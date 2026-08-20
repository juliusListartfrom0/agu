"""Pinned offline adapter for WASB-SBDT basketball heatmap inference.

The adapter intentionally does not import WASB's CUDA-only detector wrapper.
It loads the upstream MIT-licensed HRNet definition from an explicitly pinned
source file, reproduces the published affine/ImageNet preprocessing contract,
and leaves service/runtime promotion to a later evidence gate.
"""

from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
import yaml
from torch import nn

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_INPUT_WH = (512, 288)


class _AttrMapping(dict[str, Any]):
    """Recursive mapping compatible with WASB's mixed key/attribute access."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


@dataclass(frozen=True)
class LoadedWasbModel:
    model: nn.Module
    device: torch.device
    source_sha256: str
    checkpoint_sha256: str


def _recursive_attr_mapping(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _AttrMapping(
            {str(key): _recursive_attr_mapping(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_recursive_attr_mapping(item) for item in value]
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256(path: Path, expected: str, label: str) -> str:
    actual = _sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"WASB {label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def _third_point(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    direction = first - second
    return second + np.asarray([-direction[1], direction[0]], dtype=np.float32)


def frame_affine(
    frame_shape: Sequence[int],
    output_wh: tuple[int, int] = DEFAULT_INPUT_WH,
    *,
    inverse: bool = False,
) -> np.ndarray:
    """Return the exact zero-rotation affine used by the WASB image loader."""

    if len(frame_shape) < 2:
        raise ValueError("frame_shape must contain height and width")
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("frame dimensions must be positive")
    output_width, output_height = output_wh
    if output_width <= 0 or output_height <= 0:
        raise ValueError("output dimensions must be positive")

    center = np.asarray([width / 2.0, height / 2.0], dtype=np.float32)
    scale = float(max(height, width))
    source = np.zeros((3, 2), dtype=np.float32)
    target = np.zeros((3, 2), dtype=np.float32)
    source[0] = center
    source[1] = center + np.asarray([0.0, scale * -0.5], dtype=np.float32)
    target[0] = np.asarray(
        [output_width * 0.5, output_height * 0.5], dtype=np.float32
    )
    target[1] = target[0] + np.asarray(
        [0.0, output_width * -0.5], dtype=np.float32
    )
    source[2] = _third_point(source[0], source[1])
    target[2] = _third_point(target[0], target[1])
    if inverse:
        return cv2.getAffineTransform(target, source)
    return cv2.getAffineTransform(source, target)


def affine_point(point: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if point.shape != (2,):
        raise ValueError("point must have shape (2,)")
    if transform.shape != (2, 3):
        raise ValueError("transform must have shape (2, 3)")
    homogeneous = np.asarray([point[0], point[1], 1.0], dtype=np.float32)
    return np.dot(transform, homogeneous)[:2]


def preprocess_rgb_triplet(
    frames: Sequence[np.ndarray],
    *,
    input_wh: tuple[int, int] = DEFAULT_INPUT_WH,
) -> tuple[torch.Tensor, np.ndarray]:
    """Warp and normalize three same-sized RGB frames into ``[1, 9, H, W]``."""

    if len(frames) != 3:
        raise ValueError(f"WASB requires exactly 3 RGB frames, got {len(frames)}")
    reference_shape = frames[0].shape
    if any(frame.shape != reference_shape for frame in frames):
        raise ValueError("all WASB input frames must have the same shape")
    if len(reference_shape) != 3 or reference_shape[2] != 3:
        raise ValueError("WASB input frames must have shape [height, width, 3]")
    if any(frame.dtype != np.uint8 for frame in frames):
        raise ValueError("WASB input frames must use uint8 RGB pixels")

    forward = frame_affine(reference_shape, input_wh)
    inverse = frame_affine(reference_shape, input_wh, inverse=True)
    channels: list[np.ndarray] = []
    for frame in frames:
        warped = cv2.warpAffine(
            frame,
            forward,
            input_wh,
            flags=cv2.INTER_LINEAR,
        )
        normalized = (warped.astype(np.float32) / 255.0 - IMAGENET_MEAN) / (
            IMAGENET_STD
        )
        channels.append(np.transpose(normalized, (2, 0, 1)))
    contiguous = np.ascontiguousarray(np.concatenate(channels, axis=0))
    return torch.from_numpy(contiguous).unsqueeze(0), inverse


def decode_heatmap_components(
    logits: torch.Tensor,
    inverse_affine: np.ndarray,
    *,
    threshold: float = 0.5,
) -> list[list[dict[str, float | int]]]:
    """Decode all thresholded connected components using WASB's weighted center."""

    if logits.ndim != 4 or logits.shape[0] != 1:
        raise ValueError("WASB logits must have shape [1, frames, height, width]")
    if inverse_affine.shape != (2, 3):
        raise ValueError("inverse_affine must have shape (2, 3)")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between zero and one")

    probabilities = logits.detach().float().sigmoid().cpu().numpy()[0]
    decoded: list[list[dict[str, float | int]]] = []
    for heatmap in probabilities:
        binary = (heatmap > threshold).astype(np.uint8)
        label_count, labels = cv2.connectedComponents(binary)
        frame_detections: list[dict[str, float | int]] = []
        for label in range(1, label_count):
            ys, xs = np.where(labels == label)
            weights = heatmap[ys, xs]
            component_score = float(weights.sum())
            if component_score <= 0.0:
                continue
            output_point = np.asarray(
                [
                    float(np.dot(xs, weights) / component_score),
                    float(np.dot(ys, weights) / component_score),
                ],
                dtype=np.float32,
            )
            source_point = affine_point(output_point, inverse_affine)
            frame_detections.append(
                {
                    "x": float(source_point[0]),
                    "y": float(source_point[1]),
                    "peak_probability": float(weights.max()),
                    "component_score": component_score,
                    "pixel_count": int(weights.size),
                }
            )
        frame_detections.sort(
            key=lambda item: (
                float(item["component_score"]),
                float(item["peak_probability"]),
            ),
            reverse=True,
        )
        decoded.append(frame_detections)
    return decoded


def _load_source_module(source_path: Path, source_sha256: str) -> ModuleType:
    module_name = f"_agu_wasb_hrnet_{source_sha256[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load WASB source module from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def load_pinned_wasb_model(
    *,
    source_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    expected_source_sha256: str,
    expected_checkpoint_sha256: str,
    device: str = "auto",
) -> LoadedWasbModel:
    """Load a pinned WASB HRNet and a tensor-only checkpoint with strict keys."""

    source_path = source_path.resolve()
    config_path = config_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    for path, label in (
        (source_path, "source"),
        (config_path, "config"),
        (checkpoint_path, "checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"WASB {label} does not exist: {path}")

    source_sha256 = _verify_sha256(
        source_path, expected_source_sha256, "source"
    )
    checkpoint_sha256 = _verify_sha256(
        checkpoint_path, expected_checkpoint_sha256, "checkpoint"
    )
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, Mapping):
        raise ValueError("WASB model config must be a mapping")
    config = _recursive_attr_mapping(raw_config)

    module = _load_source_module(source_path, source_sha256)
    model_type = getattr(module, "HRNet", None)
    if not isinstance(model_type, type) or not issubclass(model_type, nn.Module):
        raise TypeError("pinned WASB source must define torch.nn.Module HRNet")
    model = model_type(config)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("WASB checkpoint must be a mapping")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("WASB checkpoint is missing model_state_dict")
    model.load_state_dict(state_dict, strict=True)

    resolved_device = _resolve_device(device)
    model.to(resolved_device)
    model.eval()
    return LoadedWasbModel(
        model=model,
        device=resolved_device,
        source_sha256=source_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )
