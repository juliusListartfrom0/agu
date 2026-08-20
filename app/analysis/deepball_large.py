"""Pinned offline adapter for the official WASB DeepBall-Large basketball model.

This module is deliberately separate from AGU's runtime detector.  The upstream
model is used only for hash-bound, CPU research screens until an independent
cross-broadcast gate is met.
"""

from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Mapping

import cv2
import numpy as np
import torch
import yaml
from torch import nn

from app.analysis.wasb_ball import affine_point, frame_affine

DEFAULT_INPUT_WH = (1280, 720)
DEFAULT_OUTPUT_WH = (320, 180)
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class LoadedDeepBallModel:
    model: nn.Module
    device: torch.device
    source_sha256: str
    checkpoint_sha256: str


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
            f"DeepBall-Large {label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def _load_source_module(source_path: Path, source_sha256: str) -> ModuleType:
    module_name = f"_agu_deepball_{source_sha256[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load DeepBall source module from {source_path}")
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


def load_pinned_deepball_large_model(
    *,
    source_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    expected_source_sha256: str,
    expected_checkpoint_sha256: str,
    device: str = "auto",
) -> LoadedDeepBallModel:
    """Load the upstream model with strict source/checkpoint/hash checks."""

    source_path = source_path.resolve()
    config_path = config_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    for path, label in (
        (source_path, "source"),
        (config_path, "config"),
        (checkpoint_path, "checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"DeepBall-Large {label} does not exist: {path}")

    source_sha256 = _verify_sha256(source_path, expected_source_sha256, "source")
    checkpoint_sha256 = _verify_sha256(
        checkpoint_path, expected_checkpoint_sha256, "checkpoint"
    )
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, Mapping):
        raise ValueError("DeepBall-Large config must be a mapping")

    module = _load_source_module(source_path, source_sha256)
    model_type = getattr(module, "DeepBall", None)
    if not isinstance(model_type, type) or not issubclass(model_type, nn.Module):
        raise TypeError("pinned DeepBall source must define torch.nn.Module DeepBall")

    frames_in = int(raw_config["frames_in"])
    frames_out = int(raw_config["frames_out"])
    class_out = int(raw_config["class_out"])
    model = model_type(
        frames_in * 3,
        frames_out * class_out,
        block_channels=list(raw_config["block_channels"]),
        block_maxpools=list(raw_config["block_maxpools"]),
        first_conv_kernel_size=int(raw_config["first_conv_kernel_size"]),
        first_conv_stride=int(raw_config["first_conv_stride"]),
        last_conv_kernel_size=int(raw_config["last_conv_kernel_size"]),
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("DeepBall-Large checkpoint must be a mapping")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("DeepBall-Large checkpoint is missing model_state_dict")
    model.load_state_dict(state_dict, strict=True)

    resolved_device = _resolve_device(device)
    model.to(resolved_device)
    model.eval()
    return LoadedDeepBallModel(
        model=model,
        device=resolved_device,
        source_sha256=source_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )


def preprocess_rgb_frame(
    frame: np.ndarray,
    *,
    input_wh: tuple[int, int] = DEFAULT_INPUT_WH,
) -> tuple[torch.Tensor, np.ndarray]:
    """Apply the upstream affine/ImageNet preprocessing to one RGB frame."""

    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("frame must be uint8 RGB with shape [height, width, 3]")
    forward = frame_affine(frame.shape, input_wh)
    inverse = frame_affine(frame.shape, DEFAULT_OUTPUT_WH, inverse=True)
    warped = cv2.warpAffine(frame, forward, input_wh, flags=cv2.INTER_LINEAR)
    normalized = (warped.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    channels = np.ascontiguousarray(np.transpose(normalized, (2, 0, 1)))
    return torch.from_numpy(channels).unsqueeze(0), inverse


def decode_deepball_logits(
    logits: torch.Tensor,
    inverse_affine: np.ndarray,
    *,
    score_threshold: float = 0.5,
) -> list[dict[str, float]]:
    """Decode the single foreground softmax peak into source-frame coordinates."""

    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 2:
        raise ValueError("DeepBall logits must have shape [1, 2, height, width]")
    if inverse_affine.shape != (2, 3):
        raise ValueError("inverse_affine must have shape (2, 3)")
    if not 0.0 < score_threshold < 1.0:
        raise ValueError("score_threshold must be between zero and one")

    foreground = torch.softmax(logits.float(), dim=1)[0, 1]
    score, index = torch.max(foreground.reshape(-1), dim=0)
    score_value = float(score.item())
    if score_value < score_threshold:
        return []
    height, width = foreground.shape
    y = int(index.item()) // width
    x = int(index.item()) % width
    source_point = affine_point(
        np.asarray([float(x), float(y)], dtype=np.float32), inverse_affine
    )
    return [
        {
            "x": float(source_point[0]),
            "y": float(source_point[1]),
            "score": score_value,
        }
    ]


@torch.inference_mode()
def infer_rgb_frame(
    loaded: LoadedDeepBallModel,
    frame: np.ndarray,
    *,
    score_threshold: float = 0.5,
) -> list[dict[str, float]]:
    """Run one frame through the research-only loaded model."""

    tensor, inverse_affine = preprocess_rgb_frame(frame)
    output = loaded.model(tensor.to(loaded.device))
    if not isinstance(output, Mapping) or 0 not in output:
        raise ValueError("DeepBall model output must contain scale 0")
    logits = output[0]
    if not isinstance(logits, torch.Tensor):
        raise TypeError("DeepBall scale-0 output must be a tensor")
    return decode_deepball_logits(
        logits.cpu(), inverse_affine, score_threshold=score_threshold
    )
