"""Training-only full-frame scene-state embeddings for shot screening."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from app.analysis.schemas import GameEventResponse
from app.analysis.shot_validity_sampling_window import (
    SHOT_SAMPLING_PROTOCOL,
    ShotSamplingWindow,
    validate_sampling_window_payload,
)

SCENE_EMBEDDING_SCHEMA = "agu.shot-validity-scene-embeddings.v1"
SCENE_BACKBONE = "torchvision/mobilenet_v3_small/imagenet1k_v1"
SCENE_EMBEDDING_DIMENSION = 576
PHASE_FRACTIONS = (0.15, 0.5, 0.85)


def phase_frame_indexes(start_frame: int, end_frame: int) -> tuple[int, int, int]:
    """Return deterministic start/middle/end observations inside an event window."""

    if end_frame < start_frame:
        raise ValueError("event end frame is before its start frame")
    span = end_frame - start_frame
    return tuple(
        int(round(start_frame + span * fraction)) for fraction in PHASE_FRACTIONS
    )


def load_scene_backbone(
    checkpoint_path: Path, *, device: torch.device
) -> tuple[nn.Module, nn.Module]:
    """Load the official local MobileNet checkpoint without executable pickle."""

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("scene backbone checkpoint must contain a state dict")
    model = mobilenet_v3_small(weights=None, progress=False)
    model.load_state_dict(state)
    encoder = nn.Sequential(model.features, model.avgpool, nn.Flatten())
    transform = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
    return encoder.to(device).eval(), transform


def extract_scene_phase_embeddings(
    *,
    video_path: Path,
    events: Sequence[GameEventResponse],
    backbone: nn.Module,
    transform: nn.Module,
    device: torch.device,
    batch_size: int = 16,
    sampling_windows: Sequence[ShotSamplingWindow] | None = None,
) -> list[list[list[float]]]:
    """Extract three raw-frame embeddings for each labeled event window."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if sampling_windows is not None and len(sampling_windows) != len(events):
        raise ValueError("sampling windows must align with events")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open source video: {video_path.name}")
    tensors: list[torch.Tensor] = []
    flat_embeddings: list[list[float]] = []

    def flush() -> None:
        if not tensors:
            return
        with torch.inference_mode():
            values = backbone(torch.stack(tensors).to(device))
        flat_embeddings.extend(values.detach().cpu().to(torch.float32).tolist())
        tensors.clear()

    try:
        for index, event in enumerate(events):
            window = sampling_windows[index] if sampling_windows is not None else None
            start_frame = event.start_frame if window is None else window.start_frame
            end_frame = event.end_frame if window is None else window.end_frame
            for frame_index in phase_frame_indexes(start_frame, end_frame):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ValueError(f"cannot decode source frame {frame_index}")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2, 0, 1)
                tensors.append(transform(tensor))
                if len(tensors) >= batch_size:
                    flush()
    finally:
        capture.release()
    flush()
    expected = len(events) * len(PHASE_FRACTIONS)
    if len(flat_embeddings) != expected:
        raise ValueError("scene backbone returned an unexpected embedding count")
    return [
        flat_embeddings[index : index + len(PHASE_FRACTIONS)]
        for index in range(0, expected, len(PHASE_FRACTIONS))
    ]


def seal_scene_embedding_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    artifact["schema_version"] = SCENE_EMBEDDING_SCHEMA
    artifact["runtime_consumable"] = False
    artifact.pop("artifact_sha256", None)
    _validate_scene_embedding_artifact(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_scene_embedding_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_scene_embedding_artifact(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("scene embedding artifact hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_scene_embedding_artifact(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != SCENE_EMBEDDING_SCHEMA:
        raise ValueError("unsupported scene embedding schema")
    if artifact.get("runtime_consumable") is not False:
        raise ValueError("scene embeddings must remain training-only")
    if artifact.get("backbone") != SCENE_BACKBONE:
        raise ValueError("unsupported scene embedding backbone")
    if int(artifact.get("embedding_dimension", 0)) != SCENE_EMBEDDING_DIMENSION:
        raise ValueError("scene embedding dimension mismatch")
    if tuple(artifact.get("phase_fractions") or ()) != PHASE_FRACTIONS:
        raise ValueError("scene phase fractions do not match the fixed protocol")
    if not artifact.get("training_manifest_sha256") or not artifact.get(
        "backbone_sha256"
    ):
        raise ValueError("scene embeddings require training and backbone provenance")
    examples = artifact.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("scene embedding artifact requires examples")
    sampling_protocol = artifact.get("sampling_protocol")
    if sampling_protocol is not None and sampling_protocol != SHOT_SAMPLING_PROTOCOL:
        raise ValueError("unsupported scene sampling protocol")
    for example in examples:
        if not isinstance(example, Mapping):
            raise ValueError("scene embedding example must be an object")
        if not isinstance(example.get("event_present"), bool):
            raise ValueError("scene embedding example requires a boolean label")
        if sampling_protocol is not None:
            sampling_window = example.get("sampling_window")
            if not isinstance(sampling_window, Mapping):
                raise ValueError("scene embedding example requires a sampling window")
            validate_sampling_window_payload(sampling_window)
        phases = example.get("phase_embeddings")
        if not isinstance(phases, list) or len(phases) != len(PHASE_FRACTIONS):
            raise ValueError("scene embedding example requires three phases")
        for embedding in phases:
            if (
                not isinstance(embedding, list)
                or len(embedding) != SCENE_EMBEDDING_DIMENSION
            ):
                raise ValueError("scene phase embedding dimension mismatch")
            if not all(np.isfinite(float(value)) for value in embedding):
                raise ValueError("scene phase embedding values must be finite")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
