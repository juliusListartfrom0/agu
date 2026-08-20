"""Replaceable torchvision video backbones for training-only shot screening."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.video import (
    MViT_V2_S_Weights,
    Swin3D_T_Weights,
    mvit_v2_s,
    swin3d_t,
)

from app.analysis.schemas import GameEventResponse
from app.analysis.shot_validity_sampling_window import (
    SHOT_SAMPLING_PROTOCOL,
    ShotSamplingWindow,
    validate_sampling_window_payload,
    video_frame_indexes,
)

VIDEO_BACKBONE_EMBEDDING_SCHEMA = "agu.shot-validity-video-embeddings.v1"


@dataclass(frozen=True)
class VideoBackboneSpec:
    name: str
    embedding_dimension: int
    minimum_clip_frames: int
    license: str
    weights_url: str


VIDEO_BACKBONES = {
    "torchvision/swin3d_t/kinetics400_v1": VideoBackboneSpec(
        name="torchvision/swin3d_t/kinetics400_v1",
        embedding_dimension=768,
        minimum_clip_frames=16,
        license="BSD-3-Clause (torchvision); verify upstream dataset/weight terms",
        weights_url=Swin3D_T_Weights.KINETICS400_V1.url,
    ),
    "torchvision/mvit_v2_s/kinetics400_v1": VideoBackboneSpec(
        name="torchvision/mvit_v2_s/kinetics400_v1",
        embedding_dimension=768,
        minimum_clip_frames=16,
        license="BSD-3-Clause (torchvision); verify upstream dataset/weight terms",
        weights_url=MViT_V2_S_Weights.KINETICS400_V1.url,
    ),
}


def get_video_backbone_spec(name: str) -> VideoBackboneSpec:
    try:
        return VIDEO_BACKBONES[name]
    except KeyError as exc:
        raise ValueError(f"unsupported video backbone: {name}") from exc


def load_video_backbone(
    name: str,
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[nn.Module, nn.Module]:
    """Load a local official torchvision checkpoint and return encoder/transform."""

    get_video_backbone_spec(name)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("video backbone checkpoint must contain a state dict")
    if name == "torchvision/swin3d_t/kinetics400_v1":
        model = swin3d_t(weights=None, progress=False)
        model.load_state_dict(state)
        model.head = nn.Identity()
        transform = Swin3D_T_Weights.KINETICS400_V1.transforms()
    elif name == "torchvision/mvit_v2_s/kinetics400_v1":
        model = mvit_v2_s(weights=None, progress=False)
        model.load_state_dict(state)
        model.head = nn.Identity()
        transform = MViT_V2_S_Weights.KINETICS400_V1.transforms()
    else:  # pragma: no cover - guarded by registry lookup
        raise AssertionError(name)
    return model.to(device).eval(), transform


def extract_video_embeddings(
    *,
    video_path: Path,
    events: Sequence[GameEventResponse],
    backbone: nn.Module,
    transform: nn.Module,
    device: torch.device,
    clip_frames: int = 16,
    batch_size: int = 1,
    sampling_windows: Sequence[ShotSamplingWindow] | None = None,
) -> list[list[float]]:
    """Extract raw-only embeddings from uniformly sampled event windows."""

    if clip_frames < 16 or batch_size < 1:
        raise ValueError("clip_frames must be >=16 and batch_size must be positive")
    if sampling_windows is not None and len(sampling_windows) != len(events):
        raise ValueError("sampling windows must align with events")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open source video: {video_path.name}")
    tensors: list[torch.Tensor] = []
    embeddings: list[list[float]] = []

    def flush() -> None:
        if not tensors:
            return
        batch: torch.Tensor | None = None
        values: torch.Tensor | None = None
        cpu_values: torch.Tensor | None = None
        try:
            with torch.inference_mode():
                batch = torch.stack(tensors).to(device)
                values = backbone(batch)
            cpu_values = values.detach().cpu().to(torch.float32)
            embeddings.extend(cpu_values.tolist())
        finally:
            tensors.clear()
            batch = None
            values = None
            cpu_values = None
            _release_accelerator_cache(device)

    try:
        for index, event in enumerate(events):
            window = (
                sampling_windows[index]
                if sampling_windows is not None
                else ShotSamplingWindow(
                    start_frame=event.start_frame,
                    end_frame=event.end_frame,
                    anchor_frame=event.end_frame,
                    anchor_source="legacy_event_window",
                )
            )
            indexes = video_frame_indexes(window, clip_frames=clip_frames)
            frames = []
            for frame_index in indexes:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ValueError(f"cannot decode source frame {frame_index}")
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            clip = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
            tensors.append(transform(clip))
            if len(tensors) >= batch_size:
                flush()
    finally:
        capture.release()
    flush()
    return embeddings


def _release_accelerator_cache(device: torch.device) -> None:
    """Return batch-local unified memory before the resource guard samples again."""

    if device.type == "mps":
        torch.mps.empty_cache()


def seal_video_embedding_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    artifact["schema_version"] = VIDEO_BACKBONE_EMBEDDING_SCHEMA
    artifact["runtime_consumable"] = False
    artifact.pop("artifact_sha256", None)
    _validate_video_embedding_artifact(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_video_embedding_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_video_embedding_artifact(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("video embedding artifact hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_video_embedding_artifact(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != VIDEO_BACKBONE_EMBEDDING_SCHEMA:
        raise ValueError("unsupported video embedding schema")
    if artifact.get("runtime_consumable") is not False:
        raise ValueError("labeled video embeddings must remain training-only")
    spec = get_video_backbone_spec(str(artifact.get("backbone") or ""))
    if int(artifact.get("embedding_dimension", 0)) != spec.embedding_dimension:
        raise ValueError("video embedding dimension does not match backbone")
    if int(artifact.get("clip_frames", 0)) < spec.minimum_clip_frames:
        raise ValueError("video embedding clip is shorter than backbone minimum")
    if not artifact.get("training_manifest_sha256") or not artifact.get("backbone_sha256"):
        raise ValueError("video embeddings require training and backbone provenance")
    examples = artifact.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("video embedding artifact requires examples")
    sampling_protocol = artifact.get("sampling_protocol")
    if sampling_protocol is not None and sampling_protocol != SHOT_SAMPLING_PROTOCOL:
        raise ValueError("unsupported video sampling protocol")
    for example in examples:
        if not isinstance(example, Mapping):
            raise ValueError("video embedding example must be an object")
        if not isinstance(example.get("event_present"), bool):
            raise ValueError("video embedding example requires a boolean label")
        if sampling_protocol is not None:
            sampling_window = example.get("sampling_window")
            if not isinstance(sampling_window, Mapping):
                raise ValueError("video embedding example requires a sampling window")
            validate_sampling_window_payload(sampling_window)
        embedding = example.get("embedding")
        if not isinstance(embedding, list) or len(embedding) != spec.embedding_dimension:
            raise ValueError("video embedding example dimension mismatch")
        if not all(np.isfinite(float(value)) for value in embedding):
            raise ValueError("video embedding values must be finite")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
