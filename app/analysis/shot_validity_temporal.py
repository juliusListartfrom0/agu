"""Hash-bound temporal shot-validity inference over raw event windows."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

from app.analysis.schemas import EventEvidenceResponse, GameEventResponse

TEMPORAL_SHOT_EMBEDDING_SCHEMA = "agu.shot-validity-temporal-embeddings.v1"
TEMPORAL_SHOT_MODEL_SCHEMA = "agu.shot-validity-temporal-linear-model.v1"
TEMPORAL_SHOT_BACKBONE = "torchvision/r2plus1d_18/kinetics400_v1"
TEMPORAL_SHOT_EMBEDDING_DIMENSION = 512


class TemporalShotValidityModel:
    """Dependency-light logistic head over a fixed R(2+1)D embedding."""

    def __init__(self, artifact: Mapping[str, Any]) -> None:
        self.artifact = verify_temporal_shot_model(artifact)

    def predict_probability(self, embedding: Sequence[float]) -> float:
        if len(embedding) != TEMPORAL_SHOT_EMBEDDING_DIMENSION:
            raise ValueError("temporal shot embedding dimension mismatch")
        score = float(self.artifact["intercept"])
        for index, value in enumerate(embedding):
            standardized = (
                float(value) - float(self.artifact["feature_mean"][index])
            ) / max(float(self.artifact["feature_scale"][index]), 1e-8)
            score += standardized * float(self.artifact["coefficients"][index])
        if score >= 0:
            return 1.0 / (1.0 + math.exp(-score))
        exponent = math.exp(score)
        return exponent / (1.0 + exponent)

    def accepts(self, embedding: Sequence[float]) -> bool:
        return self.predict_probability(embedding) >= float(self.artifact["threshold"])


def seal_temporal_shot_model(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    artifact["schema_version"] = TEMPORAL_SHOT_MODEL_SCHEMA
    artifact["backbone"] = TEMPORAL_SHOT_BACKBONE
    artifact["embedding_dimension"] = TEMPORAL_SHOT_EMBEDDING_DIMENSION
    artifact.pop("model_sha256", None)
    _validate_temporal_shot_model(artifact)
    artifact["model_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_temporal_shot_model(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("model_sha256", ""))
    _validate_temporal_shot_model(artifact)
    if _canonical_sha256(artifact) != claimed:
        raise ValueError("temporal shot model hash mismatch")
    artifact["model_sha256"] = claimed
    return artifact


def verify_temporal_embedding_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    if artifact.get("schema_version") != TEMPORAL_SHOT_EMBEDDING_SCHEMA:
        raise ValueError("unsupported temporal shot embedding schema")
    if artifact.get("runtime_consumable") is not False:
        raise ValueError("labeled temporal embeddings must remain training-only")
    if artifact.get("backbone") != TEMPORAL_SHOT_BACKBONE:
        raise ValueError("temporal shot embedding backbone mismatch")
    if artifact.get("embedding_dimension") != TEMPORAL_SHOT_EMBEDDING_DIMENSION:
        raise ValueError("temporal shot embedding dimension mismatch")
    examples = artifact.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("temporal shot embeddings require examples")
    for example in examples:
        embedding = example.get("embedding") if isinstance(example, Mapping) else None
        if not isinstance(embedding, list) or len(embedding) != TEMPORAL_SHOT_EMBEDDING_DIMENSION:
            raise ValueError("temporal shot example embedding dimension mismatch")
        if not isinstance(example.get("event_present"), bool):
            raise ValueError("temporal shot example requires a boolean label")
    if _canonical_sha256(artifact) != claimed:
        raise ValueError("temporal shot embedding artifact hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def seal_temporal_embedding_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    artifact["schema_version"] = TEMPORAL_SHOT_EMBEDDING_SCHEMA
    artifact["runtime_consumable"] = False
    artifact["backbone"] = TEMPORAL_SHOT_BACKBONE
    artifact["embedding_dimension"] = TEMPORAL_SHOT_EMBEDDING_DIMENSION
    artifact.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return verify_temporal_embedding_artifact(artifact)


def load_temporal_backbone(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> nn.Module:
    """Load the configured local Kinetics checkpoint without network access."""

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("temporal backbone checkpoint must contain a state dict")
    model = r2plus1d_18(weights=None, progress=False)
    model.load_state_dict(state)
    model.fc = nn.Identity()
    return model.to(device).eval()


def extract_event_embeddings(
    *,
    video_path: Path,
    events: Sequence[GameEventResponse],
    backbone: nn.Module,
    device: torch.device,
    clip_frames: int = 16,
    batch_size: int = 4,
) -> list[list[float]]:
    """Embed uniformly sampled full-frame event windows using raw video only."""

    if clip_frames < 2 or batch_size < 1:
        raise ValueError("clip_frames must be >=2 and batch_size must be positive")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open source video: {video_path.name}")
    transform = R2Plus1D_18_Weights.KINETICS400_V1.transforms()
    tensors: list[torch.Tensor] = []
    embeddings: list[list[float]] = []

    def flush() -> None:
        if not tensors:
            return
        with torch.inference_mode():
            batch = torch.stack(tensors).to(device)
            values = backbone(batch).detach().cpu().to(torch.float32).numpy()
        embeddings.extend(values.tolist())
        tensors.clear()

    try:
        for event in events:
            indexes = np.linspace(event.start_frame, event.end_frame, clip_frames)
            frames = []
            for frame_index in np.rint(indexes).astype(int):
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


def apply_temporal_shot_gate(
    events: Sequence[GameEventResponse],
    *,
    shot_events: Sequence[GameEventResponse],
    embeddings: Sequence[Sequence[float]],
    model: TemporalShotValidityModel,
) -> list[GameEventResponse]:
    """Reject false shots and dependent synthetic rebounds using raw-window embeddings."""

    if len(shot_events) != len(embeddings):
        raise ValueError("temporal shot events and embeddings must have equal length")
    predictions = {
        event.event_id: (model.predict_probability(embedding), model.accepts(embedding))
        for event, embedding in zip(shot_events, embeddings, strict=True)
    }
    rejected_shots = {
        event_id for event_id, (_probability, accepted) in predictions.items() if not accepted
    }
    output = []
    for event in events:
        prediction = predictions.get(event.event_id)
        if prediction is not None:
            probability, accepted = prediction
            if not accepted:
                continue
            evidence = EventEvidenceResponse(
                evidence_id=f"{event.event_id}:temporal-shot-validity",
                kind="traditional_temporal_shot_validity_model",
                source_video_id=event.source_video_id,
                start_frame=event.start_frame,
                end_frame=event.end_frame,
                confidence=probability,
                details={
                    "model_sha256": model.artifact["model_sha256"],
                    "backbone_sha256": model.artifact["backbone_sha256"],
                    "probability": probability,
                    "threshold": model.artifact["threshold"],
                    "clip_frames": model.artifact["clip_frames"],
                },
            )
            event = event.model_copy(
                update={
                    "evidence": [*event.evidence, evidence],
                    "reason": f"{event.reason}; temporal shot-validity gate accepted candidate",
                }
            )
        if rejected_shots.intersection(event.related_event_ids):
            continue
        output.append(event)
    return output


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_temporal_shot_model(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != TEMPORAL_SHOT_MODEL_SCHEMA:
        raise ValueError("unsupported temporal shot model schema")
    if artifact.get("backbone") != TEMPORAL_SHOT_BACKBONE:
        raise ValueError("temporal shot model backbone mismatch")
    if artifact.get("embedding_dimension") != TEMPORAL_SHOT_EMBEDDING_DIMENSION:
        raise ValueError("temporal shot model embedding dimension mismatch")
    for field in ("coefficients", "feature_mean", "feature_scale"):
        values = artifact.get(field)
        if not isinstance(values, list) or len(values) != TEMPORAL_SHOT_EMBEDDING_DIMENSION:
            raise ValueError(f"temporal shot model {field} dimension mismatch")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"temporal shot model {field} must be finite")
    if not math.isfinite(float(artifact.get("intercept", math.nan))):
        raise ValueError("temporal shot model intercept must be finite")
    threshold = float(artifact.get("threshold", -1.0))
    if not 0 <= threshold <= 1:
        raise ValueError("temporal shot model threshold must be in [0,1]")
    if int(artifact.get("clip_frames", 0)) < 2:
        raise ValueError("temporal shot model clip_frames must be at least 2")
    if not artifact.get("training_manifest_sha256") or not artifact.get("backbone_sha256"):
        raise ValueError("temporal shot model requires training and backbone provenance")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
