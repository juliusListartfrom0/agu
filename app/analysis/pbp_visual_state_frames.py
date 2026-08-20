from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from app.analysis.shot_validity_scene_state import (
    SCENE_BACKBONE,
    SCENE_EMBEDDING_DIMENSION,
)

ANCHOR_STATE_EMBEDDING_SCHEMA = "agu.pbp-anchor-state-embeddings.v1"
ANCHOR_OFFSETS_SECONDS = (-2.0, 0.0, 2.0)
PRE_ANCHOR_OFFSETS_SECONDS = (-4.0, -3.0, -2.0, -1.0, 0.0)
POST_ANCHOR_OFFSETS_SECONDS = (0.0, 5.0, 10.0, 15.0, 20.0)
WIDE_ANCHOR_OFFSETS_SECONDS = (
    -8.0,
    -6.0,
    -4.0,
    -2.0,
    0.0,
    2.0,
    4.0,
    6.0,
    8.0,
)
SUPPORTED_OFFSET_PROTOCOLS = {
    ANCHOR_OFFSETS_SECONDS,
    PRE_ANCHOR_OFFSETS_SECONDS,
    POST_ANCHOR_OFFSETS_SECONDS,
    WIDE_ANCHOR_OFFSETS_SECONDS,
}
DINO_V2_SMALL_BACKBONE = "facebook/dinov2-small"
BACKBONE_DIMENSIONS = {
    SCENE_BACKBONE: SCENE_EMBEDDING_DIMENSION,
    DINO_V2_SMALL_BACKBONE: 384,
}

_SHA256 = re.compile(r"[0-9a-f]{64}")
_TRAINING_STATES = {"field_goal", "free_throw"}


def anchor_frame_indexes(
    *,
    anchor_frame: int,
    source_fps: float,
    frame_count: int,
    offsets_seconds: Sequence[float] = ANCHOR_OFFSETS_SECONDS,
) -> tuple[int, ...]:
    if (
        isinstance(anchor_frame, bool)
        or anchor_frame < 0
        or not math.isfinite(source_fps)
        or source_fps <= 0
        or frame_count <= 0
    ):
        raise ValueError("invalid anchor-state frame sampling configuration")
    offsets = tuple(float(value) for value in offsets_seconds)
    if offsets not in SUPPORTED_OFFSET_PROTOCOLS:
        raise ValueError("unsupported anchor-state offset protocol")
    indexes = tuple(
        round(anchor_frame + offset * source_fps)
        for offset in offsets
    )
    if (
        len(set(indexes)) != len(indexes)
        or indexes[0] < 0
        or indexes[-1] >= frame_count
        or indexes[offsets.index(0.0)] != anchor_frame
    ):
        raise ValueError("anchor-state temporal window is incomplete")
    return indexes


def seal_anchor_state_embedding_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    artifact["schema_version"] = ANCHOR_STATE_EMBEDDING_SCHEMA
    artifact["runtime_consumable"] = False
    artifact["backbone"] = str(artifact.get("backbone") or SCENE_BACKBONE)
    artifact["embedding_dimension"] = BACKBONE_DIMENSIONS.get(
        artifact["backbone"],
        0,
    )
    artifact.pop("artifact_sha256", None)
    _validate_anchor_state_embedding_artifact(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_anchor_state_embedding_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_anchor_state_embedding_artifact(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("anchor-state embedding artifact hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _validate_anchor_state_embedding_artifact(
    artifact: Mapping[str, Any],
) -> None:
    if artifact.get("schema_version") != ANCHOR_STATE_EMBEDDING_SCHEMA:
        raise ValueError("unsupported anchor-state embedding schema")
    if (
        artifact.get("runtime_consumable") is not False
        or artifact.get("truth_used_for_training_only") is not True
        or artifact.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("anchor-state embeddings must remain training-only")
    backbone = str(artifact.get("backbone") or "")
    if backbone not in BACKBONE_DIMENSIONS:
        raise ValueError("unsupported anchor-state embedding backbone")
    if int(artifact.get("embedding_dimension", 0)) != BACKBONE_DIMENSIONS[backbone]:
        raise ValueError("anchor-state embedding dimension mismatch")
    offsets = tuple(artifact.get("anchor_offsets_seconds") or ())
    if offsets not in SUPPORTED_OFFSET_PROTOCOLS:
        raise ValueError("anchor-state offsets do not match the fixed protocol")

    manifests = _require_hash_list(
        artifact.get("training_manifest_sha256s"),
        field="training manifest",
    )
    clocks = _require_hash_list(
        artifact.get("clock_artifact_sha256s"),
        field="clock artifact",
    )
    sources = set(
        _require_hash_list(
            artifact.get("source_video_sha256s"),
            field="source video",
        )
    )
    blind = set(
        _require_hash_list(
            artifact.get("sealed_blind_video_sha256s"),
            field="sealed blind video",
        )
    )
    _require_sha256(artifact.get("backbone_sha256"), field="backbone")
    if not manifests or not clocks or sources & blind:
        raise ValueError("anchor-state source provenance is invalid")

    examples = artifact.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("anchor-state embedding artifact requires examples")
    seen: set[tuple[str, str]] = set()
    observed_sources: set[str] = set()
    for row in examples:
        if not isinstance(row, Mapping):
            raise ValueError("invalid anchor-state example")
        source = _require_sha256(row.get("source_video_sha256"), field="source video")
        if source in blind:
            raise ValueError("sealed blind video cannot supply anchor-state embeddings")
        if source not in sources:
            raise ValueError("anchor-state example source is not declared")
        observed_sources.add(source)
        _require_sha256(row.get("candidate_bundle_sha256"), field="candidate bundle")
        event_id = str(row.get("event_id") or "")
        key = (source, event_id)
        if not event_id or key in seen:
            raise ValueError("invalid or duplicate anchor-state event")
        seen.add(key)
        if row.get("state") not in _TRAINING_STATES:
            raise ValueError("invalid anchor-state training state")
        anchor = int(row.get("anchor_frame", -1))
        fps = float(row.get("source_fps", 0.0))
        frame_count = int(row.get("frame_count", 0))
        expected = anchor_frame_indexes(
            anchor_frame=anchor,
            source_fps=fps,
            frame_count=frame_count,
            offsets_seconds=offsets,
        )
        if tuple(row.get("frame_indexes") or ()) != expected:
            raise ValueError("anchor-state frame indexes do not match the protocol")
        embeddings = row.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(expected):
            raise ValueError("anchor-state example requires three embeddings")
        for embedding in embeddings:
            if (
                not isinstance(embedding, list)
                or len(embedding) != BACKBONE_DIMENSIONS[backbone]
                or not all(np.isfinite(float(value)) for value in embedding)
            ):
                raise ValueError("invalid anchor-state embedding values")
    if observed_sources != sources:
        raise ValueError("anchor-state examples do not cover all declared videos")


def _require_hash_list(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} hashes are required")
    hashes = tuple(_require_sha256(item, field=field) for item in value)
    if len(set(hashes)) != len(hashes):
        raise ValueError(f"duplicate {field} hash")
    return hashes


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value or "")
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"invalid {field} SHA-256")
    return text


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
