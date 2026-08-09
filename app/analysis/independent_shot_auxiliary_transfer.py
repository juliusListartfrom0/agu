"""Label-hidden transfer screening for independent shot-VLM auxiliary evidence.

This module is deliberately stricter than a normal detector wrapper.  It accepts
only a sealed, label-free VLM plan and raw frames, and emits event-level transfer
scores without selecting a threshold or pretending that target-game rows are
OOF evidence.  The output is research-only and cannot be consumed by AGU's
runtime or evidence gate as training truth.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from app.analysis.independent_shot_vlm import verify_independent_shot_vlm_plan

TRANSFER_SCHEMA = "agu.independent-shot-vlm-auxiliary-transfer.v1"
_FORBIDDEN_LABEL_FIELDS = {
    "event_present",
    "label",
    "review_note",
    "ground_truth",
    "target",
}


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a JSON payload while excluding its own artifact digest."""

    normalized = dict(payload)
    normalized.pop("artifact_sha256", None)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Any) -> str:
    """Return a streaming SHA-256 for a local raw video or model file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_unit(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{field} must be finite and within [0, 1]")
    return parsed


def summarize_event_scores(
    *,
    detections_by_frame: Sequence[Mapping[str, Any]],
    start_frame: int,
    end_frame: int,
) -> dict[str, Any]:
    """Convert frame detections into a transparent, uncalibrated event score.

    ``transfer_score`` is the geometric mean of the detector's strongest score
    and its frame presence fraction.  This is intentionally not called a
    probability: no target-game label or threshold is used to calibrate it.
    """

    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError("event frame bounds are invalid")
    if not detections_by_frame:
        raise ValueError("at least one sampled frame is required")
    frame_scores: list[float] = []
    sampled_frames: list[int] = []
    for row in detections_by_frame:
        frame = int(row.get("frame", -1))
        if frame < start_frame or frame >= end_frame:
            raise ValueError("sampled frame is outside the event bounds")
        if frame in sampled_frames:
            raise ValueError("duplicate sampled frame")
        detections = row.get("detections")
        if not isinstance(detections, Sequence) or isinstance(detections, (str, bytes)):
            raise ValueError("detections must be a sequence")
        strongest = 0.0
        for detection in detections:
            if not isinstance(detection, Mapping):
                raise ValueError("detector rows must be mappings")
            strongest = max(strongest, _finite_unit(detection.get("score"), field="score"))
        sampled_frames.append(frame)
        frame_scores.append(strongest)
    positive = [score for score in frame_scores if score > 0.0]
    presence_fraction = len(positive) / len(frame_scores)
    max_score = max(frame_scores)
    mean_positive_score = sum(positive) / len(positive) if positive else 0.0
    transfer_score = math.sqrt(max_score * presence_fraction)
    return {
        "sampled_frame_count": len(frame_scores),
        "sampled_frames": sampled_frames,
        "frame_scores": frame_scores,
        "positive_frame_count": len(positive),
        "presence_fraction": presence_fraction,
        "max_detector_score": max_score,
        "mean_positive_detector_score": mean_positive_score,
        "transfer_score": transfer_score,
    }


def seal_transfer_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Seal and verify a transfer-only artifact with empty OOF coverage."""

    artifact = dict(payload)
    artifact["schema_version"] = TRANSFER_SCHEMA
    artifact["purpose"] = "offline_independent_shot_vlm_auxiliary_transfer_screen"
    artifact["runtime_consumable"] = False
    artifact["training_eligible"] = False
    artifact["codex_runtime_answer_used"] = False
    artifact["auxiliary_oof_evidence_available"] = False
    artifact["oof_predictions"] = []
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return verify_transfer_artifact(artifact)


def verify_transfer_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Verify provenance, label absence, uniqueness, and fail-closed OOF state."""

    artifact = dict(payload)
    if artifact.get("schema_version") != TRANSFER_SCHEMA:
        raise ValueError("unsupported auxiliary transfer schema")
    if artifact.get("purpose") != "offline_independent_shot_vlm_auxiliary_transfer_screen":
        raise ValueError("auxiliary transfer must be research-only")
    for field in ("runtime_consumable", "training_eligible", "codex_runtime_answer_used"):
        if artifact.get(field) is not False:
            raise ValueError(f"auxiliary transfer {field} must be false")
    if artifact.get("auxiliary_oof_evidence_available") is not False:
        raise ValueError("transfer screen cannot claim OOF evidence")
    if artifact.get("oof_predictions") != []:
        raise ValueError("transfer screen must contain no OOF predictions")
    if canonical_sha256(artifact) != artifact.get("artifact_sha256"):
        raise ValueError("auxiliary transfer artifact hash mismatch")
    rows = artifact.get("transfer_predictions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("transfer screen requires event predictions")
    plan = artifact.get("plan")
    if not isinstance(plan, Mapping):
        raise ValueError("transfer screen must carry a plan reference")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("transfer prediction rows must be mappings")
        if _FORBIDDEN_LABEL_FIELDS.intersection(row):
            raise ValueError("transfer prediction row leaks a target label")
        for field in ("source_video_sha256", "candidate_bundle_sha256", "event_id"):
            if not str(row.get(field) or ""):
                raise ValueError(f"transfer prediction lacks {field}")
        _finite_unit(row.get("transfer_score"), field="transfer_score")
    keys = [
        (
            str(row["source_video_sha256"]),
            str(row["candidate_bundle_sha256"]),
            str(row["event_id"]),
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("transfer predictions contain duplicates")
    return artifact


def verify_label_hidden_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the upstream plan and reject any nested label-bearing fields."""

    verified = verify_independent_shot_vlm_plan(plan)
    for row in verified["examples"]:
        if _FORBIDDEN_LABEL_FIELDS.intersection(row):
            raise ValueError("auxiliary input plan leaks target labels")
    return verified
