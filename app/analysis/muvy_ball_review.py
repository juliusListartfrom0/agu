"""Hash-bound Codex review contracts for MUVY small-ball annotations."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

MUVY_BALL_REVIEW_PLAN_SCHEMA = "agu.muvy-ball-review-plan.v1"
MUVY_BALL_REVIEW_SCHEMA = "agu.muvy-ball-offline-review.v1"
MUVY_BALL_DECISIONS = {"valid_ball", "false_positive", "uncertain"}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CANDIDATE_ID = re.compile(r"muvy-ball-(\d{4})")
_INDEX_RANGE = re.compile(r"(\d+)(?:-(\d+))?")


def build_muvy_ball_review_decisions(
    candidate_ids: Sequence[str],
    *,
    valid_ranges: Sequence[str],
    uncertain_ranges: Sequence[str],
) -> list[dict[str, str]]:
    """Expand a compact, human-authored review into exact candidate coverage."""

    observed_indices: dict[int, str] = {}
    for candidate_id in candidate_ids:
        match = _CANDIDATE_ID.fullmatch(candidate_id)
        if match is None:
            raise ValueError("invalid MUVY ball candidate id")
        index = int(match.group(1))
        if index in observed_indices:
            raise ValueError("duplicate MUVY ball candidate id")
        observed_indices[index] = candidate_id

    valid = _expand_index_ranges(valid_ranges)
    uncertain = _expand_index_ranges(uncertain_ranges)
    if valid & uncertain:
        raise ValueError("MUVY ball review ranges overlap")
    selected = valid | uncertain
    if selected - observed_indices.keys():
        raise ValueError("MUVY ball review range is outside candidate coverage")

    decisions = []
    for index, candidate_id in observed_indices.items():
        if index in valid:
            decision = "valid_ball"
            notes = "Visible basketball inside the proposed box."
        elif index in uncertain:
            decision = "uncertain"
            notes = "Proposed box is too ambiguous for a reliable ball label."
        else:
            decision = "false_positive"
            notes = "No visible basketball inside the proposed box."
        decisions.append(
            {
                "candidate_id": candidate_id,
                "decision": decision,
                "notes": notes,
            }
        )
    return decisions


def _expand_index_ranges(ranges: Sequence[str]) -> set[int]:
    indices: set[int] = set()
    for value in ranges:
        match = _INDEX_RANGE.fullmatch(value)
        if match is None:
            raise ValueError("invalid MUVY ball review range")
        start = int(match.group(1))
        stop = int(match.group(2) or start)
        if start < 1 or stop < start:
            raise ValueError("invalid MUVY ball review range")
        indices.update(range(start, stop + 1))
    return indices


def parse_muvy_detection_line(line: str) -> dict[str, Any]:
    """Parse MUVY's whitespace format with a multi-token class name."""

    fields = line.split()
    if len(fields) < 7:
        raise ValueError("invalid MUVY detection line")
    try:
        frame_id = int(fields[0])
        confidence = float(fields[-5])
        bbox = [float(value) for value in fields[-4:]]
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid MUVY detection values") from exc
    object_class = " ".join(fields[1:-5])
    if (
        frame_id < 1
        or not object_class
        or not 0 <= confidence <= 1
        or not _valid_bbox(bbox)
    ):
        raise ValueError("invalid MUVY detection values")
    return {
        "frame_id": frame_id,
        "object_class": object_class,
        "confidence": confidence,
        "bbox": bbox,
    }


def is_small_ball_candidate(
    bbox: Sequence[float],
    *,
    frame_width: int,
    frame_height: int,
    maximum_area_ratio: float = 0.005,
    maximum_aspect_ratio: float = 2.5,
) -> bool:
    """Apply a label-independent small-object geometry screen."""

    if (
        frame_width < 1
        or frame_height < 1
        or maximum_area_ratio <= 0
        or maximum_aspect_ratio < 1
        or not _valid_bbox(bbox)
    ):
        raise ValueError("invalid MUVY small-ball geometry")
    x1, y1, x2, y2 = map(float, bbox)
    if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
        raise ValueError("MUVY candidate box exceeds its frame")
    width = x2 - x1
    height = y2 - y1
    area_ratio = width * height / (frame_width * frame_height)
    aspect_ratio = max(width / height, height / width)
    return (
        area_ratio <= maximum_area_ratio
        and aspect_ratio <= maximum_aspect_ratio
    )


def seal_muvy_ball_review_plan(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["schema_version"] = MUVY_BALL_REVIEW_PLAN_SCHEMA
    artifact["purpose"] = "codex_offline_muvy_ball_annotation_audit"
    artifact["runtime_consumable"] = False
    artifact["codex_runtime_answer_used"] = False
    artifact["reviewer_visible_fields"] = [
        "candidate_id",
        "raw_frame",
        "proposed_bbox",
    ]
    _validate_plan(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_muvy_ball_review_plan(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_plan(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("MUVY ball review plan hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def seal_muvy_ball_review(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    verified_plan = verify_muvy_ball_review_plan(plan)
    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["schema_version"] = MUVY_BALL_REVIEW_SCHEMA
    artifact["purpose"] = "codex_offline_muvy_ball_annotation_audit"
    artifact["runtime_consumable"] = False
    artifact["codex_runtime_answer_used"] = False
    _validate_review(artifact, plan=verified_plan)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_muvy_ball_review(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    verified_plan = verify_muvy_ball_review_plan(plan)
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_review(artifact, plan=verified_plan)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("MUVY ball offline review hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _validate_plan(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != MUVY_BALL_REVIEW_PLAN_SCHEMA:
        raise ValueError("unsupported MUVY ball review plan schema")
    if (
        artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("reviewer_visible_fields")
        != ["candidate_id", "raw_frame", "proposed_bbox"]
        or artifact.get("source_license") != "CC-BY-4.0"
        or artifact.get("source_record_url")
        != "https://zenodo.org/records/13883315"
    ):
        raise ValueError("MUVY ball review plan policy is invalid")
    _require_sha256(artifact.get("source_manifest_sha256"))
    candidates = artifact.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("MUVY ball review plan requires candidates")
    observed = set()
    for index, row in enumerate(candidates, start=1):
        candidate_id = str(row.get("candidate_id") or "")
        bbox = row.get("bbox")
        width = int(row.get("frame_width", 0))
        height = int(row.get("frame_height", 0))
        if (
            candidate_id != f"muvy-ball-{index:04d}"
            or candidate_id in observed
            or not str(row.get("video_path") or "").endswith(".mp4")
            or int(row.get("frame_id", 0)) < 1
            or int(row.get("frame_index", -1))
            != int(row.get("frame_id", 0)) - 1
            or not isinstance(bbox, list)
            or not is_small_ball_candidate(
                bbox,
                frame_width=width,
                frame_height=height,
            )
            or not math.isfinite(float(row.get("confidence", math.nan)))
        ):
            raise ValueError("invalid or non-canonical MUVY ball candidate")
        _require_sha256(row.get("video_sha256"))
        _require_sha256(row.get("annotation_sha256"))
        _require_sha256(row.get("frame_sha256"))
        observed.add(candidate_id)


def _validate_review(
    artifact: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> None:
    if artifact.get("schema_version") != MUVY_BALL_REVIEW_SCHEMA:
        raise ValueError("unsupported MUVY ball review schema")
    if (
        artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("plan_sha256") != plan["artifact_sha256"]
        or artifact.get("reviewer") != "codex_offline_source_annotation"
    ):
        raise ValueError("MUVY ball review policy is invalid")
    decisions = artifact.get("decisions")
    planned = [row["candidate_id"] for row in plan["candidates"]]
    if (
        not isinstance(decisions, list)
        or [str(row.get("candidate_id") or "") for row in decisions]
        != planned
    ):
        raise ValueError("MUVY ball review must exactly cover the plan")
    for row in decisions:
        if (
            row.get("decision") not in MUVY_BALL_DECISIONS
            or not isinstance(row.get("notes"), str)
        ):
            raise ValueError("invalid MUVY ball review decision")


def _valid_bbox(value: Sequence[float]) -> bool:
    if not isinstance(value, Sequence) or len(value) != 4:
        return False
    try:
        x1, y1, x2, y2 = map(float, value)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(x1)
        and math.isfinite(y1)
        and math.isfinite(x2)
        and math.isfinite(y2)
        and x2 > x1
        and y2 > y1
    )


def _require_sha256(value: object) -> str:
    text = str(value or "")
    if _SHA256.fullmatch(text) is None:
        raise ValueError("invalid MUVY source SHA-256")
    return text


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
