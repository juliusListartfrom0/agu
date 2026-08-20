"""Hash-bound, offline-only manual causal review for continuous VRU clips.

The artifact records only evidence that a reviewer can see in the original
frames.  It is deliberately not an inference input: no runtime endpoint may
consume this review or use its labels as an answer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from app.analysis.continuous_causal_selection import neutral_continuous_review_id

PLAN_SCHEMA = "agu.vru-causal-review-plan.v1"
REVIEW_SCHEMA = "agu.vru-causal-offline-review.v1"
PLAN_PURPOSE = "codex_offline_vru_ball_hand_rim_causal_annotation"
PLAN_ANNOTATION_SCOPE = "original_continuous_frames_only_release_rim_proximity_and_outcome"
PLAN_REVIEW_RULES = {
    "release": "earliest frame where the ball is visibly separated from the controlled shooter's hand",
    "rim": "frame nearest the ball reaching the rim plane or net; proximity is not contact",
    "outcome": "made or missed only when the rim/net result is visually resolved; otherwise unknown",
    "negative": "not_a_shot requires no visible release-to-rim chain in the planned window",
}
SHOT_SEQUENCES = {"shot", "not_a_shot", "uncertain"}
OUTCOMES = {"made", "missed", "unknown", "not_applicable"}
CONFIDENCE_VALUES = {"high", "medium", "low"}
EVIDENCE_FIELDS = (
    "controlled_ball_before_release",
    "ball_separated_from_hand",
    "ball_progresses_toward_rim",
    "rim_proximity_visible",
    "rim_contact_visible",
)
REVIEWER_VISIBLE_FIELDS = (
    "review_id",
    "source_video_filename",
    "source_fps",
    "frame_indexes",
    "frame_sha256s",
)
SELECTION_BOUND_REVIEWER_VISIBLE_FIELDS = (
    "review_id",
    "source_video_handle",
    "source_fps",
    "frame_indexes",
    "frame_sha256s",
)
SELECTION_BOUND_EXAMPLE_COUNT = 24
SELECTION_BOUND_SAMPLE_COUNT = 64
SELECTION_BOUND_SAMPLE_RATE_HZ = 8.0
_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "purpose",
        "annotation_scope",
        "runtime_consumable",
        "codex_runtime_answer_used",
        "labels_hidden_from_reviewer",
        "reviewer_visible_fields",
        "source_manifest_sha256",
        "source_video_sha256s",
        "review_rules",
        "examples",
    }
)
_SELECTION_BOUND_PLAN_FIELDS = _PLAN_FIELDS | {"source_selection_artifact_sha256"}
_PLAN_EXAMPLE_INPUT_FIELDS = frozenset(
    {
        "review_id",
        "source_video_sha256",
        "source_video_filename",
        "source_fps",
        "frame_count",
        "frame_indexes",
    }
)
_PLAN_EXAMPLE_OPTIONAL_INPUT_FIELDS = frozenset({"frame_sha256s"})
_SELECTION_BOUND_PLAN_EXAMPLE_FIELDS = (
    _PLAN_EXAMPLE_INPUT_FIELDS | _PLAN_EXAMPLE_OPTIONAL_INPUT_FIELDS | {"source_video_handle"}
)
_REVIEW_INPUT_FIELDS = (
    "review_id",
    "shot_sequence",
    "release_position",
    "rim_position",
    "outcome",
    "confidence",
    "evidence",
    "notes",
)
_OPTIONAL_REVIEW_INPUT_FIELDS = frozenset({"release_position", "rim_position"})
_SEALED_REVIEW_ALLOWED_FIELDS = frozenset(
    (
        *_REVIEW_INPUT_FIELDS,
        "source_video_sha256",
        "source_video_filename",
        "release_frame",
        "rim_frame",
    )
)
_SEALED_REVIEW_REQUIRED_FIELDS = _SEALED_REVIEW_ALLOWED_FIELDS - _OPTIONAL_REVIEW_INPUT_FIELDS
_SEALED_ARTIFACT_FIELDS = frozenset(
    {
        "plan_sha256",
        "reviewer",
        "reviews",
        "schema_version",
        "purpose",
        "runtime_consumable",
        "codex_runtime_answer_used",
        "labels_hidden_from_reviewer",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVIEW_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


def validate_vru_causal_review_id(value: Any) -> str:
    """Return one path-safe review ID or fail closed."""

    if not isinstance(value, str) or _REVIEW_ID.fullmatch(value) is None:
        raise ValueError("VRU causal review ID must be a path-safe scalar")
    return value


def build_vru_causal_review_plan(
    *,
    source_manifest_sha256: str,
    examples: Sequence[Mapping[str, Any]],
    source_selection_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a label-hidden frame plan from hash-bound continuous source clips."""

    source_manifest_sha256 = _require_sha(source_manifest_sha256, "source manifest")
    selection_sha = (
        None
        if source_selection_artifact_sha256 is None
        else _require_sha(source_selection_artifact_sha256, "source selection artifact")
    )
    if not examples:
        raise ValueError("VRU causal review plan requires examples")
    if selection_sha is not None and len(examples) != SELECTION_BOUND_EXAMPLE_COUNT:
        raise ValueError("selection-bound review plan requires exactly 24 examples")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_sources: set[str] = set()
    for ordinal, value in enumerate(examples, start=1):
        row = dict(value)
        if selection_sha is not None and not (
            _PLAN_EXAMPLE_INPUT_FIELDS <= set(row)
            and set(row) <= _PLAN_EXAMPLE_INPUT_FIELDS | _PLAN_EXAMPLE_OPTIONAL_INPUT_FIELDS
        ):
            raise ValueError("selection-bound review example fields are not canonical")
        review_id = validate_vru_causal_review_id(row.get("review_id"))
        source_sha = _require_sha(row.get("source_video_sha256"), "source video")
        filename_value = row.get("source_video_filename")
        filename = filename_value if isinstance(filename_value, str) else ""
        fps = _finite_float(row.get("source_fps"), "source FPS")
        frame_count = _positive_int(row.get("frame_count"), "frame count")
        frame_indexes = row.get("frame_indexes")
        if not review_id or review_id in seen_ids:
            raise ValueError("VRU causal review IDs must be unique")
        if selection_sha is not None and review_id != neutral_continuous_review_id(
            source_sha,
            ordinal,
        ):
            raise ValueError("selection-bound plan requires a neutral review ID")
        if not filename:
            raise ValueError("VRU causal review filename is required")
        if not isinstance(frame_indexes, list) or len(frame_indexes) < 3:
            raise ValueError("VRU causal review requires at least three frames")
        if selection_sha is not None and (
            len(frame_indexes) != SELECTION_BOUND_SAMPLE_COUNT
            or any(isinstance(index, bool) or not isinstance(index, int) for index in frame_indexes)
        ):
            raise ValueError("selection-bound review examples require 64 integer frames")
        indexes = [_frame_index(index, frame_count) for index in frame_indexes]
        if indexes != sorted(set(indexes)):
            raise ValueError("VRU causal frame indexes must be sorted and unique")
        if selection_sha is not None:
            _verify_selection_bound_sampling(indexes, fps)
        frame_sha256s = row.get("frame_sha256s", {})
        if not isinstance(frame_sha256s, Mapping):
            raise ValueError("VRU causal frame hashes must be an object")
        normalized_hashes: dict[str, str] = {}
        for key, digest in frame_sha256s.items():
            index = _frame_index(key, frame_count)
            normalized_hashes[str(index)] = _require_sha(digest, "frame")
        if any(int(key) not in indexes for key in normalized_hashes):
            raise ValueError("VRU causal frame hash is outside the planned window")
        if selection_sha is not None and set(normalized_hashes) != {str(index) for index in indexes}:
            raise ValueError("selection-bound plan requires all 64 private frame hashes")
        seen_ids.add(review_id)
        seen_sources.add(source_sha)
        normalized_row = {
            "review_id": review_id,
            "source_video_sha256": source_sha,
            "source_video_filename": (
                neutral_vru_source_filename(source_sha) if selection_sha is not None else filename
            ),
            "source_fps": fps,
            "frame_count": frame_count,
            "frame_indexes": indexes,
            "frame_sha256s": dict(sorted(normalized_hashes.items())),
        }
        if selection_sha is not None:
            normalized_row["source_video_handle"] = neutral_vru_source_handle(source_sha)
        normalized.append(normalized_row)

    if selection_sha is not None and len(seen_sources) != 1:
        raise ValueError("selection-bound plan requires exactly one private source hash")

    artifact: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "purpose": PLAN_PURPOSE,
        "annotation_scope": PLAN_ANNOTATION_SCOPE,
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "labels_hidden_from_reviewer": True,
        "reviewer_visible_fields": list(
            SELECTION_BOUND_REVIEWER_VISIBLE_FIELDS if selection_sha is not None else REVIEWER_VISIBLE_FIELDS
        ),
        "source_manifest_sha256": source_manifest_sha256,
        "source_video_sha256s": sorted(seen_sources),
        "review_rules": dict(PLAN_REVIEW_RULES),
        "examples": normalized,
    }
    if selection_sha is not None:
        artifact["source_selection_artifact_sha256"] = selection_sha
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def neutral_vru_source_handle(source_video_sha256: str) -> str:
    """Return the only source identifier exposed by selection-bound plans."""

    source_sha = _require_sha(source_video_sha256, "source video")
    token = hashlib.sha256(f"agu.vru-neutral-source.v1\0{source_sha}".encode()).hexdigest()
    return f"source-{token[:24]}"


def neutral_vru_source_filename(source_video_sha256: str) -> str:
    """Return an opaque display-safe filename for a selection-bound source."""

    return f"{neutral_vru_source_handle(source_video_sha256)}.media"


def verify_vru_causal_review_plan(
    payload: Mapping[str, Any],
    *,
    expected_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed_value = artifact.pop("artifact_sha256", "")
    claimed = claimed_value if isinstance(claimed_value, str) else ""
    _validate_plan(artifact)
    if not claimed or claimed != _canonical_sha256(artifact):
        raise ValueError("VRU causal review plan hash mismatch")
    selection_bound = artifact.get("source_selection_artifact_sha256") is not None
    if selection_bound and expected_artifact_sha256 is None:
        raise ValueError("selection-bound plan requires an external plan receipt")
    if expected_artifact_sha256 is not None:
        expected_sha = _require_sha(expected_artifact_sha256, "expected plan artifact")
        if claimed != expected_sha:
            raise ValueError("external plan receipt does not match the sealed plan")
    artifact["artifact_sha256"] = claimed
    return artifact


def seal_vru_causal_review(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    expected_plan_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Normalize reviewer positions into source frame indexes and seal them."""

    verified_plan = verify_vru_causal_review_plan(
        plan,
        expected_artifact_sha256=expected_plan_artifact_sha256,
    )
    planned = {str(row["review_id"]): row for row in verified_plan["examples"]}
    reviews = payload.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("VRU causal review requires a reviews list")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for value in reviews:
        if not isinstance(value, Mapping):
            raise ValueError("VRU causal review rows must be objects")
        row = dict(value)
        review_id = str(row.get("review_id") or "")
        if review_id not in planned or review_id in seen:
            raise ValueError("VRU causal reviews must exactly cover the plan")
        seen.add(review_id)
        sequence = str(row.get("shot_sequence") or "")
        outcome = str(row.get("outcome") or "")
        confidence = str(row.get("confidence") or "")
        if sequence not in SHOT_SEQUENCES:
            raise ValueError("unsupported VRU causal shot sequence")
        if outcome not in OUTCOMES:
            raise ValueError("unsupported VRU causal outcome")
        if confidence not in CONFIDENCE_VALUES:
            raise ValueError("unsupported VRU causal confidence")
        evidence = row.get("evidence")
        if not isinstance(evidence, Mapping) or set(evidence) != set(EVIDENCE_FIELDS):
            raise ValueError("VRU causal evidence must contain the exact evidence fields")
        normalized_evidence = {name: _bool(evidence[name], f"evidence.{name}") for name in EVIDENCE_FIELDS}
        release_position = _optional_position(row.get("release_position"), planned[review_id])
        rim_position = _optional_position(row.get("rim_position"), planned[review_id])
        if sequence == "shot":
            if release_position is None or rim_position is None:
                raise ValueError("shot review requires release and rim positions")
            if release_position >= rim_position:
                raise ValueError("release must precede rim evidence")
            if outcome == "not_applicable":
                raise ValueError("shot review cannot use not_applicable outcome")
            if not all(
                normalized_evidence[name]
                for name in (
                    "controlled_ball_before_release",
                    "ball_separated_from_hand",
                    "ball_progresses_toward_rim",
                    "rim_proximity_visible",
                )
            ):
                raise ValueError("shot review requires the visible causal evidence chain")
        elif sequence == "not_a_shot":
            if release_position is not None or rim_position is not None:
                raise ValueError("not-a-shot review cannot contain release or rim positions")
            if outcome != "not_applicable":
                raise ValueError("not-a-shot review requires not_applicable outcome")
            if any(normalized_evidence.values()):
                raise ValueError("not-a-shot review cannot claim causal evidence")
        else:
            if outcome == "not_applicable":
                raise ValueError("uncertain review requires an unknown outcome")
            if release_position is not None and rim_position is not None and release_position >= rim_position:
                raise ValueError("release must precede rim evidence")
        row["source_video_sha256"] = planned[review_id]["source_video_sha256"]
        row["source_video_filename"] = planned[review_id]["source_video_filename"]
        row["release_frame"] = _position_frame(release_position, planned[review_id])
        row["rim_frame"] = _position_frame(rim_position, planned[review_id])
        row["evidence"] = normalized_evidence
        row["shot_sequence"] = sequence
        row["outcome"] = outcome
        row["confidence"] = confidence
        row["notes"] = str(row.get("notes") or "")
        normalized.append(row)
    if seen != set(planned):
        raise ValueError("VRU causal reviews must exactly cover the plan")
    artifact: dict[str, Any] = {
        "plan_sha256": verified_plan["artifact_sha256"],
        "reviewer": str(payload.get("reviewer") or "codex_offline_training_annotation"),
        "reviews": sorted(normalized, key=lambda row: str(row["review_id"])),
        "schema_version": REVIEW_SCHEMA,
        "purpose": "codex_offline_vru_ball_hand_rim_causal_annotation",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "labels_hidden_from_reviewer": True,
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_vru_causal_review(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    expected_plan_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    verified_plan = verify_vru_causal_review_plan(
        plan,
        expected_artifact_sha256=expected_plan_artifact_sha256,
    )
    artifact = dict(payload)
    claimed_value = artifact.pop("artifact_sha256", "")
    claimed = claimed_value if isinstance(claimed_value, str) else ""
    if set(artifact) != _SEALED_ARTIFACT_FIELDS:
        raise ValueError("VRU causal review must contain the exact artifact fields")
    if artifact.get("schema_version") != REVIEW_SCHEMA:
        raise ValueError("unsupported VRU causal review schema")
    if artifact.get("runtime_consumable") is not False:
        raise ValueError("VRU causal review cannot be runtime consumable")
    if artifact.get("codex_runtime_answer_used") is not False:
        raise ValueError("VRU causal review cannot be a runtime answer channel")
    if artifact.get("plan_sha256") != verified_plan["artifact_sha256"]:
        raise ValueError("VRU causal review plan binding mismatch")
    if not claimed or claimed != _canonical_sha256(artifact):
        raise ValueError("VRU causal review hash mismatch")
    reviewer = artifact.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer:
        raise ValueError("VRU causal review reviewer must be a non-empty string")
    reviews = artifact.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("VRU causal review requires a reviews list")
    review_inputs: list[dict[str, Any]] = []
    for value in reviews:
        if not isinstance(value, Mapping):
            raise ValueError("VRU causal sealed rows must contain the exact review fields")
        fields = set(value)
        if not _SEALED_REVIEW_REQUIRED_FIELDS <= fields <= _SEALED_REVIEW_ALLOWED_FIELDS:
            raise ValueError("VRU causal sealed rows must contain the exact review fields")
        review_inputs.append({name: value[name] for name in _REVIEW_INPUT_FIELDS if name in value})

    expected = seal_vru_causal_review(
        {"reviewer": reviewer, "reviews": review_inputs},
        plan=verified_plan,
        expected_plan_artifact_sha256=expected_plan_artifact_sha256,
    )
    artifact["artifact_sha256"] = claimed
    if artifact != expected:
        raise ValueError("VRU causal review semantic validation mismatch")
    return artifact


def _validate_plan(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("unsupported VRU causal review plan schema")
    if artifact.get("runtime_consumable") is not False:
        raise ValueError("VRU causal review plan cannot be runtime consumable")
    if artifact.get("codex_runtime_answer_used") is not False:
        raise ValueError("VRU causal review plan cannot be a runtime answer channel")
    if artifact.get("labels_hidden_from_reviewer") is not True:
        raise ValueError("VRU causal review plan must hide labels")
    _require_sha(artifact.get("source_manifest_sha256"), "source manifest")
    selection_sha = artifact.get("source_selection_artifact_sha256")
    if selection_sha is not None:
        _require_sha(selection_sha, "source selection artifact")
        if set(artifact) != _SELECTION_BOUND_PLAN_FIELDS:
            raise ValueError("selection-bound plan fields are not canonical")
        if (
            artifact.get("purpose") != PLAN_PURPOSE
            or artifact.get("annotation_scope") != PLAN_ANNOTATION_SCOPE
            or artifact.get("review_rules") != PLAN_REVIEW_RULES
        ):
            raise ValueError("selection-bound plan contract is not canonical")
        if artifact.get("reviewer_visible_fields") != list(SELECTION_BOUND_REVIEWER_VISIBLE_FIELDS):
            raise ValueError("selection-bound reviewer fields are not canonical")
    elif artifact.get("reviewer_visible_fields") == list(SELECTION_BOUND_REVIEWER_VISIBLE_FIELDS):
        raise ValueError("selection-bound plan requires its selection receipt")
    examples = artifact.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("VRU causal review plan requires examples")
    if selection_sha is not None and len(examples) != SELECTION_BOUND_EXAMPLE_COUNT:
        raise ValueError("selection-bound review plan requires exactly 24 examples")
    ids = [validate_vru_causal_review_id(row.get("review_id")) for row in examples if isinstance(row, Mapping)]
    if len(ids) != len(examples) or not ids or len(ids) != len(set(ids)):
        raise ValueError("VRU causal review plan IDs are invalid")
    seen_sources: set[str] = set()
    selection_fps: float | None = None
    selection_frame_count: int | None = None
    for ordinal, row in enumerate(examples, start=1):
        if not isinstance(row, Mapping):
            raise ValueError("VRU causal review plan rows must be objects")
        source_sha = _require_sha(row.get("source_video_sha256"), "source video")
        seen_sources.add(source_sha)
        if selection_sha is not None and set(row) != _SELECTION_BOUND_PLAN_EXAMPLE_FIELDS:
            raise ValueError("selection-bound review example fields are not canonical")
        if selection_sha is not None and row.get("review_id") != (neutral_continuous_review_id(source_sha, ordinal)):
            raise ValueError("selection-bound plan requires neutral review IDs")
        if selection_sha is not None and row.get("source_video_handle") != (neutral_vru_source_handle(source_sha)):
            raise ValueError("selection-bound source handle is not canonical")
        if selection_sha is not None and row.get("source_video_filename") != (neutral_vru_source_filename(source_sha)):
            raise ValueError("selection-bound source filename is not neutral")
        source_fps = _finite_float(row.get("source_fps"), "source FPS")
        frame_count = _positive_int(row.get("frame_count"), "frame count")
        if selection_sha is not None:
            if selection_fps is None:
                selection_fps = source_fps
                selection_frame_count = frame_count
            elif source_fps != selection_fps or frame_count != selection_frame_count:
                raise ValueError("selection-bound source geometry is inconsistent")
        frame_indexes = row.get("frame_indexes")
        if not isinstance(frame_indexes, list) or len(frame_indexes) < 3:
            raise ValueError("VRU causal review requires at least three frames")
        if selection_sha is not None and (
            len(frame_indexes) != SELECTION_BOUND_SAMPLE_COUNT
            or any(isinstance(index, bool) or not isinstance(index, int) for index in frame_indexes)
        ):
            raise ValueError("selection-bound review examples require 64 integer frames")
        indexes = [_frame_index(index, frame_count) for index in frame_indexes]
        if indexes != sorted(set(indexes)):
            raise ValueError("VRU causal frame indexes must be sorted and unique")
        if selection_sha is not None:
            _verify_selection_bound_sampling(indexes, source_fps)
        frame_hashes = row.get("frame_sha256s")
        if not isinstance(frame_hashes, Mapping):
            raise ValueError("VRU causal frame hashes must be an object")
        for key, digest in frame_hashes.items():
            if _frame_index(key, frame_count) not in indexes:
                raise ValueError("VRU causal frame hash is outside the planned window")
            _require_sha(digest, "frame")
        if selection_sha is not None and set(frame_hashes) != {str(index) for index in indexes}:
            raise ValueError("selection-bound plan requires all 64 private frame hashes")
    if selection_sha is not None and artifact.get("source_video_sha256s") != sorted(seen_sources):
        raise ValueError("selection-bound source hash list is not canonical")
    if selection_sha is not None and len(seen_sources) != 1:
        raise ValueError("selection-bound plan requires exactly one private source hash")


def _verify_selection_bound_sampling(indexes: list[int], source_fps: float) -> None:
    expected = [
        round(indexes[0] + offset * source_fps / SELECTION_BOUND_SAMPLE_RATE_HZ)
        for offset in range(SELECTION_BOUND_SAMPLE_COUNT)
    ]
    if indexes != expected:
        raise ValueError("selection-bound frames must preserve the frozen 8 FPS geometry")


def _optional_position(value: Any, plan_row: Mapping[str, Any]) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("VRU causal positions must be integer indexes")
    indexes = plan_row["frame_indexes"]
    if value < 0 or value >= len(indexes):
        raise ValueError("VRU causal position is outside the planned window")
    return value


def _position_frame(position: int | None, plan_row: Mapping[str, Any]) -> int | None:
    return None if position is None else int(plan_row["frame_indexes"][position])


def _frame_index(value: Any, frame_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("VRU causal frame index must be an integer")
    try:
        index = int(value)
    except (TypeError, ValueError):
        raise ValueError("VRU causal frame index must be an integer") from None
    if index < 0 or index >= frame_count:
        raise ValueError("VRU causal frame index is outside the source video")
    return index


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _require_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} SHA-256 must be a lowercase hex digest")
    return value


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
