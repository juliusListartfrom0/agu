"""Offline, label-hidden visual review contracts for PBP anchor states."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from app.analysis.pbp_visual_state_frames import (
    verify_anchor_state_embedding_artifact,
)

REVIEW_PLAN_SCHEMA = "agu.pbp-visual-state-review-plan.v1"
OFFLINE_REVIEW_SCHEMA = "agu.pbp-visual-state-offline-review.v1"
LABEL_CORRECTION_SCHEMA = "agu.pbp-visual-state-label-corrections.v1"

VISUAL_STATES = {
    "free_throw_setup",
    "live_play",
    "stoppage_other",
    "uncertain",
}
CONFIDENCE_VALUES = {"high", "medium", "low"}
REVIEWER_VISIBLE_FIELDS = ["review_id", "frame_indexes", "raw_frames"]
FOLLOWUP_OFFSETS_SECONDS = (
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

_SHA256 = re.compile(r"[0-9a-f]{64}")


def build_visual_state_review_plan(
    embedding_artifact: Mapping[str, Any],
    *,
    additional_sealed_blind_video_sha256s: Sequence[str] = (),
) -> dict[str, Any]:
    """Create a Codex-visible plan that omits every PBP-derived state label."""

    embeddings = verify_anchor_state_embedding_artifact(embedding_artifact)
    sources = {
        _require_sha256(value, field="source video")
        for value in embeddings["source_video_sha256s"]
    }
    blind = {
        _require_sha256(value, field="sealed blind video")
        for value in [
            *embeddings["sealed_blind_video_sha256s"],
            *additional_sealed_blind_video_sha256s,
        ]
    }
    if sources & blind:
        raise ValueError("sealed blind video cannot enter visual-state review")

    examples = []
    for index, row in enumerate(embeddings["examples"], start=1):
        source = str(row["source_video_sha256"])
        if source in blind:
            raise ValueError("sealed blind video cannot enter visual-state review")
        examples.append(
            {
                "review_id": f"visual-state-{index:04d}",
                "source_video_sha256": source,
                "source_video_filename": str(
                    row.get("source_video_filename") or ""
                ),
                "candidate_bundle_sha256": str(
                    row["candidate_bundle_sha256"]
                ),
                "event_id": str(row["event_id"]),
                "anchor_frame": int(row["anchor_frame"]),
                "source_fps": float(row["source_fps"]),
                "frame_count": int(row["frame_count"]),
                "frame_indexes": [int(value) for value in row["frame_indexes"]],
            }
        )
    artifact: dict[str, Any] = {
        "schema_version": REVIEW_PLAN_SCHEMA,
        "purpose": "codex_offline_visual_state_training_annotation",
        "annotation_scope": (
            "free_throw_setup_vs_live_play_visual_alignment_only"
        ),
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "labels_hidden_from_reviewer": True,
        "reviewer_visible_fields": list(REVIEWER_VISIBLE_FIELDS),
        "embedding_artifact_sha256": embeddings["artifact_sha256"],
        "source_video_sha256s": sorted(sources),
        "sealed_blind_video_sha256s": sorted(blind),
        "anchor_offsets_seconds": list(embeddings["anchor_offsets_seconds"]),
        "examples": examples,
    }
    _validate_visual_state_review_plan(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_visual_state_review_plan(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_visual_state_review_plan(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("visual-state review plan hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def verify_visual_state_label_corrections(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    verified_plan = verify_visual_state_review_plan(plan)
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    if artifact.get("schema_version") != LABEL_CORRECTION_SCHEMA:
        raise ValueError("unsupported visual-state label-correction schema")
    if claimed != _canonical_sha256(artifact):
        raise ValueError("visual-state label-correction hash mismatch")
    if (
        artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("plan_sha256") != verified_plan["artifact_sha256"]
    ):
        raise ValueError("visual-state label-correction provenance is invalid")
    _require_sha256(
        artifact.get("embedding_artifact_sha256"),
        field="embedding artifact",
    )
    _require_sha256(artifact.get("review_sha256"), field="review")
    plan_by_id = {
        str(row["review_id"]): row for row in verified_plan["examples"]
    }
    decisions = artifact.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("visual-state label corrections require decisions")
    decision_ids = [str(row.get("review_id") or "") for row in decisions]
    if (
        len(decision_ids) != len(set(decision_ids))
        or set(decision_ids) != set(plan_by_id)
    ):
        raise ValueError("visual-state label corrections must cover the plan")
    resolved = 0
    changed = 0
    for row in decisions:
        planned = plan_by_id[str(row["review_id"])]
        if (
            str(row.get("source_video_sha256") or "")
            != str(planned["source_video_sha256"])
            or str(row.get("candidate_bundle_sha256") or "")
            != str(planned["candidate_bundle_sha256"])
            or str(row.get("event_id") or "") != str(planned["event_id"])
            or row.get("original_state") not in {"free_throw", "field_goal"}
            or row.get("corrected_state")
            not in {None, "free_throw", "field_goal"}
            or row.get("visual_state") not in VISUAL_STATES
            or row.get("review_confidence") not in CONFIDENCE_VALUES
        ):
            raise ValueError("invalid visual-state label-correction decision")
        if row["corrected_state"] is not None:
            resolved += 1
            changed += row["corrected_state"] != row["original_state"]
    expected_summary = {
        "reviewed": len(decisions),
        "resolved": resolved,
        "changed": changed,
        "unresolved": len(decisions) - resolved,
    }
    if artifact.get("summary") != expected_summary:
        raise ValueError("visual-state label-correction summary mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def build_visual_state_followup_review_plan(
    plan: Mapping[str, Any],
    *,
    corrections: Mapping[str, Any],
    offsets_seconds: Sequence[float] = FOLLOWUP_OFFSETS_SECONDS,
) -> dict[str, Any]:
    """Build a wider label-hidden plan for only unresolved first-pass rows."""

    verified_plan = verify_visual_state_review_plan(plan)
    verified_corrections = verify_visual_state_label_corrections(
        corrections,
        plan=verified_plan,
    )
    offsets = tuple(float(value) for value in offsets_seconds)
    if offsets != FOLLOWUP_OFFSETS_SECONDS:
        raise ValueError("unsupported visual-state follow-up offsets")
    unresolved = {
        str(row["review_id"])
        for row in verified_corrections["decisions"]
        if row["corrected_state"] is None
    }
    if not unresolved:
        raise ValueError("visual-state follow-up requires unresolved rows")
    examples = []
    sources: set[str] = set()
    for row in verified_plan["examples"]:
        if str(row["review_id"]) not in unresolved:
            continue
        anchor = int(row["anchor_frame"])
        fps = float(row["source_fps"])
        frame_count = int(row["frame_count"])
        frame_indexes = [
            round(anchor + offset * fps) for offset in offsets
        ]
        if (
            len(set(frame_indexes)) != len(frame_indexes)
            or frame_indexes[0] < 0
            or frame_indexes[-1] >= frame_count
            or frame_indexes[offsets.index(0.0)] != anchor
        ):
            raise ValueError("visual-state follow-up window is incomplete")
        example = dict(row)
        example["frame_indexes"] = frame_indexes
        examples.append(example)
        sources.add(str(row["source_video_sha256"]))
    artifact: dict[str, Any] = {
        "schema_version": REVIEW_PLAN_SCHEMA,
        "purpose": "codex_offline_visual_state_followup_training_annotation",
        "annotation_scope": (
            "wider_free_throw_setup_vs_live_play_visual_alignment_only"
        ),
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "labels_hidden_from_reviewer": True,
        "reviewer_visible_fields": list(REVIEWER_VISIBLE_FIELDS),
        "embedding_artifact_sha256": verified_plan[
            "embedding_artifact_sha256"
        ],
        "parent_plan_sha256": verified_plan["artifact_sha256"],
        "parent_corrections_sha256": verified_corrections[
            "artifact_sha256"
        ],
        "source_video_sha256s": sorted(sources),
        "sealed_blind_video_sha256s": list(
            verified_plan["sealed_blind_video_sha256s"]
        ),
        "anchor_offsets_seconds": list(offsets),
        "examples": examples,
    }
    _validate_visual_state_review_plan(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def seal_visual_state_review(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    verified_plan = verify_visual_state_review_plan(plan)
    artifact = dict(payload)
    artifact["schema_version"] = OFFLINE_REVIEW_SCHEMA
    artifact["purpose"] = "codex_offline_visual_state_training_annotation"
    artifact["runtime_consumable"] = False
    artifact["codex_runtime_answer_used"] = False
    artifact.pop("artifact_sha256", None)
    _validate_visual_state_review(artifact, plan=verified_plan)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_visual_state_review(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    verified_plan = verify_visual_state_review_plan(plan)
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_visual_state_review(artifact, plan=verified_plan)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("visual-state offline review hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def derive_visual_state_label_corrections(
    *,
    embedding_artifact: Mapping[str, Any],
    plan: Mapping[str, Any],
    review: Mapping[str, Any],
) -> dict[str, Any]:
    embeddings = verify_anchor_state_embedding_artifact(embedding_artifact)
    verified_plan = verify_visual_state_review_plan(plan)
    verified_review = verify_visual_state_review(review, plan=verified_plan)
    if (
        verified_plan["embedding_artifact_sha256"]
        != embeddings["artifact_sha256"]
    ):
        raise ValueError("visual-state review plan does not match embeddings")

    original_by_key = {
        (
            str(row["source_video_sha256"]),
            str(row["candidate_bundle_sha256"]),
            str(row["event_id"]),
        ): str(row["state"])
        for row in embeddings["examples"]
    }
    plan_by_id = {
        str(row["review_id"]): row for row in verified_plan["examples"]
    }
    decisions = []
    changed = 0
    resolved = 0
    for review_row in verified_review["reviews"]:
        planned = plan_by_id[str(review_row["review_id"])]
        key = (
            str(planned["source_video_sha256"]),
            str(planned["candidate_bundle_sha256"]),
            str(planned["event_id"]),
        )
        if key not in original_by_key:
            raise ValueError("visual-state review is absent from embeddings")
        original_state = original_by_key[key]
        corrected_state = _corrected_state(review_row)
        if corrected_state is not None:
            resolved += 1
            changed += corrected_state != original_state
        decisions.append(
            {
                "review_id": review_row["review_id"],
                "source_video_sha256": key[0],
                "candidate_bundle_sha256": key[1],
                "event_id": key[2],
                "original_state": original_state,
                "corrected_state": corrected_state,
                "visual_state": review_row["visual_state"],
                "review_confidence": review_row["confidence"],
            }
        )
    artifact: dict[str, Any] = {
        "schema_version": LABEL_CORRECTION_SCHEMA,
        "purpose": "pbp_visual_state_training_label_correction_only",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "embedding_artifact_sha256": embeddings["artifact_sha256"],
        "plan_sha256": verified_plan["artifact_sha256"],
        "review_sha256": verified_review["artifact_sha256"],
        "policy": (
            "high_or_medium_free_throw_setup_maps_to_free_throw; "
            "high_or_medium_live_play_maps_to_field_goal; "
            "other_low_or_uncertain_is_unresolved"
        ),
        "summary": {
            "reviewed": len(decisions),
            "resolved": resolved,
            "changed": changed,
            "unresolved": len(decisions) - resolved,
        },
        "decisions": decisions,
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def _validate_visual_state_review_plan(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != REVIEW_PLAN_SCHEMA:
        raise ValueError("unsupported visual-state review plan schema")
    if (
        artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("labels_hidden_from_reviewer") is not True
        or artifact.get("reviewer_visible_fields") != REVIEWER_VISIBLE_FIELDS
    ):
        raise ValueError("visual-state review plan must be label-hidden and offline")
    _require_sha256(
        artifact.get("embedding_artifact_sha256"),
        field="embedding artifact",
    )
    sources = _require_hashes(
        artifact.get("source_video_sha256s"),
        field="source video",
    )
    blind = _require_hashes(
        artifact.get("sealed_blind_video_sha256s"),
        field="sealed blind video",
    )
    if sources & blind:
        raise ValueError("sealed blind video cannot enter visual-state review")
    examples = artifact.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("visual-state review plan requires examples")
    seen_ids: set[str] = set()
    seen_events: set[tuple[str, str]] = set()
    observed_sources: set[str] = set()
    for row in examples:
        if not isinstance(row, Mapping):
            raise ValueError("invalid visual-state review example")
        if "state" in row or "label" in row:
            raise ValueError("visual-state review examples must hide labels")
        review_id = str(row.get("review_id") or "")
        source = _require_sha256(
            row.get("source_video_sha256"),
            field="source video",
        )
        event_id = str(row.get("event_id") or "")
        event_key = (source, event_id)
        if (
            not review_id
            or review_id in seen_ids
            or not event_id
            or event_key in seen_events
            or source not in sources
            or source in blind
        ):
            raise ValueError("invalid or duplicate visual-state review example")
        seen_ids.add(review_id)
        seen_events.add(event_key)
        observed_sources.add(source)
        _require_sha256(
            row.get("candidate_bundle_sha256"),
            field="candidate bundle",
        )
        frames = row.get("frame_indexes")
        if (
            not isinstance(frames, list)
            or not frames
            or any(isinstance(value, bool) or int(value) < 0 for value in frames)
            or len({int(value) for value in frames}) != len(frames)
        ):
            raise ValueError("invalid visual-state review frame indexes")
    if observed_sources != sources:
        raise ValueError("visual-state review does not cover declared sources")


def _validate_visual_state_review(
    artifact: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> None:
    if artifact.get("schema_version") != OFFLINE_REVIEW_SCHEMA:
        raise ValueError("unsupported visual-state offline review schema")
    if (
        artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("plan_sha256") != plan["artifact_sha256"]
        or artifact.get("reviewer") != "codex_offline_visual_review"
    ):
        raise ValueError("visual-state review provenance is invalid")
    reviews = artifact.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("visual-state review requires reviews")
    planned_ids = {str(row["review_id"]) for row in plan["examples"]}
    review_ids = [str(row.get("review_id") or "") for row in reviews]
    if len(review_ids) != len(set(review_ids)) or set(review_ids) != planned_ids:
        raise ValueError("visual-state reviews must exactly cover the plan")
    for row in reviews:
        if (
            not isinstance(row, Mapping)
            or row.get("visual_state") not in VISUAL_STATES
            or row.get("confidence") not in CONFIDENCE_VALUES
            or not isinstance(row.get("notes", ""), str)
        ):
            raise ValueError("invalid visual-state review decision")


def _corrected_state(review: Mapping[str, Any]) -> str | None:
    if review["confidence"] not in {"high", "medium"}:
        return None
    if review["visual_state"] == "free_throw_setup":
        return "free_throw"
    if review["visual_state"] == "live_play":
        return "field_goal"
    return None


def _require_hashes(value: object, *, field: str) -> set[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} hashes are required")
    hashes = {_require_sha256(item, field=field) for item in value}
    if len(hashes) != len(value):
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
