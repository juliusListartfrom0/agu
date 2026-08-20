"""Hash-bound offline annotation contracts for causal basketball shot phases."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from app.analysis.pbp_visual_state_review import (
    verify_visual_state_review_plan,
)

CAUSAL_PHASE_REVIEW_PLAN_SCHEMA = "agu.causal-shot-phase-review-plan.v1"
CAUSAL_PHASE_REVIEW_SCHEMA = "agu.causal-shot-phase-offline-review.v1"

CAUSAL_PHASE_OFFSETS_SECONDS = (
    -4.0,
    -3.5,
    -3.25,
    -3.0,
    -2.75,
    -2.5,
    -2.25,
    -2.0,
    -1.75,
    -1.5,
    -1.25,
    -1.0,
    -0.75,
    -0.5,
    -0.25,
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
    4.0,
)
REVIEWER_VISIBLE_FIELDS = [
    "phase_review_id",
    "frame_indexes",
    "anchor_offsets_seconds",
    "raw_frames",
]
FORMATION_STATES = {
    "free_throw_setup",
    "live_play",
    "stoppage_other",
    "uncertain",
}
BROADCAST_CONTEXTS = {"live_action", "replay", "non_action", "uncertain"}
SHOT_SEQUENCES = {"complete", "partial", "not_a_shot", "uncertain"}
OUTCOMES = {"made", "missed", "unknown", "not_applicable"}
CONFIDENCE_VALUES = {"high", "medium", "low"}
ANNOTATION_RULES = {
    "formation_state": (
        "use the wide context frames to classify free-throw setup, live play, "
        "other stoppage, or uncertain"
    ),
    "broadcast_context": (
        "classify the visible sequence as live action, replay, non-action, or "
        "uncertain without consulting event labels"
    ),
    "release_position": (
        "earliest sampled position with the ball visibly separated from "
        "the shooter's hand"
    ),
    "rim_arrival_position": (
        "sampled position closest to the ball reaching the rim plane or "
        "crossing the net"
    ),
    "shot_sequence": (
        "complete requires release and rim evidence from the same attempt; "
        "partial requires at least one visible phase"
    ),
    "outcome": (
        "made or missed only when the rim/net result is visually resolved; "
        "otherwise use unknown"
    ),
}

_SHA256 = re.compile(r"[0-9a-f]{64}")


def build_causal_shot_phase_review_plan(
    visual_state_plan: Mapping[str, Any],
    *,
    additional_sealed_blind_video_sha256s: Sequence[str] = (),
) -> dict[str, Any]:
    """Derive a dense, label-hidden phase plan from an existing review plan."""

    base = verify_visual_state_review_plan(visual_state_plan)
    sources = {
        _require_sha256(value, field="source video")
        for value in base["source_video_sha256s"]
    }
    blind = {
        _require_sha256(value, field="sealed blind video")
        for value in [
            *base["sealed_blind_video_sha256s"],
            *additional_sealed_blind_video_sha256s,
        ]
    }
    if sources & blind:
        raise ValueError("sealed blind video cannot enter causal phase review")

    examples = []
    for index, base_row in enumerate(base["examples"], start=1):
        source_sha = _require_sha256(
            base_row["source_video_sha256"],
            field="source video",
        )
        if source_sha in blind:
            raise ValueError("sealed blind video cannot enter causal phase review")
        anchor = int(base_row["anchor_frame"])
        fps = float(base_row["source_fps"])
        frame_count = int(base_row["frame_count"])
        frame_indexes = [
            round(anchor + offset * fps)
            for offset in CAUSAL_PHASE_OFFSETS_SECONDS
        ]
        examples.append(
            {
                "phase_review_id": f"causal-phase-{index:04d}",
                "visual_state_review_id": str(base_row["review_id"]),
                "source_video_sha256": source_sha,
                "source_video_filename": str(
                    base_row.get("source_video_filename") or ""
                ),
                "candidate_bundle_sha256": _require_sha256(
                    base_row["candidate_bundle_sha256"],
                    field="candidate bundle",
                ),
                "event_id": str(base_row["event_id"]),
                "anchor_frame": anchor,
                "source_fps": fps,
                "frame_count": frame_count,
                "frame_indexes": frame_indexes,
            }
        )

    artifact: dict[str, Any] = {
        "schema_version": CAUSAL_PHASE_REVIEW_PLAN_SCHEMA,
        "purpose": "codex_offline_causal_shot_phase_training_annotation",
        "annotation_scope": (
            "formation_release_rim_arrival_and_outcome_visual_evidence_only"
        ),
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "labels_hidden_from_reviewer": True,
        "reviewer_visible_fields": list(REVIEWER_VISIBLE_FIELDS),
        "annotation_rules": dict(ANNOTATION_RULES),
        "visual_state_plan_sha256": base["artifact_sha256"],
        "source_video_sha256s": sorted(sources),
        "sealed_blind_video_sha256s": sorted(blind),
        "anchor_offsets_seconds": list(CAUSAL_PHASE_OFFSETS_SECONDS),
        "examples": examples,
    }
    _validate_plan(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_causal_shot_phase_review_plan(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_plan(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("causal shot phase review plan hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def seal_causal_shot_phase_review(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    verified_plan = verify_causal_shot_phase_review_plan(plan)
    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["schema_version"] = CAUSAL_PHASE_REVIEW_SCHEMA
    artifact["purpose"] = "codex_offline_causal_shot_phase_training_annotation"
    artifact["runtime_consumable"] = False
    artifact["codex_runtime_answer_used"] = False
    artifact["labels_hidden_from_reviewer"] = True

    planned = {
        str(row["phase_review_id"]): row
        for row in verified_plan["examples"]
    }
    normalized_reviews = []
    for value in artifact.get("reviews", []):
        row = dict(value)
        phase_review_id = str(row.get("phase_review_id") or "")
        plan_row = planned.get(phase_review_id)
        if plan_row is not None:
            row["source_video_sha256"] = plan_row["source_video_sha256"]
            row["candidate_bundle_sha256"] = plan_row[
                "candidate_bundle_sha256"
            ]
            row["event_id"] = plan_row["event_id"]
            row["release_frame"] = _position_frame(
                row.get("release_position"),
                plan_row["frame_indexes"],
            )
            row["rim_arrival_frame"] = _position_frame(
                row.get("rim_arrival_position"),
                plan_row["frame_indexes"],
            )
        normalized_reviews.append(row)
    artifact["reviews"] = normalized_reviews

    _validate_review(artifact, plan=verified_plan)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_causal_shot_phase_review(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    verified_plan = verify_causal_shot_phase_review_plan(plan)
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_review(artifact, plan=verified_plan)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("causal shot phase offline review hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _validate_plan(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != CAUSAL_PHASE_REVIEW_PLAN_SCHEMA:
        raise ValueError("unsupported causal shot phase review plan schema")
    if (
        artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("labels_hidden_from_reviewer") is not True
        or artifact.get("reviewer_visible_fields") != REVIEWER_VISIBLE_FIELDS
        or artifact.get("annotation_rules") != ANNOTATION_RULES
        or tuple(artifact.get("anchor_offsets_seconds") or ())
        != CAUSAL_PHASE_OFFSETS_SECONDS
    ):
        raise ValueError("causal shot phase review plan policy is invalid")
    _require_sha256(
        artifact.get("visual_state_plan_sha256"),
        field="visual-state plan",
    )
    sources = {
        _require_sha256(value, field="source video")
        for value in artifact.get("source_video_sha256s") or []
    }
    blind = {
        _require_sha256(value, field="sealed blind video")
        for value in artifact.get("sealed_blind_video_sha256s") or []
    }
    if not sources or sources & blind:
        raise ValueError("sealed blind video cannot enter causal phase review")
    examples = artifact.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("causal shot phase review plan requires examples")
    review_ids = [str(row.get("phase_review_id") or "") for row in examples]
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("causal shot phase review IDs must be unique")
    observed_sources = set()
    observed_events: set[tuple[str, str, str]] = set()
    for index, row in enumerate(examples, start=1):
        if row.get("phase_review_id") != f"causal-phase-{index:04d}":
            raise ValueError("causal shot phase review IDs are not canonical")
        source = _require_sha256(
            row.get("source_video_sha256"),
            field="source video",
        )
        candidate = _require_sha256(
            row.get("candidate_bundle_sha256"),
            field="candidate bundle",
        )
        event_id = str(row.get("event_id") or "")
        if source not in sources or source in blind or not event_id:
            raise ValueError("causal shot phase example provenance is invalid")
        key = (source, candidate, event_id)
        if key in observed_events:
            raise ValueError("causal shot phase events must be unique")
        observed_events.add(key)
        observed_sources.add(source)
        anchor = int(row.get("anchor_frame", -1))
        fps = float(row.get("source_fps", 0.0))
        frame_count = int(row.get("frame_count", 0))
        expected = [
            round(anchor + offset * fps)
            for offset in CAUSAL_PHASE_OFFSETS_SECONDS
        ]
        frames = row.get("frame_indexes")
        if (
            fps <= 0.0
            or frame_count <= 0
            or frames != expected
            or len(set(expected)) != len(expected)
            or expected[0] < 0
            or expected[-1] >= frame_count
            or expected[CAUSAL_PHASE_OFFSETS_SECONDS.index(0.0)] != anchor
        ):
            raise ValueError("causal shot phase frame window is invalid")
    if observed_sources != sources:
        raise ValueError("causal shot phase source coverage is invalid")


def _validate_review(
    artifact: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> None:
    if artifact.get("schema_version") != CAUSAL_PHASE_REVIEW_SCHEMA:
        raise ValueError("unsupported causal shot phase review schema")
    if (
        artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("labels_hidden_from_reviewer") is not True
        or artifact.get("plan_sha256") != plan["artifact_sha256"]
        or not str(artifact.get("reviewer") or "").strip()
    ):
        raise ValueError("causal shot phase review provenance is invalid")
    plan_by_id = {
        str(row["phase_review_id"]): row for row in plan["examples"]
    }
    reviews = artifact.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("causal shot phase review requires reviews")
    review_ids = [str(row.get("phase_review_id") or "") for row in reviews]
    if (
        len(review_ids) != len(set(review_ids))
        or set(review_ids) != set(plan_by_id)
    ):
        raise ValueError("causal shot phase reviews must exactly cover the plan")
    for row in reviews:
        planned = plan_by_id[str(row["phase_review_id"])]
        if (
            row.get("formation_state") not in FORMATION_STATES
            or row.get("broadcast_context") not in BROADCAST_CONTEXTS
            or row.get("shot_sequence") not in SHOT_SEQUENCES
            or row.get("outcome") not in OUTCOMES
            or row.get("confidence") not in CONFIDENCE_VALUES
            or not isinstance(row.get("notes"), str)
            or row.get("source_video_sha256")
            != planned["source_video_sha256"]
            or row.get("candidate_bundle_sha256")
            != planned["candidate_bundle_sha256"]
            or row.get("event_id") != planned["event_id"]
        ):
            raise ValueError("invalid causal shot phase review decision")
        release = _optional_position(
            row.get("release_position"),
            field="release",
        )
        rim = _optional_position(
            row.get("rim_arrival_position"),
            field="rim arrival",
        )
        if release is not None and rim is not None and release >= rim:
            raise ValueError("shot release must precede rim arrival")
        if row.get("release_frame") != _position_frame(
            release,
            planned["frame_indexes"],
        ) or row.get("rim_arrival_frame") != _position_frame(
            rim,
            planned["frame_indexes"],
        ):
            raise ValueError("causal shot phase frame binding is invalid")
        sequence = row["shot_sequence"]
        outcome = row["outcome"]
        if sequence == "complete":
            if release is None or rim is None or outcome == "not_applicable":
                raise ValueError("complete shot requires release and rim evidence")
        elif sequence == "partial":
            if (
                release is None and rim is None
            ) or outcome == "not_applicable":
                raise ValueError("partial shot requires visible causal evidence")
        elif sequence == "not_a_shot":
            if (
                release is not None
                or rim is not None
                or outcome != "not_applicable"
            ):
                raise ValueError("not-a-shot decision cannot contain shot phases")
        elif (
            release is not None
            or rim is not None
            or outcome != "unknown"
        ):
            raise ValueError("uncertain shot cannot contain asserted phases")


def _optional_position(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} position must be an integer or null")
    if value < 0 or value >= len(CAUSAL_PHASE_OFFSETS_SECONDS):
        raise ValueError(f"{field} position is outside the review window")
    return value


def _position_frame(
    position: Any,
    frame_indexes: Sequence[int],
) -> int | None:
    normalized = _optional_position(position, field="phase")
    return None if normalized is None else int(frame_indexes[normalized])


def _require_sha256(value: Any, *, field: str) -> str:
    text = str(value or "")
    if not _SHA256.fullmatch(text):
        raise ValueError(f"{field} SHA-256 is invalid")
    return text


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
