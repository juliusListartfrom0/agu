"""Quantify what OCR-clock/PBP alignment can and cannot supervise."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

PBP_CAUSAL_COVERAGE_SCHEMA = "agu.pbp-causal-coverage-audit.v1"


def build_pbp_causal_coverage_audit(
    alignments: Sequence[Mapping[str, Any]],
    *,
    generated_on: str,
) -> dict[str, Any]:
    """Summarize frame-anchor coverage without claiming causal truth.

    The input is an OCR-clock alignment artifact.  A matched frame is only a
    temporal anchor for the PBP row; it is deliberately not a ball/hand/rim
    label.  The summary exposes make/miss mapping asymmetry so later training
    cannot silently treat the mapped subset as representative.
    """

    if not alignments:
        raise ValueError("at least one PBP alignment is required")
    games: list[dict[str, Any]] = []
    total = Counter()
    action_totals: dict[str, Counter[str]] = defaultdict(Counter)
    for alignment in alignments:
        events = alignment.get("events")
        if not isinstance(events, list) or not events:
            raise ValueError("PBP alignment events must be a non-empty list")
        game_id = str(alignment.get("game_id") or "").strip()
        artifact_sha = str(alignment.get("artifact_sha256") or "").strip()
        if not game_id or not artifact_sha:
            raise ValueError("PBP alignment requires game_id and artifact_sha256")
        counts = Counter()
        action_counts: dict[str, Counter[str]] = defaultdict(Counter)
        deltas = Counter()
        for event in events:
            if not isinstance(event, Mapping):
                raise ValueError("PBP alignment events must be objects")
            mapped = event.get("video_frame") is not None
            action = str(event.get("action_type") or "unknown")
            counts["event_count"] += 1
            counts["mapped_event_count"] += int(mapped)
            action_counts[action]["event_count"] += 1
            action_counts[action]["mapped_event_count"] += int(mapped)
            if mapped and event.get("clock_delta_seconds") is not None:
                deltas[str(event["clock_delta_seconds"])] += 1
            if bool(event.get("is_field_goal")):
                counts["field_goal_attempt_count"] += 1
                counts["mapped_field_goal_attempt_count"] += int(mapped)
            if action == "Made Shot":
                counts["made_shot_count"] += 1
                counts["mapped_made_shot_count"] += int(mapped)
            elif action == "Missed Shot":
                counts["missed_shot_count"] += 1
                counts["mapped_missed_shot_count"] += int(mapped)

        game = _finalize_game(
            game_id=game_id,
            alignment_artifact_sha256=artifact_sha,
            counts=counts,
            action_counts=action_counts,
            delta_counts=deltas,
        )
        games.append(game)
        total.update(counts)
        for action, values in action_counts.items():
            action_totals[action].update(values)

    overall = _finalize_game(
        game_id="__pooled__",
        alignment_artifact_sha256="pooled",
        counts=total,
        action_counts=action_totals,
        delta_counts=Counter(),
    )
    audit: dict[str, Any] = {
        "schema_version": PBP_CAUSAL_COVERAGE_SCHEMA,
        "generated_on": str(generated_on),
        "purpose": "measure_ocr_clock_pbp_anchor_coverage_without_causal_visual_labels",
        "runtime_consumable": False,
        "training_media_eligible": False,
        "accepted": False,
        "causal_label_contract": {
            "clock_anchor_is_not_ball_hand_rim_outcome_truth": True,
            "subsecond_ball_hand_rim_outcome_labels_available": False,
            "exhaustive_non_shot_hard_negatives_available": False,
        },
        "games": games,
        "pooled": overall,
        "interpretation": (
            "PBP rows with a matched OCR clock can select a temporal neighborhood only; they cannot "
            "be promoted to frame-level causal supervision. Unequal mapped make/miss fractions are "
            "a selection-bias warning."
        ),
    }
    audit["audit_sha256"] = _canonical_sha256(audit)
    return audit


def verify_pbp_causal_coverage_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("audit_sha256", ""))
    if artifact.get("schema_version") != PBP_CAUSAL_COVERAGE_SCHEMA:
        raise ValueError("invalid PBP causal coverage schema")
    if artifact.get("runtime_consumable") is not False or artifact.get("accepted") is not False:
        raise ValueError("PBP causal coverage audit must remain offline and rejected")
    if not isinstance(artifact.get("games"), list) or not artifact["games"]:
        raise ValueError("PBP causal coverage audit requires games")
    if not claimed or claimed != _canonical_sha256(artifact):
        raise ValueError("PBP causal coverage audit hash mismatch")
    artifact["audit_sha256"] = claimed
    return artifact


def _finalize_game(
    *,
    game_id: str,
    alignment_artifact_sha256: str,
    counts: Counter[str],
    action_counts: Mapping[str, Counter[str]],
    delta_counts: Counter[str],
) -> dict[str, Any]:
    event_count = int(counts["event_count"])
    mapped_count = int(counts["mapped_event_count"])
    fg_count = int(counts["field_goal_attempt_count"])
    mapped_fg = int(counts["mapped_field_goal_attempt_count"])
    made_count = int(counts["made_shot_count"])
    mapped_made = int(counts["mapped_made_shot_count"])
    missed_count = int(counts["missed_shot_count"])
    mapped_missed = int(counts["mapped_missed_shot_count"])
    action_summary = {
        action: {
            "event_count": int(values["event_count"]),
            "mapped_event_count": int(values["mapped_event_count"]),
            "mapped_fraction": _fraction(values["mapped_event_count"], values["event_count"]),
        }
        for action, values in sorted(action_counts.items())
    }
    return {
        "game_id": game_id,
        "alignment_artifact_sha256": alignment_artifact_sha256,
        "event_count": event_count,
        "mapped_event_count": mapped_count,
        "mapped_event_fraction": _fraction(mapped_count, event_count),
        "field_goal_attempt_count": fg_count,
        "mapped_field_goal_attempt_count": mapped_fg,
        "mapped_field_goal_attempt_fraction": _fraction(mapped_fg, fg_count),
        "made_shot_count": made_count,
        "mapped_made_shot_count": mapped_made,
        "mapped_made_shot_fraction": _fraction(mapped_made, made_count),
        "missed_shot_count": missed_count,
        "mapped_missed_shot_count": mapped_missed,
        "mapped_missed_shot_fraction": _fraction(mapped_missed, missed_count),
        "mapped_make_minus_miss_fraction": round(
            _fraction(mapped_made, made_count) - _fraction(mapped_missed, missed_count), 6
        ),
        "action_summary": action_summary,
        "mapped_clock_delta_seconds": dict(sorted(delta_counts.items())),
    }


def _fraction(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
