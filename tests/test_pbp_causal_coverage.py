from __future__ import annotations

import pytest

from app.analysis.pbp_causal_coverage import (
    PBP_CAUSAL_COVERAGE_SCHEMA,
    build_pbp_causal_coverage_audit,
    verify_pbp_causal_coverage_audit,
)


def _alignment() -> dict:
    return {
        "game_id": "game-a",
        "artifact_sha256": "a" * 64,
        "events": [
            {"action_type": "Made Shot", "is_field_goal": True, "video_frame": 10, "clock_delta_seconds": 0.0},
            {"action_type": "Missed Shot", "is_field_goal": True, "video_frame": None, "clock_delta_seconds": None},
            {"action_type": "Rebound", "is_field_goal": False, "video_frame": 20, "clock_delta_seconds": 1.0},
        ],
    }


def test_coverage_exposes_make_miss_mapping_bias_and_is_rejected() -> None:
    audit = build_pbp_causal_coverage_audit([_alignment()], generated_on="2026-08-08")

    assert audit["schema_version"] == PBP_CAUSAL_COVERAGE_SCHEMA
    assert audit["pooled"]["event_count"] == 3
    assert audit["pooled"]["mapped_event_count"] == 2
    assert audit["pooled"]["mapped_field_goal_attempt_fraction"] == 0.5
    assert audit["pooled"]["mapped_make_minus_miss_fraction"] == 1.0
    assert audit["causal_label_contract"]["subsecond_ball_hand_rim_outcome_labels_available"] is False
    assert verify_pbp_causal_coverage_audit(audit)["audit_sha256"] == audit["audit_sha256"]


def test_coverage_requires_alignment_events() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_pbp_causal_coverage_audit(
            [{"game_id": "game-a", "artifact_sha256": "a" * 64, "events": []}],
            generated_on="2026-08-08",
        )
