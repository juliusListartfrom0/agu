import pytest

from app.analysis.official_count_audit import (
    build_blind_count_audit,
    count_only_upper_bound,
    diagnose_prediction_bottlenecks,
)
from app.analysis.schemas import GameEventResponse


def test_count_only_upper_bound_reports_best_possible_f1_without_alignment() -> None:
    result = count_only_upper_bound(
        truth_counts={"field_goal_attempt": 2, "rebound": 1},
        prediction_counts={"field_goal_attempt": 3, "rebound": 2},
    )

    assert result["metrics"]["true_positive_upper_bound"] == 3
    assert result["metrics"]["false_positive_lower_bound"] == 2
    assert result["metrics"]["false_negative_lower_bound"] == 0
    assert result["metrics"]["precision_upper_bound"] == pytest.approx(0.6)
    assert result["metrics"]["recall_upper_bound"] == 1.0
    assert result["metrics"]["f1_upper_bound"] == pytest.approx(0.75)
    assert result["by_event_type"]["field_goal_attempt"]["f1_upper_bound"] == 0.8
    assert result["by_event_type"]["rebound"]["f1_upper_bound"] == 2 / 3


def test_blind_count_audit_separates_candidates_from_automatic_confirmations() -> None:
    official_payload = {
        "source_url": "https://www.nba.com/game/example/play-by-play",
        "playByPlay": {
            "gameId": "game-1",
            "actions": [
                {"actionType": "Made Shot"},
                {"actionType": "Missed Shot"},
                {"actionType": "Rebound"},
                {"actionType": "Free Throw"},
                {"actionType": "Turnover"},
                {"actionType": "Foul"},
            ],
        },
        "game": {
            "homeTeam": {
                "statistics": {"assists": 2, "steals": 1, "blocks": 1},
            },
            "awayTeam": {
                "statistics": {"assists": 3, "steals": 2, "blocks": 0},
            },
        },
    }
    events = [
        GameEventResponse(
            event_id="shot-confirmed",
            revision=1,
            event_type="field_goal_attempt",
            source_video_id="video_001",
            start_frame=10,
            end_frame=20,
            status="edge_vlm_confirmed",
        ),
        GameEventResponse(
            event_id="shot-review",
            revision=1,
            event_type="field_goal_attempt",
            source_video_id="video_001",
            start_frame=30,
            end_frame=40,
            status="needs_review",
        ),
        GameEventResponse(
            event_id="rebound-review",
            revision=1,
            event_type="rebound",
            source_video_id="video_001",
            start_frame=50,
            end_frame=60,
            status="needs_review",
        ),
    ]

    report = build_blind_count_audit(
        prediction_bundle_sha256="prediction-sha",
        events=events,
        official_payload=official_payload,
        target_f1=0.85,
    )

    assert report["official_event_counts"] == {
        "field_goal_attempt": 2,
        "rebound": 1,
        "free_throw_attempt": 1,
        "turnover": 1,
        "foul": 1,
        "assist": 5,
        "steal": 3,
        "block": 1,
    }
    assert report["candidate_geometry"]["prediction_counts"] == {
        "field_goal_attempt": 2,
        "rebound": 1,
    }
    assert report["automatic_confirmed"]["prediction_counts"] == {
        "field_goal_attempt": 1,
    }
    assert report["complete_scope"]["candidate_geometry"]["metrics"] == {
        "true_positive_upper_bound": 3,
        "false_positive_lower_bound": 0,
        "false_negative_lower_bound": 12,
        "precision_upper_bound": 1.0,
        "recall_upper_bound": 0.2,
        "f1_upper_bound": pytest.approx(1 / 3),
    }
    assert report["complete_scope"]["candidate_geometry"]["target_status"] == "failed"
    assert report["complete_scope"]["automatic_confirmed"]["metrics"]["f1_upper_bound"] == pytest.approx(
        0.125
    )
    assert report["scope_coverage"]["covered_event_types"] == [
        "field_goal_attempt",
        "rebound",
    ]
    assert report["scope_coverage"]["missing_event_types"] == [
        "assist",
        "block",
        "foul",
        "free_throw_attempt",
        "steal",
        "turnover",
    ]
    assert report["candidate_geometry"]["target_status"] == "passed"
    assert report["automatic_confirmed"]["target_status"] == "failed"
    assert report["diagnostic_only_not_acceptance_metric"] is True


def test_prediction_bottleneck_diagnosis_quantifies_causal_and_identity_gaps() -> None:
    events = [
        GameEventResponse(
            event_id="shot-confirmed",
            revision=1,
            event_type="field_goal_attempt",
            source_video_id="video_001",
            start_frame=10,
            end_frame=20,
            status="edge_vlm_confirmed",
            outcome="made",
            team_id="raw-dark",
            primary_player_id="player-1",
        ),
        GameEventResponse(
            event_id="shot-review",
            revision=1,
            event_type="field_goal_attempt",
            source_video_id="video_001",
            start_frame=30,
            end_frame=40,
            status="needs_review",
            outcome="unknown",
        ),
        GameEventResponse(
            event_id="rebound-review",
            revision=1,
            event_type="rebound",
            source_video_id="video_001",
            start_frame=50,
            end_frame=60,
            status="needs_review",
            reason="AGU causal gate: linked shot outcome is not an automatic confirmed miss",
        ),
    ]

    diagnosis = diagnose_prediction_bottlenecks(
        events=events,
        official_counts={"field_goal_attempt": 1, "rebound": 1},
    )

    assert diagnosis["candidate_overgeneration"]["false_positive_lower_bound"] == 1
    assert diagnosis["candidate_overgeneration"]["by_event_type"] == {
        "field_goal_attempt": 1,
        "rebound": 0,
    }
    assert diagnosis["confirmation_collapse"]["status_counts"] == {
        "edge_vlm_confirmed": 1,
        "needs_review": 2,
    }
    assert diagnosis["confirmation_collapse"]["automatic_confirmed_count"] == 1
    assert diagnosis["shot_outcome_gap"] == {
        "field_goal_attempt_count": 2,
        "unknown_or_missing_count": 1,
        "known_count": 1,
        "known_rate": 0.5,
    }
    assert diagnosis["causal_gate_blocks"] == {
        "linked_shot_not_confirmed_miss": 1,
        "rebound_after_confirmed_make": 0,
    }
    assert diagnosis["identity_gap"] == {
        "event_count": 3,
        "missing_team_id_count": 2,
        "missing_primary_player_id_count": 2,
        "team_id_coverage": pytest.approx(1 / 3),
        "primary_player_id_coverage": pytest.approx(1 / 3),
    }
