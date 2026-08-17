from __future__ import annotations

import pytest

from app.analysis.official_evaluation import seal_raw_only_predictions
from app.analysis.schemas import (
    BoundingBoxResponse,
    EventEvidenceResponse,
    GameEventResponse,
    PerceptionDetectionResponse,
)
from app.analysis.scoreboard_causal import (
    ScoreboardCausalConfig,
    attach_scoreboard_causal_evidence,
)
from scripts.enrich_scoreboard_causal_evidence import (
    _verify_equivalent_event_geometry,
)


def _detection(
    detection_id: str,
    frame: int,
    object_type: str,
    x: float,
    y: float,
) -> PerceptionDetectionResponse:
    return PerceptionDetectionResponse(
        detection_id=detection_id,
        frame=frame,
        object_type=object_type,
        bbox=BoundingBoxResponse(x1=x, y1=y, x2=x + 8, y2=y + 8),
        confidence=0.9,
        backend="fixture",
    )


def _score_event() -> GameEventResponse:
    return GameEventResponse(
        event_id="score-1",
        revision=1,
        event_type="field_goal_attempt",
        source_video_id="video_001",
        start_frame=10,
        end_frame=30,
        outcome="made",
        status="needs_review",
        confidence=0.9,
        evidence=[
            EventEvidenceResponse(
                evidence_id="score-1:delta",
                kind="scoreboard_delta",
                source_video_id="video_001",
                start_frame=0,
                end_frame=30,
                confidence=0.9,
                details={"points": 2},
            )
        ],
    )


def test_attach_scoreboard_causal_evidence_preserves_score_and_adds_physics() -> None:
    detections = [
        _detection("ball-1", 16, "basketball", 30, 50),
        _detection("ball-2", 18, "basketball", 35, 38),
        _detection("ball-3", 20, "basketball", 40, 24),
        _detection("rim", 20, "rim", 42, 22),
    ]

    result, stats = attach_scoreboard_causal_evidence(
        [_score_event()],
        detections,
        config=ScoreboardCausalConfig(
            max_rim_frame_gap=3,
            cluster_gap_frames=5,
            maximum_cluster_span_frames=10,
            minimum_approach_points=3,
            minimum_approach_rise_px=10,
        ),
    )

    assert stats == {
        "field_goal_count": 1,
        "matched_field_goal_count": 1,
        "physical_proposal_count": 1,
    }
    assert result[0].outcome == "made"
    assert result[0].start_frame == 10
    assert [item.kind for item in result[0].evidence] == [
        "scoreboard_delta",
        "ball_rim_proximity_cluster",
    ]
    assert (
        result[0].evidence[1].details["maximum_approach_point_count"] == 3
    )
    assert (
        result[0].evidence[1].details["causal_evidence_version"]
        == "agu_window_ball_rim_pose_v1"
    )


def test_attach_scoreboard_causal_evidence_abstains_without_trajectory() -> None:
    result, stats = attach_scoreboard_causal_evidence(
        [_score_event()],
        [_detection("ball", 20, "basketball", 40, 24)],
    )

    assert stats["matched_field_goal_count"] == 0
    assert [item.kind for item in result[0].evidence] == ["scoreboard_delta"]


def test_pose_source_equivalence_requires_identical_event_geometry(tmp_path) -> None:
    video = tmp_path / "game.mp4"
    video.write_bytes(b"raw")
    source = seal_raw_only_predictions(
        game_id="game",
        raw_video_paths=[video],
        events=[_score_event()],
        config={"version": 1},
    )
    equivalent = seal_raw_only_predictions(
        game_id="game",
        raw_video_paths=[video],
        events=[_score_event()],
        config={"version": 2},
    )

    _verify_equivalent_event_geometry(source, equivalent)

    changed = seal_raw_only_predictions(
        game_id="game",
        raw_video_paths=[video],
        events=[_score_event().model_copy(update={"start_frame": 11})],
        config={"version": 3},
    )
    with pytest.raises(ValueError, match="event geometry"):
        _verify_equivalent_event_geometry(source, changed)
