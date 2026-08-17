from app.analysis.schemas import EventEvidenceResponse, GameEventResponse
from scripts.run_candidate_component_scoreboard import (
    candidate_anchor,
    candidate_sample_frames,
)


def test_candidate_anchor_prefers_highest_hit_count() -> None:
    event = GameEventResponse(
        event_id="shot",
        revision=1,
        event_type="field_goal_attempt",
        source_video_id="video_001",
        start_frame=90,
        end_frame=180,
        status="needs_review",
        evidence=[
            EventEvidenceResponse(
                evidence_id="weak",
                kind="ball_rim_proximity_cluster",
                source_video_id="video_001",
                start_frame=100,
                end_frame=110,
                details={"hit_count": 2, "review_anchor_frame": 105},
            ),
            EventEvidenceResponse(
                evidence_id="strong",
                kind="ball_rim_proximity_cluster",
                source_video_id="video_001",
                start_frame=130,
                end_frame=150,
                details={"hit_count": 5, "review_anchor_frame": 140},
            ),
        ],
    )

    assert candidate_anchor(event) == 140


def test_candidate_sample_frames_preserve_frozen_gap() -> None:
    before, after = candidate_sample_frames(
        anchor=900,
        fps=30,
        frame_count=2000,
        sample_interval_sec=2,
        before_start_sec=20,
        before_end_sec=4,
        after_end_sec=20,
    )

    assert before == list(range(300, 781, 60))
    assert after == list(range(900, 1501, 60))
