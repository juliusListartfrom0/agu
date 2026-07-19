from __future__ import annotations

import numpy as np

from app.analysis.schemas import EventEvidenceResponse, GameEventResponse
from scripts.run_official_autonomous_inference import (
    _evenly_select_values,
    _overlay_candidate_identities,
    _review_frame_bounds,
    _select_identity_overlays,
)


def _event() -> GameEventResponse:
    return GameEventResponse(
        event_id="shot-1",
        revision=1,
        event_type="field_goal_attempt",
        source_video_id="video-1",
        start_frame=100,
        end_frame=130,
        evidence=[
            EventEvidenceResponse(
                evidence_id="e1",
                kind="traditional_cv",
                source_video_id="video-1",
                start_frame=100,
                end_frame=130,
                details={
                    "candidate_player_observations": [
                        {
                            "player_id": "track-7",
                            "team_id": "raw-dark",
                            "frame": 110,
                            "bbox": {"x1": 2, "y1": 2, "x2": 20, "y2": 28},
                        }
                    ]
                },
            )
        ],
    )


def test_identity_overlay_only_draws_near_observation_frame() -> None:
    frame = np.zeros((40, 40, 3), dtype=np.uint8)

    distant = _overlay_candidate_identities(frame, _event(), frame_number=100, maximum_frame_gap=3)
    nearby = _overlay_candidate_identities(frame, _event(), frame_number=108, maximum_frame_gap=3)

    assert distant is frame
    assert nearby is not frame
    assert np.any(nearby != frame)


def test_identity_overlay_limits_dense_frame_to_two_ball_nearest_players() -> None:
    event = _event()
    event.evidence[0].details["candidate_player_ids"] = ["track-7", "track-8", "track-9"]
    event.evidence[0].details["candidate_player_observations"] = [
        {
            "player_id": player_id,
            "team_id": "raw-dark",
            "frame": 110,
            "bbox": {"x1": x1, "y1": 2, "x2": x1 + 8, "y2": 28},
            "ball_player_distance": distance,
        }
        for player_id, x1, distance in (
            ("track-7", 2, 0.1),
            ("track-8", 14, 0.2),
            ("track-9", 28, 0.3),
        )
    ]
    selected = _select_identity_overlays(event.evidence[0].details["candidate_player_observations"], maximum=2)

    assert [item["player_id"] for item in selected] == ["track-7", "track-8"]


def test_identity_overlay_deduplicates_multiple_observations_of_same_player() -> None:
    observations = [
        {"player_id": "same", "ball_player_distance": 0.1, "frame": 10},
        {"player_id": "same", "ball_player_distance": 0.2, "frame": 12},
        {"player_id": "other", "ball_player_distance": 0.3, "frame": 10},
    ]

    selected = _select_identity_overlays(observations, maximum=2)

    assert [item["player_id"] for item in selected] == ["same", "other"]


def test_rebound_overlay_prioritizes_stable_control_before_transient_nearest_box() -> None:
    observations = [
        {
            "player_id": "transient-nearest",
            "ball_player_distance": 0.05,
            "stable_control_score": None,
        },
        {
            "player_id": "stable-control",
            "ball_player_distance": 0.25,
            "stable_control_score": 0.15,
        },
    ]

    selected = _select_identity_overlays(
        observations,
        maximum=1,
        prefer_stable_control=True,
    )

    assert selected[0]["player_id"] == "stable-control"


def test_identity_overlay_prefers_pose_wrist_contact_over_box_center_distance() -> None:
    observations = [
        {
            "player_id": "box-nearest",
            "ball_player_distance": 0.10,
            "wrist_ball_distance": 0.40,
        },
        {
            "player_id": "wrist-nearest",
            "ball_player_distance": 0.30,
            "wrist_ball_distance": 0.05,
        },
    ]

    selected = _select_identity_overlays(observations, maximum=1)

    assert selected[0]["player_id"] == "wrist-nearest"


def test_identity_overlay_prioritizes_traditional_action_owner_score() -> None:
    observations = [
        {
            "player_id": "ball-nearest",
            "ball_player_distance": 0.1,
            "action_owner_probability": 0.2,
        },
        {
            "player_id": "model-top1",
            "ball_player_distance": 0.4,
            "action_owner_probability": 0.9,
        },
    ]

    selected = _select_identity_overlays(observations, maximum=1)

    assert selected[0]["player_id"] == "model-top1"


def test_identity_overlay_keeps_ball_nearest_alongside_model_top1() -> None:
    observations = [
        {
            "player_id": "ball-nearest",
            "ball_player_distance": 0.1,
            "action_owner_probability": 0.2,
        },
        {
            "player_id": "model-top1",
            "ball_player_distance": 0.4,
            "action_owner_probability": 0.9,
        },
        {
            "player_id": "neither",
            "ball_player_distance": 0.3,
            "action_owner_probability": 0.5,
        },
    ]

    selected = _select_identity_overlays(observations, maximum=2)

    assert [item["player_id"] for item in selected] == ["model-top1", "ball-nearest"]


def test_review_bounds_use_traditional_outcome_anchor_instead_of_wide_candidate() -> None:
    event = _event().model_copy(update={"start_frame": 100, "end_frame": 1000, "outcome_frame": 700})

    assert _review_frame_bounds(event, source_fps=30.0, pre_seconds=3.0, post_seconds=2.0) == (610, 760)


def test_review_bounds_fall_back_to_evidence_anchor() -> None:
    event = _event()
    event.evidence[0].details["candidate_event_frame"] = 110

    assert _review_frame_bounds(event, source_fps=30.0, pre_seconds=1.0, post_seconds=1.0) == (80, 140)


def test_evenly_selected_frame_numbers_match_displayed_images() -> None:
    sampled = list(range(468, 649, 8))

    assert _evenly_select_values(sampled, 12) == [
        468,
        484,
        500,
        516,
        532,
        548,
        564,
        580,
        596,
        612,
        628,
        644,
    ]


def test_rebound_review_bounds_center_after_related_shot_release() -> None:
    event = GameEventResponse(
        event_id="rebound-1",
        revision=1,
        event_type="rebound",
        source_video_id="video-1",
        start_frame=150,
        end_frame=220,
        evidence=[
            EventEvidenceResponse(
                evidence_id="e1",
                kind="traditional_cv",
                source_video_id="video-1",
                start_frame=150,
                end_frame=220,
                details={"related_shot_release_frame": 120},
            )
        ],
    )

    assert _review_frame_bounds(event, source_fps=30.0, pre_seconds=1.0, post_seconds=1.0) == (120, 180)
