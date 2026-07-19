from __future__ import annotations

from app.analysis.box_score import EventLedger
from app.analysis.game_state import (
    ControlSpan,
    DefensiveContact,
    EventGraphBuilder,
    PassObservation,
    PossessionObservation,
    PossessionStateMachine,
    assess_shot_trajectory,
    classify_shot_value,
)
from app.analysis.game_state.candidates import _nearby_player_candidates
from app.analysis.perception import BallTracker
from app.analysis.schemas import (
    BallTrackPointResponse,
    BoundingBoxResponse,
    CourtCalibrationResponse,
    PerceptionDetectionResponse,
    Point2DResponse,
)


def _ball_detection(frame: int, x: float, y: float) -> PerceptionDetectionResponse:
    return PerceptionDetectionResponse(
        detection_id=f"d{frame}",
        frame=frame,
        object_type="basketball",
        bbox=BoundingBoxResponse(x1=x - 2, y1=y - 2, x2=x + 2, y2=y + 2),
        confidence=0.9,
        backend="test",
    )


def _player_detection(frame: int, player_id: str) -> PerceptionDetectionResponse:
    return PerceptionDetectionResponse(
        detection_id=f"{player_id}-{frame}",
        frame=frame,
        object_type="player",
        bbox=BoundingBoxResponse(x1=0, y1=0, x2=20, y2=40),
        confidence=0.9,
        backend="test",
        player_id=player_id,
        team_id="raw-dark",
    )


def test_actor_candidates_keep_temporal_observations_for_vlm_alias_overlay() -> None:
    balls = [_ball_detection(frame, 10, 10) for frame in (10, 11, 12)]
    players = [_player_detection(frame, "p1") for frame in (10, 11, 12)]
    detections = [*balls, *players]

    player_ids, _, observations = _nearby_player_candidates(
        detections,
        ball_ids=[ball.detection_id for ball in balls],
        detections_by_id={item.detection_id: item for item in detections},
        start_frame=10,
        end_frame=12,
        maximum_frame_gap=1,
    )

    assert player_ids == ["p1"]
    assert [item["frame"] for item in observations] == [10, 11, 12]


def test_actor_candidates_suppress_persistent_duplicate_boxes() -> None:
    balls = [_ball_detection(frame, 10, 10) for frame in (10, 11, 12)]
    first = [_player_detection(frame, "backend-a") for frame in (10, 11, 12)]
    duplicate = [
        _player_detection(frame, "backend-b").model_copy(
            update={
                "bbox": BoundingBoxResponse(x1=1, y1=0, x2=21, y2=40),
            }
        )
        for frame in (10, 11, 12)
    ]
    distinct = [
        _player_detection(frame, "distinct").model_copy(
            update={
                "bbox": BoundingBoxResponse(x1=60, y1=0, x2=80, y2=40),
            }
        )
        for frame in (10, 11, 12)
    ]
    detections = [*balls, *first, *duplicate, *distinct]

    player_ids, _, _ = _nearby_player_candidates(
        detections,
        ball_ids=[ball.detection_id for ball in balls],
        detections_by_id={item.detection_id: item for item in detections},
        start_frame=10,
        end_frame=12,
        maximum=2,
        maximum_frame_gap=1,
    )

    assert len({"backend-a", "backend-b"}.intersection(player_ids)) == 1
    assert "distinct" in player_ids


def test_ball_tracker_links_motion_and_interpolates_short_occlusion() -> None:
    tracker = BallTracker(max_distance_px=20, max_gap_frames=2)
    tracks = tracker.track([_ball_detection(1, 10, 10), _ball_detection(2, 15, 9), _ball_detection(4, 25, 7)])

    assert len(tracks) == 1
    interpolated = tracker.interpolate(tracks[0])
    assert [point.frame for point in interpolated.points] == [1, 2, 3, 4]
    assert interpolated.points[2].predicted is True


def test_possession_machine_distinguishes_control_release_shot_and_miss() -> None:
    machine = PossessionStateMachine(control_radius_px=20, shot_upward_velocity_px=5)
    hand = Point2DResponse(x=100, y=200)
    rim = Point2DResponse(x=110, y=80)

    controlled = machine.update(
        PossessionObservation(
            frame=1,
            ball=Point2DResponse(x=102, y=198),
            player_hands={"p1": (hand,)},
            player_teams={"p1": "dark"},
            rim_center=rim,
        )
    )
    released = machine.update(
        PossessionObservation(
            frame=2,
            ball=Point2DResponse(x=104, y=185),
            player_hands={},
            player_teams={"p1": "dark"},
            rim_center=rim,
        )
    )
    missed = machine.update(
        PossessionObservation(frame=3, ball=Point2DResponse(x=110, y=90), missed=True, rim_center=rim)
    )

    assert controlled is not None and controlled.to_state == "CONTROLLED"
    assert released is not None and released.to_state == "SHOT_FLIGHT"
    assert released.player_id == "p1"
    assert missed is not None and missed.to_state == "LOOSE_BALL"


def test_shot_trajectory_requires_downward_rim_crossing_for_make() -> None:
    rim = BoundingBoxResponse(x1=90, y1=95, x2=110, y2=105)
    made = assess_shot_trajectory(
        [
            BallTrackPointResponse(frame=1, center=Point2DResponse(x=100, y=80), confidence=0.95),
            BallTrackPointResponse(frame=2, center=Point2DResponse(x=101, y=101), confidence=0.94),
            BallTrackPointResponse(frame=3, center=Point2DResponse(x=101, y=120), confidence=0.93),
        ],
        rim,
    )
    missed = assess_shot_trajectory(
        [
            BallTrackPointResponse(frame=1, center=Point2DResponse(x=82, y=88), confidence=0.9),
            BallTrackPointResponse(frame=2, center=Point2DResponse(x=82, y=98), confidence=0.9),
            BallTrackPointResponse(frame=3, center=Point2DResponse(x=130, y=120), confidence=0.9),
        ],
        rim,
    )

    assert made.outcome == "made"
    assert missed.outcome == "missed"


def test_shot_value_uses_valid_homography_and_defers_line_calls() -> None:
    calibration = CourtCalibrationResponse(
        calibration_id="c1",
        source_video_id="video_001",
        start_frame=0,
        end_frame=100,
        image_points=[Point2DResponse(x=0, y=0)] * 4,
        court_points=[Point2DResponse(x=0, y=0)] * 4,
        homography=[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]],
        reprojection_error_px=2,
        status="valid",
        method="manual_opencv",
    )

    assert classify_shot_value(Point2DResponse(x=30, y=0), calibration)[0] == 2
    assert classify_shot_value(Point2DResponse(x=80, y=0), calibration)[0] == 3
    assert classify_shot_value(Point2DResponse(x=67.5, y=0), calibration)[0] is None


def test_event_graph_builds_rebound_turnover_steal_block_and_assist_relations() -> None:
    builder = EventGraphBuilder(source_video_id="video_001")
    missed = _official_event("miss", outcome="missed", confidence=0.96)
    rebound = builder.rebound_from_miss(
        event_id="rebound",
        missed_shot=missed,
        first_control=ControlSpan("light-1", "light", 140, 150, 0.97),
    )
    block = builder.block_from_contact(
        event_id="block",
        shot=missed,
        contact=DefensiveContact("light-2", "light", 120, 0.96, True),
    )
    turnover, steal = builder.turnover_and_optional_steal(
        turnover_event_id="turnover",
        steal_event_id="steal",
        prior_control=ControlSpan("dark-1", "dark", 200, 210, 0.96),
        next_control=ControlSpan("light-3", "light", 212, 220, 0.95),
        defender_caused_loss=True,
    )
    made = _official_event(
        "made",
        outcome="made",
        release_frame=310,
        start_frame=300,
        end_frame=330,
        primary_player_id="dark-2",
        confidence=0.97,
    )
    assist = builder.assist_from_pass(
        event_id="assist",
        made_shot=made,
        last_pass=PassObservation("dark-4", "dark-2", "dark", 290, 300, 0.96),
        shooter_control_changes=0,
        max_receive_to_release_frames=30,
    )

    assert rebound.rebound_type == "defensive" and rebound.related_event_ids == ["miss"]
    assert block.related_event_ids == ["miss"]
    assert steal is not None and steal.related_event_ids == ["turnover"]
    assert turnover.outcome_frame == steal.outcome_frame == 212
    assert assist is not None and assist.related_event_ids == ["made"]
    result = EventLedger([missed, rebound, block, turnover, steal, made, assist]).aggregate()
    assert result.reconciliation.valid is True


def test_event_graph_does_not_award_steal_or_assist_without_causal_evidence() -> None:
    builder = EventGraphBuilder(source_video_id="video_001")
    turnover, steal = builder.turnover_and_optional_steal(
        turnover_event_id="turnover",
        prior_control=ControlSpan("dark-1", "dark", 200, 210, 0.95),
        next_control=ControlSpan("light-1", "light", 212, 220, 0.95),
        defender_caused_loss=False,
    )
    made = _official_event(
        "made",
        outcome="made",
        release_frame=400,
        primary_player_id="dark-2",
        confidence=0.96,
    )
    assist = builder.assist_from_pass(
        event_id="assist",
        made_shot=made,
        last_pass=PassObservation("dark-4", "dark-2", "dark", 300, 310, 0.96),
        shooter_control_changes=1,
        max_receive_to_release_frames=30,
    )

    assert turnover.event_type == "turnover"
    assert steal is None
    assert assist is None


def _official_event(event_id: str, **updates: object):
    payload: dict[str, object] = {
        "event_id": event_id,
        "revision": 1,
        "event_type": "field_goal_attempt",
        "source_video_id": "video_001",
        "start_frame": 100,
        "end_frame": 130,
        "team_id": "dark",
        "primary_player_id": "dark-1",
        "shot_value": 2,
        "status": "vision_confirmed",
    }
    payload.update(updates)
    from app.analysis.schemas import GameEventResponse

    return GameEventResponse.model_validate(payload)
