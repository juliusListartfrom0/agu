import hashlib
import json
from pathlib import Path

from app.analysis.game_state import (
    LegacyAnalysisCandidateConfig,
    ShotCandidateConfig,
    attach_traditional_identity_candidates,
    link_causal_candidate_relations,
    merge_temporal_review_candidates,
    propose_legacy_analysis_candidates,
    propose_shot_and_rebound_candidates,
)
from app.analysis.schemas import (
    BoundingBoxResponse,
    GameEventResponse,
    OfficialCanonicalIdentityResponse,
    OfficialIdentityGraphArtifactResponse,
    OfficialIdentityTrackletResponse,
    PerceptionDetectionResponse,
    Point2DResponse,
)
from scripts.build_official_event_candidates import (
    _face_gallery_anchored_player_ids,
    _load_detection,
    build_candidates,
)


def _detection(identifier: str, frame: int, kind: str, box: tuple[float, float, float, float]):
    return PerceptionDetectionResponse(
        detection_id=identifier,
        frame=frame,
        object_type=kind,
        bbox=BoundingBoxResponse(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
        confidence=0.9,
        backend="test",
    )


def test_ball_rim_hits_cluster_into_linked_review_candidates() -> None:
    detections = [
        _detection("rim-1", 100, "rim", (90, 90, 130, 105)),
        _detection("ball-1", 100, "basketball", (105, 60, 115, 70)),
        _detection("rim-2", 110, "rim", (90, 90, 130, 105)),
        _detection("ball-2", 110, "basketball", (108, 88, 118, 98)),
        _detection("rim-3", 300, "rim", (90, 90, 130, 105)),
        _detection("ball-3", 300, "basketball", (108, 88, 118, 98)),
    ]

    events = propose_shot_and_rebound_candidates(
        detections,
        source_video_id="video_001",
        config=ShotCandidateConfig(cluster_gap_frames=30, max_rim_frame_gap=1),
    )

    assert [event.event_type for event in events] == [
        "field_goal_attempt",
        "rebound",
        "field_goal_attempt",
        "rebound",
    ]
    assert events[1].related_event_ids == [events[0].event_id]
    assert events[0].status == "needs_review"
    assert events[0].evidence[0].details["candidate_event_frame"] == 100
    assert events[1].evidence[0].details["candidate_event_frame"] == 155


def test_ball_rim_cluster_span_prevents_chain_across_a_possession() -> None:
    detections = []
    for frame in (100, 130, 160, 190):
        detections.extend(
            [
                _detection(f"rim-{frame}", frame, "rim", (90, 90, 130, 105)),
                _detection(f"ball-{frame}", frame, "basketball", (108, 88, 118, 98)),
            ]
        )

    events = propose_shot_and_rebound_candidates(
        detections,
        source_video_id="video_001",
        config=ShotCandidateConfig(
            cluster_gap_frames=31,
            maximum_cluster_span_frames=60,
            max_rim_frame_gap=1,
        ),
    )

    shots = [event for event in events if event.event_type == "field_goal_attempt"]
    assert len(shots) == 2
    assert [event.evidence[0].details["frames"] for event in shots] == [
        [100, 130, 160],
        [190],
    ]


def test_ball_far_from_rim_does_not_create_candidate() -> None:
    detections = [
        _detection("rim", 100, "rim", (90, 90, 130, 105)),
        _detection("ball", 100, "basketball", (500, 500, 510, 510)),
    ]

    assert propose_shot_and_rebound_candidates(detections, source_video_id="video_001") == []


def test_ball_approach_gate_rejects_stationary_near_rim_false_positive() -> None:
    detections = [
        _detection("rim", 100, "rim", (90, 90, 130, 105)),
        _detection("ball-1", 90, "basketball", (108, 88, 118, 98)),
        _detection("ball-2", 95, "basketball", (108, 88, 118, 98)),
        _detection("ball-3", 100, "basketball", (108, 88, 118, 98)),
    ]

    assert (
        propose_shot_and_rebound_candidates(
            detections,
            source_video_id="video_001",
            config=ShotCandidateConfig(
                max_rim_frame_gap=6,
                minimum_approach_points=3,
                minimum_approach_rise_px=20,
            ),
        )
        == []
    )


def test_ball_approach_gate_keeps_continuous_rising_trajectory() -> None:
    detections = [
        _detection("rim", 100, "rim", (90, 90, 130, 105)),
        _detection("ball-1", 90, "basketball", (108, 150, 118, 160)),
        _detection("ball-2", 95, "basketball", (108, 115, 118, 125)),
        _detection("ball-3", 100, "basketball", (108, 88, 118, 98)),
    ]

    events = propose_shot_and_rebound_candidates(
        detections,
        source_video_id="video_001",
        config=ShotCandidateConfig(
            max_rim_frame_gap=6,
            minimum_approach_points=3,
            minimum_approach_rise_px=50,
        ),
    )

    assert [event.event_type for event in events] == ["field_goal_attempt", "rebound"]
    assert events[0].evidence[0].details["maximum_approach_point_count"] == 3
    assert events[0].evidence[0].details["maximum_approach_rise_px"] == 62.0


def test_additional_tracker_ids_are_namespaced_by_perception_artifact() -> None:
    item = _detection("player", 100, "player", (0, 0, 10, 20)).model_copy(
        update={"track_id": "track-7", "player_id": "raw-player-track-7"}
    )

    primary = _load_detection(item.model_dump(mode="json"), 0)
    additional = _load_detection(item.model_dump(mode="json"), 2)

    assert primary.player_id == "raw-player-track-7"
    assert additional.player_id == "perception-2:raw-player-track-7"
    assert additional.track_id == "perception-2:track-7"
    assert additional.detection_id == "perception-2:player"


def test_canonical_player_id_survives_additional_perception_namespacing() -> None:
    item = _detection("player", 100, "player", (0, 0, 10, 20)).model_copy(
        update={"track_id": "track-7", "player_id": "canonical-dark-player"}
    )

    canonical = _load_detection(
        item.model_dump(mode="json"),
        2,
        preserve_player_id=True,
    )

    assert canonical.player_id == "canonical-dark-player"
    assert canonical.track_id == "perception-2:track-7"


def test_traditional_tracking_ids_are_attached_as_bounded_vlm_candidates() -> None:
    player = _detection("player", 100, "player", (80, 40, 125, 180)).model_copy(
        update={"track_id": "track-7", "player_id": "raw-player-track-7", "team_id": "raw-dark"}
    )
    events = propose_shot_and_rebound_candidates(
        [
            player,
            _detection("ball", 100, "basketball", (105, 60, 115, 70)),
            _detection("rim", 100, "rim", (90, 90, 130, 105)),
        ],
        source_video_id="video_001",
        config=ShotCandidateConfig(cluster_gap_frames=10),
    )

    shot = events[0]
    assert shot.primary_player_id == "raw-player-track-7"
    assert shot.team_id == "raw-dark"
    assert shot.evidence[0].details["candidate_player_ids"] == ["raw-player-track-7"]


def test_shot_candidate_player_limit_is_explicitly_configurable() -> None:
    detections = [
        _detection("ball", 100, "basketball", (105, 60, 115, 70)),
        _detection("rim", 100, "rim", (90, 90, 130, 105)),
    ]
    for index, x1 in enumerate((90, 120, 150), start=1):
        detections.append(
            _detection(f"player-{index}", 100, "player", (x1, 40, x1 + 30, 180)).model_copy(
                update={"player_id": f"player-{index}"}
            )
        )

    events = propose_shot_and_rebound_candidates(
        detections,
        source_video_id="video_001",
        config=ShotCandidateConfig(
            cluster_gap_frames=10,
            maximum_player_candidates=2,
        ),
    )

    assert len(events[0].evidence[0].details["candidate_player_ids"]) == 2


def test_shooter_candidates_exclude_post_rim_rebounder() -> None:
    before = _detection("before", 90, "player", (80, 40, 125, 180)).model_copy(
        update={"player_id": "raw-dark-shooter", "team_id": "raw-dark"}
    )
    after = _detection("after", 110, "player", (80, 40, 125, 180)).model_copy(
        update={"player_id": "raw-light-rebounder", "team_id": "raw-light"}
    )
    events = propose_shot_and_rebound_candidates(
        [
            before,
            after,
            _detection("pre-ball", 90, "basketball", (105, 60, 115, 70)),
            _detection("rim-ball", 100, "basketball", (108, 88, 118, 98)),
            _detection("rim", 100, "rim", (90, 90, 130, 105)),
            _detection("post-ball", 110, "basketball", (105, 60, 115, 70)),
        ],
        source_video_id="video_001",
        config=ShotCandidateConfig(max_rim_frame_gap=1, shooter_lookback_frames=30),
    )

    shot = events[0]
    assert shot.evidence[0].details["candidate_player_ids"] == ["raw-dark-shooter"]
    assert shot.team_id == "raw-dark"


def test_shooter_lookback_can_extend_before_compact_review_preroll() -> None:
    shooter = _detection("shooter", 60, "player", (80, 40, 125, 180)).model_copy(
        update={"player_id": "raw-dark-shooter", "team_id": "raw-dark"}
    )
    events = propose_shot_and_rebound_candidates(
        [
            shooter,
            _detection("release-ball", 60, "basketball", (105, 60, 115, 70)),
            _detection("rim-ball", 100, "basketball", (108, 88, 118, 98)),
            _detection("rim", 100, "rim", (90, 90, 130, 105)),
        ],
        source_video_id="video_001",
        config=ShotCandidateConfig(
            max_rim_frame_gap=1,
            pre_roll_frames=10,
            shooter_lookback_frames=50,
        ),
    )

    shot = events[0]
    assert shot.start_frame == 90
    assert shot.evidence[0].details["candidate_player_ids"] == ["raw-dark-shooter"]


def test_rebound_candidates_prioritize_first_stable_control_over_one_frame_proximity() -> None:
    detections = [
        _detection("rim", 100, "rim", (90, 90, 130, 105)),
        _detection("rim-ball", 100, "basketball", (108, 88, 118, 98)),
    ]
    for frame in (110, 112, 114):
        detections.append(_detection(f"ball-{frame}", frame, "basketball", (108, 130, 118, 140)))
        detections.append(
            _detection(f"stable-{frame}", frame, "player", (100, 100, 145, 220)).model_copy(
                update={"player_id": "stable-control", "team_id": "raw-light"}
            )
        )
    detections.append(
        _detection("transient", 110, "player", (105, 105, 125, 220)).model_copy(
            update={"player_id": "transient-tip", "team_id": "raw-dark"}
        )
    )

    events = propose_shot_and_rebound_candidates(
        detections,
        source_video_id="video_001",
        config=ShotCandidateConfig(max_rim_frame_gap=2, rebound_window_frames=30),
    )

    rebound = next(event for event in events if event.event_type == "rebound")
    details = rebound.evidence[0].details
    assert details["candidate_player_ids"][0] == "stable-control"
    stable = next(item for item in details["candidate_player_observations"] if item["player_id"] == "stable-control")
    assert stable["stable_contact_count"] == 3
    assert stable["first_stable_control_frame"] == 110


def test_candidate_observation_exposes_pose_wrist_ball_distance() -> None:
    player = _detection("player", 100, "player", (80, 40, 125, 180)).model_copy(
        update={
            "player_id": "pose-player",
            "team_id": "raw-dark",
            "keypoints": {
                "left_wrist": Point2DResponse(x=108.0, y=65.0),
                "right_wrist": Point2DResponse(x=90.0, y=90.0),
            },
        }
    )
    events = propose_shot_and_rebound_candidates(
        [
            player,
            _detection("ball", 100, "basketball", (105, 60, 115, 70)),
            _detection("rim", 100, "rim", (90, 90, 130, 105)),
        ],
        source_video_id="video_001",
    )

    observation = events[0].evidence[0].details["candidate_player_observations"][0]
    assert observation["wrist_ball_distance"] < observation["ball_player_distance"]


def test_legacy_analysis_stays_needs_review_and_respects_bounds() -> None:
    analysis = {
        "records": [
            {"start_frame": 10, "end_frame": 20, "final": {"action": "shoot", "confidence": 0.8}},
            {"start_frame": 110, "end_frame": 120, "final": {"action": "shoot", "confidence": 0.9}},
        ],
        "long_video": {
            "event_candidates": [
                {
                    "event_type": "steal_candidate",
                    "start_frame": 25,
                    "end_frame": 35,
                    "confidence": 0.3,
                    "player_id": "p2",
                }
            ]
        },
    }
    events = propose_legacy_analysis_candidates(
        analysis,
        source_video_id="video_001",
        config=LegacyAnalysisCandidateConfig(min_frame=0, max_frame=100),
    )
    assert [event.event_type for event in events] == ["field_goal_attempt", "turnover", "steal"]
    assert all(event.status == "needs_review" for event in events)
    assert events[2].related_event_ids == [events[1].event_id]


def test_legacy_time_candidate_rebinds_to_current_canonical_observations() -> None:
    events = propose_legacy_analysis_candidates(
        {
            "records": [
                {
                    "start_frame": 100,
                    "end_frame": 120,
                    "local_player_id": "stale-segment-player",
                    "final": {"action": "shoot", "confidence": 0.8},
                }
            ]
        },
        source_video_id="video_001",
    )
    current = _detection("current", 110, "player", (20, 20, 50, 100)).model_copy(
        update={"player_id": "canonical-dark-1", "team_id": "raw-dark"}
    )

    grounded = attach_traditional_identity_candidates(events, [current], context_frames=30)

    assert grounded[0].primary_player_id == "canonical-dark-1"
    assert grounded[0].team_id == "raw-dark"
    assert grounded[0].evidence[0].details["candidate_player_ids"] == ["canonical-dark-1"]
    assert grounded[0].evidence[0].details["candidate_event_frame"] == 110


def test_candidate_merge_keeps_priority_timing_and_remaps_relations() -> None:
    primary = propose_shot_and_rebound_candidates(
        [
            _detection("ball", 100, "basketball", (105, 60, 115, 70)),
            _detection("rim", 100, "rim", (90, 90, 130, 105)),
        ],
        source_video_id="video_001",
        config=ShotCandidateConfig(cluster_gap_frames=10),
    )
    secondary = propose_shot_and_rebound_candidates(
        [
            _detection("ball-2", 105, "basketball", (105, 60, 115, 70)),
            _detection("rim-2", 105, "rim", (90, 90, 130, 105)),
        ],
        source_video_id="video_001",
        config=ShotCandidateConfig(cluster_gap_frames=10),
        event_id_prefix="secondary",
    )
    merged = merge_temporal_review_candidates([[primary[0]], secondary], max_center_gap_frames=20)
    shots = [event for event in merged if event.event_type == "field_goal_attempt"]
    rebound = next(event for event in merged if event.event_type == "rebound")
    assert len(shots) == 1
    assert shots[0].event_id == primary[0].event_id
    assert len(shots[0].evidence) == 2
    assert rebound.related_event_ids == [shots[0].event_id]


def test_possession_change_merge_extends_to_latest_control_frame() -> None:
    analysis = {
        "long_video": {
            "event_candidates": [
                {"event_type": "steal_candidate", "start_frame": 100, "end_frame": 120, "confidence": 0.3},
                {"event_type": "steal_candidate", "start_frame": 115, "end_frame": 140, "confidence": 0.3},
            ]
        }
    }
    events = propose_legacy_analysis_candidates(analysis, source_video_id="video_001")
    merged = merge_temporal_review_candidates([events], max_center_gap_frames=30)
    steal = next(event for event in merged if event.event_type == "steal")
    turnover = next(event for event in merged if event.event_type == "turnover")

    assert (steal.start_frame, steal.end_frame, steal.outcome_frame) == (100, 140, 140)
    assert (turnover.start_frame, turnover.end_frame, turnover.outcome_frame) == (100, 140, 140)
    assert steal.related_event_ids == [turnover.event_id]


def test_dense_candidate_relations_link_only_compatible_nearby_parents() -> None:
    def event(event_id: str, event_type: str, start: int) -> GameEventResponse:
        return GameEventResponse(
            event_id=event_id,
            revision=1,
            event_type=event_type,
            source_video_id="video_001",
            start_frame=start,
            end_frame=start + 10,
            status="needs_review",
        )

    shot = event("shot", "field_goal_attempt", 100)
    turnover = event("turnover", "turnover", 200)
    assist = event("assist", "assist", 90)
    block = event("block", "block", 105)
    rebound = event("rebound", "rebound", 120)
    steal = event("steal", "steal", 205)
    far_rebound = event("far-rebound", "rebound", 500)

    linked = link_causal_candidate_relations(
        [shot, turnover, assist, block, rebound, steal, far_rebound],
        max_gap_frames=40,
    )
    by_id = {item.event_id: item for item in linked}

    assert by_id["assist"].related_event_ids == ["shot"]
    assert by_id["block"].related_event_ids == ["shot"]
    assert by_id["rebound"].related_event_ids == ["shot"]
    assert by_id["steal"].related_event_ids == ["turnover"]
    assert by_id["far-rebound"].related_event_ids == []


def test_player_perception_role_excludes_its_ball_and_rim_detections(tmp_path: Path) -> None:
    video = tmp_path / "game.mov"
    video.write_bytes(b"raw-video")
    raw = {
        "filename": video.name,
        "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "source_fps": 30.0,
        "frame_count": 600,
    }
    primary = tmp_path / "primary.json"
    primary.write_text(
        json.dumps(
            {
                "schema_version": "agu.official-perception.v1",
                "raw_video": raw,
                "sampling": {"stride_frames": 1, "end_frame": 600},
                "detector": {"model_sha256": "primary"},
                "detections": [
                    _detection("ball-100", 100, "basketball", (100, 60, 110, 70)).model_dump(mode="json"),
                    _detection("rim-100", 100, "rim", (90, 90, 130, 105)).model_dump(mode="json"),
                ],
            }
        ),
        encoding="utf-8",
    )
    player_source = tmp_path / "players.json"
    player = _detection("player-100", 100, "player", (80, 40, 125, 180)).model_copy(
        update={"player_id": "canonical-player", "team_id": "raw-dark"}
    )
    player_source.write_text(
        json.dumps(
            {
                "schema_version": "agu.official-perception.v1",
                "raw_video": raw,
                "detector": {"model_sha256": "player-source"},
                "detections": [
                    player.model_dump(mode="json"),
                    _detection("ball-500", 500, "basketball", (100, 60, 110, 70)).model_dump(mode="json"),
                    _detection("rim-500", 500, "rim", (90, 90, 130, 105)).model_dump(mode="json"),
                ],
            }
        ),
        encoding="utf-8",
    )

    bundle = build_candidates(
        perception_path=primary,
        player_perception_paths=[player_source],
        video_path=video,
        game_id="game",
        max_normalized_distance=4.0,
        cluster_gap_sec=3.0,
    )

    assert [event.event_type for event in bundle.events] == ["field_goal_attempt", "rebound"]
    assert bundle.events[0].primary_player_id == "perception-1:canonical-player"


def test_face_gallery_enforcement_accepts_only_enrolled_canonical_identity() -> None:
    artifact = OfficialIdentityGraphArtifactResponse(
        raw_videos=[],
        config_sha256="config",
        tracklets=[
            OfficialIdentityTrackletResponse(
                tracklet_id="enrolled-track",
                source_video_id="video_001",
                source_player_id="raw-track-1",
                team_id="raw-dark",
                start_frame=1,
                end_frame=10,
                observation_count=4,
                crop_count=4,
                embedding_model="fixture",
                gallery_person_id="roster-player-7",
                gallery_confidence=0.93,
            ),
            OfficialIdentityTrackletResponse(
                tracklet_id="anonymous-track",
                source_video_id="video_001",
                source_player_id="raw-track-2",
                team_id="raw-dark",
                start_frame=20,
                end_frame=30,
                observation_count=4,
                crop_count=4,
                embedding_model="fixture",
            ),
        ],
        identities=[
            OfficialCanonicalIdentityResponse(
                player_id="roster-player-7",
                team_id="raw-dark",
                tracklet_ids=["enrolled-track"],
                source_player_ids=["raw-track-1"],
                jersey_number="7",
                confidence=0.93,
                evidence=["face_gallery=roster-player-7", "trusted_jersey=7"],
            ),
            OfficialCanonicalIdentityResponse(
                player_id="raw-dark-player-anonymous",
                team_id="raw-dark",
                tracklet_ids=["anonymous-track"],
                source_player_ids=["raw-track-2"],
                confidence=0.8,
            ),
        ],
    )

    assert _face_gallery_anchored_player_ids(artifact) == {"roster-player-7"}
