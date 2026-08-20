from __future__ import annotations

import copy

import pytest

from app.analysis.shot_reason_evidence import (
    FEATURE_NAMES,
    build_reason_evidence_artifact,
    extract_reason_evidence_features,
    seal_reason_evidence_artifact,
    verify_reason_evidence_artifact,
)
from app.analysis.shot_validity_scene_state import (
    SCENE_BACKBONE,
    SCENE_EMBEDDING_DIMENSION,
    seal_scene_embedding_artifact,
)


def _player(frame: int, *, x1: float, raised: bool = False) -> dict[str, object]:
    shoulder_y = 40.0
    wrist_y = 20.0 if raised else 60.0
    return {
        "frame": frame,
        "object_type": "player",
        "confidence": 0.9,
        "bbox": {"x1": x1, "y1": 20.0, "x2": x1 + 20.0, "y2": 100.0},
        "keypoints": {
            "left_shoulder": {"x": x1 + 8.0, "y": shoulder_y},
            "left_wrist": {"x": x1 + 8.0, "y": wrist_y},
        },
    }


def test_reason_features_use_only_window_frames_and_are_finite() -> None:
    detections = [
        _player(9, x1=0.0),
        _player(10, x1=10.0),
        _player(10, x1=60.0, raised=True),
        _player(12, x1=20.0),
        _player(12, x1=70.0, raised=True),
        _player(13, x1=0.0),
    ]
    event = {
        "evidence": [
            {
                "kind": "ball_rim_proximity_cluster",
                "details": {
                    "hit_count": 3,
                    "minimum_normalized_distance": 0.25,
                    "maximum_approach_point_count": 4,
                    "maximum_approach_rise_px": 24.0,
                    "candidate_player_observations": [
                        {
                            "player_id": "p1",
                            "team_id": "dark",
                            "frame": 10,
                            "wrist_ball_distance": 0.1,
                            "stable_control_score": 0.8,
                            "stable_contact_count": 2,
                        },
                        {
                            "player_id": "p1",
                            "team_id": "dark",
                            "frame": 12,
                            "wrist_ball_distance": 0.4,
                            "stable_control_score": 0.2,
                            "stable_contact_count": 0,
                        }
                    ],
                    "review_rim_observations": [
                        {
                            "frame": 10,
                            "ball_bbox": {
                                "x1": 0.0,
                                "y1": 0.0,
                                "x2": 10.0,
                                "y2": 10.0,
                            },
                            "rim_bbox": {
                                "x1": 40.0,
                                "y1": 40.0,
                                "x2": 50.0,
                                "y2": 50.0,
                            },
                        },
                        {
                            "frame": 12,
                            "ball_bbox": {
                                "x1": 20.0,
                                "y1": 20.0,
                                "x2": 30.0,
                                "y2": 30.0,
                            },
                            "rim_bbox": {
                                "x1": 40.0,
                                "y1": 40.0,
                                "x2": 50.0,
                                "y2": 50.0,
                            },
                        },
                    ],
                },
            }
        ]
    }

    features = extract_reason_evidence_features(
        detections=detections,
        start_frame=10,
        end_frame=12,
        frame_width=100,
        frame_height=100,
        event=event,
    )

    assert set(features) == set(FEATURE_NAMES)
    assert features["sampled_frame_count"] == 2.0
    assert features["player_count_mean"] == 2.0
    assert features["raised_wrist_fraction"] == 0.5
    assert features["ball_rim_hit_count"] == 3.0
    assert features["candidate_observation_count"] == 2.0
    assert features["nearest_player_motion_mean"] == pytest.approx(0.1)
    assert features["wrist_ball_distance_range"] == pytest.approx(0.3)
    assert features["wrist_ball_distance_last_minus_first"] == pytest.approx(0.3)
    assert features["ball_rim_distance_last_minus_first"] < 0
    assert features["ball_path_displacement"] > 0
    assert all(value == pytest.approx(float(value)) for value in features.values())


def test_reason_evidence_artifact_is_label_free_and_tamper_evident() -> None:
    artifact = seal_reason_evidence_artifact(
        {
            "source_scene_artifact_sha256": "a" * 64,
            "pose_source_sha256s": ["b" * 64],
            "candidate_bundle_sha256s": ["c" * 64],
            "examples": [
                {
                    "source_video_sha256": "d" * 64,
                    "candidate_bundle_sha256": "c" * 64,
                    "event_id": "event-1",
                    "features": [0.0] * len(FEATURE_NAMES),
                }
            ],
        }
    )

    assert artifact["runtime_consumable"] is False
    assert artifact["schema_version"] == "agu.shot-reason-evidence.v2"
    assert artifact["feature_names"] == list(FEATURE_NAMES)
    assert "event_present" not in str(artifact)
    assert verify_reason_evidence_artifact(artifact)["artifact_sha256"]

    leaked = copy.deepcopy(artifact)
    leaked["examples"][0]["event_present"] = True
    with pytest.raises(ValueError, match="labels"):
        seal_reason_evidence_artifact(leaked)

    tampered = copy.deepcopy(artifact)
    tampered["examples"][0]["features"][0] = 1.0
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_reason_evidence_artifact(tampered)


def test_reason_evidence_builder_aligns_scene_pose_and_candidate_sources() -> None:
    scene = seal_scene_embedding_artifact(
        {
            "purpose": "test",
            "producer": "test",
            "training_manifest_sha256": "a" * 64,
            "training_annotation_sha256": ["b" * 64],
            "source_video_sha256": ["d" * 64],
            "backbone": SCENE_BACKBONE,
            "backbone_sha256": "e" * 64,
            "backbone_license": "BSD-3-Clause",
            "backbone_weights_url": "https://example.invalid/model",
            "embedding_dimension": SCENE_EMBEDDING_DIMENSION,
            "phase_fractions": [0.15, 0.5, 0.85],
            "examples": [
                {
                    "source_video_sha256": "d" * 64,
                    "candidate_bundle_sha256": "c" * 64,
                    "event_id": "event-1",
                    "event_present": True,
                    "phase_embeddings": [
                        [0.0] * SCENE_EMBEDDING_DIMENSION,
                        [0.0] * SCENE_EMBEDDING_DIMENSION,
                        [0.0] * SCENE_EMBEDDING_DIMENSION,
                    ],
                }
            ],
        }
    )
    pose = {
        "schema_version": "agu.official-pose.v1",
        "raw_video": {"sha256": "d" * 64},
        "detections": [
            _player(10, x1=10.0),
            _player(20, x1=20.0),
        ],
    }
    candidate = {
        "bundle_sha256": "c" * 64,
        "events": [
            {
                "event_id": "event-1",
                "start_frame": 10,
                "end_frame": 10,
                "evidence": [],
            }
        ],
    }

    artifact = build_reason_evidence_artifact(
        scene_artifact=scene,
        pose_artifacts=[pose],
        pose_source_sha256s=["f" * 64],
        candidate_bundles=[candidate],
        video_dimensions={"d" * 64: (100, 100)},
    )

    assert artifact["source_scene_artifact_sha256"] == scene["artifact_sha256"]
    assert artifact["examples"][0]["features"][0] == 1.0
    assert "event_present" not in str(artifact)
