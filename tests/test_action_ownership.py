from __future__ import annotations

import pytest

from app.analysis.action_ownership import (
    ACTION_OWNER_FEATURES,
    ActionOwnerModel,
    extract_action_owner_features,
    seal_action_owner_model,
    seal_extra_trees_action_owner_model,
)


def _observation(player_id: str, frame: int, distance: float, wrist_y: float) -> dict:
    return {
        "player_id": player_id,
        "frame": frame,
        "ball_player_distance": distance,
        "ball_frame": frame,
        "ball_center": {"x": 30, "y": wrist_y + distance * 100},
        "bbox": {"x1": 0, "y1": 0, "x2": 50, "y2": 100},
        "keypoints": {
            "right_shoulder": {"x": 20, "y": 50},
            "right_elbow": {"x": 25, "y": (50 + wrist_y) / 2},
            "right_wrist": {"x": 30, "y": wrist_y},
        },
    }


def test_action_owner_features_capture_release_motion() -> None:
    observations = [
        _observation("p1", 8, 0.2, 60),
        _observation("p1", 10, 0.1, 30),
        _observation("p1", 12, 0.8, 20),
    ]

    features = extract_action_owner_features(observations, anchor_frame=10)

    assert features["minimum_ball_distance"] == pytest.approx(0.1)
    assert features["ball_distance_exit_slope"] == pytest.approx(0.35)
    assert features["minimum_wrist_ball_distance"] == pytest.approx(0.1)
    assert features["mean_wrist_ball_distance"] == pytest.approx(1.1 / 3)
    assert features["wrist_ball_exit_slope"] == pytest.approx(0.35)
    assert features["wrist_ball_contact_coverage"] == 1.0
    assert features["maximum_wrist_above_shoulder"] == pytest.approx(0.3)
    assert features["maximum_wrist_upward_velocity"] > 0
    assert features["pose_coverage"] == 1.0


def test_sealed_action_owner_model_ranks_candidates_and_detects_tampering() -> None:
    coefficients = [0.0] * len(ACTION_OWNER_FEATURES)
    coefficients[ACTION_OWNER_FEATURES.index("maximum_wrist_above_shoulder")] = 4.0
    artifact = seal_action_owner_model(
        {
            "training_manifest_sha256": "training-sha",
            "coefficients": coefficients,
            "intercept": 0.0,
            "feature_mean": [0.0] * len(ACTION_OWNER_FEATURES),
            "feature_scale": [1.0] * len(ACTION_OWNER_FEATURES),
        }
    )
    model = ActionOwnerModel(artifact)
    observations = [
        _observation("shooter", 10, 0.2, 20),
        _observation("other", 10, 0.1, 70),
    ]

    ranked = model.rank(observations, anchor_frame=10)

    assert [item.player_id for item in ranked] == ["shooter", "other"]
    artifact["intercept"] = 9.0
    with pytest.raises(ValueError, match="hash mismatch"):
        ActionOwnerModel(artifact)


def test_sealed_extra_trees_model_ranks_without_sklearn_runtime() -> None:
    release_feature = ACTION_OWNER_FEATURES.index("maximum_wrist_above_shoulder")
    artifact = seal_extra_trees_action_owner_model(
        {
            "training_manifest_sha256": "training-sha",
            "trees": [
                {
                    "children_left": [1, -1, -1],
                    "children_right": [2, -1, -1],
                    "feature": [release_feature, -2, -2],
                    "threshold": [0.0, -2.0, -2.0],
                    "positive_probability": [0.5, 0.1, 0.9],
                }
            ],
        }
    )
    observations = [
        _observation("shooter", 10, 0.2, 20),
        _observation("other", 10, 0.1, 70),
    ]

    ranked = ActionOwnerModel(artifact).rank(observations, anchor_frame=10)

    assert [item.player_id for item in ranked] == ["shooter", "other"]
    assert [item.probability for item in ranked] == [0.9, 0.1]
    artifact["trees"][0]["threshold"][0] = 9.0
    with pytest.raises(ValueError, match="hash mismatch"):
        ActionOwnerModel(artifact)


def test_action_owner_runtime_features_ignore_distant_event_observations() -> None:
    artifact = seal_extra_trees_action_owner_model(
        {
            "training_manifest_sha256": "training-sha",
            "trees": [
                {
                    "children_left": [-1],
                    "children_right": [-1],
                    "feature": [-2],
                    "threshold": [-2.0],
                    "positive_probability": [0.7],
                }
            ],
        }
    )
    predictions = ActionOwnerModel(artifact).rank(
        [
            _observation("near", 100, 0.4, 30),
            _observation("near", 500, 0.01, 30),
            _observation("far-only", 500, 0.01, 30),
        ],
        anchor_frame=100,
        maximum_frame_distance=20,
    )

    assert [item.player_id for item in predictions] == ["near"]
    assert predictions[0].features["minimum_ball_distance"] == pytest.approx(0.4)
