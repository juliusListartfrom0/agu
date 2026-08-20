from __future__ import annotations

import hashlib
import json

import pytest

from app.analysis.schemas import EventEvidenceResponse, GameEventResponse
from app.analysis.shot_validity import (
    SHOT_VALIDITY_FEATURES_V1,
    SHOT_VALIDITY_FEATURES_V2,
    SHOT_VALIDITY_MODEL_SCHEMA_V1,
    SHOT_VALIDITY_MODEL_SCHEMA_V2,
    ShotValidityModel,
    apply_shot_validity_gate,
    extract_shot_validity_features,
    seal_shot_validity_model,
)


def _shot(event_id: str, confidence: float) -> GameEventResponse:
    return GameEventResponse(
        event_id=event_id,
        revision=1,
        event_type="field_goal_attempt",
        source_video_id="video_001",
        start_frame=90,
        end_frame=130,
        release_frame=110,
        outcome="unknown",
        status="needs_review",
        confidence=confidence,
        evidence=[
            EventEvidenceResponse(
                evidence_id=f"{event_id}:vision",
                kind="ball_rim_proximity_cluster",
                source_video_id="video_001",
                start_frame=90,
                end_frame=130,
                confidence=confidence,
                details={
                    "frames": [100, 105, 110],
                    "hit_count": 3,
                    "minimum_normalized_distance": 0.4,
                    "maximum_approach_point_count": 2,
                    "maximum_approach_rise_px": 30,
                    "trajectory_outcome": "unknown",
                    "candidate_player_observations": [
                        {
                            "player_id": "p1",
                            "frame": 108,
                            "bbox": {"x1": 0, "y1": 0, "x2": 40, "y2": 100},
                            "ball_player_distance": 0.3,
                            "ball_center": {"x": 20, "y": 20},
                            "keypoints": {
                                "left_shoulder": {"x": 20, "y": 50},
                                "left_wrist": {"x": 20, "y": 22},
                            },
                        }
                    ],
                },
            )
        ],
    )


def _model() -> ShotValidityModel:
    return ShotValidityModel(
        seal_shot_validity_model(
            {
                "training_manifest_sha256": "manifest",
                "threshold": 0.8,
                "trees": [
                    {
                        "children_left": [1, -1, -1],
                        "children_right": [2, -1, -1],
                        "feature": [0, -2, -2],
                        "threshold": [0.5, -2.0, -2.0],
                        "positive_probability": [0.5, 0.1, 0.9],
                    }
                ],
            }
        )
    )


def test_extract_shot_validity_features_uses_raw_candidate_evidence() -> None:
    features = extract_shot_validity_features(_shot("shot", 0.9))

    assert features["candidate_confidence"] == 0.9
    assert features["hit_count"] == 3
    assert features["normalized_approach_rise"] == 0.3
    assert features["ball_frame_count"] == 3
    assert features["minimum_ball_player_distance"] == 0.3
    assert features["best_minimum_wrist_ball_distance"] == 0.02


def test_extract_features_prefers_window_causal_evidence() -> None:
    shot = _shot("shot", 0.9)
    window_evidence = shot.evidence[0].model_copy(
        update={
            "evidence_id": "shot:window-causal",
            "details": {
                **shot.evidence[0].details,
                "causal_evidence_version": "agu_window_ball_rim_pose_v1",
                "hit_count": 7,
                "minimum_normalized_distance": 0.2,
            },
        }
    )

    features = extract_shot_validity_features(
        shot.model_copy(update={"evidence": [*shot.evidence, window_evidence]})
    )

    assert features["hit_count"] == 3
    assert features["minimum_normalized_distance"] == 0.4
    assert features["window_causal_present"] == 1
    assert features["window_maximum_approach_point_count"] == 2
    assert features["window_minimum_normalized_distance"] == 0.2


def test_gate_drops_rejected_shot_and_related_rebound() -> None:
    accepted = _shot("accepted", 0.9)
    rejected = _shot("rejected", 0.2)
    rebound = GameEventResponse(
        event_id="rebound",
        revision=1,
        event_type="rebound",
        source_video_id="video_001",
        start_frame=130,
        end_frame=160,
        status="needs_review",
        related_event_ids=["rejected"],
    )

    events = apply_shot_validity_gate(
        [accepted, rejected, rebound], model=_model()
    )

    assert [event.event_id for event in events] == ["accepted"]
    assert events[0].status == "needs_review"
    assert events[0].evidence[-1].kind == "traditional_shot_validity_model"
    assert events[0].evidence[-1].confidence == 0.9


def test_shot_validity_model_rejects_hash_tampering() -> None:
    artifact = dict(_model().artifact)
    artifact["threshold"] = 0.1

    with pytest.raises(ValueError, match="hash mismatch"):
        ShotValidityModel(artifact)


def test_shot_validity_model_keeps_v1_artifacts_readable() -> None:
    artifact = dict(_model().artifact)
    artifact["schema_version"] = SHOT_VALIDITY_MODEL_SCHEMA_V1
    artifact["feature_names"] = list(SHOT_VALIDITY_FEATURES_V1)
    artifact.pop("model_sha256")
    artifact["model_sha256"] = hashlib.sha256(
        json.dumps(
            artifact,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    assert ShotValidityModel(artifact).predict(_shot("legacy", 0.9)).accepted is True


def test_shot_validity_model_keeps_v2_artifacts_readable() -> None:
    artifact = dict(_model().artifact)
    artifact["schema_version"] = SHOT_VALIDITY_MODEL_SCHEMA_V2
    artifact["feature_names"] = list(SHOT_VALIDITY_FEATURES_V2)
    artifact.pop("model_sha256")
    artifact["model_sha256"] = hashlib.sha256(
        json.dumps(
            artifact,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    assert ShotValidityModel(artifact).predict(_shot("legacy-v2", 0.9)).accepted is True
