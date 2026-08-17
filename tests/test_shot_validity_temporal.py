from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.analysis.schemas import EventEvidenceResponse, GameEventResponse
from app.analysis.shot_validity_temporal import (
    TEMPORAL_SHOT_EMBEDDING_DIMENSION,
    TemporalShotValidityModel,
    apply_temporal_shot_gate,
    seal_temporal_embedding_artifact,
    seal_temporal_shot_model,
    verify_temporal_shot_model,
)
from scripts.train_shot_validity_temporal_model import train_temporal_model


def _vector(value: float) -> list[float]:
    return [value] * TEMPORAL_SHOT_EMBEDDING_DIMENSION


def _shot(event_id: str, confidence: float) -> GameEventResponse:
    return GameEventResponse(
        event_id=event_id,
        revision=1,
        event_type="field_goal_attempt",
        source_video_id="video_001",
        start_frame=90,
        end_frame=130,
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
            )
        ],
    )


def _model() -> dict[str, object]:
    return seal_temporal_shot_model(
        {
            "training_manifest_sha256": "manifest",
            "backbone_sha256": "backbone",
            "clip_frames": 16,
            "threshold": 0.5,
            "coefficients": _vector(1.0),
            "intercept": 0.0,
            "feature_mean": _vector(0.0),
            "feature_scale": _vector(1.0),
        }
    )


def test_temporal_shot_model_is_hash_bound_and_predicts() -> None:
    artifact = _model()
    model = TemporalShotValidityModel(artifact)

    assert model.predict_probability(_vector(0.1)) > 0.99
    assert model.accepts(_vector(0.1)) is True

    artifact["threshold"] = 0.9
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_temporal_shot_model(artifact)


def test_temporal_training_uses_leave_one_video_out_calibration(tmp_path: Path) -> None:
    examples = []
    for game, shift in (("game-a", 0.0), ("game-b", 0.1)):
        examples.extend(
            [
                {
                    "source_video_sha256": game,
                    "event_id": f"{game}-positive-{index}",
                    "event_present": True,
                    "embedding": _vector(2.0 + shift + index * 0.1),
                }
                for index in range(3)
            ]
        )
        examples.extend(
            [
                {
                    "source_video_sha256": game,
                    "event_id": f"{game}-negative-{index}",
                    "event_present": False,
                    "embedding": _vector(-2.0 + shift - index * 0.1),
                }
                for index in range(3)
            ]
        )
    embeddings = seal_temporal_embedding_artifact(
        {
            "purpose": "model_training_only",
            "producer": "agu",
            "training_manifest_sha256": "manifest",
            "training_annotation_sha256": ["labels"],
            "backbone_sha256": "backbone",
            "clip_frames": 16,
            "examples": examples,
        }
    )
    path = tmp_path / "embeddings.json"
    path.write_text(json.dumps(embeddings), encoding="utf-8")

    artifact = train_temporal_model(embedding_path=path)

    assert artifact["metrics"]["training_video_count"] == 2
    assert artifact["metrics"]["leave_one_video_out_precision"] == 1.0
    assert artifact["metrics"]["leave_one_video_out_recall"] == 1.0
    assert artifact["metrics"]["minimum_observed_per_video_recall"] == 1.0
    assert artifact["metrics"]["every_video_precision_gate"] is True
    assert artifact["metrics"]["promotion_gate"] is True
    assert set(artifact["metrics"]["leave_one_video_out_per_video"]) == {
        "game-a",
        "game-b",
    }


def test_temporal_gate_rejects_shot_and_dependent_rebound() -> None:
    accepted = _shot("accepted", 0.9)
    rejected = _shot("rejected", 0.2)
    rebound = _shot("rebound", 0.8).model_copy(
        update={"event_type": "rebound", "related_event_ids": ["rejected"]}
    )
    model = TemporalShotValidityModel(_model())

    events = apply_temporal_shot_gate(
        [accepted, rejected, rebound],
        shot_events=[accepted, rejected],
        embeddings=[_vector(0.1), _vector(-0.1)],
        model=model,
    )

    assert [event.event_id for event in events] == ["accepted"]
    assert events[0].evidence[-1].kind == "traditional_temporal_shot_validity_model"
