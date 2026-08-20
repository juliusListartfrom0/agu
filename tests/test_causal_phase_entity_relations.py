from __future__ import annotations

import copy

import numpy as np
import pytest

from app.analysis.causal_phase_entity_relations import (
    ENTITY_RELATION_FEATURE_NAMES,
    attach_dense_entity_relation_features,
    extract_entity_relation_frame_features,
    seal_dense_entity_relation_artifact,
    verify_dense_entity_relation_artifact,
)


def _detection(
    *,
    object_type: str,
    x: float,
    y: float,
    width: float,
    height: float,
    confidence: float = 0.9,
    team_id: str | None = None,
) -> dict:
    return {
        "object_type": object_type,
        "bbox": {
            "x1": x - width / 2.0,
            "y1": y - height / 2.0,
            "x2": x + width / 2.0,
            "y2": y + height / 2.0,
        },
        "confidence": confidence,
        "team_id": team_id,
    }


def _frame_detections() -> tuple[list[dict], list[dict], list[dict]]:
    players = [
        _detection(
            object_type="player",
            x=360,
            y=260,
            width=42,
            height=120,
            team_id="raw-light",
        ),
        _detection(
            object_type="player",
            x=440,
            y=275,
            width=40,
            height=116,
            team_id="raw-light",
        ),
        _detection(
            object_type="player",
            x=600,
            y=265,
            width=44,
            height=122,
            team_id="raw-dark",
        ),
        _detection(
            object_type="player",
            x=680,
            y=280,
            width=43,
            height=118,
            team_id="raw-dark",
        ),
    ]
    rims = [
        _detection(
            object_type="rim",
            x=520,
            y=120,
            width=60,
            height=20,
        )
    ]
    balls = [
        _detection(
            object_type="basketball",
            x=500,
            y=170,
            width=15,
            height=15,
        )
    ]
    return players, rims, balls


def test_entity_relations_are_player_order_and_horizontal_mirror_invariant() -> None:
    players, rims, balls = _frame_detections()

    original = extract_entity_relation_frame_features(
        players=players,
        rims=rims,
        balls=balls,
        frame_width=1000,
        frame_height=500,
    )
    mirrored_players = []
    for row in reversed(players):
        mirrored = copy.deepcopy(row)
        box = mirrored["bbox"]
        box["x1"], box["x2"] = 1000 - box["x2"], 1000 - box["x1"]
        mirrored_players.append(mirrored)
    mirrored_rims = copy.deepcopy(rims)
    mirrored_rims[0]["bbox"]["x1"], mirrored_rims[0]["bbox"]["x2"] = (
        1000 - mirrored_rims[0]["bbox"]["x2"],
        1000 - mirrored_rims[0]["bbox"]["x1"],
    )
    mirrored_balls = copy.deepcopy(balls)
    mirrored_balls[0]["bbox"]["x1"], mirrored_balls[0]["bbox"]["x2"] = (
        1000 - mirrored_balls[0]["bbox"]["x2"],
        1000 - mirrored_balls[0]["bbox"]["x1"],
    )
    mirrored = extract_entity_relation_frame_features(
        players=mirrored_players,
        rims=mirrored_rims,
        balls=mirrored_balls,
        frame_width=1000,
        frame_height=500,
    )

    assert len(original) == len(ENTITY_RELATION_FEATURE_NAMES)
    assert mirrored == pytest.approx(original)


def test_entity_relations_are_stable_under_uniform_pixel_scaling() -> None:
    players, rims, balls = _frame_detections()
    original = extract_entity_relation_frame_features(
        players=players,
        rims=rims,
        balls=balls,
        frame_width=1000,
        frame_height=500,
    )

    scaled_groups = []
    for group in (players, rims, balls):
        scaled = copy.deepcopy(group)
        for row in scaled:
            row["bbox"] = {
                key: value * 2.0 for key, value in row["bbox"].items()
            }
        scaled_groups.append(scaled)
    doubled = extract_entity_relation_frame_features(
        players=scaled_groups[0],
        rims=scaled_groups[1],
        balls=scaled_groups[2],
        frame_width=2000,
        frame_height=1000,
    )

    assert doubled == pytest.approx(original)


def test_entity_relations_fail_closed_to_finite_values_without_entities() -> None:
    values = extract_entity_relation_frame_features(
        players=[],
        rims=[],
        balls=[],
        frame_width=1280,
        frame_height=720,
    )

    assert len(values) == len(ENTITY_RELATION_FEATURE_NAMES)
    assert np.isfinite(values).all()
    assert values == pytest.approx([0.0] * len(values))


def _artifact() -> dict:
    features = [
        [float(index) for index in range(len(ENTITY_RELATION_FEATURE_NAMES))]
        for _ in range(24)
    ]
    return seal_dense_entity_relation_artifact(
        {
            "review_plan_sha256": "1" * 64,
            "source_video_sha256s": ["a" * 64],
            "sealed_blind_video_sha256s": ["b" * 64],
            "source_artifact_sha256s": ["c" * 64],
            "examples": [
                {
                    "phase_review_id": "phase-1",
                    "source_video_sha256": "a" * 64,
                    "candidate_bundle_sha256": "d" * 64,
                    "event_id": "event-1",
                    "frame_indexes": list(range(24)),
                    "features": features,
                }
            ],
        }
    )


def test_entity_relation_artifact_is_hash_sealed_and_exactly_joined() -> None:
    artifact = _artifact()
    assert verify_dense_entity_relation_artifact(artifact) == artifact
    examples = [
        {
            "phase_review_id": "phase-1",
            "source_video_sha256": "a" * 64,
            "candidate_bundle_sha256": "d" * 64,
            "event_id": "event-1",
            "embeddings": [[0.5, 0.25] for _ in range(24)],
        }
    ]

    fused, provenance = attach_dense_entity_relation_features(
        examples,
        relation_artifact=artifact,
    )

    assert len(fused[0]["embeddings"][0]) == (
        2 + len(ENTITY_RELATION_FEATURE_NAMES)
    )
    assert provenance["entity_relation_artifact_sha256"] == artifact[
        "artifact_sha256"
    ]

    tampered = copy.deepcopy(artifact)
    tampered["examples"][0]["features"][0][0] += 1.0
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_dense_entity_relation_artifact(tampered)

    with pytest.raises(ValueError, match="cover"):
        attach_dense_entity_relation_features(
            [{**examples[0], "event_id": "missing"}],
            relation_artifact=artifact,
        )


def test_entity_relation_artifact_rejects_blind_source_overlap() -> None:
    artifact = _artifact()
    artifact.pop("artifact_sha256")
    artifact["sealed_blind_video_sha256s"] = ["a" * 64]

    with pytest.raises(ValueError, match="sealed blind"):
        seal_dense_entity_relation_artifact(artifact)
