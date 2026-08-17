from __future__ import annotations

from app.analysis.shot_reason_evidence import (
    FEATURE_NAMES,
    seal_reason_evidence_artifact,
)
from app.analysis.shot_reason_fusion import screen_shot_reason_fusion
from app.analysis.shot_validity_scene_state import (
    SCENE_BACKBONE,
    SCENE_EMBEDDING_DIMENSION,
    seal_scene_embedding_artifact,
)


def test_nested_reason_scores_never_train_on_outer_held_or_scored_game() -> None:
    examples = []
    evidence_examples = []
    reviews = []
    for game_index in range(4):
        game = f"{game_index + 1:064x}"
        bundle = f"{game_index + 101:064x}"
        for example_index in range(2):
            event_id = f"event-{game_index}-{example_index}"
            examples.append(
                {
                    "source_video_sha256": game,
                    "candidate_bundle_sha256": bundle,
                    "event_id": event_id,
                    "event_present": example_index == 1,
                    "phase_embeddings": [
                        [float(example_index)] * SCENE_EMBEDDING_DIMENSION,
                        [float(example_index)] * SCENE_EMBEDDING_DIMENSION,
                        [float(example_index)] * SCENE_EMBEDDING_DIMENSION,
                    ],
                }
            )
            evidence_examples.append(
                {
                    "source_video_sha256": game,
                    "candidate_bundle_sha256": bundle,
                    "event_id": event_id,
                    "features": [
                        float(example_index),
                        *([float(game_index)] * (len(FEATURE_NAMES) - 1)),
                    ],
                }
            )
            reviews.append(
                {
                    "source_video_sha256": game,
                    "candidate_bundle_sha256": bundle,
                    "event_id": event_id,
                    "release_observed": example_index == 1,
                    "ball_moves_toward_rim": example_index == 1,
                    "free_throw_formation": example_index == 0,
                    "replay_or_stoppage": example_index == 0,
                    "review_confidence": "high",
                }
            )
    scene = seal_scene_embedding_artifact(
        {
            "purpose": "test",
            "producer": "test",
            "training_manifest_sha256": "a" * 64,
            "training_annotation_sha256": ["b" * 64],
            "source_video_sha256": [f"{index + 1:064x}" for index in range(4)],
            "backbone": SCENE_BACKBONE,
            "backbone_sha256": "c" * 64,
            "backbone_license": "BSD-3-Clause",
            "backbone_weights_url": "https://example.invalid/model",
            "embedding_dimension": SCENE_EMBEDDING_DIMENSION,
            "phase_fractions": [0.15, 0.5, 0.85],
            "examples": examples,
        }
    )
    evidence = seal_reason_evidence_artifact(
        {
            "source_scene_artifact_sha256": scene["artifact_sha256"],
            "pose_source_sha256s": ["d" * 64],
            "candidate_bundle_sha256s": [
                f"{game_index + 101:064x}" for game_index in range(4)
            ],
            "examples": evidence_examples,
        }
    )

    result = screen_shot_reason_fusion(
        scene_artifact=scene,
        video_artifacts=[],
        reason_evidence_artifact=evidence,
        review_rows=reviews,
        pca_components=2,
        regularization_c=0.01,
    )

    assert {row["name"] for row in result["variants"]} == {
        "base",
        "base+continuous_raw",
        "base+nested_reason_scores",
        "base+continuous_raw+nested_reason_scores",
    }
    for outer in result["auxiliary_provenance"]:
        held_game = outer["held_game_sha256"]
        for reason in outer["reasons"]:
            assert held_game not in reason["held_score_training_game_sha256s"]
            for training_score in reason["training_scores"]:
                assert held_game not in training_score["training_game_sha256s"]
                assert (
                    training_score["scored_game_sha256"]
                    not in training_score["training_game_sha256s"]
                )
