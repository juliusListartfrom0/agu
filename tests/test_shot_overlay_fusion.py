from __future__ import annotations

from app.analysis.shot_broadcast_state import (
    FEATURE_NAMES as BROADCAST_FEATURE_NAMES,
)
from app.analysis.shot_broadcast_state import (
    seal_broadcast_state_evidence_artifact,
)
from app.analysis.shot_overlay_fusion import screen_shot_overlay_fusion
from app.analysis.shot_overlay_state import (
    FEATURE_NAMES as OVERLAY_FEATURE_NAMES,
)
from app.analysis.shot_overlay_state import (
    seal_overlay_state_evidence_artifact,
)
from app.analysis.shot_validity_scene_state import (
    SCENE_BACKBONE,
    SCENE_EMBEDDING_DIMENSION,
    seal_scene_embedding_artifact,
)


def test_overlay_fusion_keeps_fixed_variants_and_outer_game_isolation() -> None:
    scene_examples = []
    broadcast_examples = []
    overlay_examples = []
    for game_index in range(4):
        game = f"{game_index + 1:064x}"
        bundle = f"{game_index + 101:064x}"
        for example_index in range(4):
            event_id = f"event-{game_index}-{example_index}"
            positive = example_index % 2 == 1
            key = {
                "source_video_sha256": game,
                "candidate_bundle_sha256": bundle,
                "event_id": event_id,
            }
            scene_examples.append(
                {
                    **key,
                    "event_present": positive,
                    "phase_embeddings": [
                        [float(example_index)] * SCENE_EMBEDDING_DIMENSION,
                        [float(example_index + game_index)] * SCENE_EMBEDDING_DIMENSION,
                        [float(example_index + 1)] * SCENE_EMBEDDING_DIMENSION,
                    ],
                }
            )
            broadcast_features = [0.0] * len(BROADCAST_FEATURE_NAMES)
            broadcast_features[
                BROADCAST_FEATURE_NAMES.index("clock_elapsed_seconds")
            ] = float(positive)
            overlay_features = [0.0] * len(OVERLAY_FEATURE_NAMES)
            overlay_features[
                OVERLAY_FEATURE_NAMES.index("score_change_flag")
            ] = float(positive)
            broadcast_examples.append({**key, "features": broadcast_features})
            overlay_examples.append({**key, "features": overlay_features})
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
            "examples": scene_examples,
        }
    )
    broadcast = seal_broadcast_state_evidence_artifact(
        {
            "source_scene_artifact_sha256": scene["artifact_sha256"],
            "source_clock_artifact_sha256s": ["d" * 64],
            "examples": broadcast_examples,
        }
    )
    overlay = seal_overlay_state_evidence_artifact(
        {
            "source_scene_artifact_sha256": scene["artifact_sha256"],
            "source_clock_artifact_sha256s": ["d" * 64],
            "source_scoreboard_timeline_sha256s": [
                f"{index + 10:064x}" for index in range(4)
            ],
            "examples": overlay_examples,
        }
    )

    result = screen_shot_overlay_fusion(
        scene_artifact=scene,
        video_artifacts=[],
        broadcast_state_artifact=broadcast,
        overlay_state_artifact=overlay,
        pca_components=2,
        regularization_c=0.01,
    )

    assert result["selection_protocol"] == "outer_game_held_overlay_state_oof"
    assert {row["name"] for row in result["variants"]} == {
        "base",
        "base+broadcast_raw",
        "base+overlay_raw",
        "base+broadcast+overlay_raw",
    }
    assert len(result["fold_provenance"]) == 4
    for fold in result["fold_provenance"]:
        assert fold["held_game_sha256"] not in fold["training_game_sha256s"]
        assert len(fold["training_game_sha256s"]) == 3
