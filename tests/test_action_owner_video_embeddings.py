import copy

import pytest

from scripts.extract_action_owner_video_embeddings import (
    ACTION_OWNER_FEATURES,
    EMBEDDING_SCHEMA,
    PRETRAINED_BACKBONE,
    _json_sha256,
    _sample_frame_indexes,
    _square_crop_bounds,
    verify_embedding_artifact,
)


def test_sample_frame_indexes_is_deterministic_and_pads_short_tracks() -> None:
    assert _sample_frame_indexes([12], 4) == [12, 12, 12, 12]
    assert _sample_frame_indexes([20, 10, 30], 5) == [10, 10, 20, 30, 30]


def test_square_crop_bounds_clamps_to_video() -> None:
    bounds = _square_crop_bounds(
        {"x1": -10, "y1": 5, "x2": 30, "y2": 105},
        frame_width=80,
        frame_height=120,
        padding=0.25,
    )
    assert bounds == (0, 0, 80, 120)


def test_embedding_artifact_is_non_runtime_and_hash_sealed() -> None:
    artifact = {
        "schema_version": EMBEDDING_SCHEMA,
        "producer": "agu",
        "purpose": "model_training_only",
        "runtime_consumable": False,
        "training_manifest_sha256": "manifest",
        "training_annotation_sha256": ["annotation"],
        "source_video_sha256": ["video"],
        "backbone": PRETRAINED_BACKBONE,
        "embedding_dimension": 512,
        "clip_frames": 16,
        "crop_padding": 0.25,
        "traditional_feature_names": list(ACTION_OWNER_FEATURES),
        "examples": [],
    }
    artifact["artifact_sha256"] = _json_sha256(artifact)
    assert verify_embedding_artifact(artifact)["runtime_consumable"] is False

    tampered = copy.deepcopy(artifact)
    tampered["runtime_consumable"] = True
    with pytest.raises(ValueError, match="non-runtime"):
        verify_embedding_artifact(tampered)
