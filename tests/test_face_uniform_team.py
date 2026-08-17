from __future__ import annotations

import numpy as np
import pytest

from app.analysis.face_uniform_team import (
    aggregate_cluster_team,
    extract_face_anchored_uniform_features,
    fit_uniform_team_model,
    predict_uniform_team_probability,
    select_uniform_team_candidates,
)


def test_face_anchored_features_separate_blue_and_white_uniforms() -> None:
    white = np.full((120, 100, 3), 240, dtype=np.uint8)
    white[50:110, 30:70] = (20, 20, 220)
    blue = np.full((120, 100, 3), 20, dtype=np.uint8)
    blue[50:110, 30:70] = (220, 80, 20)
    bbox = [40, 10, 20, 20]

    white_features = extract_face_anchored_uniform_features(white, bbox)
    blue_features = extract_face_anchored_uniform_features(blue, bbox)

    assert white_features.shape == blue_features.shape
    assert white_features[8] > blue_features[8]
    assert blue_features[4] > white_features[4]


def test_uniform_model_is_group_cross_validated_and_serializable() -> None:
    features = np.asarray(
        [
            [0.0, 0.9],
            [0.1, 0.8],
            [0.0, 0.7],
            [0.2, 0.9],
            [0.8, 0.1],
            [0.9, 0.0],
            [0.7, 0.2],
            [1.0, 0.1],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    groups = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])

    model, audit = fit_uniform_team_model(
        features,
        labels,
        groups,
        class_ids=("hou", "orl"),
        minimum_cross_validated_accuracy=0.80,
    )

    assert audit["cross_validated_accuracy"] == 1.0
    assert audit["group_count"] == 4
    assert 0.0 <= predict_uniform_team_probability(model, [0.05, 0.85]) < 0.5
    assert 0.5 < predict_uniform_team_probability(model, [0.85, 0.05]) <= 1.0


def test_uniform_model_fails_closed_when_cross_validation_is_weak() -> None:
    features = np.asarray([[0.0], [1.0], [0.0], [1.0]])
    labels = np.asarray([0, 1, 1, 0])
    groups = np.asarray(["a", "b", "c", "d"])

    with pytest.raises(ValueError, match="cross-validated"):
        fit_uniform_team_model(
            features,
            labels,
            groups,
            class_ids=("hou", "orl"),
            minimum_cross_validated_accuracy=0.90,
        )


def test_cluster_team_requires_multiple_consistent_observations() -> None:
    assert aggregate_cluster_team(
        [0.91, 0.83, 0.72],
        class_ids=("hou", "orl"),
        minimum_observations=2,
        minimum_probability=0.65,
        minimum_consensus=0.75,
    ) == ("orl", 1.0, 0.82)
    assert aggregate_cluster_team(
        [0.9],
        class_ids=("hou", "orl"),
        minimum_observations=2,
        minimum_probability=0.65,
        minimum_consensus=0.75,
    )[0] is None
    assert aggregate_cluster_team(
        [0.9, 0.1],
        class_ids=("hou", "orl"),
        minimum_observations=2,
        minimum_probability=0.65,
        minimum_consensus=0.75,
    )[0] is None


def test_uniform_team_subset_preserves_only_sealed_matching_clusters() -> None:
    import hashlib
    import json

    def seal(payload: dict) -> dict:
        payload["manifest_sha256"] = ""
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        payload["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
        return payload

    candidates = seal(
        {
            "schema_version": "agu.face-enrollment-candidates.v1",
            "benchmark_disjoint": True,
            "model_id": "face",
            "sources": [],
            "config": {},
            "clusters": [
                {"cluster_id": "a", "samples": []},
                {"cluster_id": "b", "samples": []},
            ],
        }
    )
    context = seal(
        {
            "schema_version": "agu.face-uniform-team-context.v1",
            "benchmark_disjoint": True,
            "source_candidate_manifest_sha256": candidates["manifest_sha256"],
            "clusters": [
                {"cluster_id": "a", "team_id": "orl"},
                {"cluster_id": "b", "team_id": "hou"},
            ],
        }
    )

    subset = select_uniform_team_candidates(
        candidates,
        context,
        team_id="orl",
    )

    assert [row["cluster_id"] for row in subset["clusters"]] == ["a"]
    assert subset["manifest_sha256"]
