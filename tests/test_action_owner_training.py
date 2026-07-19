from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.analysis.action_ownership import ActionOwnerModel
from app.analysis.official_evaluation import seal_raw_only_predictions
from app.analysis.training_annotation import seal_training_annotation_manifest
from scripts.build_action_owner_training_labels import _bounded_observations
from scripts.train_action_owner_model import train_action_owner_model


def _observation(player: str, frame: int, wrist_y: float) -> dict:
    return {
        "player_id": player,
        "frame": frame,
        "ball_player_distance": 0.1 if player == "shooter" else 1.2,
        "ball_frame": frame,
        "ball_center": {
            "x": 30,
            "y": wrist_y + (10 if player == "shooter" else 120),
        },
        "bbox": {"x1": 0, "y1": 0, "x2": 50, "y2": 100},
        "keypoints": {
            "right_shoulder": {"x": 20, "y": 50},
            "right_elbow": {"x": 25, "y": 40},
            "right_wrist": {"x": 30, "y": wrist_y},
        },
    }


def test_training_requires_sha_bound_disjoint_annotations(tmp_path: Path) -> None:
    source = tmp_path / "training.mp4"
    source.write_bytes(b"training-video")
    benchmark = tmp_path / "benchmark.mp4"
    benchmark.write_bytes(b"benchmark-video")
    annotation = tmp_path / "labels.json"
    examples = []
    for anchor in (10, 30):
        observations = []
        for frame, wrist in ((anchor - 2, 60), (anchor, 20), (anchor + 2, 25)):
            observations.extend(
                [_observation("shooter", frame, wrist), _observation("other", frame, 70)]
            )
        examples.append(
            {
                "anchor_frame": anchor,
                "positive_player_id": "shooter",
                "candidate_player_observations": observations,
            }
        )
    annotation.write_text(
        json.dumps(
            {
                "schema_version": "agu.action-ownership-labels.v1",
                "source_video_sha256": __import__("hashlib").sha256(b"training-video").hexdigest(),
                "examples": examples,
            }
        ),
        encoding="utf-8",
    )
    manifest = seal_training_annotation_manifest(
        producer="codex",
        source_video_paths=[source],
        annotation_paths=[annotation],
        task_types=["shot_release_actor"],
        benchmark_bundles=[
            seal_raw_only_predictions(
                game_id="acceptance",
                raw_video_paths=[benchmark],
                events=[],
                config={},
            ).model_dump(mode="json")
        ],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    artifact = train_action_owner_model(
        manifest_path=manifest_path,
        annotation_paths=[annotation],
    )

    assert artifact["metrics"]["training_actions"] == 2
    assert ActionOwnerModel(artifact).artifact["training_manifest_sha256"] == manifest["manifest_sha256"]
    tree_artifact = train_action_owner_model(
        manifest_path=manifest_path,
        annotation_paths=[annotation],
        model_type="extra_trees",
        tree_count=5,
        tree_max_depth=2,
        tree_min_samples_leaf=1,
    )
    assert tree_artifact["schema_version"] == "agu.action-owner-extra-trees-model.v1"
    assert len(tree_artifact["trees"]) == 5
    assert ActionOwnerModel(tree_artifact).artifact["model_sha256"] == tree_artifact["model_sha256"]
    annotation.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not SHA-bound"):
        train_action_owner_model(manifest_path=manifest_path, annotation_paths=[annotation])


def test_codex_training_selection_can_bound_verified_track_segment() -> None:
    observations = [
        {"player_id": "shooter", "frame": frame} for frame in (90, 100, 110, 160)
    ]

    bounded = _bounded_observations(
        observations,
        {
            "anchor_frame": 105,
            "observation_start_frame": 100,
            "observation_end_frame": 110,
        },
        event_id="shot-1",
    )

    assert [item["frame"] for item in bounded] == [100, 110]
