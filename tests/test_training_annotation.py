from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis.official_evaluation import RawOnlyEvaluationError, seal_raw_only_predictions
from app.analysis.training_annotation import (
    seal_training_annotation_manifest,
    verify_training_annotation_manifest,
)


def _benchmark_bundle(path: Path) -> dict:
    return seal_raw_only_predictions(
        game_id="acceptance",
        raw_video_paths=[path],
        events=[],
        config={},
    ).model_dump(mode="json")


def test_codex_training_manifest_is_hash_bound_and_benchmark_disjoint(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.mov"
    training = tmp_path / "training.mov"
    labels = tmp_path / "labels.json"
    benchmark.write_bytes(b"acceptance")
    training.write_bytes(b"different game")
    labels.write_text('{"boxes": []}', encoding="utf-8")

    manifest = seal_training_annotation_manifest(
        producer="codex",
        source_video_paths=[training],
        annotation_paths=[labels],
        task_types=["player_bbox", "ball_bbox"],
        benchmark_bundles=[_benchmark_bundle(benchmark)],
    )

    verified = verify_training_annotation_manifest(manifest)
    assert verified["runtime_consumable"] is False
    assert verified["benchmark_overlap"] is False


def test_training_manifest_rejects_acceptance_video_reuse(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.mov"
    labels = tmp_path / "labels.json"
    benchmark.write_bytes(b"acceptance")
    labels.write_text('{"boxes": []}', encoding="utf-8")

    with pytest.raises(RawOnlyEvaluationError, match="overlap acceptance"):
        seal_training_annotation_manifest(
            producer="codex",
            source_video_paths=[benchmark],
            annotation_paths=[labels],
            task_types=["player_bbox"],
            benchmark_bundles=[_benchmark_bundle(benchmark)],
        )


def test_training_manifest_rejects_mutation(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.mov"
    training = tmp_path / "training.mov"
    labels = tmp_path / "labels.json"
    benchmark.write_bytes(b"acceptance")
    training.write_bytes(b"different game")
    labels.write_text('{"boxes": []}', encoding="utf-8")
    manifest = seal_training_annotation_manifest(
        producer="codex",
        source_video_paths=[training],
        annotation_paths=[labels],
        task_types=["player_bbox"],
        benchmark_bundles=[_benchmark_bundle(benchmark)],
    )
    manifest["runtime_consumable"] = True

    with pytest.raises(RawOnlyEvaluationError, match="hash mismatch"):
        verify_training_annotation_manifest(manifest)
