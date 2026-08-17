from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.analysis.official_evaluation import seal_raw_only_predictions
from app.analysis.schemas import EventEvidenceResponse, GameEventResponse
from app.analysis.shot_validity import SHOT_VALIDITY_LABEL_SCHEMA, ShotValidityModel
from app.analysis.training_annotation import seal_training_annotation_manifest
from scripts.train_shot_validity_model import (
    _calibrate_group_threshold,
    _calibrate_threshold,
    train_shot_validity_model,
)


def _event(event_id: str, confidence: float) -> GameEventResponse:
    return GameEventResponse(
        event_id=event_id,
        revision=1,
        event_type="field_goal_attempt",
        source_video_id="video_001",
        start_frame=10,
        end_frame=30,
        confidence=confidence,
        evidence=[
            EventEvidenceResponse(
                evidence_id=f"{event_id}:candidate",
                kind="ball_rim_proximity_cluster",
                source_video_id="video_001",
                start_frame=10,
                end_frame=30,
                confidence=confidence,
                details={"frames": [15, 20], "hit_count": 2},
            )
        ],
    )


def _write_training_game(
    root: Path,
    name: str,
    *,
    model_provenance: dict[str, str] | None = None,
) -> tuple[Path, Path, Path, object]:
    video = root / f"{name}.mp4"
    video.write_bytes(f"raw-{name}".encode())
    bundle = seal_raw_only_predictions(
        game_id=name,
        raw_video_paths=[video],
        events=[_event(f"{name}-positive", 0.9), _event(f"{name}-negative", 0.1)],
        config={"fixture": True},
        model_provenance=model_provenance,
    )
    bundle_path = root / f"{name}-candidates.json"
    bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    labels = {
        "schema_version": SHOT_VALIDITY_LABEL_SCHEMA,
        "source_video_sha256": bundle.raw_videos[0].sha256,
        "candidate_bundle_sha256": bundle.bundle_sha256,
        "examples": [
            {"event_id": f"{name}-positive", "event_present": True},
            {"event_id": f"{name}-negative", "event_present": False},
        ],
    }
    labels_path = root / f"{name}-labels.json"
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    return video, bundle_path, labels_path, bundle


@pytest.mark.parametrize(
    "model_provenance",
    [
        {"traditional_feature_training_eligible": "false"},
        {"training_geometry": "label_hidden_window_only"},
    ],
)
def test_train_shot_validity_model_rejects_window_only_bundle_before_training_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_provenance: dict[str, str],
) -> None:
    training_game = _write_training_game(
        tmp_path,
        "closure-game",
        model_provenance=model_provenance,
    )
    benchmark_video = tmp_path / "benchmark.mp4"
    benchmark_video.write_bytes(b"blind-benchmark")
    benchmark = seal_raw_only_predictions(
        game_id="benchmark",
        raw_video_paths=[benchmark_video],
        events=[],
        config={"fixture": True},
    )
    manifest = seal_training_annotation_manifest(
        producer="codex-assisted-review",
        source_video_paths=[training_game[0]],
        annotation_paths=[training_game[2]],
        task_types=["shot_validity"],
        benchmark_bundles=[benchmark.model_dump(mode="json")],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def forbidden_training_work(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("window-only bundle reached feature extraction or fitting")

    monkeypatch.setattr(
        "scripts.train_shot_validity_model.extract_shot_validity_features",
        forbidden_training_work,
    )
    monkeypatch.setattr(
        "scripts.train_shot_validity_model._classifier",
        forbidden_training_work,
    )

    with pytest.raises(
        ValueError,
        match="candidate bundle is not eligible for traditional feature training",
    ):
        train_shot_validity_model(
            manifest_path=manifest_path,
            candidate_bundle_paths=[training_game[1]],
            annotation_paths=[training_game[2]],
            tree_count=5,
            tree_max_depth=2,
            tree_min_samples_leaf=1,
        )


def test_train_shot_validity_model_is_manifest_bound_and_group_calibrated(
    tmp_path: Path,
) -> None:
    first = _write_training_game(tmp_path, "game-a")
    second = _write_training_game(tmp_path, "game-b")
    supplemental_bundle = seal_raw_only_predictions(
        game_id="game-a-supplemental",
        raw_video_paths=[first[0]],
        events=[
            _event("game-a-supp-positive", 0.95),
            _event("game-a-supp-negative", 0.05),
        ],
        config={"fixture": "supplemental"},
    )
    supplemental_bundle_path = tmp_path / "game-a-supp-candidates.json"
    supplemental_bundle_path.write_text(
        supplemental_bundle.model_dump_json(indent=2), encoding="utf-8"
    )
    supplemental_labels = {
        "schema_version": SHOT_VALIDITY_LABEL_SCHEMA,
        "source_video_sha256": supplemental_bundle.raw_videos[0].sha256,
        "candidate_bundle_sha256": supplemental_bundle.bundle_sha256,
        "examples": [
            {"event_id": "game-a-supp-positive", "event_present": True},
            {"event_id": "game-a-supp-negative", "event_present": False},
        ],
    }
    supplemental_labels_path = tmp_path / "game-a-supp-labels.json"
    supplemental_labels_path.write_text(
        json.dumps(supplemental_labels), encoding="utf-8"
    )
    benchmark_video = tmp_path / "benchmark.mp4"
    benchmark_video.write_bytes(b"blind-benchmark")
    benchmark = seal_raw_only_predictions(
        game_id="benchmark",
        raw_video_paths=[benchmark_video],
        events=[],
        config={"fixture": True},
    )
    manifest = seal_training_annotation_manifest(
        producer="codex-assisted-review",
        source_video_paths=[first[0], second[0]],
        annotation_paths=[first[2], second[2], supplemental_labels_path],
        task_types=["shot_validity"],
        benchmark_bundles=[benchmark.model_dump(mode="json")],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    artifact = train_shot_validity_model(
        manifest_path=manifest_path,
        candidate_bundle_paths=[first[1], second[1], supplemental_bundle_path],
        annotation_paths=[first[2], second[2], supplemental_labels_path],
        tree_count=5,
        tree_max_depth=2,
        tree_min_samples_leaf=1,
        minimum_precision=0.95,
    )

    assert artifact["training_benchmark_overlap"] is False
    assert artifact["metrics"]["training_video_count"] == 2
    assert artifact["metrics"]["training_rows"] == 6
    assert artifact["metrics"]["minimum_precision_gate"] is True
    model = ShotValidityModel(artifact)
    assert model.predict(_event("new-positive", 0.9)).accepted is True
    assert model.predict(_event("new-negative", 0.1)).accepted is False


def test_threshold_calibration_prioritizes_recall_at_precision_gate() -> None:
    threshold, precision, recall, _f1 = _calibrate_threshold(
        np.asarray([1, 1, 0, 0]),
        np.asarray([0.9, 0.7, 0.6, 0.1]),
        minimum_precision=0.95,
    )

    assert threshold == 0.7
    assert precision == 1.0
    assert recall == 1.0


def test_group_calibration_requires_every_video_to_pass() -> None:
    threshold, gate = _calibrate_group_threshold(
        np.asarray([1, 0, 1, 0]),
        np.asarray([0.9, 0.1, 0.4, 0.8]),
        np.asarray(["game-a", "game-a", "game-b", "game-b"]),
        minimum_precision=0.95,
        minimum_per_video_recall=0.85,
    )

    assert threshold == 1.0
    assert gate["promoted"] is False
