from __future__ import annotations

import pytest

from app.analysis.uvy_detector_training import (
    UVY_TRAINING_SCHEMA,
    build_uvy_training_plan,
    seal_uvy_training_result,
    verify_uvy_training_result,
)
from scripts.train_uvy_yolo_auxiliary import _write_absolute_data_yaml


def _manifest() -> dict[str, object]:
    return {
        "schema_version": "agu.uvy-yolo-auxiliary.v1",
        "source_manifest_sha256": "a" * 64,
        "runtime_consumable": False,
        "training_media_eligible": True,
        "training_scope": "auxiliary_detector_and_hard_negative_only",
        "causal_truth_eligible": False,
        "sequence_splits": {"basketball_V01": "train", "basketball_V02": "val", "basketball_V04": "test"},
        "data_yaml": "data.yaml",
        "manifest_sha256": "b" * 64,
    }


def test_build_uvy_training_plan_is_bounded_and_non_promotable() -> None:
    plan = build_uvy_training_plan(
        _manifest(),
        model_sha256="c" * 64,
        epochs=3,
        imgsz=320,
        batch=4,
        workers=0,
        device="cpu",
        output_dir="analysis_outputs/public_research/uvy_train",
        max_memory_percent=85.0,
        min_available_memory_gib=4.0,
    )

    assert plan["schema_version"] == UVY_TRAINING_SCHEMA
    assert plan["promotion_eligible"] is False
    assert plan["runtime_consumable"] is False
    assert plan["training_scope"] == "auxiliary_detector_and_hard_negative_only"
    assert plan["parameters"]["device"] == "cpu"
    assert plan["resource_thresholds"]["max_system_memory_percent"] == 85.0
    assert len(plan["plan_sha256"]) == 64


def test_build_uvy_training_plan_rejects_unsafe_or_missing_split() -> None:
    bad = _manifest()
    bad["sequence_splits"] = {"basketball_V01": "train", "basketball_V02": "val"}
    with pytest.raises(ValueError, match="train, val and test"):
        build_uvy_training_plan(
            bad,
            model_sha256="c" * 64,
            epochs=3,
            imgsz=320,
            batch=4,
            workers=0,
            device="cpu",
            output_dir="out",
            max_memory_percent=85.0,
            min_available_memory_gib=4.0,
        )

    with pytest.raises(ValueError, match="epochs"):
        build_uvy_training_plan(
            _manifest(),
            model_sha256="c" * 64,
            epochs=0,
            imgsz=320,
            batch=4,
            workers=0,
            device="cpu",
            output_dir="out",
            max_memory_percent=85.0,
            min_available_memory_gib=4.0,
        )


def test_uvy_training_result_seal_and_verify_fail_closed() -> None:
    result = seal_uvy_training_result(
        {
            "plan_sha256": "d" * 64,
            "source_yolo_manifest_sha256": "b" * 64,
            "model_sha256": "c" * 64,
            "metrics": {"f1": 0.0},
            "resource_guard": {"exit_code": 0},
        }
    )
    assert verify_uvy_training_result(result) == result

    promoted = dict(result)
    promoted["promotion_eligible"] = True
    with pytest.raises(ValueError, match="promotion"):
        verify_uvy_training_result(seal_uvy_training_result(promoted))

    checkpoint_promoted = dict(result)
    checkpoint_promoted["checkpoint_promoted"] = True
    with pytest.raises(ValueError, match="checkpoint"):
        verify_uvy_training_result(seal_uvy_training_result(checkpoint_promoted))


def test_training_yaml_binds_paths_to_dataset_root(tmp_path) -> None:
    dataset_root = tmp_path / "yolo"
    run_dir = tmp_path / "run"
    dataset_root.mkdir()
    run_dir.mkdir()
    for split in ("train", "val", "test"):
        (dataset_root / f"{split}.txt").write_text(
            f"images/{split}/example.jpg\n", encoding="utf-8"
        )
    yaml_path = _write_absolute_data_yaml(dataset_root, run_dir)
    text = yaml_path.read_text(encoding="utf-8")
    assert f"path: '{dataset_root}'" in text
    assert f"val: '{run_dir / 'val.absolute.txt'}'" in text
    assert (run_dir / "train.absolute.txt").is_file()
