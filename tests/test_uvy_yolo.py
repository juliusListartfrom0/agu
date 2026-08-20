from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis.uvy_yolo import (
    UVY_YOLO_SCHEMA,
    assign_uvy_sequence_splits,
    materialize_uvy_yolo_subset,
    verify_uvy_yolo_manifest,
)


def _write_sequence(root: Path, name: str, *, frame: int = 1) -> None:
    sequence = root / "UVY" / name
    (sequence / "img1").mkdir(parents=True)
    (sequence / "gt").mkdir()
    (sequence / "video_info.txt").write_text(
        "imgDir=img1\nimWidth=100\nimHeight=100\nnumFrames=1\n", encoding="utf-8"
    )
    (sequence / "gt" / "labels.txt").write_text(
        "player\nsports ball\ngoal\nreferee\n", encoding="utf-8"
    )
    (sequence / "gt" / "gt.txt").write_text(
        f"{frame},1,10,20,20,10,1,1,1\n{frame},2,40,50,5,5,1,2,1\n",
        encoding="utf-8",
    )
    (sequence / "img1" / f"{frame:06d}.jpg").write_bytes(b"image")


def test_uvy_sequence_split_is_complete_and_deterministic() -> None:
    assert assign_uvy_sequence_splits(
        ["basketball_V04", "basketball_V01", "basketball_V03"],
        validation_sequence="basketball_V03",
        test_sequence="basketball_V04",
    ) == {
        "basketball_V01": "train",
        "basketball_V03": "val",
        "basketball_V04": "test",
    }
    with pytest.raises(ValueError, match="distinct"):
        assign_uvy_sequence_splits(
            ["basketball_V01"],
            validation_sequence="basketball_V01",
            test_sequence="basketball_V01",
        )


def test_uvy_yolo_materializer_keeps_media_auxiliary_only(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for name in ("basketball_V01", "basketball_V03", "basketball_V04"):
        _write_sequence(source, name)
    output = tmp_path / "yolo"
    manifest = materialize_uvy_yolo_subset(
        source,
        output,
        source_manifest_sha256="a" * 64,
        sequence_splits={
            "basketball_V01": "train",
            "basketball_V03": "val",
            "basketball_V04": "test",
        },
    )
    assert manifest["schema_version"] == UVY_YOLO_SCHEMA
    assert manifest["split_counts"] == {"test": 1, "train": 1, "val": 1}
    assert manifest["split_box_counts"] == {"test": 2, "train": 2, "val": 2}
    assert manifest["training_scope"] == "auxiliary_detector_and_hard_negative_only"
    assert manifest["runtime_consumable"] is False
    assert verify_uvy_yolo_manifest(manifest)["manifest_sha256"] == manifest["manifest_sha256"]
    label = (output / "labels" / "train" / "basketball_V01_000001.txt").read_text()
    assert label.splitlines()[0].startswith("0 ")
    assert (output / "train.txt").read_text().strip() == "images/train/basketball_V01_000001.jpg"


def test_uvy_yolo_materializer_rejects_bad_manifest_hash(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for name in ("basketball_V01", "basketball_V03", "basketball_V04"):
        _write_sequence(source, name)
    manifest = materialize_uvy_yolo_subset(
        source,
        tmp_path / "yolo",
        source_manifest_sha256="b" * 64,
        sequence_splits={
            "basketball_V01": "train",
            "basketball_V03": "val",
            "basketball_V04": "test",
        },
    )
    manifest["split_counts"]["train"] = 99
    with pytest.raises(ValueError, match="hash"):
        verify_uvy_yolo_manifest(manifest)
