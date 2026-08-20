from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.analysis.open_tracking_datasets import (
    build_open_tracking_catalog,
    verify_open_tracking_catalog,
)


def _write_sequence(root: Path, name: str, rows: list[str]) -> None:
    sequence = root / "basketball_side" / "train" / name
    annotation = sequence / "gt" / "gt.txt"
    annotation.parent.mkdir(parents=True)
    annotation.write_text("\n".join(rows) + "\n", encoding="utf-8")
    (sequence / "img1.mp4").write_bytes(b"fixture-media")


def test_open_tracking_catalog_seals_mot_annotations_and_media(tmp_path: Path) -> None:
    _write_sequence(
        tmp_path,
        "sequence-a",
        [
            "1,7,10,20,30,40,1,1,1",
            "2,7,11,21,30,40,1,1,0.8",
            "2,8,100,120,20,30,1,1,1",
        ],
    )

    catalog = build_open_tracking_catalog(
        tmp_path,
        dataset_id="teamtrack",
        source_revision="revision-1",
    )
    verified = verify_open_tracking_catalog(json.loads(json.dumps(catalog)))

    assert verified["runtime_consumable"] is False
    assert verified["acceptance_media_allowed"] is False
    assert verified["annotation_count"] == 3
    assert verified["track_count"] == 2
    assert verified["sequences"][0]["frame_count"] == 2
    assert verified["sequences"][0]["coordinate_extent"] == [10.0, 20.0, 120.0, 150.0]
    assert verified["sequences"][0]["media_sha256"]


def test_open_tracking_catalog_rejects_bad_boxes_and_tampering(tmp_path: Path) -> None:
    _write_sequence(tmp_path, "bad", ["1,7,10,20,-30,40,1,1,1"])
    with pytest.raises(ValueError, match="bounding box"):
        build_open_tracking_catalog(
            tmp_path,
            dataset_id="trackid3x3",
            source_revision="revision-1",
        )

    clean_root = tmp_path / "clean"
    _write_sequence(clean_root, "good", ["1,7,10,20,30,40,1,1,1"])
    catalog = build_open_tracking_catalog(
        clean_root,
        dataset_id="trackid3x3",
        source_revision="revision-1",
    )
    catalog["runtime_consumable"] = True
    with pytest.raises(ValueError, match="runtime-consumable"):
        verify_open_tracking_catalog(catalog)


def test_trackid3x3_catalog_imports_official_named_mot_files_and_video_layout(
    tmp_path: Path,
) -> None:
    annotation = (
        tmp_path
        / "ground_truth"
        / "Indoor"
        / "MOT"
        / "basket_S1T1_pre.txt"
    )
    annotation.parent.mkdir(parents=True)
    annotation.write_text(
        "1,1,10,20,30,40,1,1,1\n"
        "1,2,50,60,30,40,1,1,1\n"
        "2,1,11,21,30,40,1,1,1\n",
        encoding="utf-8",
    )
    media = tmp_path / "Indoor" / "raw" / "basket_S1T1_pre.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"fixture-media")

    catalog = build_open_tracking_catalog(
        tmp_path,
        dataset_id="trackid3x3",
        source_revision="revision-1",
    )

    assert catalog["sequence_count"] == 1
    assert catalog["annotation_count"] == 3
    assert catalog["sequences"][0]["sequence"] == "basket_S1T1_pre"
    assert catalog["sequences"][0]["annotation_path"] == (
        "ground_truth/Indoor/MOT/basket_S1T1_pre.txt"
    )
    assert catalog["sequences"][0]["media_path"] == (
        "Indoor/raw/basket_S1T1_pre.mp4"
    )
    assert catalog["sequences"][0]["media_sha256"]

    media.unlink()
    with pytest.raises(ValueError, match="requires its matching video"):
        build_open_tracking_catalog(
            tmp_path,
            dataset_id="trackid3x3",
            source_revision="revision-1",
        )
