from __future__ import annotations

import csv
from pathlib import Path

import pytest

from app.analysis.infactory_ball_tracking import (
    build_infactory_ball_tracking_manifest,
    match_ball_boxes,
    verify_infactory_ball_tracking_manifest,
)


def _write_fixture(root: Path) -> None:
    rows: list[dict[str, str | int]] = []
    for clip_index in range(5):
        clip = f"clip-{clip_index}"
        visible = f"data/visible/{clip}_000000.jpg"
        negative = f"data/not_visible/{clip}_000001.jpg"
        (root / visible).parent.mkdir(parents=True, exist_ok=True)
        (root / negative).parent.mkdir(parents=True, exist_ok=True)
        (root / visible).write_bytes(f"visible-{clip}".encode())
        (root / negative).write_bytes(f"negative-{clip}".encode())
        (root / visible).with_suffix(".txt").write_text(
            "0 0.5 0.5 0.02 0.03\n", encoding="utf-8"
        )
        rows.extend(
            [
                {
                    "file_path": visible,
                    "video_source": clip,
                    "frame_index": "000000",
                    "visibility": "visible",
                    "bboxes_count": 1,
                },
                {
                    "file_path": negative,
                    "video_source": clip,
                    "frame_index": "000001",
                    "visibility": "not_visible",
                    "bboxes_count": 0,
                },
            ]
        )
    with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file_path",
                "video_source",
                "frame_index",
                "visibility",
                "bboxes_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_infactory_manifest_is_clip_disjoint_and_hash_bound(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    manifest = build_infactory_ball_tracking_manifest(
        tmp_path,
        source_revision="a" * 40,
        validation_clip_count=1,
        test_clip_count=1,
        split_seed="fixture-v1",
    )

    assert manifest["runtime_consumable"] is False
    assert manifest["codex_runtime_answer_used"] is False
    assert manifest["license"] == "CC-BY-NC-4.0"
    assert manifest["row_counts"] == {"visible": 5, "not_visible": 5}
    assert manifest["split_counts"] == {"train": 6, "val": 2, "test": 2}
    split_clips = manifest["split_clip_ids"]
    assert set(split_clips) == {"train", "val", "test"}
    assert not (set(split_clips["train"]) & set(split_clips["val"]))
    assert not (set(split_clips["train"]) & set(split_clips["test"]))
    assert not (set(split_clips["val"]) & set(split_clips["test"]))
    assert verify_infactory_ball_tracking_manifest(manifest) == manifest


def test_infactory_manifest_rejects_mismatched_negative_labels(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    negative = next((tmp_path / "data/not_visible").glob("*.jpg"))
    negative.with_suffix(".txt").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="not_visible frame must not have a label"):
        build_infactory_ball_tracking_manifest(
            tmp_path,
            source_revision="b" * 40,
            validation_clip_count=1,
            test_clip_count=1,
            split_seed="fixture-v1",
        )


def test_match_ball_boxes_is_one_to_one_and_counts_unmatched_predictions() -> None:
    assert match_ball_boxes(
        [(0, 0, 10, 10), (20, 20, 30, 30)],
        [(1, 1, 9, 9), (20, 20, 30, 30), (40, 40, 50, 50)],
        iou_threshold=0.25,
    ) == {"tp": 2, "fp": 1, "fn": 0}
