from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.analysis.perception.tiled_ball import horizontal_tile_specs
from scripts.materialize_tiled_ball_yolo import (
    YoloLabel,
    materialize_tiled_dataset,
    parse_yolo_labels,
    project_labels_to_tile,
)


def test_parse_yolo_labels_rejects_malformed_or_out_of_range_rows() -> None:
    assert parse_yolo_labels(["0 0.5 0.5 0.2 0.4\n"]) == (
        YoloLabel(class_id=0, center_x=0.5, center_y=0.5, width=0.2, height=0.4),
    )

    with pytest.raises(ValueError, match="five fields"):
        parse_yolo_labels(["0 0.5 0.5 0.2\n"])
    with pytest.raises(ValueError, match="normalized"):
        parse_yolo_labels(["0 1.1 0.5 0.2 0.4\n"])


def test_project_labels_to_tile_clips_intersections_and_drops_disjoint_boxes() -> None:
    labels = (
        YoloLabel(class_id=0, center_x=0.62, center_y=0.5, width=0.10, height=0.20),
        YoloLabel(class_id=0, center_x=0.05, center_y=0.5, width=0.04, height=0.10),
    )
    tile = horizontal_tile_specs(640, overlap_ratio=0.24)[1]

    projected = project_labels_to_tile(labels, frame_width=640, frame_height=360, tile=tile)

    assert len(projected) == 1
    assert projected[0].class_id == 0
    assert 0.0 < projected[0].center_x < 1.0
    assert projected[0].width > 0.0
    assert projected[0].height == pytest.approx(0.20)


def test_materialize_tiled_dataset_writes_hash_bound_train_and_val_tiles(tmp_path) -> None:
    source = tmp_path / "source"
    for split in ("train", "val"):
        (source / "images" / split).mkdir(parents=True)
        (source / "labels" / split).mkdir(parents=True)
        image = np.zeros((20, 32, 3), dtype=np.uint8)
        image[:, :16] = (10, 20, 30)
        image[:, 16:] = (40, 50, 60)
        image_path = source / "images" / split / "game-001.jpg"
        assert cv2.imwrite(str(image_path), image)
        (source / "labels" / split / "game-001.txt").write_text(
            "0 0.75 0.5 0.2 0.5\n", encoding="utf-8"
        )

    output = tmp_path / "tiles"
    manifest = materialize_tiled_dataset(source, output, jpeg_quality=90)

    assert manifest["split_counts"] == {"train": 2, "val": 2}
    assert manifest["tile_count"] == 4
    assert manifest["runtime_consumable"] is False
    assert (output / "data.yaml").is_file()
    assert len(list((output / "images" / "train").glob("*.jpg"))) == 2
    assert len(list((output / "labels" / "val").glob("*.txt"))) == 2
    assert manifest["artifact_sha256"]
