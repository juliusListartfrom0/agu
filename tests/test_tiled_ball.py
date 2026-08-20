from __future__ import annotations

import pytest

from app.analysis.perception.tiled_ball import (
    deduplicate_ball_detections,
    horizontal_tile_specs,
    project_tile_box,
)


def test_horizontal_tile_specs_match_two_tile_screen_protocol() -> None:
    tiles = horizontal_tile_specs(640, overlap_ratio=0.24)

    assert [(tile.x1, tile.x2) for tile in tiles] == [(0, 397), (243, 640)]
    assert tiles[0].frame_width == 640
    assert tiles[0].width == 397


def test_project_tile_box_adds_offset_and_clips_to_frame() -> None:
    tile = horizontal_tile_specs(640, overlap_ratio=0.24)[1]

    assert project_tile_box((-4, 2, 30, 18), tile) == (243.0, 2.0, 273.0, 18.0)


def test_deduplicate_ball_detections_keeps_highest_confidence_per_frame_cluster() -> None:
    detections = [
        {
            "detection_id": "right",
            "frame": 10,
            "bbox": {"x1": 245.0, "y1": 20.0, "x2": 255.0, "y2": 30.0},
            "confidence": 0.7,
        },
        {
            "detection_id": "left",
            "frame": 10,
            "bbox": {"x1": 246.0, "y1": 21.0, "x2": 256.0, "y2": 31.0},
            "confidence": 0.8,
        },
        {
            "detection_id": "next-frame",
            "frame": 11,
            "bbox": {"x1": 246.0, "y1": 21.0, "x2": 256.0, "y2": 31.0},
            "confidence": 0.4,
        },
    ]

    assert [item["detection_id"] for item in deduplicate_ball_detections(detections)] == [
        "left",
        "next-frame",
    ]


def test_tiled_ball_helpers_reject_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="frame width"):
        horizontal_tile_specs(0)
    with pytest.raises(ValueError, match="overlap ratio"):
        horizontal_tile_specs(640, overlap_ratio=1.0)
