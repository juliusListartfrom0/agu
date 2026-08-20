from __future__ import annotations

import pytest

from app.analysis.broadcast_ball_detector_screen import (
    box_iou,
    detector_frame_metrics,
)


def test_box_iou_handles_overlap_and_disjoint_boxes() -> None:
    assert box_iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)
    assert box_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0


def test_detector_frame_metrics_counts_conservative_false_positive_frames() -> None:
    rows = [
        {"decision": "valid_ball", "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
        {"decision": "valid_ball", "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
        {"decision": "false_positive", "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
    ]
    metrics = detector_frame_metrics(
        rows,
        [
            [(1, 1, 9, 9)],
            [],
            [(2, 2, 4, 4), (20, 20, 30, 30)],
        ],
        iou_threshold=0.25,
    )
    assert metrics == {
        "tp": 1,
        "fp": 2,
        "fn": 1,
        "precision": pytest.approx(1 / 3),
        "recall": pytest.approx(0.5),
        "f1": pytest.approx(0.4),
        "valid_frames": 2,
        "false_positive_frames": 1,
    }
