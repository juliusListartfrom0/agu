from __future__ import annotations

import pytest

from scripts.materialize_apidis_ball_yolo import _label_rows_by_frame, _yolo_line


def test_label_rows_use_median_for_nearest_frame_duplicates() -> None:
    rows = _label_rows_by_frame(
        [
            {"frame": 10, "x": 20.0, "y": 30.0},
            {"frame": 10, "x": 24.0, "y": 34.0},
            {"frame": 20, "x": 5.0, "y": 6.0},
        ]
    )
    assert rows == {10: (22.0, 32.0), 20: (5.0, 6.0)}


def test_yolo_line_normalizes_fixed_ball_box() -> None:
    assert _yolo_line(
        (400.0, 300.0),
        width=800,
        height=600,
        box_width_px=13.5,
        box_height_px=15.5,
    ) == "0 0.5000000000 0.5000000000 0.0168750000 0.0258333333\n"


def test_yolo_line_rejects_out_of_frame_center() -> None:
    with pytest.raises(ValueError, match="outside"):
        _yolo_line(
            (801.0, 300.0),
            width=800,
            height=600,
            box_width_px=13.5,
            box_height_px=15.5,
        )
