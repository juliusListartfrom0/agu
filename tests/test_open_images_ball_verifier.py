from __future__ import annotations

from scripts.screen_open_images_ball_verifier import _iou, _pixel_box


def test_iou_is_zero_for_disjoint_boxes_and_positive_for_overlap() -> None:
    assert _iou((0, 0, 10, 10), (10, 10, 20, 20)) == 0.0
    assert _iou((0, 0, 10, 10), (5, 5, 15, 15)) == 25 / 175


def test_pixel_box_clamps_normalized_coordinates_to_image() -> None:
    assert _pixel_box(
        {"xmin": -0.1, "ymin": 0.2, "xmax": 1.2, "ymax": 0.8},
        100,
        50,
    ) == (0, 10, 100, 40)
