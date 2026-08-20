from __future__ import annotations

import numpy as np

from scripts.run_tiled_ball_perception import _tile_predictions


class _Tensor:
    def __init__(self, value: object) -> None:
        self.value = value

    def cpu(self) -> _Tensor:
        return self

    def tolist(self) -> object:
        return self.value


class _Boxes:
    xyxy = _Tensor([[10.0, 5.0, 20.0, 15.0]])
    conf = _Tensor([0.7])


class _Result:
    boxes = _Boxes()


class _Model:
    def predict(self, *, source: list[np.ndarray], **_: object) -> list[_Result]:
        return [_Result() for _ in source]


def test_tile_predictions_projects_boxes_and_deduplicates_overlap() -> None:
    frame = np.zeros((40, 640, 3), dtype=np.uint8)
    detections = _tile_predictions(
        _Model(),
        [frame],
        [15],
        image_size=960,
        confidence=0.05,
        overlap_ratio=0.24,
        device="cpu",
        dedup_center_distance=18.0,
    )

    assert len(detections) == 2
    assert [round(item["bbox"]["x1"], 1) for item in detections] == [10.0, 253.0]
