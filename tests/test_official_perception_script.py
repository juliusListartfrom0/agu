from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis.schemas import BoundingBoxResponse, PerceptionDetectionResponse
from scripts.run_official_perception import (
    _filter_object_types,
    _filter_rims_by_image_geometry,
    run_perception_scan,
)


def test_official_perception_scan_rejects_invalid_sampling(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        run_perception_scan(
            video_path=tmp_path / "video.mov",
            model_path=tmp_path / "model.pt",
            sample_fps=0,
            batch_size=1,
            device="cpu",
            image_size=704,
            confidence=0.1,
            start_sec=0,
            duration_sec=1,
        )


def test_rim_geometry_filter_rejects_only_lower_frame_rims() -> None:
    import numpy as np

    detections = [
        PerceptionDetectionResponse(
            detection_id="upper",
            frame=10,
            object_type="rim",
            bbox=BoundingBoxResponse(x1=0, y1=20, x2=20, y2=30),
            confidence=0.8,
            backend="test",
        ),
        PerceptionDetectionResponse(
            detection_id="lower",
            frame=10,
            object_type="rim",
            bbox=BoundingBoxResponse(x1=0, y1=70, x2=10, y2=80),
            confidence=0.8,
            backend="test",
        ),
        PerceptionDetectionResponse(
            detection_id="ball",
            frame=10,
            object_type="basketball",
            bbox=BoundingBoxResponse(x1=0, y1=70, x2=10, y2=80),
            confidence=0.8,
            backend="test",
        ),
    ]

    filtered = _filter_rims_by_image_geometry(
        detections,
        [np.zeros((100, 100, 3), dtype=np.uint8)],
        [10],
        max_center_y_ratio=0.5,
        min_aspect_ratio=1.5,
    )

    assert [item.detection_id for item in filtered] == ["upper", "ball"]


def test_object_type_filter_supports_specialist_perception_passes() -> None:
    detections = [
        PerceptionDetectionResponse(
            detection_id=object_type,
            frame=10,
            object_type=object_type,
            bbox=BoundingBoxResponse(x1=0, y1=0, x2=10, y2=10),
            confidence=0.8,
            backend="test",
        )
        for object_type in ("player", "basketball", "rim")
    ]

    assert [
        item.object_type for item in _filter_object_types(detections, ("basketball",))
    ] == ["basketball"]
    assert _filter_object_types(detections, ()) == detections
