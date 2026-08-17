from __future__ import annotations

import pytest

from app.analysis.uvy_detector_screen import (
    ball_detection_metrics,
    seal_uvy_detector_screen,
    verify_uvy_detector_screen,
)


def test_uvy_ball_metrics_use_one_to_one_matching() -> None:
    metrics = ball_detection_metrics(
        [[[0, 0, 10, 10], [20, 20, 30, 30]], []],
        [[[0, 0, 10, 10], [0, 0, 10, 10], [40, 40, 50, 50]], [[1, 1, 2, 2]]],
        iou_threshold=0.5,
    )
    assert metrics["tp"] == 1
    assert metrics["fp"] == 3
    assert metrics["fn"] == 1


def test_uvy_screen_is_sealed_non_promotable() -> None:
    artifact = seal_uvy_detector_screen(
        {
            "purpose": "test",
            "metrics": {"precision": 1.0},
            "promotion_eligible": False,
        }
    )
    assert verify_uvy_detector_screen(artifact)["artifact_sha256"] == artifact["artifact_sha256"]
    bad = dict(artifact)
    bad["promotion_eligible"] = True
    with pytest.raises(ValueError, match="cannot promote"):
        verify_uvy_detector_screen(bad)
