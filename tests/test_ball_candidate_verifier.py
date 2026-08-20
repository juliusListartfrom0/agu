from __future__ import annotations

import numpy as np

from app.analysis.ball_candidate_verifier import (
    ball_track_features,
    choose_recall_threshold,
    stratified_review_metrics,
)


def test_choose_recall_threshold_maximizes_precision_at_recall_floor() -> None:
    labels = np.asarray([1, 1, 1, 0, 0, 0], dtype=np.int64)
    scores = np.asarray([0.9, 0.8, 0.4, 0.7, 0.3, 0.1], dtype=np.float64)

    result = choose_recall_threshold(labels, scores, recall_floor=2 / 3)

    assert result["threshold"] == 0.8
    assert result["precision"] == 1.0
    assert result["recall"] == 2 / 3


def test_stratified_review_metrics_weights_each_band_by_population() -> None:
    reviewed = [
        {"band": "low", "label": 1, "kept": True},
        {"band": "low", "label": 0, "kept": True},
        {"band": "high", "label": 1, "kept": True},
        {"band": "high", "label": 1, "kept": False},
    ]

    result = stratified_review_metrics(
        reviewed,
        band_population={"low": 90, "high": 10},
    )

    assert result["weighted_tp"] == 50.0
    assert result["weighted_fp"] == 45.0
    assert result["weighted_fn"] == 5.0
    assert result["precision"] == 50.0 / 95.0
    assert result["recall"] == 50.0 / 55.0


def test_ball_track_features_bind_visible_points_to_detections() -> None:
    perception = {
        "detections": [
            {
                "detection_id": "d:1",
                "frame": 10,
                "bbox": {"x1": 0.0, "y1": 0.0, "x2": 4.0, "y2": 6.0},
                "confidence": 0.4,
            },
            {
                "detection_id": "d:2",
                "frame": 12,
                "bbox": {"x1": 3.0, "y1": 0.0, "x2": 7.0, "y2": 6.0},
                "confidence": 0.8,
            },
        ],
        "ball_tracks": [
            {
                "points": [
                    {
                        "frame": 10,
                        "center": {"x": 2.0, "y": 3.0},
                        "confidence": 0.4,
                        "visible": True,
                        "predicted": False,
                    },
                    {
                        "frame": 11,
                        "center": {"x": 3.5, "y": 3.0},
                        "confidence": 0.28,
                        "visible": False,
                        "predicted": True,
                    },
                    {
                        "frame": 12,
                        "center": {"x": 5.0, "y": 3.0},
                        "confidence": 0.8,
                        "visible": True,
                        "predicted": False,
                    },
                ]
            }
        ],
    }

    names, rows = ball_track_features(perception)

    assert set(rows) == {"d:1", "d:2"}
    assert rows["d:1"].shape == (len(names),)
    assert rows["d:1"][names.index("track_visible_points")] == 2.0
    assert rows["d:1"][names.index("track_span_frames")] == 2.0
    assert rows["d:1"][names.index("track_mean_speed_px_per_frame")] == 1.5
    assert rows["d:1"][names.index("candidate_count_same_frame")] == 1.0
