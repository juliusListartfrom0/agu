"""Leakage-safe utilities for screening basketball candidate verifiers."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

BALL_TRACK_FEATURE_NAMES = (
    "detection_confidence",
    "bbox_log_area",
    "bbox_aspect_ratio",
    "candidate_count_same_frame",
    "track_found",
    "track_visible_points",
    "track_total_points",
    "track_span_frames",
    "track_mean_confidence",
    "track_max_confidence",
    "track_mean_speed_px_per_frame",
    "track_max_speed_px_per_frame",
    "track_speed_std_px_per_frame",
    "track_displacement_px_per_frame",
)


def _point_key(frame: int, x: float, y: float) -> tuple[int, float, float]:
    return frame, round(x, 5), round(y, 5)


def ball_track_features(
    perception: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    """Derive detector/trajectory features without decoding source pixels."""
    detections = list(perception.get("detections") or [])
    frame_counts = Counter(int(row["frame"]) for row in detections)
    track_by_point: dict[tuple[int, float, float], np.ndarray] = {}
    for track in perception.get("ball_tracks") or []:
        points = list(track.get("points") or [])
        visible = [
            point
            for point in points
            if bool(point.get("visible", True))
            and not bool(point.get("predicted", False))
        ]
        if not visible:
            continue
        speeds = [
            math.hypot(
                float(right["center"]["x"]) - float(left["center"]["x"]),
                float(right["center"]["y"]) - float(left["center"]["y"]),
            )
            / max(1, int(right["frame"]) - int(left["frame"]))
            for left, right in zip(visible, visible[1:])
        ]
        first, last = visible[0], visible[-1]
        span = max(0, int(last["frame"]) - int(first["frame"]))
        displacement = math.hypot(
            float(last["center"]["x"]) - float(first["center"]["x"]),
            float(last["center"]["y"]) - float(first["center"]["y"]),
        ) / max(1, span)
        confidences = [float(point["confidence"]) for point in visible]
        track_values = np.asarray(
            [
                1.0,
                float(len(visible)),
                float(len(points)),
                float(span),
                float(np.mean(confidences)),
                float(max(confidences)),
                float(np.mean(speeds)) if speeds else 0.0,
                float(max(speeds)) if speeds else 0.0,
                float(np.std(speeds)) if speeds else 0.0,
                float(displacement),
            ],
            dtype=np.float64,
        )
        for point in visible:
            key = _point_key(
                int(point["frame"]),
                float(point["center"]["x"]),
                float(point["center"]["y"]),
            )
            if key in track_by_point:
                raise ValueError("a visible ball point belongs to multiple tracks")
            track_by_point[key] = track_values

    output: dict[str, np.ndarray] = {}
    for row in detections:
        bbox = row["bbox"]
        width = max(1e-6, float(bbox["x2"]) - float(bbox["x1"]))
        height = max(1e-6, float(bbox["y2"]) - float(bbox["y1"]))
        frame = int(row["frame"])
        key = _point_key(
            frame,
            (float(bbox["x1"]) + float(bbox["x2"])) / 2.0,
            (float(bbox["y1"]) + float(bbox["y2"])) / 2.0,
        )
        track_values = track_by_point.get(key)
        if track_values is None:
            track_values = np.zeros(10, dtype=np.float64)
        output[str(row["detection_id"])] = np.concatenate(
            [
                np.asarray(
                    [
                        float(row["confidence"]),
                        math.log(width * height),
                        width / height,
                        float(frame_counts[frame]),
                    ],
                    dtype=np.float64,
                ),
                track_values,
            ]
        )
    return BALL_TRACK_FEATURE_NAMES, output


def choose_recall_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    recall_floor: float,
) -> dict[str, float]:
    """Choose the highest-precision observed threshold meeting a recall floor."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or labels.size == 0:
        raise ValueError("labels and scores must be non-empty aligned vectors")
    if not 0.0 < recall_floor <= 1.0 or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("invalid labels or recall floor")

    positives = int(labels.sum())
    if positives == 0:
        raise ValueError("threshold selection requires positive examples")
    best: tuple[float, float, float] | None = None
    for threshold in sorted(set(float(value) for value in scores), reverse=True):
        kept = scores >= threshold
        tp = int(np.logical_and(kept, labels == 1).sum())
        fp = int(np.logical_and(kept, labels == 0).sum())
        recall = tp / positives
        if recall < recall_floor:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        candidate = (precision, threshold, recall)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise ValueError("no threshold satisfies recall floor")
    precision, threshold, recall = best
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
    }


def stratified_review_metrics(
    reviewed: Sequence[Mapping[str, object]],
    *,
    band_population: Mapping[str, int],
) -> dict[str, float]:
    """Estimate population precision/recall from equal-size band review rows."""
    weighted_tp = weighted_fp = weighted_fn = 0.0
    for band, population in band_population.items():
        rows = [row for row in reviewed if str(row["band"]) == band]
        if not rows or population < 0:
            raise ValueError("every population band requires reviewed rows")
        weight = float(population) / len(rows)
        for row in rows:
            label = int(row["label"])
            kept = bool(row["kept"])
            if label not in (0, 1):
                raise ValueError("review labels must be binary")
            weighted_tp += weight * float(label == 1 and kept)
            weighted_fp += weight * float(label == 0 and kept)
            weighted_fn += weight * float(label == 1 and not kept)
    precision = weighted_tp / (weighted_tp + weighted_fp) if weighted_tp + weighted_fp else 0.0
    recall = weighted_tp / (weighted_tp + weighted_fn) if weighted_tp + weighted_fn else 0.0
    return {
        "weighted_tp": weighted_tp,
        "weighted_fp": weighted_fp,
        "weighted_fn": weighted_fn,
        "precision": precision,
        "recall": recall,
    }
