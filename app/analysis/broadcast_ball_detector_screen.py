"""Metrics for the offline full-frame broadcast ball-detector screen."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = (float(value) for value in left)
    rx1, ry1, rx2, ry2 = (float(value) for value in right)
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _target_box(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    bbox = row["bbox"]
    return tuple(float(bbox[key]) for key in ("x1", "y1", "x2", "y2"))  # type: ignore[return-value]


def detector_frame_metrics(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[Sequence[float]]],
    *,
    iou_threshold: float,
) -> dict[str, Any]:
    """Compute a conservative frame/object lower bound for reviewed rows.

    ``false_positive`` rows count every model box as a false positive because the
    reviewed candidate is known to be invalid.  On positive rows, one matching box
    is a true positive and all other boxes are false positives.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    if len(rows) != len(predictions):
        raise ValueError("prediction rows do not align with reviewed rows")
    tp = fp = fn = 0
    valid_frames = false_positive_frames = 0
    for row, boxes in zip(rows, predictions):
        decision = str(row["decision"])
        if decision == "valid_ball":
            valid_frames += 1
            target = _target_box(row)
            matched = any(box_iou(target, box) >= iou_threshold for box in boxes)
            if matched:
                tp += 1
                fp += max(0, len(boxes) - 1)
            else:
                fn += 1
                fp += len(boxes)
        elif decision == "false_positive":
            false_positive_frames += 1
            fp += len(boxes)
        else:
            raise ValueError("metrics only accept determinate reviewed rows")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "valid_frames": valid_frames,
        "false_positive_frames": false_positive_frames,
    }
