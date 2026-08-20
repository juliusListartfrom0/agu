"""Metrics and contracts for a held-sequence UVY ball-detector screen."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

UVY_DETECTOR_SCREEN_SCHEMA = "agu.uvy-detector-screen.v1"


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


def ball_detection_metrics(
    targets: Sequence[Sequence[Sequence[float]]],
    predictions: Sequence[Sequence[Sequence[float]]],
    *,
    iou_threshold: float,
) -> dict[str, Any]:
    """Compute one-to-one ball-box precision/recall over sampled frames."""

    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    if len(targets) != len(predictions):
        raise ValueError("target/prediction frame counts differ")
    tp = fp = fn = 0
    frames_with_target = frames_with_prediction = 0
    for target_boxes, predicted_boxes in zip(targets, predictions):
        if target_boxes:
            frames_with_target += 1
        if predicted_boxes:
            frames_with_prediction += 1
        remaining = set(range(len(predicted_boxes)))
        for target in target_boxes:
            best_index = None
            best_iou = 0.0
            for index in remaining:
                overlap = box_iou(target, predicted_boxes[index])
                if overlap >= iou_threshold and overlap > best_iou:
                    best_index, best_iou = index, overlap
            if best_index is None:
                fn += 1
            else:
                remaining.remove(best_index)
                tp += 1
        fp += len(remaining)
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
        "frames": len(targets),
        "frames_with_target": frames_with_target,
        "frames_with_prediction": frames_with_prediction,
    }


def seal_uvy_detector_screen(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    artifact.setdefault("schema_version", UVY_DETECTOR_SCREEN_SCHEMA)
    artifact.setdefault("runtime_consumable", False)
    artifact.setdefault("codex_runtime_answer_used", False)
    artifact.setdefault("promotion_eligible", False)
    artifact.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_uvy_detector_screen(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    if artifact.get("schema_version") != UVY_DETECTOR_SCREEN_SCHEMA:
        raise ValueError("invalid UVY detector screen schema")
    if artifact.get("runtime_consumable") is not False:
        raise ValueError("UVY detector screen cannot be runtime consumable")
    if artifact.get("promotion_eligible") is not False:
        raise ValueError("UVY detector screen cannot promote a model")
    if not claimed or claimed != _canonical_sha256(artifact):
        raise ValueError("UVY detector screen hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
