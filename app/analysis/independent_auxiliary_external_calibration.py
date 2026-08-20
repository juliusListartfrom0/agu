"""Game-held diagnostic calibration for an uncalibrated auxiliary detector score.

The transfer detector emits a bounded score, not a probability.  This module
can select a descriptive threshold from independent labeled game screens and
evaluate a separately held target only after inference.  It never emits OOF
rows, never reads target labels during threshold selection, and is explicitly
research-only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from app.analysis.independent_shot_auxiliary_transfer import summarize_event_scores


def build_screen_score_rows(screen_artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Derive transfer scores from a labeled detector screen after inference."""

    source = str(screen_artifact.get("video_sha256") or "")
    if len(source) != 64:
        raise ValueError("screen artifact video_sha256 is invalid")
    raw_rows = screen_artifact.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("screen artifact rows are required")
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("screen row must be a mapping")
        event_id = str(raw.get("event_id") or "")
        if not event_id or (source, event_id) in seen:
            raise ValueError("screen rows contain duplicate or empty event IDs")
        label = raw.get("event_present")
        if not isinstance(label, bool):
            raise ValueError("screen rows require boolean event_present labels")
        sampled_frames = raw.get("sampled_frames")
        if not isinstance(sampled_frames, list) or not sampled_frames:
            raise ValueError("screen row sampled_frames are required")
        detections_by_frame: dict[int, list[dict[str, Any]]] = {
            int(frame): [] for frame in sampled_frames
        }
        detections = raw.get("detections")
        if not isinstance(detections, list):
            raise ValueError("screen row detections are required")
        for detection in detections:
            if not isinstance(detection, Mapping):
                raise ValueError("screen detection must be a mapping")
            frame = int(detection.get("frame", -1))
            if frame not in detections_by_frame:
                raise ValueError("screen detection frame is not sampled")
            detections_by_frame[frame].append({"score": detection.get("score")})
        summary = summarize_event_scores(
            detections_by_frame=[
                {"frame": frame, "detections": detections_by_frame[frame]}
                for frame in sampled_frames
            ],
            start_frame=int(raw.get("start_frame", -1)),
            end_frame=int(raw.get("end_frame", -1)),
        )
        output.append(
            {
                "source_video_sha256": source,
                "event_id": event_id,
                "score": float(summary["transfer_score"]),
                "event_present": label,
                "sampled_frame_count": int(summary["sampled_frame_count"]),
                "positive_frame_count": int(summary["positive_frame_count"]),
            }
        )
        seen.add((source, event_id))
    return output


def select_external_score_threshold(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_recall: float = 0.85,
) -> dict[str, Any]:
    """Select a threshold from calibration games only.

    This is intentionally separate from the runtime evidence gate: the field
    is ``score`` rather than ``probability`` and the result is diagnostic.
    """

    if not 0.0 < float(minimum_recall) <= 1.0:
        raise ValueError("minimum_recall must be within (0, 1]")
    normalized = [_normalize_score_row(row) for row in rows]
    if not normalized:
        raise ValueError("calibration rows are required")
    groups = sorted({row["source_video_sha256"] for row in normalized})
    thresholds = sorted({0.0, 1.0, *(row["score"] for row in normalized)})
    candidates = [_metrics(normalized, threshold) for threshold in thresholds]
    eligible = [row for row in candidates if row["recall"] >= minimum_recall]
    chosen = max(
        eligible or candidates,
        key=lambda row: (row["precision"], row["recall"], row["f1"], row["threshold"]),
    )
    return {
        "training_groups": groups,
        "training_count": len(normalized),
        "minimum_recall": float(minimum_recall),
        "selected_from_target_labels": False,
        **chosen,
    }


def evaluate_labeled_scores(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate scores against labels supplied after inference."""

    normalized = [_normalize_score_row(row) for row in rows]
    if not normalized:
        raise ValueError("labeled score rows are required")
    if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be within [0, 1]")
    return {
        "threshold": float(threshold),
        "evaluated_count": len(normalized),
        **_metrics(normalized, float(threshold)),
        "labels_used_after_inference": True,
    }


def _normalize_score_row(row: Mapping[str, Any]) -> dict[str, Any]:
    source = str(row.get("source_video_sha256") or "")
    event_id = str(row.get("event_id") or "")
    if len(source) != 64 or not event_id:
        raise ValueError("score row key is invalid")
    try:
        score = float(row.get("score"))
    except (TypeError, ValueError) as exc:
        raise ValueError("score row score is invalid") from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("score row score must be within [0, 1]")
    label = row.get("event_present")
    if not isinstance(label, bool):
        raise ValueError("score rows require boolean labels")
    return {
        "source_video_sha256": source,
        "event_id": event_id,
        "score": score,
        "event_present": label,
    }


def _metrics(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for row in rows:
        predicted = float(row["score"]) >= threshold
        label = bool(row["event_present"])
        if predicted and label:
            tp += 1
        elif predicted:
            fp += 1
        elif label:
            fn += 1
        else:
            tn += 1
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {
        "threshold": float(threshold),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
    }


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
