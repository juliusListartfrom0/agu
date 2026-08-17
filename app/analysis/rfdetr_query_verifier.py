"""Utilities for binding exported RF-DETR detections to decoder queries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_TRACK_BASE_FEATURE_NAMES = (
    "track_found",
    "track_query_count",
    "track_span_frames",
    "context_current_to_mean_cosine",
    "context_adjacent_cosine_mean",
    "context_adjacent_cosine_min",
    "context_adjacent_l2_mean",
    "context_adjacent_l2_max",
)


def _as_single_image_array(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(f"{name} must contain exactly one image")
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [queries, features]")
    return array


def _object_index(detection_id: str) -> int:
    try:
        value = int(detection_id.rsplit(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"detection_id does not end in an object index: {detection_id}"
        ) from exc
    if value < 0:
        raise ValueError(f"detection object index must be non-negative: {detection_id}")
    return value


def _pixel_xyxy(box: np.ndarray, *, width: int, height: int) -> np.ndarray:
    center_x, center_y, box_width, box_height = map(float, box)
    return np.asarray(
        [
            (center_x - box_width / 2.0) * width,
            (center_y - box_height / 2.0) * height,
            (center_x + box_width / 2.0) * width,
            (center_y + box_height / 2.0) * height,
        ],
        dtype=np.float64,
    )


def _candidate_xyxy(candidate: Mapping[str, Any]) -> np.ndarray:
    bbox = candidate.get("bbox")
    if not isinstance(bbox, Mapping):
        raise ValueError("candidate bbox must be an x1/y1/x2/y2 mapping")
    try:
        return np.asarray(
            [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("candidate bbox must contain numeric x1/y1/x2/y2") from exc


def query_feature_vector(
    hidden_states: Any,
    logits: Any,
    pred_boxes: Any,
    *,
    query_index: int,
) -> np.ndarray:
    """Return one frozen query's context, class logits and normalized box."""
    query_hidden = _as_single_image_array(hidden_states, name="hidden_states")
    query_logits = _as_single_image_array(logits, name="logits")
    query_boxes = _as_single_image_array(pred_boxes, name="pred_boxes")
    query_count = query_hidden.shape[0]
    if (
        query_logits.shape[0] != query_count
        or query_boxes.shape != (query_count, 4)
        or query_index < 0
        or query_index >= query_count
    ):
        raise ValueError("query feature tensors or query_index are incompatible")
    return np.concatenate(
        [
            query_hidden[query_index],
            query_logits[query_index],
            query_boxes[query_index],
        ]
    ).astype(np.float64, copy=False)


def _point_key(frame: int, x: float, y: float) -> tuple[int, float, float]:
    return frame, round(x, 5), round(y, 5)


def _track_sequences(
    perception: Mapping[str, Any],
) -> tuple[dict[str, tuple[str, ...]], dict[str, int]]:
    detections = list(perception.get("detections") or [])
    detection_by_point: dict[tuple[int, float, float], str] = {}
    frame_by_id: dict[str, int] = {}
    for row in detections:
        detection_id = str(row["detection_id"])
        if detection_id in frame_by_id:
            raise ValueError("perception detection IDs must be unique")
        bbox = row["bbox"]
        key = _point_key(
            int(row["frame"]),
            (float(bbox["x1"]) + float(bbox["x2"])) / 2.0,
            (float(bbox["y1"]) + float(bbox["y2"])) / 2.0,
        )
        if key in detection_by_point:
            raise ValueError("multiple detections share one track point")
        detection_by_point[key] = detection_id
        frame_by_id[detection_id] = int(row["frame"])

    sequence_by_id: dict[str, tuple[str, ...]] = {}
    for track in perception.get("ball_tracks") or []:
        members: list[str] = []
        for point in track.get("points") or []:
            if not bool(point.get("visible", True)) or bool(
                point.get("predicted", False)
            ):
                continue
            key = _point_key(
                int(point["frame"]),
                float(point["center"]["x"]),
                float(point["center"]["y"]),
            )
            detection_id = detection_by_point.get(key)
            if detection_id is None:
                raise ValueError("visible track point has no matching detection")
            members.append(detection_id)
        if not members:
            continue
        ordered = tuple(sorted(set(members), key=lambda value: frame_by_id[value]))
        for detection_id in ordered:
            if detection_id in sequence_by_id:
                raise ValueError("a detection belongs to multiple ball tracks")
            sequence_by_id[detection_id] = ordered
    return sequence_by_id, frame_by_id


def expand_detection_ids_to_tracks(
    perception: Mapping[str, Any],
    detection_ids: set[str],
) -> set[str]:
    """Expand reviewed candidates to visible members of their detector tracks."""
    sequences, frame_by_id = _track_sequences(perception)
    if not detection_ids.issubset(frame_by_id):
        raise ValueError("selected candidate is missing from perception")
    expanded = set(detection_ids)
    for detection_id in detection_ids:
        expanded.update(sequences.get(detection_id, ()))
    return expanded


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def track_query_features(
    perception: Mapping[str, Any],
    feature_by_id: Mapping[str, np.ndarray],
    *,
    context_width: int,
    label_width: int,
) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    """Summarize frozen RF-DETR query changes along each visible ball track."""
    total_width = context_width + label_width + 4
    if context_width <= 0 or label_width <= 0 or not feature_by_id:
        raise ValueError("invalid query feature contract")
    normalized = {
        str(detection_id): np.asarray(value, dtype=np.float64)
        for detection_id, value in feature_by_id.items()
    }
    if any(value.shape != (total_width,) for value in normalized.values()):
        raise ValueError("query feature width does not match the contract")
    sequences, frame_by_id = _track_sequences(perception)
    tail_names = tuple(
        [f"raw_logit_{index}" for index in range(label_width)]
        + ["box_cx", "box_cy", "box_width", "box_height"]
    )
    names = _TRACK_BASE_FEATURE_NAMES + tuple(
        f"track_{stat}_{name}"
        for name in tail_names
        for stat in ("mean", "std", "min", "max")
    )
    output: dict[str, np.ndarray] = {}
    for detection_id, current in normalized.items():
        full_sequence = sequences.get(detection_id, ())
        sequence = [
            member for member in full_sequence if member in normalized
        ] or [detection_id]
        values = np.stack([normalized[member] for member in sequence])
        contexts = values[:, :context_width]
        adjacent_cosines = [
            _cosine(left, right)
            for left, right in zip(contexts, contexts[1:], strict=False)
        ]
        adjacent_l2 = [
            float(np.linalg.norm(right - left))
            for left, right in zip(contexts, contexts[1:], strict=False)
        ]
        frames = [frame_by_id[member] for member in sequence]
        base = np.asarray(
            [
                float(bool(full_sequence)),
                float(len(sequence)),
                float(max(frames) - min(frames)),
                _cosine(current[:context_width], contexts.mean(axis=0)),
                float(np.mean(adjacent_cosines)) if adjacent_cosines else 0.0,
                float(min(adjacent_cosines)) if adjacent_cosines else 0.0,
                float(np.mean(adjacent_l2)) if adjacent_l2 else 0.0,
                float(max(adjacent_l2)) if adjacent_l2 else 0.0,
            ],
            dtype=np.float64,
        )
        tail = values[:, context_width:]
        statistics = np.asarray(
            [
                value
                for column in tail.T
                for value in (
                    float(np.mean(column)),
                    float(np.std(column)),
                    float(np.min(column)),
                    float(np.max(column)),
                )
            ],
            dtype=np.float64,
        )
        output[detection_id] = np.concatenate([base, statistics])
    return names, output


def match_candidates_to_queries(
    logits: Any,
    pred_boxes: Any,
    candidates: Sequence[Mapping[str, Any]],
    *,
    image_width: int,
    image_height: int,
    confidence_threshold: float,
    ball_label_id: int,
    confidence_tolerance: float = 1e-5,
    bbox_tolerance_px: float = 1e-3,
) -> dict[str, dict[str, int | float]]:
    """Reproduce RF-DETR post-processing and map artifact rows to query indices.

    RF-DETR object indices are positions in the thresholded top-query output,
    not decoder-query indices. Reconstructing the exact post-processing order
    prevents a verifier from learning against the wrong decoder embedding.
    """
    query_logits = _as_single_image_array(logits, name="logits")
    query_boxes = _as_single_image_array(pred_boxes, name="pred_boxes")
    if query_logits.shape[0] != query_boxes.shape[0] or query_boxes.shape[1] != 4:
        raise ValueError("logits and pred_boxes have incompatible query shapes")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between zero and one")
    if ball_label_id < 0 or ball_label_id >= query_logits.shape[1]:
        raise ValueError("ball_label_id is outside the detector label range")
    if confidence_tolerance < 0.0 or bbox_tolerance_px < 0.0:
        raise ValueError("matching tolerances must be non-negative")

    # Match AutoImageProcessor.post_process_object_detection: sigmoid every
    # query/class logit, flatten, take num_queries highest values, then filter.
    import torch

    flat_scores = torch.as_tensor(query_logits).sigmoid().flatten()
    query_count, class_count = query_logits.shape
    top_scores, top_indices = torch.topk(flat_scores, query_count)
    selected: list[tuple[int, int, float]] = []
    for score_value, flat_index_value in zip(top_scores, top_indices, strict=True):
        score = float(score_value)
        if score > confidence_threshold:
            flat_index = int(flat_index_value)
            selected.append(
                (flat_index // class_count, flat_index % class_count, score)
            )

    output: dict[str, dict[str, int | float]] = {}
    for candidate in candidates:
        detection_id = str(candidate.get("detection_id", ""))
        if not detection_id or detection_id in output:
            raise ValueError("candidate detection_id must be present and unique")
        index = _object_index(detection_id)
        if index >= len(selected):
            raise ValueError(
                f"detection object index is outside reconstructed output: {detection_id}"
            )
        query_index, label_id, score = selected[index]
        if label_id != ball_label_id:
            raise ValueError(
                f"detection does not reconstruct as the ball label: {detection_id}"
            )
        try:
            artifact_score = float(candidate["confidence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("candidate confidence must be numeric") from exc
        if not np.isclose(
            score,
            artifact_score,
            rtol=0.0,
            atol=confidence_tolerance,
        ):
            raise ValueError(
                f"confidence mismatch for {detection_id}: "
                f"{score:.8f} != {artifact_score:.8f}"
            )
        reconstructed_bbox = _pixel_xyxy(
            query_boxes[query_index],
            width=image_width,
            height=image_height,
        )
        artifact_bbox = _candidate_xyxy(candidate)
        if not np.allclose(
            reconstructed_bbox,
            artifact_bbox,
            rtol=0.0,
            atol=bbox_tolerance_px,
        ):
            delta = float(np.max(np.abs(reconstructed_bbox - artifact_bbox)))
            raise ValueError(
                f"bbox mismatch for {detection_id}: max delta {delta:.6f}px"
            )
        output[detection_id] = {
            "query_index": query_index,
            "label_id": label_id,
            "score": score,
        }
    return output
