"""Offline-only adapter and external screen for the frozen DEIM basketball detector."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from app.analysis.ball_candidate_review import canonical_sha256, verify_artifact
from app.analysis.broadcast_ball_detector_screen import detector_frame_metrics

DEIM_MODEL_ID = "ortizeg/basketball-deim-m-640"
DEIM_MODEL_REVISION = "1b4f378bc1fa8d3a7980768d3b08486b6f3fe357"
DEIM_OFFICIAL_CODE_REVISION = "26d25268a56c0a32436023f29be66ab3f0352ca2"
DEIM_RAW10_LABELS = (
    "ball",
    "ball-in-basket",
    "number",
    "player",
    "player-in-possession",
    "player-jump-shot",
    "player-layup-dunk",
    "player-shot-block",
    "referee",
    "rim",
)
BALL_CLASS_IDS = frozenset({0, 1})
DEIM_SCREEN_THRESHOLDS = (0.1, 0.25, 0.5)


class _SessionInput(Protocol):
    name: str


class _InferenceSession(Protocol):
    def get_inputs(self) -> Sequence[_SessionInput]: ...

    def get_outputs(self) -> Sequence[_SessionInput]: ...

    def run(
        self,
        output_names: None,
        feed: Mapping[str, np.ndarray],
    ) -> Sequence[np.ndarray]: ...


SessionFactory = Callable[[str, Sequence[str]], _InferenceSession]


def _strict_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _confidence_floor(value: Any) -> float:
    result = _finite_number(value, field="confidence floor")
    if not 0 <= result < 1:
        raise ValueError("confidence floor must be in [0, 1)")
    return result


def preprocess_deim_frame(frame: np.ndarray, *, input_size: int = 640) -> np.ndarray:
    """Apply the official DEIM BGR-to-RGB square-resize preprocessing contract.

    Source contract (fixed commit):
    https://github.com/ortizeg/object-detection-eval/blob/26d25268a56c0a32436023f29be66ab3f0352ca2/src/object_detection_eval/inference/preprocess.py
    """
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("DEIM input must be a uint8 HWC BGR frame")
    if type(input_size) is not int or input_size <= 0:
        raise ValueError("input size must be a positive integer")
    rgb = frame[:, :, ::-1]
    image = Image.fromarray(rgb).resize(
        (input_size, input_size),
        resample=Image.Resampling.BILINEAR,
    )
    array = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...])


def _strip_batch(array: Any, *, field: str) -> np.ndarray:
    result = np.asarray(array)
    if result.ndim < 1 or result.shape[0] != 1:
        raise ValueError(f"DEIM {field} must have a single batch dimension")
    return result[0]


def decode_deim_ball_outputs(
    outputs: Sequence[Any],
    *,
    confidence_floor: float,
) -> list[dict[str, Any]]:
    """Decode official labels/boxes/scores outputs and retain both ball classes."""
    floor = _confidence_floor(confidence_floor)
    if not isinstance(outputs, Sequence) or len(outputs) != 3:
        raise ValueError("DEIM must return exactly three outputs")
    labels = _strip_batch(outputs[0], field="labels")
    boxes = _strip_batch(outputs[1], field="boxes")
    scores = _strip_batch(outputs[2], field="scores")
    if labels.ndim != 1 or scores.ndim != 1 or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("DEIM output shapes are invalid")
    if not (len(labels) == len(boxes) == len(scores)):
        raise ValueError("DEIM labels, boxes, and scores do not align")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("DEIM labels must be integers")
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("DEIM boxes and scores must be finite")

    decoded: list[dict[str, Any]] = []
    for class_value, box_value, score_value in zip(labels, boxes, scores):
        class_id = int(class_value)
        score = float(score_value)
        if class_id not in BALL_CLASS_IDS or score <= floor:
            continue
        x1, y1, x2, y2 = (float(value) for value in box_value)
        if x2 < x1 or y2 < y1:
            raise ValueError("DEIM returned an inverted bounding box")
        decoded.append(
            {
                "class_id": class_id,
                "class_name": DEIM_RAW10_LABELS[class_id],
                "confidence": score,
                "bbox_xyxy": [x1, y1, x2, y2],
            }
        )
    return decoded


class DeimBasketballDetector:
    """Small ONNX Runtime wrapper for the frozen official DEIM-M export."""

    def __init__(
        self,
        model_path: Path,
        *,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        session_factory: SessionFactory | None = None,
    ) -> None:
        resolved_model = model_path.resolve()
        if not resolved_model.is_file():
            raise ValueError("DEIM model path must be a regular file")
        if not providers or any(not isinstance(provider, str) or not provider for provider in providers):
            raise ValueError("DEIM providers must be non-empty strings")
        if session_factory is None:
            import onnxruntime as ort

            session_factory = lambda path, selected: ort.InferenceSession(  # noqa: E731
                path,
                providers=list(selected),
            )
        self._session = session_factory(str(resolved_model), tuple(providers))
        inputs = tuple(self._session.get_inputs())
        outputs = tuple(self._session.get_outputs())
        if len(inputs) != 2 or len(outputs) != 3:
            raise ValueError("DEIM session must expose two inputs and three outputs")
        self._image_input = inputs[0].name
        self._size_input = inputs[1].name

    def predict(
        self,
        frame: np.ndarray,
        *,
        confidence_floor: float = 0.01,
    ) -> list[dict[str, Any]]:
        image = preprocess_deim_frame(frame)
        height, width = frame.shape[:2]
        original_size = np.asarray([[width, height]], dtype=np.int64)
        outputs = self._session.run(
            None,
            {
                self._image_input: image,
                self._size_input: original_size,
            },
        )
        return decode_deim_ball_outputs(outputs, confidence_floor=confidence_floor)


def _prediction_boxes(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> list[list[float]]:
    boxes: list[list[float]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("prediction rows must be objects")
        if set(row) != {"class_id", "class_name", "confidence", "bbox_xyxy"}:
            raise ValueError("prediction rows require exact fields")
        class_id = row["class_id"]
        if type(class_id) is not int or class_id not in BALL_CLASS_IDS:
            raise ValueError("prediction class ID must be a canonical ball class")
        if row["class_name"] != DEIM_RAW10_LABELS[class_id]:
            raise ValueError("prediction class name does not match its class ID")
        confidence = _finite_number(row.get("confidence"), field="prediction confidence")
        if not 0 <= confidence <= 1:
            raise ValueError("prediction confidence must be in [0, 1]")
        bbox = row.get("bbox_xyxy")
        if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
            raise ValueError("prediction boxes must be four-element sequences")
        box = [_finite_number(value, field="prediction box coordinate") for value in bbox]
        if box[2] < box[0] or box[3] < box[1]:
            raise ValueError("prediction boxes must not be inverted")
        if confidence > threshold:
            boxes.append(box)
    return boxes


def build_deim_external_screen(
    plan_payload: Mapping[str, Any],
    review_payload: Mapping[str, Any],
    predictions_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    model_sha256: str,
    raw_video_sha256: str,
    iou_threshold: float = 0.25,
) -> dict[str, Any]:
    """Build a sealed, permanently non-promotable external-screen artifact."""
    model_sha = _strict_sha256(model_sha256, field="model SHA-256")
    raw_sha = _strict_sha256(raw_video_sha256, field="raw video SHA-256")
    iou = _finite_number(iou_threshold, field="IoU threshold")
    if not 0 < iou <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    plan = verify_artifact(plan_payload)
    review = verify_artifact(review_payload)
    if review.get("plan_sha256") != plan["artifact_sha256"]:
        raise ValueError("review is not bound to the plan")
    if plan.get("raw_video_sha256") != raw_sha:
        raise ValueError("plan is not bound to the raw video")

    candidates = plan.get("candidates")
    decisions = review.get("decisions")
    if not isinstance(candidates, list) or not isinstance(decisions, list):
        raise ValueError("plan candidates and review decisions must be lists")
    candidate_map: dict[str, Mapping[str, Any]] = {}
    for row in candidates:
        if not isinstance(row, Mapping) or not isinstance(row.get("candidate_id"), str):
            raise ValueError("plan candidate shape is invalid")
        candidate_id = row["candidate_id"]
        if not candidate_id or candidate_id in candidate_map:
            raise ValueError("plan candidate IDs must be unique and non-empty")
        frame = row.get("frame")
        if type(frame) is not int or frame < 0:
            raise ValueError("plan candidate frame must be a non-negative integer")
        bbox = row.get("bbox")
        if not isinstance(bbox, Mapping) or set(bbox) != {"x1", "y1", "x2", "y2"}:
            raise ValueError("plan candidate bbox requires exact fields")
        target = [_finite_number(bbox[key], field="target box coordinate") for key in ("x1", "y1", "x2", "y2")]
        if target[2] < target[0] or target[3] < target[1]:
            raise ValueError("target boxes must not be inverted")
        candidate_map[candidate_id] = row
    decision_map: dict[str, str] = {}
    for row in decisions:
        if not isinstance(row, Mapping) or not isinstance(row.get("candidate_id"), str):
            raise ValueError("review decision shape is invalid")
        candidate_id = row["candidate_id"]
        decision = row.get("decision")
        if candidate_id in decision_map or decision not in {
            "valid_ball",
            "false_positive",
            "uncertain",
        }:
            raise ValueError("review decisions must be unique and canonical")
        decision_map[candidate_id] = decision
    if set(decision_map) != set(candidate_map):
        raise ValueError("review coverage must exactly match the plan")

    determinate_ids = {candidate_id for candidate_id, decision in decision_map.items() if decision != "uncertain"}
    if set(predictions_by_candidate) != determinate_ids:
        raise ValueError("predictions must exactly cover determinate reviewed candidates")

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for candidate_id, candidate in candidate_map.items():
        decision = decision_map[candidate_id]
        if decision == "uncertain":
            continue
        rows = predictions_by_candidate[candidate_id]
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise ValueError("candidate predictions must be a sequence")
        normalized = [dict(row) for row in rows]
        # Validate every prediction once before metrics are derived.
        _prediction_boxes(normalized, threshold=1.0)
        metric_rows.append(
            {
                "candidate_id": candidate_id,
                "decision": decision,
                "bbox": dict(candidate["bbox"]),
            }
        )
        prediction_rows.append(
            {
                "candidate_id": candidate_id,
                "frame": int(candidate["frame"]),
                "decision": decision,
                "target_bbox": dict(candidate["bbox"]),
                "ball_predictions": normalized,
            }
        )

    metrics_by_threshold: list[dict[str, Any]] = []
    for threshold in DEIM_SCREEN_THRESHOLDS:
        metrics = detector_frame_metrics(
            metric_rows,
            [_prediction_boxes(row["ball_predictions"], threshold=threshold) for row in prediction_rows],
            iou_threshold=iou,
        )
        metrics_by_threshold.append({"confidence_threshold": threshold, **metrics})
    meets_followup_gate = any(row["precision"] >= 0.85 and row["recall"] >= 0.85 for row in metrics_by_threshold)
    artifact: dict[str, Any] = {
        "schema_version": "agu.deim-basketball-detector-external-screen.v1",
        "purpose": "offline_independent_broadcast_ball_detector_diagnostic",
        "runtime_consumable": False,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
        "codex_runtime_answer_used": False,
        "model": {
            "repository": DEIM_MODEL_ID,
            "revision": DEIM_MODEL_REVISION,
            "official_code_revision": DEIM_OFFICIAL_CODE_REVISION,
            "artifact_sha256": model_sha,
            "backend": "onnxruntime_cpu",
            "class_policy": "raw10 class ids 0 ball and 1 ball-in-basket are merged as ball",
        },
        "raw_video_sha256": raw_sha,
        "plan_sha256": plan["artifact_sha256"],
        "review_sha256": review["artifact_sha256"],
        "thresholds": list(DEIM_SCREEN_THRESHOLDS),
        "iou_threshold": iou,
        "metrics_by_threshold": metrics_by_threshold,
        "meets_followup_gate": meets_followup_gate,
        "limitations": [
            "The sealed review covers candidates produced by another detector, not exhaustive full-frame ball truth.",
            "Official model-card metrics are clip-held but not game-held; all official test games also appear in training clips.",
            "This artifact is an offline diagnostic and cannot promote runtime readiness.",
        ],
        "predictions": prediction_rows,
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return artifact
