"""Training-only camera-compensated motion screening on MUVS frame pairs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.analysis.muvs_event_state import (
    verify_muvs_event_state_frames,
    verify_muvs_event_state_plan,
    verify_muvs_event_state_review,
)

MUVS_PAIR_MOTION_FEATURES_SCHEMA = "agu.muvs-pair-motion-features.v1"
MUVS_PAIR_MOTION_SCREEN_SCHEMA = "agu.muvs-pair-motion-screen.v1"
MUVS_PAIR_MOTION_DIMENSION = 54
MUVS_PAIR_MOTION_STATES = {
    "live_play",
    "free_throw_setup",
    "dead_ball_timeout",
    "uncertain",
}
_GRID_ROWS = 3
_GRID_COLUMNS = 4
_SHA256_LENGTH = 64


def extract_muvs_pair_motion_features(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
) -> dict[str, Any]:
    """Extract fixed camera-compensated residual and dense-flow statistics."""

    before = _prepare_gray(before_bgr)
    after = _prepare_gray(after_bgr)
    if before.shape != after.shape:
        raise ValueError("MUVS pair motion frames must have the same shape")
    height, width = before.shape
    transform, match_count, inlier_ratio, applied = _estimate_transform(
        before,
        after,
    )
    compensated = cv2.warpPerspective(
        after,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    residual = cv2.absdiff(before, compensated).astype(np.float32) / 255.0
    flow = cv2.calcOpticalFlowFarneback(
        before,
        compensated,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    magnitude = np.linalg.norm(flow, axis=2) / math.hypot(width, height)
    linear = transform[:2, :2]
    scale = math.sqrt(abs(float(np.linalg.det(linear))))
    diagnostics = [
        float(applied),
        min(match_count, 200) / 200.0,
        inlier_ratio,
        float(transform[0, 2]) / width,
        float(transform[1, 2]) / height,
        math.log(max(scale, 1e-6)),
    ]
    grid_features: list[float] = []
    roi_top = height // 5
    row_edges = np.linspace(roi_top, height, _GRID_ROWS + 1, dtype=int)
    column_edges = np.linspace(0, width, _GRID_COLUMNS + 1, dtype=int)
    for row_index in range(_GRID_ROWS):
        for column_index in range(_GRID_COLUMNS):
            row_slice = slice(row_edges[row_index], row_edges[row_index + 1])
            column_slice = slice(
                column_edges[column_index],
                column_edges[column_index + 1],
            )
            residual_cell = residual[row_slice, column_slice]
            magnitude_cell = magnitude[row_slice, column_slice]
            grid_features.extend(
                (
                    float(residual_cell.mean()),
                    float(np.quantile(residual_cell, 0.9)),
                    float(magnitude_cell.mean()),
                    float(np.quantile(magnitude_cell, 0.9)),
                )
            )
    features = diagnostics + grid_features
    if (
        len(features) != MUVS_PAIR_MOTION_DIMENSION
        or not np.isfinite(features).all()
    ):
        raise ValueError("invalid MUVS pair motion feature vector")
    return {
        "homography_applied": applied,
        "match_count": match_count,
        "inlier_ratio": inlier_ratio,
        "features": features,
    }


def build_muvs_pair_motion_features(
    *,
    plan: Mapping[str, Any],
    frames: Mapping[str, Any],
    review: Mapping[str, Any],
    frame_loader: Callable[[Mapping[str, Any]], np.ndarray],
) -> dict[str, Any]:
    """Join verified source rows and extract one fixed feature vector per pair."""

    verified_plan = verify_muvs_event_state_plan(plan)
    verified_frames = verify_muvs_event_state_frames(
        frames,
        plan=verified_plan,
    )
    verified_review = verify_muvs_event_state_review(
        review,
        plan=verified_plan,
        frames=verified_frames,
    )
    frame_rows = {
        str(row["sample_id"]): row for row in verified_frames["files"]
    }
    decisions = {
        str(row["sample_id"]): row for row in verified_review["decisions"]
    }
    examples = []
    for sample in verified_plan["samples"]:
        sample_id = str(sample["sample_id"])
        frame_row = frame_rows[sample_id]
        result = extract_muvs_pair_motion_features(
            frame_loader(frame_row["frame_before"]),
            frame_loader(frame_row["frame_after"]),
        )
        examples.append(
            {
                "sample_id": sample_id,
                "event_id": str(sample["event_id"]),
                "split": str(sample["split"]),
                "state": str(decisions[sample_id]["state"]),
                "frame_before_sha256": str(
                    frame_row["frame_before"]["sha256"]
                ),
                "frame_after_sha256": str(
                    frame_row["frame_after"]["sha256"]
                ),
                **result,
            }
        )
    return seal_muvs_pair_motion_features(
        {
            "frame_artifact_sha256": verified_frames["artifact_sha256"],
            "review_artifact_sha256": verified_review["artifact_sha256"],
            "examples": examples,
        }
    )


def seal_muvs_pair_motion_features(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["schema_version"] = MUVS_PAIR_MOTION_FEATURES_SCHEMA
    artifact["purpose"] = "offline_muvs_source_pair_motion_pretraining"
    artifact["runtime_consumable"] = False
    artifact["codex_runtime_answer_used"] = False
    artifact["feature_dimension"] = MUVS_PAIR_MOTION_DIMENSION
    _validate_features(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_muvs_pair_motion_features(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_features(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("MUVS pair motion feature hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def screen_muvs_pair_motion(
    feature_artifact: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payloads = (
        [feature_artifact]
        if isinstance(feature_artifact, Mapping)
        else list(feature_artifact)
    )
    if not payloads:
        raise ValueError("MUVS pair motion screen requires features")
    artifacts = [verify_muvs_pair_motion_features(row) for row in payloads]
    rows_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for artifact in artifacts:
        for row in artifact["examples"]:
            if row["state"] == "uncertain":
                continue
            key = (
                str(row["frame_before_sha256"]),
                str(row["frame_after_sha256"]),
            )
            candidate = {
                **row,
                "_feature_artifact_sha256": artifact["artifact_sha256"],
            }
            previous = rows_by_pair.get(key)
            if previous is not None:
                if any(
                    previous[field] != candidate[field]
                    for field in (
                        "event_id",
                        "state",
                        "homography_applied",
                        "match_count",
                        "inlier_ratio",
                        "features",
                    )
                ):
                    raise ValueError("duplicate MUVS pair motion disagrees")
                continue
            rows_by_pair[key] = candidate
    rows = list(rows_by_pair.values())
    labels = np.asarray(
        [row["state"] == "free_throw_setup" for row in rows],
        dtype=np.int64,
    )
    groups = np.asarray([str(row["event_id"]) for row in rows])
    if len(set(labels.tolist())) != 2 or len(set(groups.tolist())) < 3:
        raise ValueError("MUVS pair motion screen requires both classes and events")
    features = np.asarray([row["features"] for row in rows], dtype=np.float64)
    scores = np.zeros(len(rows), dtype=np.float64)
    folds = []
    for train, test in LeaveOneGroupOut().split(features, labels, groups):
        if len(set(labels[train].tolist())) != 2:
            raise ValueError("each MUVS pair motion fold must retain both classes")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=5000,
                random_state=0,
            ),
        )
        model.fit(features[train], labels[train])
        fold_scores = model.decision_function(features[test])
        scores[test] = fold_scores
        folds.append(
            _fold_metrics(
                event_id=str(groups[test][0]),
                labels=labels[test],
                scores=fold_scores,
            )
        )
    predictions = scores >= 0.0
    positive_recall = _class_recall(labels, predictions, positive_class=1)
    negative_specificity = _class_recall(
        labels,
        predictions,
        positive_class=0,
    )
    positive_folds = [
        float(row["positive_recall"])
        for row in folds
        if row["positive_examples"] > 0
    ]
    negative_folds = [
        float(row["negative_specificity"])
        for row in folds
        if row["negative_examples"] > 0
    ]
    metrics = {
        "balanced_accuracy": float(
            (positive_recall + negative_specificity) / 2.0
        ),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "positive_recall": positive_recall,
        "negative_specificity": negative_specificity,
        "worst_positive_event_recall": min(positive_folds),
        "worst_negative_event_specificity": min(negative_folds),
    }
    artifact: dict[str, Any] = {
        "schema_version": MUVS_PAIR_MOTION_SCREEN_SCHEMA,
        "purpose": "offline_source_feature_acceptance_only",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "feature_artifact_sha256s": [
            row["artifact_sha256"] for row in artifacts
        ],
        "resolved_examples": len(rows),
        "event_count": len(set(groups.tolist())),
        "positive_examples": int(labels.sum()),
        "negative_examples": int((labels == 0).sum()),
        "protocol": {
            "grouping": "leave_one_event_out",
            "classifier": "StandardScaler+balanced_LogisticRegression_C1",
            "threshold": "fixed_zero_logistic_margin",
            "selection_data": "MUVS_source_only",
            "acceptance_metric": "source_oof_balanced_accuracy",
            "acceptance_threshold": 0.85,
            "minimum_positive_event_recall": 0.5,
            "minimum_negative_specificity": 0.85,
        },
        "metrics": metrics,
        "folds": folds,
        "oof_predictions": [
            {
                "sample_id": str(row["sample_id"]),
                "event_id": str(row["event_id"]),
                "feature_artifact_sha256": str(
                    row["_feature_artifact_sha256"]
                ),
                "label": bool(label),
                "score": float(score),
                "prediction": bool(score >= 0.0),
            }
            for row, label, score in zip(rows, labels, scores, strict=True)
        ],
        "accepted": bool(
            metrics["balanced_accuracy"] >= 0.85
            and metrics["worst_positive_event_recall"] >= 0.5
            and metrics["negative_specificity"] >= 0.85
        ),
    }
    _validate_screen(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_muvs_pair_motion_screen(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_screen(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("MUVS pair motion screen hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _prepare_gray(image: np.ndarray) -> np.ndarray:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) < 32
    ):
        raise ValueError("invalid MUVS pair motion frame")
    height, width = image.shape[:2]
    if width > 320:
        target_height = max(32, round(height * 320 / width))
        image = cv2.resize(
            image,
            (320, target_height),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _estimate_transform(
    before: np.ndarray,
    after: np.ndarray,
) -> tuple[np.ndarray, int, float, bool]:
    identity = np.eye(3, dtype=np.float64)
    detector = cv2.ORB_create(nfeatures=800)
    before_points, before_descriptors = detector.detectAndCompute(before, None)
    after_points, after_descriptors = detector.detectAndCompute(after, None)
    if before_descriptors is None or after_descriptors is None:
        return identity, 0, 0.0, False
    matches = sorted(
        cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True).match(
            before_descriptors,
            after_descriptors,
        ),
        key=lambda row: row.distance,
    )[:300]
    match_count = len(matches)
    if match_count < 8:
        return identity, match_count, 0.0, False
    source = np.float32(
        [after_points[row.trainIdx].pt for row in matches]
    ).reshape(-1, 1, 2)
    target = np.float32(
        [before_points[row.queryIdx].pt for row in matches]
    ).reshape(-1, 1, 2)
    transform, inliers = cv2.findHomography(
        source,
        target,
        cv2.RANSAC,
        3.0,
    )
    inlier_ratio = (
        float(np.asarray(inliers, dtype=np.float64).mean())
        if inliers is not None
        else 0.0
    )
    if transform is None or not _reasonable_transform(transform, before.shape):
        return identity, match_count, inlier_ratio, False
    return transform.astype(np.float64), match_count, inlier_ratio, True


def _reasonable_transform(
    transform: np.ndarray,
    shape: tuple[int, int],
) -> bool:
    if transform.shape != (3, 3) or not np.isfinite(transform).all():
        return False
    height, width = shape
    corners = np.float32(
        [[0, 0], [width, 0], [width, height], [0, height]]
    ).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, transform)
    if not np.isfinite(projected).all():
        return False
    area = abs(float(cv2.contourArea(projected)))
    ratio = area / float(width * height)
    return 0.4 <= ratio <= 2.5


def _fold_metrics(
    *,
    event_id: str,
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    predictions = scores >= 0.0
    positives = int(labels.sum())
    negatives = int((labels == 0).sum())
    return {
        "event_id": event_id,
        "examples": len(labels),
        "positive_examples": positives,
        "negative_examples": negatives,
        "positive_recall": (
            _class_recall(labels, predictions, positive_class=1)
            if positives
            else None
        ),
        "negative_specificity": (
            _class_recall(labels, predictions, positive_class=0)
            if negatives
            else None
        ),
    }


def _class_recall(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    positive_class: int,
) -> float:
    mask = labels == positive_class
    if not mask.any():
        raise ValueError("class recall requires at least one example")
    return float((predictions[mask] == bool(positive_class)).mean())


def _validate_features(artifact: Mapping[str, Any]) -> None:
    examples = artifact.get("examples")
    if (
        artifact.get("schema_version") != MUVS_PAIR_MOTION_FEATURES_SCHEMA
        or artifact.get("purpose")
        != "offline_muvs_source_pair_motion_pretraining"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("feature_dimension") != MUVS_PAIR_MOTION_DIMENSION
        or not isinstance(examples, list)
        or not examples
    ):
        raise ValueError("invalid MUVS pair motion feature artifact")
    _require_sha256(artifact.get("frame_artifact_sha256"))
    _require_sha256(artifact.get("review_artifact_sha256"))
    sample_ids = []
    for row in examples:
        values = np.asarray(row.get("features"), dtype=np.float64)
        if (
            set(row)
            != {
                "sample_id",
                "event_id",
                "split",
                "state",
                "frame_before_sha256",
                "frame_after_sha256",
                "homography_applied",
                "match_count",
                "inlier_ratio",
                "features",
            }
            or not str(row.get("sample_id") or "")
            or not str(row.get("event_id") or "")
            or row.get("split") not in {"train", "test"}
            or row.get("state") not in MUVS_PAIR_MOTION_STATES
            or not isinstance(row.get("homography_applied"), bool)
            or not isinstance(row.get("match_count"), int)
            or isinstance(row.get("match_count"), bool)
            or int(row["match_count"]) < 0
            or not 0.0 <= float(row.get("inlier_ratio", math.nan)) <= 1.0
            or values.shape != (MUVS_PAIR_MOTION_DIMENSION,)
            or not np.isfinite(values).all()
        ):
            raise ValueError("invalid MUVS pair motion feature row")
        _require_sha256(row.get("frame_before_sha256"))
        _require_sha256(row.get("frame_after_sha256"))
        sample_ids.append(str(row["sample_id"]))
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate MUVS pair motion sample")


def _validate_screen(artifact: Mapping[str, Any]) -> None:
    metrics = artifact.get("metrics")
    protocol = artifact.get("protocol")
    if (
        artifact.get("schema_version") != MUVS_PAIR_MOTION_SCREEN_SCHEMA
        or artifact.get("purpose") != "offline_source_feature_acceptance_only"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or not isinstance(metrics, Mapping)
        or not isinstance(protocol, Mapping)
        or protocol.get("grouping") != "leave_one_event_out"
        or protocol.get("threshold") != "fixed_zero_logistic_margin"
        or int(artifact.get("resolved_examples", 0)) < 1
        or int(artifact.get("event_count", 0)) < 3
        or not isinstance(artifact.get("accepted"), bool)
    ):
        raise ValueError("invalid MUVS pair motion screen")
    hashes = artifact.get("feature_artifact_sha256s")
    if not isinstance(hashes, list) or not hashes:
        raise ValueError("MUVS pair motion screen requires artifact hashes")
    for value in hashes:
        _require_sha256(value)
    for key in (
        "balanced_accuracy",
        "roc_auc",
        "positive_recall",
        "negative_specificity",
        "worst_positive_event_recall",
        "worst_negative_event_specificity",
    ):
        value = float(metrics.get(key, math.nan))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("invalid MUVS pair motion metric")


def _require_sha256(value: object) -> str:
    text = str(value or "")
    if len(text) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError("invalid MUVS pair motion SHA-256")
    return text


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
