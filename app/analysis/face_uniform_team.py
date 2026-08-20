"""Face-anchored uniform color model for offline enrollment team filtering."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Sequence

import cv2
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = (
    "red_low_fraction",
    "orange_fraction",
    "yellow_green_fraction",
    "cyan_fraction",
    "blue_fraction",
    "purple_fraction",
    "red_high_fraction",
    "low_saturation_fraction",
    "white_fraction",
    "dark_fraction",
    "mean_saturation",
    "mean_value",
)


def extract_face_anchored_uniform_features(
    frame: np.ndarray,
    face_bbox: Sequence[float],
) -> np.ndarray:
    """Measure HSV color below a face without using a broad player box."""

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("uniform feature frame must be a BGR image")
    if len(face_bbox) != 4:
        raise ValueError("face bbox must contain x, y, width, height")
    x, y, width, height = (float(value) for value in face_bbox)
    if width <= 0 or height <= 0:
        raise ValueError("face bbox dimensions must be positive")
    frame_height, frame_width = frame.shape[:2]
    x1 = max(0, int(round(x - 0.40 * width)))
    x2 = min(frame_width, int(round(x + 1.40 * width)))
    y1 = max(0, int(round(y + 1.00 * height)))
    y2 = min(frame_height, int(round(y + 3.20 * height)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("face-anchored uniform crop is empty")
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    chromatic = (saturation >= 60) & (value >= 40)
    hue_ranges = (
        (0, 15),
        (15, 35),
        (35, 75),
        (75, 100),
        (100, 130),
        (130, 170),
        (170, 180),
    )
    features = [
        float(((hue >= low) & (hue < high) & chromatic).mean())
        for low, high in hue_ranges
    ]
    features.extend(
        [
            float((saturation < 55).mean()),
            float(((saturation < 55) & (value > 130)).mean()),
            float((value < 60).mean()),
            float(saturation.mean() / 255.0),
            float(value.mean() / 255.0),
        ]
    )
    return np.asarray(features, dtype=np.float64)


def fit_uniform_team_model(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    class_ids: tuple[str, str],
    minimum_cross_validated_accuracy: float = 0.85,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit a serializable logistic model after leave-one-person-out audit."""

    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    group_ids = np.asarray(groups)
    if matrix.ndim != 2 or matrix.shape[0] != target.size:
        raise ValueError("uniform feature matrix and labels do not align")
    if target.size != group_ids.size or target.size < 4:
        raise ValueError("uniform labels and groups do not align")
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("uniform training labels must contain both classes")
    if len(class_ids) != 2 or any(not str(value).strip() for value in class_ids):
        raise ValueError("uniform team class IDs must contain two values")
    if not 0.5 <= minimum_cross_validated_accuracy <= 1.0:
        raise ValueError("minimum cross-validated accuracy must be in [0.5, 1.0]")
    unique_groups = np.unique(group_ids)
    if unique_groups.size < 4:
        raise ValueError("uniform training requires at least four identity groups")
    for held_out_group in unique_groups:
        training_labels = target[group_ids != held_out_group]
        if set(np.unique(training_labels)) != {0, 1}:
            raise ValueError("every held-out identity fold must retain both classes")

    pipeline = _new_pipeline()
    predicted = cross_val_predict(
        pipeline,
        matrix,
        target,
        groups=group_ids,
        cv=LeaveOneGroupOut(),
        method="predict",
    )
    accuracy = float(accuracy_score(target, predicted))
    recalls = recall_score(target, predicted, labels=[0, 1], average=None)
    audit = {
        "validation": "leave_one_identity_out",
        "cross_validated_accuracy": accuracy,
        "class_recall": {
            str(class_ids[0]): float(recalls[0]),
            str(class_ids[1]): float(recalls[1]),
        },
        "sample_count": int(target.size),
        "group_count": int(unique_groups.size),
        "class_sample_counts": {
            str(class_ids[0]): int((target == 0).sum()),
            str(class_ids[1]): int((target == 1).sum()),
        },
        "minimum_cross_validated_accuracy": minimum_cross_validated_accuracy,
    }
    if accuracy < minimum_cross_validated_accuracy:
        raise ValueError(
            "uniform model cross-validated accuracy "
            f"{accuracy:.4f} is below {minimum_cross_validated_accuracy:.4f}"
        )

    pipeline.fit(matrix, target)
    scaler = pipeline.named_steps["standardscaler"]
    classifier = pipeline.named_steps["logisticregression"]
    model = {
        "schema_version": "agu.face-uniform-team-model.v1",
        "class_ids": [str(class_ids[0]), str(class_ids[1])],
        "feature_names": list(FEATURE_NAMES[: matrix.shape[1]]),
        "scaler_mean": [float(value) for value in scaler.mean_],
        "scaler_scale": [float(value) for value in scaler.scale_],
        "coefficient": [float(value) for value in classifier.coef_[0]],
        "intercept": float(classifier.intercept_[0]),
        "logistic_regression_c": 0.2,
        "class_weight": "balanced",
        "sklearn_version": sklearn.__version__,
    }
    return model, audit


def predict_uniform_team_probability(
    model: dict[str, Any],
    features: Sequence[float],
) -> float:
    """Return the probability of the second serialized team class."""

    values = np.asarray(features, dtype=np.float64)
    mean = np.asarray(model.get("scaler_mean") or [], dtype=np.float64)
    scale = np.asarray(model.get("scaler_scale") or [], dtype=np.float64)
    coefficient = np.asarray(model.get("coefficient") or [], dtype=np.float64)
    if (
        values.ndim != 1
        or values.size == 0
        or values.shape != mean.shape
        or values.shape != scale.shape
        or values.shape != coefficient.shape
        or np.any(scale <= 0)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("uniform prediction features do not match model")
    logit = float(coefficient @ ((values - mean) / scale)) + float(
        model.get("intercept") or 0.0
    )
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def aggregate_cluster_team(
    probabilities: Sequence[float],
    *,
    class_ids: tuple[str, str],
    minimum_observations: int,
    minimum_probability: float,
    minimum_consensus: float,
) -> tuple[str | None, float | None, float | None]:
    """Require multiple confident sample votes before assigning a team."""

    values = [float(value) for value in probabilities]
    if minimum_observations < 1:
        raise ValueError("minimum observations must be positive")
    if not 0.5 < minimum_probability < 1.0:
        raise ValueError("minimum probability must be in (0.5, 1.0)")
    if not 0.5 <= minimum_consensus <= 1.0:
        raise ValueError("minimum consensus must be in [0.5, 1.0]")
    if len(values) < minimum_observations:
        return None, None, None
    votes = [
        1 if value >= minimum_probability else 0
        for value in values
        if value >= minimum_probability or value <= 1.0 - minimum_probability
    ]
    if not votes:
        return None, 0.0, float(np.mean(values))
    counts = np.bincount(votes, minlength=2)
    winning_class = int(np.argmax(counts))
    consensus = float(counts[winning_class] / len(values))
    mean_probability = float(np.mean(values))
    if consensus < minimum_consensus:
        return None, consensus, mean_probability
    return str(class_ids[winning_class]), consensus, mean_probability


def select_uniform_team_candidates(
    candidate_payload: dict[str, Any],
    context_payload: dict[str, Any],
    *,
    team_id: str,
) -> dict[str, Any]:
    """Seal candidates whose face-anchored uniform consensus matches a team."""

    candidate_sha256 = _validate_manifest(
        candidate_payload,
        schema_version="agu.face-enrollment-candidates.v1",
    )
    context_sha256 = _validate_manifest(
        context_payload,
        schema_version="agu.face-uniform-team-context.v1",
    )
    if context_payload.get("source_candidate_manifest_sha256") != candidate_sha256:
        raise ValueError("uniform team context does not match candidate manifest")
    normalized_team_id = str(team_id).strip()
    if not normalized_team_id:
        raise ValueError("uniform team selection requires a team ID")
    contexts = {
        str(row.get("cluster_id") or ""): row
        for row in context_payload.get("clusters") or []
    }
    output = {
        "schema_version": "agu.face-enrollment-candidates.v1",
        "benchmark_disjoint": True,
        "producer": "agu_face_anchored_uniform_team_subset",
        "model_id": candidate_payload.get("model_id"),
        "sources": list(candidate_payload.get("sources") or []),
        "config": {
            "selection": "face_anchored_uniform_team_consensus",
            "source_candidate_manifest_sha256": candidate_sha256,
            "source_config": dict(candidate_payload.get("config") or {}),
            "uniform_team_selection": {
                "team_id": normalized_team_id,
                "source_context_manifest_sha256": context_sha256,
            },
        },
        "clusters": [
            dict(cluster)
            for cluster in candidate_payload.get("clusters") or []
            if contexts.get(str(cluster.get("cluster_id") or ""), {}).get(
                "team_id"
            )
            == normalized_team_id
        ],
        "manifest_sha256": "",
    }
    output["manifest_sha256"] = _canonical_sha256(output)
    return output


def _new_pipeline():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.2,
            class_weight="balanced",
            max_iter=2000,
            random_state=0,
        ),
    )


def _validate_manifest(payload: dict[str, Any], *, schema_version: str) -> str:
    if payload.get("schema_version") != schema_version:
        raise ValueError(f"unsupported manifest schema: {payload.get('schema_version')}")
    if payload.get("benchmark_disjoint") is not True:
        raise ValueError("uniform team inputs must be benchmark-disjoint")
    manifest_sha256 = str(payload.get("manifest_sha256") or "")
    if manifest_sha256 != _canonical_sha256({**payload, "manifest_sha256": ""}):
        raise ValueError("uniform team manifest hash mismatch")
    return manifest_sha256


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
