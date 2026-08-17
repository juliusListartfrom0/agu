"""Offline E-BARD object-role transfer helpers.

This module deliberately stops at an evidence artifact.  It can fit a small
object-role baseline from the licensed E-BARD crop archive and score crops from
an already-reviewed development game, but it does not expose a runtime model
or a blind-game answer channel.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.analysis.e_bard_object_classification import (
    _SPLIT_MANIFESTS,
    OBJECT_CLASSES,
    _game_id,
    _normalize_image_path,
    _row_label,
)

ROLE_TRANSFER_SCHEMA = "agu.e-bard-role-transfer-screen.v1"
_FEATURE_IMAGE_SIZE = (16, 16)
_HISTOGRAM_BINS = 16


def image_role_features(image_data: bytes | Image.Image) -> np.ndarray:
    """Return a small, deterministic RGB/HSV/edge feature vector.

    The representation is intentionally lightweight and CPU-only.  It is a
    baseline for deciding whether a detector crop looks like a basketball;
    it is not intended as a production VLM embedding.
    """

    if isinstance(image_data, Image.Image):
        image = image_data.convert("RGB")
    else:
        with Image.open(io.BytesIO(bytes(image_data))) as decoded:
            image = decoded.convert("RGB")
    resized = image.resize(_FEATURE_IMAGE_SIZE, Image.Resampling.BILINEAR)
    rgb = np.asarray(resized, dtype=np.float32) / 255.0
    hsv = np.asarray(resized.convert("HSV"), dtype=np.float32) / 255.0
    gray = rgb.mean(axis=2)
    gradient_x = np.diff(gray, axis=1).ravel()
    gradient_y = np.diff(gray, axis=0).ravel()
    moments = np.concatenate((rgb.mean(axis=(0, 1)), rgb.std(axis=(0, 1))))
    histograms = []
    for channel in range(3):
        histogram, _ = np.histogram(
            hsv[:, :, channel],
            bins=_HISTOGRAM_BINS,
            range=(0.0, 1.0),
        )
        histograms.append(histogram.astype(np.float32) / float(hsv.shape[0] * hsv.shape[1]))
    return np.concatenate(
        (
            rgb.ravel(),
            np.asarray(histograms, dtype=np.float32).ravel(),
            moments.astype(np.float32),
            gradient_x.astype(np.float32),
            gradient_y.astype(np.float32),
        )
    ).astype(np.float32, copy=False)


def split_group_indices(
    groups: np.ndarray,
    *,
    holdout_fraction: float = 0.2,
    seed: int = 20260802,
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows deterministically while keeping every group intact."""

    group_array = np.asarray(groups)
    if group_array.ndim != 1 or not len(group_array):
        raise ValueError("groups must be a non-empty one-dimensional array")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between zero and one")
    unique_groups = np.asarray(sorted({str(value) for value in group_array}))
    if len(unique_groups) < 2:
        raise ValueError("at least two groups are required")
    holdout_count = min(
        len(unique_groups) - 1,
        max(1, int(np.ceil(len(unique_groups) * holdout_fraction))),
    )
    rng = np.random.default_rng(seed)
    holdout_groups = set(rng.permutation(unique_groups)[:holdout_count].tolist())
    valid_mask = np.asarray([str(value) in holdout_groups for value in group_array])
    valid_indices = np.flatnonzero(valid_mask)
    train_indices = np.flatnonzero(~valid_mask)
    return train_indices, valid_indices


def choose_precision_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    minimum_precision: float = 0.85,
) -> float:
    """Choose the lowest score threshold meeting a precision gate.

    Ties are resolved in favor of recall.  If no threshold can meet the gate,
    positive infinity is returned, which is a safe reject-all default.
    """

    truth = np.asarray(labels, dtype=bool)
    probabilities = np.asarray(scores, dtype=np.float64)
    if truth.ndim != 1 or probabilities.ndim != 1 or len(truth) != len(probabilities):
        raise ValueError("labels and scores must be one-dimensional with equal length")
    if not len(truth) or not np.isfinite(probabilities).all():
        raise ValueError("labels and scores must be non-empty and finite")
    if not 0.0 < minimum_precision <= 1.0:
        raise ValueError("minimum_precision must be in (0, 1]")
    positives = int(truth.sum())
    if positives == 0:
        return float("inf")
    best: tuple[float, int, int] | None = None
    for threshold in np.sort(np.unique(probabilities))[::-1]:
        predicted = probabilities >= threshold
        true_positive = int(np.logical_and(predicted, truth).sum())
        predicted_positive = int(predicted.sum())
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        if precision + 1e-12 < minimum_precision:
            continue
        recall_numerator = true_positive
        candidate = (float(threshold), recall_numerator, -predicted_positive)
        if best is None or candidate[1:] > best[1:]:
            best = candidate
    return float(best[0]) if best is not None else float("inf")


def binary_gate_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """Return confusion counts and precision/recall for a score threshold."""

    truth = np.asarray(labels, dtype=bool)
    probabilities = np.asarray(scores, dtype=np.float64)
    if truth.ndim != 1 or probabilities.ndim != 1 or len(truth) != len(probabilities):
        raise ValueError("labels and scores must be one-dimensional with equal length")
    predicted = probabilities >= float(threshold)
    tp = int(np.logical_and(predicted, truth).sum())
    fp = int(np.logical_and(predicted, ~truth).sum())
    fn = int(np.logical_and(~predicted, truth).sum())
    tn = int(np.logical_and(~predicted, ~truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "examples": int(len(truth)),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def load_archive_features(
    archive_path: Path,
    *,
    splits: tuple[str, ...] = ("train", "valid", "test"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Decode manifest-backed E-BARD images into features and labels."""

    if not splits or any(split not in _SPLIT_MANIFESTS for split in splits):
        raise ValueError("splits must be a non-empty subset of train/valid/test")
    features: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    split_counts: dict[str, int] = {}
    with zipfile.ZipFile(Path(archive_path)) as archive:
        for split in splits:
            manifest = json.loads(archive.read(_SPLIT_MANIFESTS[split]))
            if not isinstance(manifest, list):
                raise ValueError(f"manifest is not a list: {split}")
            split_counts[split] = len(manifest)
            for row in manifest:
                image_path = _normalize_image_path(row.get("image"))
                label = _row_label(row)
                group = _game_id(image_path)
                if not group or label not in OBJECT_CLASSES:
                    raise ValueError(f"invalid E-BARD row: {image_path}")
                features.append(image_role_features(archive.read(image_path)))
                labels.append(label)
                groups.append(group)
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels),
        np.asarray(groups),
        split_counts,
    )


def fit_role_classifier(features: np.ndarray, labels: np.ndarray) -> Any:
    """Fit a CPU-only multiclass baseline; import sklearn lazily."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("features and labels have incompatible shapes")
    if len(set(y.tolist())) < 2:
        raise ValueError("at least two object classes are required")
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1200,
            random_state=0,
        ),
    ).fit(x, y)


def canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
