"""Training-only MUVS source-event representation screening."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.analysis.shot_validity_scene_state import (
    SCENE_BACKBONE,
    SCENE_EMBEDDING_DIMENSION,
)

MUVS_EVENT_STATE_EMBEDDING_SCHEMA = "agu.muvs-event-state-embeddings.v1"
MUVS_FREE_THROW_SOURCE_SCREEN_SCHEMA = "agu.muvs-free-throw-source-screen.v1"
_SHA256_LENGTH = 64
_FEATURE_CANDIDATES = ("mean", "concat", "mean_abs_delta")


def seal_muvs_event_state_embeddings(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["schema_version"] = MUVS_EVENT_STATE_EMBEDDING_SCHEMA
    artifact["purpose"] = "offline_muvs_source_event_state_pretraining"
    artifact["runtime_consumable"] = False
    artifact["codex_runtime_answer_used"] = False
    _validate_embeddings(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_muvs_event_state_embeddings(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_embeddings(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("MUVS event-state embedding hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def screen_muvs_free_throw_source(
    embedding_artifact: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Run deterministic leave-one-event-out free-throw screening."""

    payloads = [embedding_artifact] if isinstance(embedding_artifact, Mapping) else list(embedding_artifact)
    if not payloads:
        raise ValueError("MUVS source screen requires embeddings")
    embeddings = [verify_muvs_event_state_embeddings(payload) for payload in payloads]
    rows_by_frame_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for artifact in embeddings:
        for row in artifact["examples"]:
            if row["state"] == "uncertain":
                continue
            key = (
                str(row["frame_before_sha256"]),
                str(row["frame_after_sha256"]),
            )
            candidate = {
                **row,
                "_embedding_artifact_sha256": artifact["artifact_sha256"],
            }
            previous = rows_by_frame_pair.get(key)
            if previous is not None:
                if (
                    previous["event_id"] != candidate["event_id"]
                    or previous["state"] != candidate["state"]
                    or previous["embeddings"] != candidate["embeddings"]
                ):
                    raise ValueError("duplicate MUVS source frames disagree")
                continue
            rows_by_frame_pair[key] = candidate
    rows = list(rows_by_frame_pair.values())
    labels = np.asarray(
        [row["state"] == "free_throw_setup" for row in rows],
        dtype=np.int64,
    )
    groups = np.asarray([str(row["event_id"]) for row in rows])
    if len(set(labels.tolist())) != 2 or len(set(groups.tolist())) < 3:
        raise ValueError("MUVS source screen requires both classes and events")

    candidate_results = []
    for candidate in _FEATURE_CANDIDATES:
        features = _feature_matrix(rows, candidate=candidate)
        scores = np.zeros(len(rows), dtype=np.float64)
        fold_rows = []
        splitter = LeaveOneGroupOut()
        for train, test in splitter.split(features, labels, groups):
            if len(set(labels[train].tolist())) != 2:
                raise ValueError("each MUVS source fold must retain both training classes")
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
            fold_rows.append(
                _fold_metrics(
                    event_id=str(groups[test][0]),
                    labels=labels[test],
                    scores=fold_scores,
                )
            )

        predictions = scores >= 0.0
        positive_recall = _class_recall(
            labels,
            predictions,
            positive_class=1,
        )
        negative_specificity = _class_recall(
            labels,
            predictions,
            positive_class=0,
        )
        positive_event_recalls = [float(row["positive_recall"]) for row in fold_rows if row["positive_examples"] > 0]
        negative_event_specificities = [
            float(row["negative_specificity"]) for row in fold_rows if row["negative_examples"] > 0
        ]
        candidate_results.append(
            {
                "feature": candidate,
                "classifier": ("StandardScaler+balanced_LogisticRegression_C1"),
                "balanced_accuracy": float((positive_recall + negative_specificity) / 2.0),
                "roc_auc": float(roc_auc_score(labels, scores)),
                "positive_recall": positive_recall,
                "negative_specificity": negative_specificity,
                "worst_positive_event_recall": min(positive_event_recalls),
                "worst_negative_event_specificity": min(negative_event_specificities),
                "folds": fold_rows,
                "oof_predictions": [
                    {
                        "sample_id": str(row["sample_id"]),
                        "event_id": str(row["event_id"]),
                        "embedding_artifact_sha256": str(row["_embedding_artifact_sha256"]),
                        "label": bool(label),
                        "score": float(score),
                        "prediction": bool(score >= 0.0),
                    }
                    for row, label, score in zip(
                        rows,
                        labels,
                        scores,
                        strict=True,
                    )
                ],
            }
        )

    best = max(
        candidate_results,
        key=lambda row: (
            float(row["balanced_accuracy"]),
            float(row["worst_positive_event_recall"]),
            float(row["roc_auc"]),
            -_FEATURE_CANDIDATES.index(str(row["feature"])),
        ),
    )
    best_summary = {
        key: best[key]
        for key in (
            "feature",
            "classifier",
            "balanced_accuracy",
            "roc_auc",
            "positive_recall",
            "negative_specificity",
            "worst_positive_event_recall",
            "worst_negative_event_specificity",
        )
    }
    accepted = bool(
        float(best["balanced_accuracy"]) >= 0.85
        and float(best["worst_positive_event_recall"]) >= 0.5
        and float(best["negative_specificity"]) >= 0.85
    )
    artifact: dict[str, Any] = {
        "schema_version": MUVS_FREE_THROW_SOURCE_SCREEN_SCHEMA,
        "purpose": "offline_source_feature_acceptance_only",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "embedding_artifact_sha256s": [artifact["artifact_sha256"] for artifact in embeddings],
        "resolved_examples": len(rows),
        "event_count": len(set(groups.tolist())),
        "positive_examples": int(labels.sum()),
        "negative_examples": int((labels == 0).sum()),
        "protocol": {
            "grouping": "leave_one_event_out",
            "threshold": "fixed_zero_logistic_margin",
            "candidate_features": list(_FEATURE_CANDIDATES),
            "selection_data": "MUVS_source_only",
            "acceptance_metric": "source_oof_balanced_accuracy",
            "acceptance_threshold": 0.85,
            "minimum_positive_event_recall": 0.5,
            "minimum_negative_specificity": 0.85,
        },
        "candidates": candidate_results,
        "best_candidate": best_summary,
        "accepted": accepted,
    }
    _validate_screen(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_muvs_free_throw_source_screen(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_screen(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("MUVS free-throw source screen hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _feature_matrix(
    rows: list[Mapping[str, Any]],
    *,
    candidate: str,
) -> np.ndarray:
    features = []
    for row in rows:
        values = np.asarray(row["embeddings"], dtype=np.float32)
        before, after = values
        mean = (before + after) / 2.0
        if candidate == "mean":
            feature = mean
        elif candidate == "concat":
            feature = np.concatenate((before, after))
        elif candidate == "mean_abs_delta":
            feature = np.concatenate((mean, np.abs(after - before)))
        else:
            raise ValueError("unsupported MUVS feature candidate")
        features.append(feature)
    return np.stack(features)


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
        "positive_recall": (_class_recall(labels, predictions, positive_class=1) if positives else None),
        "negative_specificity": (_class_recall(labels, predictions, positive_class=0) if negatives else None),
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


def _validate_embeddings(artifact: Mapping[str, Any]) -> None:
    examples = artifact.get("examples")
    if (
        artifact.get("schema_version") != MUVS_EVENT_STATE_EMBEDDING_SCHEMA
        or artifact.get("purpose") != "offline_muvs_source_event_state_pretraining"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("backbone") != SCENE_BACKBONE
        or int(artifact.get("embedding_dimension", 0)) != SCENE_EMBEDDING_DIMENSION
        or not isinstance(examples, list)
        or not examples
    ):
        raise ValueError("invalid MUVS event-state embedding artifact")
    for key in (
        "plan_sha256",
        "frames_sha256",
        "review_sha256",
        "backbone_sha256",
    ):
        _require_sha256(artifact.get(key))
    sample_ids = []
    for row in examples:
        values = np.asarray(row.get("embeddings"), dtype=np.float64)
        if (
            set(row)
            != {
                "sample_id",
                "event_id",
                "split",
                "state",
                "frame_before_sha256",
                "frame_after_sha256",
                "embeddings",
            }
            or not str(row.get("sample_id") or "")
            or not str(row.get("event_id") or "")
            or row.get("split") not in {"train", "test"}
            or row.get("state")
            not in {
                "live_play",
                "free_throw_setup",
                "dead_ball_timeout",
                "uncertain",
            }
            or values.shape != (2, SCENE_EMBEDDING_DIMENSION)
            or not np.isfinite(values).all()
        ):
            raise ValueError("invalid MUVS event-state embedding example")
        _require_sha256(row.get("frame_before_sha256"))
        _require_sha256(row.get("frame_after_sha256"))
        sample_ids.append(str(row["sample_id"]))
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate MUVS event-state embedding sample")


def _validate_screen(artifact: Mapping[str, Any]) -> None:
    candidates = artifact.get("candidates")
    best = artifact.get("best_candidate")
    protocol = artifact.get("protocol")
    if (
        artifact.get("schema_version") != MUVS_FREE_THROW_SOURCE_SCREEN_SCHEMA
        or artifact.get("purpose") != "offline_source_feature_acceptance_only"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or not isinstance(protocol, Mapping)
        or protocol.get("grouping") != "leave_one_event_out"
        or not isinstance(candidates, list)
        or [row.get("feature") for row in candidates] != list(_FEATURE_CANDIDATES)
        or not isinstance(best, Mapping)
        or best.get("feature") not in _FEATURE_CANDIDATES
        or not isinstance(artifact.get("accepted"), bool)
        or int(artifact.get("resolved_examples", 0)) < 1
        or int(artifact.get("event_count", 0)) < 3
    ):
        raise ValueError("invalid MUVS free-throw source screen")
    embedding_hashes = artifact.get("embedding_artifact_sha256s")
    if not isinstance(embedding_hashes, list) or not embedding_hashes:
        raise ValueError("MUVS source screen requires embedding hashes")
    for value in embedding_hashes:
        _require_sha256(value)
    for row in candidates:
        for key in (
            "balanced_accuracy",
            "roc_auc",
            "positive_recall",
            "negative_specificity",
            "worst_positive_event_recall",
            "worst_negative_event_specificity",
        ):
            value = float(row.get(key, math.nan))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("invalid MUVS source screen metric")


def _require_sha256(value: object) -> str:
    text = str(value or "")
    if len(text) != _SHA256_LENGTH or any(character not in "0123456789abcdef" for character in text):
        raise ValueError("invalid MUVS source SHA-256")
    return text


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
