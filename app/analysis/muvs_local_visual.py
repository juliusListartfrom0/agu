"""Training-only local DINOv2 view screening on reviewed MUVS frames."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MUVS_LOCAL_VISUAL_EMBEDDINGS_SCHEMA = "agu.muvs-local-visual-embeddings.v1"
MUVS_LOCAL_VISUAL_SCREEN_SCHEMA = "agu.muvs-local-visual-screen.v1"
MUVS_LOCAL_VISUAL_BACKBONE = "facebook/dinov2-small"
MUVS_LOCAL_VISUAL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
MUVS_LOCAL_VISUAL_DIMENSION = 384
MUVS_LOCAL_VISUAL_STATES = {
    "live_play",
    "free_throw_setup",
    "dead_ball_timeout",
    "uncertain",
}
MUVS_LOCAL_VISUAL_VIEWS = [
    {"name": "left_65", "x_range": [0.0, 0.65]},
    {"name": "center_65", "x_range": [0.175, 0.825]},
    {"name": "right_65", "x_range": [0.35, 1.0]},
]
_FEATURE_CANDIDATES = (
    "selected_context",
    "selected_directed_temporal",
    "symmetric_context",
)
_LEGACY_FEATURE_CANDIDATES = ("selected_context", "symmetric_context")
_PCA_COMPONENTS = 8
_SHA256_LENGTH = 64


def seal_muvs_local_visual_embeddings(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["schema_version"] = MUVS_LOCAL_VISUAL_EMBEDDINGS_SCHEMA
    artifact["purpose"] = "offline_muvs_source_local_visual_pretraining"
    artifact["runtime_consumable"] = False
    artifact["codex_runtime_answer_used"] = False
    _validate_embeddings(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_muvs_local_visual_embeddings(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_embeddings(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("MUVS local visual embedding hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def screen_muvs_local_visual(
    embedding_artifact: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payloads = (
        [embedding_artifact]
        if isinstance(embedding_artifact, Mapping)
        else list(embedding_artifact)
    )
    if not payloads:
        raise ValueError("MUVS local visual screen requires embeddings")
    artifacts = [
        verify_muvs_local_visual_embeddings(payload) for payload in payloads
    ]
    rows_by_frame_pair: dict[tuple[str, str], dict[str, Any]] = {}
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
                "_embedding_artifact_sha256": artifact["artifact_sha256"],
            }
            previous = rows_by_frame_pair.get(key)
            if previous is not None:
                if (
                    previous["event_id"] != candidate["event_id"]
                    or previous["state"] != candidate["state"]
                    or previous["selected_view_indices"]
                    != candidate["selected_view_indices"]
                    or previous["embeddings"] != candidate["embeddings"]
                ):
                    raise ValueError("duplicate MUVS local views disagree")
                continue
            rows_by_frame_pair[key] = candidate

    rows = list(rows_by_frame_pair.values())
    labels = np.asarray(
        [row["state"] == "free_throw_setup" for row in rows],
        dtype=np.int64,
    )
    groups = np.asarray([str(row["event_id"]) for row in rows])
    if len(set(labels.tolist())) != 2 or len(set(groups.tolist())) < 3:
        raise ValueError("MUVS local visual screen requires both classes and events")

    candidate_results = [
        _screen_candidate(
            rows,
            labels=labels,
            groups=groups,
            candidate=candidate,
        )
        for candidate in _FEATURE_CANDIDATES
    ]
    best = max(
        candidate_results,
        key=lambda row: (
            float(row["balanced_accuracy"]),
            float(row["worst_positive_event_recall"]),
            float(row["roc_auc"]),
            -_FEATURE_CANDIDATES.index(str(row["feature"])),
        ),
    )
    summary_keys = (
        "feature",
        "classifier",
        "feature_dimension",
        "balanced_accuracy",
        "roc_auc",
        "positive_recall",
        "negative_specificity",
        "worst_positive_event_recall",
        "worst_negative_event_specificity",
    )
    accepted = bool(
        float(best["balanced_accuracy"]) >= 0.85
        and float(best["worst_positive_event_recall"]) >= 0.5
        and float(best["negative_specificity"]) >= 0.85
    )
    artifact: dict[str, Any] = {
        "schema_version": MUVS_LOCAL_VISUAL_SCREEN_SCHEMA,
        "purpose": "offline_source_feature_acceptance_only",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "embedding_artifact_sha256s": [
            artifact["artifact_sha256"] for artifact in artifacts
        ],
        "resolved_examples": len(rows),
        "event_count": len(set(groups.tolist())),
        "positive_examples": int(labels.sum()),
        "negative_examples": int((labels == 0).sum()),
        "protocol": {
            "grouping": "leave_one_event_out",
            "threshold": "fixed_zero_logistic_margin",
            "candidate_features": list(_FEATURE_CANDIDATES),
            "pca_components": _PCA_COMPONENTS,
            "selection_data": "MUVS_source_only",
            "acceptance_metric": "source_oof_balanced_accuracy",
            "acceptance_threshold": 0.85,
            "minimum_positive_event_recall": 0.5,
            "minimum_negative_specificity": 0.85,
        },
        "candidates": candidate_results,
        "best_candidate": {key: best[key] for key in summary_keys},
        "accepted": accepted,
    }
    _validate_screen(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_muvs_local_visual_screen(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_screen(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("MUVS local visual screen hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def choose_muvs_local_view(frame: Mapping[str, Any]) -> int:
    """Choose left/center/right context from detector evidence only."""

    detections = frame["detections"]
    rims = [row for row in detections if row["object_type"] == "rim"]
    if rims:
        rim = max(rims, key=lambda row: row["confidence"])
        x1, _, x2, _ = rim["bbox"]
        center_x = (x1 + x2) / 2.0
        if center_x < 0.42:
            return 0
        if center_x > 0.58:
            return 2
        return 1
    player_centers = [
        (row["bbox"][0] + row["bbox"][2]) / 2.0
        for row in detections
        if row["object_type"] == "player"
    ]
    counts = [
        sum(start <= center <= end for center in player_centers)
        for start, end in (
            (0.0, 0.65),
            (0.175, 0.825),
            (0.35, 1.0),
        )
    ]
    return max(range(3), key=lambda index: (counts[index], index == 1))


def _screen_candidate(
    rows: list[Mapping[str, Any]],
    *,
    labels: np.ndarray,
    groups: np.ndarray,
    candidate: str,
) -> dict[str, Any]:
    features = _feature_matrix(rows, candidate=candidate)
    scores = np.zeros(len(rows), dtype=np.float64)
    fold_rows = []
    for train, test in LeaveOneGroupOut().split(features, labels, groups):
        if len(set(labels[train].tolist())) != 2:
            raise ValueError("each MUVS local visual fold must retain both classes")
        components = min(
            _PCA_COMPONENTS,
            len(train) - 1,
            features.shape[1],
        )
        model = make_pipeline(
            StandardScaler(),
            PCA(
                n_components=components,
                whiten=True,
                random_state=0,
            ),
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
                pca_components=components,
            )
        )

    predictions = scores >= 0.0
    positive_recall = _class_recall(labels, predictions, positive_class=1)
    negative_specificity = _class_recall(
        labels,
        predictions,
        positive_class=0,
    )
    positive_event_recalls = [
        float(row["positive_recall"])
        for row in fold_rows
        if row["positive_examples"] > 0
    ]
    negative_event_specificities = [
        float(row["negative_specificity"])
        for row in fold_rows
        if row["negative_examples"] > 0
    ]
    return {
        "feature": candidate,
        "classifier": (
            "StandardScaler+PCA8_whiten+balanced_LogisticRegression_C1"
        ),
        "feature_dimension": int(features.shape[1]),
        "balanced_accuracy": float(
            (positive_recall + negative_specificity) / 2.0
        ),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "positive_recall": positive_recall,
        "negative_specificity": negative_specificity,
        "worst_positive_event_recall": min(positive_event_recalls),
        "worst_negative_event_specificity": min(
            negative_event_specificities
        ),
        "folds": fold_rows,
        "oof_predictions": [
            {
                "sample_id": str(row["sample_id"]),
                "event_id": str(row["event_id"]),
                "embedding_artifact_sha256": str(
                    row["_embedding_artifact_sha256"]
                ),
                "label": bool(label),
                "score": float(score),
                "prediction": bool(score >= 0.0),
            }
            for row, label, score in zip(rows, labels, scores, strict=True)
        ],
    }


def _feature_matrix(
    rows: list[Mapping[str, Any]],
    *,
    candidate: str,
) -> np.ndarray:
    output = []
    for row in rows:
        values = np.asarray(row["embeddings"], dtype=np.float32)
        if candidate == "selected_context":
            indices = row["selected_view_indices"]
            before = values[0, int(indices[0])]
            after = values[1, int(indices[1])]
        elif candidate == "selected_directed_temporal":
            indices = row["selected_view_indices"]
            before = values[0, int(indices[0])]
            after = values[1, int(indices[1])]
        elif candidate == "symmetric_context":
            before = _symmetric_frame(values[0])
            after = _symmetric_frame(values[1])
        else:
            raise ValueError("unsupported MUVS local visual feature")
        if candidate == "selected_directed_temporal":
            output.append(
                np.concatenate(
                    (
                        (before + after) / 2.0,
                        after - before,
                        np.abs(after - before),
                    )
                )
            )
        else:
            output.append(
                np.concatenate(
                    (
                        (before + after) / 2.0,
                        np.abs(after - before),
                    )
                )
            )
    return np.stack(output)


def _symmetric_frame(views: np.ndarray) -> np.ndarray:
    left, center, right = views
    return np.concatenate(
        (
            (left + right) / 2.0,
            np.abs(left - right),
            center,
        )
    )


def _fold_metrics(
    *,
    event_id: str,
    labels: np.ndarray,
    scores: np.ndarray,
    pca_components: int,
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
        "pca_components": pca_components,
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
        artifact.get("schema_version")
        != MUVS_LOCAL_VISUAL_EMBEDDINGS_SCHEMA
        or artifact.get("purpose")
        != "offline_muvs_source_local_visual_pretraining"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("backbone") != MUVS_LOCAL_VISUAL_BACKBONE
        or artifact.get("backbone_revision") != MUVS_LOCAL_VISUAL_REVISION
        or int(artifact.get("embedding_dimension", 0))
        != MUVS_LOCAL_VISUAL_DIMENSION
        or artifact.get("views") != MUVS_LOCAL_VISUAL_VIEWS
        or not isinstance(examples, list)
        or not examples
    ):
        raise ValueError("invalid MUVS local visual embedding artifact")
    for key in ("detection_artifact_sha256", "backbone_sha256"):
        _require_sha256(artifact.get(key))
    sample_ids = []
    for row in examples:
        values = np.asarray(row.get("embeddings"), dtype=np.float64)
        indices = row.get("selected_view_indices")
        if (
            set(row)
            != {
                "sample_id",
                "event_id",
                "split",
                "state",
                "frame_before_sha256",
                "frame_after_sha256",
                "selected_view_indices",
                "embeddings",
            }
            or not str(row.get("sample_id") or "")
            or not str(row.get("event_id") or "")
            or row.get("split") not in {"train", "test"}
            or row.get("state") not in MUVS_LOCAL_VISUAL_STATES
            or not isinstance(indices, list)
            or len(indices) != 2
            or any(
                not isinstance(index, int)
                or isinstance(index, bool)
                or index not in {0, 1, 2}
                for index in indices
            )
            or values.shape
            != (2, 3, MUVS_LOCAL_VISUAL_DIMENSION)
            or not np.isfinite(values).all()
        ):
            raise ValueError("invalid MUVS local visual embedding example")
        _require_sha256(row.get("frame_before_sha256"))
        _require_sha256(row.get("frame_after_sha256"))
        sample_ids.append(str(row["sample_id"]))
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate MUVS local visual embedding sample")


def _validate_screen(artifact: Mapping[str, Any]) -> None:
    candidates = artifact.get("candidates")
    best = artifact.get("best_candidate")
    protocol = artifact.get("protocol")
    candidate_names = (
        tuple(row.get("feature") for row in candidates)
        if isinstance(candidates, list)
        else ()
    )
    if (
        artifact.get("schema_version") != MUVS_LOCAL_VISUAL_SCREEN_SCHEMA
        or artifact.get("purpose") != "offline_source_feature_acceptance_only"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or not isinstance(protocol, Mapping)
        or protocol.get("grouping") != "leave_one_event_out"
        or protocol.get("pca_components") != _PCA_COMPONENTS
        or not isinstance(candidates, list)
        or candidate_names
        not in {_LEGACY_FEATURE_CANDIDATES, _FEATURE_CANDIDATES}
        or protocol.get("candidate_features") != list(candidate_names)
        or not isinstance(best, Mapping)
        or best.get("feature") not in candidate_names
        or not isinstance(artifact.get("accepted"), bool)
        or int(artifact.get("resolved_examples", 0)) < 1
        or int(artifact.get("event_count", 0)) < 3
    ):
        raise ValueError("invalid MUVS local visual screen")
    artifact_hashes = artifact.get("embedding_artifact_sha256s")
    if not isinstance(artifact_hashes, list) or not artifact_hashes:
        raise ValueError("MUVS local visual screen requires artifact hashes")
    for value in artifact_hashes:
        _require_sha256(value)
    for row in candidates:
        if int(row.get("feature_dimension", 0)) < 1:
            raise ValueError("invalid MUVS local visual feature dimension")
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
                raise ValueError("invalid MUVS local visual metric")


def _require_sha256(value: object) -> str:
    text = str(value or "")
    if len(text) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError("invalid MUVS local visual SHA-256")
    return text


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
