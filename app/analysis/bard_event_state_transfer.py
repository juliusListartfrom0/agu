"""BARD-to-AGU event-state embedding transfer screening."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.analysis.causal_phase_dense_temporal import (
    verify_dense_causal_frame_embeddings,
)

BARD_FRAME_EMBEDDING_SCHEMA = "agu.bard-frame-embeddings.v1"
BARD_TRANSFER_PREDICTION_SCHEMA = "agu.bard-transfer-predictions.v1"
BARD_TRANSFER_EVALUATION_SCHEMA = "agu.bard-transfer-evaluation.v1"
BARD_SAMPLING_PROTOCOL = "normalized_full_clip_5_to_95_percent_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def normalized_frame_indexes(
    frame_count: int,
    *,
    positions: int = 24,
) -> list[int]:
    """Return fixed normalized observations away from codec edge frames."""

    if positions < 2 or frame_count < positions:
        raise ValueError("video has too few frames for normalized sampling")
    indexes = [
        int(round(value))
        for value in np.linspace(
            0.05 * (frame_count - 1),
            0.95 * (frame_count - 1),
            positions,
        )
    ]
    if len(set(indexes)) != positions:
        raise ValueError("normalized frame indexes are not unique")
    return indexes


def sequence_features(
    values: np.ndarray,
    *,
    representation: str,
) -> np.ndarray:
    """Pool variable-length frame sequences without target-label access."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError("sequence features require a finite 2D sequence")
    if not np.isfinite(array).all():
        raise ValueError("sequence features must be finite")
    if representation == "mean":
        return array.mean(axis=0)
    if representation == "mean_std":
        return np.concatenate((array.mean(axis=0), array.std(axis=0)))
    if representation == "quarters":
        return np.concatenate(
            [chunk.mean(axis=0) for chunk in np.array_split(array, 4)]
        )
    raise ValueError("unsupported BARD temporal representation")


def seal_bard_frame_embeddings(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["schema_version"] = BARD_FRAME_EMBEDDING_SCHEMA
    artifact["runtime_consumable"] = False
    _validate_bard_embeddings(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_bard_frame_embeddings(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = json.loads(json.dumps(payload))
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_bard_embeddings(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("BARD frame embedding hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def train_bard_transfer_predictor(
    *,
    bard_embedding_artifact: Mapping[str, Any],
    target_embedding_artifact: Mapping[str, Any],
    configurations: Sequence[Mapping[str, Any]],
    seed: int,
) -> dict[str, Any]:
    """Select only on BARD OOF labels, then emit label-free target predictions."""

    source = verify_bard_frame_embeddings(bard_embedding_artifact)
    target = verify_dense_causal_frame_embeddings(target_embedding_artifact)
    if (
        source["backbone"] != target["backbone"]
        or source["backbone_sha256"] != target["backbone_sha256"]
        or source["embedding_dimension"] != target["embedding_dimension"]
    ):
        raise ValueError("BARD and target embedding backbones do not match")
    configs = _normalize_configurations(configurations)
    source_labels = np.asarray(
        [
            1 if row["event_state"] == "free_throw" else 0
            for row in source["examples"]
        ],
        dtype=np.int64,
    )
    source_groups = np.asarray(
        [str(row["game_id"]) for row in source["examples"]]
    )
    unique_groups = sorted(set(source_groups.tolist()))
    if len(unique_groups) < 4 or set(source_labels.tolist()) != {0, 1}:
        raise ValueError("BARD transfer requires four groups and both labels")
    split_count = min(5, len(unique_groups))
    splitter = StratifiedGroupKFold(
        n_splits=split_count,
        shuffle=True,
        random_state=seed,
    )

    candidates = []
    for config_index, config in enumerate(configs):
        source_features = _feature_matrix(
            source["examples"],
            representation=config["representation"],
        )
        probabilities = np.zeros(len(source_labels), dtype=np.float64)
        folds = []
        for fold_index, (train_indexes, held_indexes) in enumerate(
            splitter.split(
                source_features,
                source_labels,
                groups=source_groups,
            )
        ):
            model = _fit_model(
                source_features[train_indexes],
                source_labels[train_indexes],
                c=config["c"],
                seed=seed + config_index * 100 + fold_index,
            )
            held_probabilities = model.predict_proba(
                source_features[held_indexes]
            )[:, 1]
            probabilities[held_indexes] = held_probabilities
            folds.append(
                {
                    "held_groups": sorted(
                        set(source_groups[held_indexes].tolist())
                    ),
                    "training_groups": sorted(
                        set(source_groups[train_indexes].tolist())
                    ),
                }
            )
        threshold, metrics = _select_threshold(
            source_labels,
            probabilities,
        )
        candidates.append(
            {
                "configuration": config,
                "threshold": threshold,
                "metrics": metrics,
                "folds": folds,
                "probabilities": probabilities.tolist(),
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            row["metrics"]["balanced_accuracy"],
            row["metrics"]["f1"],
            -_representation_rank(
                row["configuration"]["representation"]
            ),
            -row["configuration"]["c"],
        ),
    )
    representation = selected["configuration"]["representation"]
    source_features = _feature_matrix(
        source["examples"],
        representation=representation,
    )
    target_features = _feature_matrix(
        target["examples"],
        representation=representation,
    )
    final_model = _fit_model(
        source_features,
        source_labels,
        c=selected["configuration"]["c"],
        seed=seed + 10_000,
    )
    target_probabilities = final_model.predict_proba(target_features)[:, 1]
    threshold = float(selected["threshold"])
    target_predictions = (target_probabilities >= threshold).astype(np.int64)

    artifact: dict[str, Any] = {
        "schema_version": BARD_TRANSFER_PREDICTION_SCHEMA,
        "purpose": "training_only_external_event_state_transfer",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "target_labels_opened_during_training": False,
        "bard_embedding_artifact_sha256": source["artifact_sha256"],
        "target_embedding_artifact_sha256": target["artifact_sha256"],
        "backbone": source["backbone"],
        "backbone_sha256": source["backbone_sha256"],
        "embedding_dimension": source["embedding_dimension"],
        "protocol": {
            "source_selection": "stratified_group_kfold_bard_games",
            "source_label": "live_field_goal_vs_free_throw",
            "target_prediction": "label_free_before_delayed_evaluation",
            "decision_threshold_source": "bard_oof_only",
        },
        "configurations": [
            {
                "configuration": row["configuration"],
                "threshold": row["threshold"],
                "metrics": row["metrics"],
                "folds": row["folds"],
            }
            for row in candidates
        ],
        "selected_configuration": selected["configuration"],
        "decision_threshold": threshold,
        "source_oof_metrics": selected["metrics"],
        "predictions": [
            {
                "phase_review_id": row["phase_review_id"],
                "source_video_sha256": row["source_video_sha256"],
                "event_id": row["event_id"],
                "prediction": int(prediction),
                "probability": float(probability),
            }
            for row, prediction, probability in zip(
                target["examples"],
                target_predictions,
                target_probabilities,
                strict=True,
            )
        ],
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return _verify_prediction_artifact(artifact)


def evaluate_bard_transfer_predictions(
    *,
    prediction_artifact: Mapping[str, Any],
    reference_predictions: Sequence[Mapping[str, Any]],
    reference_artifact_sha256: str,
    acceptance_threshold: float,
) -> dict[str, Any]:
    """Open frozen development labels only after exact prediction sealing."""

    predictions = _verify_prediction_artifact(prediction_artifact)
    _require_sha256(reference_artifact_sha256, "reference artifact")
    if not 0.5 <= acceptance_threshold <= 1.0:
        raise ValueError("BARD transfer acceptance threshold is invalid")
    predicted = {
        _prediction_key(row): row for row in predictions["predictions"]
    }
    references = {}
    for row in reference_predictions:
        key = _prediction_key(row)
        if key in references or row.get("target") not in {0, 1}:
            raise ValueError("BARD transfer reference row is invalid")
        references[key] = row
    if set(predicted) != set(references):
        raise ValueError(
            "BARD transfer reference labels must exactly match predictions"
        )
    ordered_keys = sorted(predicted)
    labels = np.asarray(
        [int(references[key]["target"]) for key in ordered_keys],
        dtype=np.int64,
    )
    values = np.asarray(
        [int(predicted[key]["prediction"]) for key in ordered_keys],
        dtype=np.int64,
    )
    probabilities = np.asarray(
        [float(predicted[key]["probability"]) for key in ordered_keys],
        dtype=np.float64,
    )
    metrics = _metrics(labels, values, probabilities)
    groups = sorted({key[0] for key in ordered_keys})
    per_group = []
    for group in groups:
        indexes = [
            index
            for index, key in enumerate(ordered_keys)
            if key[0] == group
        ]
        group_labels = labels[indexes]
        group_values = values[indexes]
        group_probabilities = probabilities[indexes]
        per_group.append(
            {
                "source_video_sha256": group,
                "example_count": len(indexes),
                "metrics": _metrics(
                    group_labels,
                    group_values,
                    group_probabilities,
                ),
            }
        )
    metrics["worst_game_balanced_accuracy"] = min(
        row["metrics"]["balanced_accuracy"] for row in per_group
    )
    accepted = (
        metrics["balanced_accuracy"] >= acceptance_threshold
        and metrics["f1"] >= acceptance_threshold
        and metrics["worst_game_balanced_accuracy"] >= 0.8
    )
    artifact: dict[str, Any] = {
        "schema_version": BARD_TRANSFER_EVALUATION_SCHEMA,
        "purpose": "training_only_external_event_state_transfer_evaluation",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "prediction_artifact_sha256": predictions["artifact_sha256"],
        "reference_artifact_sha256": reference_artifact_sha256,
        "target_example_count": len(ordered_keys),
        "target_group_count": len(groups),
        "metrics": metrics,
        "per_game": per_group,
        "acceptance_gate": {
            "minimum_balanced_accuracy": acceptance_threshold,
            "minimum_f1": acceptance_threshold,
            "minimum_worst_game_balanced_accuracy": 0.8,
        },
        "accepted": accepted,
        "promotion_decision": (
            "eligible_for_checkpoint_training"
            if accepted
            else "rejected_below_external_transfer_gate"
        ),
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def _validate_bard_embeddings(artifact: Mapping[str, Any]) -> None:
    if (
        artifact.get("schema_version") != BARD_FRAME_EMBEDDING_SCHEMA
        or artifact.get("purpose") != "event_state_pretraining_only"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("sampling_protocol") != BARD_SAMPLING_PROTOCOL
    ):
        raise ValueError("BARD frame embedding boundary is invalid")
    _require_sha256(artifact.get("subset_sha256"), "BARD subset")
    _require_sha256(artifact.get("backbone_sha256"), "BARD backbone")
    dimension = int(artifact.get("embedding_dimension") or 0)
    positions = int(artifact.get("positions") or 0)
    if dimension < 1 or positions < 2 or not artifact.get("backbone"):
        raise ValueError("BARD frame embedding shape is invalid")
    examples = artifact.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("BARD frame embeddings require examples")
    identities: set[str] = set()
    for row in examples:
        clip_id = str(row.get("clip_id") or "")
        values = np.asarray(row.get("embeddings"), dtype=np.float32)
        indexes = row.get("frame_indexes")
        if (
            not clip_id
            or clip_id in identities
            or row.get("event_state")
            not in {"live_field_goal", "free_throw"}
            or not str(row.get("game_id") or "")
            or not isinstance(indexes, list)
            or len(indexes) != positions
            or len(set(indexes)) != positions
            or values.shape != (positions, dimension)
            or not np.isfinite(values).all()
        ):
            raise ValueError("BARD frame embedding example is invalid")
        identities.add(clip_id)


def _normalize_configurations(
    values: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for value in values:
        row = {
            "representation": str(value.get("representation") or ""),
            "c": float(value.get("c") or 0.0),
        }
        if (
            row["representation"] not in {"mean", "mean_std", "quarters"}
            or not 0.0001 <= row["c"] <= 100.0
        ):
            raise ValueError("BARD transfer configuration is invalid")
        rows.append(row)
    if not rows or len({_canonical_sha256(row) for row in rows}) != len(rows):
        raise ValueError("BARD transfer configurations are empty or duplicate")
    return rows


def _feature_matrix(
    examples: Sequence[Mapping[str, Any]],
    *,
    representation: str,
) -> np.ndarray:
    values = np.stack(
        [
            sequence_features(
                np.asarray(row["embeddings"], dtype=np.float32),
                representation=representation,
            )
            for row in examples
        ]
    )
    if not np.isfinite(values).all():
        raise ValueError("BARD transfer feature matrix is invalid")
    return values


def _fit_model(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    c: float,
    seed: int,
) -> Any:
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("BARD transfer fold lacks one label")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            solver="liblinear",
        ),
    )
    return model.fit(features, labels)


def _select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, dict[str, float]]:
    candidates = []
    for threshold in np.linspace(0.2, 0.8, 13):
        predictions = (probabilities >= threshold).astype(np.int64)
        metrics = _metrics(labels, predictions, probabilities)
        candidates.append((float(threshold), metrics))
    return max(
        candidates,
        key=lambda item: (
            item[1]["balanced_accuracy"],
            item[1]["f1"],
            -abs(item[0] - 0.5),
        ),
    )


def _metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(labels, probabilities))
            if len(set(labels.tolist())) == 2
            else 0.5
        ),
    }


def _verify_prediction_artifact(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = json.loads(json.dumps(value))
    claimed = str(artifact.pop("artifact_sha256", ""))
    if claimed != _canonical_sha256(artifact):
        raise ValueError("BARD transfer prediction hash mismatch")
    if (
        artifact.get("schema_version") != BARD_TRANSFER_PREDICTION_SCHEMA
        or artifact.get("purpose")
        != "training_only_external_event_state_transfer"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("target_labels_opened_during_training") is not False
    ):
        raise ValueError("BARD transfer prediction boundary is invalid")
    for field in (
        "bard_embedding_artifact_sha256",
        "target_embedding_artifact_sha256",
        "backbone_sha256",
    ):
        _require_sha256(artifact.get(field), field)
    rows = artifact.get("predictions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("BARD transfer predictions are missing")
    keys = set()
    for row in rows:
        key = _prediction_key(row)
        if (
            key in keys
            or row.get("prediction") not in {0, 1}
            or "target" in row
            or not 0.0 <= float(row.get("probability", -1.0)) <= 1.0
        ):
            raise ValueError("BARD transfer prediction row is invalid")
        keys.add(key)
    artifact["artifact_sha256"] = claimed
    return artifact


def _prediction_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    key = (
        str(row.get("source_video_sha256") or ""),
        str(row.get("event_id") or ""),
        str(row.get("phase_review_id") or ""),
    )
    if not _SHA256.fullmatch(key[0]) or not key[1] or not key[2]:
        raise ValueError("BARD transfer prediction identity is invalid")
    return key


def _representation_rank(value: str) -> int:
    return {"mean": 0, "mean_std": 1, "quarters": 2}[value]


def _require_sha256(value: object, label: str) -> None:
    if not _SHA256.fullmatch(str(value or "")):
        raise ValueError(f"{label} SHA-256 is invalid")


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
