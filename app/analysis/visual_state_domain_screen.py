"""Game-disjoint source-assisted screening for reviewed broadcast visual states."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.analysis.basketball51 import (
    BASKETBALL51_LABELS,
    verify_basketball51_embedding_artifact,
)
from app.analysis.pbp_visual_state_review import (
    verify_visual_state_label_corrections,
    verify_visual_state_review_plan,
)
from app.analysis.shot_validity_video_backbone import (
    verify_video_embedding_artifact,
)

VISUAL_STATE_DOMAIN_SCREEN_SCHEMA = "agu.visual-state-domain-screen.v1"
VISUAL_STATE_LABELS = ("field_goal", "free_throw")
DEFAULT_CONFIGURATIONS = (
    {"c": 0.1, "source_weight": 0.0, "pca_components": None},
    {"c": 1.0, "source_weight": 0.0, "pca_components": None},
    {"c": 0.1, "source_weight": 0.25, "pca_components": None},
    {"c": 1.0, "source_weight": 0.25, "pca_components": None},
    {"c": 0.1, "source_weight": 0.5, "pca_components": None},
    {"c": 1.0, "source_weight": 0.5, "pca_components": None},
    {"c": 0.1, "source_weight": 0.25, "pca_components": 32},
    {"c": 1.0, "source_weight": 0.25, "pca_components": 32},
    {"c": 0.1, "source_weight": 0.5, "pca_components": 32},
    {"c": 1.0, "source_weight": 0.5, "pca_components": 32},
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


def build_visual_state_domain_screen(
    *,
    source_embedding_artifact: Mapping[str, Any],
    target_embedding_artifact: Mapping[str, Any],
    base_review_plan: Mapping[str, Any],
    base_label_corrections: Mapping[str, Any],
    followup_review_plan: Mapping[str, Any],
    followup_label_corrections: Mapping[str, Any],
    configurations: Sequence[Mapping[str, Any]] | None = None,
    acceptance_threshold: float = 0.85,
) -> dict[str, Any]:
    """Verify the complete review chain and run a training-only nested screen."""

    source = verify_basketball51_embedding_artifact(source_embedding_artifact)
    target = verify_video_embedding_artifact(target_embedding_artifact)
    base_plan = verify_visual_state_review_plan(base_review_plan)
    base_corrections = verify_visual_state_label_corrections(
        base_label_corrections,
        plan=base_plan,
    )
    followup_plan = verify_visual_state_review_plan(followup_review_plan)
    followup_corrections = verify_visual_state_label_corrections(
        followup_label_corrections,
        plan=followup_plan,
    )

    if (
        followup_plan.get("parent_plan_sha256")
        != base_plan["artifact_sha256"]
        or followup_plan.get("parent_corrections_sha256")
        != base_corrections["artifact_sha256"]
        or followup_plan.get("embedding_artifact_sha256")
        != base_plan["embedding_artifact_sha256"]
        or followup_plan.get("sealed_blind_video_sha256s")
        != base_plan["sealed_blind_video_sha256s"]
        or base_corrections.get("embedding_artifact_sha256")
        != base_plan["embedding_artifact_sha256"]
        or followup_corrections.get("embedding_artifact_sha256")
        != followup_plan["embedding_artifact_sha256"]
    ):
        raise ValueError("visual-state follow-up chain is invalid")
    if (
        source["backbone"] != target["backbone"]
        or source["backbone_sha256"] != target["backbone_sha256"]
        or int(source["embedding_dimension"])
        != int(target["embedding_dimension"])
    ):
        raise ValueError("source and target embeddings require the same backbone")

    target_rows = join_reviewed_target_embeddings(
        target["examples"],
        base_decisions=base_corrections["decisions"],
        followup_decisions=followup_corrections["decisions"],
        sealed_blind_video_sha256s=base_plan[
            "sealed_blind_video_sha256s"
        ],
    )
    return screen_visual_state_domain_embeddings(
        source_examples=source["examples"],
        target_examples=target_rows,
        backbone=str(source["backbone"]),
        backbone_sha256=str(source["backbone_sha256"]),
        source_artifact_sha256=str(source["artifact_sha256"]),
        target_artifact_sha256=str(target["artifact_sha256"]),
        label_correction_sha256s=[
            str(base_corrections["artifact_sha256"]),
            str(followup_corrections["artifact_sha256"]),
        ],
        configurations=configurations,
        acceptance_threshold=acceptance_threshold,
    )


def join_reviewed_target_embeddings(
    target_examples: Sequence[Mapping[str, Any]],
    *,
    base_decisions: Sequence[Mapping[str, Any]],
    followup_decisions: Sequence[Mapping[str, Any]],
    sealed_blind_video_sha256s: Sequence[str],
) -> list[dict[str, Any]]:
    """Exact-join reviewed labels, allowing follow-up only on unresolved rows."""

    blind = {
        _require_sha256(value, field="sealed blind video")
        for value in sealed_blind_video_sha256s
    }
    target_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in target_examples:
        key = _event_key(row)
        if key in target_by_key:
            raise ValueError("duplicate target embedding event")
        if key[0] in blind:
            raise ValueError("sealed blind video cannot enter domain screening")
        target_by_key[key] = row

    base_by_key = _decision_index(base_decisions, field="base")
    unresolved = {
        key for key, row in base_by_key.items() if row.get("corrected_state") is None
    }
    followup_by_key = _decision_index(followup_decisions, field="follow-up")
    if set(followup_by_key) != unresolved:
        raise ValueError(
            "follow-up decisions must exactly cover unresolved base decisions"
        )

    joined = []
    for key, base in base_by_key.items():
        if key not in target_by_key:
            raise ValueError("reviewed event is missing from target embeddings")
        decision = followup_by_key.get(key, base)
        label = decision.get("corrected_state")
        if label is None:
            continue
        if label not in VISUAL_STATE_LABELS:
            raise ValueError("invalid reviewed visual-state label")
        target = target_by_key[key]
        joined.append(
            {
                "source_video_sha256": key[0],
                "candidate_bundle_sha256": key[1],
                "event_id": key[2],
                "label": label,
                "embedding": list(target["embedding"]),
            }
        )
    if not joined:
        raise ValueError("domain screening requires resolved target labels")
    return sorted(
        joined,
        key=lambda row: (
            row["source_video_sha256"],
            row["event_id"],
        ),
    )


def screen_visual_state_domain_embeddings(
    *,
    source_examples: Sequence[Mapping[str, Any]],
    target_examples: Sequence[Mapping[str, Any]],
    backbone: str,
    backbone_sha256: str,
    source_artifact_sha256: str,
    target_artifact_sha256: str,
    label_correction_sha256s: Sequence[str],
    configurations: Sequence[Mapping[str, Any]] | None = None,
    acceptance_threshold: float = 0.85,
) -> dict[str, Any]:
    """Run nested leave-one-target-game-out selection without held-game leakage."""

    if not 0.5 <= float(acceptance_threshold) <= 1.0:
        raise ValueError("acceptance threshold must be between 0.5 and 1.0")
    provenance_hashes = [
        _require_sha256(backbone_sha256, field="backbone"),
        _require_sha256(source_artifact_sha256, field="source artifact"),
        _require_sha256(target_artifact_sha256, field="target artifact"),
        *[
            _require_sha256(value, field="label correction")
            for value in label_correction_sha256s
        ],
    ]
    if len(set(provenance_hashes[3:])) != len(provenance_hashes[3:]):
        raise ValueError("duplicate label-correction artifact")
    if not str(backbone).strip():
        raise ValueError("backbone name is required")

    source_x, source_y, source_groups = _source_arrays(source_examples)
    target_x, target_y, target_groups, target_ids = _target_arrays(
        target_examples
    )
    if source_x.shape[1] != target_x.shape[1]:
        raise ValueError("source and target embedding dimension mismatch")
    groups = sorted(set(target_groups))
    if len(groups) < 4:
        raise ValueError("domain screening requires at least four target groups")
    for group in groups:
        labels = set(target_y[target_groups == group].tolist())
        if labels != {0, 1}:
            raise ValueError("every target group must contain both visual states")

    configs = _normalize_configurations(
        configurations or DEFAULT_CONFIGURATIONS
    )
    outer_predictions = np.zeros(len(target_y), dtype=np.int64)
    outer_probabilities = np.zeros(len(target_y), dtype=np.float64)
    outer_folds = []
    for held_group in groups:
        held_mask = target_groups == held_group
        development_groups = [
            group for group in groups if group != held_group
        ]
        selection = _select_configuration(
            configs=configs,
            source_x=source_x,
            source_y=source_y,
            target_x=target_x,
            target_y=target_y,
            target_groups=target_groups,
            development_groups=development_groups,
        )
        development_mask = np.isin(target_groups, development_groups)
        estimator = _fit_estimator(
            config=selection["configuration"],
            source_x=source_x,
            source_y=source_y,
            target_x=target_x[development_mask],
            target_y=target_y[development_mask],
        )
        probabilities = estimator.predict_proba(target_x[held_mask])[:, 1]
        predictions = (probabilities >= 0.5).astype(np.int64)
        outer_predictions[held_mask] = predictions
        outer_probabilities[held_mask] = probabilities
        outer_folds.append(
            {
                "held_target_group": held_group,
                "training_target_groups": development_groups,
                "selection_target_groups": development_groups,
                "selected_configuration": selection["configuration"],
                "inner_selection": selection["metrics"],
                "held_metrics": _metrics(
                    target_y[held_mask],
                    predictions,
                    probabilities,
                ),
                "held_example_count": int(held_mask.sum()),
            }
        )

    pooled_metrics = _metrics(
        target_y,
        outer_predictions,
        outer_probabilities,
    )
    per_game = {
        group: _metrics(
            target_y[target_groups == group],
            outer_predictions[target_groups == group],
            outer_probabilities[target_groups == group],
        )
        for group in groups
    }
    worst_game_balanced_accuracy = min(
        metrics["balanced_accuracy"] for metrics in per_game.values()
    )
    metrics = {
        **pooled_metrics,
        "worst_game_balanced_accuracy": worst_game_balanced_accuracy,
        "per_game": per_game,
    }
    accepted = (
        worst_game_balanced_accuracy >= acceptance_threshold
        and pooled_metrics["f1"] >= acceptance_threshold
    )
    artifact: dict[str, Any] = {
        "schema_version": VISUAL_STATE_DOMAIN_SCREEN_SCHEMA,
        "purpose": "training_only_cross_broadcast_visual_state_screen",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "backbone": str(backbone),
        "backbone_sha256": backbone_sha256,
        "embedding_dimension": int(source_x.shape[1]),
        "source_embedding_artifact_sha256": source_artifact_sha256,
        "target_embedding_artifact_sha256": target_artifact_sha256,
        "label_correction_artifact_sha256s": list(
            label_correction_sha256s
        ),
        "source_example_count": int(len(source_y)),
        "source_group_count": int(len(set(source_groups))),
        "target_example_count": int(len(target_y)),
        "target_group_count": int(len(groups)),
        "target_event_ids_sha256": _canonical_sha256(target_ids),
        "protocol": {
            "outer_split": "leave_one_target_video_out",
            "inner_selection": "leave_one_remaining_target_video_out",
            "source_usage": "training_fold_only_with_configured_sample_weight",
            "decision_threshold": 0.5,
            "random_seed": 0,
        },
        "configurations": configs,
        "outer_folds": outer_folds,
        "metrics": metrics,
        "acceptance_gate": {
            "minimum_game_balanced_accuracy": float(acceptance_threshold),
            "minimum_pooled_f1": float(acceptance_threshold),
        },
        "accepted": bool(accepted),
        "promotion_decision": (
            "eligible_for_next_training_stage"
            if accepted
            else "rejected_below_cross_game_gate"
        ),
    }
    _validate_visual_state_domain_screen(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_visual_state_domain_screen(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_visual_state_domain_screen(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("visual-state domain screen hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _select_configuration(
    *,
    configs: Sequence[dict[str, Any]],
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    target_groups: np.ndarray,
    development_groups: Sequence[str],
) -> dict[str, Any]:
    candidates = []
    for config in configs:
        predictions = []
        probabilities = []
        labels = []
        group_metrics = {}
        for inner_held in development_groups:
            validation_mask = target_groups == inner_held
            training_mask = np.isin(
                target_groups,
                [group for group in development_groups if group != inner_held],
            )
            estimator = _fit_estimator(
                config=config,
                source_x=source_x,
                source_y=source_y,
                target_x=target_x[training_mask],
                target_y=target_y[training_mask],
            )
            group_probabilities = estimator.predict_proba(
                target_x[validation_mask]
            )[:, 1]
            group_predictions = (group_probabilities >= 0.5).astype(np.int64)
            group_labels = target_y[validation_mask]
            group_metrics[inner_held] = _metrics(
                group_labels,
                group_predictions,
                group_probabilities,
            )
            predictions.extend(group_predictions.tolist())
            probabilities.extend(group_probabilities.tolist())
            labels.extend(group_labels.tolist())
        pooled = _metrics(
            np.asarray(labels),
            np.asarray(predictions),
            np.asarray(probabilities),
        )
        worst = min(
            metrics["balanced_accuracy"] for metrics in group_metrics.values()
        )
        candidates.append(
            {
                "configuration": config,
                "metrics": {
                    **pooled,
                    "worst_game_balanced_accuracy": worst,
                    "per_game": group_metrics,
                },
            }
        )
    return max(
        candidates,
        key=lambda row: (
            row["metrics"]["worst_game_balanced_accuracy"],
            row["metrics"]["balanced_accuracy"],
            row["metrics"]["f1"],
            -configs.index(row["configuration"]),
        ),
    )


def _fit_estimator(
    *,
    config: Mapping[str, Any],
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> Pipeline:
    source_weight = float(config["source_weight"])
    if source_weight > 0:
        features = np.concatenate((target_x, source_x), axis=0)
        labels = np.concatenate((target_y, source_y), axis=0)
        sample_weight = np.concatenate(
            (
                np.ones(len(target_y), dtype=np.float64),
                np.full(len(source_y), source_weight, dtype=np.float64),
            )
        )
    else:
        features = target_x
        labels = target_y
        sample_weight = np.ones(len(target_y), dtype=np.float64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("every training fold must contain both visual states")

    steps: list[tuple[str, Any]] = [("scale", StandardScaler())]
    requested_components = config["pca_components"]
    if requested_components is not None:
        effective_components = min(
            int(requested_components),
            int(features.shape[0] - 1),
            int(features.shape[1]),
        )
        if effective_components < 1:
            raise ValueError("PCA requires at least two training examples")
        steps.append(
            (
                "pca",
                PCA(
                    n_components=effective_components,
                    whiten=True,
                    svd_solver="full",
                ),
            )
        )
    steps.append(
        (
            "classifier",
            LogisticRegression(
                C=float(config["c"]),
                class_weight="balanced",
                max_iter=2000,
                random_state=0,
                solver="liblinear",
            ),
        )
    )
    estimator = Pipeline(steps)
    estimator.fit(
        features,
        labels,
        classifier__sample_weight=sample_weight,
    )
    return estimator


def _source_arrays(
    examples: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not examples:
        raise ValueError("source embeddings are required")
    features = []
    labels = []
    groups = []
    for row in examples:
        label = str(row.get("label") or "")
        group = str(row.get("source_group") or "")
        if label not in BASKETBALL51_LABELS or not group:
            raise ValueError("invalid Basketball-51 source example")
        features.append(_finite_embedding(row))
        labels.append(1 if label.startswith("ft") else 0)
        groups.append(group)
    return _stack_embeddings(features), np.asarray(labels), groups


def _target_arrays(
    examples: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]]]:
    if not examples:
        raise ValueError("target embeddings are required")
    features = []
    labels = []
    groups = []
    identities = []
    seen = set()
    for row in examples:
        source = _require_sha256(
            row.get("source_video_sha256"),
            field="target source video",
        )
        event_id = str(row.get("event_id") or "")
        label = str(row.get("label") or "")
        identity = (source, event_id)
        if (
            not event_id
            or identity in seen
            or label not in VISUAL_STATE_LABELS
        ):
            raise ValueError("invalid or duplicate target example")
        seen.add(identity)
        features.append(_finite_embedding(row))
        labels.append(1 if label == "free_throw" else 0)
        groups.append(source)
        identities.append(
            {"source_video_sha256": source, "event_id": event_id}
        )
    return (
        _stack_embeddings(features),
        np.asarray(labels),
        np.asarray(groups),
        identities,
    )


def _finite_embedding(row: Mapping[str, Any]) -> list[float]:
    embedding = row.get("embedding")
    if (
        not isinstance(embedding, list)
        or not embedding
        or not all(math.isfinite(float(value)) for value in embedding)
    ):
        raise ValueError("embedding must contain finite values")
    return [float(value) for value in embedding]


def _stack_embeddings(features: Sequence[Sequence[float]]) -> np.ndarray:
    dimensions = {len(row) for row in features}
    if len(dimensions) != 1:
        raise ValueError("embedding dimension mismatch")
    return np.asarray(features, dtype=np.float64)


def _decision_index(
    decisions: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    indexed = {}
    for row in decisions:
        key = _event_key(row)
        if key in indexed:
            raise ValueError(f"duplicate {field} review decision")
        indexed[key] = row
    return indexed


def _event_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    source = _require_sha256(
        row.get("source_video_sha256"),
        field="source video",
    )
    bundle = _require_sha256(
        row.get("candidate_bundle_sha256"),
        field="candidate bundle",
    )
    event_id = str(row.get("event_id") or "")
    if not event_id:
        raise ValueError("event id is required")
    return source, bundle, event_id


def _normalize_configurations(
    configurations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized = []
    for row in configurations:
        c_value = float(row.get("c", 0.0))
        source_weight = float(row.get("source_weight", -1.0))
        pca_value = row.get("pca_components")
        if (
            not math.isfinite(c_value)
            or c_value <= 0
            or not math.isfinite(source_weight)
            or not 0.0 <= source_weight <= 1.0
            or (
                pca_value is not None
                and (
                    isinstance(pca_value, bool)
                    or int(pca_value) < 1
                )
            )
        ):
            raise ValueError("invalid domain-screen configuration")
        normalized.append(
            {
                "c": c_value,
                "source_weight": source_weight,
                "pca_components": (
                    None if pca_value is None else int(pca_value)
                ),
            }
        )
    if not normalized:
        raise ValueError("domain screening requires configurations")
    if len({_canonical_sha256(row) for row in normalized}) != len(normalized):
        raise ValueError("duplicate domain-screen configuration")
    return normalized


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
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def _validate_visual_state_domain_screen(
    artifact: Mapping[str, Any],
) -> None:
    if (
        artifact.get("schema_version") != VISUAL_STATE_DOMAIN_SCREEN_SCHEMA
        or artifact.get("purpose")
        != "training_only_cross_broadcast_visual_state_screen"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("invalid visual-state domain screen")
    _require_sha256(artifact.get("backbone_sha256"), field="backbone")
    _require_sha256(
        artifact.get("source_embedding_artifact_sha256"),
        field="source artifact",
    )
    _require_sha256(
        artifact.get("target_embedding_artifact_sha256"),
        field="target artifact",
    )
    _require_sha256(
        artifact.get("target_event_ids_sha256"),
        field="target event identities",
    )
    correction_hashes = artifact.get("label_correction_artifact_sha256s")
    if not isinstance(correction_hashes, list) or not correction_hashes:
        raise ValueError("domain screen requires label corrections")
    for value in correction_hashes:
        _require_sha256(value, field="label correction")
    if (
        int(artifact.get("embedding_dimension", 0)) < 1
        or int(artifact.get("source_example_count", 0)) < 1
        or int(artifact.get("target_example_count", 0)) < 1
        or int(artifact.get("target_group_count", 0)) < 4
    ):
        raise ValueError("domain screen has invalid dimensions or counts")
    metrics = artifact.get("metrics")
    gate = artifact.get("acceptance_gate")
    folds = artifact.get("outer_folds")
    if (
        not isinstance(metrics, Mapping)
        or not isinstance(gate, Mapping)
        or not isinstance(folds, list)
        or len(folds) != int(artifact["target_group_count"])
    ):
        raise ValueError("domain screen metrics are incomplete")
    expected_accepted = (
        float(metrics["worst_game_balanced_accuracy"])
        >= float(gate["minimum_game_balanced_accuracy"])
        and float(metrics["f1"]) >= float(gate["minimum_pooled_f1"])
    )
    if artifact.get("accepted") is not expected_accepted:
        raise ValueError("domain screen acceptance decision is inconsistent")
    per_game = metrics.get("per_game")
    if not isinstance(per_game, Mapping):
        raise ValueError("domain screen per-game metrics are missing")
    group_names = set(per_game)
    held_names = {
        str(fold.get("held_target_group") or "") for fold in folds
    }
    if (
        len(group_names) != int(artifact["target_group_count"])
        or held_names != group_names
    ):
        raise ValueError("domain screen target-group coverage is inconsistent")
    for fold in folds:
        held = str(fold.get("held_target_group") or "")
        training = fold.get("training_target_groups")
        selection = fold.get("selection_target_groups")
        if (
            not held
            or not isinstance(training, list)
            or not isinstance(selection, list)
            or held in training
            or held in selection
            or set(training) != group_names - {held}
            or set(selection) != group_names - {held}
            or fold.get("held_metrics") != per_game[held]
        ):
            raise ValueError("held target group leaked into domain screening")


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value or "")
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"invalid {field} SHA-256")
    return text


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
