"""Nested game-disjoint screening of causal phase auxiliary supervision."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from app.analysis.visual_state_temporal_screen import (
    _fit_estimator,
    _metrics,
    _normalize_configurations,
    _temporal_arrays,
)

CAUSAL_PHASE_AUXILIARY_SCREEN_SCHEMA = (
    "agu.causal-phase-auxiliary-screen.v1"
)
PHASE_TARGET_NAMES = (
    "formation_free_throw",
    "formation_live_play",
    "context_replay",
    "context_live_action",
    "sequence_complete",
    "sequence_not_a_shot",
    "release_visible",
    "rim_visible",
)


def join_causal_phase_targets(
    temporal_examples: Sequence[Mapping[str, Any]],
    phase_reviews: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Exact-join sealed phase decisions to temporal embedding examples."""

    def key(row: Mapping[str, Any]) -> tuple[str, str, str]:
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
            raise ValueError("phase auxiliary event ID is required")
        return source, bundle, event_id

    temporal_by_key = {}
    for row in temporal_examples:
        event_key = key(row)
        if event_key in temporal_by_key:
            raise ValueError("duplicate temporal phase auxiliary event")
        temporal_by_key[event_key] = row
    reviews_by_key = {}
    for row in phase_reviews:
        event_key = key(row)
        if event_key in reviews_by_key:
            raise ValueError("duplicate causal phase review event")
        reviews_by_key[event_key] = row
    if not set(temporal_by_key).issubset(reviews_by_key):
        raise ValueError(
            "causal phase reviews must exactly cover all temporal examples"
        )

    joined = []
    for row in temporal_examples:
        review = reviews_by_key[key(row)]
        joined.append(
            {
                **row,
                "phase_targets": {
                    "formation_free_throw": int(
                        review.get("formation_state")
                        == "free_throw_setup"
                    ),
                    "formation_live_play": int(
                        review.get("formation_state") == "live_play"
                    ),
                    "context_replay": int(
                        review.get("broadcast_context") == "replay"
                    ),
                    "context_live_action": int(
                        review.get("broadcast_context") == "live_action"
                    ),
                    "sequence_complete": int(
                        review.get("shot_sequence") == "complete"
                    ),
                    "sequence_not_a_shot": int(
                        review.get("shot_sequence") == "not_a_shot"
                    ),
                    "release_visible": int(
                        review.get("release_position") is not None
                    ),
                    "rim_visible": int(
                        review.get("rim_arrival_position") is not None
                    ),
                },
            }
        )
    return joined


def screen_causal_phase_auxiliary_examples(
    *,
    examples: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    configurations: Sequence[Mapping[str, Any]],
    acceptance_threshold: float = 0.85,
) -> dict[str, Any]:
    """Compare a base temporal probe with strictly OOF phase stacking."""

    if not 0.5 <= float(acceptance_threshold) <= 1.0:
        raise ValueError("acceptance threshold must be between 0.5 and 1.0")
    labels, groups, event_ids, representations = _temporal_arrays(examples)
    phase_targets = _phase_target_array(examples)
    unique_groups = sorted(set(groups.tolist()))
    if len(unique_groups) < 4:
        raise ValueError("phase auxiliary screening requires four games")
    for group in unique_groups:
        if set(labels[groups == group].tolist()) != {0, 1}:
            raise ValueError("every game must contain both visual states")
    configs = _normalize_configurations(configurations)
    if any(row["representation"] not in representations for row in configs):
        raise ValueError("phase auxiliary representation is unavailable")

    variants = {
        name: _screen_variant(
            name=name,
            configs=configs,
            representations=representations,
            labels=labels,
            phase_targets=phase_targets,
            groups=groups,
            unique_groups=unique_groups,
        )
        for name in ("baseline", "phase_auxiliary")
    }
    phase_metrics = variants["phase_auxiliary"]["metrics"]
    accepted = (
        phase_metrics["worst_game_balanced_accuracy"]
        >= acceptance_threshold
        and phase_metrics["f1"] >= acceptance_threshold
    )
    artifact: dict[str, Any] = {
        "schema_version": CAUSAL_PHASE_AUXILIARY_SCREEN_SCHEMA,
        "purpose": "training_only_causal_phase_auxiliary_screen",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "backbone": str(provenance.get("backbone") or ""),
        "backbone_sha256": _require_sha256(
            provenance.get("backbone_sha256"),
            field="backbone",
        ),
        "embedding_dimension": int(
            provenance.get("embedding_dimension", 0)
        ),
        "embedding_artifact_sha256s": _hash_list(
            provenance.get("embedding_artifact_sha256s"),
            field="embedding artifact",
        ),
        "label_correction_artifact_sha256s": _hash_list(
            provenance.get("label_correction_artifact_sha256s"),
            field="label correction artifact",
        ),
        "phase_review_artifact_sha256": _require_sha256(
            provenance.get("phase_review_artifact_sha256"),
            field="phase review artifact",
        ),
        "phase_target_names": list(PHASE_TARGET_NAMES),
        "phase_review_example_count": int(
            provenance.get("phase_review_example_count", len(labels))
        ),
        "excluded_unresolved_target_count": int(
            provenance.get("excluded_unresolved_target_count", 0)
        ),
        "target_example_count": len(labels),
        "target_group_count": len(unique_groups),
        "target_event_ids_sha256": _canonical_sha256(event_ids),
        "protocol": {
            "outer_split": "leave_one_source_video_out",
            "inner_selection": "leave_one_remaining_source_video_out",
            "phase_features": "other_game_predictions_only",
            "meta_training": "leave_one_training_game_out_oof",
            "decision_threshold": 0.5,
            "random_seed": 0,
        },
        "configurations": configs,
        "variants": variants,
        "acceptance_gate": {
            "minimum_game_balanced_accuracy": float(
                acceptance_threshold
            ),
            "minimum_pooled_f1": float(acceptance_threshold),
        },
        "accepted": bool(accepted),
        "promotion_decision": (
            "eligible_for_next_training_stage"
            if accepted
            else "rejected_below_cross_game_gate"
        ),
    }
    _validate_screen(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_causal_phase_auxiliary_screen(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_screen(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("causal phase auxiliary screen hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _screen_variant(
    *,
    name: str,
    configs: Sequence[dict[str, Any]],
    representations: Mapping[str, np.ndarray],
    labels: np.ndarray,
    phase_targets: np.ndarray,
    groups: np.ndarray,
    unique_groups: Sequence[str],
) -> dict[str, Any]:
    predictions = np.zeros(len(labels), dtype=np.int64)
    probabilities = np.zeros(len(labels), dtype=np.float64)
    folds = []
    for held_group in unique_groups:
        training_groups = [
            value for value in unique_groups if value != held_group
        ]
        selection = _select_configuration(
            name=name,
            configs=configs,
            representations=representations,
            labels=labels,
            phase_targets=phase_targets,
            groups=groups,
            development_groups=training_groups,
        )
        config = selection["configuration"]
        features = representations[config["representation"]]
        held_probabilities, provenance = _fold_probabilities(
            name=name,
            features=features,
            labels=labels,
            phase_targets=phase_targets,
            groups=groups,
            training_groups=training_groups,
            scored_group=held_group,
            config=config,
        )
        held_mask = groups == held_group
        held_predictions = (
            held_probabilities >= 0.5
        ).astype(np.int64)
        probabilities[held_mask] = held_probabilities
        predictions[held_mask] = held_predictions
        folds.append(
            {
                "held_target_group": held_group,
                "training_target_groups": training_groups,
                "selection_target_groups": training_groups,
                "selected_configuration": config,
                "inner_selection": selection["metrics"],
                "held_metrics": _metrics(
                    labels[held_mask],
                    held_predictions,
                    held_probabilities,
                ),
                "held_example_count": int(held_mask.sum()),
                **(
                    {"meta_oof_provenance": provenance}
                    if name == "phase_auxiliary"
                    else {}
                ),
            }
        )
    per_game = {
        group: _metrics(
            labels[groups == group],
            predictions[groups == group],
            probabilities[groups == group],
        )
        for group in unique_groups
    }
    return {
        "outer_folds": folds,
        "metrics": {
            **_metrics(labels, predictions, probabilities),
            "worst_game_balanced_accuracy": min(
                row["balanced_accuracy"] for row in per_game.values()
            ),
            "per_game": per_game,
        },
    }


def _select_configuration(
    *,
    name: str,
    configs: Sequence[dict[str, Any]],
    representations: Mapping[str, np.ndarray],
    labels: np.ndarray,
    phase_targets: np.ndarray,
    groups: np.ndarray,
    development_groups: Sequence[str],
) -> dict[str, Any]:
    candidates = []
    for config in configs:
        features = representations[config["representation"]]
        predictions = []
        probabilities = []
        expected = []
        per_game = {}
        for held_group in development_groups:
            training_groups = [
                value
                for value in development_groups
                if value != held_group
            ]
            held_probabilities, _ = _fold_probabilities(
                name=name,
                features=features,
                labels=labels,
                phase_targets=phase_targets,
                groups=groups,
                training_groups=training_groups,
                scored_group=held_group,
                config=config,
            )
            held_mask = groups == held_group
            held_labels = labels[held_mask]
            held_predictions = (
                held_probabilities >= 0.5
            ).astype(np.int64)
            per_game[held_group] = _metrics(
                held_labels,
                held_predictions,
                held_probabilities,
            )
            predictions.extend(held_predictions.tolist())
            probabilities.extend(held_probabilities.tolist())
            expected.extend(held_labels.tolist())
        pooled = _metrics(
            np.asarray(expected),
            np.asarray(predictions),
            np.asarray(probabilities),
        )
        candidates.append(
            {
                "configuration": config,
                "metrics": {
                    **pooled,
                    "worst_game_balanced_accuracy": min(
                        row["balanced_accuracy"]
                        for row in per_game.values()
                    ),
                    "per_game": per_game,
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


def _fold_probabilities(
    *,
    name: str,
    features: np.ndarray,
    labels: np.ndarray,
    phase_targets: np.ndarray,
    groups: np.ndarray,
    training_groups: Sequence[str],
    scored_group: str,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    training_mask = np.isin(groups, training_groups)
    scored_mask = groups == scored_group
    base = _fit_estimator(
        features[training_mask],
        labels[training_mask],
        config=config,
    )
    base_scores = base.predict_proba(features[scored_mask])[:, 1]
    if name == "baseline":
        return base_scores, []

    meta_rows = []
    meta_labels = []
    provenance = []
    for meta_group in training_groups:
        phase_training_groups = [
            value for value in training_groups if value != meta_group
        ]
        phase_training_mask = np.isin(groups, phase_training_groups)
        meta_mask = groups == meta_group
        meta_base = _fit_estimator(
            features[phase_training_mask],
            labels[phase_training_mask],
            config=config,
        )
        meta_base_scores = meta_base.predict_proba(
            features[meta_mask]
        )[:, 1]
        meta_phase_scores = _phase_probabilities(
            train_features=features[phase_training_mask],
            train_targets=phase_targets[phase_training_mask],
            score_features=features[meta_mask],
            config=config,
        )
        meta_rows.append(
            np.column_stack((meta_base_scores, meta_phase_scores))
        )
        meta_labels.append(labels[meta_mask])
        provenance.append(
            {
                "scored_group": meta_group,
                "phase_training_groups": phase_training_groups,
            }
        )
    meta_estimator = _fit_meta(
        np.vstack(meta_rows),
        np.concatenate(meta_labels),
        c_value=float(config["c"]),
    )
    held_phase_scores = _phase_probabilities(
        train_features=features[training_mask],
        train_targets=phase_targets[training_mask],
        score_features=features[scored_mask],
        config=config,
    )
    stacked = np.column_stack((base_scores, held_phase_scores))
    return meta_estimator.predict_proba(stacked)[:, 1], provenance


def _phase_probabilities(
    *,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    score_features: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    components = min(
        int(config["pca_components"]),
        int(train_features.shape[0] - 1),
        int(train_features.shape[1]),
    )
    if components < 1:
        raise ValueError("phase PCA requires at least two examples")
    reducer = PCA(
        n_components=components,
        whiten=True,
        svd_solver="randomized",
        random_state=0,
    )
    train_reduced = reducer.fit_transform(train_features)
    score_reduced = reducer.transform(score_features)
    columns = []
    for index in range(train_targets.shape[1]):
        target = train_targets[:, index]
        classes = set(target.tolist())
        if len(classes) == 1:
            columns.append(
                np.full(len(score_features), float(target[0]))
            )
            continue
        estimator = LogisticRegression(
            C=float(config["c"]),
            class_weight="balanced",
            max_iter=2000,
            random_state=0,
            solver="liblinear",
        )
        estimator.fit(train_reduced, target)
        columns.append(estimator.predict_proba(score_reduced)[:, 1])
    return np.column_stack(columns)


def _fit_meta(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    c_value: float,
) -> LogisticRegression:
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("meta training requires both visual states")
    estimator = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=2000,
        random_state=0,
        solver="liblinear",
    )
    estimator.fit(features, labels)
    return estimator


def _phase_target_array(
    examples: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    rows = []
    for row in examples:
        values = row.get("phase_targets")
        if not isinstance(values, Mapping):
            raise ValueError("phase targets are required")
        encoded = []
        for name in PHASE_TARGET_NAMES:
            value = values.get(name)
            if value not in (0, 1, False, True):
                raise ValueError("phase targets must be binary")
            encoded.append(int(value))
        rows.append(encoded)
    return np.asarray(rows, dtype=np.int64)


def _validate_screen(artifact: Mapping[str, Any]) -> None:
    if (
        artifact.get("schema_version")
        != CAUSAL_PHASE_AUXILIARY_SCREEN_SCHEMA
        or artifact.get("purpose")
        != "training_only_causal_phase_auxiliary_screen"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("invalid causal phase auxiliary screen")
    _require_sha256(artifact.get("backbone_sha256"), field="backbone")
    _hash_list(
        artifact.get("embedding_artifact_sha256s"),
        field="embedding artifact",
    )
    _hash_list(
        artifact.get("label_correction_artifact_sha256s"),
        field="label correction artifact",
    )
    _require_sha256(
        artifact.get("phase_review_artifact_sha256"),
        field="phase review artifact",
    )
    groups = int(artifact.get("target_group_count", 0))
    examples = int(artifact.get("target_example_count", 0))
    phase_examples = int(artifact.get("phase_review_example_count", 0))
    excluded = int(artifact.get("excluded_unresolved_target_count", -1))
    variants = artifact.get("variants")
    if (
        groups < 4
        or examples < groups * 2
        or excluded < 0
        or phase_examples != examples + excluded
        or not isinstance(variants, Mapping)
    ):
        raise ValueError("invalid causal phase auxiliary coverage")
    if set(variants) != {"baseline", "phase_auxiliary"}:
        raise ValueError("causal phase auxiliary variants are incomplete")
    for name, variant in variants.items():
        folds = variant.get("outer_folds")
        if not isinstance(folds, list) or len(folds) != groups:
            raise ValueError("causal phase auxiliary folds are incomplete")
        for fold in folds:
            held = fold.get("held_target_group")
            if (
                held in fold.get("training_target_groups", [])
                or held in fold.get("selection_target_groups", [])
            ):
                raise ValueError("held game leaked into phase training")
            if name == "phase_auxiliary":
                for row in fold.get("meta_oof_provenance", []):
                    if (
                        held in row.get("phase_training_groups", [])
                        or row.get("scored_group")
                        in row.get("phase_training_groups", [])
                    ):
                        raise ValueError(
                            "held or scored game leaked into phase stacking"
                        )
    metrics = variants["phase_auxiliary"].get("metrics")
    gate = artifact.get("acceptance_gate")
    if not isinstance(metrics, Mapping) or not isinstance(gate, Mapping):
        raise ValueError("causal phase auxiliary metrics are missing")
    expected = (
        float(metrics["worst_game_balanced_accuracy"])
        >= float(gate["minimum_game_balanced_accuracy"])
        and float(metrics["f1"])
        >= float(gate["minimum_pooled_f1"])
    )
    if bool(artifact.get("accepted")) != expected:
        raise ValueError("causal phase auxiliary acceptance is inconsistent")


def _hash_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} hashes are required")
    values = [_require_sha256(item, field=field) for item in value]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {field} hash")
    return values


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"invalid {field} SHA-256")
    return text


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
