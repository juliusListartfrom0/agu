"""Offline, game-held screening for continuous shot-causal support features.

The screen joins the frozen shot-validity OOF probabilities with the
training-only reason evidence artifact and evaluates a small logistic model
under outer game isolation.  It is deliberately not a runtime component:
the artifact is evidence for deciding whether a future experiment is worth
promoting, never an AGU answer source.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from app.analysis.shot_reason_evidence import verify_reason_evidence_artifact

CAUSAL_SUPPORT_SCHEMA = "agu.shot-causal-support-screen.v2"
DEFAULT_CAUSAL_FEATURE_NAMES = (
    "ball_rim_hit_count",
    "min_ball_rim_distance",
    "approach_rise_height_ratio",
    "candidate_observation_count",
    "candidate_player_count",
    "max_stable_contact_count",
    "candidate_team_count",
)
MINIMUM_PRECISION = 0.85
MINIMUM_RECALL = 0.85


def screen_shot_causal_support(
    *,
    base_artifact: Mapping[str, Any],
    reason_evidence_artifact: Mapping[str, Any],
    causal_feature_names: Sequence[str] = DEFAULT_CAUSAL_FEATURE_NAMES,
    regularization_c: float = 0.01,
) -> dict[str, Any]:
    """Screen reason evidence with a frozen base under outer game isolation.

    The base probability is retained as a model input so the experiment asks
    a narrow question: can independently computed continuous causal support
    improve a known shot-validity signal? Thresholds are selected from
    leave-one-training-game-out scores inside each outer fold and are then
    applied once to the held game. This keeps threshold selection from using
    either the held game or in-sample training predictions.
    """

    if regularization_c <= 0 or not math.isfinite(float(regularization_c)):
        raise ValueError("causal support regularization must be positive")
    selected_features = tuple(str(name) for name in causal_feature_names)
    if not selected_features or any(not name for name in selected_features):
        raise ValueError("causal support requires named features")
    if len(set(selected_features)) != len(selected_features):
        raise ValueError("causal support feature names must be unique")

    base_rows = _extract_base_rows(base_artifact)
    reason = _verify_reason_for_join(reason_evidence_artifact)
    reason_names = tuple(str(name) for name in reason["feature_names"])
    unknown = [name for name in selected_features if name not in reason_names]
    if unknown:
        raise ValueError(f"unknown causal feature: {unknown[0]}")

    normalized_base = [_normalize_base_row(row) for row in base_rows]
    normalized_reason = [_normalize_reason_row(row) for row in reason["examples"]]
    base_lookup = _index_rows(normalized_base, "base")
    reason_lookup = _index_rows(normalized_reason, "reason")
    if set(base_lookup) != set(reason_lookup):
        raise ValueError(
            "base and reason evidence must exactly cover the same join keys"
        )

    ordered_keys = [
        (row["source_video_sha256"], row["event_id"])
        for row in normalized_base
    ]
    groups = np.asarray([key[0] for key in ordered_keys], dtype=object)
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 4:
        raise ValueError("causal support screening requires at least four games")
    labels = np.asarray(
        [int(base_lookup[key]["event_present"]) for key in ordered_keys],
        dtype=np.int64,
    )
    base_probabilities = np.asarray(
        [float(base_lookup[key]["probability"]) for key in ordered_keys],
        dtype=np.float64,
    )
    feature_indexes = {name: reason_names.index(name) for name in selected_features}
    reason_matrix = np.asarray(
        [
            [float(reason_lookup[key]["features"][feature_indexes[name]]) for name in selected_features]
            for key in ordered_keys
        ],
        dtype=np.float64,
    )
    if not np.isfinite(reason_matrix).all():
        raise ValueError("causal support features must be finite")
    causal_matrix = np.column_stack([base_probabilities, reason_matrix])

    baseline_scores = np.zeros(len(ordered_keys), dtype=np.float64)
    causal_scores = np.zeros(len(ordered_keys), dtype=np.float64)
    folds: list[dict[str, Any]] = []
    for held_group in unique_groups:
        training_mask = groups != held_group
        held_mask = ~training_mask
        training_groups = sorted(set(groups[training_mask]))
        if len(set(labels[training_mask])) < 2:
            raise ValueError("causal support training fold requires both labels")

        baseline_scores[held_mask] = base_probabilities[held_mask]
        scaler, classifier = _fit_causal_classifier(
            causal_matrix[training_mask],
            labels[training_mask],
            regularization_c=regularization_c,
        )
        causal_scores[held_mask] = classifier.predict_proba(
            scaler.transform(causal_matrix[held_mask])
        )[:, 1]

        # Threshold selection is itself nested.  The previous implementation
        # selected from ``causal_scores[training_mask]`` even though that
        # buffer is only populated for the outer held game, so those scores
        # were all zero and the threshold frequently collapsed to zero.
        inner_training_scores = np.zeros(int(training_mask.sum()), dtype=np.float64)
        outer_training_indices = np.flatnonzero(training_mask)
        inner_training_groups = sorted(set(groups[training_mask]))
        inner_score_provenance = []
        for scored_group in inner_training_groups:
            scored_mask = training_mask & (groups == scored_group)
            inner_fit_mask = training_mask & (groups != scored_group)
            if len(set(labels[inner_fit_mask])) < 2:
                raise ValueError("causal support inner fold requires both labels")
            inner_scaler, inner_classifier = _fit_causal_classifier(
                causal_matrix[inner_fit_mask],
                labels[inner_fit_mask],
                regularization_c=regularization_c,
            )
            scored_indices = np.flatnonzero(scored_mask)
            inner_training_scores[
                np.searchsorted(outer_training_indices, scored_indices)
            ] = inner_classifier.predict_proba(
                inner_scaler.transform(causal_matrix[scored_mask])
            )[:, 1]
            inner_score_provenance.append(
                {
                    "scored_game_sha256": scored_group,
                    "training_game_sha256s": sorted(
                        set(groups[inner_fit_mask])
                    ),
                    "scored_example_count": int(scored_mask.sum()),
                }
            )

        baseline_threshold = _select_group_held_threshold(
            base_probabilities[training_mask],
            labels[training_mask],
            groups[training_mask],
        )
        causal_threshold = _select_group_held_threshold(
            inner_training_scores,
            labels[training_mask],
            groups[training_mask],
        )
        held_baseline = _metrics(
            labels[held_mask],
            base_probabilities[held_mask] >= baseline_threshold["threshold"],
        )
        held_causal = _metrics(
            labels[held_mask],
            causal_scores[held_mask] >= causal_threshold["threshold"],
        )
        folds.append(
            {
                "held_game_sha256": held_group,
                "training_game_sha256s": training_groups,
                "training_example_count": int(training_mask.sum()),
                "held_example_count": int(held_mask.sum()),
                "baseline_threshold": baseline_threshold,
                "causal_support_threshold": causal_threshold,
                "causal_threshold_score_provenance": inner_score_provenance,
                "held_metrics": {
                    "baseline": held_baseline,
                    "causal_support": held_causal,
                },
            }
        )

    predictions = []
    thresholds_by_group = {
        str(fold["held_game_sha256"]): fold for fold in folds
    }
    for index, key in enumerate(ordered_keys):
        fold = thresholds_by_group[key[0]]
        baseline_threshold = float(fold["baseline_threshold"]["threshold"])
        causal_threshold = float(fold["causal_support_threshold"]["threshold"])
        predictions.append(
            {
                "source_video_sha256": key[0],
                "event_id": key[1],
                "event_present": bool(labels[index]),
                "base_probability": float(base_probabilities[index]),
                "causal_support_probability": float(causal_scores[index]),
                "baseline_threshold": baseline_threshold,
                "causal_support_threshold": causal_threshold,
                "baseline_predicted": bool(base_probabilities[index] >= baseline_threshold),
                "causal_support_predicted": bool(causal_scores[index] >= causal_threshold),
            }
        )

    metrics = {
        "baseline": _screen_metrics(predictions, "base_probability", "baseline_threshold"),
        "causal_support": _screen_metrics(
            predictions,
            "causal_support_probability",
            "causal_support_threshold",
        ),
    }
    baseline_gate = bool(metrics["baseline"]["promoted"])
    causal_gate = bool(metrics["causal_support"]["promoted"])
    causal_f1 = float(metrics["causal_support"]["pooled"]["f1"])
    baseline_f1 = float(metrics["baseline"]["pooled"]["f1"])
    causal_precision = float(metrics["causal_support"]["pooled"]["precision"])
    baseline_precision = float(metrics["baseline"]["pooled"]["precision"])
    accepted = bool(
        causal_gate
        and (causal_f1 > baseline_f1 or causal_precision > baseline_precision)
    )
    payload: dict[str, Any] = {
        "schema_version": CAUSAL_SUPPORT_SCHEMA,
        "purpose": "offline_shot_causal_support_screening",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "selection_protocol": "outer_game_held_with_inner_game_oof_threshold",
        "base_artifact_sha256": base_artifact.get("artifact_sha256"),
        "reason_evidence_artifact_sha256": reason["artifact_sha256"],
        "causal_feature_names": list(selected_features),
        "regularization_c": float(regularization_c),
        "minimum_precision": MINIMUM_PRECISION,
        "minimum_recall": MINIMUM_RECALL,
        "folds": folds,
        "predictions": predictions,
        "metrics": metrics,
        "accepted": accepted,
        "promotion_decision": (
            "accepted_causal_support_improvement"
            if accepted
            else "rejected_below_cross_game_gate_or_no_improvement"
        ),
        "baseline_gate_promoted": baseline_gate,
        "causal_support_gate_promoted": causal_gate,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return payload


def _extract_base_rows(base_artifact: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if base_artifact.get("runtime_consumable") is not False:
        raise ValueError("base shot artifact must remain offline-only")
    rows = base_artifact.get("oof_predictions")
    if not isinstance(rows, list):
        best = base_artifact.get("best_variant")
        rows = best.get("oof_predictions") if isinstance(best, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("base shot artifact requires OOF predictions")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("base OOF predictions must be objects")
    return list(rows)


def _verify_reason_for_join(payload: Mapping[str, Any]) -> dict[str, Any]:
    examples = payload.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("reason evidence must exactly cover base examples")
    # Check the join contract before hash verification so a malformed duplicate
    # cannot be mistaken for a missing example when reporting the screen error.
    keys = []
    for row in examples:
        if not isinstance(row, Mapping):
            raise ValueError("reason evidence must exactly cover base examples")
        key = (str(row.get("source_video_sha256") or ""), str(row.get("event_id") or ""))
        if not all(key) or key in keys:
            raise ValueError("reason evidence must exactly cover base examples")
        keys.append(key)
    return verify_reason_evidence_artifact(payload)


def _normalize_base_row(row: Mapping[str, Any]) -> dict[str, Any]:
    source = str(row.get("source_video_sha256") or "")
    event = str(row.get("event_id") or "")
    probability = row.get("probability")
    label = row.get("event_present")
    try:
        probability = float(probability)
    except (TypeError, ValueError):
        probability = math.nan
    if (
        not source
        or not event
        or not isinstance(label, bool)
        or not math.isfinite(probability)
        or not 0.0 <= probability <= 1.0
    ):
        raise ValueError("base OOF predictions have invalid key, label, or probability")
    return {
        "source_video_sha256": source,
        "event_id": event,
        "event_present": label,
        "probability": probability,
    }


def _normalize_reason_row(row: Mapping[str, Any]) -> dict[str, Any]:
    source = str(row.get("source_video_sha256") or "")
    event = str(row.get("event_id") or "")
    features = row.get("features")
    if not source or not event or not isinstance(features, list):
        raise ValueError("reason evidence has an invalid join row")
    try:
        values = [float(value) for value in features]
    except (TypeError, ValueError):
        raise ValueError("reason evidence has a non-numeric feature") from None
    if not all(math.isfinite(value) for value in values):
        raise ValueError("reason evidence has a non-finite feature")
    return {
        "source_video_sha256": source,
        "event_id": event,
        "features": values,
    }


def _fit_causal_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    regularization_c: float,
) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(
        C=float(regularization_c),
        class_weight="balanced",
        max_iter=5000,
        random_state=0,
    ).fit(scaler.transform(features), labels)
    return scaler, classifier


def _index_rows(
    rows: Sequence[Mapping[str, Any]], label: str
) -> dict[tuple[str, str], Mapping[str, Any]]:
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["source_video_sha256"]), str(row["event_id"]))
        if key in indexed:
            raise ValueError(f"{label} rows contain duplicate join keys")
        indexed[key] = row
    return indexed


def _select_group_held_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    thresholds = sorted({0.0, 1.0, *(float(score) for score in scores)})
    candidates = []
    for threshold in thresholds:
        predicted = scores >= threshold
        pooled = _metrics(labels, predicted)
        per_group = {
            str(group): _metrics(labels[groups == group], predicted[groups == group])
            for group in sorted(set(groups))
        }
        min_precision = min(row["precision"] for row in per_group.values())
        min_recall = min(row["recall"] for row in per_group.values())
        candidates.append(
            {
                "threshold": float(threshold),
                "pooled": pooled,
                "per_game": per_group,
                "minimum_precision": float(min_precision),
                "minimum_recall": float(min_recall),
                "meets_training_recall": bool(min_recall >= MINIMUM_RECALL),
            }
        )
    return max(
        candidates,
        key=lambda row: (
            int(row["meets_training_recall"]),
            row["minimum_precision"],
            row["minimum_recall"],
            row["pooled"]["f1"],
            row["pooled"]["precision"],
            -row["threshold"],
        ),
    )


def _screen_metrics(
    rows: Sequence[Mapping[str, Any]], score_field: str, threshold_field: str
) -> dict[str, Any]:
    labels = np.asarray([int(bool(row["event_present"])) for row in rows])
    predicted = np.asarray(
        [float(row[score_field]) >= float(row[threshold_field]) for row in rows],
        dtype=bool,
    )
    groups = [str(row["source_video_sha256"]) for row in rows]
    per_game = {
        group: _metrics(
            labels[np.asarray([item == group for item in groups], dtype=bool)],
            predicted[np.asarray([item == group for item in groups], dtype=bool)],
        )
        for group in sorted(set(groups))
    }
    pooled = _metrics(labels, predicted)
    promoted = bool(
        pooled["precision"] >= MINIMUM_PRECISION
        and pooled["recall"] >= MINIMUM_RECALL
        and all(
            row["precision"] >= MINIMUM_PRECISION
            and row["recall"] >= MINIMUM_RECALL
            for row in per_game.values()
        )
    )
    return {
        "pooled": pooled,
        "per_game": per_game,
        "promoted": promoted,
    }


def _metrics(labels: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int(np.logical_and(predicted, labels).sum())
    fp = int(np.logical_and(predicted, ~labels).sum())
    fn = int(np.logical_and(~predicted, labels).sum())
    tn = int(np.logical_and(~predicted, ~labels).sum())
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2.0 * precision * recall, precision + recall),
    }


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("artifact_sha256", None)
    return hashlib.sha256(
        json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
