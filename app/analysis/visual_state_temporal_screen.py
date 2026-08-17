"""Nested game-held screening of existing temporal DINOv2 state embeddings."""

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

from app.analysis.pbp_visual_state_frames import (
    ANCHOR_OFFSETS_SECONDS,
    POST_ANCHOR_OFFSETS_SECONDS,
    PRE_ANCHOR_OFFSETS_SECONDS,
    WIDE_ANCHOR_OFFSETS_SECONDS,
    verify_anchor_state_embedding_artifact,
)
from app.analysis.pbp_visual_state_review import (
    verify_visual_state_label_corrections,
    verify_visual_state_review_plan,
)

VISUAL_STATE_TEMPORAL_SCREEN_SCHEMA = "agu.visual-state-temporal-screen.v1"
VISUAL_STATE_LABELS = ("field_goal", "free_throw")
TEMPORAL_REPRESENTATIONS = (
    "anchor_summary",
    "pre_summary",
    "full_summary",
    "full_flat",
    "wide_summary",
    "wide_flat",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


def build_visual_state_temporal_screen(
    *,
    pre_anchor_artifact: Mapping[str, Any],
    anchor_artifact: Mapping[str, Any],
    post_anchor_artifact: Mapping[str, Any],
    wide_anchor_artifact: Mapping[str, Any] | None = None,
    base_review_plan: Mapping[str, Any],
    base_label_corrections: Mapping[str, Any],
    followup_review_plan: Mapping[str, Any],
    followup_label_corrections: Mapping[str, Any],
    configurations: Sequence[Mapping[str, Any]] | None = None,
    acceptance_threshold: float = 0.85,
) -> dict[str, Any]:
    """Verify the complete review chain before temporal representation screening."""

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
        followup_plan.get("parent_plan_sha256") != base_plan["artifact_sha256"]
        or followup_plan.get("parent_corrections_sha256")
        != base_corrections["artifact_sha256"]
        or followup_plan.get("sealed_blind_video_sha256s")
        != base_plan["sealed_blind_video_sha256s"]
    ):
        raise ValueError("visual-state follow-up chain is invalid")

    examples, provenance = build_visual_state_temporal_examples(
        pre_anchor_artifact=pre_anchor_artifact,
        anchor_artifact=anchor_artifact,
        post_anchor_artifact=post_anchor_artifact,
        wide_anchor_artifact=wide_anchor_artifact,
        base_decisions=base_corrections["decisions"],
        followup_decisions=followup_corrections["decisions"],
    )
    if (
        set(provenance["sealed_blind_video_sha256s"])
        != set(base_plan["sealed_blind_video_sha256s"])
        or {row["source_video_sha256"] for row in examples}
        != set(base_plan["source_video_sha256s"])
    ):
        raise ValueError("temporal artifacts do not match review-plan provenance")
    return screen_visual_state_temporal_examples(
        examples=examples,
        provenance=provenance,
        label_correction_artifact_sha256s=[
            base_corrections["artifact_sha256"],
            followup_corrections["artifact_sha256"],
        ],
        configurations=configurations,
        acceptance_threshold=acceptance_threshold,
    )


def build_visual_state_temporal_examples(
    *,
    pre_anchor_artifact: Mapping[str, Any],
    anchor_artifact: Mapping[str, Any],
    post_anchor_artifact: Mapping[str, Any],
    wide_anchor_artifact: Mapping[str, Any] | None = None,
    base_decisions: Sequence[Mapping[str, Any]],
    followup_decisions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exact-join three temporal artifacts and the reviewed correction overlay."""

    artifacts = [
        verify_anchor_state_embedding_artifact(pre_anchor_artifact),
        verify_anchor_state_embedding_artifact(anchor_artifact),
        verify_anchor_state_embedding_artifact(post_anchor_artifact),
    ]
    expected_offsets: tuple[tuple[float, ...], ...] = (
        PRE_ANCHOR_OFFSETS_SECONDS,
        ANCHOR_OFFSETS_SECONDS,
        POST_ANCHOR_OFFSETS_SECONDS,
    )
    if wide_anchor_artifact is not None:
        artifacts.append(
            verify_anchor_state_embedding_artifact(wide_anchor_artifact)
        )
        expected_offsets += (WIDE_ANCHOR_OFFSETS_SECONDS,)
    if any(
        tuple(artifact["anchor_offsets_seconds"]) != offsets
        for artifact, offsets in zip(artifacts, expected_offsets, strict=True)
    ):
        raise ValueError("temporal artifacts do not match the fixed offset protocols")

    provenance_fields = (
        "backbone",
        "backbone_sha256",
        "embedding_dimension",
        "source_video_sha256s",
        "sealed_blind_video_sha256s",
    )
    for field in provenance_fields:
        if any(artifact[field] != artifacts[0][field] for artifact in artifacts[1:]):
            raise ValueError(f"temporal artifact {field} provenance mismatch")

    indexed = [_example_index(artifact["examples"]) for artifact in artifacts]
    keys = set(indexed[0])
    if any(set(rows) != keys for rows in indexed[1:]):
        raise ValueError("temporal artifacts require exact event coverage")

    base_by_key = _decision_index(base_decisions, field="base")
    unresolved = {
        key for key, row in base_by_key.items() if row.get("corrected_state") is None
    }
    followup_by_key = _decision_index(followup_decisions, field="follow-up")
    if set(followup_by_key) != unresolved:
        raise ValueError("follow-up decisions must exactly cover unresolved base decisions")
    if set(base_by_key) != keys:
        raise ValueError("review decisions must exactly cover temporal artifacts")

    rows = []
    for key in sorted(keys):
        decision = followup_by_key.get(key, base_by_key[key])
        label = decision.get("corrected_state")
        if label is None:
            continue
        if label not in VISUAL_STATE_LABELS:
            raise ValueError("invalid reviewed visual-state label")

        by_offset: dict[float, list[float]] = {}
        for artifact, artifact_rows in zip(artifacts, indexed, strict=True):
            example = artifact_rows[key]
            for offset, embedding in zip(
                artifact["anchor_offsets_seconds"],
                example["embeddings"],
                strict=True,
            ):
                offset_value = float(offset)
                values = [float(value) for value in embedding]
                prior = by_offset.get(offset_value)
                if prior is not None and prior != values:
                    raise ValueError("shared temporal offsets have different embeddings")
                by_offset[offset_value] = values
        offsets = sorted(by_offset)
        rows.append(
            {
                "source_video_sha256": key[0],
                "candidate_bundle_sha256": key[1],
                "event_id": key[2],
                "label": label,
                "offsets_seconds": offsets,
                "embeddings": [by_offset[offset] for offset in offsets],
            }
        )
    if not rows:
        raise ValueError("temporal screening requires resolved examples")

    provenance = {
        "backbone": artifacts[0]["backbone"],
        "backbone_sha256": artifacts[0]["backbone_sha256"],
        "embedding_dimension": artifacts[0]["embedding_dimension"],
        "embedding_artifact_sha256s": [
            artifact["artifact_sha256"] for artifact in artifacts
        ],
        "sealed_blind_video_sha256s": artifacts[0][
            "sealed_blind_video_sha256s"
        ],
    }
    return rows, provenance


def screen_visual_state_temporal_examples(
    *,
    examples: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    label_correction_artifact_sha256s: Sequence[str],
    configurations: Sequence[Mapping[str, Any]] | None = None,
    acceptance_threshold: float = 0.85,
) -> dict[str, Any]:
    """Select temporal representation only inside each outer training split."""

    if not 0.5 <= float(acceptance_threshold) <= 1.0:
        raise ValueError("acceptance threshold must be between 0.5 and 1.0")
    backbone = str(provenance.get("backbone") or "")
    backbone_sha256 = _require_sha256(
        provenance.get("backbone_sha256"),
        field="backbone",
    )
    embedding_artifact_sha256s = _hash_list(
        provenance.get("embedding_artifact_sha256s"),
        field="embedding artifact",
    )
    correction_hashes = _hash_list(
        label_correction_artifact_sha256s,
        field="label correction artifact",
    )
    if not backbone or int(provenance.get("embedding_dimension", 0)) < 1:
        raise ValueError("temporal embedding provenance is incomplete")

    labels, groups, event_ids, representations = _temporal_arrays(examples)
    unique_groups = sorted(set(groups.tolist()))
    if len(unique_groups) < 4:
        raise ValueError("temporal screening requires at least four target groups")
    for group in unique_groups:
        if set(labels[groups == group].tolist()) != {0, 1}:
            raise ValueError("every target group must contain both visual states")
    configs = _normalize_configurations(
        configurations or _default_configurations(representations)
    )
    if any(config["representation"] not in representations for config in configs):
        raise ValueError("configured temporal representation is unavailable")

    outer_predictions = np.zeros(len(labels), dtype=np.int64)
    outer_probabilities = np.zeros(len(labels), dtype=np.float64)
    outer_folds = []
    for held_group in unique_groups:
        held_mask = groups == held_group
        development_groups = [
            group for group in unique_groups if group != held_group
        ]
        selection = _select_configuration(
            configs=configs,
            representations=representations,
            labels=labels,
            groups=groups,
            development_groups=development_groups,
        )
        config = selection["configuration"]
        features = representations[config["representation"]]
        development_mask = np.isin(groups, development_groups)
        estimator = _fit_estimator(
            features[development_mask],
            labels[development_mask],
            config=config,
        )
        probabilities = estimator.predict_proba(features[held_mask])[:, 1]
        predictions = (probabilities >= 0.5).astype(np.int64)
        outer_predictions[held_mask] = predictions
        outer_probabilities[held_mask] = probabilities
        outer_folds.append(
            {
                "held_target_group": held_group,
                "training_target_groups": development_groups,
                "selection_target_groups": development_groups,
                "selected_configuration": config,
                "inner_selection": selection["metrics"],
                "held_metrics": _metrics(
                    labels[held_mask],
                    predictions,
                    probabilities,
                ),
                "held_example_count": int(held_mask.sum()),
            }
        )

    pooled = _metrics(labels, outer_predictions, outer_probabilities)
    per_game = {
        group: _metrics(
            labels[groups == group],
            outer_predictions[groups == group],
            outer_probabilities[groups == group],
        )
        for group in unique_groups
    }
    worst_game = min(row["balanced_accuracy"] for row in per_game.values())
    metrics = {
        **pooled,
        "worst_game_balanced_accuracy": worst_game,
        "per_game": per_game,
    }
    accepted = (
        worst_game >= acceptance_threshold
        and pooled["f1"] >= acceptance_threshold
    )
    artifact: dict[str, Any] = {
        "schema_version": VISUAL_STATE_TEMPORAL_SCREEN_SCHEMA,
        "purpose": "training_only_temporal_visual_state_screen",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "backbone": backbone,
        "backbone_sha256": backbone_sha256,
        "embedding_dimension": int(provenance["embedding_dimension"]),
        "embedding_artifact_sha256s": embedding_artifact_sha256s,
        "label_correction_artifact_sha256s": correction_hashes,
        "target_example_count": int(len(labels)),
        "target_group_count": int(len(unique_groups)),
        "target_event_ids_sha256": _canonical_sha256(event_ids),
        "protocol": {
            "outer_split": "leave_one_target_video_out",
            "inner_selection": "leave_one_remaining_target_video_out",
            "decision_threshold": 0.5,
            "pca_solver": "randomized",
            "pca_whiten": True,
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
    _validate_screen(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_visual_state_temporal_screen(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_screen(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("visual-state temporal screen hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _temporal_arrays(
    examples: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]], dict[str, np.ndarray]]:
    if not examples:
        raise ValueError("temporal examples are required")
    labels = []
    groups = []
    event_ids = []
    representation_rows = {
        name: []
        for name in (
            "anchor_summary",
            "pre_summary",
            "full_summary",
            "full_flat",
            "wide_summary",
            "wide_flat",
        )
    }
    wide_mode: bool | None = None
    seen = set()
    for row in examples:
        source = _require_sha256(row.get("source_video_sha256"), field="source video")
        event_id = str(row.get("event_id") or "")
        label = str(row.get("label") or "")
        key = (source, event_id)
        if not event_id or key in seen or label not in VISUAL_STATE_LABELS:
            raise ValueError("invalid or duplicate temporal example")
        seen.add(key)
        offsets = [float(value) for value in row.get("offsets_seconds") or []]
        embeddings = np.asarray(row.get("embeddings"), dtype=np.float64)
        if (
            offsets != sorted(set(offsets))
            or embeddings.ndim != 2
            or embeddings.shape[0] != len(offsets)
            or not np.isfinite(embeddings).all()
        ):
            raise ValueError("invalid temporal embedding sequence")
        by_offset = {offset: embeddings[index] for index, offset in enumerate(offsets)}
        required = set(
            PRE_ANCHOR_OFFSETS_SECONDS + (2.0,) + POST_ANCHOR_OFFSETS_SECONDS
        )
        wide_required = required | set(WIDE_ANCHOR_OFFSETS_SECONDS)
        if set(offsets) not in (required, wide_required):
            raise ValueError("temporal embedding offsets are incomplete")
        has_wide = set(offsets) == wide_required
        if wide_mode is None:
            wide_mode = has_wide
        elif wide_mode != has_wide:
            raise ValueError("temporal examples mix offset protocols")
        sequences = {
            "anchor_summary": np.stack([by_offset[-2.0], by_offset[0.0], by_offset[2.0]]),
            "pre_summary": np.stack(
                [by_offset[offset] for offset in PRE_ANCHOR_OFFSETS_SECONDS]
            ),
            "full_summary": embeddings,
        }
        for name, sequence in sequences.items():
            representation_rows[name].append(_summary_features(sequence))
        representation_rows["full_flat"].append(embeddings.reshape(-1))
        if has_wide:
            wide_sequence = np.stack(
                [by_offset[offset] for offset in WIDE_ANCHOR_OFFSETS_SECONDS]
            )
            representation_rows["wide_summary"].append(
                _summary_features(wide_sequence)
            )
            representation_rows["wide_flat"].append(wide_sequence.reshape(-1))
        labels.append(1 if label == "free_throw" else 0)
        groups.append(source)
        event_ids.append({"source_video_sha256": source, "event_id": event_id})
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups),
        event_ids,
        {
            name: np.asarray(values, dtype=np.float64)
            for name, values in representation_rows.items()
            if values
        },
    )


def _summary_features(sequence: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            sequence.mean(axis=0),
            sequence.std(axis=0),
            sequence[-1] - sequence[0],
        )
    )


def _select_configuration(
    *,
    configs: Sequence[dict[str, Any]],
    representations: Mapping[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    development_groups: Sequence[str],
) -> dict[str, Any]:
    candidates = []
    for config in configs:
        features = representations[config["representation"]]
        predictions = []
        probabilities = []
        inner_labels = []
        per_game = {}
        for held_group in development_groups:
            validation_mask = groups == held_group
            training_mask = np.isin(
                groups,
                [group for group in development_groups if group != held_group],
            )
            estimator = _fit_estimator(
                features[training_mask],
                labels[training_mask],
                config=config,
            )
            group_probabilities = estimator.predict_proba(features[validation_mask])[:, 1]
            group_predictions = (group_probabilities >= 0.5).astype(np.int64)
            group_labels = labels[validation_mask]
            per_game[held_group] = _metrics(
                group_labels,
                group_predictions,
                group_probabilities,
            )
            predictions.extend(group_predictions.tolist())
            probabilities.extend(group_probabilities.tolist())
            inner_labels.extend(group_labels.tolist())
        pooled = _metrics(
            np.asarray(inner_labels),
            np.asarray(predictions),
            np.asarray(probabilities),
        )
        candidates.append(
            {
                "configuration": config,
                "metrics": {
                    **pooled,
                    "worst_game_balanced_accuracy": min(
                        row["balanced_accuracy"] for row in per_game.values()
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


def _fit_estimator(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    config: Mapping[str, Any],
) -> Pipeline:
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("every training fold must contain both visual states")
    components = min(
        int(config["pca_components"]),
        int(features.shape[0] - 1),
        int(features.shape[1]),
    )
    if components < 1:
        raise ValueError("PCA requires at least two training examples")
    estimator = Pipeline(
        [
            (
                "pca",
                PCA(
                    n_components=components,
                    whiten=True,
                    svd_solver="randomized",
                    random_state=0,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=float(config["c"]),
                    class_weight="balanced",
                    l1_ratio=0.0,
                    max_iter=2000,
                    random_state=0,
                    solver="liblinear",
                ),
            ),
        ]
    )
    estimator.fit(features, labels)
    return estimator


def _normalize_configurations(
    configurations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized = []
    for row in configurations:
        representation = str(row.get("representation") or "")
        c_value = float(row.get("c", 0.0))
        components = row.get("pca_components")
        if (
            representation not in TEMPORAL_REPRESENTATIONS
            or not math.isfinite(c_value)
            or c_value <= 0
            or isinstance(components, bool)
            or int(components or 0) < 1
        ):
            raise ValueError("invalid temporal-screen configuration")
        normalized.append(
            {
                "representation": representation,
                "c": c_value,
                "pca_components": int(components),
            }
        )
    if not normalized:
        raise ValueError("temporal screening requires configurations")
    if len({_canonical_sha256(row) for row in normalized}) != len(normalized):
        raise ValueError("duplicate temporal-screen configuration")
    return normalized


def _default_configurations(
    representations: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    return [
        {
            "representation": representation,
            "c": c_value,
            "pca_components": components,
        }
        for representation in representations
        for c_value in (0.01, 0.1, 1.0)
        for components in (8, 16, 32)
    ]


def _metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def _validate_screen(artifact: Mapping[str, Any]) -> None:
    if (
        artifact.get("schema_version") != VISUAL_STATE_TEMPORAL_SCREEN_SCHEMA
        or artifact.get("purpose") != "training_only_temporal_visual_state_screen"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("invalid visual-state temporal screen")
    _require_sha256(artifact.get("backbone_sha256"), field="backbone")
    _hash_list(artifact.get("embedding_artifact_sha256s"), field="embedding artifact")
    _hash_list(
        artifact.get("label_correction_artifact_sha256s"),
        field="label correction artifact",
    )
    groups = int(artifact.get("target_group_count", 0))
    examples = int(artifact.get("target_example_count", 0))
    folds = artifact.get("outer_folds")
    if groups < 4 or examples < groups * 2 or not isinstance(folds, list):
        raise ValueError("invalid temporal-screen coverage")
    if len(folds) != groups:
        raise ValueError("temporal-screen fold count is inconsistent")
    for fold in folds:
        held = fold.get("held_target_group")
        if (
            held in fold.get("training_target_groups", [])
            or held in fold.get("selection_target_groups", [])
        ):
            raise ValueError("held game leaked into temporal training")
    metrics = artifact.get("metrics")
    gate = artifact.get("acceptance_gate")
    if not isinstance(metrics, Mapping) or not isinstance(gate, Mapping):
        raise ValueError("temporal-screen metrics are missing")
    expected_accepted = (
        float(metrics["worst_game_balanced_accuracy"])
        >= float(gate["minimum_game_balanced_accuracy"])
        and float(metrics["f1"]) >= float(gate["minimum_pooled_f1"])
    )
    if bool(artifact.get("accepted")) != expected_accepted:
        raise ValueError("temporal-screen acceptance is inconsistent")


def _example_index(
    examples: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    indexed = {}
    for row in examples:
        key = _event_key(row)
        if key in indexed:
            raise ValueError("duplicate temporal embedding event")
        indexed[key] = row
    return indexed


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
    source = _require_sha256(row.get("source_video_sha256"), field="source video")
    bundle = _require_sha256(
        row.get("candidate_bundle_sha256"),
        field="candidate bundle",
    )
    event_id = str(row.get("event_id") or "")
    if not event_id:
        raise ValueError("event id is required")
    return source, bundle, event_id


def _hash_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} hashes are required")
    hashes = [_require_sha256(item, field=field) for item in value]
    if len(set(hashes)) != len(hashes):
        raise ValueError(f"duplicate {field} hash")
    return hashes


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value or "")
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"invalid {field} SHA-256")
    return text


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
