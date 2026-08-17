"""Strict held-game screen for cut-aligned dense-video state embeddings."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from app.analysis.cut_aligned_video_state import (
    CUT_ALIGNED_SEGMENT_ROLES,
    verify_cut_aligned_video_embedding_artifact,
)
from app.analysis.visual_state_temporal_screen import (
    VISUAL_STATE_LABELS,
    _fit_estimator,
    _metrics,
    _select_configuration,
    _summary_features,
)

CUT_ALIGNED_VIDEO_STATE_SCREEN_SCHEMA = "agu.cut-aligned-video-state-screen.v1"
CUT_ALIGNED_REPRESENTATIONS = (
    "anchor_embedding",
    "neighbor_summary",
    "role_concat",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def build_cut_aligned_video_state_examples(
    *,
    embedding_artifacts: Sequence[Mapping[str, Any]],
    base_decisions: Sequence[Mapping[str, Any]],
    followup_decisions: Sequence[Mapping[str, Any]],
    sealed_blind_video_sha256s: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exact-join complete clip artifacts with reviewed development labels."""

    if not embedding_artifacts or not base_decisions:
        raise ValueError("cut-aligned screening requires embeddings and decisions")
    artifacts = [
        verify_cut_aligned_video_embedding_artifact(row)
        for row in embedding_artifacts
    ]
    if not all(artifact["complete"] is True for artifact in artifacts):
        raise ValueError("cut-aligned screening rejects incomplete embeddings")
    sealed = {
        _require_sha256(value, field="sealed blind video")
        for value in sealed_blind_video_sha256s
    }
    shared_fields = (
        "review_plan_sha256",
        "backbone",
        "backbone_sha256",
        "embedding_dimension",
        "clip_frames",
    )
    for field in shared_fields:
        if any(artifact[field] != artifacts[0][field] for artifact in artifacts[1:]):
            raise ValueError(f"cut-aligned embedding {field} mismatch")

    embedded = {}
    embedding_hashes = []
    transition_hashes = []
    for artifact in artifacts:
        source = _require_sha256(
            artifact["raw_video_sha256"],
            field="embedding source video",
        )
        if source in sealed:
            raise ValueError("cut-aligned screening rejects sealed blind source")
        embedding_hashes.append(artifact["artifact_sha256"])
        transition_hashes.append(artifact["transition_artifact_sha256"])
        for row in artifact["examples"]:
            key = _event_key(row)
            if key in embedded:
                raise ValueError("duplicate cut-aligned embedding event")
            embedded[key] = row
    if len(set(embedding_hashes)) != len(embedding_hashes):
        raise ValueError("duplicate cut-aligned embedding artifact")
    if len(set(transition_hashes)) != len(transition_hashes):
        raise ValueError("duplicate cut-aligned transition artifact")

    base = _decision_index(base_decisions, field="base")
    unresolved = {
        key for key, row in base.items() if row.get("corrected_state") is None
    }
    followup = _decision_index(followup_decisions, field="follow-up")
    if set(followup) != unresolved:
        raise ValueError("follow-up decisions must exactly cover unresolved labels")
    if set(base) != set(embedded):
        raise ValueError("review decisions must exactly cover clip embeddings")

    examples = []
    for key in sorted(embedded):
        decision = followup.get(key, base[key])
        label = decision.get("corrected_state")
        if label is None:
            continue
        if label not in VISUAL_STATE_LABELS:
            raise ValueError("invalid cut-aligned visual-state label")
        row = dict(embedded[key])
        row["label"] = label
        examples.append(row)
    if not examples:
        raise ValueError("cut-aligned screening has no resolved examples")
    provenance = {
        "review_plan_sha256": artifacts[0]["review_plan_sha256"],
        "backbone": artifacts[0]["backbone"],
        "backbone_sha256": artifacts[0]["backbone_sha256"],
        "embedding_dimension": artifacts[0]["embedding_dimension"],
        "clip_frames": artifacts[0]["clip_frames"],
        "embedding_artifact_sha256s": sorted(embedding_hashes),
        "transition_artifact_sha256s": sorted(transition_hashes),
    }
    return examples, provenance


def screen_cut_aligned_video_state_examples(
    *,
    examples: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    label_correction_artifact_sha256s: Sequence[str],
    configurations: Sequence[Mapping[str, Any]] | None = None,
    acceptance_threshold: float = 0.85,
) -> dict[str, Any]:
    """Select the clip representation only inside nested game-held folds."""

    if not 0.5 <= float(acceptance_threshold) <= 1.0:
        raise ValueError("acceptance threshold must be between 0.5 and 1.0")
    backbone = str(provenance.get("backbone") or "").strip()
    if not backbone:
        raise ValueError("cut-aligned backbone provenance is required")
    embedding_dimension = int(provenance.get("embedding_dimension", 0))
    clip_frames = int(provenance.get("clip_frames", 0))
    if embedding_dimension < 1 or clip_frames < 1:
        raise ValueError("cut-aligned embedding provenance is invalid")
    labels, groups, event_ids, representations = _feature_arrays(examples)
    unique_groups = sorted(set(groups.tolist()))
    if len(unique_groups) < 4:
        raise ValueError("cut-aligned screening requires at least four games")
    for group in unique_groups:
        if set(labels[groups == group].tolist()) != {0, 1}:
            raise ValueError("every cut-aligned game must contain both labels")
    configs = _normalize_configurations(
        configurations or _default_configurations(representations)
    )

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
    metrics = {
        **pooled,
        "worst_game_balanced_accuracy": min(
            row["balanced_accuracy"] for row in per_game.values()
        ),
        "per_game": per_game,
    }
    accepted = (
        metrics["worst_game_balanced_accuracy"] >= acceptance_threshold
        and metrics["f1"] >= acceptance_threshold
    )
    artifact: dict[str, Any] = {
        "schema_version": CUT_ALIGNED_VIDEO_STATE_SCREEN_SCHEMA,
        "purpose": "training_only_cut_aligned_video_state_screen",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "transition_semantics_used": False,
        "review_plan_sha256": _require_sha256(
            provenance.get("review_plan_sha256"),
            field="review plan",
        ),
        "backbone": backbone,
        "backbone_sha256": _require_sha256(
            provenance.get("backbone_sha256"),
            field="backbone",
        ),
        "embedding_dimension": embedding_dimension,
        "clip_frames": clip_frames,
        "embedding_artifact_sha256s": _hash_list(
            provenance.get("embedding_artifact_sha256s"),
            field="embedding artifact",
        ),
        "transition_artifact_sha256s": _hash_list(
            provenance.get("transition_artifact_sha256s"),
            field="transition artifact",
        ),
        "label_correction_artifact_sha256s": _hash_list(
            label_correction_artifact_sha256s,
            field="label correction artifact",
        ),
        "target_example_count": len(labels),
        "target_group_count": len(unique_groups),
        "target_event_ids_sha256": _canonical_sha256(event_ids),
        "protocol": {
            "outer_split": "leave_one_target_video_out",
            "inner_selection": "leave_one_remaining_target_video_out",
            "decision_threshold": 0.5,
            "transition_semantics_used": False,
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


def verify_cut_aligned_video_state_screen(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_screen(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("cut-aligned video state screen hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _feature_arrays(
    examples: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]], dict[str, np.ndarray]]:
    rows = {name: [] for name in CUT_ALIGNED_REPRESENTATIONS}
    labels = []
    groups = []
    event_ids = []
    seen = set()
    expected_dimension = None
    for example in examples:
        key = _event_key(example)
        label = str(example.get("label") or "")
        if key in seen or label not in VISUAL_STATE_LABELS:
            raise ValueError("invalid or duplicate cut-aligned screen example")
        seen.add(key)
        by_role = {
            str(segment["role"]): segment for segment in example["segments"]
        }
        if set(by_role) != set(CUT_ALIGNED_SEGMENT_ROLES):
            raise ValueError("cut-aligned screen segment roles are incomplete")
        anchor = np.asarray(by_role["anchor"]["embedding"], dtype=np.float64)
        if anchor.ndim != 1 or not np.isfinite(anchor).all():
            raise ValueError("invalid cut-aligned anchor embedding")
        if expected_dimension is None:
            expected_dimension = len(anchor)
        elif len(anchor) != expected_dimension:
            raise ValueError("cut-aligned embedding dimensions differ")
        vectors = []
        presence = []
        durations = []
        for role in CUT_ALIGNED_SEGMENT_ROLES:
            segment = by_role[role]
            available = bool(segment["available"])
            vector = (
                np.asarray(segment["embedding"], dtype=np.float64)
                if available
                else anchor
            )
            if vector.shape != anchor.shape or not np.isfinite(vector).all():
                raise ValueError("invalid cut-aligned role embedding")
            vectors.append(vector)
            presence.append(1.0 if available else 0.0)
            durations.append(
                max(
                    0.0,
                    float(segment["end_offset_seconds"])
                    - float(segment["start_offset_seconds"]),
                )
            )
        geometry = np.asarray([*presence, *durations], dtype=np.float64)
        sequence = np.stack(vectors)
        rows["anchor_embedding"].append(anchor)
        rows["neighbor_summary"].append(
            np.concatenate((_summary_features(sequence), geometry))
        )
        rows["role_concat"].append(
            np.concatenate((sequence.reshape(-1), geometry))
        )
        labels.append(1 if label == "free_throw" else 0)
        groups.append(key[0])
        event_ids.append({"source_video_sha256": key[0], "event_id": key[2]})
    if not rows["anchor_embedding"]:
        raise ValueError("cut-aligned screen examples are required")
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups),
        event_ids,
        {name: np.asarray(values) for name, values in rows.items()},
    )


def _normalize_configurations(
    configurations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized = []
    for row in configurations:
        representation = str(row.get("representation") or "")
        c_value = float(row.get("c", 0.0))
        components = row.get("pca_components")
        if (
            representation not in CUT_ALIGNED_REPRESENTATIONS
            or not math.isfinite(c_value)
            or c_value <= 0
            or isinstance(components, bool)
            or int(components or 0) < 1
        ):
            raise ValueError("invalid cut-aligned screen configuration")
        normalized.append(
            {
                "representation": representation,
                "c": c_value,
                "pca_components": int(components),
            }
        )
    if not normalized or len(
        {_canonical_sha256(row) for row in normalized}
    ) != len(normalized):
        raise ValueError("duplicate or missing cut-aligned configurations")
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


def _validate_screen(artifact: Mapping[str, Any]) -> None:
    if (
        artifact.get("schema_version") != CUT_ALIGNED_VIDEO_STATE_SCREEN_SCHEMA
        or artifact.get("purpose")
        != "training_only_cut_aligned_video_state_screen"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("transition_semantics_used") is not False
    ):
        raise ValueError("invalid cut-aligned video state screen")
    if (
        not str(artifact.get("backbone") or "").strip()
        or int(artifact.get("embedding_dimension", 0)) < 1
        or int(artifact.get("clip_frames", 0)) < 1
    ):
        raise ValueError("invalid cut-aligned embedding provenance")
    _require_sha256(artifact.get("review_plan_sha256"), field="review plan")
    _require_sha256(artifact.get("backbone_sha256"), field="backbone")
    _hash_list(
        artifact.get("embedding_artifact_sha256s"),
        field="embedding artifact",
    )
    _hash_list(
        artifact.get("transition_artifact_sha256s"),
        field="transition artifact",
    )
    _hash_list(
        artifact.get("label_correction_artifact_sha256s"),
        field="label correction artifact",
    )
    groups = int(artifact.get("target_group_count", 0))
    examples = int(artifact.get("target_example_count", 0))
    folds = artifact.get("outer_folds")
    if groups < 4 or examples < groups * 2 or not isinstance(folds, list):
        raise ValueError("invalid cut-aligned screen coverage")
    if len(folds) != groups:
        raise ValueError("cut-aligned screen fold count is inconsistent")
    for fold in folds:
        held = fold.get("held_target_group")
        if (
            held in fold.get("training_target_groups", [])
            or held in fold.get("selection_target_groups", [])
        ):
            raise ValueError("held game leaked into cut-aligned training")
    metrics = artifact.get("metrics")
    gate = artifact.get("acceptance_gate")
    if not isinstance(metrics, Mapping) or not isinstance(gate, Mapping):
        raise ValueError("cut-aligned screen metrics are missing")
    expected = (
        float(metrics["worst_game_balanced_accuracy"])
        >= float(gate["minimum_game_balanced_accuracy"])
        and float(metrics["f1"]) >= float(gate["minimum_pooled_f1"])
    )
    if bool(artifact.get("accepted")) != expected:
        raise ValueError("cut-aligned screen acceptance is inconsistent")


def _decision_index(
    decisions: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    indexed = {}
    for row in decisions:
        key = _event_key(row)
        if key in indexed:
            raise ValueError(f"duplicate {field} decision")
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
        raise ValueError("cut-aligned event id is required")
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
