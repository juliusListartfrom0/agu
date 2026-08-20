"""Offline, game-held evidence gating for independent shot-VLM screens.

This module is deliberately a research contract. It can abstain on a VLM
decision when a required visual observable is contradicted or when a
game-held auxiliary score is unavailable, but it must never become an AGU
runtime answer source. Inputs are fail-closed: every VLM event must have an
auxiliary OOF row, required observables must be unique, and numeric scores
must be finite probabilities in ``[0, 1]``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from app.analysis.independent_shot_vlm import (
    OBSERVABLE_FIELDS,
    seal_independent_shot_vlm_predictions,
    verify_independent_shot_vlm_predictions,
    verify_nested_fusion_shot_vlm_plan,
)

EVIDENCE_GATE_SCHEMA = "agu.independent-shot-vlm-evidence-gate.v1"
DEFAULT_REQUIRED_OBSERVABLES = (
    "continuous_live_play",
    "controlled_ball_before_release",
    "ball_separates_from_hands",
)
_KEY_FIELDS = ("source_video_sha256", "candidate_bundle_sha256", "event_id")
_AUXILIARY_KEY_FIELDS = _KEY_FIELDS
_NESTED_AUXILIARY_SCHEMA = "agu.shot-validity-scene-fusion-screen.v2"
_NESTED_SELECTION_PROTOCOL = "nested_outer_game_held_v2"
_NESTED_AUXILIARY_PURPOSE = "scene_video_fusion_screening_training_only"
_NESTED_THRESHOLD_SELECTION = (
    "nested_inner_game_held_minimum_recall_0_85"
)
_PLAN_BOUND_SUBSET_SELECTION_PROTOCOL = (
    "sealed_independent_shot_vlm_plan_three_part_key_order_v1"
)
_FROZEN_AUXILIARY_ROW_FIELDS = frozenset((*_KEY_FIELDS, "probability"))
_AUXILIARY_TOP_FIELDS = frozenset(
    {
        "artifact_sha256",
        "codex_runtime_answer_used",
        "fusion_screen_contract",
        "fusion_screen_contract_sha256",
        "gate",
        "outer_folds",
        "pca_components",
        "plan_sha256",
        "predeclared_variants",
        "promotion_requirements",
        "purpose",
        "random_seed",
        "regularization_c",
        "runtime_consumable",
        "scene_embedding_artifact_sha256",
        "scene_subset_embedding_artifact_sha256",
        "schema_version",
        "selection_protocol",
        "subset_selection_protocol",
        "threshold_selection",
        "training_manifest_sha256",
        "video_embedding_artifact_sha256s",
        "video_subset_embedding_artifact_sha256s",
    }
)
_OUTER_FOLD_FIELDS = frozenset(
    {
        "fit_game_sha256s",
        "held_game_sha256",
        "held_metrics",
        "inner_selection_game_sha256s",
        "inner_selection_gate",
        "oof_predictions",
        "selected_variant",
        "threshold",
        "variant_selection",
    }
)
_VARIANT_SELECTION_FIELDS = frozenset(
    {"gate", "inner_folds", "input_dimension", "name", "threshold"}
)
_INNER_FOLD_FIELDS = frozenset(
    {"fit_game_sha256s", "held_game_sha256", "row_count"}
)
_GATE_FIELDS = frozenset({"per_game", "pooled", "promoted"})
_METRIC_FIELDS = frozenset(
    {"f1", "fn", "fp", "precision", "recall", "tn", "tp"}
)
_VLM_TOP_REQUIRED_FIELDS = frozenset(
    {
        "artifact_sha256",
        "codex_runtime_answer_used",
        "model",
        "plan_sha256",
        "predictions",
        "purpose",
        "runtime_consumable",
        "schema_version",
    }
)
_VLM_TOP_OPTIONAL_FIELDS = frozenset(
    {"prompt_sha256", "prompt_variant", "sampling"}
)
_VLM_MODEL_FIELDS = frozenset(
    {"independent_from_codex", "name", "revision", "source", "weights_sha256"}
)
_VLM_ROW_REQUIRED_FIELDS = frozenset(
    {*_KEY_FIELDS, "confidence", "field_goal_state"}
)
_VLM_ROW_OPTIONAL_FIELDS = frozenset(
    {"available", "observables", "raw_response", "reason"}
)
_VLM_SAMPLING_FIELDS = frozenset(
    {
        "context_length",
        "image_width",
        "max_frames",
        "max_pixels",
        "native_temporal_position_encoding",
        "sample_fps",
    }
)
_NESTED_PROMOTION_REQUIREMENTS = {
    "minimum_pooled_precision": 0.85,
    "minimum_pooled_recall": 0.85,
    "minimum_per_game_precision": 0.85,
    "minimum_per_game_recall": 0.85,
}
_FORBIDDEN_LABEL_FIELDS = frozenset(
    {
        "annotation",
        "annotations",
        "actual",
        "class",
        "event_present",
        "ground_truth",
        "gold_label",
        "is_made",
        "is_positive",
        "label",
        "labels",
        "made",
        "missed",
        "outcome",
        "review_note",
        "review_notes",
        "shot_outcome",
        "shot_sequence",
        "target",
        "target_label",
        "target_labels",
        "target_truth",
        "true_label",
        "truth",
        "y_true",
    }
)


def apply_evidence_gate(
    row: Mapping[str, Any],
    *,
    auxiliary_probability: float | None,
    threshold: float,
    required_observables: Sequence[str] = DEFAULT_REQUIRED_OBSERVABLES,
) -> dict[str, Any]:
    """Apply one conservative VLM-plus-auxiliary decision without labels."""

    threshold_value = _strict_probability(
        threshold, field="evidence gate threshold"
    )
    required = tuple(str(name) for name in required_observables)
    if not required or any(not name for name in required):
        raise ValueError("evidence gate requires named observables")
    if len(set(required)) != len(required):
        raise ValueError("evidence gate required observables contain duplicate names")
    if any(name not in OBSERVABLE_FIELDS for name in required):
        raise ValueError(
            "evidence gate requirements must use canonical visual observables"
        )
    if auxiliary_probability is not None:
        auxiliary_value = _strict_probability(
            auxiliary_probability,
            field="auxiliary probability",
        )
    else:
        auxiliary_value = None
    observables = row.get("observables")
    if not isinstance(observables, Mapping):
        observables = {}
    state = str(row.get("field_goal_state") or "").strip().lower()
    try:
        confidence = float(row.get("confidence"))
    except (TypeError, ValueError):
        confidence = math.nan
    reason = "vlm_not_live"
    accepted = False
    if row.get("available") is False:
        reason = "vlm_unavailable"
    elif state == "live_field_goal":
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            reason = "vlm_confidence_invalid"
        elif any(observables.get(name) is not True for name in required):
            reason = "required_observable_missing"
        elif observables.get("replay_or_highlight") is True:
            reason = "replay_or_highlight_veto"
        elif observables.get("free_throw") is True:
            reason = "free_throw_veto"
        elif auxiliary_value is None:
            reason = "auxiliary_score_missing"
        elif auxiliary_value < threshold_value:
            reason = "auxiliary_below_group_held_threshold"
        else:
            accepted = True
            reason = "vlm_and_auxiliary_confirm"
    return {
        **{field: row.get(field) for field in _KEY_FIELDS},
        "field_goal_state": "live_field_goal" if accepted else "unknown",
        "confidence": confidence if accepted else 0.0,
        "auxiliary_probability": (
            None if auxiliary_value is None else auxiliary_value
        ),
        "threshold": threshold_value,
        "required_observables": list(required),
        "decision_reason": reason,
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
    }


def select_group_held_threshold(
    rows: Sequence[Mapping[str, Any]],
    *,
    held_group: str,
    minimum_recall: float = 0.85,
) -> dict[str, Any]:
    """Select an auxiliary threshold from groups other than ``held_group``."""

    if not held_group:
        raise ValueError("held group is required")
    if not 0.0 < float(minimum_recall) <= 1.0:
        raise ValueError("minimum recall must be within (0, 1]")
    normalized = [_normalize_auxiliary_row(row) for row in rows]
    _assert_unique_auxiliary_rows(normalized)
    training = [row for row in normalized if row["source_video_sha256"] != held_group]
    target_count = sum(row["source_video_sha256"] == held_group for row in normalized)
    if not training:
        raise ValueError("group-held threshold requires training groups")
    thresholds = sorted({0.0, 1.0, *(row["probability"] for row in training)})
    candidates = [_threshold_metrics(training, threshold) for threshold in thresholds]
    eligible = [row for row in candidates if row["recall"] >= minimum_recall]
    chosen = max(
        eligible or candidates,
        key=lambda row: (
            row["precision"],
            row["recall"],
            row["f1"],
            row["threshold"],
        ),
    )
    return {
        "held_group": held_group,
        "training_groups": sorted({row["source_video_sha256"] for row in training}),
        "training_count": len(training),
        "target_count": target_count,
        "minimum_recall": float(minimum_recall),
        **chosen,
    }


def screen_evidence_gate(
    *,
    vlm_predictions: Sequence[Mapping[str, Any]],
    auxiliary_oof_rows: Sequence[Mapping[str, Any]],
    target_truth: Mapping[tuple[str, str, str], bool],
    minimum_precision: float = 0.85,
    minimum_recall: float = 0.85,
    required_observables: Sequence[str] = DEFAULT_REQUIRED_OBSERVABLES,
) -> dict[str, Any]:
    """Screen the gate with group-held thresholds and exact target coverage."""

    if not 0.0 < float(minimum_precision) <= 1.0:
        raise ValueError("minimum precision must be within (0, 1]")
    if not 0.0 < float(minimum_recall) <= 1.0:
        raise ValueError("minimum recall must be within (0, 1]")
    vlm = [dict(row) for row in vlm_predictions]
    if not vlm:
        raise ValueError("VLM predictions are required")
    keys = [_key(row) for row in vlm]
    if len(set(keys)) != len(keys):
        raise ValueError("VLM predictions contain duplicates")
    if set(target_truth) != set(keys):
        raise ValueError("target truth must exactly cover VLM predictions")
    auxiliary = [_normalize_auxiliary_row(row) for row in auxiliary_oof_rows]
    _assert_unique_auxiliary_rows(auxiliary)
    auxiliary_lookup = {
        tuple(str(row[field]) for field in _AUXILIARY_KEY_FIELDS): row[
            "probability"
        ]
        for row in auxiliary
    }
    required_auxiliary_keys = {
        _key(row) for row in vlm
    }
    if not required_auxiliary_keys.issubset(auxiliary_lookup):
        raise ValueError("auxiliary OOF rows must exactly cover VLM events")
    groups = sorted({str(row["source_video_sha256"]) for row in vlm})
    folds = []
    output_rows: list[dict[str, Any]] = []
    counts: defaultdict[str, int] = defaultdict(int)
    per_group: dict[str, dict[str, Any]] = {}
    for held_group in groups:
        threshold = select_group_held_threshold(
            auxiliary,
            held_group=held_group,
            minimum_recall=minimum_recall,
        )
        held_rows = [row for row in vlm if str(row["source_video_sha256"]) == held_group]
        group_counts: defaultdict[str, int] = defaultdict(int)
        for row in held_rows:
            key = _key(row)
            aux_probability = auxiliary_lookup.get(key)
            decision = apply_evidence_gate(
                row,
                auxiliary_probability=aux_probability,
                threshold=float(threshold["threshold"]),
                required_observables=required_observables,
            )
            predicted = decision["field_goal_state"] == "live_field_goal"
            label = bool(target_truth[key])
            bucket = (
                "true_positive"
                if predicted and label
                else "false_positive"
                if predicted
                else "false_negative"
                if label
                else "true_negative"
            )
            counts[bucket] += 1
            group_counts[bucket] += 1
            output_rows.append(decision)
        per_group[held_group] = _metrics(group_counts)
        folds.append(threshold)
    metrics = _metrics(counts)
    observed_gate_would_have_met_thresholds = bool(
        metrics["precision"] >= minimum_precision
        and metrics["recall"] >= minimum_recall
        and all(
            row["precision"] >= minimum_precision
            and row["recall"] >= minimum_recall
            for row in per_group.values()
        )
    )
    metrics.update(
        {
            "evaluated_count": len(vlm),
            "minimum_precision": float(minimum_precision),
            "minimum_recall": float(minimum_recall),
            "promotion_eligible": False,
            "observed_gate_would_have_met_thresholds": (
                observed_gate_would_have_met_thresholds
            ),
            "per_group": per_group,
        }
    )
    payload: dict[str, Any] = {
        "schema_version": EVIDENCE_GATE_SCHEMA,
        "purpose": "offline_independent_shot_vlm_evidence_gate_screening",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "legacy_provenance_only": True,
        "promotion_eligible": False,
        "promotion_decision": "ineligible_legacy_unbound_evidence_gate_v1",
        "required_observables": list(required_observables),
        "folds": folds,
        "predictions": output_rows,
        "metrics": metrics,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return payload


def fuse_vlm_with_frozen_auxiliary(
    *,
    plan: Mapping[str, Any],
    vlm_predictions: Mapping[str, Any],
    auxiliary_artifact: Mapping[str, Any],
    expected_auxiliary_artifact_sha256: str,
    required_observables: Sequence[str] = DEFAULT_REQUIRED_OBSERVABLES,
) -> dict[str, Any]:
    """Apply a pre-frozen nested screen without reading target labels.

    The expected SHA-256 must come from outside ``auxiliary_artifact``.  The
    function verifies the producer-shaped contract but does not attest that
    the producer code actually ran, so the result remains a non-promotable,
    training-only diagnostic.
    """

    if tuple(str(name) for name in required_observables) != tuple(
        DEFAULT_REQUIRED_OBSERVABLES
    ):
        raise ValueError(
            "frozen auxiliary fusion requires the fixed default observables"
        )
    verified_plan = verify_nested_fusion_shot_vlm_plan(plan)
    verified_vlm = verify_independent_shot_vlm_predictions(vlm_predictions)
    _reject_label_fields(verified_plan, context="frozen plan")
    _reject_label_fields(verified_vlm, context="VLM predictions")
    _verify_vlm_artifact_shape(verified_vlm)
    vlm_model = verified_vlm.get("model")
    if (
        not isinstance(vlm_model, Mapping)
        or vlm_model.get("independent_from_codex") is not True
    ):
        raise ValueError("VLM model must declare that it is independent from Codex")
    for row in verified_vlm["predictions"]:
        _verify_vlm_observable_fields(row)
    if verified_vlm.get("plan_sha256") != verified_plan["plan_sha256"]:
        raise ValueError("VLM predictions do not match the frozen plan")
    plan_keys = {_key(row) for row in verified_plan["examples"]}
    vlm_keys = {_key(row) for row in verified_vlm["predictions"]}
    if any(not _is_sha256(key[0]) for key in plan_keys):
        raise ValueError("frozen plan source video SHA-256 is invalid")
    if any(not _is_sha256(key[1]) for key in plan_keys):
        raise ValueError("frozen plan candidate bundle SHA-256 is invalid")
    if vlm_keys != plan_keys:
        raise ValueError("VLM predictions must exactly cover the frozen plan")
    if auxiliary_artifact.get("runtime_consumable") is not False:
        raise ValueError("auxiliary screen must be training-only")
    if (
        auxiliary_artifact.get("schema_version") != _NESTED_AUXILIARY_SCHEMA
        or auxiliary_artifact.get("selection_protocol")
        != _NESTED_SELECTION_PROTOCOL
    ):
        raise ValueError(
            "auxiliary screen must use the nested outer-game-held v2 contract"
        )
    if auxiliary_artifact.get("purpose") != _NESTED_AUXILIARY_PURPOSE:
        raise ValueError("auxiliary screen purpose is invalid")
    if auxiliary_artifact.get("codex_runtime_answer_used") is not False:
        raise ValueError("Codex must not provide an auxiliary runtime answer")
    _reject_label_fields(auxiliary_artifact, context="auxiliary artifact")
    _verify_auxiliary_artifact_shape(auxiliary_artifact)
    expected_auxiliary_sha = expected_auxiliary_artifact_sha256
    if not _is_sha256(expected_auxiliary_sha):
        raise ValueError("expected auxiliary artifact SHA-256 is invalid")
    auxiliary_sha = auxiliary_artifact.get("artifact_sha256")
    if not _is_sha256(auxiliary_sha) or _canonical_sha256(auxiliary_artifact) != auxiliary_sha:
        raise ValueError("auxiliary artifact hash mismatch")
    if auxiliary_sha != expected_auxiliary_sha:
        raise ValueError("expected auxiliary artifact SHA-256 does not match")
    if auxiliary_artifact.get("plan_sha256") != verified_plan["plan_sha256"]:
        raise ValueError("auxiliary artifact does not match the frozen plan")
    training_manifest_sha256 = verified_plan.get("training_manifest_sha256")
    if not _is_sha256(training_manifest_sha256):
        raise ValueError("frozen plan has no valid training manifest binding")
    if (
        auxiliary_artifact.get("training_manifest_sha256")
        != training_manifest_sha256
    ):
        raise ValueError("auxiliary artifact training manifest does not match the plan")

    _verify_fusion_screen_contract(
        plan=verified_plan,
        auxiliary_artifact=auxiliary_artifact,
    )

    plan_groups = sorted({key[0] for key in plan_keys})
    if len(plan_groups) < 3 or any(not _is_sha256(group) for group in plan_groups):
        raise ValueError("frozen plan requires at least three valid game groups")
    plan_keys_by_group = {
        group: {key for key in plan_keys if key[0] == group}
        for group in plan_groups
    }
    plan_counts_by_group = {
        group: len(keys) for group, keys in plan_keys_by_group.items()
    }
    top_gate = auxiliary_artifact.get("gate")
    _nested_gate_score(
        top_gate,
        expected_groups=plan_groups,
        expected_counts=plan_counts_by_group,
    )
    expected_positive_counts = {
        group: top_gate["per_game"][group]["tp"]
        + top_gate["per_game"][group]["fn"]
        for group in plan_groups
    }
    expected_negative_counts = {
        group: top_gate["per_game"][group]["fp"]
        + top_gate["per_game"][group]["tn"]
        for group in plan_groups
    }
    predeclared_variants = _normalize_predeclared_variants(
        auxiliary_artifact.get("predeclared_variants")
    )
    contract_variants = _normalize_predeclared_variants(
        verified_plan["fusion_screen_contract"]["predeclared_variants"]
    )
    if predeclared_variants != contract_variants:
        raise ValueError("auxiliary predeclared variants differ from the plan contract")

    folds = auxiliary_artifact.get("outer_folds")
    if not isinstance(folds, list) or len(folds) != len(plan_groups):
        raise ValueError("auxiliary outer folds must exactly cover plan games")

    thresholds: dict[str, float] = {}
    auxiliary: list[dict[str, Any]] = []
    for expected_held_group, fold in zip(plan_groups, folds, strict=True):
        if not isinstance(fold, Mapping):
            raise ValueError("auxiliary screen fold is invalid")
        held_group = fold.get("held_game_sha256")
        if not _is_sha256(held_group):
            raise ValueError("auxiliary held game SHA-256 is invalid")
        if held_group != expected_held_group or held_group in thresholds:
            raise ValueError("auxiliary screen fold threshold is invalid")
        expected_development_groups = [
            group for group in plan_groups if group != held_group
        ]
        inner_groups = _normalize_game_groups(
            fold.get("inner_selection_game_sha256s"),
            field="inner-selection",
        )
        fit_groups = _normalize_game_groups(
            fold.get("fit_game_sha256s"),
            field="fit",
        )
        if (
            inner_groups != expected_development_groups
            or fit_groups != expected_development_groups
        ):
            raise ValueError(
                "auxiliary fold training groups must equal plan games minus held game"
            )
        selected_variant = _verify_variant_selection(
            fold=fold,
            predeclared_variants=predeclared_variants,
            development_groups=expected_development_groups,
            plan_keys_by_group=plan_keys_by_group,
            expected_positive_counts=expected_positive_counts,
            expected_negative_counts=expected_negative_counts,
        )
        if _metric_row_count(fold["held_metrics"]) != plan_counts_by_group[
            held_group
        ]:
            raise ValueError("auxiliary held metric row count is inconsistent")
        held_metrics = fold["held_metrics"]
        if held_metrics != auxiliary_artifact["gate"]["per_game"][held_group]:
            raise ValueError("auxiliary held metrics are inconsistent")
        try:
            threshold = _strict_probability(
                fold["threshold"],
                field="auxiliary screen fold threshold",
            )
        except KeyError as exc:
            raise ValueError("auxiliary screen fold threshold is invalid") from exc
        if threshold != selected_variant["threshold"]:
            raise ValueError("auxiliary fold threshold differs from selected variant")
        thresholds[held_group] = threshold
        oof_rows = fold.get("oof_predictions")
        if not isinstance(oof_rows, list) or not oof_rows:
            raise ValueError("auxiliary screen fold has no OOF predictions")
        for raw_row in oof_rows:
            if not isinstance(raw_row, Mapping):
                raise ValueError("auxiliary OOF rows must be objects")
            row = _normalize_frozen_auxiliary_row(raw_row)
            if row["source_video_sha256"] != held_group:
                raise ValueError("auxiliary OOF row does not belong to its held game")
            auxiliary.append(row)
        predicted_positive_count = sum(
            row["probability"] >= threshold for row in oof_rows
        )
        if held_metrics["tp"] + held_metrics["fp"] != predicted_positive_count:
            raise ValueError(
                "auxiliary held metrics disagree with frozen threshold predictions"
            )
        held_keys = {
            (
                row["source_video_sha256"],
                row["candidate_bundle_sha256"],
                row["event_id"],
            )
            for row in auxiliary
            if row["source_video_sha256"] == held_group
        }
        if held_keys != plan_keys_by_group[held_group]:
            raise ValueError(
                "auxiliary OOF rows must exactly cover the frozen plan"
            )

    _assert_unique_frozen_auxiliary_rows(auxiliary)
    auxiliary_keys = {
        (
            row["source_video_sha256"],
            row["candidate_bundle_sha256"],
            row["event_id"],
        )
        for row in auxiliary
    }
    if auxiliary_keys != plan_keys:
        unexpected_keys = auxiliary_keys - plan_keys
        plan_source_events = {(key[0], key[2]) for key in plan_keys}
        if any((key[0], key[2]) in plan_source_events for key in unexpected_keys):
            raise ValueError("auxiliary candidate bundle does not match the frozen plan")
        raise ValueError("auxiliary OOF rows must exactly cover the frozen plan")
    auxiliary_lookup = {
        (
            row["source_video_sha256"],
            row["candidate_bundle_sha256"],
            row["event_id"],
        ): row["probability"]
        for row in auxiliary
    }
    output_rows: list[dict[str, Any]] = []
    for raw_row in verified_vlm["predictions"]:
        row = dict(raw_row)
        source = row["source_video_sha256"]
        candidate_bundle = row["candidate_bundle_sha256"]
        event_id = row["event_id"]
        threshold = thresholds[source]
        auxiliary_probability = auxiliary_lookup[
            (source, candidate_bundle, event_id)
        ]
        decision = apply_evidence_gate(
            row,
            auxiliary_probability=auxiliary_probability,
            threshold=threshold,
            required_observables=required_observables,
        )
        observables = row.get("observables")
        if isinstance(observables, Mapping):
            decision["observables"] = {
                name: observables[name]
                for name in OBSERVABLE_FIELDS
                if isinstance(observables.get(name), bool)
            }
        decision["reason"] = (
            f"frozen auxiliary gate: {decision['decision_reason']}"
        )
        output_rows.append(decision)

    return seal_independent_shot_vlm_predictions(
        {
            "plan_sha256": verified_vlm["plan_sha256"],
            "model": {
                "name": "independent-vlm-plus-frozen-scene-video-oof-gate",
                "input_model_independent_from_codex_declared": True,
                "input_model_provenance_verification": (
                    "self_declared_structural_only"
                ),
                "codex_runtime_answer_used_declared_false": True,
                "fusion_rule": (
                    "vlm_live_and_auxiliary_oof_above_frozen_game_held_threshold"
                ),
                "auxiliary_artifact_sha256": auxiliary_sha,
                "expected_auxiliary_artifact_sha256": expected_auxiliary_sha,
                "input_vlm_prediction_artifact_sha256": verified_vlm[
                    "artifact_sha256"
                ],
                "selection_protocol": _NESTED_SELECTION_PROTOCOL,
                "threshold_selection": _NESTED_THRESHOLD_SELECTION,
                "thresholds_by_source_video_sha256": dict(sorted(thresholds.items())),
                "raw_target_label_fields_read_by_fuser": False,
                "auxiliary_fit_used_training_labels": True,
                "outer_held_labels_used_for_selection": False,
                "outer_held_metrics_present_in_auxiliary": True,
                "outer_held_metrics_used_for_decision": False,
                "auxiliary_provenance_verification": "structural_only",
                "formal_promotion_eligible": False,
            },
            "coverage": {
                "vlm_event_count": len(output_rows),
                "auxiliary_event_count": len(output_rows),
                "missing_auxiliary_event_ids": [],
                "missing_auxiliary_keys": [],
            },
            "predictions": output_rows,
        }
    )


def _reject_label_fields(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            normalized_key = _normalize_contract_key(key)
            if normalized_key in _FORBIDDEN_LABEL_FIELDS:
                raise ValueError(f"{context} contains forbidden label field")
            _reject_label_fields(nested_value, context=context)
    elif isinstance(value, list):
        for nested_value in value:
            _reject_label_fields(nested_value, context=context)


def _verify_vlm_artifact_shape(artifact: Mapping[str, Any]) -> None:
    fields = set(artifact)
    if not _VLM_TOP_REQUIRED_FIELDS.issubset(fields) or not fields.issubset(
        _VLM_TOP_REQUIRED_FIELDS | _VLM_TOP_OPTIONAL_FIELDS
    ):
        raise ValueError("VLM prediction artifact fields are not canonical")
    model = artifact.get("model")
    if not isinstance(model, Mapping) or not {"name", "independent_from_codex"}.issubset(
        model
    ) or not set(model).issubset(_VLM_MODEL_FIELDS):
        raise ValueError("VLM model provenance fields are not canonical")
    if not isinstance(model.get("name"), str) or not model["name"]:
        raise ValueError("VLM model provenance name is invalid")
    for field in ("revision", "source"):
        if field in model and not isinstance(model[field], str):
            raise ValueError("VLM model provenance fields are invalid")
    if "weights_sha256" in model and not _is_sha256(model["weights_sha256"]):
        raise ValueError("VLM model weights SHA-256 is invalid")
    if "prompt_sha256" in artifact and not _is_sha256(
        artifact["prompt_sha256"]
    ):
        raise ValueError("VLM prompt SHA-256 is invalid")
    if "prompt_variant" in artifact and not isinstance(
        artifact["prompt_variant"], str
    ):
        raise ValueError("VLM prompt variant is invalid")
    sampling = artifact.get("sampling")
    if sampling is not None:
        if not isinstance(sampling, Mapping) or not set(sampling).issubset(
            _VLM_SAMPLING_FIELDS
        ):
            raise ValueError("VLM sampling fields are not canonical")
        if any(
            not isinstance(value, (bool, int, float))
            for value in sampling.values()
        ):
            raise ValueError("VLM sampling values are invalid")
    rows = artifact.get("predictions")
    if not isinstance(rows, list):
        raise ValueError("VLM prediction rows are invalid")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("VLM prediction rows are invalid")
        row_fields = set(row)
        if not _VLM_ROW_REQUIRED_FIELDS.issubset(
            row_fields
        ) or not row_fields.issubset(
            _VLM_ROW_REQUIRED_FIELDS | _VLM_ROW_OPTIONAL_FIELDS
        ):
            raise ValueError("VLM prediction row fields are not canonical")
        _strict_probability(row.get("confidence"), field="VLM confidence")
        if "available" in row and not isinstance(row["available"], bool):
            raise ValueError("VLM prediction availability is invalid")
        if (
            row.get("available") is False
            and row.get("field_goal_state") != "unknown"
        ):
            raise ValueError(
                "unavailable VLM predictions must use the unknown state"
            )
        for field in ("raw_response", "reason"):
            if field in row and not isinstance(row[field], str):
                raise ValueError("VLM prediction text fields are invalid")


def _verify_auxiliary_artifact_shape(artifact: Mapping[str, Any]) -> None:
    if set(artifact) != _AUXILIARY_TOP_FIELDS:
        raise ValueError("auxiliary artifact top-level fields are not canonical")
    folds = artifact.get("outer_folds")
    if not isinstance(folds, list):
        raise ValueError("auxiliary outer folds are invalid")
    for fold in folds:
        if not isinstance(fold, Mapping) or set(fold) != _OUTER_FOLD_FIELDS:
            raise ValueError("auxiliary outer fold fields are not canonical")
        _verify_gate_shape(fold.get("inner_selection_gate"))
        _verify_metric_shape(fold.get("held_metrics"))
        variants = fold.get("variant_selection")
        if not isinstance(variants, list):
            raise ValueError("auxiliary variant selection is invalid")
        for variant in variants:
            if (
                not isinstance(variant, Mapping)
                or set(variant) != _VARIANT_SELECTION_FIELDS
            ):
                raise ValueError("auxiliary variant fields are not canonical")
            _verify_gate_shape(variant.get("gate"))
            inner_folds = variant.get("inner_folds")
            if not isinstance(inner_folds, list) or any(
                not isinstance(inner_fold, Mapping)
                or set(inner_fold) != _INNER_FOLD_FIELDS
                for inner_fold in inner_folds
            ):
                raise ValueError("auxiliary inner fold fields are not canonical")
        oof_rows = fold.get("oof_predictions")
        if not isinstance(oof_rows, list) or any(
            not isinstance(row, Mapping)
            or set(row) != _FROZEN_AUXILIARY_ROW_FIELDS
            for row in oof_rows
        ):
            raise ValueError("auxiliary OOF row fields are not canonical")
    _verify_gate_shape(artifact.get("gate"))


def _verify_gate_shape(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != _GATE_FIELDS:
        raise ValueError("auxiliary gate fields are not canonical")
    if not isinstance(value.get("promoted"), bool):
        raise ValueError("auxiliary gate promotion flag is invalid")
    _verify_metric_shape(value.get("pooled"))
    per_game = value.get("per_game")
    if not isinstance(per_game, Mapping) or not per_game:
        raise ValueError("auxiliary gate per-game metrics are invalid")
    for metrics in per_game.values():
        _verify_metric_shape(metrics)


def _verify_metric_shape(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != _METRIC_FIELDS:
        raise ValueError("auxiliary metric fields are not canonical")
    for field in ("tp", "fp", "fn", "tn"):
        count = value.get(field)
        if type(count) is not int or count < 0:
            raise ValueError("auxiliary metric counts are invalid")
    observed = {
        field: _strict_probability(
            value.get(field),
            field="auxiliary metric probability",
        )
        for field in ("precision", "recall", "f1")
    }
    tp, fp, fn, _tn = (value[field] for field in ("tp", "fp", "fn", "tn"))
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * precision * recall, precision + recall)
    expected = {"precision": precision, "recall": recall, "f1": f1}
    if any(
        not math.isclose(
            observed[field],
            expected[field],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for field in expected
    ):
        raise ValueError("auxiliary metric arithmetic is inconsistent")


def _metric_row_count(value: Mapping[str, Any]) -> int:
    return sum(value[field] for field in ("tp", "fp", "fn", "tn"))


def _verify_vlm_observable_fields(row: Mapping[str, Any]) -> None:
    observables = row.get("observables")
    if observables is None:
        return
    if not isinstance(observables, Mapping):
        raise ValueError("VLM prediction observable fields are invalid")
    unexpected = set(observables) - set(OBSERVABLE_FIELDS)
    if unexpected:
        raise ValueError("VLM prediction observable fields are not canonical")
    if any(
        value is not None and not isinstance(value, bool)
        for value in observables.values()
    ):
        raise ValueError("VLM prediction observable fields are invalid")


def _verify_fusion_screen_contract(
    *,
    plan: Mapping[str, Any],
    auxiliary_artifact: Mapping[str, Any],
) -> None:
    raw_contract = plan.get("fusion_screen_contract")
    if not isinstance(raw_contract, Mapping):
        raise ValueError("frozen plan requires a fusion screen contract")
    contract = dict(raw_contract)
    required_contract_fields = {
        "scene_embedding_artifact_sha256",
        "video_embedding_artifacts",
        "predeclared_variants",
        "pca_components",
        "regularization_c",
        "threshold_selection",
        "promotion_requirements",
    }
    if set(contract) != required_contract_fields:
        raise ValueError("frozen plan fusion screen contract fields are invalid")
    scene_source_sha = contract["scene_embedding_artifact_sha256"]
    if not _is_sha256(scene_source_sha):
        raise ValueError("fusion screen contract scene artifact SHA-256 is invalid")
    source_videos = contract["video_embedding_artifacts"]
    if not isinstance(source_videos, list):
        raise ValueError("fusion screen contract video artifacts are invalid")
    normalized_source_videos: list[tuple[str, str]] = []
    for row in source_videos:
        if not isinstance(row, Mapping):
            raise ValueError("fusion screen contract video artifacts are invalid")
        if set(row) != {"backbone", "artifact_sha256"}:
            raise ValueError("fusion screen contract video artifact fields are invalid")
        backbone = row.get("backbone")
        artifact_sha = row.get("artifact_sha256")
        if (
            not isinstance(backbone, str)
            or not backbone
            or not _is_sha256(artifact_sha)
        ):
            raise ValueError("fusion screen contract video artifacts are invalid")
        normalized_source_videos.append((backbone, artifact_sha))
    if (
        normalized_source_videos != sorted(normalized_source_videos)
        or len({row[0] for row in normalized_source_videos})
        != len(normalized_source_videos)
        or len(set(normalized_source_videos)) != len(normalized_source_videos)
    ):
        raise ValueError("fusion screen contract video artifacts are not canonical")

    expected_variant_names = ["scene_phase_all"]
    for count in range(1, len(normalized_source_videos) + 1):
        for combination in itertools.combinations(normalized_source_videos, count):
            expected_variant_names.append(
                "scene_phase_all+" + "+".join(row[0] for row in combination)
            )
    contract_variants = _normalize_predeclared_variants(
        contract["predeclared_variants"]
    )
    if [row["name"] for row in contract_variants] != expected_variant_names:
        raise ValueError("fusion screen contract variants are invalid")
    pca_components = contract["pca_components"]
    if (
        isinstance(pca_components, bool)
        or not isinstance(pca_components, int)
        or pca_components < 1
    ):
        raise ValueError("fusion screen contract PCA components are invalid")
    raw_regularization_c = contract["regularization_c"]
    if isinstance(raw_regularization_c, bool) or not isinstance(
        raw_regularization_c, (int, float)
    ):
        raise ValueError("fusion screen contract regularization C is invalid")
    regularization_c = float(raw_regularization_c)
    if not math.isfinite(regularization_c) or regularization_c <= 0.0:
        raise ValueError("fusion screen contract regularization C is invalid")
    if contract["threshold_selection"] != _NESTED_THRESHOLD_SELECTION:
        raise ValueError("fusion screen contract threshold selection is invalid")
    if contract["promotion_requirements"] != _NESTED_PROMOTION_REQUIREMENTS:
        raise ValueError("fusion screen contract promotion requirements are invalid")

    raw_auxiliary_contract = auxiliary_artifact.get("fusion_screen_contract")
    if not isinstance(raw_auxiliary_contract, Mapping):
        raise ValueError("auxiliary artifact has no fusion screen contract")
    if dict(raw_auxiliary_contract) != contract:
        raise ValueError("auxiliary fusion screen contract does not match the plan")
    if (
        auxiliary_artifact.get("fusion_screen_contract_sha256")
        != _canonical_sha256(contract)
    ):
        raise ValueError("auxiliary fusion screen contract digest is invalid")
    if auxiliary_artifact.get("threshold_selection") != contract[
        "threshold_selection"
    ]:
        raise ValueError("auxiliary threshold selection does not match the contract")
    if (
        auxiliary_artifact.get("subset_selection_protocol")
        != _PLAN_BOUND_SUBSET_SELECTION_PROTOCOL
    ):
        raise ValueError("auxiliary subset selection protocol is invalid")
    random_seed = auxiliary_artifact.get("random_seed")
    if type(random_seed) is not int or random_seed != 0:
        raise ValueError("auxiliary random seed is invalid")
    auxiliary_pca_components = auxiliary_artifact.get("pca_components")
    if (
        type(auxiliary_pca_components) is not int
        or auxiliary_pca_components != pca_components
    ):
        raise ValueError("auxiliary fusion screen contract PCA value drifted")
    raw_auxiliary_regularization_c = auxiliary_artifact.get(
        "regularization_c", math.nan
    )
    if isinstance(raw_auxiliary_regularization_c, bool) or not isinstance(
        raw_auxiliary_regularization_c, (int, float)
    ):
        raise ValueError(
            "auxiliary fusion screen contract regularization value drifted"
        )
    auxiliary_regularization_c = float(raw_auxiliary_regularization_c)
    if auxiliary_regularization_c != regularization_c:
        raise ValueError("auxiliary fusion screen contract regularization value drifted")
    if auxiliary_artifact.get("promotion_requirements") != contract[
        "promotion_requirements"
    ]:
        raise ValueError("auxiliary fusion screen contract promotion policy drifted")
    if auxiliary_artifact.get("scene_embedding_artifact_sha256") != scene_source_sha:
        raise ValueError("auxiliary fusion screen contract scene artifact drifted")
    if auxiliary_artifact.get("video_embedding_artifact_sha256s") != [
        row[1] for row in normalized_source_videos
    ]:
        raise ValueError("auxiliary fusion screen contract video artifacts drifted")
    scene_subset_sha = auxiliary_artifact.get(
        "scene_subset_embedding_artifact_sha256"
    )
    video_subset_shas = auxiliary_artifact.get(
        "video_subset_embedding_artifact_sha256s"
    )
    if not _is_sha256(scene_subset_sha):
        raise ValueError("auxiliary scene subset artifact SHA-256 is invalid")
    if (
        not isinstance(video_subset_shas, list)
        or len(video_subset_shas) != len(normalized_source_videos)
        or any(not _is_sha256(value) for value in video_subset_shas)
        or len(set(video_subset_shas)) != len(video_subset_shas)
    ):
        raise ValueError("auxiliary video subset artifact SHA-256s are invalid")


def _normalize_predeclared_variants(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("auxiliary predeclared variants are missing")
    variants: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {
            "name",
            "input_dimension",
        }:
            raise ValueError("auxiliary predeclared variants are invalid")
        name = row.get("name")
        dimension = row.get("input_dimension")
        if (
            not isinstance(name, str)
            or not name
            or isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 1
        ):
            raise ValueError("auxiliary predeclared variants are invalid")
        variants.append({"name": name, "input_dimension": dimension})
    names = [row["name"] for row in variants]
    if len(set(names)) != len(names):
        raise ValueError("auxiliary predeclared variants contain duplicates")
    return variants


def _verify_variant_selection(
    *,
    fold: Mapping[str, Any],
    predeclared_variants: Sequence[Mapping[str, Any]],
    development_groups: Sequence[str],
    plan_keys_by_group: Mapping[str, set[tuple[str, str, str]]],
    expected_positive_counts: Mapping[str, int],
    expected_negative_counts: Mapping[str, int],
) -> dict[str, Any]:
    variants = fold.get("variant_selection")
    if not isinstance(variants, list) or len(variants) != len(
        predeclared_variants
    ):
        raise ValueError("auxiliary fold variant selection is incomplete")
    normalized: list[dict[str, Any]] = []
    for expected, raw_variant in zip(
        predeclared_variants, variants, strict=True
    ):
        if not isinstance(raw_variant, Mapping):
            raise ValueError("auxiliary fold variant selection is invalid")
        raw_dimension = raw_variant.get("input_dimension")
        if (
            raw_variant.get("name") != expected["name"]
            or type(raw_dimension) is not int
            or raw_dimension != expected["input_dimension"]
        ):
            raise ValueError("auxiliary fold variant differs from its declaration")
        try:
            threshold = _strict_probability(
                raw_variant["threshold"],
                field="auxiliary variant threshold",
            )
        except KeyError as exc:
            raise ValueError("auxiliary variant threshold is invalid") from exc
        inner_folds = raw_variant.get("inner_folds")
        if not isinstance(inner_folds, list) or len(inner_folds) != len(
            development_groups
        ):
            raise ValueError("auxiliary variant inner folds are incomplete")
        for expected_inner_held, inner_fold in zip(
            development_groups, inner_folds, strict=True
        ):
            if not isinstance(inner_fold, Mapping):
                raise ValueError("auxiliary variant inner folds are invalid")
            expected_fit = [
                group
                for group in development_groups
                if group != expected_inner_held
            ]
            row_count = inner_fold.get("row_count")
            if (
                inner_fold.get("held_game_sha256") != expected_inner_held
                or inner_fold.get("fit_game_sha256s") != expected_fit
                or type(row_count) is not int
                or row_count != len(plan_keys_by_group[expected_inner_held])
            ):
                raise ValueError("auxiliary variant inner folds are inconsistent")
        gate = raw_variant.get("gate")
        score = _nested_gate_score(
            gate,
            expected_groups=development_groups,
            expected_counts={
                group: len(plan_keys_by_group[group])
                for group in development_groups
            },
            expected_positive_counts={
                group: expected_positive_counts[group]
                for group in development_groups
            },
            expected_negative_counts={
                group: expected_negative_counts[group]
                for group in development_groups
            },
        )
        normalized.append(
            {
                "name": expected["name"],
                "threshold": threshold,
                "gate": gate,
                "score": score,
            }
        )
    selected_index = max(
        range(len(normalized)),
        key=lambda index: normalized[index]["score"],
    )
    selected = normalized[selected_index]
    if fold.get("selected_variant") != selected["name"]:
        raise ValueError("auxiliary fold selected variant is inconsistent")
    if fold.get("inner_selection_gate") != selected["gate"]:
        raise ValueError("auxiliary fold selected variant gate is inconsistent")
    return selected


def _nested_gate_score(
    gate: Any,
    *,
    expected_groups: Sequence[str],
    expected_counts: Mapping[str, int],
    expected_positive_counts: Mapping[str, int] | None = None,
    expected_negative_counts: Mapping[str, int] | None = None,
) -> tuple[float, ...]:
    if not isinstance(gate, Mapping) or not isinstance(
        gate.get("promoted"), bool
    ):
        raise ValueError("auxiliary variant gate is invalid")
    pooled = gate.get("pooled")
    per_game = gate.get("per_game")
    if not isinstance(pooled, Mapping) or not isinstance(per_game, Mapping):
        raise ValueError("auxiliary variant gate is invalid")
    if list(per_game) != list(expected_groups):
        raise ValueError("auxiliary variant gate game groups are inconsistent")
    if list(expected_counts) != list(expected_groups):
        raise ValueError("auxiliary variant gate row counts are inconsistent")
    if any(
        _metric_row_count(per_game[group]) != expected_counts[group]
        for group in expected_groups
    ):
        raise ValueError("auxiliary variant gate per-game row counts are inconsistent")
    if (expected_positive_counts is None) != (expected_negative_counts is None):
        raise ValueError("auxiliary variant gate class margins are incomplete")
    if expected_positive_counts is not None and expected_negative_counts is not None:
        if (
            list(expected_positive_counts) != list(expected_groups)
            or list(expected_negative_counts) != list(expected_groups)
            or any(
                per_game[group]["tp"] + per_game[group]["fn"]
                != expected_positive_counts[group]
                or per_game[group]["fp"] + per_game[group]["tn"]
                != expected_negative_counts[group]
                for group in expected_groups
            )
        ):
            raise ValueError("auxiliary variant gate class counts are inconsistent")
    for count_field in ("tp", "fp", "fn", "tn"):
        if pooled[count_field] != sum(
            per_game[group][count_field] for group in expected_groups
        ):
            raise ValueError("auxiliary variant gate pooled counts are inconsistent")

    def metric(row: Mapping[str, Any], field: str) -> float:
        try:
            raw_value = row[field]
        except KeyError as exc:
            raise ValueError("auxiliary variant gate metrics are invalid") from exc
        return _strict_probability(
            raw_value,
            field="auxiliary variant gate metrics",
        )

    pooled_precision = metric(pooled, "precision")
    pooled_recall = metric(pooled, "recall")
    pooled_f1 = metric(pooled, "f1")
    if any(not isinstance(per_game[group], Mapping) for group in expected_groups):
        raise ValueError("auxiliary variant gate metrics are invalid")
    precisions = [
        metric(per_game[group], "precision") for group in expected_groups
    ]
    recalls = [metric(per_game[group], "recall") for group in expected_groups]
    expected_promoted = bool(
        pooled_precision
        >= _NESTED_PROMOTION_REQUIREMENTS["minimum_pooled_precision"]
        and pooled_recall
        >= _NESTED_PROMOTION_REQUIREMENTS["minimum_pooled_recall"]
        and all(
            value
            >= _NESTED_PROMOTION_REQUIREMENTS["minimum_per_game_precision"]
            for value in precisions
        )
        and all(
            value
            >= _NESTED_PROMOTION_REQUIREMENTS["minimum_per_game_recall"]
            for value in recalls
        )
    )
    if gate["promoted"] is not expected_promoted:
        raise ValueError("auxiliary variant gate promotion flag is inconsistent")
    recall_eligible = all(
        value >= _NESTED_PROMOTION_REQUIREMENTS["minimum_per_game_recall"]
        for value in recalls
    )
    return (
        float(gate["promoted"]),
        float(recall_eligible),
        min(precisions),
        min(recalls),
        pooled_f1,
    )


def _normalize_frozen_auxiliary_row(row: Mapping[str, Any]) -> dict[str, Any]:
    missing_fields = _FROZEN_AUXILIARY_ROW_FIELDS - set(row)
    if missing_fields:
        raise ValueError(
            "auxiliary OOF rows require a complete three-part key including "
            "candidate_bundle_sha256 and probability"
        )
    unexpected_fields = set(row) - _FROZEN_AUXILIARY_ROW_FIELDS
    if unexpected_fields:
        raise ValueError(
            "auxiliary OOF rows contain label-bearing or unexpected fields"
        )
    values = tuple(row.get(field) for field in _KEY_FIELDS)
    if any(not isinstance(value, str) for value in values):
        raise ValueError("auxiliary OOF three-part key fields must be strings")
    key = values
    if not all(key):
        raise ValueError(
            "auxiliary OOF rows require a complete three-part key including "
            "candidate_bundle_sha256"
        )
    if not _is_sha256(key[0]) or not _is_sha256(key[1]):
        raise ValueError("auxiliary OOF key SHA-256 bindings are invalid")
    probability = _strict_probability(
        row.get("probability"),
        field="auxiliary OOF probability",
    )
    return {
        "source_video_sha256": key[0],
        "candidate_bundle_sha256": key[1],
        "event_id": key[2],
        "probability": probability,
    }


def _assert_unique_frozen_auxiliary_rows(
    rows: Sequence[Mapping[str, Any]],
) -> None:
    keys = [tuple(str(row[field]) for field in _KEY_FIELDS) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("auxiliary OOF rows contain duplicate three-part keys")


def _normalize_game_groups(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"auxiliary screen {field} game groups are missing")
    if any(not _is_sha256(item) for item in value):
        raise ValueError(f"auxiliary screen {field} game groups are invalid")
    groups = list(value)
    if len(set(groups)) != len(groups):
        raise ValueError(f"auxiliary screen {field} game groups are invalid")
    return groups


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _normalize_contract_key(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _strict_probability(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite and within [0, 1]")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{field} must be finite and within [0, 1]")
    return probability


def _normalize_auxiliary_row(row: Mapping[str, Any]) -> dict[str, Any]:
    source = str(row.get("source_video_sha256") or "")
    candidate_bundle = str(row.get("candidate_bundle_sha256") or "")
    event = str(row.get("event_id") or "")
    try:
        probability = float(row.get("probability"))
    except (TypeError, ValueError):
        probability = math.nan
    if (
        not source
        or not candidate_bundle
        or not event
        or not math.isfinite(probability)
        or not 0.0 <= probability <= 1.0
    ):
        raise ValueError("auxiliary OOF rows have invalid key or probability")
    label = row.get("event_present")
    if not isinstance(label, bool):
        raise ValueError("auxiliary OOF rows require boolean training labels")
    return {
        "source_video_sha256": source,
        "candidate_bundle_sha256": candidate_bundle,
        "event_id": event,
        "probability": probability,
        "event_present": label,
    }


def _assert_unique_auxiliary_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    keys = [
        tuple(str(row[field]) for field in _AUXILIARY_KEY_FIELDS)
        for row in rows
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("auxiliary OOF rows contain duplicates")


def _threshold_metrics(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for row in rows:
        predicted = float(row["probability"]) >= threshold
        label = bool(row["event_present"])
        if predicted and label:
            tp += 1
        elif predicted:
            fp += 1
        elif label:
            fn += 1
        else:
            tn += 1
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * precision * recall, precision + recall)
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _metrics(counts: Mapping[str, int]) -> dict[str, Any]:
    tp = int(counts.get("true_positive", counts.get("tp", 0)))
    fp = int(counts.get("false_positive", counts.get("fp", 0)))
    fn = int(counts.get("false_negative", counts.get("fn", 0)))
    tn = int(counts.get("true_negative", counts.get("tn", 0)))
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
    }


def _key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    values = tuple(row.get(field) for field in _KEY_FIELDS)
    if any(not isinstance(value, str) for value in values):
        raise ValueError("VLM prediction key fields must be strings")
    key = values
    if not all(key):
        raise ValueError("VLM prediction key is incomplete")
    return key  # type: ignore[return-value]


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
