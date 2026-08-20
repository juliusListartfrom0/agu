"""Training-only contracts for an independent basketball VLM shot reviewer."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from typing import Any, Mapping

INDEPENDENT_SHOT_VLM_PLAN_SCHEMA = "agu.independent-shot-vlm-plan.v1"
INDEPENDENT_SHOT_VLM_PREDICTIONS_SCHEMA = "agu.independent-shot-vlm-predictions.v1"
INDEPENDENT_SHOT_VLM_EVALUATION_SCHEMA = "agu.independent-shot-vlm-evaluation.v1"
SHOT_STATES = frozenset({"live_field_goal", "not_field_goal", "unknown"})
OBSERVABLE_FIELDS = (
    "continuous_live_play",
    "controlled_ball_before_release",
    "ball_separates_from_hands",
    "ball_moves_toward_rim",
    "replay_or_highlight",
    "free_throw",
)
LEGACY_V1_MINIMUM_PRECISION = 0.95
LEGACY_V1_MINIMUM_RECALL = 0.85
LEGACY_V1_PROVENANCE_VERIFICATION = "open_world_structural_only"
_FORBIDDEN_LABEL_OR_REVIEW_FIELDS = frozenset(
    {
        "actual",
        "annotation",
        "annotations",
        "class",
        "event_present",
        "gold_label",
        "ground_truth",
        "held_label",
        "human_review_label",
        "human_reviewed_label",
        "is_made",
        "is_positive",
        "label",
        "labels",
        "made",
        "missed",
        "outcome",
        "review_note",
        "review_notes",
        "reviewer_notes",
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
_FALSE_PROVENANCE_DECLARATIONS = frozenset(
    {
        "codex_involved",
        "codex_runtime_answer_used",
        "labels_or_review_notes_exposed_to_model",
        "learned_on_target_labels",
        "outer_held_labels_used_for_selection",
        "raw_target_label_fields_read_by_fuser",
        "runtime_consumable",
        "target_labels_used",
    }
)
_TRUE_PROVENANCE_DECLARATIONS = frozenset(
    {
        "codex_runtime_answer_used_declared_false",
        "label_selection_not_used",
    }
)
_BOOLEAN_PROVENANCE_DISCLOSURES = frozenset(
    {
        "auxiliary_fit_used_training_labels",
        "independent_from_codex",
        "input_model_independent_from_codex_declared",
    }
)
_SAFE_LABEL_OR_REVIEW_DECLARATIONS = frozenset(
    {
        "auxiliary_fit_used_training_labels",
        "label_selection_not_used",
        "labels_or_review_notes_exposed_to_model",
        "learned_on_target_labels",
        "outer_held_labels_used_for_selection",
        "raw_target_label_fields_read_by_fuser",
        "target_labels_used",
    }
)
_CANONICAL_RESERVED_FIELDS = frozenset(
    {
        "codex_runtime_answer_used",
        "purpose",
        "runtime_consumable",
        "schema_version",
    }
)
_NESTED_FUSION_PLAN_REQUIRED_FIELDS = frozenset(
    {
        "codex_runtime_answer_used",
        "examples",
        "fusion_screen_contract",
        "input_contract",
        "plan_sha256",
        "purpose",
        "runtime_consumable",
        "schema_version",
        "selection",
        "training_manifest_sha256",
    }
)
_NESTED_FUSION_PLAN_OPTIONAL_FIELDS = frozenset({"training_annotation_sha256"})
_NESTED_FUSION_PLAN_EXAMPLE_FIELDS = frozenset(
    {
        "candidate_bundle_sha256",
        "end_frame",
        "event_id",
        "source_fps",
        "source_video_filename",
        "source_video_sha256",
        "start_frame",
    }
)
_NESTED_FUSION_PLAN_SELECTION_FIELDS = frozenset(
    {
        "example_count",
        "method",
        "negative_per_video",
        "positive_per_video",
        "video_count",
    }
)
_NESTED_FUSION_PLAN_INPUT_FIELDS = frozenset(
    {
        "chronological_even_sampling",
        "image_width",
        "labels_or_review_notes_exposed_to_model",
        "max_frames",
        "raw_frames_only",
    }
)


def canonical_sha256(payload: Mapping[str, Any], *, hash_field: str) -> str:
    """Hash a JSON artifact while excluding its own digest field."""
    normalized = dict(payload)
    normalized.pop(hash_field, None)
    raw = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def seal_independent_shot_vlm_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Seal an open-world legacy, development-only VLM plan.

    This generic v1 boundary applies best-effort provenance hygiene but cannot
    prove label-free or Codex-independent semantics for arbitrary extension
    fields. Closed consumers must apply their own exact schemas.
    """
    _reject_conflicting_seal_metadata(
        payload,
        schema_version=INDEPENDENT_SHOT_VLM_PLAN_SCHEMA,
        context="independent-shot VLM plan",
    )
    _reject_explicit_unsafe_provenance(payload, context="independent-shot VLM plan")
    sealed = dict(payload)
    sealed["schema_version"] = INDEPENDENT_SHOT_VLM_PLAN_SCHEMA
    sealed["purpose"] = "offline_development_model_screening"
    sealed["runtime_consumable"] = False
    sealed["codex_runtime_answer_used"] = False
    sealed["plan_sha256"] = canonical_sha256(sealed, hash_field="plan_sha256")
    return verify_independent_shot_vlm_plan(sealed)


def verify_independent_shot_vlm_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Verify legacy structural integrity, uniqueness and non-runtime status.

    The v1 payload is intentionally open-world. Passing this verifier does not
    establish label-free or Codex-independent provenance.
    """
    plan = dict(payload)
    if plan.get("schema_version") != INDEPENDENT_SHOT_VLM_PLAN_SCHEMA:
        raise ValueError("unsupported independent-shot VLM plan schema")
    if plan.get("purpose") != "offline_development_model_screening":
        raise ValueError("independent-shot VLM plan must be development-only")
    if plan.get("runtime_consumable") is not False:
        raise ValueError("independent-shot VLM plan must not be runtime-consumable")
    if plan.get("codex_runtime_answer_used") is not False:
        raise ValueError("Codex must not provide an independent VLM runtime answer")
    _reject_explicit_unsafe_provenance(
        plan,
        context="independent-shot VLM plan",
    )
    if canonical_sha256(plan, hash_field="plan_sha256") != plan.get("plan_sha256"):
        raise ValueError("independent-shot VLM plan hash mismatch")
    rows = plan.get("examples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("independent-shot VLM plan requires examples")
    _reject_label_or_review_fields(plan, context="independent-shot VLM plan")
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("independent-shot VLM plan rows must be objects")
        key = _example_key(row)
        if key in seen:
            raise ValueError("independent-shot VLM plan contains duplicate examples")
        seen.add(key)
        start = int(row.get("start_frame", -1))
        end = int(row.get("end_frame", -1))
        if start < 0 or end <= start:
            raise ValueError("independent-shot VLM plan has invalid frame bounds")
        if float(row.get("source_fps", 0.0)) <= 0:
            raise ValueError("independent-shot VLM plan has invalid source FPS")
    return plan


def verify_nested_fusion_shot_vlm_plan(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the closed, label-hidden plan contract used by nested fusion.

    The generic independent-shot plan remains extensible for other research
    flows. Nested fusion is deliberately narrower so an unknown field cannot
    become a covert held-label or review-note channel.
    """

    plan = verify_independent_shot_vlm_plan(payload)
    fields = set(plan)
    if not _NESTED_FUSION_PLAN_REQUIRED_FIELDS.issubset(fields) or not fields.issubset(
        _NESTED_FUSION_PLAN_REQUIRED_FIELDS | _NESTED_FUSION_PLAN_OPTIONAL_FIELDS
    ):
        raise ValueError("nested fusion plan top-level fields are not canonical")
    if not _is_lower_sha256(plan.get("plan_sha256")):
        raise ValueError("nested fusion plan SHA-256 binding is invalid")
    if not _is_lower_sha256(plan.get("training_manifest_sha256")):
        raise ValueError("nested fusion plan training manifest is invalid")
    annotation_shas = plan.get("training_annotation_sha256")
    if annotation_shas is not None and (
        not isinstance(annotation_shas, list)
        or not annotation_shas
        or any(not _is_lower_sha256(value) for value in annotation_shas)
        or annotation_shas != sorted(set(annotation_shas))
    ):
        raise ValueError("nested fusion plan annotation bindings are invalid")

    examples = plan.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("nested fusion plan examples are invalid")
    for row in examples:
        if not isinstance(row, Mapping) or set(row) != _NESTED_FUSION_PLAN_EXAMPLE_FIELDS:
            raise ValueError("nested fusion plan example fields are not canonical")
        if not _is_lower_sha256(row.get("source_video_sha256")):
            raise ValueError("nested fusion plan source video must be a lowercase SHA-256 string")
        if not _is_lower_sha256(row.get("candidate_bundle_sha256")):
            raise ValueError("nested fusion plan candidate bundle must be a lowercase SHA-256 string")
        if not isinstance(row.get("source_video_filename"), str) or not row["source_video_filename"]:
            raise ValueError("nested fusion plan source filename is invalid")
        if not isinstance(row.get("event_id"), str) or not row["event_id"]:
            raise ValueError("nested fusion plan three-part key fields must be strings")
        start = row.get("start_frame")
        end = row.get("end_frame")
        fps = row.get("source_fps")
        if (
            type(start) is not int
            or type(end) is not int
            or start < 0
            or end <= start
            or isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not math.isfinite(float(fps))
            or float(fps) <= 0.0
        ):
            raise ValueError("nested fusion plan example geometry is invalid")

    selection = plan.get("selection")
    if (
        not isinstance(selection, Mapping)
        or set(selection) != _NESTED_FUSION_PLAN_SELECTION_FIELDS
        or selection.get("method") != "sha256_rank_within_video_and_boolean_class"
        or type(selection.get("positive_per_video")) is not int
        or selection["positive_per_video"] < 0
        or type(selection.get("negative_per_video")) is not int
        or selection["negative_per_video"] < 0
        or (selection["positive_per_video"] + selection["negative_per_video"] < 1)
        or type(selection.get("video_count")) is not int
        or selection["video_count"] != len({row["source_video_sha256"] for row in examples})
        or type(selection.get("example_count")) is not int
        or selection["example_count"] != len(examples)
    ):
        raise ValueError("nested fusion plan selection fields are not canonical")

    input_contract = plan.get("input_contract")
    if (
        not isinstance(input_contract, Mapping)
        or set(input_contract) != _NESTED_FUSION_PLAN_INPUT_FIELDS
        or input_contract.get("raw_frames_only") is not True
        or input_contract.get("labels_or_review_notes_exposed_to_model") is not False
        or input_contract.get("chronological_even_sampling") is not True
        or type(input_contract.get("max_frames")) is not int
        or input_contract["max_frames"] < 2
        or type(input_contract.get("image_width")) is not int
        or input_contract["image_width"] < 1
        or not isinstance(plan.get("fusion_screen_contract"), Mapping)
    ):
        raise ValueError("nested fusion plan input contract fields are not canonical")
    per_video_count = selection["positive_per_video"] + selection["negative_per_video"]
    source_counts: dict[str, int] = defaultdict(int)
    for row in examples:
        source_counts[row["source_video_sha256"]] += 1
    if (
        any(count != per_video_count for count in source_counts.values())
        or len(examples) != selection["video_count"] * per_video_count
    ):
        raise ValueError("nested fusion plan rows do not match the declared per-video quotas")
    return plan


def _is_lower_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def derive_native_video_shot_vlm_plan(
    source_plan: Mapping[str, Any],
    *,
    sample_fps: float,
    max_pixels: int,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Derive a native-video contract with known-field provenance hygiene.

    The source is the open-world legacy v1 schema, so this transformation does
    not independently prove label-free provenance for arbitrary extensions.
    """
    source = verify_independent_shot_vlm_plan(source_plan)
    if sample_fps <= 0:
        raise ValueError("native-video sample FPS must be positive")
    if max_pixels <= 0:
        raise ValueError("native-video max pixels must be positive")
    if max_examples is not None and max_examples <= 0:
        raise ValueError("native-video resource probe size must be positive")
    if max_examples is not None and max_examples > len(source["examples"]):
        raise ValueError("native-video resource probe exceeds source examples")
    examples = [dict(row) for row in source["examples"]]
    selection = dict(source.get("selection") or {})
    if max_examples is not None:
        examples = examples[:max_examples]
        selection.update(
            {
                "resource_probe_only": True,
                "source_example_count": len(source["examples"]),
                "example_count": len(examples),
            }
        )
    return seal_independent_shot_vlm_plan(
        {
            "source_plan_sha256": source["plan_sha256"],
            "selection": selection,
            "input_contract": {
                "raw_video_clips_only": True,
                "labels_or_review_notes_exposed_to_model": False,
                "native_temporal_position_encoding": True,
                "sample_fps": float(sample_fps),
                "max_pixels": int(max_pixels),
            },
            "examples": examples,
        }
    )


def derive_frame_sampled_shot_vlm_plan(
    source_plan: Mapping[str, Any],
    *,
    max_frames: int,
    image_width: int,
) -> dict[str, Any]:
    """Derive a frame-sampling contract with known-field provenance hygiene.

    The source is the open-world legacy v1 schema, so this transformation does
    not independently prove label-free provenance for arbitrary extensions.
    """
    source = verify_independent_shot_vlm_plan(source_plan)
    if max_frames < 2:
        raise ValueError("frame-sampled max frames must be at least two")
    if image_width <= 0:
        raise ValueError("frame-sampled image width must be positive")
    return seal_independent_shot_vlm_plan(
        {
            "source_plan_sha256": source["plan_sha256"],
            "selection": dict(source.get("selection") or {}),
            "input_contract": {
                "raw_frames_only": True,
                "labels_or_review_notes_exposed_to_model": False,
                "chronological_even_sampling": True,
                "max_frames": int(max_frames),
                "image_width": int(image_width),
            },
            "examples": [dict(row) for row in source["examples"]],
        }
    )


def verify_independent_shot_vlm_annotation_plan(
    evaluation_plan: Mapping[str, Any],
    annotation_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the label-bearing provenance plan for a derived evaluation plan."""
    evaluation = verify_independent_shot_vlm_plan(evaluation_plan)
    annotation = verify_independent_shot_vlm_plan(annotation_plan)
    expected_source = evaluation.get("source_plan_sha256")
    if expected_source is None:
        if evaluation["plan_sha256"] != annotation["plan_sha256"]:
            raise ValueError("annotation plan does not match the evaluation plan")
    elif expected_source != annotation["plan_sha256"]:
        raise ValueError("annotation plan does not match the derived source plan")
    annotation_hashes = annotation.get("training_annotation_sha256")
    if (
        not isinstance(annotation_hashes, list)
        or not annotation_hashes
        or len(set(annotation_hashes)) != len(annotation_hashes)
        or any(not isinstance(value, str) or len(value) != 64 for value in annotation_hashes)
    ):
        raise ValueError("annotation plan has invalid training annotation hashes")
    evaluation_keys = {_example_key(row) for row in evaluation["examples"]}
    annotation_keys = {_example_key(row) for row in annotation["examples"]}
    if not evaluation_keys.issubset(annotation_keys):
        raise ValueError("annotation plan does not cover the evaluation examples")
    return annotation


def parse_independent_shot_vlm_decision(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one strict independent-model decision."""
    state = str(payload.get("field_goal_state") or "").strip().lower()
    if state not in SHOT_STATES:
        state = "unknown"
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    observables = {field: _optional_bool(payload.get(field)) for field in OBSERVABLE_FIELDS}
    return {
        "field_goal_state": state,
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": str(payload.get("reason") or "")[:500],
        "observables": observables,
    }


def seal_independent_shot_vlm_predictions(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Seal open-world legacy decisions without accepting them as runtime truth."""
    _reject_conflicting_seal_metadata(
        payload,
        schema_version=INDEPENDENT_SHOT_VLM_PREDICTIONS_SCHEMA,
        context="independent-shot VLM predictions",
    )
    _reject_explicit_unsafe_provenance(
        payload,
        context="independent-shot VLM predictions",
    )
    sealed = dict(payload)
    sealed["schema_version"] = INDEPENDENT_SHOT_VLM_PREDICTIONS_SCHEMA
    sealed["purpose"] = "offline_development_model_screening"
    sealed["runtime_consumable"] = False
    sealed["codex_runtime_answer_used"] = False
    sealed["artifact_sha256"] = canonical_sha256(sealed, hash_field="artifact_sha256")
    return verify_independent_shot_vlm_predictions(sealed)


def verify_independent_shot_vlm_predictions(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify legacy prediction structure without proving semantic provenance."""
    predictions = dict(payload)
    if predictions.get("schema_version") != INDEPENDENT_SHOT_VLM_PREDICTIONS_SCHEMA:
        raise ValueError("unsupported independent-shot VLM predictions schema")
    if predictions.get("purpose") != "offline_development_model_screening":
        raise ValueError("independent-shot VLM predictions must be development-only")
    if predictions.get("runtime_consumable") is not False:
        raise ValueError("independent-shot VLM predictions must not be runtime-consumable")
    if predictions.get("codex_runtime_answer_used") is not False:
        raise ValueError("Codex must not provide an independent VLM runtime answer")
    _reject_explicit_unsafe_provenance(
        predictions,
        context="independent-shot VLM predictions",
    )
    if canonical_sha256(predictions, hash_field="artifact_sha256") != predictions.get("artifact_sha256"):
        raise ValueError("independent-shot VLM predictions hash mismatch")
    _reject_label_or_review_fields(
        predictions,
        context="independent-shot VLM predictions",
    )
    rows = predictions.get("predictions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("independent-shot VLM predictions require rows")
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = _example_key(row)
        if key in seen:
            raise ValueError("independent-shot VLM predictions contain duplicates")
        seen.add(key)
        normalized = parse_independent_shot_vlm_decision(row)
        if normalized["field_goal_state"] != row.get("field_goal_state"):
            raise ValueError("independent-shot VLM prediction is not normalized")
    return predictions


def evaluate_independent_shot_vlm(
    *,
    plan: Mapping[str, Any],
    predictions: Mapping[str, Any],
    truth: Mapping[tuple[str, str, str], bool],
    minimum_precision: float = LEGACY_V1_MINIMUM_PRECISION,
    minimum_recall: float = LEGACY_V1_MINIMUM_RECALL,
    expected_plan_sha256: str | None = None,
    expected_prediction_artifact_sha256: str | None = None,
    prediction_file_sha256: str | None = None,
    expected_prediction_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Evaluate legacy v1 predictions as a permanently non-promotable diagnostic."""
    minimum_precision = _strict_gate_threshold(
        minimum_precision,
        field="minimum_precision",
        required_minimum=LEGACY_V1_MINIMUM_PRECISION,
    )
    minimum_recall = _strict_gate_threshold(
        minimum_recall,
        field="minimum_recall",
        required_minimum=LEGACY_V1_MINIMUM_RECALL,
    )
    verified_plan = verify_independent_shot_vlm_plan(plan)
    verified_predictions = verify_independent_shot_vlm_predictions(predictions)
    plan_artifact_receipt_verified = False
    if expected_plan_sha256 is not None:
        if not _is_lower_sha256(expected_plan_sha256):
            raise ValueError("expected plan SHA-256 is invalid")
        if expected_plan_sha256 != verified_plan["plan_sha256"]:
            raise ValueError("expected plan SHA-256 does not match")
        plan_artifact_receipt_verified = True
    prediction_artifact_receipt_verified = False
    if expected_prediction_artifact_sha256 is not None:
        if not _is_lower_sha256(expected_prediction_artifact_sha256):
            raise ValueError("expected prediction artifact SHA-256 is invalid")
        if expected_prediction_artifact_sha256 != verified_predictions["artifact_sha256"]:
            raise ValueError("expected prediction artifact SHA-256 does not match")
        prediction_artifact_receipt_verified = True
    prediction_file_receipt_verified = False
    if prediction_file_sha256 is not None and not _is_lower_sha256(prediction_file_sha256):
        raise ValueError("prediction file SHA-256 is invalid")
    if expected_prediction_file_sha256 is not None:
        if not _is_lower_sha256(expected_prediction_file_sha256):
            raise ValueError("expected prediction file SHA-256 is invalid")
        if prediction_file_sha256 is None:
            raise ValueError("prediction file SHA-256 is required for receipt verification")
        if expected_prediction_file_sha256 != prediction_file_sha256:
            raise ValueError("expected prediction file SHA-256 does not match")
        prediction_file_receipt_verified = True
    prediction_model = verified_predictions.get("model")
    input_prediction_formal_promotion_eligible: bool | None = None
    if isinstance(prediction_model, Mapping) and "formal_promotion_eligible" in prediction_model:
        formal_marker = prediction_model["formal_promotion_eligible"]
        if not isinstance(formal_marker, bool):
            raise ValueError("prediction formal-promotion marker must be boolean")
        input_prediction_formal_promotion_eligible = formal_marker
        if formal_marker is True and prediction_model.get("independent_from_codex") is not True:
            raise ValueError("formal prediction provenance must be independent from Codex")
    if verified_predictions.get("plan_sha256") != verified_plan["plan_sha256"]:
        raise ValueError("prediction artifact does not match the frozen plan")
    expected = {_example_key(row) for row in verified_plan["examples"]}
    actual = {_example_key(row) for row in verified_predictions["predictions"]}
    if actual != expected:
        raise ValueError("prediction rows do not exactly cover the frozen plan")
    if set(truth) != expected:
        raise ValueError("truth rows do not exactly cover the frozen plan")
    if any(type(value) is not bool for value in truth.values()):
        raise ValueError("truth values must be exact booleans")

    counts = defaultdict(int)
    per_video_counts: dict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    unknown_event_ids: list[str] = []
    false_positive_event_ids: list[str] = []
    by_key = {_example_key(row): row for row in verified_predictions["predictions"]}
    for key in sorted(expected):
        state = str(by_key[key]["field_goal_state"])
        predicted = state == "live_field_goal"
        label = truth[key]
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
        per_video_counts[key[0]][bucket] += 1
        if state == "unknown":
            counts["unknown"] += 1
            per_video_counts[key[0]]["unknown"] += 1
            unknown_event_ids.append(key[2])
        if bucket == "false_positive":
            false_positive_event_ids.append(key[2])
    precision = _ratio(counts["true_positive"], counts["true_positive"] + counts["false_positive"])
    recall = _ratio(counts["true_positive"], counts["true_positive"] + counts["false_negative"])
    per_video = []
    for video_sha, video_counts in sorted(per_video_counts.items()):
        video_precision = _ratio(
            video_counts["true_positive"],
            video_counts["true_positive"] + video_counts["false_positive"],
        )
        video_recall = _ratio(
            video_counts["true_positive"],
            video_counts["true_positive"] + video_counts["false_negative"],
        )
        per_video.append(
            {
                "raw_video_sha256": video_sha,
                **{
                    name: video_counts[name]
                    for name in (
                        "true_positive",
                        "false_positive",
                        "false_negative",
                        "true_negative",
                        "unknown",
                    )
                },
                "precision": video_precision,
                "recall": video_recall,
            }
        )
    observed_metric_requirements_met = (
        precision >= minimum_precision
        and recall >= minimum_recall
        and all(row["precision"] >= minimum_precision and row["recall"] >= minimum_recall for row in per_video)
    )
    metrics = {
        "evaluated_count": len(expected),
        **{
            name: counts[name]
            for name in (
                "true_positive",
                "false_positive",
                "false_negative",
                "true_negative",
                "unknown",
            )
        },
        "precision": precision,
        "recall": recall,
        "minimum_precision": minimum_precision,
        "minimum_recall": minimum_recall,
        "observed_metric_requirements_met": observed_metric_requirements_met,
        "input_prediction_formal_promotion_eligible": (input_prediction_formal_promotion_eligible),
        "prediction_artifact_receipt_verified": (prediction_artifact_receipt_verified),
        "prediction_file_receipt_verified": prediction_file_receipt_verified,
        "plan_artifact_receipt_verified": plan_artifact_receipt_verified,
        "input_provenance_verification": LEGACY_V1_PROVENANCE_VERIFICATION,
        "input_label_free_verified": False,
        "input_codex_independence_verified": False,
        "legacy_provenance_only": True,
        "promotion_eligible": False,
        "false_positive_event_ids": false_positive_event_ids,
        "unknown_event_ids": unknown_event_ids,
        "per_video": per_video,
    }
    artifact = {
        "schema_version": INDEPENDENT_SHOT_VLM_EVALUATION_SCHEMA,
        "purpose": "offline_development_model_screening",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "input_provenance_verification": LEGACY_V1_PROVENANCE_VERIFICATION,
        "input_label_free_verified": False,
        "input_codex_independence_verified": False,
        "legacy_provenance_only": True,
        "plan_sha256": verified_plan["plan_sha256"],
        "expected_plan_sha256": expected_plan_sha256,
        "prediction_artifact_sha256": verified_predictions["artifact_sha256"],
        "expected_prediction_artifact_sha256": (expected_prediction_artifact_sha256),
        "prediction_file_sha256": prediction_file_sha256,
        "expected_prediction_file_sha256": expected_prediction_file_sha256,
        "metrics": metrics,
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact, hash_field="artifact_sha256")
    return artifact


def stable_example_rank(raw_sha256: str, candidate_sha256: str, event_id: str) -> str:
    """Return a deterministic selection rank independent of label ordering."""
    return hashlib.sha256(f"{raw_sha256}:{candidate_sha256}:{event_id}".encode()).hexdigest()


def _example_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    key = (
        str(row.get("source_video_sha256") or ""),
        str(row.get("candidate_bundle_sha256") or ""),
        str(row.get("event_id") or ""),
    )
    if not all(key):
        raise ValueError("independent-shot VLM example key is incomplete")
    return key


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _reject_explicit_unsafe_provenance(
    payload: Any,
    *,
    context: str,
) -> None:
    if isinstance(payload, Mapping):
        normalized_keys: set[str] = set()
        for key, nested_value in payload.items():
            if not isinstance(key, str):
                raise ValueError(f"{context} fields must have string names")
            normalized_key = _normalize_contract_key(key)
            if normalized_key in normalized_keys:
                raise ValueError(f"{context} contains normalized field aliases")
            normalized_keys.add(normalized_key)
            if normalized_key in _CANONICAL_RESERVED_FIELDS and key != normalized_key:
                raise ValueError(f"{context} contains a noncanonical reserved field")
            if normalized_key in _FALSE_PROVENANCE_DECLARATIONS and nested_value is not False:
                raise ValueError(f"{context} contains unsafe provenance")
            if normalized_key in _TRUE_PROVENANCE_DECLARATIONS and nested_value is not True:
                raise ValueError(f"{context} contains unsafe provenance")
            if normalized_key in _BOOLEAN_PROVENANCE_DISCLOSURES and type(nested_value) is not bool:
                raise ValueError(f"{context} contains noncanonical provenance")
            if (
                "codex" in normalized_key.split("_")
                and normalized_key
                not in (
                    _FALSE_PROVENANCE_DECLARATIONS | _TRUE_PROVENANCE_DECLARATIONS | _BOOLEAN_PROVENANCE_DISCLOSURES
                )
                and nested_value is not False
            ):
                raise ValueError(f"{context} contains unsafe Codex provenance")
            _reject_explicit_unsafe_provenance(nested_value, context=context)
    elif isinstance(payload, (list, tuple)):
        for nested_value in payload:
            _reject_explicit_unsafe_provenance(nested_value, context=context)


def _normalize_contract_key(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _reject_conflicting_seal_metadata(
    payload: Mapping[str, Any],
    *,
    schema_version: str,
    context: str,
) -> None:
    if "schema_version" in payload and payload["schema_version"] != schema_version:
        raise ValueError(f"{context} has a conflicting schema version")
    if "purpose" in payload and payload["purpose"] != "offline_development_model_screening":
        raise ValueError(f"{context} has a conflicting purpose")


def _reject_label_or_review_fields(value: Any, *, context: str) -> None:
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            normalized_key = _normalize_contract_key(key)
            tokens = set(normalized_key.split("_"))
            label_or_truth_field = bool(tokens & {"label", "labels", "truth"})
            review_content_field = bool(tokens & {"review", "reviewed", "reviewer"}) and bool(
                tokens & {"answer", "label", "labels", "note", "notes", "outcome", "truth", "verdict"}
            )
            if normalized_key not in _SAFE_LABEL_OR_REVIEW_DECLARATIONS and (
                normalized_key in _FORBIDDEN_LABEL_OR_REVIEW_FIELDS or label_or_truth_field or review_content_field
            ):
                raise ValueError(f"{context} contains a forbidden label field or review field")
            _reject_label_or_review_fields(nested_value, context=context)
    elif isinstance(value, (list, tuple)):
        for nested_value in value:
            _reject_label_or_review_fields(nested_value, context=context)


def _strict_gate_threshold(
    value: Any,
    *,
    field: str,
    required_minimum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite numeric threshold")
    parsed = float(value)
    if not math.isfinite(parsed) or not required_minimum <= parsed <= 1.0:
        raise ValueError(f"{field} must remain within [{required_minimum}, 1.0]")
    return parsed


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
