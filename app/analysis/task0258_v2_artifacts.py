"""TASK-0258 Amendment-001 v2 artifact schemas: candidate gate + mechanical failure.

Closed-field, fail-closed validators for the two pre/post publication artifact
schemas. They compose the receipt primitives (`task0258_module_a_v2`) and the
gate vocabulary (`task0258_v2_gate`). Deep provider-slot / subject-receipt
sub-validation is layered on in later phases; these enforce the exact top-level
field set and every pinned constant field.
"""

from __future__ import annotations

from collections.abc import Mapping

from app.analysis.task0258_module_a_v2 import (
    CANDIDATE_GATE_SCHEMA_V2,
    CANDIDATE_RECEIPT_BUNDLE_SCHEMA,
    MECHANICAL_FAILURE_SCHEMA_V2,
    MODULE_ID,
    POSTPUBLICATION_FAILURE_SCHEMA_V2,
    POSTPUBLICATION_VERIFICATION_SCHEMA_V2,
    is_sha256,
    verify_artifact_file_receipt,
    verify_authorization_receipts,
    verify_exact_field_set,
    verify_history_artifact_receipt,
    verify_internal_artifact_hash,
    verify_provider_slot,
    verify_provider_slots,
    verify_run_history_contract_receipt,
    verify_static_input_contract,
)
from app.analysis.task0258_v2_gate import (
    CANDIDATE_PREPUBLICATION_CHECKS,
    RESULT_ORDERED_CHECKS,
    verify_candidate_metric_outcome,
    verify_failed_phase,
    verify_failed_phase_stop_reason,
    verify_failure_check_sequence,
    verify_ordered_checks,
)

_COMMON_FALSE_FIELDS = frozenset(
    {
        "schema_version",
        "module_id",
        "purpose",
        "runtime_consumable",
        "training_consumable",
        "formal_evaluation_eligible",
        "promotion_eligible",
        "promoted",
    }
)

# candidate_gate.v2 exact top-level field set.
CANDIDATE_GATE_FIELDS = _COMMON_FALSE_FIELDS | frozenset(
    {
        "input_receipts",
        "producer_attempt_chain",
        "verification_attempt_receipt",
        "ordered_prepublication_check_results",
        "error_bound_result",
        "candidate_metric_outcome",
        "publication_state",
        "postpublication_verification_required",
        "conditional_downstream",
        "artifact_sha256",
    }
)

# mechanical_failure.v2 exact top-level field set.
MECHANICAL_FAILURE_FIELDS = _COMMON_FALSE_FIELDS | frozenset(
    {
        "provider_slots",
        "producer_attempt_chain",
        "failed_phase",
        "ordered_check_results",
        "resource_summary",
        "publication_summary",
        "decision",
        "stop_reason",
        "conditional_downstream",
        "artifact_sha256",
    }
)


# candidate_receipt_bundle.v1 exact top-level field set.
CANDIDATE_RECEIPT_BUNDLE_FIELDS = _COMMON_FALSE_FIELDS | frozenset(
    {
        "authorization_receipt",
        "run_identity_receipt",
        "run_admission_receipt",
        "static_input_contract",
        "candidate_published_history_head_receipt",
        "candidate_generation_name",
        "ordered_member_receipts",
        "observed_at_utc",
        "artifact_sha256",
    }
)

# postpublication verification (verified_result_v2) exact top-level field set.
POSTPUBLICATION_VERIFICATION_FIELDS = _COMMON_FALSE_FIELDS | frozenset(
    {
        "candidate_generation_name",
        "authorization_receipts",
        "run_history_contract_receipt",
        "run_admission_receipt",
        "static_input_contract",
        "candidate_receipt_bundle_receipt",
        "candidate_member_receipts",
        "prior_attempt_receipts",
        "producer_embedding_receipt",
        "verification_embedding_receipt",
        "verification_attempt_receipt",
        "ordered_check_results",
        "error_bound_result",
        "decision",
        "stop_reason",
        "evaluator_receipts",
        "publication_observation",
        "conditional_downstream",
        "artifact_sha256",
    }
)

# postverification failure exact top-level field set.
POSTPUBLICATION_FAILURE_FIELDS = _COMMON_FALSE_FIELDS | frozenset(
    {
        "dependency_provider_slots",
        "failed_check_name",
        "observed_result",
        "decision",
        "stop_reason",
        "final_result_published",
        "conditional_downstream",
        "artifact_sha256",
    }
)

# postverification failure: closed failed-check names and observation states.
POSTVERIFICATION_FAILED_CHECKS: tuple[str, ...] = (
    "producer_extraction",
    "verification_extraction",
    "projection_equality",
    "held_label_invariance",
    "per_fit_environment",
    "retrospective",
    "baseline_evaluator",
    "candidate_evaluator",
    "global_resource_caps",
)

_POSTVERIFICATION_STATES = {name: "computation_mismatch" for name in POSTVERIFICATION_FAILED_CHECKS}
_POSTVERIFICATION_STATES["global_resource_caps"] = "limit_failure"

# candidate_v2 exact ten members, in literal order (amendment §"Candidate generation").
CANDIDATE_MEMBER_PATHS: tuple[str, ...] = (
    "terminal_attempt/resource_guard.jsonl",
    "terminal_attempt/attempt_record.json",
    "producer_tiled_swin_embeddings.json",
    "verification_attempt/resource_guard.jsonl",
    "verification_attempt/attempt_record.json",
    "verification_tiled_swin_embeddings.json",
    "temporal_retrospective.json",
    "baseline_final_evaluator.json",
    "candidate_final_evaluator.json",
    "candidate_gate.json",
)

_PRODUCER_ATTEMPT_MEMBERS = frozenset(
    {
        "producer_attempt/resource_guard.jsonl",
        "producer_attempt/attempt_record.json",
    }
)
_PRE_VERIFICATION_MEMBERS = frozenset(
    {
        "terminal_attempt/resource_guard.jsonl",
        "terminal_attempt/attempt_record.json",
        "producer_tiled_swin_embeddings.json",
    }
)
_VERIFICATION_BEFORE_EMBEDDING_MEMBERS = _PRE_VERIFICATION_MEMBERS | frozenset(
    {
        "verification_attempt/resource_guard.jsonl",
        "verification_attempt/attempt_record.json",
    }
)
_VERIFICATION_AFTER_EMBEDDING_MEMBERS = _VERIFICATION_BEFORE_EMBEDDING_MEMBERS | frozenset(
    {
        "verification_tiled_swin_embeddings.json",
    }
)
_RETROSPECTIVE_MEMBERS = _VERIFICATION_AFTER_EMBEDDING_MEMBERS
_EVALUATOR_MEMBERS = _RETROSPECTIVE_MEMBERS | frozenset(
    {
        "temporal_retrospective.json",
    }
)

# Failure directory members (excluding the always-present `mechanical_failure.json`).
FAILED_PHASE_MEMBERS: dict[str, frozenset[str]] = {
    "producer_attempt_1": _PRODUCER_ATTEMPT_MEMBERS,
    "producer_attempt_2": _PRODUCER_ATTEMPT_MEMBERS,
    "producer_attempt_3": _PRODUCER_ATTEMPT_MEMBERS,
    "pre_verification": _PRE_VERIFICATION_MEMBERS,
    "verification_before_embedding": _VERIFICATION_BEFORE_EMBEDDING_MEMBERS,
    "verification_after_embedding": _VERIFICATION_AFTER_EMBEDDING_MEMBERS,
    "projection_equality": _RETROSPECTIVE_MEMBERS,
    "held_label_invariance": _RETROSPECTIVE_MEMBERS,
    "per_fit_environment": _RETROSPECTIVE_MEMBERS,
    "retrospective": _RETROSPECTIVE_MEMBERS,
    "baseline_evaluator": _EVALUATOR_MEMBERS,
    "candidate_evaluator": _EVALUATOR_MEMBERS,
    "global_resource_caps": _EVALUATOR_MEMBERS,
    "candidate_sealing": _EVALUATOR_MEMBERS,
}


def verify_candidate_member_coverage(paths: object) -> None:
    """Validate an exact candidate_v2 ten-member relative-path listing."""
    if not isinstance(paths, (list, tuple)) or tuple(paths) != CANDIDATE_MEMBER_PATHS:
        raise ValueError("candidate member coverage is invalid (must be the exact ten members in order)")


def verify_failure_member_coverage(phase: str, paths: object) -> None:
    """Validate an exact terminal_failure_v2 member listing for ``phase``.

    The listing must equal the phase's exact members plus ``mechanical_failure.json``
    in lexical relative-path order.
    """
    verify_failed_phase(phase)
    expected = sorted(FAILED_PHASE_MEMBERS[phase] | {"mechanical_failure.json"})
    if not isinstance(paths, (list, tuple)) or list(paths) != expected:
        raise ValueError("failure member coverage is invalid")


def verify_generation_member_receipt(row: object) -> None:
    """Validate an exact ``GenerationMemberReceipt`` row.

    ``{relative_path, receipt_kind, size_bytes, internal_sha256_field,
    artifact_sha256, file_sha256}`` with ``receipt_kind`` ``file_only`` or
    ``json``. For ``file_only``, internal/artifact are null; for ``json``,
    internal_sha256_field is a nonempty safe name and artifact_sha256 is a
    lowercase SHA-256. ``file_sha256`` is always lowercase SHA-256.
    """
    if not isinstance(row, Mapping) or set(row) != {
        "relative_path",
        "receipt_kind",
        "size_bytes",
        "internal_sha256_field",
        "artifact_sha256",
        "file_sha256",
    }:
        raise ValueError("GenerationMemberReceipt shape is invalid")
    if not isinstance(row["relative_path"], str) or not row["relative_path"]:
        raise ValueError("GenerationMemberReceipt relative_path is invalid")
    if not isinstance(row["size_bytes"], int) or isinstance(row["size_bytes"], bool) or row["size_bytes"] < 0:
        raise ValueError("GenerationMemberReceipt size_bytes is invalid")
    if not is_sha256(row["file_sha256"]):
        raise ValueError("GenerationMemberReceipt file_sha256 is invalid")
    kind = row["receipt_kind"]
    if kind == "file_only":
        if row["internal_sha256_field"] is not None or row["artifact_sha256"] is not None:
            raise ValueError("file_only member must have null internal/artifact hashes")
    elif kind == "json":
        if not isinstance(row["internal_sha256_field"], str) or not row["internal_sha256_field"]:
            raise ValueError("json member internal_sha256_field is invalid")
        if not is_sha256(row["artifact_sha256"]):
            raise ValueError("json member artifact_sha256 is invalid")
    else:
        raise ValueError("GenerationMemberReceipt receipt_kind is invalid")


# Exact provider-slot orders (amendment §"Failure generation",
# §"Candidate generation", §"Post-publication verification").
MECHANICAL_FAILURE_PROVIDERS: tuple[str, ...] = (
    "parent_spec_approval",
    "amendment_implementation_approval",
    "amended_implementation_review",
    "exact_v2_rerun_authorization",
    "run_history_ledger",
    "run_admission",
    "static_inputs",
    "producer_attempt",
    "producer_embedding",
    "verification_attempt",
    "verification_embedding",
    "retrospective",
)

CANDIDATE_INPUT_RECEIPT_PROVIDERS: tuple[str, ...] = (
    "parent_spec_approval",
    "amendment_implementation_approval",
    "amended_implementation_review",
    "exact_v2_rerun_authorization",
    "run_history_ledger",
    "run_admission",
    "static_inputs",
    "producer_embedding",
    "verification_embedding",
    "verification_attempt",
    "retrospective",
    "baseline_evaluator",
    "candidate_evaluator",
)

_POSTVERIFICATION_PREFIX: tuple[str, ...] = (
    "parent_spec_approval",
    "amendment_implementation_approval",
    "amended_implementation_review",
    "exact_v2_rerun_authorization",
    "run_history_ledger",
    "run_admission",
    "static_inputs",
)

# Role-fixed receipt kinds for the trust-spine providers; computation providers
# use the default ``artifact``/``generic`` kinds.
_TRUST_SPINE_RECEIPT_KINDS = {
    "run_history_ledger": "run_history",
    "static_inputs": "static_inputs",
}

_STORED_ARTIFACT_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "internal_sha256_field",
        "internal_sha256",
        "file_sha256",
        "filename",
        "size_bytes",
    }
)
_PRIOR_ATTEMPT_RECEIPT_FIELDS = frozenset(
    {"attempt_ordinal", "attempt_record_receipt", "resource_log_receipt", "resume_receipt"}
)


def _verify_file_receipt(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {"file_sha256", "filename", "size_bytes"}:
        raise ValueError("FileReceipt shape is invalid")
    if not is_sha256(value["file_sha256"]):
        raise ValueError("FileReceipt file_sha256 is invalid")
    if not isinstance(value["filename"], str) or not value["filename"] or "/" in value["filename"]:
        raise ValueError("FileReceipt filename is invalid")
    if not isinstance(value["size_bytes"], int) or isinstance(value["size_bytes"], bool) or value["size_bytes"] < 0:
        raise ValueError("FileReceipt size_bytes is invalid")


def _verify_stored_artifact_receipt(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != _STORED_ARTIFACT_RECEIPT_FIELDS:
        raise ValueError("StoredArtifactReceipt shape is invalid")
    if not isinstance(value["schema_version"], str) or not value["schema_version"]:
        raise ValueError("StoredArtifactReceipt schema_version is invalid")
    if not isinstance(value["internal_sha256_field"], str) or not value["internal_sha256_field"]:
        raise ValueError("StoredArtifactReceipt internal_sha256_field is invalid")
    if not is_sha256(value["internal_sha256"]) or not is_sha256(value["file_sha256"]):
        raise ValueError("StoredArtifactReceipt hashes are invalid")
    if not isinstance(value["filename"], str) or not value["filename"] or "/" in value["filename"]:
        raise ValueError("StoredArtifactReceipt filename is invalid")
    if not isinstance(value["size_bytes"], int) or isinstance(value["size_bytes"], bool) or value["size_bytes"] < 0:
        raise ValueError("StoredArtifactReceipt size_bytes is invalid")


def _verify_prior_attempt_receipts(value: object) -> None:
    if not isinstance(value, (list, tuple)) or len(value) > 2:
        raise ValueError("prior_attempt_receipts must contain zero to two rows")
    for ordinal, row in enumerate(value, start=1):
        if not isinstance(row, Mapping) or set(row) != _PRIOR_ATTEMPT_RECEIPT_FIELDS:
            raise ValueError("prior attempt receipt row shape is invalid")
        if row["attempt_ordinal"] != ordinal or isinstance(row["attempt_ordinal"], bool):
            raise ValueError("prior attempt receipt ordinal is invalid")
        _verify_stored_artifact_receipt(row["attempt_record_receipt"])
        _verify_file_receipt(row["resource_log_receipt"])
        if row["resume_receipt"] is not None:
            _verify_stored_artifact_receipt(row["resume_receipt"])


def verify_common_false_fields(payload: Mapping[str, object], *, schema_version: str) -> None:
    """Validate schema_version/module_id/purpose and the five hard-false flags."""
    if payload.get("schema_version") != schema_version or payload.get("module_id") != MODULE_ID:
        raise ValueError("artifact identity is invalid")
    if payload.get("purpose") != "development_diagnostic_only":
        raise ValueError("artifact purpose is invalid")
    for field in (
        "runtime_consumable",
        "training_consumable",
        "formal_evaluation_eligible",
        "promotion_eligible",
        "promoted",
    ):
        if payload.get(field) is not False:
            raise ValueError(f"eligibility flag {field!r} must be False")


def verify_candidate_gate(payload: Mapping[str, object]) -> None:
    """Validate a `agu.vru-causal-temporal-candidate-gate.v2` payload."""
    verify_exact_field_set(payload, CANDIDATE_GATE_FIELDS)
    verify_common_false_fields(payload, schema_version=CANDIDATE_GATE_SCHEMA_V2)
    verify_internal_artifact_hash(payload)
    if payload["publication_state"] != "not_yet_observed":
        raise ValueError("candidate gate publication_state must be not_yet_observed")
    if payload["postpublication_verification_required"] is not True:
        raise ValueError("candidate gate postpublication_verification_required must be True")
    verify_candidate_metric_outcome(payload["candidate_metric_outcome"])
    verify_ordered_checks(
        payload["ordered_prepublication_check_results"],
        CANDIDATE_PREPUBLICATION_CHECKS,
    )
    verify_provider_slots(
        payload["input_receipts"],
        CANDIDATE_INPUT_RECEIPT_PROVIDERS,
        receipt_kinds=_TRUST_SPINE_RECEIPT_KINDS,
    )
    producer_chain = payload["producer_attempt_chain"]
    if not isinstance(producer_chain, (list, tuple)):
        raise ValueError("candidate producer_attempt_chain must be a list")
    _verify_prior_attempt_receipts(producer_chain)
    verify_artifact_file_receipt(payload["verification_attempt_receipt"])
    downstream = payload["conditional_downstream"]
    if not isinstance(downstream, Mapping) or any(value is not False for value in downstream.values()):
        raise ValueError("candidate gate conditional_downstream must set every authority false")


def verify_mechanical_failure(payload: Mapping[str, object]) -> None:
    """Validate a `agu.vru-causal-temporal-mechanical-failure.v2` payload."""
    verify_exact_field_set(payload, MECHANICAL_FAILURE_FIELDS)
    verify_common_false_fields(payload, schema_version=MECHANICAL_FAILURE_SCHEMA_V2)
    verify_internal_artifact_hash(payload)
    if payload["decision"] != "mechanical_failure":
        raise ValueError("mechanical failure decision must be mechanical_failure")
    failed_phase = payload["failed_phase"]
    verify_failed_phase(failed_phase)
    verify_failed_phase_stop_reason(failed_phase, payload["stop_reason"])
    verify_failure_check_sequence(failed_phase, payload["ordered_check_results"])
    verify_provider_slots(
        payload["provider_slots"],
        MECHANICAL_FAILURE_PROVIDERS,
        receipt_kinds=_TRUST_SPINE_RECEIPT_KINDS,
    )
    if payload["publication_summary"] != {
        "target": "terminal_failure_v2",
        "publication_state": "not_yet_observed",
    }:
        raise ValueError("mechanical failure publication_summary is invalid")
    resource_summary = payload["resource_summary"]
    if not isinstance(resource_summary, Mapping) or set(resource_summary) != {
        "cumulative_active_runtime_nanoseconds",
        "cumulative_resource_samples",
        "cumulative_resource_log_bytes",
    }:
        raise ValueError("mechanical failure resource_summary shape is invalid")
    for value in resource_summary.values():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("mechanical failure resource_summary values are invalid")


def verify_candidate_receipt_bundle(payload: Mapping[str, object]) -> None:
    """Validate a `agu.vru-causal-temporal-candidate-receipt-bundle.v1` payload."""
    verify_exact_field_set(payload, CANDIDATE_RECEIPT_BUNDLE_FIELDS)
    verify_common_false_fields(payload, schema_version=CANDIDATE_RECEIPT_BUNDLE_SCHEMA)
    verify_internal_artifact_hash(payload)
    if payload["candidate_generation_name"] != "candidate_v2":
        raise ValueError("bundle candidate_generation_name must be candidate_v2")
    verify_artifact_file_receipt(payload["authorization_receipt"])
    verify_history_artifact_receipt(payload["run_identity_receipt"])
    verify_artifact_file_receipt(payload["run_admission_receipt"])
    verify_history_artifact_receipt(payload["candidate_published_history_head_receipt"])
    verify_static_input_contract(payload["static_input_contract"])
    members = payload["ordered_member_receipts"]
    if not isinstance(members, (list, tuple)) or len(members) != len(CANDIDATE_MEMBER_PATHS):
        raise ValueError("bundle ordered_member_receipts must be the ten candidate members")
    for i, row in enumerate(members):
        verify_generation_member_receipt(row)
        if row["relative_path"] != CANDIDATE_MEMBER_PATHS[i]:
            raise ValueError("bundle member row order mismatch")


def verify_postpublication_verification(payload: Mapping[str, object]) -> None:
    """Validate a `agu.vru-causal-temporal-postpublication-verification.v2` payload."""
    verify_exact_field_set(payload, POSTPUBLICATION_VERIFICATION_FIELDS)
    verify_common_false_fields(payload, schema_version=POSTPUBLICATION_VERIFICATION_SCHEMA_V2)
    verify_internal_artifact_hash(payload)
    verify_authorization_receipts(payload["authorization_receipts"])
    verify_run_history_contract_receipt(payload["run_history_contract_receipt"])
    verify_artifact_file_receipt(payload["run_admission_receipt"])
    verify_static_input_contract(payload["static_input_contract"])
    verify_artifact_file_receipt(payload["candidate_receipt_bundle_receipt"])
    candidate_members = payload["candidate_member_receipts"]
    if not isinstance(candidate_members, (list, tuple)) or len(candidate_members) != len(CANDIDATE_MEMBER_PATHS):
        raise ValueError("result candidate_member_receipts must contain all ten members")
    for expected_path, row in zip(CANDIDATE_MEMBER_PATHS, candidate_members):
        verify_generation_member_receipt(row)
        if row["relative_path"] != expected_path:
            raise ValueError("result candidate member receipt order mismatch")
    _verify_prior_attempt_receipts(payload["prior_attempt_receipts"])
    if payload["candidate_generation_name"] != "candidate_v2":
        raise ValueError("result candidate_generation_name must be candidate_v2")
    expected_member_rows = {row["relative_path"]: row for row in candidate_members}
    for field, expected_path in (
        ("producer_embedding_receipt", "producer_tiled_swin_embeddings.json"),
        ("verification_embedding_receipt", "verification_tiled_swin_embeddings.json"),
        ("verification_attempt_receipt", "verification_attempt/attempt_record.json"),
    ):
        verify_generation_member_receipt(payload[field])
        if payload[field] != expected_member_rows[expected_path]:
            raise ValueError(f"result {field} is not bound to its candidate member")
    evaluator_receipts = payload["evaluator_receipts"]
    if not isinstance(evaluator_receipts, Mapping) or set(evaluator_receipts) != {"baseline", "candidate"}:
        raise ValueError("result evaluator_receipts shape is invalid")
    for name, expected_path in (
        ("baseline", "baseline_final_evaluator.json"),
        ("candidate", "candidate_final_evaluator.json"),
    ):
        verify_generation_member_receipt(evaluator_receipts[name])
        if evaluator_receipts[name] != expected_member_rows[expected_path]:
            raise ValueError(f"result evaluator receipt {name} is not bound to its candidate member")
    checks = payload["ordered_check_results"]
    # all rows except the final error-bound row must pass
    verify_ordered_checks(checks[:-1], RESULT_ORDERED_CHECKS[:-1])
    last = checks[-1]
    if set(last) != {"check_name", "passed"} or last["check_name"] != "frozen_error_bounds":
        raise ValueError("result final check row is invalid")
    if not isinstance(last["passed"], bool):
        raise ValueError("result final check passed must be an exact boolean")
    last_passed = last["passed"]
    decision = payload["decision"]
    stop_reason = payload["stop_reason"]
    if last_passed:
        if decision != "mechanical_pass" or stop_reason is not None:
            raise ValueError("result decision/stop_reason inconsistent with passed error bounds")
    else:
        if decision != "temporal-hypothesis-rejected" or stop_reason != "temporal-hypothesis-rejected":
            raise ValueError("result decision/stop_reason inconsistent with failed error bounds")
    if payload["publication_observation"] != {
        "candidate_visible": True,
        "result_publication_state": "not_yet_observed",
        "bound_active_stage_is_only_stage": True,
    }:
        raise ValueError("result publication_observation is invalid")


def verify_postpublication_failure(payload: Mapping[str, object]) -> None:
    """Validate a `agu.vru-causal-temporal-postpublication-failure.v2` payload."""
    verify_exact_field_set(payload, POSTPUBLICATION_FAILURE_FIELDS)
    verify_common_false_fields(payload, schema_version=POSTPUBLICATION_FAILURE_SCHEMA_V2)
    verify_internal_artifact_hash(payload)
    if payload["decision"] != "mechanical_failure":
        raise ValueError("postverification failure decision must be mechanical_failure")
    if payload["final_result_published"] is not False:
        raise ValueError("postverification failure final_result_published must be False")
    failed_check = payload["failed_check_name"]
    if failed_check not in POSTVERIFICATION_FAILED_CHECKS:
        raise ValueError("postverification failure failed_check_name is invalid")
    if payload["stop_reason"] != failed_check:
        raise ValueError("postverification failure stop_reason must equal failed_check_name")
    expected_state = _POSTVERIFICATION_STATES[failed_check]
    if payload["observed_result"] != {
        "subject_kind": "check",
        "subject": failed_check,
        "observation_state": expected_state,
    }:
        raise ValueError("postverification failure observed_result is invalid")
    _verify_dependency_provider_slots(payload["dependency_provider_slots"])


def _verify_dependency_provider_slots(slots: object) -> None:
    """Validate postverification-failure ``dependency_provider_slots`` order.

    Exact structure: the seven trust-spine prefix in order, then zero or more
    per-attempt triples (record/log/resume) in increasing ordinal, then the
    final ``candidate_receipt_bundle`` slot.
    """
    if not isinstance(slots, (list, tuple)):
        raise ValueError("dependency_provider_slots must be a list")
    if len(slots) < len(_POSTVERIFICATION_PREFIX) + 1:
        raise ValueError("dependency_provider_slots is too short")
    prefix = slots[: len(_POSTVERIFICATION_PREFIX)]
    verify_provider_slots(
        prefix,
        _POSTVERIFICATION_PREFIX,
        receipt_kinds=_TRUST_SPINE_RECEIPT_KINDS,
    )
    middle = slots[len(_POSTVERIFICATION_PREFIX) : -1]
    if len(middle) % 3 != 0:
        raise ValueError("dependency_provider_slots attempt triple count is invalid")
    for slot in middle:
        verify_provider_slot(slot, receipt_kind="generic")
    final_slot = slots[-1]
    verify_provider_slot(final_slot, receipt_kind="artifact")
    if final_slot["provider"] != "candidate_receipt_bundle":
        raise ValueError("dependency_provider_slots must end with candidate_receipt_bundle")


__all__ = [
    "CANDIDATE_GATE_FIELDS",
    "MECHANICAL_FAILURE_FIELDS",
    "CANDIDATE_RECEIPT_BUNDLE_FIELDS",
    "POSTPUBLICATION_VERIFICATION_FIELDS",
    "POSTPUBLICATION_FAILURE_FIELDS",
    "POSTVERIFICATION_FAILED_CHECKS",
    "CANDIDATE_MEMBER_PATHS",
    "FAILED_PHASE_MEMBERS",
    "verify_common_false_fields",
    "verify_candidate_gate",
    "verify_mechanical_failure",
    "verify_candidate_receipt_bundle",
    "verify_postpublication_verification",
    "verify_postpublication_failure",
    "verify_candidate_member_coverage",
    "verify_failure_member_coverage",
    "verify_generation_member_receipt",
]
