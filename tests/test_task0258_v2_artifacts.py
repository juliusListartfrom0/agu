"""Tests for TASK-0258 v2 artifact schemas (candidate gate + mechanical failure)."""

from __future__ import annotations

import pytest

from app.analysis.task0258_module_a_v2 import (
    CANDIDATE_GATE_SCHEMA_V2,
    CANDIDATE_RECEIPT_BUNDLE_SCHEMA,
    MECHANICAL_FAILURE_SCHEMA_V2,
    MODULE_ID,
    POSTPUBLICATION_FAILURE_SCHEMA_V2,
    POSTPUBLICATION_VERIFICATION_SCHEMA_V2,
    canonical_artifact_sha256,
)
from app.analysis.task0258_v2_artifacts import (
    CANDIDATE_INPUT_RECEIPT_PROVIDERS,
    CANDIDATE_MEMBER_PATHS,
    MECHANICAL_FAILURE_PROVIDERS,
    verify_candidate_gate,
    verify_candidate_member_coverage,
    verify_candidate_receipt_bundle,
    verify_common_false_fields,
    verify_failure_member_coverage,
    verify_generation_member_receipt,
    verify_mechanical_failure,
    verify_postpublication_failure,
    verify_postpublication_verification,
)
from app.analysis.task0258_v2_gate import (
    CANDIDATE_PREPUBLICATION_CHECKS,
    RESULT_ORDERED_CHECKS,
    expected_failure_check_sequence,
)


def _false_common(schema_version):
    return {
        "schema_version": schema_version,
        "module_id": MODULE_ID,
        "purpose": "development_diagnostic_only",
        "runtime_consumable": False,
        "training_consumable": False,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
    }


def _receipt():
    return {"artifact_sha256": "0" * 64, "file_sha256": "0" * 64}


def _seal(payload):
    payload["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )
    return payload


def _slot(provider, state="verified"):
    return {"provider": provider, "verification_state": state, "receipt": _receipt() if state == "verified" else None}


def _run_history_receipt():
    return {"run_identity_receipt": _receipt(), "head_receipt": _receipt(), "marker_count": 0}


def _static_input_contract():
    return {
        "temporal_plan_artifact_sha256": "0" * 64,
        "temporal_plan_file_sha256": "0" * 64,
        "task0257_receipts_projection_sha256": "0" * 64,
    }


def _trust_slots(providers):
    out = []
    for p in providers:
        if p == "run_history_ledger":
            out.append({"provider": p, "verification_state": "verified", "receipt": _run_history_receipt()})
        elif p == "static_inputs":
            out.append({"provider": p, "verification_state": "verified", "receipt": _static_input_contract()})
        else:
            out.append(_slot(p))
    return out


def _member_row(path):
    is_jsonl = path.endswith(".jsonl")
    return {
        "relative_path": path,
        "receipt_kind": "file_only" if is_jsonl else "json",
        "size_bytes": 0,
        "internal_sha256_field": None if is_jsonl else "artifact_sha256",
        "artifact_sha256": None if is_jsonl else "0" * 64,
        "file_sha256": "0" * 64,
    }


def _candidate_gate():
    p = _false_common(CANDIDATE_GATE_SCHEMA_V2)
    p.update(
        {
            "input_receipts": _trust_slots(CANDIDATE_INPUT_RECEIPT_PROVIDERS),
            "producer_attempt_chain": [],
            "verification_attempt_receipt": None,
            "ordered_prepublication_check_results": [
                {"check_name": name, "passed": True} for name in CANDIDATE_PREPUBLICATION_CHECKS
            ],
            "error_bound_result": {},
            "candidate_metric_outcome": "within_frozen_error_bounds",
            "publication_state": "not_yet_observed",
            "postpublication_verification_required": True,
            "conditional_downstream": {},
            "artifact_sha256": "0" * 64,
        }
    )
    return _seal(p)


def _mechanical_failure(phase="candidate_sealing"):
    from app.analysis.task0258_v2_gate import FAILED_PHASE_ALLOWED_REASONS

    stop_reason = next(iter(sorted(FAILED_PHASE_ALLOWED_REASONS[phase])))
    p = _false_common(MECHANICAL_FAILURE_SCHEMA_V2)
    p.update(
        {
            "provider_slots": _trust_slots(MECHANICAL_FAILURE_PROVIDERS),
            "producer_attempt_chain": [],
            "failed_phase": phase,
            "ordered_check_results": [
                {"check_name": name, "passed": (i < len(expected_failure_check_sequence(phase)) - 1)}
                for i, name in enumerate(expected_failure_check_sequence(phase))
            ],
            "resource_summary": {
                "cumulative_active_runtime_nanoseconds": 0,
                "cumulative_resource_samples": 0,
                "cumulative_resource_log_bytes": 0,
            },
            "publication_summary": {
                "target": "terminal_failure_v2",
                "publication_state": "not_yet_observed",
            },
            "decision": "mechanical_failure",
            "stop_reason": stop_reason,
            "conditional_downstream": {},
            "artifact_sha256": "0" * 64,
        }
    )
    return _seal(p)


def test_verify_common_false_fields():
    verify_common_false_fields(_false_common(CANDIDATE_GATE_SCHEMA_V2), schema_version=CANDIDATE_GATE_SCHEMA_V2)
    with pytest.raises(ValueError):
        verify_common_false_fields(
            {**_false_common(CANDIDATE_GATE_SCHEMA_V2), "runtime_consumable": True},
            schema_version=CANDIDATE_GATE_SCHEMA_V2,
        )
    with pytest.raises(ValueError):
        verify_common_false_fields(
            {**_false_common(CANDIDATE_GATE_SCHEMA_V2), "module_id": "bogus"}, schema_version=CANDIDATE_GATE_SCHEMA_V2
        )


def test_candidate_gate_valid():
    verify_candidate_gate(_candidate_gate())


def test_candidate_gate_rejects():
    # publication_state must be not_yet_observed
    bad = _candidate_gate()
    bad["publication_state"] = "verified_terminal_result"
    with pytest.raises(ValueError):
        verify_candidate_gate(bad)
    # postpublication_verification_required must be True
    bad = _candidate_gate()
    bad["postpublication_verification_required"] = False
    with pytest.raises(ValueError):
        verify_candidate_gate(bad)
    # candidate_metric_outcome may not be mechanical_pass
    bad = _candidate_gate()
    bad["candidate_metric_outcome"] = "mechanical_pass"
    with pytest.raises(ValueError):
        verify_candidate_gate(bad)
    # a check row must not be False
    bad = _candidate_gate()
    bad["ordered_prepublication_check_results"][0]["passed"] = False
    with pytest.raises(ValueError):
        verify_candidate_gate(bad)


def test_mechanical_failure_valid():
    verify_mechanical_failure(_mechanical_failure("candidate_sealing"))
    verify_mechanical_failure(_mechanical_failure("producer_attempt_1"))


def test_mechanical_failure_rejects():
    bad = _mechanical_failure("candidate_sealing")
    bad["decision"] = "temporal-hypothesis-rejected"
    with pytest.raises(ValueError):
        verify_mechanical_failure(bad)
    # stop_reason must be allowed for the phase
    bad = _mechanical_failure("candidate_sealing")
    bad["stop_reason"] = "decode_failure"
    with pytest.raises(ValueError):
        verify_mechanical_failure(bad)
    # publication_summary pinned
    bad = _mechanical_failure("candidate_sealing")
    bad["publication_summary"]["target"] = "verified_result_v2"
    with pytest.raises(ValueError):
        verify_mechanical_failure(bad)


def _bundle():
    p = _false_common(CANDIDATE_RECEIPT_BUNDLE_SCHEMA)
    p.update(
        {
            "authorization_receipt": {"artifact_sha256": "0" * 64, "file_sha256": "0" * 64},
            "run_identity_receipt": {"artifact_sha256": "0" * 64, "file_sha256": "0" * 64},
            "run_admission_receipt": {"artifact_sha256": "0" * 64, "file_sha256": "0" * 64},
            "static_input_contract": {
                "temporal_plan_artifact_sha256": "0" * 64,
                "temporal_plan_file_sha256": "0" * 64,
                "task0257_receipts_projection_sha256": "0" * 64,
            },
            "candidate_published_history_head_receipt": {"artifact_sha256": "0" * 64, "file_sha256": "0" * 64},
            "candidate_generation_name": "candidate_v2",
            "ordered_member_receipts": [_member_row(p) for p in CANDIDATE_MEMBER_PATHS],
            "observed_at_utc": "2026-08-17T00:00:00Z",
            "artifact_sha256": "0" * 64,
        }
    )
    return _seal(p)


def test_candidate_receipt_bundle_valid():
    verify_candidate_receipt_bundle(_bundle())


def test_candidate_receipt_bundle_rejects():
    bad = _bundle()
    bad["candidate_generation_name"] = "candidate_v1"
    with pytest.raises(ValueError):
        verify_candidate_receipt_bundle(bad)
    bad = _bundle()
    bad["static_input_contract"] = {"bogus": "x"}
    with pytest.raises(ValueError):
        verify_candidate_receipt_bundle(bad)


def _result(pass_error_bounds=True):
    p = _false_common(POSTPUBLICATION_VERIFICATION_SCHEMA_V2)
    p.update(
        {
            "candidate_generation_name": "candidate_v2",
            "authorization_receipts": {
                "parent_spec_approval": _receipt(),
                "amendment_implementation_approval": _receipt(),
                "amended_implementation_review": _receipt(),
                "rerun_authorization": _receipt(),
            },
            "run_history_contract_receipt": {
                "run_identity_receipt": _receipt(),
                "head_receipt": _receipt(),
                "marker_count": 0,
            },
            "run_admission_receipt": _receipt(),
            "static_input_contract": {
                "temporal_plan_artifact_sha256": "0" * 64,
                "temporal_plan_file_sha256": "0" * 64,
                "task0257_receipts_projection_sha256": "0" * 64,
            },
            "candidate_receipt_bundle_receipt": _receipt(),
            "candidate_member_receipts": [],
            "prior_attempt_receipts": [],
            "producer_embedding_receipt": _receipt(),
            "verification_embedding_receipt": _receipt(),
            "verification_attempt_receipt": _receipt(),
            "ordered_check_results": [
                {"check_name": name, "passed": (name != "frozen_error_bounds" or pass_error_bounds)}
                for name in RESULT_ORDERED_CHECKS
            ],
            "error_bound_result": {},
            "decision": "mechanical_pass" if pass_error_bounds else "temporal-hypothesis-rejected",
            "stop_reason": None if pass_error_bounds else "temporal-hypothesis-rejected",
            "evaluator_receipts": {},
            "publication_observation": {
                "candidate_visible": True,
                "result_publication_state": "not_yet_observed",
                "bound_active_stage_is_only_stage": True,
            },
            "conditional_downstream": {},
            "artifact_sha256": "0" * 64,
        }
    )
    return _seal(p)


def test_postpublication_verification_valid():
    verify_postpublication_verification(_result(pass_error_bounds=True))
    verify_postpublication_verification(_result(pass_error_bounds=False))


def test_postpublication_verification_rejects():
    bad = _result(pass_error_bounds=True)
    bad["decision"] = "temporal-hypothesis-rejected"
    with pytest.raises(ValueError):
        verify_postpublication_verification(bad)
    bad = _result(pass_error_bounds=False)
    bad["stop_reason"] = None
    with pytest.raises(ValueError):
        verify_postpublication_verification(bad)
    bad = _result(pass_error_bounds=True)
    bad["publication_observation"]["result_publication_state"] = "published"
    with pytest.raises(ValueError):
        verify_postpublication_verification(bad)


def _postfailure(check="retrospective"):
    p = _false_common(POSTPUBLICATION_FAILURE_SCHEMA_V2)
    p.update(
        {
            "dependency_provider_slots": _trust_slots(
                (
                    "parent_spec_approval",
                    "amendment_implementation_approval",
                    "amended_implementation_review",
                    "exact_v2_rerun_authorization",
                    "run_history_ledger",
                    "run_admission",
                    "static_inputs",
                )
            )
            + [_slot("candidate_receipt_bundle")],
            "failed_check_name": check,
            "observed_result": {
                "subject_kind": "check",
                "subject": check,
                "observation_state": "computation_mismatch",
            },
            "decision": "mechanical_failure",
            "stop_reason": check,
            "final_result_published": False,
            "conditional_downstream": {},
        }
    )
    return _seal(p)


def test_postpublication_failure_valid():
    verify_postpublication_failure(_postfailure("retrospective"))
    # global_resource_caps maps to limit_failure
    g = _postfailure("global_resource_caps")
    g["observed_result"]["observation_state"] = "limit_failure"
    g = _seal(g)
    verify_postpublication_failure(g)


def test_postpublication_failure_rejects():
    bad = _postfailure("retrospective")
    bad["final_result_published"] = True
    with pytest.raises(ValueError):
        verify_postpublication_failure(bad)
    bad = _postfailure("retrospective")
    bad["stop_reason"] = "other"
    with pytest.raises(ValueError):
        verify_postpublication_failure(bad)
    bad = _postfailure("bogus_check")
    with pytest.raises(ValueError):
        verify_postpublication_failure(bad)
    # global_resource_caps may not be computation_mismatch
    bad = _postfailure("global_resource_caps")
    with pytest.raises(ValueError):
        verify_postpublication_failure(bad)


def test_candidate_member_paths_exact():
    assert CANDIDATE_MEMBER_PATHS == (
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
    verify_candidate_member_coverage(list(CANDIDATE_MEMBER_PATHS))
    with pytest.raises(ValueError):
        verify_candidate_member_coverage(list(CANDIDATE_MEMBER_PATHS[:-1]))


def test_failure_member_coverage():
    verify_failure_member_coverage(
        "producer_attempt_1",
        ["mechanical_failure.json", "producer_attempt/attempt_record.json", "producer_attempt/resource_guard.jsonl"],
    )
    # candidate_evaluator includes temporal_retrospective but not baseline evaluator
    verify_failure_member_coverage(
        "candidate_evaluator",
        sorted(
            [
                "mechanical_failure.json",
                "terminal_attempt/resource_guard.jsonl",
                "terminal_attempt/attempt_record.json",
                "producer_tiled_swin_embeddings.json",
                "verification_attempt/resource_guard.jsonl",
                "verification_attempt/attempt_record.json",
                "verification_tiled_swin_embeddings.json",
                "temporal_retrospective.json",
            ]
        ),
    )
    with pytest.raises(ValueError):
        verify_failure_member_coverage("producer_attempt_1", ["mechanical_failure.json"])


def test_generation_member_receipt():
    verify_generation_member_receipt(
        {
            "relative_path": "candidate_gate.json",
            "receipt_kind": "json",
            "size_bytes": 10,
            "internal_sha256_field": "artifact_sha256",
            "artifact_sha256": "0" * 64,
            "file_sha256": "0" * 64,
        }
    )
    verify_generation_member_receipt(
        {
            "relative_path": "terminal_attempt/resource_guard.jsonl",
            "receipt_kind": "file_only",
            "size_bytes": 0,
            "internal_sha256_field": None,
            "artifact_sha256": None,
            "file_sha256": "0" * 64,
        }
    )
    with pytest.raises(ValueError):
        verify_generation_member_receipt(
            {
                "relative_path": "x.json",
                "receipt_kind": "file_only",
                "size_bytes": 0,
                "internal_sha256_field": None,
                "artifact_sha256": "0" * 64,
                "file_sha256": "0" * 64,
            }
        )
    with pytest.raises(ValueError):
        verify_generation_member_receipt(
            {
                "relative_path": "x.json",
                "receipt_kind": "bogus",
                "size_bytes": 0,
                "internal_sha256_field": None,
                "artifact_sha256": None,
                "file_sha256": "0" * 64,
            }
        )


def test_candidate_gate_rejects_downstream_true():
    p = _candidate_gate()
    p["conditional_downstream"] = {"module_b": True}
    with pytest.raises(ValueError):
        verify_candidate_gate(p)
