"""Small, schema-valid candidate members used by v2 publication tests."""

from __future__ import annotations

from hashlib import sha256

from app.analysis.task0258_module_a_v2 import (
    MODULE_ID,
    TILED_SWIN_ATTEMPT_SCHEMA_V2,
    TILED_SWIN_EMBEDDING_SCHEMA_V2,
    V2_PRODUCER_ATTEMPT_FIELDS,
    V2_PRODUCER_EMBEDDING_FIELDS,
    VERIFICATION_ATTEMPT_SCHEMA_V2,
    VERIFICATION_EMBEDDING_SCHEMA_V2,
    canonical_artifact_sha256,
    compact_canonical_json,
)
from app.analysis.task0258_v2_artifacts import CANDIDATE_MEMBER_PATHS
from app.analysis.task0258_v2_read_isolation import (
    MAXIMUM_POLICY_BYTES,
    MAXIMUM_READ_EVENT_BYTES,
    MAXIMUM_READ_EVENT_ROWS,
    READ_EVENT_PROJECTION_PROTOCOL,
    READ_ISOLATION_ATTESTATION_SCHEMA,
    READ_ISOLATION_POLICY_SCHEMA,
)
from app.analysis.task0258_v2_verification import ROLE


def _receipt() -> dict[str, str]:
    return {"artifact_sha256": "0" * 64, "file_sha256": "0" * 64}


def _allowed_read_row() -> dict[str, object]:
    return {
        "locator_kind": "path",
        "path_role": "runtime_contract",
        "absolute_path": "/runtime/contract.json",
        "fd_number": None,
        "match_kind": "exact_regular",
        "entry_kind": "regular",
        "expected_device": 1,
        "expected_inode": 2,
        "expected_size_bytes": 1,
        "expected_file_sha256": "0" * 64,
        "expected_symlink_target_text": None,
        "expected_code_signature": None,
        "ordered_allowed_operations": ["open"],
    }


def _allowed_read_event() -> dict[str, object]:
    return {
        "ordinal": 1,
        "operation": "open",
        "path_role": "runtime_contract",
        "locator_kind": "path",
        "normalized_path": "/runtime/contract.json",
        "fd_number": None,
        "opened_fd_number": 3,
        "follow_symlinks": False,
        "access_mode": 0,
        "access_granted": True,
        "xattr_name": None,
        "result_state": "success",
        "errno": None,
        "entry_kind": "regular",
        "mode_bits": 0o600,
        "device": 1,
        "inode": 2,
        "size_bytes": 1,
        "file_sha256": "0" * 64,
        "symlink_target_text": None,
        "ordered_child_names": None,
        "xattr_value_sha256": None,
    }


def _auth() -> dict[str, dict[str, str]]:
    return {
        "parent_spec_approval": _receipt(),
        "amendment_implementation_approval": _receipt(),
        "amended_implementation_review": _receipt(),
        "rerun_authorization": _receipt(),
    }


def _common(schema: str) -> dict[str, object]:
    return {
        "schema_version": schema,
        "module_id": MODULE_ID,
        "purpose": "development_diagnostic_only",
        "runtime_consumable": False,
        "training_consumable": False,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
    }


def _policy() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": READ_ISOLATION_POLICY_SCHEMA,
        "module_id": MODULE_ID,
        "provider_receipt": _receipt(),
        "run_identity_receipt": _receipt(),
        "worker_role": "verification",
        "child_nonce": "0" * 64,
        "output_root_identity": {"absolute_path": "/output", "device": 1, "inode": 2},
        "runtime_snapshot_contract_input": {
            "runtime_root_absolute_path": "/runtime",
            "runtime_root_device": 1,
            "runtime_root_inode": 2,
            "runtime_contract_absolute_path": "/runtime/contract.json",
            "runtime_contract_size_bytes": 1,
            "runtime_contract_receipt": _receipt(),
            "runtime_manifest_absolute_path": "/runtime/manifest.json",
            "runtime_manifest_size_bytes": 1,
            "runtime_manifest_receipt": _receipt(),
            "runtime_tree_projection_sha256": "0" * 64,
        },
        "ordered_allowed_read_rows": [_allowed_read_row()],
        "ordered_denied_read_rows": [],
        "read_event_projection_protocol": READ_EVENT_PROJECTION_PROTOCOL,
        "maximum_read_event_rows": MAXIMUM_READ_EVENT_ROWS,
        "maximum_read_event_bytes": MAXIMUM_READ_EVENT_BYTES,
        "maximum_policy_bytes": MAXIMUM_POLICY_BYTES,
    }
    payload["artifact_sha256"] = canonical_artifact_sha256(payload)
    return payload


def _attestation(policy: dict[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": READ_ISOLATION_ATTESTATION_SCHEMA,
        "module_id": MODULE_ID,
        "policy_artifact_sha256": policy["artifact_sha256"],
        "provider_receipt": policy["provider_receipt"],
        "run_identity_receipt": policy["run_identity_receipt"],
        "worker_role": policy["worker_role"],
        "child_pid": 42,
        "ordered_observed_process_ids": [42],
        "provider_process_instance_id": "1" * 64,
        "child_nonce": policy["child_nonce"],
        "prepare_artifact_sha256": "0" * 64,
        "prepared_artifact_sha256": "0" * 64,
        "child_started_artifact_sha256": "0" * 64,
        "permit_artifact_sha256": "0" * 64,
        "finalize_artifact_sha256": "0" * 64,
        "audit_started_before_spawn": True,
        "audit_ended_after_child_exit": True,
        "audit_overflow": False,
        "ordered_observed_read_events": [_allowed_read_event()],
        "read_event_projection_sha256": "",
        "denied_read_attempt_count": 0,
        "unknown_read_attempt_count": 0,
    }
    payload["read_event_projection_sha256"] = sha256(
        compact_canonical_json(payload["ordered_observed_read_events"]).encode()
    ).hexdigest()
    payload["artifact_sha256"] = canonical_artifact_sha256(payload)
    return payload


def _producer_attempt(log: bytes) -> dict[str, object]:
    payload = _common(TILED_SWIN_ATTEMPT_SCHEMA_V2)
    payload.update(
        {
            "authorization_receipts": _auth(),
            "run_identity_receipt": _receipt(),
            "history_head_receipt": _receipt(),
            "run_admission_receipt": _receipt(),
            "worker_request": {},
            "child_observation": {},
            "worker_payload": {},
            "read_isolation_policy": {},
            "read_isolation_attestation": {},
            "attempt_ordinal": 1,
            "prior_attempt_receipt": None,
            "plan_receipt": _receipt(),
            "task0257_input_receipts": {},
            "started_prefix_count": 0,
            "completed_prefix_count": 0,
            "new_rows_verified": 0,
            "cumulative_active_runtime_nanoseconds": 0,
            "cumulative_resource_samples": 0,
            "cumulative_resource_log_bytes": len(log),
            "resource_log_receipt": {"size_bytes": len(log), "file_sha256": sha256(log).hexdigest()},
            "resume_input_cas": None,
            "resume_output_cas": None,
            "received_signal": None,
            "disposition": "completed",
            "stop_reason": None,
        }
    )
    payload["artifact_sha256"] = canonical_artifact_sha256(payload)
    assert set(payload) == V2_PRODUCER_ATTEMPT_FIELDS
    return payload


def _producer_embedding() -> dict[str, object]:
    payload = _common(TILED_SWIN_EMBEDDING_SCHEMA_V2)
    payload.update(
        {
            "authorization_receipts": _auth(),
            "run_identity_receipt": _receipt(),
            "history_head_receipt": _receipt(),
            "run_admission_receipt": _receipt(),
            "attempt_chain": [],
            "examples": [],
            "formal_evaluation_eligible": False,
            "plan_receipt": _receipt(),
            "producer_environment": {},
            "representation": "test",
            "row_count": 45,
            "task0257_input_receipts": {},
        }
    )
    payload["artifact_sha256"] = canonical_artifact_sha256(payload)
    assert set(payload) == V2_PRODUCER_EMBEDDING_FIELDS
    return payload


def _verification_embedding() -> dict[str, object]:
    payload = _common(VERIFICATION_EMBEDDING_SCHEMA_V2)
    payload.update(
        {
            "role": ROLE,
            "authorization_receipts": _auth(),
            "run_identity_receipt": _receipt(),
            "history_head_receipt": _receipt(),
            "run_admission_receipt": _receipt(),
            "plan_receipt": _receipt(),
            "task0257_input_receipts": [],
            "representation": "test",
            "producer_environment": {},
            "checkpoint_receipt": _receipt(),
            "source_video_receipts": [],
            "row_count": 45,
            "examples": [],
        }
    )
    payload["artifact_sha256"] = canonical_artifact_sha256(payload)
    return payload


def _verification_attempt(log: bytes) -> dict[str, object]:
    policy = _policy()
    payload = _common(VERIFICATION_ATTEMPT_SCHEMA_V2)
    payload.update(
        {
            "verification_ordinal": 1,
            "authorization_receipts": _auth(),
            "run_identity_receipt": _receipt(),
            "history_head_receipt": _receipt(),
            "run_admission_receipt": _receipt(),
            "plan_receipt": _receipt(),
            "task0257_input_receipts": [],
            "checkpoint_receipt": _receipt(),
            "source_video_receipts": [],
            "producer_attempt_chain_receipts": [],
            "producer_embedding_receipt": _receipt(),
            "worker_request": {},
            "child_observation": {},
            "worker_payload": {},
            "read_isolation_policy": policy,
            "read_isolation_attestation": _attestation(policy),
            "verification_embedding_slot": {
                "provider": "verification_tiled_swin_embeddings",
                "verification_state": "verified",
                "receipt": _receipt(),
            },
            "resource_log_receipt": {"size_bytes": len(log), "file_sha256": sha256(log).hexdigest()},
            "started_cumulative_active_runtime_nanoseconds": 0,
            "ended_cumulative_active_runtime_nanoseconds": 0,
            "started_cumulative_resource_samples": 0,
            "ended_cumulative_resource_samples": 0,
            "started_cumulative_resource_log_bytes": 0,
            "ended_cumulative_resource_log_bytes": len(log),
            "computational_projection_sha256": "0" * 64,
            "received_signal": None,
            "disposition": "completed",
            "stop_reason": None,
        }
    )
    payload["artifact_sha256"] = canonical_artifact_sha256(payload)
    return payload


def _parent_artifact(schema: str, *, role: str | None = None) -> dict[str, object]:
    if schema.endswith("retrospective.v1"):
        payload = {
            **_common(schema),
            "input_receipts": {},
            "selection_protocol": "test",
            "source_selection_limitation": {},
            "outer_folds": [],
            "per_fit_environment": {},
            "held_label_invariance": {},
            "archived_task0257_comparator": {},
            "error_bound_result": {},
            "temporal_hypothesis_decision": "test",
        }
    else:
        payload = {
            **_common(schema),
            "evaluator_role": role,
            "input_receipts": {},
            "representation_candidates": [],
            "ordered_training_rows": [],
            "logo_selection": [],
            "selected_evaluator": {},
            "all_45_refit": {},
            "conditional_downstream": {},
        }
    payload["artifact_sha256"] = canonical_artifact_sha256(payload)
    return payload


def candidate_members(candidate_gate: dict[str, object] | None = None) -> dict[str, bytes]:
    terminal_log = b""
    verification_log = b""
    payloads: dict[str, object] = {
        "terminal_attempt/attempt_record.json": _producer_attempt(terminal_log),
        "producer_tiled_swin_embeddings.json": _producer_embedding(),
        "verification_attempt/attempt_record.json": _verification_attempt(verification_log),
        "verification_tiled_swin_embeddings.json": _verification_embedding(),
        "temporal_retrospective.json": _parent_artifact("agu.vru-causal-temporal-retrospective.v1"),
        "baseline_final_evaluator.json": _parent_artifact("agu.vru-causal-final-evaluator.v1", role="baseline"),
        "candidate_final_evaluator.json": _parent_artifact("agu.vru-causal-final-evaluator.v1", role="candidate"),
        "candidate_gate.json": candidate_gate,
    }
    if candidate_gate is None:
        raise ValueError("candidate fixture requires a caller-provided candidate gate")
    bound_gate = dict(candidate_gate)
    verification_bytes = (compact_canonical_json(payloads["verification_attempt/attempt_record.json"]) + "\n").encode()
    bound_gate["verification_attempt_receipt"] = {
        "artifact_sha256": payloads["verification_attempt/attempt_record.json"]["artifact_sha256"],
        "file_sha256": sha256(verification_bytes).hexdigest(),
    }
    bound_gate["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in bound_gate.items() if key != "artifact_sha256"}
    )
    payloads["candidate_gate.json"] = bound_gate
    return {
        relative_path: (
            terminal_log
            if relative_path == "terminal_attempt/resource_guard.jsonl"
            else verification_log
            if relative_path == "verification_attempt/resource_guard.jsonl"
            else (compact_canonical_json(payloads[relative_path]) + "\n").encode("utf-8")
        )
        for relative_path in CANDIDATE_MEMBER_PATHS
    }
