"""Adversarial tests for TASK-0258 v2 review-only contexts/loaders."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from app.analysis.task0258_module_a_v2 import (
    VERIFICATION_ATTEMPT_SCHEMA_V2,
    canonical_artifact_sha256,
    compact_canonical_json,
)
from app.analysis.task0258_run_history import ADMISSION_SCHEMA, CLAIM_SCHEMA, COMPLETION_SCHEMA, MODULE_ID
from app.analysis.task0258_v2_artifacts import CANDIDATE_MEMBER_PATHS
from app.analysis.task0258_v2_capabilities import (
    VerifiedImplementationReviewSandboxContext,
    VerifiedReviewDiscoveryContext,
    VerifiedReviewImplementationApproval,
    VerifiedReviewNoWritePreflight,
    VerifiedReviewReadIsolationBinding,
    VerifiedReviewVerificationAttempt,
    VerifiedReviewVerificationAttemptRunSpine,
    bind_implementation_review_discovery_context,
    bind_implementation_review_sandbox_context,
    bind_synthetic_module_a_discovery_context,
    bind_synthetic_module_a_transaction_context,
    bind_verified_review_attempt_to_run_spine,
    bind_verified_review_no_write_preflight,
    exercise_module_a_v2_state_machine_for_discovery,
    exercise_module_a_v2_state_machine_for_review,
    load_verified_candidate_receipt_bundle,
    load_verified_read_isolation_binding,
    load_verified_review_implementation_approval,
    load_verified_run_admission,
    load_verified_run_history_ledger,
    load_verified_terminal_artifact,
    load_verified_verification_attempt,
    replay_module_a_read_traversal_for_discovery,
)
from app.analysis.task0258_v2_pipeline import (
    build_candidate_gate_payload,
    build_candidate_receipt_bundle_payload,
    build_member_receipts,
    build_postpublication_failure_payload,
    build_postpublication_verification_payload,
    seal_candidate_v2,
    seal_postverification_failure,
    seal_verified_result,
)
from app.analysis.task0258_v2_read_isolation import (
    MAXIMUM_POLICY_BYTES,
    MAXIMUM_READ_EVENT_BYTES,
    MAXIMUM_READ_EVENT_ROWS,
    READ_EVENT_PROJECTION_PROTOCOL,
    READ_ISOLATION_ATTESTATION_SCHEMA,
    READ_ISOLATION_POLICY_SCHEMA,
)
from app.analysis.task0258_v2_registry import seal_run_consumption_claim, seal_run_consumption_completed


def _receipt():
    return {"artifact_sha256": "0" * 64, "file_sha256": "0" * 64}


def _trust_slots(providers):
    out = []
    for provider in providers:
        if provider == "run_history_ledger":
            receipt = {"run_identity_receipt": _receipt(), "head_receipt": _receipt(), "marker_count": 0}
        elif provider == "static_inputs":
            receipt = {
                "temporal_plan_artifact_sha256": "0" * 64,
                "temporal_plan_file_sha256": "0" * 64,
                "task0257_receipts_projection_sha256": "0" * 64,
            }
        else:
            receipt = _receipt()
        out.append({"provider": provider, "verification_state": "verified", "receipt": receipt})
    return out


def _gate():
    return build_candidate_gate_payload(
        input_receipts=_trust_slots(
            (
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
        ),
        producer_attempt_chain=[],
        verification_attempt_receipt=_receipt(),
        error_bound_result={},
        candidate_metric_outcome="within_frozen_error_bounds",
    )


def _members(*, gate=None):
    members = {}
    for rel in CANDIDATE_MEMBER_PATHS:
        if rel.endswith(".jsonl"):
            members[rel] = b'{"role":"test"}\n'
        elif rel == "candidate_gate.json":
            members[rel] = (compact_canonical_json(gate or _gate()) + "\n").encode()
        else:
            payload = {"schema_version": "agu.test"}
            payload["artifact_sha256"] = canonical_artifact_sha256(payload)
            members[rel] = (compact_canonical_json(payload) + "\n").encode()
    return members


def _payload_receipt(payload):
    raw = (compact_canonical_json(payload) + "\n").encode()
    return {"artifact_sha256": payload["artifact_sha256"], "file_sha256": hashlib.sha256(raw).hexdigest()}


def _file_hashes(path, payload):
    return payload["artifact_sha256"], hashlib.sha256(path.read_bytes()).hexdigest()


def _read_isolation_fixture(tmp_path):
    observed_file = tmp_path / "runtime-contract.json"
    observed_file.write_text("{}\n")
    observed_stat = observed_file.stat()
    policy = {
        "schema_version": READ_ISOLATION_POLICY_SCHEMA,
        "module_id": MODULE_ID,
        "provider_receipt": _receipt(),
        "run_identity_receipt": _receipt(),
        "worker_role": "verification-worker",
        "child_nonce": "1" * 64,
        "output_root_identity": {
            "absolute_path": str(tmp_path),
            "device": tmp_path.stat().st_dev,
            "inode": tmp_path.stat().st_ino,
        },
        "runtime_snapshot_contract_input": {
            "runtime_root_absolute_path": str(tmp_path),
            "runtime_root_device": tmp_path.stat().st_dev,
            "runtime_root_inode": tmp_path.stat().st_ino,
            "runtime_contract_absolute_path": str(observed_file),
            "runtime_contract_size_bytes": observed_stat.st_size,
            "runtime_contract_receipt": _receipt(),
            "runtime_manifest_absolute_path": str(observed_file),
            "runtime_manifest_size_bytes": observed_stat.st_size,
            "runtime_manifest_receipt": _receipt(),
            "runtime_tree_projection_sha256": "2" * 64,
        },
        "ordered_allowed_read_rows": [
            {
                "locator_kind": "path",
                "path_role": "runtime_contract",
                "absolute_path": str(observed_file),
                "fd_number": None,
                "match_kind": "exact_regular",
                "entry_kind": "regular",
                "expected_device": observed_stat.st_dev,
                "expected_inode": observed_stat.st_ino,
                "expected_size_bytes": observed_stat.st_size,
                "expected_file_sha256": hashlib.sha256(observed_file.read_bytes()).hexdigest(),
                "expected_symlink_target_text": None,
                "expected_code_signature": None,
                "ordered_allowed_operations": ["open", "read"],
            }
        ],
        "ordered_denied_read_rows": [],
        "read_event_projection_protocol": READ_EVENT_PROJECTION_PROTOCOL,
        "maximum_read_event_rows": MAXIMUM_READ_EVENT_ROWS,
        "maximum_read_event_bytes": MAXIMUM_READ_EVENT_BYTES,
        "maximum_policy_bytes": MAXIMUM_POLICY_BYTES,
    }
    policy["artifact_sha256"] = canonical_artifact_sha256(policy)
    event = {
        "ordinal": 1,
        "operation": "open",
        "path_role": "runtime_contract",
        "locator_kind": "path",
        "normalized_path": str(observed_file),
        "fd_number": None,
        "opened_fd_number": 9,
        "follow_symlinks": False,
        "access_mode": 0,
        "access_granted": True,
        "xattr_name": None,
        "result_state": "success",
        "errno": None,
        "entry_kind": "regular",
        "mode_bits": observed_stat.st_mode,
        "device": observed_stat.st_dev,
        "inode": observed_stat.st_ino,
        "size_bytes": observed_stat.st_size,
        "file_sha256": hashlib.sha256(observed_file.read_bytes()).hexdigest(),
        "symlink_target_text": None,
        "ordered_child_names": None,
        "xattr_value_sha256": None,
    }
    event_projection = hashlib.sha256(compact_canonical_json([event]).encode()).hexdigest()
    attestation = {
        "schema_version": READ_ISOLATION_ATTESTATION_SCHEMA,
        "module_id": MODULE_ID,
        "policy_artifact_sha256": policy["artifact_sha256"],
        "provider_receipt": policy["provider_receipt"],
        "run_identity_receipt": policy["run_identity_receipt"],
        "worker_role": policy["worker_role"],
        "child_pid": 1234,
        "ordered_observed_process_ids": [1234],
        "provider_process_instance_id": "3" * 64,
        "child_nonce": policy["child_nonce"],
        "prepare_artifact_sha256": "4" * 64,
        "prepared_artifact_sha256": "5" * 64,
        "child_started_artifact_sha256": "6" * 64,
        "permit_artifact_sha256": "7" * 64,
        "finalize_artifact_sha256": "8" * 64,
        "audit_started_before_spawn": True,
        "audit_ended_after_child_exit": True,
        "audit_overflow": False,
        "ordered_observed_read_events": [event],
        "read_event_projection_sha256": event_projection,
        "denied_read_attempt_count": 0,
        "unknown_read_attempt_count": 0,
    }
    attestation["artifact_sha256"] = canonical_artifact_sha256(attestation)
    policy_path = tmp_path / "read-isolation-policy.json"
    attestation_path = tmp_path / "read-isolation-attestation.json"
    policy_raw = (compact_canonical_json(policy) + "\n").encode()
    attestation_raw = (compact_canonical_json(attestation) + "\n").encode()
    policy_path.write_bytes(policy_raw)
    attestation_path.write_bytes(attestation_raw)
    return {
        "policy": policy,
        "attestation": attestation,
        "policy_path": policy_path,
        "attestation_path": attestation_path,
        "policy_file_sha": hashlib.sha256(policy_raw).hexdigest(),
        "attestation_file_sha": hashlib.sha256(attestation_raw).hexdigest(),
    }


def _verification_attempt_fixture(tmp_path):
    read_isolation = _read_isolation_fixture(tmp_path)
    authorization_receipts = {
        "parent_spec_approval": _receipt(),
        "amendment_implementation_approval": _receipt(),
        "amended_implementation_review": _receipt(),
        "rerun_authorization": _receipt(),
    }
    attempt = {
        "schema_version": VERIFICATION_ATTEMPT_SCHEMA_V2,
        "module_id": MODULE_ID,
        "purpose": "development_diagnostic_only",
        "runtime_consumable": False,
        "training_consumable": False,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
        "verification_ordinal": 1,
        "authorization_receipts": authorization_receipts,
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
        "read_isolation_policy": read_isolation["policy"],
        "read_isolation_attestation": read_isolation["attestation"],
        "verification_embedding_slot": {
            "provider": "verification_tiled_swin_embeddings",
            "verification_state": "verified",
            "receipt": _receipt(),
        },
        "resource_log_receipt": _receipt(),
        "started_cumulative_active_runtime_nanoseconds": 0,
        "ended_cumulative_active_runtime_nanoseconds": 0,
        "started_cumulative_resource_samples": 0,
        "ended_cumulative_resource_samples": 0,
        "started_cumulative_resource_log_bytes": 0,
        "ended_cumulative_resource_log_bytes": 0,
        "computational_projection_sha256": "0" * 64,
        "received_signal": None,
        "disposition": "completed",
        "stop_reason": None,
    }
    attempt["artifact_sha256"] = canonical_artifact_sha256(attempt)
    attempt_path = tmp_path / "verification-attempt.json"
    raw = (compact_canonical_json(attempt) + "\n").encode()
    attempt_path.write_bytes(raw)
    return {
        "attempt": attempt,
        "attempt_path": attempt_path,
        "attempt_file_sha": hashlib.sha256(raw).hexdigest(),
        "review": bind_implementation_review_sandbox_context(
            expected_check_name="focused_pytest", expected_command_sha256="0" * 64
        ),
    }


def _bound_gate(admission, history):
    history_contract = {
        "run_identity_receipt": _payload_receipt(history.payloads[1]),
        "head_receipt": _payload_receipt(history.payloads[-1]),
        "marker_count": len(history.payloads) - 2,
    }
    slots = _trust_slots(
        (
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
    )
    by_provider = {slot["provider"]: slot for slot in slots}
    by_provider["exact_v2_rerun_authorization"]["receipt"] = admission.admission.payload["authorization_receipt"]
    by_provider["run_history_ledger"]["receipt"] = history_contract
    by_provider["run_admission"]["receipt"] = _payload_receipt(admission.admission.payload)
    by_provider["static_inputs"]["receipt"] = admission.admission.payload["static_input_contract"]
    return build_candidate_gate_payload(
        input_receipts=slots,
        producer_attempt_chain=[],
        verification_attempt_receipt=_receipt(),
        error_bound_result={},
        candidate_metric_outcome="within_frozen_error_bounds",
    )


def _admission_fixture(tmp_path):
    auth = "a" * 64
    authorization_receipt = {"artifact_sha256": auth, "file_sha256": "b" * 64}
    root = tmp_path / "vru_causal_temporal_retrospective_v2"
    registry = tmp_path / "registry"
    registry.mkdir()
    claim = {
        "schema_version": CLAIM_SCHEMA,
        "module_id": MODULE_ID,
        "authorization_receipt": authorization_receipt,
        "run_id": "run-1",
        "output_root_absolute_path": str(root),
        "nonce": "c" * 64,
        "state": "claimed",
        "created_at_utc": "2026-08-21T00:00:00Z",
    }
    claim["artifact_sha256"] = canonical_artifact_sha256(claim)
    seal_run_consumption_claim(registry, auth, claim)
    claim_path = registry / f"{auth}.claim.json"
    admission = {
        "schema_version": ADMISSION_SCHEMA,
        "module_id": MODULE_ID,
        "authorization_receipt": authorization_receipt,
        "claim_receipt": _payload_receipt(claim),
        "nonce": claim["nonce"],
        "run_id": claim["run_id"],
        "output_root_absolute_path": str(root),
        "static_input_contract": {
            "temporal_plan_artifact_sha256": "d" * 64,
            "temporal_plan_file_sha256": "e" * 64,
            "task0257_receipts_projection_sha256": "f" * 64,
        },
        "maximum_run_count": 1,
        "admission_state": "admitted",
        "module_b_authorized": False,
        "created_at_utc": "2026-08-21T00:00:01Z",
    }
    admission["artifact_sha256"] = canonical_artifact_sha256(admission)
    admission_bytes = (compact_canonical_json(admission) + "\n").encode()
    root_lock = tmp_path / "output-root.lock"
    from app.analysis.task0258_v2_fs import seal_generation_directory

    seal_generation_directory(
        root.parent,
        root.name,
        {"run_admission.json": admission_bytes},
        ("run_admission.json",),
        flock_path=root_lock,
    )
    admission_path = root / "run_admission.json"
    root_stat = os.stat(root, follow_symlinks=False)
    admission_stat = os.stat(admission_path, follow_symlinks=False)
    completion = {
        "schema_version": COMPLETION_SCHEMA,
        "module_id": MODULE_ID,
        "authorization_receipt": authorization_receipt,
        "claim_receipt": _payload_receipt(claim),
        "admission_receipt": _payload_receipt(admission),
        "nonce": claim["nonce"],
        "run_id": claim["run_id"],
        "output_root_absolute_path": str(root),
        "consumption_count": 1,
        "state": "completed",
        "root_identity": {"device": root_stat.st_dev, "inode": root_stat.st_ino},
        "admission_identity": {
            "device": admission_stat.st_dev,
            "inode": admission_stat.st_ino,
            "size_bytes": admission_stat.st_size,
            "internal_sha256": admission["artifact_sha256"],
            "file_sha256": hashlib.sha256(admission_path.read_bytes()).hexdigest(),
        },
        "created_at_utc": "2026-08-21T00:00:02Z",
    }
    completion["artifact_sha256"] = canonical_artifact_sha256(completion)
    seal_run_consumption_completed(registry, auth, completion)
    completion_path = registry / f"{auth}.completed.json"
    review = bind_implementation_review_sandbox_context(
        expected_check_name="focused_pytest", expected_command_sha256="0" * 64
    )
    claim_sha, claim_file_sha = _file_hashes(claim_path, claim)
    admission_sha, admission_file_sha = _file_hashes(admission_path, admission)
    completion_sha, completion_file_sha = _file_hashes(completion_path, completion)
    loaded = load_verified_run_admission(
        execution_context=review,
        claim_path=claim_path,
        admission_path=admission_path,
        completion_path=completion_path,
        expected_authorization_sha256=auth,
        expected_claim_artifact_sha256=claim_sha,
        expected_claim_file_sha256=claim_file_sha,
        expected_admission_artifact_sha256=admission_sha,
        expected_admission_file_sha256=admission_file_sha,
        expected_completion_artifact_sha256=completion_sha,
        expected_completion_file_sha256=completion_file_sha,
    )
    history = load_verified_run_history_ledger(registry_directory=registry, authorization_sha256=auth)
    return {
        "auth": auth,
        "root": root,
        "registry": registry,
        "lock": root_lock,
        "review": review,
        "admission": loaded,
        "history": history,
    }


def _candidate_bundle_fixture(fixture):
    root = fixture["root"]
    admission = fixture["admission"].admission
    history = fixture["history"]
    candidate = seal_candidate_v2(
        root,
        _members(gate=_bound_gate(fixture["admission"], history)),
        flock_path=fixture["lock"],
    )
    history_contract = {
        "run_identity_receipt": _payload_receipt(history.payloads[1]),
        "head_receipt": _payload_receipt(history.payloads[-1]),
        "marker_count": len(history.payloads) - 2,
    }
    candidate_rows = build_member_receipts(candidate)
    bundle = build_candidate_receipt_bundle_payload(
        authorization_receipt=admission.payload["authorization_receipt"],
        run_identity_receipt=history_contract["run_identity_receipt"],
        run_admission_receipt=_payload_receipt(admission.payload),
        static_input_contract=admission.payload["static_input_contract"],
        candidate_published_history_head_receipt=history_contract["head_receipt"],
        candidate_dir=candidate,
        observed_at_utc="2026-08-21T00:00:03Z",
    )
    bundle_parent = root.parent / "bundle-parent"
    bundle_parent.mkdir()
    bundle_path = bundle_parent / "candidate-receipt-bundle.json"
    bundle_raw = (compact_canonical_json(bundle) + "\n").encode()
    bundle_path.write_bytes(bundle_raw)
    member_by_path = {row["relative_path"]: row for row in candidate_rows}
    result = build_postpublication_verification_payload(
        error_bounds_pass=True,
        authorization_receipts={
            "parent_spec_approval": _receipt(),
            "amendment_implementation_approval": _receipt(),
            "amended_implementation_review": _receipt(),
            "rerun_authorization": admission.payload["authorization_receipt"],
        },
        run_history_contract_receipt=history_contract,
        run_admission_receipt=_payload_receipt(admission.payload),
        static_input_contract=admission.payload["static_input_contract"],
        candidate_receipt_bundle_receipt={
            "artifact_sha256": bundle["artifact_sha256"],
            "file_sha256": hashlib.sha256(bundle_raw).hexdigest(),
        },
        candidate_member_receipts=candidate_rows,
        prior_attempt_receipts=[],
        producer_embedding_receipt=member_by_path["producer_tiled_swin_embeddings.json"],
        verification_embedding_receipt=member_by_path["verification_tiled_swin_embeddings.json"],
        verification_attempt_receipt=member_by_path["verification_attempt/attempt_record.json"],
        error_bound_result={},
        evaluator_receipts={
            "baseline": member_by_path["baseline_final_evaluator.json"],
            "candidate": member_by_path["candidate_final_evaluator.json"],
        },
    )
    return {
        **fixture,
        "candidate": candidate,
        "bundle": bundle,
        "bundle_path": bundle_path,
        "bundle_file_sha": hashlib.sha256(bundle_raw).hexdigest(),
        "result": result,
    }


def test_review_contexts_are_opaque_and_distinct():
    with pytest.raises(TypeError):
        VerifiedReviewDiscoveryContext()
    with pytest.raises(TypeError):
        VerifiedImplementationReviewSandboxContext()
    with pytest.raises(TypeError):
        VerifiedReviewReadIsolationBinding()
    with pytest.raises(TypeError):
        VerifiedReviewNoWritePreflight()
    with pytest.raises(TypeError):
        VerifiedReviewImplementationApproval()
    with pytest.raises(TypeError):
        VerifiedReviewVerificationAttempt()
    with pytest.raises(TypeError):
        VerifiedReviewVerificationAttemptRunSpine()
    discovery = bind_implementation_review_discovery_context(
        expected_check_name="focused_pytest", expected_command_sha256="0" * 64
    )
    review = bind_implementation_review_sandbox_context(
        expected_check_name="focused_pytest", expected_command_sha256="0" * 64
    )
    assert type(discovery) is not type(review)
    with pytest.raises(PermissionError):
        bind_synthetic_module_a_transaction_context(review_sandbox=discovery, isolated_temp_ancestor=None)


def test_discovery_manifest_is_temp_bound_and_observation_only(tmp_path):
    discovery = bind_implementation_review_discovery_context(
        expected_check_name="focused_pytest", expected_command_sha256="0" * 64
    )
    payload = {"schema_version": "agu.test"}
    payload["artifact_sha256"] = canonical_artifact_sha256(payload)
    manifest = tmp_path / "manifest.json"
    raw = (compact_canonical_json(payload) + "\n").encode()
    manifest.write_bytes(raw)
    observation = replay_module_a_read_traversal_for_discovery(
        discovery_context=discovery,
        operation="manifest_replay",
        operation_input_manifest_path=manifest,
        expected_manifest_artifact_sha256=payload["artifact_sha256"],
        expected_manifest_file_sha256=hashlib.sha256(raw).hexdigest(),
    )
    assert observation["production_capability"] is False
    with pytest.raises(ValueError):
        replay_module_a_read_traversal_for_discovery(
            discovery_context=discovery,
            operation="manifest_replay",
            operation_input_manifest_path=tmp_path.parent / "outside.json",
            expected_manifest_artifact_sha256=payload["artifact_sha256"],
            expected_manifest_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_implementation_approval_loader_replays_closed_review_artifact(tmp_path):
    source = (
        Path(__file__).resolve().parents[1]
        / "analysis_outputs/public_research/task0258_module_a_amendment_approval"
        / "amendment_implementation_approval.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    repository_root = Path(__file__).resolve().parents[1]
    payload["repository_root_device"] = repository_root.stat().st_dev
    payload["repository_root_inode"] = repository_root.stat().st_ino
    payload["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )
    raw = (compact_canonical_json(payload) + "\n").encode()
    approval_path = tmp_path / "implementation-approval.json"
    approval_path.write_bytes(raw)
    review = bind_implementation_review_sandbox_context(
        expected_check_name="focused_pytest", expected_command_sha256="0" * 64
    )

    loaded = load_verified_review_implementation_approval(
        execution_context=review,
        approval_path=approval_path,
        expected_artifact_sha256=payload["artifact_sha256"],
        expected_file_sha256=hashlib.sha256(raw).hexdigest(),
    )

    assert isinstance(loaded, VerifiedReviewImplementationApproval)
    assert loaded.production_capability is False
    assert loaded.repository_root_identity["device"] == payload["repository_root_device"]
    assert loaded.artifact.payload["model_execution_authorized"] is False

    drifted = dict(payload)
    drifted["model_execution_authorized"] = True
    drifted["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in drifted.items() if key != "artifact_sha256"}
    )
    drifted_raw = (compact_canonical_json(drifted) + "\n").encode()
    approval_path.write_bytes(drifted_raw)
    with pytest.raises(ValueError, match="execution flags"):
        load_verified_review_implementation_approval(
            execution_context=review,
            approval_path=approval_path,
            expected_artifact_sha256=drifted["artifact_sha256"],
            expected_file_sha256=hashlib.sha256(drifted_raw).hexdigest(),
        )


def test_discovery_manifest_rejects_symlinked_temp_parent(tmp_path):
    discovery = bind_implementation_review_discovery_context(
        expected_check_name="focused_pytest", expected_command_sha256="0" * 64
    )
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    symlinked_parent = tmp_path / "symlinked-parent"
    symlinked_parent.symlink_to(real_parent, target_is_directory=True)
    payload = {"schema_version": "agu.test"}
    payload["artifact_sha256"] = canonical_artifact_sha256(payload)
    raw = (compact_canonical_json(payload) + "\n").encode()
    manifest = real_parent / "manifest.json"
    manifest.write_bytes(raw)

    with pytest.raises(ValueError):
        replay_module_a_read_traversal_for_discovery(
            discovery_context=discovery,
            operation="manifest_replay",
            operation_input_manifest_path=symlinked_parent / manifest.name,
            expected_manifest_artifact_sha256=payload["artifact_sha256"],
            expected_manifest_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_read_isolation_loader_binds_policy_and_attestation_for_review_only(tmp_path):
    fixture = _read_isolation_fixture(tmp_path)
    review = bind_implementation_review_sandbox_context(
        expected_check_name="focused_pytest", expected_command_sha256="0" * 64
    )

    loaded = load_verified_read_isolation_binding(
        execution_context=review,
        policy_path=fixture["policy_path"],
        expected_policy_artifact_sha256=fixture["policy"]["artifact_sha256"],
        expected_policy_file_sha256=fixture["policy_file_sha"],
        attestation_path=fixture["attestation_path"],
        expected_attestation_artifact_sha256=fixture["attestation"]["artifact_sha256"],
        expected_attestation_file_sha256=fixture["attestation_file_sha"],
    )

    assert loaded.policy.payload["artifact_sha256"] == fixture["policy"]["artifact_sha256"]
    assert loaded.attestation.payload["policy_artifact_sha256"] == fixture["policy"]["artifact_sha256"]
    assert type(loaded).__name__ == "VerifiedReviewReadIsolationBinding"
    with pytest.raises(PermissionError):
        load_verified_read_isolation_binding(
            execution_context=object(),
            policy_path=fixture["policy_path"],
            expected_policy_artifact_sha256=fixture["policy"]["artifact_sha256"],
            expected_policy_file_sha256=fixture["policy_file_sha"],
            attestation_path=fixture["attestation_path"],
            expected_attestation_artifact_sha256=fixture["attestation"]["artifact_sha256"],
            expected_attestation_file_sha256=fixture["attestation_file_sha"],
        )


def test_read_isolation_loader_rejects_cross_artifact_binding_drift(tmp_path):
    fixture = _read_isolation_fixture(tmp_path)
    drifted = dict(fixture["attestation"])
    drifted["worker_role"] = "producer-worker"
    drifted["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in drifted.items() if key != "artifact_sha256"}
    )
    raw = (compact_canonical_json(drifted) + "\n").encode()
    fixture["attestation_path"].write_bytes(raw)

    review = bind_implementation_review_sandbox_context(
        expected_check_name="focused_pytest", expected_command_sha256="0" * 64
    )
    with pytest.raises(ValueError, match="worker role"):
        load_verified_read_isolation_binding(
            execution_context=review,
            policy_path=fixture["policy_path"],
            expected_policy_artifact_sha256=fixture["policy"]["artifact_sha256"],
            expected_policy_file_sha256=fixture["policy_file_sha"],
            attestation_path=fixture["attestation_path"],
            expected_attestation_artifact_sha256=drifted["artifact_sha256"],
            expected_attestation_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_read_isolation_loader_rejects_symlinked_parent(tmp_path):
    fixture = _read_isolation_fixture(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    review = bind_implementation_review_sandbox_context(
        expected_check_name="focused_pytest", expected_command_sha256="0" * 64
    )
    with pytest.raises(ValueError):
        load_verified_read_isolation_binding(
            execution_context=review,
            policy_path=alias / fixture["policy_path"].name,
            expected_policy_artifact_sha256=fixture["policy"]["artifact_sha256"],
            expected_policy_file_sha256=fixture["policy_file_sha"],
            attestation_path=fixture["attestation_path"],
            expected_attestation_artifact_sha256=fixture["attestation"]["artifact_sha256"],
            expected_attestation_file_sha256=fixture["attestation_file_sha"],
        )


def test_verification_attempt_loader_replays_bound_read_isolation(tmp_path):
    fixture = _verification_attempt_fixture(tmp_path)
    loaded = load_verified_verification_attempt(
        execution_context=fixture["review"],
        attempt_path=fixture["attempt_path"],
        expected_artifact_sha256=fixture["attempt"]["artifact_sha256"],
        expected_file_sha256=fixture["attempt_file_sha"],
    )

    assert isinstance(loaded, VerifiedReviewVerificationAttempt)
    assert loaded.artifact.payload["schema_version"] == VERIFICATION_ATTEMPT_SCHEMA_V2
    assert (
        loaded.artifact.payload["read_isolation_attestation"]["policy_artifact_sha256"]
        == (loaded.artifact.payload["read_isolation_policy"]["artifact_sha256"])
    )


def test_verification_attempt_loader_rejects_nested_read_isolation_drift(tmp_path):
    fixture = _verification_attempt_fixture(tmp_path)
    drifted = dict(fixture["attempt"])
    drifted_attestation = dict(drifted["read_isolation_attestation"])
    drifted_attestation["worker_role"] = "producer-worker"
    drifted_attestation["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in drifted_attestation.items() if key != "artifact_sha256"}
    )
    drifted["read_isolation_attestation"] = drifted_attestation
    drifted["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in drifted.items() if key != "artifact_sha256"}
    )
    raw = (compact_canonical_json(drifted) + "\n").encode()
    fixture["attempt_path"].write_bytes(raw)

    with pytest.raises(ValueError, match="worker role"):
        load_verified_verification_attempt(
            execution_context=fixture["review"],
            attempt_path=fixture["attempt_path"],
            expected_artifact_sha256=drifted["artifact_sha256"],
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        )


def test_verification_attempt_binds_to_run_admission_and_history_spine(tmp_path):
    admission_fixture = _admission_fixture(tmp_path)
    attempt_fixture = _verification_attempt_fixture(admission_fixture["root"])
    attempt = dict(attempt_fixture["attempt"])
    history = admission_fixture["history"]
    admission = admission_fixture["admission"].admission
    history_contract = {
        "run_identity_receipt": _payload_receipt(history.payloads[1]),
        "head_receipt": _payload_receipt(history.payloads[-1]),
        "marker_count": len(history.payloads) - 2,
    }
    attempt["authorization_receipts"] = {
        **attempt["authorization_receipts"],
        "rerun_authorization": admission.payload["authorization_receipt"],
    }
    attempt["run_identity_receipt"] = history_contract["run_identity_receipt"]
    attempt["history_head_receipt"] = history_contract["head_receipt"]
    attempt["run_admission_receipt"] = _payload_receipt(admission.payload)
    policy = dict(attempt["read_isolation_policy"])
    policy["run_identity_receipt"] = attempt["run_identity_receipt"]
    policy["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in policy.items() if key != "artifact_sha256"}
    )
    attestation = dict(attempt["read_isolation_attestation"])
    attestation["policy_artifact_sha256"] = policy["artifact_sha256"]
    attestation["run_identity_receipt"] = attempt["run_identity_receipt"]
    attestation["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in attestation.items() if key != "artifact_sha256"}
    )
    attempt["read_isolation_policy"] = policy
    attempt["read_isolation_attestation"] = attestation
    attempt["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in attempt.items() if key != "artifact_sha256"}
    )
    raw = (compact_canonical_json(attempt) + "\n").encode()
    attempt_fixture["attempt_path"].write_bytes(raw)
    loaded_attempt = load_verified_verification_attempt(
        execution_context=attempt_fixture["review"],
        attempt_path=attempt_fixture["attempt_path"],
        expected_artifact_sha256=attempt["artifact_sha256"],
        expected_file_sha256=hashlib.sha256(raw).hexdigest(),
    )

    bound = bind_verified_review_attempt_to_run_spine(
        attempt=loaded_attempt,
        run_admission=admission_fixture["admission"],
        run_history=history,
    )

    assert isinstance(bound, VerifiedReviewVerificationAttemptRunSpine)
    assert bound.attempt is loaded_attempt


def test_review_no_write_preflight_replays_identities_without_writing(tmp_path):
    admission_fixture = _admission_fixture(tmp_path)
    attempt_root = tmp_path / "attempt-artifacts"
    attempt_root.mkdir()
    attempt_fixture = _verification_attempt_fixture(attempt_root)
    attempt = dict(attempt_fixture["attempt"])
    history = admission_fixture["history"]
    admission = admission_fixture["admission"].admission
    history_contract = {
        "run_identity_receipt": _payload_receipt(history.payloads[1]),
        "head_receipt": _payload_receipt(history.payloads[-1]),
        "marker_count": len(history.payloads) - 2,
    }
    attempt["authorization_receipts"] = {
        **attempt["authorization_receipts"],
        "rerun_authorization": admission.payload["authorization_receipt"],
    }
    attempt["run_identity_receipt"] = history_contract["run_identity_receipt"]
    attempt["history_head_receipt"] = history_contract["head_receipt"]
    attempt["run_admission_receipt"] = _payload_receipt(admission.payload)
    policy = dict(attempt["read_isolation_policy"])
    policy["run_identity_receipt"] = attempt["run_identity_receipt"]
    policy["output_root_identity"] = {
        "absolute_path": str(admission_fixture["root"]),
        "device": admission_fixture["root"].stat().st_dev,
        "inode": admission_fixture["root"].stat().st_ino,
    }
    policy["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in policy.items() if key != "artifact_sha256"}
    )
    attestation = dict(attempt["read_isolation_attestation"])
    attestation["policy_artifact_sha256"] = policy["artifact_sha256"]
    attestation["run_identity_receipt"] = attempt["run_identity_receipt"]
    attestation["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in attestation.items() if key != "artifact_sha256"}
    )
    attempt["read_isolation_policy"] = policy
    attempt["read_isolation_attestation"] = attestation
    attempt["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in attempt.items() if key != "artifact_sha256"}
    )
    raw = (compact_canonical_json(attempt) + "\n").encode()
    attempt_fixture["attempt_path"].write_bytes(raw)
    loaded_attempt = load_verified_verification_attempt(
        execution_context=attempt_fixture["review"],
        attempt_path=attempt_fixture["attempt_path"],
        expected_artifact_sha256=attempt["artifact_sha256"],
        expected_file_sha256=hashlib.sha256(raw).hexdigest(),
    )
    spine = bind_verified_review_attempt_to_run_spine(
        attempt=loaded_attempt,
        run_admission=admission_fixture["admission"],
        run_history=history,
    )
    candidate_bundle_path = tmp_path / "bundle-parent" / "candidate-receipt-bundle.json"
    candidate_bundle_path.parent.mkdir()
    before_registry = sorted(path.name for path in admission_fixture["registry"].iterdir())
    before_root = sorted(path.name for path in admission_fixture["root"].iterdir())

    preflight = bind_verified_review_no_write_preflight(
        execution_context=admission_fixture["review"],
        attempt_spine=spine,
        candidate_bundle_path=candidate_bundle_path,
    )

    assert isinstance(preflight, VerifiedReviewNoWritePreflight)
    assert preflight.production_capability is False
    assert preflight.output_root_identity == {
        "device": admission_fixture["root"].stat().st_dev,
        "inode": admission_fixture["root"].stat().st_ino,
    }
    assert not candidate_bundle_path.exists()
    assert sorted(path.name for path in admission_fixture["registry"].iterdir()) == before_registry
    assert sorted(path.name for path in admission_fixture["root"].iterdir()) == before_root
    (admission_fixture["root"] / "candidate_v2").mkdir()
    with pytest.raises(ValueError, match="reserved output"):
        bind_verified_review_no_write_preflight(
            execution_context=admission_fixture["review"],
            attempt_spine=spine,
            candidate_bundle_path=candidate_bundle_path,
        )


def test_review_no_write_preflight_rejects_unbound_spine(tmp_path):
    admission_fixture = _admission_fixture(tmp_path)
    with pytest.raises(TypeError, match="attempt spine"):
        bind_verified_review_no_write_preflight(
            execution_context=admission_fixture["review"],
            attempt_spine=object(),
            candidate_bundle_path=tmp_path / "bundle.json",
        )


def test_verification_attempt_rejects_run_admission_receipt_drift(tmp_path):
    admission_fixture = _admission_fixture(tmp_path)
    attempt_fixture = _verification_attempt_fixture(admission_fixture["root"])
    loaded_attempt = attempt_fixture["attempt"]
    loaded_attempt["run_admission_receipt"] = _receipt()
    loaded_attempt["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in loaded_attempt.items() if key != "artifact_sha256"}
    )
    raw = (compact_canonical_json(loaded_attempt) + "\n").encode()
    attempt_fixture["attempt_path"].write_bytes(raw)
    attempt = load_verified_verification_attempt(
        execution_context=attempt_fixture["review"],
        attempt_path=attempt_fixture["attempt_path"],
        expected_artifact_sha256=loaded_attempt["artifact_sha256"],
        expected_file_sha256=hashlib.sha256(raw).hexdigest(),
    )

    with pytest.raises(ValueError, match="admission receipt"):
        bind_verified_review_attempt_to_run_spine(
            attempt=attempt,
            run_admission=admission_fixture["admission"],
            run_history=admission_fixture["history"],
        )


def test_synthetic_context_reopens_and_rejects_identity_drift(tmp_path):
    discovery = bind_implementation_review_discovery_context(
        expected_check_name="focused_pytest", expected_command_sha256="0" * 64
    )
    review = bind_implementation_review_sandbox_context(
        expected_check_name="focused_pytest", expected_command_sha256="0" * 64
    )
    discovery_context = bind_synthetic_module_a_discovery_context(
        discovery_context=discovery, isolated_temp_ancestor=tmp_path
    )
    review_context = bind_synthetic_module_a_transaction_context(review_sandbox=review, isolated_temp_ancestor=tmp_path)
    scenario = {"scenario_id": "claim_then_publish", "steps": [{"state": "fresh", "event": "claim"}]}
    assert (
        exercise_module_a_v2_state_machine_for_discovery(synthetic_context=discovery_context, scenario=scenario)[
            "production_capability"
        ]
        is False
    )
    assert (
        exercise_module_a_v2_state_machine_for_review(synthetic_context=review_context, scenario=scenario)[
            "observed_step_count"
        ]
        == 1
    )
    replacement = tmp_path.with_name(tmp_path.name + "-replacement")
    tmp_path.rename(replacement)
    with pytest.raises(ValueError):
        exercise_module_a_v2_state_machine_for_review(synthetic_context=review_context, scenario=scenario)


def test_candidate_bundle_loader_replays_bytes_and_rejects_wrong_context(tmp_path):
    root = tmp_path / "output"
    root.mkdir()
    lock = root / ".lock"
    lock.write_text("")
    candidate = seal_candidate_v2(root, _members(), flock_path=lock)
    bundle = build_candidate_receipt_bundle_payload(
        authorization_receipt=_receipt(),
        run_identity_receipt=_receipt(),
        run_admission_receipt=_receipt(),
        static_input_contract={
            "temporal_plan_artifact_sha256": "0" * 64,
            "temporal_plan_file_sha256": "0" * 64,
            "task0257_receipts_projection_sha256": "0" * 64,
        },
        candidate_published_history_head_receipt=_receipt(),
        candidate_dir=candidate,
        observed_at_utc="2026-08-17T00:00:00Z",
    )
    bundle_parent = tmp_path / "bundle-parent"
    bundle_parent.mkdir()
    bundle_path = bundle_parent / "bundle.json"
    bundle_raw = (compact_canonical_json(bundle) + "\n").encode()
    bundle_path.write_bytes(bundle_raw)
    review = bind_implementation_review_sandbox_context(
        expected_check_name="focused_pytest", expected_command_sha256="0" * 64
    )
    loaded = load_verified_candidate_receipt_bundle(
        execution_context=review,
        bundle_path=bundle_path,
        candidate_dir=candidate,
        expected_artifact_sha256=bundle["artifact_sha256"],
        expected_file_sha256=hashlib.sha256(bundle_raw).hexdigest(),
    )
    assert loaded.payload["candidate_generation_name"] == "candidate_v2"
    with pytest.raises(PermissionError):
        load_verified_candidate_receipt_bundle(
            execution_context=object(),
            bundle_path=bundle_path,
            candidate_dir=candidate,
            expected_artifact_sha256=bundle["artifact_sha256"],
            expected_file_sha256=hashlib.sha256(bundle_raw).hexdigest(),
        )
    symlinked_parent = tmp_path / "bundle-alias"
    symlinked_parent.symlink_to(bundle_parent, target_is_directory=True)
    with pytest.raises(ValueError):
        load_verified_candidate_receipt_bundle(
            execution_context=review,
            bundle_path=symlinked_parent / bundle_path.name,
            candidate_dir=candidate,
            expected_artifact_sha256=bundle["artifact_sha256"],
            expected_file_sha256=hashlib.sha256(bundle_raw).hexdigest(),
        )
    with pytest.raises(ValueError):
        load_verified_candidate_receipt_bundle(
            execution_context=review,
            bundle_path=bundle_path,
            candidate_dir=Path("/private/etc/candidate_v2"),
            expected_artifact_sha256=bundle["artifact_sha256"],
            expected_file_sha256=hashlib.sha256(bundle_raw).hexdigest(),
        )


def test_candidate_bundle_loader_rejects_candidate_member_drift(tmp_path):
    root = tmp_path / "output"
    root.mkdir()
    lock = root / ".lock"
    lock.write_text("")
    candidate = seal_candidate_v2(root, _members(), flock_path=lock)
    bundle = build_candidate_receipt_bundle_payload(
        authorization_receipt=_receipt(),
        run_identity_receipt=_receipt(),
        run_admission_receipt=_receipt(),
        static_input_contract={
            "temporal_plan_artifact_sha256": "0" * 64,
            "temporal_plan_file_sha256": "0" * 64,
            "task0257_receipts_projection_sha256": "0" * 64,
        },
        candidate_published_history_head_receipt=_receipt(),
        candidate_dir=candidate,
        observed_at_utc="2026-08-17T00:00:00Z",
    )
    bundle_parent = tmp_path / "bundle-parent"
    bundle_parent.mkdir()
    bundle_path = bundle_parent / "bundle.json"
    bundle_raw = (compact_canonical_json(bundle) + "\n").encode()
    bundle_path.write_bytes(bundle_raw)
    (candidate / "temporal_retrospective.json").write_bytes(
        (candidate / "temporal_retrospective.json").read_bytes() + b" "
    )
    review = bind_implementation_review_sandbox_context(
        expected_check_name="focused_pytest", expected_command_sha256="0" * 64
    )

    with pytest.raises(ValueError, match="member receipts"):
        load_verified_candidate_receipt_bundle(
            execution_context=review,
            bundle_path=bundle_path,
            candidate_dir=candidate,
            expected_artifact_sha256=bundle["artifact_sha256"],
            expected_file_sha256=hashlib.sha256(bundle_raw).hexdigest(),
        )


def test_run_admission_loader_replays_physical_receipt_spine(tmp_path):
    fixture = _admission_fixture(tmp_path)
    loaded = fixture["admission"]
    assert loaded.admission.payload["admission_state"] == "admitted"
    assert loaded.completion.payload["admission_identity"]["internal_sha256"] == loaded.admission.artifact_sha256
    with pytest.raises(PermissionError):
        load_verified_run_admission(
            execution_context=object(),
            claim_path=loaded.claim.path,
            admission_path=loaded.admission.path,
            completion_path=loaded.completion.path,
            expected_authorization_sha256=fixture["auth"],
            expected_claim_artifact_sha256=loaded.claim.artifact_sha256,
            expected_claim_file_sha256=loaded.claim.file_sha256,
            expected_admission_artifact_sha256=loaded.admission.artifact_sha256,
            expected_admission_file_sha256=loaded.admission.file_sha256,
            expected_completion_artifact_sha256=loaded.completion.artifact_sha256,
            expected_completion_file_sha256=loaded.completion.file_sha256,
        )


def test_terminal_loader_replays_result_and_rejects_candidate_drift(tmp_path):
    fixture = _candidate_bundle_fixture(_admission_fixture(tmp_path))
    result_dir = seal_verified_result(fixture["root"], fixture["result"], flock_path=fixture["lock"])
    loaded = load_verified_terminal_artifact(
        execution_context=fixture["review"],
        terminal_kind="verified_result",
        output_root=fixture["root"],
        candidate_dir=fixture["candidate"],
        candidate_bundle_path=fixture["bundle_path"],
        expected_candidate_bundle_artifact_sha256=fixture["bundle"]["artifact_sha256"],
        expected_candidate_bundle_file_sha256=fixture["bundle_file_sha"],
        run_admission=fixture["admission"],
        run_history=fixture["history"],
        terminal_path=result_dir / "verification_registry.json",
        expected_terminal_artifact_sha256=fixture["result"]["artifact_sha256"],
        expected_terminal_file_sha256=hashlib.sha256(
            (result_dir / "verification_registry.json").read_bytes()
        ).hexdigest(),
    )
    assert loaded.payload["decision"] == "mechanical_pass"
    mutated = fixture["candidate"] / "temporal_retrospective.json"
    mutated.write_bytes(mutated.read_bytes() + b" ")
    with pytest.raises(ValueError, match="member receipts"):
        load_verified_terminal_artifact(
            execution_context=fixture["review"],
            terminal_kind="verified_result",
            output_root=fixture["root"],
            candidate_dir=fixture["candidate"],
            candidate_bundle_path=fixture["bundle_path"],
            expected_candidate_bundle_artifact_sha256=fixture["bundle"]["artifact_sha256"],
            expected_candidate_bundle_file_sha256=fixture["bundle_file_sha"],
            run_admission=fixture["admission"],
            run_history=fixture["history"],
            terminal_path=result_dir / "verification_registry.json",
            expected_terminal_artifact_sha256=fixture["result"]["artifact_sha256"],
            expected_terminal_file_sha256=hashlib.sha256(
                (result_dir / "verification_registry.json").read_bytes()
            ).hexdigest(),
        )


def test_terminal_loader_replays_postverification_failure(tmp_path):
    case = tmp_path / "failure-case"
    case.mkdir()
    fixture = _candidate_bundle_fixture(_admission_fixture(case))
    history_contract = {
        "run_identity_receipt": _payload_receipt(fixture["history"].payloads[1]),
        "head_receipt": _payload_receipt(fixture["history"].payloads[-1]),
        "marker_count": len(fixture["history"].payloads) - 2,
    }
    admission = fixture["admission"].admission
    bundle_receipt = {
        "artifact_sha256": fixture["bundle"]["artifact_sha256"],
        "file_sha256": fixture["bundle_file_sha"],
    }
    slots = [
        {"provider": "parent_spec_approval", "verification_state": "verified", "receipt": _receipt()},
        {"provider": "amendment_implementation_approval", "verification_state": "verified", "receipt": _receipt()},
        {"provider": "amended_implementation_review", "verification_state": "verified", "receipt": _receipt()},
        {
            "provider": "exact_v2_rerun_authorization",
            "verification_state": "verified",
            "receipt": admission.payload["authorization_receipt"],
        },
        {"provider": "run_history_ledger", "verification_state": "verified", "receipt": history_contract},
        {
            "provider": "run_admission",
            "verification_state": "verified",
            "receipt": _payload_receipt(admission.payload),
        },
        {
            "provider": "static_inputs",
            "verification_state": "verified",
            "receipt": admission.payload["static_input_contract"],
        },
        {"provider": "candidate_receipt_bundle", "verification_state": "verified", "receipt": bundle_receipt},
    ]
    failure = build_postpublication_failure_payload(
        failed_check_name="retrospective",
        dependency_provider_slots=slots,
    )
    failure_dir = seal_postverification_failure(fixture["root"], failure, flock_path=fixture["lock"])
    loaded = load_verified_terminal_artifact(
        execution_context=fixture["review"],
        terminal_kind="postverification_failure",
        output_root=fixture["root"],
        candidate_dir=fixture["candidate"],
        candidate_bundle_path=fixture["bundle_path"],
        expected_candidate_bundle_artifact_sha256=fixture["bundle"]["artifact_sha256"],
        expected_candidate_bundle_file_sha256=fixture["bundle_file_sha"],
        run_admission=fixture["admission"],
        run_history=fixture["history"],
        terminal_path=failure_dir / "failure.json",
        expected_terminal_artifact_sha256=failure["artifact_sha256"],
        expected_terminal_file_sha256=hashlib.sha256((failure_dir / "failure.json").read_bytes()).hexdigest(),
    )
    assert loaded.payload["failed_check_name"] == "retrospective"
