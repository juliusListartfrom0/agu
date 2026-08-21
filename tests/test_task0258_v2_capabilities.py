"""Adversarial tests for TASK-0258 v2 review-only contexts/loaders."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest

from app.analysis import vru_causal_temporal_retrospective as temporal_module
from app.analysis.task0258_module_a_v2 import (
    VERIFICATION_ATTEMPT_SCHEMA_V2,
    canonical_artifact_sha256,
    compact_canonical_json,
)
from app.analysis.task0258_run_history import ADMISSION_SCHEMA, CLAIM_SCHEMA, COMPLETION_SCHEMA, MODULE_ID
from app.analysis.task0258_v2_artifacts import CANDIDATE_MEMBER_PATHS
from app.analysis.task0258_v2_capabilities import (
    VerifiedImplementationReviewSandboxContext,
    VerifiedModuleAStaticInputs,
    VerifiedReviewAmendedImplementationReview,
    VerifiedReviewDiscoveryContext,
    VerifiedReviewImplementationApproval,
    VerifiedReviewNoWritePreflight,
    VerifiedReviewParentModuleASpecApproval,
    VerifiedReviewReadIsolationBinding,
    VerifiedReviewRerunAuthorization,
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
    load_verified_amended_implementation_review,
    load_verified_amendment_implementation_approval,
    load_verified_candidate_receipt_bundle,
    load_verified_module_a_static_inputs,
    load_verified_parent_module_a_spec_approval,
    load_verified_read_isolation_binding,
    load_verified_review_rerun_authorization,
    load_verified_run_admission,
    load_verified_run_history_ledger,
    load_verified_terminal_artifact,
    load_verified_verification_attempt,
    replay_module_a_read_traversal_for_discovery,
    replay_verified_module_a_static_inputs,
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
from app.analysis.vru_causal_temporal_retrospective import (
    FileReceipt,
    StoredArtifactReceipt,
    Task0257ExpectedReceipts,
    Task0257InputPaths,
)


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
        VerifiedReviewAmendedImplementationReview()
    with pytest.raises(TypeError):
        VerifiedReviewRerunAuthorization()
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


def test_amendment_implementation_loader_replays_all_review_receipts(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    approval_dir = repository_root / "analysis_outputs/public_research/task0258_module_a_amendment_approval"
    parent_source = (
        repository_root
        / "analysis_outputs/public_research/task0258_module_a_spec_registry_v2/module_a_spec_approval.json"
    )
    parent_payload = json.loads(parent_source.read_text(encoding="utf-8"))
    parent_raw = (compact_canonical_json(parent_payload) + "\n").encode()
    parent_path = tmp_path / "parent-spec-approval.json"
    parent_path.write_bytes(parent_raw)
    approved_specs = {
        name: repository_root / "docs/specs/TASK-0258-temporal-canary" / name
        for name in ("requirement.md", "solution.md", "gate-review.md")
    }
    parent = load_verified_parent_module_a_spec_approval(
        execution_context=bind_implementation_review_sandbox_context(
            expected_check_name="focused_pytest", expected_command_sha256="0" * 64
        ),
        approval_path=parent_path,
        expected_artifact_sha256=parent_payload["artifact_sha256"],
        expected_file_sha256=hashlib.sha256(parent_raw).hexdigest(),
        approved_spec_paths=approved_specs,
        expected_fresh_review_internal_sha256=parent_payload["fresh_review_receipt"]["internal_sha256"],
        expected_fresh_review_file_sha256=parent_payload["fresh_review_receipt"]["file_sha256"],
        expected_approval_statement_sha256=parent_payload["approval_statement_sha256"],
    )
    assert isinstance(parent, VerifiedReviewParentModuleASpecApproval)

    baseline_payload = {
        "schema_version": "agu.task0258-module-a-implementation-scope-baseline.v1",
        "module_id": "existing-45-temporal-retrospective",
        "repository_root_absolute_path": str(repository_root),
        "repository_root_device": repository_root.stat().st_dev,
        "repository_root_inode": repository_root.stat().st_ino,
        "ordered_root_paths": ["."],
        "check_output_directory_absolute_path": str(tmp_path),
        "check_output_directory_device": tmp_path.stat().st_dev,
        "check_output_directory_inode": tmp_path.stat().st_ino,
        "ordered_entry_receipts": [
            {
                "path": ".",
                "entry_kind": "directory",
                "mode_bits": 0o700,
                "link_count": None,
                "size_bytes": None,
                "file_sha256": None,
                "symlink_target_text": None,
                "hardlink_group_sha256": None,
                "ordered_child_names": [],
            }
        ],
        "ordered_repository_executable_receipts": [],
        "captured_at_utc": "2026-08-22T00:00:00Z",
    }
    baseline_payload["artifact_sha256"] = canonical_artifact_sha256(baseline_payload)
    baseline_raw = (compact_canonical_json(baseline_payload) + "\n").encode()
    baseline_path = tmp_path / "implementation-scope-baseline.json"
    baseline_path.write_bytes(baseline_raw)

    review_path = approval_dir / "amendment_fresh_review.md"
    amendment_path = repository_root / "docs/specs/TASK-0258-temporal-canary/amendment-001-postpublication-proof.md"
    approval_payload = json.loads((approval_dir / "amendment_implementation_approval.json").read_text(encoding="utf-8"))
    approval_payload["repository_root_device"] = repository_root.stat().st_dev
    approval_payload["repository_root_inode"] = repository_root.stat().st_ino
    approval_payload["parent_spec_approval_receipt"] = {
        "artifact_sha256": parent_payload["artifact_sha256"],
        "file_sha256": hashlib.sha256(parent_raw).hexdigest(),
    }
    approval_payload["amendment_fresh_review_receipt"] = {
        "artifact_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
        "file_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
    }
    approval_payload["implementation_scope_baseline_receipt"] = {
        "artifact_sha256": baseline_payload["artifact_sha256"],
        "file_sha256": hashlib.sha256(baseline_raw).hexdigest(),
    }
    approval_payload["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in approval_payload.items() if key != "artifact_sha256"}
    )
    approval_raw = (compact_canonical_json(approval_payload) + "\n").encode()
    approval_path = tmp_path / "amendment-implementation-approval.json"
    approval_path.write_bytes(approval_raw)

    loaded = load_verified_amendment_implementation_approval(
        execution_context=bind_implementation_review_sandbox_context(
            expected_check_name="focused_pytest", expected_command_sha256="0" * 64
        ),
        repository_root=repository_root,
        approval_path=approval_path,
        expected_artifact_sha256=approval_payload["artifact_sha256"],
        expected_file_sha256=hashlib.sha256(approval_raw).hexdigest(),
        parent_approval=parent,
        amendment_path=amendment_path,
        expected_amendment_file_sha256=approval_payload["approved_amendment_file_receipt"]["file_sha256"],
        amendment_review_path=review_path,
        expected_amendment_review_artifact_sha256=approval_payload["amendment_fresh_review_receipt"]["artifact_sha256"],
        expected_amendment_review_file_sha256=approval_payload["amendment_fresh_review_receipt"]["file_sha256"],
        implementation_scope_baseline_path=baseline_path,
        expected_implementation_scope_baseline_artifact_sha256=baseline_payload["artifact_sha256"],
        expected_implementation_scope_baseline_file_sha256=hashlib.sha256(baseline_raw).hexdigest(),
    )

    assert isinstance(loaded, VerifiedReviewImplementationApproval)
    assert loaded.production_capability is False
    assert loaded.parent_approval is parent
    assert loaded.repository_root_identity == {
        "device": repository_root.stat().st_dev,
        "inode": repository_root.stat().st_ino,
    }


def test_amended_implementation_review_loader_replays_fresh_review_and_checks(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    approval_dir = repository_root / "analysis_outputs/public_research/task0258_module_a_amendment_approval"
    parent_source = (
        repository_root
        / "analysis_outputs/public_research/task0258_module_a_spec_registry_v2/module_a_spec_approval.json"
    )
    parent_payload = json.loads(parent_source.read_text(encoding="utf-8"))
    parent_raw = (compact_canonical_json(parent_payload) + "\n").encode()
    parent_path = tmp_path / "parent-spec-approval.json"
    parent_path.write_bytes(parent_raw)
    approved_specs = {
        name: repository_root / "docs/specs/TASK-0258-temporal-canary" / name
        for name in ("requirement.md", "solution.md", "gate-review.md")
    }
    review_context = bind_implementation_review_sandbox_context(
        expected_check_name="focused_pytest", expected_command_sha256="0" * 64
    )
    parent = load_verified_parent_module_a_spec_approval(
        execution_context=review_context,
        approval_path=parent_path,
        expected_artifact_sha256=parent_payload["artifact_sha256"],
        expected_file_sha256=hashlib.sha256(parent_raw).hexdigest(),
        approved_spec_paths=approved_specs,
        expected_fresh_review_internal_sha256=parent_payload["fresh_review_receipt"]["internal_sha256"],
        expected_fresh_review_file_sha256=parent_payload["fresh_review_receipt"]["file_sha256"],
        expected_approval_statement_sha256=parent_payload["approval_statement_sha256"],
    )

    baseline_scope_paths = sorted(
        (
            "app/analysis/vru_causal_temporal_retrospective.py",
            "scripts/extract_vru_causal_tiled_swin_embeddings.py",
            "scripts/screen_vru_causal_temporal_retrospective.py",
            "scripts/seal_vru_causal_temporal_feature_plan.py",
            "scripts/task0258_module_a_verified_bootstrap.py",
            "tests/test_task0258_module_a_cli.py",
            "tests/test_vru_causal_final_evaluator.py",
            "tests/test_vru_causal_temporal_feature_plan.py",
            "tests/test_vru_causal_temporal_retrospective.py",
            "tests/test_vru_causal_tiled_swin_embeddings.py",
        )
    )
    baseline_entries = [
        {
            "path": ".",
            "entry_kind": "directory",
            "mode_bits": 0o700,
            "link_count": None,
            "size_bytes": None,
            "file_sha256": None,
            "symlink_target_text": None,
            "hardlink_group_sha256": None,
            "ordered_child_names": [],
        },
        {
            "path": "scripts",
            "entry_kind": "directory",
            "mode_bits": 0o700,
            "link_count": None,
            "size_bytes": None,
            "file_sha256": None,
            "symlink_target_text": None,
            "hardlink_group_sha256": None,
            "ordered_child_names": ["task0258_module_a_verified_bootstrap.py"],
        },
    ]
    for path_text in baseline_scope_paths:
        path_stat = (repository_root / path_text).stat()
        baseline_entries.append(
            {
                "path": path_text,
                "entry_kind": "regular",
                "mode_bits": path_stat.st_mode,
                "link_count": 1,
                "size_bytes": path_stat.st_size,
                "file_sha256": hashlib.sha256((repository_root / path_text).read_bytes()).hexdigest(),
                "symlink_target_text": None,
                "hardlink_group_sha256": hashlib.sha256(compact_canonical_json([path_text]).encode()).hexdigest(),
                "ordered_child_names": None,
            }
        )
    baseline_entries = [baseline_entries[0], *sorted(baseline_entries[1:], key=lambda entry: entry["path"])]
    baseline_payload = {
        "schema_version": "agu.task0258-module-a-implementation-scope-baseline.v1",
        "module_id": "existing-45-temporal-retrospective",
        "repository_root_absolute_path": str(repository_root),
        "repository_root_device": repository_root.stat().st_dev,
        "repository_root_inode": repository_root.stat().st_ino,
        "ordered_root_paths": ["."],
        "check_output_directory_absolute_path": str(tmp_path),
        "check_output_directory_device": tmp_path.stat().st_dev,
        "check_output_directory_inode": tmp_path.stat().st_ino,
        "ordered_entry_receipts": baseline_entries,
        "ordered_repository_executable_receipts": [],
        "captured_at_utc": "2026-08-22T00:00:00Z",
    }
    baseline_payload["artifact_sha256"] = canonical_artifact_sha256(baseline_payload)
    baseline_raw = (compact_canonical_json(baseline_payload) + "\n").encode()
    baseline_path = tmp_path / "implementation-scope-baseline.json"
    baseline_path.write_bytes(baseline_raw)

    amendment_path = repository_root / "docs/specs/TASK-0258-temporal-canary/amendment-001-postpublication-proof.md"
    amendment_review_path = approval_dir / "amendment_fresh_review.md"
    approval_payload = json.loads((approval_dir / "amendment_implementation_approval.json").read_text())
    approval_payload["repository_root_device"] = repository_root.stat().st_dev
    approval_payload["repository_root_inode"] = repository_root.stat().st_ino
    approval_payload["parent_spec_approval_receipt"] = {
        "artifact_sha256": parent_payload["artifact_sha256"],
        "file_sha256": hashlib.sha256(parent_raw).hexdigest(),
    }
    review_bytes = amendment_review_path.read_bytes()
    approval_payload["amendment_fresh_review_receipt"] = {
        "artifact_sha256": hashlib.sha256(review_bytes).hexdigest(),
        "file_sha256": hashlib.sha256(review_bytes).hexdigest(),
    }
    approval_payload["implementation_scope_baseline_receipt"] = {
        "artifact_sha256": baseline_payload["artifact_sha256"],
        "file_sha256": hashlib.sha256(baseline_raw).hexdigest(),
    }
    approval_payload["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in approval_payload.items() if key != "artifact_sha256"}
    )
    approval_raw = (compact_canonical_json(approval_payload) + "\n").encode()
    approval_path = tmp_path / "amendment-implementation-approval.json"
    approval_path.write_bytes(approval_raw)
    implementation_approval = load_verified_amendment_implementation_approval(
        execution_context=review_context,
        repository_root=repository_root,
        approval_path=approval_path,
        expected_artifact_sha256=approval_payload["artifact_sha256"],
        expected_file_sha256=hashlib.sha256(approval_raw).hexdigest(),
        parent_approval=parent,
        amendment_path=amendment_path,
        expected_amendment_file_sha256=approval_payload["approved_amendment_file_receipt"]["file_sha256"],
        amendment_review_path=amendment_review_path,
        expected_amendment_review_artifact_sha256=approval_payload["amendment_fresh_review_receipt"]["artifact_sha256"],
        expected_amendment_review_file_sha256=approval_payload["amendment_fresh_review_receipt"]["file_sha256"],
        implementation_scope_baseline_path=baseline_path,
        expected_implementation_scope_baseline_artifact_sha256=baseline_payload["artifact_sha256"],
        expected_implementation_scope_baseline_file_sha256=hashlib.sha256(baseline_raw).hexdigest(),
    )

    def reviewed(path_text):
        raw = (repository_root / path_text).read_bytes()
        return {"path": path_text, "size_bytes": len(raw), "file_sha256": hashlib.sha256(raw).hexdigest()}

    code_paths = (
        "app/analysis/vru_causal_temporal_retrospective.py",
        "scripts/extract_vru_causal_tiled_swin_embeddings.py",
        "scripts/screen_vru_causal_temporal_retrospective.py",
        "scripts/seal_vru_causal_temporal_feature_plan.py",
        "scripts/task0258_module_a_verified_bootstrap.py",
    )
    test_paths = (
        "tests/test_task0258_module_a_cli.py",
        "tests/test_vru_causal_final_evaluator.py",
        "tests/test_vru_causal_temporal_feature_plan.py",
        "tests/test_vru_causal_temporal_retrospective.py",
        "tests/test_vru_causal_tiled_swin_embeddings.py",
    )
    code_rows = [reviewed(path) for path in code_paths]
    test_rows = [reviewed(path) for path in test_paths]
    runtime_rows = [reviewed(path) for path in code_paths]
    config_rows = [reviewed(path) for path in ("pyproject.toml", "pytest.ini")]
    baseline_by_path = {entry["path"]: entry for entry in baseline_entries}
    delta = {
        "ordered_leaf_rows": [
            {
                "path": row["path"],
                "before_entry": baseline_by_path[row["path"]],
                "after_file": row,
                "change_kind": "created"
                if row["path"].endswith("task0258_module_a_verified_bootstrap.py")
                else "modified",
            }
            for row in sorted(code_rows + test_rows, key=lambda item: item["path"])
        ],
        "ordered_directory_rows": [
            {
                "path": "scripts",
                "before_child_names": [],
                "after_child_names": ["task0258_module_a_verified_bootstrap.py"],
                "authorized_added_children": ["task0258_module_a_verified_bootstrap.py"],
                "authorized_removed_children": [],
            }
        ],
    }
    bootstrap = {
        "protocol": "task0258-module-a-verified-python-bootstrap-v2",
        "source_sha256": code_rows[-1]["file_sha256"],
        "runtime_snapshot_receipt": {
            "contract_absolute_path": str(tmp_path / "runtime-contract.json"),
            "artifact_sha256": "4" * 64,
            "file_sha256": "5" * 64,
            "postpublication_free_bytes": 0,
        },
    }
    checks = ("focused_pytest", "full_pytest", "ruff_check", "ruff_format_check", "diff_check")
    query_path = code_paths[0]
    query_file = repository_root / query_path
    query_stat = query_file.stat()
    query_stat_result = {
        "entry_kind": "regular",
        "mode": query_stat.st_mode,
        "device": query_stat.st_dev,
        "inode": query_stat.st_ino,
        "nlink": query_stat.st_nlink,
        "uid": query_stat.st_uid,
        "gid": query_stat.st_gid,
        "rdev": query_stat.st_rdev,
        "size_bytes": query_stat.st_size,
        "atime_ns": query_stat.st_atime_ns,
        "mtime_ns": query_stat.st_mtime_ns,
        "ctime_ns": query_stat.st_ctime_ns,
        "birthtime_ns": getattr(query_stat, "st_birthtime_ns", 0),
        "flags": getattr(query_stat, "st_flags", 0),
    }
    query = {
        "operation": "open",
        "path": query_path,
        "arguments": {"flags": 0, "creation_mode": None},
        "follow_policy": "follow",
        "result_kind": "success",
        "errno": None,
        "stat_result": query_stat_result,
        "access_result": None,
        "readlink_target_text": None,
        "ordered_directory_entries": None,
        "xattr_result": None,
        "file_content": {"size_bytes": query_stat.st_size, "file_sha256": code_rows[0]["file_sha256"]},
    }
    query_projection = hashlib.sha256(compact_canonical_json([query]).encode()).hexdigest()
    check_inputs = [
        {"check_name": check, "ordered_query_receipts": [query], "projection_sha256": query_projection}
        for check in checks
    ]
    check_rows = []
    for check in checks:
        output_path = tmp_path / f"{check}.out"
        output_raw = f"{check}: ok\n".encode()
        output_path.write_bytes(output_raw)
        check_rows.append(
            {
                "check_name": check,
                "execution_protocol": "task0258-review-snapshot-fd-v2",
                "sandbox_attestation_sha256": "6" * 64,
                "command_sha256": "7" * 64,
                "exit_code": 0,
                "output_path": str(output_path),
                "output_size_bytes": len(output_raw),
                "output_artifact_sha256": None,
                "output_file_sha256": hashlib.sha256(output_raw).hexdigest(),
                "completed_at_utc": "2026-08-22T00:00:00Z",
            }
        )
    governance = {
        "review_scope": "external_governance_only",
        "model_execution_scope": "review_reasoning_only",
        "local_module_a_worker_or_media_execution_performed": False,
    }
    fresh_payload = {
        "schema_version": "agu.task0258-module-a-fresh-code-review.v1",
        "module_id": "existing-45-temporal-retrospective",
        "repository_root_absolute_path": str(repository_root),
        "repository_root_device": repository_root.stat().st_dev,
        "repository_root_inode": repository_root.stat().st_ino,
        "amendment_implementation_approval_receipt": {
            "artifact_sha256": implementation_approval.artifact.artifact_sha256,
            "file_sha256": implementation_approval.artifact.file_sha256,
        },
        "implementation_context_id": "implementation-context",
        "reviewer_context_id": "fresh-review-context",
        "reviewer_independence": "different_fresh_context",
        "ordered_code_file_receipts": code_rows,
        "ordered_test_file_receipts": test_rows,
        "ordered_runtime_dependency_file_receipts": runtime_rows,
        "ordered_check_configuration_file_receipts": config_rows,
        "ordered_check_input_receipt_sets": check_inputs,
        "implementation_scope_baseline_receipt": approval_payload["implementation_scope_baseline_receipt"],
        "implementation_scope_delta": delta,
        "bootstrap_launcher_receipt": bootstrap,
        "ordered_pre_review_check_receipts": check_rows,
        "fresh_review_governance_observation": governance,
        "critical_count": 0,
        "required_count": 0,
        "optional_count": 0,
        "heavy_execution_performed": False,
        "reviewed_at_utc": "2026-08-22T00:00:01Z",
    }
    fresh_payload["artifact_sha256"] = canonical_artifact_sha256(fresh_payload)
    fresh_raw = (compact_canonical_json(fresh_payload) + "\n").encode()
    fresh_path = tmp_path / "fresh_context_code_review.out"
    fresh_path.write_bytes(fresh_raw)
    fresh_file_sha = hashlib.sha256(fresh_raw).hexdigest()
    check_rows.append(
        {
            "check_name": "fresh_context_code_review",
            "execution_protocol": "fresh-context-read-only-code-review-v1",
            "sandbox_attestation_sha256": None,
            "command_sha256": "8" * 64,
            "exit_code": 0,
            "output_path": str(fresh_path),
            "output_size_bytes": len(fresh_raw),
            "output_artifact_sha256": fresh_payload["artifact_sha256"],
            "output_file_sha256": fresh_file_sha,
            "completed_at_utc": "2026-08-22T00:00:02Z",
        }
    )
    review_payload = {
        **{
            key: value
            for key, value in fresh_payload.items()
            if key not in {"ordered_pre_review_check_receipts", "fresh_review_governance_observation"}
        },
        "ordered_check_receipts": check_rows,
        "review_resource_summary": {
            "runtime_build_observation": {
                "build_resource_limits": {},
                "build_resource_observation": {},
                "postpublication_free_bytes": 0,
            },
            "ordered_check_observations": [
                {"check_name": check, "process_resource_observation": {}, "output_publication_observation": {}}
                for check in checks
            ],
            "fresh_review_observation": {
                "governance_observation": governance,
                "fresh_review_artifact_bytes": len(fresh_raw),
            },
        },
    }
    review_payload["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in review_payload.items() if key != "artifact_sha256"}
    )
    review_raw = (compact_canonical_json(review_payload) + "\n").encode()
    implementation_review_path = tmp_path / "amended-implementation-review.json"
    implementation_review_path.write_bytes(review_raw)
    loaded = load_verified_amended_implementation_review(
        execution_context=review_context,
        repository_root=repository_root,
        implementation_approval=implementation_approval,
        implementation_review_path=implementation_review_path,
        expected_implementation_review_artifact_sha256=review_payload["artifact_sha256"],
        expected_implementation_review_file_sha256=hashlib.sha256(review_raw).hexdigest(),
        fresh_review_path=fresh_path,
        expected_fresh_review_artifact_sha256=fresh_payload["artifact_sha256"],
        expected_fresh_review_file_sha256=fresh_file_sha,
    )
    assert isinstance(loaded, VerifiedReviewAmendedImplementationReview)
    assert loaded.production_capability is False
    assert loaded.implementation_approval is implementation_approval
    assert loaded.fresh_review.payload["critical_count"] == 0

    registry_directory = tmp_path / "consumption-registry"
    registry_directory.mkdir()
    candidate_bundle_parent = tmp_path / "candidate-bundle-parent"
    candidate_bundle_parent.mkdir()
    candidate_bundle_path = candidate_bundle_parent / "candidate-receipt-bundle.json"
    output_root = tmp_path / "vru_causal_temporal_retrospective_v2"
    authorization_payload = {
        "schema_version": "agu.task0258-module-a-v2-rerun-authorization.v1",
        "module_id": "existing-45-temporal-retrospective",
        "repository_root_absolute_path": str(repository_root),
        "repository_root_device": repository_root.stat().st_dev,
        "repository_root_inode": repository_root.stat().st_ino,
        "amendment_implementation_approval_receipt": {
            "artifact_sha256": implementation_approval.artifact.artifact_sha256,
            "file_sha256": implementation_approval.artifact.file_sha256,
        },
        "implementation_review_receipt": {
            "artifact_sha256": loaded.artifact.artifact_sha256,
            "file_sha256": loaded.artifact.file_sha256,
        },
        "approved_amendment_file_receipt": approval_payload["approved_amendment_file_receipt"],
        "approved_code_receipts": review_payload["ordered_code_file_receipts"],
        "approved_test_receipts": review_payload["ordered_test_file_receipts"],
        "approved_runtime_dependency_receipts": review_payload["ordered_runtime_dependency_file_receipts"],
        "approved_check_configuration_receipts": review_payload["ordered_check_configuration_file_receipts"],
        "approved_check_input_receipt_sets": review_payload["ordered_check_input_receipt_sets"],
        "approved_check_receipts": review_payload["ordered_check_receipts"],
        "approved_implementation_scope_baseline_receipt": review_payload["implementation_scope_baseline_receipt"],
        "approved_implementation_scope_delta": review_payload["implementation_scope_delta"],
        "approved_bootstrap_launcher_receipt": review_payload["bootstrap_launcher_receipt"],
        "approved_static_input_contract": {
            "temporal_plan_artifact_sha256": "a" * 64,
            "temporal_plan_file_sha256": "b" * 64,
            "task0257_receipts_projection_sha256": "c" * 64,
        },
        "allowed_operations": [
            "preflight_and_publish_admission",
            "recover_admission_completion",
            "publish_candidate",
            "seal_candidate_receipt_bundle",
            "postpublication_verify",
            "load_existing_terminal",
        ],
        "output_root_absolute_path": str(output_root),
        "candidate_receipt_bundle_absolute_path": str(candidate_bundle_path),
        "run_id": "local-review-rerun",
        "authorization_scope": "one_module_a_v2_rerun",
        "maximum_run_count": 1,
        "run_admission_relative_path": "run_admission.json",
        "run_consumption_registry_directory_absolute_path": str(registry_directory),
        "run_consumption_registry_directory_identity": {
            "device": registry_directory.stat().st_dev,
            "inode": registry_directory.stat().st_ino,
        },
        "module_b_authorized": False,
        "approval_statement_sha256": "d" * 64,
        "approved_at_utc": "2026-08-22T00:00:03Z",
    }
    authorization_payload["artifact_sha256"] = canonical_artifact_sha256(authorization_payload)
    authorization_raw = (compact_canonical_json(authorization_payload) + "\n").encode()
    authorization_path = tmp_path / "rerun-authorization.json"
    authorization_path.write_bytes(authorization_raw)
    loaded_authorization = load_verified_review_rerun_authorization(
        execution_context=review_context,
        repository_root=repository_root,
        authorization_path=authorization_path,
        expected_artifact_sha256=authorization_payload["artifact_sha256"],
        expected_file_sha256=hashlib.sha256(authorization_raw).hexdigest(),
        implementation_approval=implementation_approval,
        implementation_review=loaded,
        expected_static_input_contract=authorization_payload["approved_static_input_contract"],
        output_root=output_root,
        candidate_bundle_path=candidate_bundle_path,
    )
    assert isinstance(loaded_authorization, VerifiedReviewRerunAuthorization)
    assert loaded_authorization.production_capability is False
    assert loaded_authorization.implementation_review is loaded

    drifted_authorization = dict(authorization_payload)
    drifted_authorization["module_b_authorized"] = True
    drifted_authorization["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in drifted_authorization.items() if key != "artifact_sha256"}
    )
    drifted_raw = (compact_canonical_json(drifted_authorization) + "\n").encode()
    authorization_path.write_bytes(drifted_raw)
    with pytest.raises(ValueError, match="Module B"):
        load_verified_review_rerun_authorization(
            execution_context=review_context,
            repository_root=repository_root,
            authorization_path=authorization_path,
            expected_artifact_sha256=drifted_authorization["artifact_sha256"],
            expected_file_sha256=hashlib.sha256(drifted_raw).hexdigest(),
            implementation_approval=implementation_approval,
            implementation_review=loaded,
            expected_static_input_contract=authorization_payload["approved_static_input_contract"],
            output_root=output_root,
            candidate_bundle_path=candidate_bundle_path,
        )


def test_static_input_loader_replays_complete_parent_graph_as_review_only(tmp_path, monkeypatch):
    repository_root = Path(__file__).resolve().parents[1]
    contract_path = (
        repository_root
        / "analysis_outputs/public_research/task0258_module_a_spec_registry_v1/task0257_input_contract.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    path_values = {
        field: tuple(Path(item) for item in value) if isinstance(value, list) else Path(value)
        for field, value in contract["paths"].items()
    }
    paths = Task0257InputPaths(**path_values)

    stored_sequences = {"parent_candidate_children", "old_embedding_files"}
    file_sequences = {
        "parent_label_children",
        "parent_review_jpegs",
        "harwood_review_jpegs",
        "source_videos",
        "checkpoints",
    }
    file_scalars = {"parent_source_manifest", "v2_label_child", "harwood_source_manifest"}
    receipt_values = {}
    for field, value in contract["expected_receipts"].items():
        if field in stored_sequences:
            receipt_values[field] = tuple(StoredArtifactReceipt(**row) for row in value)
        elif field in file_sequences:
            receipt_values[field] = tuple(FileReceipt(**row) for row in value)
        elif field in file_scalars:
            receipt_values[field] = FileReceipt(**value)
        else:
            receipt_values[field] = StoredArtifactReceipt(**value)
    receipts = Task0257ExpectedReceipts(**receipt_values)

    plan_path = tmp_path / "temporal-plan.json"
    plan_path.write_bytes(b"review-only-plan\n")
    plan = temporal_module.VerifiedTemporalFeaturePlan(temporal_module._VERIFIED_PLAN_TOKEN)
    plan._payload = {"task0257_receipts": json.loads(json.dumps(asdict(receipts)))}
    plan._path = plan_path
    monkeypatch.setattr(
        "app.analysis.task0258_v2_capabilities.load_verified_temporal_feature_plan",
        lambda **_kwargs: plan,
    )
    calls = []
    monkeypatch.setattr(
        "app.analysis.task0258_v2_capabilities.verify_task0257_temporal_inputs",
        lambda **kwargs: calls.append(kwargs) or object(),
    )

    projection = (compact_canonical_json(asdict(receipts)) + "\n").encode("utf-8")
    capability = load_verified_module_a_static_inputs(
        execution_context=bind_implementation_review_sandbox_context(
            expected_check_name="focused_pytest", expected_command_sha256="0" * 64
        ),
        temporal_plan_path=plan_path,
        expected_temporal_plan_artifact_sha256="1" * 64,
        expected_temporal_plan_file_sha256="2" * 64,
        task0257_input_paths=paths,
        expected_task0257_receipts=receipts,
        expected_task0257_receipts_projection_sha256=hashlib.sha256(projection).hexdigest(),
    )

    assert isinstance(capability, VerifiedModuleAStaticInputs)
    assert capability.production_capability is False
    assert dict(capability.static_input_contract) == {
        "temporal_plan_artifact_sha256": "1" * 64,
        "temporal_plan_file_sha256": "2" * 64,
        "task0257_receipts_projection_sha256": hashlib.sha256(projection).hexdigest(),
    }
    assert capability.task0257_receipts_snapshot == projection
    replay_verified_module_a_static_inputs(capability)
    assert calls == [
        {"paths": paths, "expected_receipts": receipts},
        {"paths": paths, "expected_receipts": receipts},
    ]
    capability._task0257_receipts_snapshot = b"drift"
    with pytest.raises(ValueError, match="mutated"):
        replay_verified_module_a_static_inputs(capability)
    with pytest.raises(TypeError):
        VerifiedModuleAStaticInputs()

    with pytest.raises(PermissionError, match="review context"):
        load_verified_module_a_static_inputs(
            execution_context=object(),
            temporal_plan_path=plan_path,
            expected_temporal_plan_artifact_sha256="1" * 64,
            expected_temporal_plan_file_sha256="2" * 64,
            task0257_input_paths=paths,
            expected_task0257_receipts=receipts,
            expected_task0257_receipts_projection_sha256=hashlib.sha256(projection).hexdigest(),
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
