"""Adversarial tests for TASK-0258 v2 review-only contexts/loaders."""

from __future__ import annotations

import hashlib

import pytest

from app.analysis.task0258_module_a_v2 import canonical_artifact_sha256, compact_canonical_json
from app.analysis.task0258_v2_artifacts import CANDIDATE_MEMBER_PATHS
from app.analysis.task0258_v2_capabilities import (
    VerifiedImplementationReviewSandboxContext,
    VerifiedReviewDiscoveryContext,
    bind_implementation_review_discovery_context,
    bind_implementation_review_sandbox_context,
    bind_synthetic_module_a_discovery_context,
    bind_synthetic_module_a_transaction_context,
    exercise_module_a_v2_state_machine_for_discovery,
    exercise_module_a_v2_state_machine_for_review,
    load_verified_candidate_receipt_bundle,
    replay_module_a_read_traversal_for_discovery,
)
from app.analysis.task0258_v2_pipeline import (
    build_candidate_gate_payload,
    build_candidate_receipt_bundle_payload,
    seal_candidate_v2,
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


def _members():
    members = {}
    for rel in CANDIDATE_MEMBER_PATHS:
        if rel.endswith(".jsonl"):
            members[rel] = b'{"role":"test"}\n'
        elif rel == "candidate_gate.json":
            members[rel] = (compact_canonical_json(_gate()) + "\n").encode()
        else:
            payload = {"schema_version": "agu.test"}
            payload["artifact_sha256"] = canonical_artifact_sha256(payload)
            members[rel] = (compact_canonical_json(payload) + "\n").encode()
    return members


def test_review_contexts_are_opaque_and_distinct():
    with pytest.raises(TypeError):
        VerifiedReviewDiscoveryContext()
    with pytest.raises(TypeError):
        VerifiedImplementationReviewSandboxContext()
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
