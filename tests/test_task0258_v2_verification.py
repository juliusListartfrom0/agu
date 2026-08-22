"""Tests for the TASK-0258 v2 verification worker artifact schemas."""

from __future__ import annotations

import pytest

from app.analysis.task0258_module_a_v2 import (
    MODULE_ID,
    VERIFICATION_ATTEMPT_SCHEMA_V2,
    VERIFICATION_EMBEDDING_SCHEMA_V2,
    VERIFICATION_RESOURCE_SAMPLE_SCHEMA,
    canonical_artifact_sha256,
)
from app.analysis.task0258_v2_verification import (
    ROLE,
    verify_verification_attempt,
    verify_verification_embedding,
    verify_verification_resource_sample,
)


def _receipt():
    return {"artifact_sha256": "0" * 64, "file_sha256": "0" * 64}


def _auth():
    return {
        "parent_spec_approval": _receipt(),
        "amendment_implementation_approval": _receipt(),
        "amended_implementation_review": _receipt(),
        "rerun_authorization": _receipt(),
    }


def _common(schema_version):
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


def _embedding():
    p = _common(VERIFICATION_EMBEDDING_SCHEMA_V2)
    p.update(
        {
            "role": ROLE,
            "authorization_receipts": _auth(),
            "run_identity_receipt": _receipt(),
            "history_head_receipt": _receipt(),
            "run_admission_receipt": _receipt(),
            "plan_receipt": _receipt(),
            "task0257_input_receipts": [],
            "representation": "swin3d-t-tiled-4x2s-mean-delta-v1",
            "producer_environment": {},
            "checkpoint_receipt": _receipt(),
            "source_video_receipts": [],
            "row_count": 45,
            "examples": [],
        }
    )
    p["artifact_sha256"] = canonical_artifact_sha256(p)
    return p


def _attempt(disposition="completed"):
    policy = _policy()
    p = _common(VERIFICATION_ATTEMPT_SCHEMA_V2)
    p.update(
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
            "read_isolation_attestation": _attestation(policy=policy),
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
            "disposition": disposition,
            "stop_reason": None,
        }
    )
    p["artifact_sha256"] = canonical_artifact_sha256(p)
    return p


def _resource_sample():
    return {
        "schema_version": VERIFICATION_RESOURCE_SAMPLE_SCHEMA,
        "attempt_role": ROLE,
        "verification_ordinal": 1,
        "attempt_sample_ordinal": 1,
        "cumulative_sample_ordinal": 1,
        "scheduled_cumulative_active_seconds": 2,
        "observed_attempt_active_nanoseconds": 0,
        "system_memory_percent": 0.0,
        "available_memory_bytes": 100,
        "free_swap_bytes": 100,
        "system_cpu_percent": 0.0,
        "consecutive_breach_count": 0,
        "breached_limits": [],
    }


def test_verification_resource_sample_valid():
    verify_verification_resource_sample(_resource_sample())
    bad = _resource_sample()
    bad["attempt_role"] = "producer"
    with pytest.raises(ValueError):
        verify_verification_resource_sample(bad)


def test_verification_embedding_valid():
    verify_verification_embedding(_embedding())
    bad = _embedding()
    bad["role"] = "producer"
    with pytest.raises(ValueError):
        verify_verification_embedding(bad)
    stale = _embedding()
    stale["examples"].append({"tampered": True})
    with pytest.raises(ValueError, match="artifact_sha256"):
        verify_verification_embedding(stale)
    bad = _embedding()
    bad["row_count"] = 44
    with pytest.raises(ValueError):
        verify_verification_embedding(bad)


def test_verification_attempt_completed_valid():
    verify_verification_attempt(_attempt("completed"))
    # completed requires verified slot; fail if slot not verified
    bad = _attempt("completed")
    bad["verification_embedding_slot"]["verification_state"] = "failed"
    bad["verification_embedding_slot"]["receipt"] = None
    with pytest.raises(ValueError):
        verify_verification_attempt(bad)
    stale = _attempt("completed")
    stale["worker_payload"]["tampered"] = True
    with pytest.raises(ValueError, match="artifact_sha256"):
        verify_verification_attempt(stale)


def test_verification_attempt_terminal_failure():
    # null signal -> inner stop reason
    bad = _attempt("terminal_failure")
    bad["received_signal"] = None
    bad["stop_reason"] = "determinism_failure"
    bad["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in bad.items() if key != "artifact_sha256"}
    )
    verify_verification_attempt(bad)
    # SIGINT -> external_sigint
    bad2 = _attempt("terminal_failure")
    bad2["received_signal"] = "SIGINT"
    bad2["stop_reason"] = "external_sigint"
    bad2["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in bad2.items() if key != "artifact_sha256"}
    )
    verify_verification_attempt(bad2)
    # wrong pairing rejected
    bad3 = _attempt("terminal_failure")
    bad3["received_signal"] = "SIGINT"
    bad3["stop_reason"] = "determinism_failure"
    with pytest.raises(ValueError):
        verify_verification_attempt(bad3)


def test_verification_attempt_rejects_bad_ordinal_and_provider():
    bad = _attempt("completed")
    bad["verification_ordinal"] = 2
    with pytest.raises(ValueError):
        verify_verification_attempt(bad)
    bad = _attempt("completed")
    bad["verification_embedding_slot"]["provider"] = "producer_tiled_swin_embeddings"
    with pytest.raises(ValueError):
        verify_verification_attempt(bad)


def test_float32_normalization():
    from app.analysis.task0258_v2_verification import normalize_float32_leaf

    # 0.1 cannot be represented exactly in float32
    assert normalize_float32_leaf(0.1) != 0.1
    assert normalize_float32_leaf(0.5) == 0.5
    assert normalize_float32_leaf([0.1, [0.5]]) == [normalize_float32_leaf(0.1), [0.5]]


def test_compute_verification_projection_deterministic():
    from app.analysis.task0258_v2_verification import compute_verification_projection

    examples = [
        {
            "ordinal": 1,
            "key": {"source_video_sha256": "0" * 64, "candidate_bundle_sha256": "0" * 64, "event_id": "e1"},
            "tile_frame_indexes": [[0, 1, 2, 3] * 4] * 4,
            "derivation_only_tile_embeddings": [[0.1, 0.2] * 384] * 4,
            "model_input": [0.1, 0.2] * 768,
        },
        {
            "ordinal": 2,
            "key": {"source_video_sha256": "0" * 64, "candidate_bundle_sha256": "0" * 64, "event_id": "e2"},
            "tile_frame_indexes": [[4, 5, 6, 7] * 4] * 4,
            "derivation_only_tile_embeddings": [[0.3, 0.4] * 384] * 4,
            "model_input": [0.3, 0.4] * 768,
        },
    ]
    h1 = compute_verification_projection("swin3d-t-tiled-4x2s-mean-delta-v1", examples)
    h2 = compute_verification_projection("swin3d-t-tiled-4x2s-mean-delta-v1", examples)
    assert h1 == h2
    assert len(h1) == 64
    # representation change changes the hash
    assert h1 != compute_verification_projection("other", examples)


def test_build_verification_embedding_payload():
    from app.analysis.task0258_v2_verification import build_verification_embedding_payload

    examples = [
        {
            "ordinal": i,
            "key": {"source_video_sha256": "0" * 64, "candidate_bundle_sha256": "0" * 64, "event_id": f"e{i}"},
            "tile_frame_indexes": [[0] * 16] * 4,
            "derivation_only_tile_embeddings": [[0.0] * 768] * 4,
            "model_input": [0.0] * 1536,
        }
        for i in range(1, 46)
    ]
    payload = build_verification_embedding_payload(
        authorization_receipts=_auth(),
        run_identity_receipt=_receipt(),
        history_head_receipt=_receipt(),
        run_admission_receipt=_receipt(),
        plan_receipt=_receipt(),
        task0257_input_receipts=[],
        representation="swin3d-t-tiled-4x2s-mean-delta-v1",
        producer_environment={},
        checkpoint_receipt=_receipt(),
        source_video_receipts=[],
        examples=examples,
    )
    verify_verification_embedding(payload)
    assert payload["row_count"] == 45
    assert len(payload["artifact_sha256"]) == 64


def _policy():
    from app.analysis.task0258_v2_read_isolation import (
        MAXIMUM_POLICY_BYTES,
        MAXIMUM_READ_EVENT_BYTES,
        MAXIMUM_READ_EVENT_ROWS,
        READ_EVENT_PROJECTION_PROTOCOL,
        READ_ISOLATION_POLICY_SCHEMA,
    )

    payload = {
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
        "ordered_allowed_read_rows": [],
        "ordered_denied_read_rows": [],
        "read_event_projection_protocol": READ_EVENT_PROJECTION_PROTOCOL,
        "maximum_read_event_rows": MAXIMUM_READ_EVENT_ROWS,
        "maximum_read_event_bytes": MAXIMUM_READ_EVENT_BYTES,
        "maximum_policy_bytes": MAXIMUM_POLICY_BYTES,
    }
    payload["artifact_sha256"] = canonical_artifact_sha256(payload)
    return payload


def _attestation(*, policy=None):
    from app.analysis.task0258_v2_read_isolation import READ_ISOLATION_ATTESTATION_SCHEMA

    payload = {
        "schema_version": READ_ISOLATION_ATTESTATION_SCHEMA,
        "module_id": MODULE_ID,
        "policy_artifact_sha256": (policy or {}).get("artifact_sha256", "0" * 64),
        "provider_receipt": _receipt(),
        "run_identity_receipt": _receipt(),
        "worker_role": "verification",
        "child_pid": 42,
        "ordered_observed_process_ids": [42],
        "provider_process_instance_id": "1" * 64,
        "child_nonce": "0" * 64,
        "prepare_artifact_sha256": "0" * 64,
        "prepared_artifact_sha256": "0" * 64,
        "child_started_artifact_sha256": "0" * 64,
        "permit_artifact_sha256": "0" * 64,
        "finalize_artifact_sha256": "0" * 64,
        "audit_started_before_spawn": True,
        "audit_ended_after_child_exit": True,
        "audit_overflow": False,
        "ordered_observed_read_events": [],
        "read_event_projection_sha256": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        "denied_read_attempt_count": 0,
        "unknown_read_attempt_count": 0,
    }
    payload["artifact_sha256"] = canonical_artifact_sha256(payload)
    return payload


def test_verification_attempt_binds_read_isolation():
    p = _attempt("completed")
    verify_verification_attempt(p)
    # attestation with a denied read attempt is rejected
    bad = _attempt("completed")
    bad["read_isolation_attestation"]["denied_read_attempt_count"] = 1
    with pytest.raises(ValueError):
        verify_verification_attempt(bad)


def test_verification_attempt_rejects_read_isolation_tuple_drift():
    bad = _attempt("completed")
    bad["read_isolation_attestation"]["worker_role"] = "producer"
    bad["read_isolation_attestation"]["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in bad["read_isolation_attestation"].items() if key != "artifact_sha256"}
    )
    bad["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in bad.items() if key != "artifact_sha256"}
    )

    with pytest.raises(ValueError, match="worker role"):
        verify_verification_attempt(bad)
