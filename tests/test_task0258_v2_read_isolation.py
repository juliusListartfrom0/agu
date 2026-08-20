"""Tests for the TASK-0258 v2 read-isolation policy/attestation schemas."""

from __future__ import annotations

import pytest

from app.analysis.task0258_module_a_v2 import MODULE_ID, canonical_artifact_sha256
from app.analysis.task0258_v2_read_isolation import (
    MAXIMUM_POLICY_BYTES,
    MAXIMUM_READ_EVENT_BYTES,
    MAXIMUM_READ_EVENT_ROWS,
    READ_EVENT_PROJECTION_PROTOCOL,
    READ_ISOLATION_ATTESTATION_SCHEMA,
    READ_ISOLATION_POLICY_SCHEMA,
    verify_read_isolation_attestation,
    verify_read_isolation_policy,
)


def _receipt():
    return {"artifact_sha256": "0" * 64, "file_sha256": "0" * 64}


def _policy():
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


def _attestation():
    payload = {
        "schema_version": READ_ISOLATION_ATTESTATION_SCHEMA,
        "module_id": MODULE_ID,
        "policy_artifact_sha256": "0" * 64,
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


def test_policy_valid():
    verify_read_isolation_policy(_policy())
    bad = _policy()
    bad["maximum_policy_bytes"] = 1
    with pytest.raises(ValueError):
        verify_read_isolation_policy(bad)
    bad = _policy()
    bad["worker_role"] = "bad role!"
    with pytest.raises(ValueError):
        verify_read_isolation_policy(bad)


def test_attestation_valid():
    verify_read_isolation_attestation(_attestation())
    bad = _attestation()
    bad["denied_read_attempt_count"] = 1
    with pytest.raises(ValueError):
        verify_read_isolation_attestation(bad)
    bad = _attestation()
    bad["audit_overflow"] = True
    with pytest.raises(ValueError):
        verify_read_isolation_attestation(bad)
    bad = _attestation()
    bad["audit_started_before_spawn"] = False
    with pytest.raises(ValueError):
        verify_read_isolation_attestation(bad)
