"""Tests for the TASK-0258 v2 read-isolation policy/attestation schemas."""

from __future__ import annotations

import hashlib

import pytest

from app.analysis.task0258_module_a_v2 import MODULE_ID, canonical_artifact_sha256, compact_canonical_json
from app.analysis.task0258_v2_read_isolation import (
    MAXIMUM_POLICY_BYTES,
    MAXIMUM_READ_EVENT_BYTES,
    MAXIMUM_READ_EVENT_ROWS,
    READ_EVENT_PROJECTION_PROTOCOL,
    READ_ISOLATION_ATTESTATION_SCHEMA,
    READ_ISOLATION_POLICY_SCHEMA,
    verify_read_isolation_attestation,
    verify_read_isolation_binding,
    verify_read_isolation_policy,
)


def _receipt():
    return {"artifact_sha256": "0" * 64, "file_sha256": "0" * 64}


def _policy(*, allowed_rows=None, denied_rows=None):
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
        "ordered_allowed_read_rows": list(allowed_rows or []),
        "ordered_denied_read_rows": list(denied_rows or []),
        "read_event_projection_protocol": READ_EVENT_PROJECTION_PROTOCOL,
        "maximum_read_event_rows": MAXIMUM_READ_EVENT_ROWS,
        "maximum_read_event_bytes": MAXIMUM_READ_EVENT_BYTES,
        "maximum_policy_bytes": MAXIMUM_POLICY_BYTES,
    }
    payload["artifact_sha256"] = canonical_artifact_sha256(payload)
    return payload


def _attestation(*, policy=None, events=None):
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
        "ordered_observed_read_events": list(events or []),
        "read_event_projection_sha256": "",
        "denied_read_attempt_count": 0,
        "unknown_read_attempt_count": 0,
    }
    payload["read_event_projection_sha256"] = hashlib.sha256(
        compact_canonical_json(payload["ordered_observed_read_events"]).encode("utf-8")
    ).hexdigest()
    payload["artifact_sha256"] = canonical_artifact_sha256(payload)
    return payload


def _allowed_regular_row():
    return {
        "locator_kind": "path",
        "path_role": "input_file",
        "absolute_path": "/input.json",
        "fd_number": None,
        "match_kind": "exact_regular",
        "entry_kind": "regular",
        "expected_device": 1,
        "expected_inode": 2,
        "expected_size_bytes": 3,
        "expected_file_sha256": "1" * 64,
        "expected_symlink_target_text": None,
        "expected_code_signature": None,
        "ordered_allowed_operations": ["open"],
    }


def _allowed_regular_event():
    return {
        "ordinal": 1,
        "operation": "open",
        "path_role": "input_file",
        "locator_kind": "path",
        "normalized_path": "/input.json",
        "fd_number": None,
        "opened_fd_number": 3,
        "follow_symlinks": False,
        "access_mode": 0,
        "access_granted": None,
        "xattr_name": None,
        "result_state": "success",
        "errno": None,
        "entry_kind": "regular",
        "mode_bits": 0o600,
        "device": 1,
        "inode": 2,
        "size_bytes": 3,
        "file_sha256": "1" * 64,
        "symlink_target_text": None,
        "ordered_child_names": None,
        "xattr_value_sha256": None,
    }


def _allowed_fd_row():
    row = _allowed_regular_row()
    row.update(
        {
            "locator_kind": "verified_inherited_fd",
            "path_role": "bootstrap_source",
            "absolute_path": None,
            "fd_number": 4,
            "match_kind": "exact_regular",
            "ordered_allowed_operations": ["fstat"],
        }
    )
    return row


def _allowed_fd_event():
    event = _allowed_regular_event()
    event.update(
        {
            "operation": "fstat",
            "path_role": "bootstrap_source",
            "locator_kind": "verified_inherited_fd",
            "normalized_path": None,
            "fd_number": 4,
            "opened_fd_number": None,
        }
    )
    return event


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
    bad_row = _allowed_regular_row()
    bad_row["expected_device"] = None
    bad = _policy(allowed_rows=[bad_row])
    with pytest.raises(ValueError, match="regular device"):
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


def test_attestation_binding_replays_policy_and_event_identity():
    policy = _policy(allowed_rows=[_allowed_regular_row()])
    attestation = _attestation(policy=policy, events=[_allowed_regular_event()])

    verify_read_isolation_binding(policy, attestation)


def test_attestation_binding_rejects_policy_tuple_drift():
    policy = _policy(allowed_rows=[_allowed_regular_row()])
    attestation = _attestation(policy=policy, events=[_allowed_regular_event()])
    attestation["worker_role"] = "producer"
    attestation["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in attestation.items() if key != "artifact_sha256"}
    )

    with pytest.raises(ValueError, match="worker role"):
        verify_read_isolation_binding(policy, attestation)


def test_attestation_binding_rejects_unknown_event_and_denied_path():
    policy = _policy(
        allowed_rows=[_allowed_regular_row()],
        denied_rows=[{"path_role": "denied_input", "absolute_path": "/private", "match_kind": "subtree"}],
    )
    unknown = _allowed_regular_event()
    unknown["normalized_path"] = "/other.json"
    unknown["path_role"] = "unknown"
    unknown["file_sha256"] = "2" * 64
    with pytest.raises(ValueError, match="unknown read event"):
        verify_read_isolation_binding(policy, _attestation(policy=policy, events=[unknown]))

    denied = _allowed_regular_event()
    denied["normalized_path"] = "/private/secret.json"
    denied["path_role"] = "input_file"
    with pytest.raises(ValueError, match="denied read event"):
        verify_read_isolation_binding(policy, _attestation(policy=policy, events=[denied]))


def test_attestation_binding_supports_fixed_inherited_fd_rows():
    policy = _policy(allowed_rows=[_allowed_fd_row()])
    verify_read_isolation_binding(policy, _attestation(policy=policy, events=[_allowed_fd_event()]))


def test_attestation_binding_rejects_ambiguous_longest_match():
    first = _allowed_regular_row()
    second = dict(first)
    second["path_role"] = "input_file_alt"
    policy = _policy(allowed_rows=[first, second])

    with pytest.raises(ValueError, match="ambiguous"):
        verify_read_isolation_binding(policy, _attestation(policy=policy, events=[_allowed_regular_event()]))
