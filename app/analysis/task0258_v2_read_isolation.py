"""TASK-0258 Amendment-001 v2 verification worker — read-isolation schemas.

Closed validators for the kernel-audit read-isolation policy and attestation
(amendment §"Review sandbox and authorized pre-import bootstrap" / the
read-isolation objects carried by worker attempts). The *production* of the
attestation requires a platform kernel-audit provider (macOS Endpoint Security /
syscall audit); this module pins the exact artifact schemas and invariants that
the production boundary must satisfy.
"""

from __future__ import annotations

from collections.abc import Mapping

from app.analysis.task0258_module_a_v2 import (
    MODULE_ID,
    is_safe_slug,
    is_sha256,
    verify_artifact_file_receipt,
    verify_exact_field_set,
)

READ_ISOLATION_POLICY_SCHEMA = "agu.task0258-module-a-worker-read-isolation-policy.v1"
READ_ISOLATION_ATTESTATION_SCHEMA = "agu.task0258-module-a-worker-read-isolation-attestation.v1"

MAXIMUM_POLICY_BYTES = 2_097_152  # 2 MiB

READ_ISOLATION_POLICY_FIELDS = frozenset(
    {
        "schema_version",
        "module_id",
        "provider_receipt",
        "run_identity_receipt",
        "worker_role",
        "child_nonce",
        "output_root_identity",
        "runtime_snapshot_contract_input",
        "ordered_allowed_read_rows",
        "ordered_denied_read_rows",
        "read_event_projection_protocol",
        "maximum_read_event_rows",
        "maximum_read_event_bytes",
        "maximum_policy_bytes",
        "artifact_sha256",
    }
)

READ_ISOLATION_ATTESTATION_FIELDS = frozenset(
    {
        "schema_version",
        "module_id",
        "policy_artifact_sha256",
        "provider_receipt",
        "run_identity_receipt",
        "worker_role",
        "child_pid",
        "ordered_observed_process_ids",
        "provider_process_instance_id",
        "child_nonce",
        "prepare_artifact_sha256",
        "prepared_artifact_sha256",
        "child_started_artifact_sha256",
        "permit_artifact_sha256",
        "finalize_artifact_sha256",
        "audit_started_before_spawn",
        "audit_ended_after_child_exit",
        "audit_overflow",
        "ordered_observed_read_events",
        "read_event_projection_sha256",
        "denied_read_attempt_count",
        "unknown_read_attempt_count",
        "artifact_sha256",
    }
)


def _verify_nonnegative_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} is invalid")


def verify_read_isolation_policy(payload: Mapping[str, object]) -> None:
    """Validate a `agu.task0258-module-a-worker-read-isolation-policy.v1` payload."""
    verify_exact_field_set(payload, READ_ISOLATION_POLICY_FIELDS)
    if payload["schema_version"] != READ_ISOLATION_POLICY_SCHEMA or payload["module_id"] != MODULE_ID:
        raise ValueError("read-isolation policy identity is invalid")
    verify_artifact_file_receipt(payload["provider_receipt"])
    verify_artifact_file_receipt(payload["run_identity_receipt"])
    if not is_safe_slug(payload["worker_role"]):
        raise ValueError("read-isolation policy worker_role is invalid")
    if not is_sha256(payload["child_nonce"]):
        raise ValueError("read-isolation policy child_nonce is invalid")
    root_identity = payload["output_root_identity"]
    if not isinstance(root_identity, Mapping) or set(root_identity) != {"device", "inode"}:
        raise ValueError("read-isolation policy output_root_identity is invalid")
    _verify_nonnegative_int(root_identity["device"], "output root device")
    _verify_nonnegative_int(root_identity["inode"], "output root inode")
    if payload["maximum_policy_bytes"] != MAXIMUM_POLICY_BYTES:
        raise ValueError("read-isolation policy maximum_policy_bytes must be 2097152")
    _verify_nonnegative_int(payload["maximum_read_event_rows"], "maximum_read_event_rows")
    _verify_nonnegative_int(payload["maximum_read_event_bytes"], "maximum_read_event_bytes")
    for field in ("ordered_allowed_read_rows", "ordered_denied_read_rows"):
        if not isinstance(payload[field], (list, tuple)):
            raise ValueError(f"read-isolation policy {field} must be a list")
    if not isinstance(payload["runtime_snapshot_contract_input"], Mapping):
        raise ValueError("read-isolation policy runtime_snapshot_contract_input is invalid")


def verify_read_isolation_attestation(payload: Mapping[str, object]) -> None:
    """Validate a `agu.task0258-module-a-worker-read-isolation-attestation.v1` payload."""
    verify_exact_field_set(payload, READ_ISOLATION_ATTESTATION_FIELDS)
    if payload["schema_version"] != READ_ISOLATION_ATTESTATION_SCHEMA or payload["module_id"] != MODULE_ID:
        raise ValueError("read-isolation attestation identity is invalid")
    for field in ("policy_artifact_sha256", "provider_receipt", "run_identity_receipt"):
        # policy hash is a plain SHA; receipts are ArtifactFileReceipt
        pass
    verify_artifact_file_receipt(payload["provider_receipt"])
    verify_artifact_file_receipt(payload["run_identity_receipt"])
    if not is_sha256(payload["policy_artifact_sha256"]):
        raise ValueError("read-isolation attestation policy_artifact_sha256 is invalid")
    if not is_safe_slug(payload["worker_role"]):
        raise ValueError("read-isolation attestation worker_role is invalid")
    if not is_sha256(payload["child_nonce"]):
        raise ValueError("read-isolation attestation child_nonce is invalid")
    _verify_nonnegative_int(payload["child_pid"], "child_pid")
    _verify_nonnegative_int(payload["provider_process_instance_id"], "provider_process_instance_id")
    for field in ("ordered_observed_process_ids", "ordered_observed_read_events"):
        if not isinstance(payload[field], (list, tuple)):
            raise ValueError(f"read-isolation attestation {field} must be a list")
    for field in (
        "prepare_artifact_sha256",
        "prepared_artifact_sha256",
        "child_started_artifact_sha256",
        "permit_artifact_sha256",
        "finalize_artifact_sha256",
    ):
        if not is_sha256(payload[field]):
            raise ValueError(f"read-isolation attestation {field} is invalid")
    if payload["audit_started_before_spawn"] is not True or payload["audit_ended_after_child_exit"] is not True:
        raise ValueError("read-isolation attestation audit bounds must be True")
    if payload["audit_overflow"] is not False:
        raise ValueError("read-isolation attestation audit_overflow must be False")
    if payload["denied_read_attempt_count"] != 0 or payload["unknown_read_attempt_count"] != 0:
        raise ValueError("read-isolation attestation denied/unknown counts must be zero")
    if not is_sha256(payload["read_event_projection_sha256"]):
        raise ValueError("read-isolation attestation read_event_projection_sha256 is invalid")


__all__ = [
    "READ_ISOLATION_POLICY_SCHEMA",
    "READ_ISOLATION_ATTESTATION_SCHEMA",
    "MAXIMUM_POLICY_BYTES",
    "READ_ISOLATION_POLICY_FIELDS",
    "READ_ISOLATION_ATTESTATION_FIELDS",
    "verify_read_isolation_policy",
    "verify_read_isolation_attestation",
]
