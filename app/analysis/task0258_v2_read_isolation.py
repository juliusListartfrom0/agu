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
    canonical_artifact_sha256,
    compact_canonical_json,
    is_safe_slug,
    is_sha256,
    verify_artifact_file_receipt,
    verify_exact_field_set,
)

READ_ISOLATION_POLICY_SCHEMA = "agu.task0258-module-a-worker-read-isolation-policy.v1"
READ_ISOLATION_ATTESTATION_SCHEMA = "agu.task0258-module-a-worker-read-isolation-attestation.v1"
READ_EVENT_PROJECTION_PROTOCOL = "compact-canonical-ordered-worker-read-events-sha256-v1"
MAXIMUM_READ_EVENT_ROWS = 21600
MAXIMUM_READ_EVENT_BYTES = 16777216

_READ_EVENT_OPERATIONS = frozenset(
    {"stat", "lstat", "fstat", "access", "readlink", "getdents", "xattr", "read", "open", "mmap", "exec"}
)
_READ_EVENT_LOCATORS = frozenset(
    {
        "path",
        "verified_inherited_fd",
        "manifest_bound_runtime_subtree",
        "envelope_bound_marker_fd",
        "owned_stage_subtree",
    }
)
_READ_EVENT_KINDS = frozenset({"regular", "directory", "symlink", "other"})
_RUNTIME_SNAPSHOT_FIELDS = frozenset(
    {
        "runtime_root_absolute_path",
        "runtime_root_device",
        "runtime_root_inode",
        "runtime_contract_absolute_path",
        "runtime_contract_size_bytes",
        "runtime_contract_receipt",
        "runtime_manifest_absolute_path",
        "runtime_manifest_size_bytes",
        "runtime_manifest_receipt",
        "runtime_tree_projection_sha256",
    }
)
_READ_EVENT_FIELDS = frozenset(
    {
        "ordinal",
        "operation",
        "path_role",
        "locator_kind",
        "normalized_path",
        "fd_number",
        "opened_fd_number",
        "follow_symlinks",
        "access_mode",
        "access_granted",
        "xattr_name",
        "result_state",
        "errno",
        "entry_kind",
        "mode_bits",
        "device",
        "inode",
        "size_bytes",
        "file_sha256",
        "symlink_target_text",
        "ordered_child_names",
        "xattr_value_sha256",
    }
)

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
    unsigned = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    if payload["artifact_sha256"] != canonical_artifact_sha256(unsigned):
        raise ValueError("read-isolation policy artifact hash is invalid")
    if payload["schema_version"] != READ_ISOLATION_POLICY_SCHEMA or payload["module_id"] != MODULE_ID:
        raise ValueError("read-isolation policy identity is invalid")
    verify_artifact_file_receipt(payload["provider_receipt"])
    verify_artifact_file_receipt(payload["run_identity_receipt"])
    if not is_safe_slug(payload["worker_role"]):
        raise ValueError("read-isolation policy worker_role is invalid")
    if not is_sha256(payload["child_nonce"]):
        raise ValueError("read-isolation policy child_nonce is invalid")
    root_identity = payload["output_root_identity"]
    if not isinstance(root_identity, Mapping) or set(root_identity) != {"absolute_path", "device", "inode"}:
        raise ValueError("read-isolation policy output_root_identity is invalid")
    if not isinstance(root_identity["absolute_path"], str) or not root_identity["absolute_path"].startswith("/"):
        raise ValueError("read-isolation policy output root path is invalid")
    _verify_nonnegative_int(root_identity["device"], "output root device")
    _verify_nonnegative_int(root_identity["inode"], "output root inode")
    if payload["maximum_policy_bytes"] != MAXIMUM_POLICY_BYTES:
        raise ValueError("read-isolation policy maximum_policy_bytes must be 2097152")
    if payload["maximum_read_event_rows"] != MAXIMUM_READ_EVENT_ROWS:
        raise ValueError("read-isolation policy maximum_read_event_rows is not the frozen cap")
    if payload["maximum_read_event_bytes"] != MAXIMUM_READ_EVENT_BYTES:
        raise ValueError("read-isolation policy maximum_read_event_bytes is not the frozen cap")
    if payload["read_event_projection_protocol"] != READ_EVENT_PROJECTION_PROTOCOL:
        raise ValueError("read-isolation policy projection protocol is invalid")
    for field in ("ordered_allowed_read_rows", "ordered_denied_read_rows"):
        if not isinstance(payload[field], (list, tuple)):
            raise ValueError(f"read-isolation policy {field} must be a list")
    runtime = payload["runtime_snapshot_contract_input"]
    if not isinstance(runtime, Mapping) or set(runtime) != _RUNTIME_SNAPSHOT_FIELDS:
        raise ValueError("read-isolation policy runtime_snapshot_contract_input is invalid")
    for field in ("runtime_root_absolute_path", "runtime_contract_absolute_path", "runtime_manifest_absolute_path"):
        if not isinstance(runtime[field], str) or not runtime[field].startswith("/"):
            raise ValueError(f"read-isolation policy runtime path is invalid: {field}")
    for field in (
        "runtime_root_device",
        "runtime_root_inode",
        "runtime_contract_size_bytes",
        "runtime_manifest_size_bytes",
    ):
        _verify_nonnegative_int(runtime[field], field)
    for field in ("runtime_contract_receipt", "runtime_manifest_receipt"):
        verify_artifact_file_receipt(runtime[field])
    if not is_sha256(runtime["runtime_tree_projection_sha256"]):
        raise ValueError("read-isolation policy runtime projection is invalid")
    _verify_policy_rows(payload["ordered_allowed_read_rows"], allowed=True)
    _verify_policy_rows(payload["ordered_denied_read_rows"], allowed=False)


def verify_read_isolation_attestation(payload: Mapping[str, object]) -> None:
    """Validate a `agu.task0258-module-a-worker-read-isolation-attestation.v1` payload."""
    verify_exact_field_set(payload, READ_ISOLATION_ATTESTATION_FIELDS)
    if payload["schema_version"] != READ_ISOLATION_ATTESTATION_SCHEMA or payload["module_id"] != MODULE_ID:
        raise ValueError("read-isolation attestation identity is invalid")
    verify_artifact_file_receipt(payload["provider_receipt"])
    verify_artifact_file_receipt(payload["run_identity_receipt"])
    if not is_sha256(payload["policy_artifact_sha256"]):
        raise ValueError("read-isolation attestation policy_artifact_sha256 is invalid")
    if not is_safe_slug(payload["worker_role"]):
        raise ValueError("read-isolation attestation worker_role is invalid")
    if not is_sha256(payload["child_nonce"]):
        raise ValueError("read-isolation attestation child_nonce is invalid")
    _verify_nonnegative_int(payload["child_pid"], "child_pid")
    if not is_sha256(payload["provider_process_instance_id"]):
        raise ValueError("provider_process_instance_id is invalid")
    for field in ("ordered_observed_process_ids", "ordered_observed_read_events"):
        if not isinstance(payload[field], (list, tuple)):
            raise ValueError(f"read-isolation attestation {field} must be a list")
    if payload["ordered_observed_process_ids"] != [payload["child_pid"]]:
        raise ValueError("read-isolation attestation process lineage is invalid")
    events = payload["ordered_observed_read_events"]
    if len(events) > MAXIMUM_READ_EVENT_ROWS:
        raise ValueError("read-isolation attestation event count exceeds the frozen cap")
    for ordinal, row in enumerate(events, start=1):
        _verify_read_event(row, ordinal)
    event_bytes = compact_canonical_json(events).encode("utf-8")
    if len(event_bytes) > MAXIMUM_READ_EVENT_BYTES:
        raise ValueError("read-isolation attestation event bytes exceed the frozen cap")
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
    import hashlib

    expected_projection = hashlib.sha256(event_bytes).hexdigest()
    if payload["read_event_projection_sha256"] != expected_projection:
        raise ValueError("read-isolation attestation event projection does not match rows")
    unsigned = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    if payload["artifact_sha256"] != canonical_artifact_sha256(unsigned):
        raise ValueError("read-isolation attestation artifact hash is invalid")


def _verify_policy_rows(rows: object, *, allowed: bool) -> None:
    if not isinstance(rows, (list, tuple)):
        raise ValueError("read-isolation policy rows must be lists")
    previous = None
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("read-isolation policy row is invalid")
        if allowed:
            expected = {
                "locator_kind",
                "path_role",
                "absolute_path",
                "fd_number",
                "match_kind",
                "entry_kind",
                "expected_device",
                "expected_inode",
                "expected_size_bytes",
                "expected_file_sha256",
                "expected_symlink_target_text",
                "expected_code_signature",
                "ordered_allowed_operations",
            }
        else:
            expected = {"path_role", "absolute_path", "match_kind"}
        if set(row) != expected:
            raise ValueError("read-isolation policy row field set is invalid")
        if not is_safe_slug(row["path_role"]):
            raise ValueError("read-isolation policy row path_role is invalid")
        if previous is not None and row["path_role"] <= previous:
            raise ValueError("read-isolation policy rows are not strictly ordered")
        previous = row["path_role"]
        if allowed:
            if row["locator_kind"] not in _READ_EVENT_LOCATORS or row["entry_kind"] not in {
                "regular",
                "directory",
                "symlink",
            }:
                raise ValueError("read-isolation policy allowed row discriminator is invalid")
            if row["locator_kind"] in {"verified_inherited_fd", "envelope_bound_marker_fd"}:
                if row["absolute_path"] is not None:
                    raise ValueError("read-isolation FD row must not carry a path")
            elif not isinstance(row["absolute_path"], str) or not row["absolute_path"].startswith("/"):
                raise ValueError("read-isolation policy row absolute_path is invalid")
            if row["fd_number"] is not None and (
                not isinstance(row["fd_number"], int) or isinstance(row["fd_number"], bool) or row["fd_number"] < 0
            ):
                raise ValueError("read-isolation policy fd_number is invalid")
            if row["match_kind"] not in {
                "exact_regular",
                "exact_directory",
                "exact_symlink",
                "exact_os_signed_regular",
                "owned_stage_subtree",
            }:
                raise ValueError("read-isolation policy match_kind is invalid")
            if row["entry_kind"] == "regular":
                if (
                    not isinstance(row["expected_size_bytes"], int)
                    or isinstance(row["expected_size_bytes"], bool)
                    or row["expected_size_bytes"] < 0
                ):
                    raise ValueError("read-isolation policy regular size is invalid")
                if not is_sha256(row["expected_file_sha256"]):
                    raise ValueError("read-isolation policy regular hash is invalid")
            elif row["entry_kind"] == "directory":
                if row["expected_size_bytes"] is not None or row["expected_file_sha256"] is not None:
                    raise ValueError("read-isolation policy directory nullability is invalid")
            elif row["entry_kind"] == "symlink" and not isinstance(row["expected_symlink_target_text"], str):
                raise ValueError("read-isolation policy symlink target is invalid")
            if not isinstance(row["ordered_allowed_operations"], (list, tuple)) or any(
                operation not in _READ_EVENT_OPERATIONS for operation in row["ordered_allowed_operations"]
            ):
                raise ValueError("read-isolation policy allowed operations are invalid")
        else:
            if not isinstance(row["absolute_path"], str) or not row["absolute_path"].startswith("/"):
                raise ValueError("read-isolation policy denied path is invalid")
            if row["match_kind"] not in {"exact_file", "subtree", "remaining_output_root"}:
                raise ValueError("read-isolation policy denied row discriminator is invalid")


def _verify_read_event(row: object, ordinal: int) -> None:
    if not isinstance(row, Mapping) or set(row) != _READ_EVENT_FIELDS:
        raise ValueError("read-isolation event field set is invalid")
    if row["ordinal"] != ordinal:
        raise ValueError("read-isolation event ordinal is invalid")
    if row["operation"] not in _READ_EVENT_OPERATIONS or not is_safe_slug(row["path_role"]):
        raise ValueError("read-isolation event operation/path role is invalid")
    if row["locator_kind"] not in _READ_EVENT_LOCATORS:
        raise ValueError("read-isolation event locator is invalid")
    path = row["normalized_path"]
    if path is not None and (not isinstance(path, str) or not path.startswith("/")):
        raise ValueError("read-isolation event normalized_path is invalid")
    for field in ("fd_number", "opened_fd_number", "access_mode", "mode_bits", "device", "inode", "size_bytes"):
        value = row[field]
        if value is not None:
            _verify_nonnegative_int(value, f"read-isolation event {field}")
    if row["access_mode"] is not None and row["access_mode"] > 7:
        raise ValueError("read-isolation event access_mode is invalid")
    if not isinstance(row["follow_symlinks"], bool):
        raise ValueError("read-isolation event follow_symlinks is invalid")
    if row["access_granted"] is not None and not isinstance(row["access_granted"], bool):
        raise ValueError("read-isolation event access_granted is invalid")
    if row["result_state"] not in {"success", "error"}:
        raise ValueError("read-isolation event result_state is invalid")
    if row["result_state"] == "error":
        if not isinstance(row["errno"], int) or isinstance(row["errno"], bool) or row["errno"] <= 0:
            raise ValueError("read-isolation event errno is invalid")
        if any(
            row[field] is not None
            for field in (
                "entry_kind",
                "mode_bits",
                "device",
                "inode",
                "size_bytes",
                "file_sha256",
                "symlink_target_text",
                "ordered_child_names",
                "xattr_value_sha256",
            )
        ):
            raise ValueError("read-isolation error event contains returned values")
    else:
        if row["errno"] is not None or row["entry_kind"] not in _READ_EVENT_KINDS:
            raise ValueError("read-isolation success event result is invalid")
    if row["file_sha256"] is not None and not is_sha256(row["file_sha256"]):
        raise ValueError("read-isolation event file hash is invalid")
    if row["xattr_value_sha256"] is not None and not is_sha256(row["xattr_value_sha256"]):
        raise ValueError("read-isolation event xattr hash is invalid")
    children = row["ordered_child_names"]
    if children is not None and (not isinstance(children, (list, tuple)) or list(children) != sorted(children)):
        raise ValueError("read-isolation event child listing is invalid")


__all__ = [
    "READ_ISOLATION_POLICY_SCHEMA",
    "READ_ISOLATION_ATTESTATION_SCHEMA",
    "MAXIMUM_POLICY_BYTES",
    "READ_EVENT_PROJECTION_PROTOCOL",
    "MAXIMUM_READ_EVENT_ROWS",
    "MAXIMUM_READ_EVENT_BYTES",
    "READ_ISOLATION_POLICY_FIELDS",
    "READ_ISOLATION_ATTESTATION_FIELDS",
    "verify_read_isolation_policy",
    "verify_read_isolation_attestation",
]
