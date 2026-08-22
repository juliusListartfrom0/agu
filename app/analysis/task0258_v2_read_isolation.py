"""TASK-0258 Amendment-001 v2 verification worker — read-isolation schemas.

Closed validators for the kernel-audit read-isolation policy and attestation
(amendment §"Review sandbox and authorized pre-import bootstrap" / the
read-isolation objects carried by worker attempts). The *production* of the
attestation requires a platform kernel-audit provider (macOS Endpoint Security /
syscall audit); this module pins the exact artifact schemas and invariants that
the production boundary must satisfy.
"""

from __future__ import annotations

import posixpath
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
        "verified_inherited_directory_fd",
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
    _verify_absolute_path(root_identity["absolute_path"], "read-isolation policy output root path")
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
        _verify_absolute_path(runtime[field], f"read-isolation policy runtime path: {field}")
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
    if not isinstance(payload["child_pid"], int) or isinstance(payload["child_pid"], bool) or payload["child_pid"] <= 0:
        raise ValueError("child_pid is invalid")
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


def verify_read_isolation_binding(policy: Mapping[str, object], attestation: Mapping[str, object]) -> None:
    """Bind an externally produced attestation to its exact read policy.

    The function only verifies an already-produced provider artifact; it never
    creates one.  In particular, the caller must still obtain the attestation
    from an externally authenticated kernel-audit provider.  This local
    binding closes the separate seam where two individually schema-valid
    objects could otherwise describe different workers or an unlisted read.
    """
    if not isinstance(policy, Mapping) or not isinstance(attestation, Mapping):
        raise TypeError("read-isolation policy and attestation must be mappings")
    verify_read_isolation_policy(policy)
    verify_read_isolation_attestation(attestation)

    if attestation["policy_artifact_sha256"] != policy["artifact_sha256"]:
        raise ValueError("read-isolation attestation policy hash does not match policy bytes")
    for field in ("provider_receipt", "run_identity_receipt", "worker_role", "child_nonce"):
        if attestation[field] != policy[field]:
            raise ValueError(f"read-isolation attestation {field.replace('_', ' ')} does not match policy")

    allowed_rows = policy["ordered_allowed_read_rows"]
    denied_rows = policy["ordered_denied_read_rows"]
    observed_events = attestation["ordered_observed_read_events"]
    if not allowed_rows:
        raise ValueError("read-isolation policy must contain allowed reads")
    if not observed_events:
        raise ValueError("read-isolation attestation must contain observed allowed reads")

    matched_allowed_rows: set[int] = set()
    for event in observed_events:
        matching_denied = [row for row in denied_rows if _denied_row_matches_event(row, event)]
        if matching_denied:
            raise ValueError("denied read event matched the read-isolation policy")

        matching_allowed = [
            (index, row) for index, row in enumerate(allowed_rows) if _allowed_row_matches_event(row, event)
        ]
        if not matching_allowed:
            raise ValueError("unknown read event is not covered by the read-isolation policy")
        matched_allowed_rows.update(index for index, _row in matching_allowed)
        selected = _select_unique_allowed_row([row for _index, row in matching_allowed])
        if event["path_role"] != selected["path_role"]:
            raise ValueError("read event path role does not match the policy row")
        _verify_event_against_allowed_row(event, selected)

    if len(matched_allowed_rows) != len(allowed_rows):
        raise ValueError("read-isolation attestation does not cover every allowed read row")


def _verify_absolute_path(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or "\x00" in value
        or value.startswith("//")
        or posixpath.normpath(value) != value
    ):
        raise ValueError(f"{name} is invalid")


def _is_lower_hex(value: object) -> bool:
    return isinstance(value, str) and bool(value) and all(char in "0123456789abcdef" for char in value)


def _path_is_within(path: str, root: str) -> bool:
    return path == root or root == "/" or path.startswith(root.rstrip("/") + "/")


def _denied_row_matches_event(row: Mapping[str, object], event: Mapping[str, object]) -> bool:
    if event["locator_kind"] != "path" or not isinstance(event["normalized_path"], str):
        return False
    path = event["normalized_path"]
    denied_path = row["absolute_path"]
    if row["match_kind"] == "exact_file":
        return path == denied_path
    return _path_is_within(path, denied_path)


def _allowed_row_matches_event(row: Mapping[str, object], event: Mapping[str, object]) -> bool:
    if event["locator_kind"] != row["locator_kind"]:
        return False
    if event["operation"] not in row["ordered_allowed_operations"]:
        return False
    locator_kind = row["locator_kind"]
    if locator_kind in {"verified_inherited_fd", "verified_inherited_directory_fd", "envelope_bound_marker_fd"}:
        return event["normalized_path"] is None and event["fd_number"] == row["fd_number"]
    if not isinstance(event["normalized_path"], str) or not isinstance(row["absolute_path"], str):
        return False
    if row["match_kind"] in {"owned_stage_subtree", "manifest_bound_runtime_subtree"}:
        return _path_is_within(event["normalized_path"], row["absolute_path"])
    return event["normalized_path"] == row["absolute_path"]


def _select_unique_allowed_row(rows: list[Mapping[str, object]]) -> Mapping[str, object]:
    if len(rows) == 1:
        return rows[0]
    path_rows = [row for row in rows if isinstance(row["absolute_path"], str)]
    if len(path_rows) != len(rows):
        raise ValueError("read event matches ambiguous policy rows")
    longest = max(len(row["absolute_path"]) for row in path_rows)
    selected = [row for row in path_rows if len(row["absolute_path"]) == longest]
    if len(selected) != 1:
        raise ValueError("read event matches ambiguous policy rows")
    return selected[0]


def _verify_event_against_allowed_row(event: Mapping[str, object], row: Mapping[str, object]) -> None:
    if event["result_state"] == "error":
        return
    if event["entry_kind"] != row["entry_kind"]:
        raise ValueError("read event entry kind does not match the policy row")
    expected_fields = {
        "expected_device": "device",
        "expected_inode": "inode",
        "expected_size_bytes": "size_bytes",
        "expected_file_sha256": "file_sha256",
        "expected_symlink_target_text": "symlink_target_text",
    }
    for expected_field, event_field in expected_fields.items():
        expected = row[expected_field]
        if expected is not None and event[event_field] != expected:
            raise ValueError(f"read event {event_field} does not match the policy row")
    if event["operation"] == "open" and event["opened_fd_number"] is None:
        raise ValueError("successful open event lacks its opened descriptor")
    if event["operation"] != "open" and event["opened_fd_number"] is not None:
        raise ValueError("non-open event carries an opened descriptor")


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
            if row["locator_kind"] in {
                "verified_inherited_fd",
                "verified_inherited_directory_fd",
                "envelope_bound_marker_fd",
            }:
                if row["absolute_path"] is not None:
                    raise ValueError("read-isolation FD row must not carry a path")
                if not isinstance(row["fd_number"], int) or isinstance(row["fd_number"], bool) or row["fd_number"] < 0:
                    raise ValueError("read-isolation FD row must carry a fixed descriptor")
            else:
                _verify_absolute_path(row["absolute_path"], "read-isolation policy row absolute_path")
            if (
                row["locator_kind"]
                not in {
                    "verified_inherited_fd",
                    "verified_inherited_directory_fd",
                    "envelope_bound_marker_fd",
                }
                and row["fd_number"] is not None
                and (
                    not isinstance(row["fd_number"], int) or isinstance(row["fd_number"], bool) or row["fd_number"] < 0
                )
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
                _verify_nonnegative_int(row["expected_device"], "read-isolation policy regular device")
                _verify_nonnegative_int(row["expected_inode"], "read-isolation policy regular inode")
                if not is_sha256(row["expected_file_sha256"]):
                    raise ValueError("read-isolation policy regular hash is invalid")
            elif row["entry_kind"] == "directory":
                _verify_nonnegative_int(row["expected_device"], "read-isolation policy directory device")
                _verify_nonnegative_int(row["expected_inode"], "read-isolation policy directory inode")
                if (
                    row["expected_size_bytes"] is not None
                    or row["expected_file_sha256"] is not None
                    or row["expected_symlink_target_text"] is not None
                ):
                    raise ValueError("read-isolation policy directory nullability is invalid")
            elif row["entry_kind"] == "symlink":
                _verify_nonnegative_int(row["expected_device"], "read-isolation policy symlink device")
                _verify_nonnegative_int(row["expected_inode"], "read-isolation policy symlink inode")
                if not isinstance(row["expected_symlink_target_text"], str):
                    raise ValueError("read-isolation policy symlink target is invalid")
                if row["expected_size_bytes"] is not None or row["expected_file_sha256"] is not None:
                    raise ValueError("read-isolation policy symlink nullability is invalid")
            if row["match_kind"] == "exact_os_signed_regular":
                signature = row["expected_code_signature"]
                if (
                    not isinstance(signature, Mapping)
                    or set(signature) != {"absolute_path", "team_identifier", "cdhash"}
                    or signature["absolute_path"] != row["absolute_path"]
                    or not is_safe_slug(signature["team_identifier"])
                    or not _is_lower_hex(signature["cdhash"])
                ):
                    raise ValueError("read-isolation OS-signed code signature is invalid")
            elif row["expected_code_signature"] is not None:
                raise ValueError("read-isolation code signature is only valid for OS-signed rows")
            if not isinstance(row["ordered_allowed_operations"], (list, tuple)) or any(
                operation not in _READ_EVENT_OPERATIONS for operation in row["ordered_allowed_operations"]
            ):
                raise ValueError("read-isolation policy allowed operations are invalid")
        else:
            _verify_absolute_path(row["absolute_path"], "read-isolation policy denied path")
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
    if path is not None:
        _verify_absolute_path(path, "read-isolation event normalized_path")
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
    if children is not None:
        if (
            not isinstance(children, (list, tuple))
            or any(not isinstance(child, str) or "/" in child or "\x00" in child for child in children)
            or list(children) != sorted(children)
        ):
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
    "verify_read_isolation_binding",
]
