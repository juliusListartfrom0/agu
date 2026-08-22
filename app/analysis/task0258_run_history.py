"""TASK-0258 Amendment-001 v2 run-history registry (foundation / trust spine).

This module implements the exact run-history transition graph and filename
grammar from amendment-001 §"Exact run-history registry encoding". It is the
durable trust spine every v2 producer/verification/candidate/result artifact
binds through `history_head_receipt` and `run_identity_receipt`.

The registry is a monotone, append-only directory owned by an external authority:
- `<AUTH_SHA>.claim.json` and `<AUTH_SHA>.completed.json` bookend the run;
- `<AUTH_SHA>.history-<NN>-<EVENT>.json` markers record each state transition.

This module currently provides the exact event set, the transition graph, the
marker filename grammar, and the marker scalar/edge validators. Registry
creation, subject-receipt binding, root-subject CAS, and loaders are added in
later phases.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from app.analysis.task0258_module_a_v2 import (
    compact_canonical_json,
    is_rfc3339,
    is_safe_slug,
    is_sha256,
    verify_artifact_file_receipt,
    verify_internal_artifact_hash,
    verify_static_input_contract,
)
from app.analysis.task0258_v2_fs import (
    FlockHandle,
    _open_existing_directory_no_follow,
    active_flock,
    exclusive_flock_at,
)

MARKER_SCHEMA = "agu.task0258-module-a-v2-run-history-marker.v1"
CLAIM_SCHEMA = "agu.task0258-module-a-v2-run-consumption-claim.v1"
ADMISSION_SCHEMA = "agu.task0258-module-a-v2-run-admission.v1"
COMPLETION_SCHEMA = "agu.task0258-module-a-v2-run-consumption-completed.v1"
MODULE_ID = "existing-45-temporal-retrospective"

MAX_SEQUENCE_ORDINAL = 10
MAX_MARKER_COUNT = 10

# The exact 14 legal history events, in spec order.
HISTORY_EVENTS: tuple[str, ...] = (
    "producer_attempt_1_admitted",
    "producer_attempt_1_recoverable_published",
    "producer_attempt_1_completed_private",
    "producer_attempt_2_admitted",
    "producer_attempt_2_recoverable_published",
    "producer_attempt_2_completed_private",
    "producer_attempt_3_admitted",
    "producer_attempt_3_completed_private",
    "verification_attempt_admitted",
    "verification_attempt_completed_private",
    "pre_candidate_failure_published",
    "candidate_published",
    "verified_result_published",
    "postverification_failure_published",
)

# Terminal events have no legal successor.
_TERMINAL_EVENTS = frozenset(
    {
        "pre_candidate_failure_published",
        "verified_result_published",
        "postverification_failure_published",
    }
)

# The exact transition graph. The completed ledger starts the chain; the
# "completed" node is a synthetic predecessor (the <AUTH_SHA>.completed.json
# file) rather than a history event.
TRANSITIONS: dict[str, frozenset[str]] = {
    "producer_attempt_1_admitted": frozenset(
        {
            "producer_attempt_1_recoverable_published",
            "producer_attempt_1_completed_private",
            "pre_candidate_failure_published",
        }
    ),
    "producer_attempt_1_recoverable_published": frozenset(
        {
            "producer_attempt_2_admitted",
        }
    ),
    "producer_attempt_1_completed_private": frozenset(
        {
            "verification_attempt_admitted",
            "pre_candidate_failure_published",
        }
    ),
    "producer_attempt_2_admitted": frozenset(
        {
            "producer_attempt_2_recoverable_published",
            "producer_attempt_2_completed_private",
            "pre_candidate_failure_published",
        }
    ),
    "producer_attempt_2_recoverable_published": frozenset(
        {
            "producer_attempt_3_admitted",
        }
    ),
    "producer_attempt_2_completed_private": frozenset(
        {
            "verification_attempt_admitted",
            "pre_candidate_failure_published",
        }
    ),
    "producer_attempt_3_admitted": frozenset(
        {
            "producer_attempt_3_completed_private",
            "pre_candidate_failure_published",
        }
    ),
    "producer_attempt_3_completed_private": frozenset(
        {
            "verification_attempt_admitted",
            "pre_candidate_failure_published",
        }
    ),
    "verification_attempt_admitted": frozenset(
        {
            "verification_attempt_completed_private",
            "pre_candidate_failure_published",
        }
    ),
    "verification_attempt_completed_private": frozenset(
        {
            "candidate_published",
            "pre_candidate_failure_published",
        }
    ),
    "candidate_published": frozenset(
        {
            "verified_result_published",
            "postverification_failure_published",
        }
    ),
}

_MARKER_FILENAME_RE = re.compile(r"^(?P<auth>[0-9a-f]{64})\.history-(?P<nn>[0-9]{2})-(?P<event>[a-z0-9_]+)\.json$")


def is_history_event(event: str) -> bool:
    """Return True when ``event`` is one of the 14 legal history events."""
    return event in HISTORY_EVENTS


def is_terminal_event(event: str) -> bool:
    """Return True when ``event`` has no legal successor."""
    return event in _TERMINAL_EVENTS


def legal_successors(event: str) -> frozenset[str]:
    """Return the exact legal successor events for ``event``.

    A terminal event has no successors. ``event`` must itself be a legal history
    event; the synthetic "completed" predecessor is not accepted here.
    """
    if not is_history_event(event):
        raise ValueError(f"unknown history event: {event!r}")
    return TRANSITIONS.get(event, frozenset())


def verify_history_transition(prior_event: str, next_event: str) -> None:
    """Raise ``ValueError`` when ``next_event`` is not a legal successor.

    ``prior_event`` is a legal history event; the transition from the completed
    ledger to ``producer_attempt_1_admitted`` is validated separately by
    :func:`verify_completed_successor`.
    """
    if not is_history_event(next_event):
        raise ValueError(f"unknown history event: {next_event!r}")
    if next_event not in legal_successors(prior_event):
        raise ValueError(f"illegal history transition {prior_event!r} -> {next_event!r}")


def verify_completed_successor(event: str) -> None:
    """Validate the completed ledger's single legal successor."""
    if event != "producer_attempt_1_admitted":
        raise ValueError(f"completed ledger may only transition to producer_attempt_1_admitted, got {event!r}")


def parse_history_filename(filename: str, auth_sha256: str) -> tuple[int, str]:
    """Parse a marker basename and return ``(sequence_ordinal, event)``.

    The basename must match ``<AUTH_SHA>.history-<NN>-<EVENT>.json`` with the
    exact lowercase rerun-authorization internal SHA-256, ``NN`` exactly two
    decimal digits in ``[01, 10]``, and a legal event. Raises ``ValueError``
    otherwise.
    """
    m = _MARKER_FILENAME_RE.match(filename)
    if m is None:
        raise ValueError(f"invalid history marker basename: {filename!r}")
    if m.group("auth") != auth_sha256:
        raise ValueError(f"history marker basename auth mismatch: {filename!r}")
    nn = int(m.group("nn"), 10)
    if not (1 <= nn <= MAX_SEQUENCE_ORDINAL):
        raise ValueError(f"history marker ordinal out of range: {filename!r}")
    event = m.group("event")
    if not is_history_event(event):
        raise ValueError(f"history marker event invalid: {filename!r}")
    return nn, event


def verify_marker_scalars(payload: Mapping[str, object]) -> None:
    """Validate the exact marker scalar fields (schema, id, ordinal, event).

    Full subject-receipt/root-CAS/projection validation is added in a later
    phase; this enforces the fields the spec pins as exact at §2783-2810.
    """
    if payload.get("schema_version") != MARKER_SCHEMA:
        raise ValueError("marker schema_version mismatch")
    if payload.get("module_id") != MODULE_ID:
        raise ValueError("marker module_id mismatch")
    seq = payload.get("sequence_ordinal")
    if not isinstance(seq, int) or isinstance(seq, bool) or not (1 <= seq <= MAX_SEQUENCE_ORDINAL):
        raise ValueError("marker sequence_ordinal invalid")
    event = payload.get("event")
    if not isinstance(event, str) or not is_history_event(event):
        raise ValueError("marker event invalid")


def claim_filename(auth_sha256: str) -> str:
    """Return the exact claim basename ``<AUTH_SHA>.claim.json``."""
    return f"{auth_sha256}.claim.json"


def completion_filename(auth_sha256: str) -> str:
    """Return the exact completion basename ``<AUTH_SHA>.completed.json``."""
    return f"{auth_sha256}.completed.json"


def history_filename(auth_sha256: str, sequence_ordinal: int, event: str) -> str:
    """Return the exact marker basename ``<AUTH_SHA>.history-<NN>-<EVENT>.json``."""
    if not (1 <= sequence_ordinal <= MAX_SEQUENCE_ORDINAL):
        raise ValueError("history marker ordinal out of range")
    if not is_history_event(event):
        raise ValueError(f"unknown history event: {event!r}")
    return f"{auth_sha256}.history-{sequence_ordinal:02d}-{event}.json"


def verify_registry_listing(
    filenames: object,
    auth_sha256: str,
) -> None:
    """Validate the complete ordered registry listing.

    The order must be exactly ``claim, completion,`` then the history markers in
    contiguous ``NN`` order (01..N, no gaps), each with the exact auth SHA and a
    legal event. Missing, duplicate, extra, reordered, or unknown basenames fail.
    """
    if not isinstance(filenames, (list, tuple)):
        raise ValueError("registry listing must be a list")
    if not is_sha256(auth_sha256):
        raise ValueError("registry auth SHA is invalid")
    if len(filenames) < 2:
        raise ValueError("registry listing must include claim and completion")
    if filenames[0] != claim_filename(auth_sha256):
        raise ValueError("registry listing first entry is not the claim")
    if filenames[1] != completion_filename(auth_sha256):
        raise ValueError("registry listing second entry is not the completion")
    seen = {filenames[0], filenames[1]}
    prev_nn = 0
    for name in filenames[2:]:
        if name in seen:
            raise ValueError(f"registry listing duplicate: {name!r}")
        seen.add(name)
        nn, event = parse_history_filename(name, auth_sha256)
        if nn != prev_nn + 1:
            raise ValueError(f"registry history ordinal gap at {name!r}")
        prev_nn = nn
    events = [parse_history_filename(name, auth_sha256)[1] for name in filenames[2:]]
    if events:
        verify_completed_successor(events[0])
        for prior, current in zip(events, events[1:]):
            verify_history_transition(prior, current)


def registry_history_filenames(registry_dir: Path, auth_sha256: str) -> list[str]:
    """Return the JSON registry listing in the canonical replay order."""
    if not registry_dir.is_dir() or registry_dir.is_symlink():
        raise ValueError("registry directory must be a real directory")
    filenames = [entry.name for entry in registry_dir.iterdir() if entry.name.endswith(".json")]
    if not filenames:
        raise ValueError("registry directory is empty")
    ordered = [claim_filename(auth_sha256), completion_filename(auth_sha256)]
    if any(required not in filenames for required in ordered):
        raise ValueError("registry is missing claim or completion ledger")
    markers = sorted(name for name in filenames if ".history-" in name)
    unexpected = set(filenames) - set(ordered) - set(markers)
    if unexpected:
        raise ValueError(f"registry contains unexpected JSON files: {sorted(unexpected)!r}")
    ordered.extend(markers)
    verify_registry_listing(ordered, auth_sha256)
    return ordered


_MAX_REGISTRY_MEMBER_BYTES = 16_777_216


def _registry_history_filenames_from_fd(directory_fd: int, auth_sha256: str) -> list[str]:
    """Return the stable registry listing from one already-open directory FD."""
    try:
        names = os.listdir(directory_fd)
    except OSError as exc:
        raise ValueError("registry directory cannot be listed through its descriptor") from exc
    allowed_lock_name = f".{auth_sha256}.history.lock"
    if any(not isinstance(name, str) or (not name.endswith(".json") and name != allowed_lock_name) for name in names):
        raise ValueError("registry contains non-JSON residue")
    ordered = [claim_filename(auth_sha256), completion_filename(auth_sha256)]
    markers = sorted(name for name in names if ".history-" in name)
    unexpected = set(names) - set(ordered) - set(markers) - {allowed_lock_name}
    if unexpected:
        raise ValueError(f"registry contains unexpected files: {sorted(unexpected)!r}")
    if any(required not in names for required in ordered):
        raise ValueError("registry is missing claim or completion ledger")
    ordered.extend(markers)
    verify_registry_listing(ordered, auth_sha256)
    return ordered


def _read_registry_member_from_fd(directory_fd: int, filename: str) -> bytes:
    """Read one bounded regular registry member without following the leaf link."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    try:
        member_fd = os.open(filename, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise ValueError(f"registry member cannot be opened without following links: {filename}") from exc
    try:
        member_stat = os.fstat(member_fd)
        if not stat.S_ISREG(member_stat.st_mode) or member_stat.st_size > _MAX_REGISTRY_MEMBER_BYTES:
            raise ValueError(f"registry member is not a bounded regular file: {filename}")
        remaining = member_stat.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(member_fd, min(1 << 20, remaining))
            if not chunk:
                raise ValueError(f"registry member ended before its recorded size: {filename}")
            chunks.append(chunk)
            remaining -= len(chunk)
        post_stat = os.fstat(member_fd)
        snapshot = (
            member_stat.st_dev,
            member_stat.st_ino,
            member_stat.st_size,
            getattr(member_stat, "st_mtime_ns", 0),
            getattr(member_stat, "st_ctime_ns", 0),
        )
        post_snapshot = (
            post_stat.st_dev,
            post_stat.st_ino,
            post_stat.st_size,
            getattr(post_stat, "st_mtime_ns", 0),
            getattr(post_stat, "st_ctime_ns", 0),
        )
        if post_snapshot != snapshot:
            raise ValueError(f"registry member changed during bounded read: {filename}")
        return b"".join(chunks)
    except OSError as exc:
        raise ValueError(f"registry member cannot be read: {filename}") from exc
    finally:
        os.close(member_fd)


def replay_run_history_registry(
    registry_dir: Path,
    auth_sha256: str,
    *,
    held_lock: FlockHandle | None = None,
    registry_fd: int | None = None,
) -> list[Mapping[str, object]]:
    """Load and replay every durable ledger/marker binding in order.

    The replay runs under the registry's fixed history lock by default. Writers
    that already hold that lock pass its unforgeable ``FlockHandle`` to avoid
    reacquiring the same advisory lock through a second file descriptor.
    """
    lock_path = Path(registry_dir) / f".{auth_sha256}.history.lock"
    if held_lock is None:
        held_lock = active_flock(lock_path)
        if held_lock is None:
            try:
                owned_directory_fd = _open_existing_directory_no_follow(Path(registry_dir))
            except OSError as exc:
                raise ValueError("registry directory cannot be opened without following links") from exc
            try:
                with exclusive_flock_at(owned_directory_fd, lock_path.name, lock_path) as acquired_lock:
                    return replay_run_history_registry(
                        registry_dir,
                        auth_sha256,
                        held_lock=acquired_lock,
                        registry_fd=owned_directory_fd,
                    )
            finally:
                os.close(owned_directory_fd)
    if registry_fd is not None and not isinstance(registry_fd, int):
        raise ValueError("registry directory descriptor is invalid")
    owns_directory_fd = registry_fd is None
    if owns_directory_fd:
        try:
            registry_fd = _open_existing_directory_no_follow(Path(registry_dir))
        except OSError as exc:
            raise ValueError("registry directory cannot be opened without following links") from exc
    try:
        assert registry_fd is not None
        held_lock.assert_held(lock_path, directory_fd=registry_fd)
        directory_stat = os.fstat(registry_fd)
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise ValueError("registry path is not a directory")
        filenames = _registry_history_filenames_from_fd(registry_fd, auth_sha256)
        payloads: list[Mapping[str, object]] = []
        for filename in filenames:
            raw = _read_registry_member_from_fd(registry_fd, filename)
            if not raw.endswith(b"\n"):
                raise ValueError(f"registry member is missing its final LF: {filename}")
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"registry member is not canonical JSON: {filename}") from exc
            if not isinstance(payload, Mapping) or raw != (compact_canonical_json(payload) + "\n").encode("utf-8"):
                raise ValueError(f"registry member bytes are not canonical: {filename}")
            verify_internal_artifact_hash(payload)
            if filename.endswith(".claim.json"):
                verify_run_consumption_claim(payload)
            elif filename.endswith(".completed.json"):
                verify_run_consumption_completed(payload)
            else:
                verify_run_history_marker(payload)
                ordinal, event = parse_history_filename(filename, auth_sha256)
                if payload["sequence_ordinal"] != ordinal or payload["event"] != event:
                    raise ValueError(f"registry marker payload does not match its basename: {filename}")
            payloads.append(payload)
    finally:
        if owns_directory_fd:
            os.close(registry_fd)
    claim = payloads[0]
    completion = payloads[1]
    auth_receipt = claim["authorization_receipt"]
    if not isinstance(auth_receipt, Mapping) or auth_receipt["artifact_sha256"] != auth_sha256:
        raise ValueError("registry claim authorization is not bound to the registry name")
    for payload in (completion,):
        if payload["authorization_receipt"] != auth_receipt:
            raise ValueError("registry completion authorization binding drifted")
        for field in ("run_id", "nonce", "output_root_absolute_path"):
            claim_field = "output_root_absolute_path" if field == "output_root_absolute_path" else field
            if payload[field] != claim[claim_field]:
                raise ValueError(f"registry completion {field} binding drifted")
    completion_receipt = _file_receipt(completion)
    previous_receipt = completion_receipt
    previous_event = None
    for marker in payloads[2:]:
        if marker["authorization_receipt"] != auth_receipt:
            raise ValueError("registry marker authorization binding drifted")
        if marker["run_identity_receipt"] != completion_receipt:
            raise ValueError("registry marker run identity binding drifted")
        for field in ("run_id", "nonce", "output_root"):
            claim_field = "output_root_absolute_path" if field == "output_root" else field
            if marker[field] != claim[claim_field]:
                raise ValueError(f"registry marker {field} binding drifted")
        if marker["prior_marker_receipt"] != previous_receipt:
            raise ValueError("registry marker prior-head receipt does not replay")
        event = marker["event"]
        if previous_event is None:
            verify_completed_successor(event)
        else:
            verify_history_transition(previous_event, event)
        previous_event = event
        previous_receipt = _file_receipt(marker)
    return payloads


def _file_receipt(payload: Mapping[str, object]) -> dict[str, str]:
    data = (compact_canonical_json(payload) + "\n").encode("utf-8")
    return {"artifact_sha256": str(payload["artifact_sha256"]), "file_sha256": hashlib.sha256(data).hexdigest()}


def verify_history_subject_receipt(row: object) -> None:
    """Validate an exact ``HistorySubjectReceipt`` row.

    JSON row: ``{provider, receipt_kind=json, artifact_sha256, file_sha256}``.
    JSONL row: ``{provider, receipt_kind=file_only, artifact_sha256=null,
    file_sha256}``. Provider is a nonempty safe slug. No other nullability or
    kind is accepted.
    """
    if not isinstance(row, Mapping) or set(row) != {
        "provider",
        "receipt_kind",
        "artifact_sha256",
        "file_sha256",
    }:
        raise ValueError("HistorySubjectReceipt shape is invalid")
    provider = row["provider"]
    if not is_safe_slug(provider):
        raise ValueError("HistorySubjectReceipt provider is invalid")
    if not is_sha256(row["file_sha256"]):
        raise ValueError("HistorySubjectReceipt file_sha256 is invalid")
    kind = row["receipt_kind"]
    if kind == "json":
        if not is_sha256(row["artifact_sha256"]):
            raise ValueError("HistorySubjectReceipt json artifact_sha256 is invalid")
    elif kind == "file_only":
        if row["artifact_sha256"] is not None:
            raise ValueError("HistorySubjectReceipt file_only artifact_sha256 must be null")
    else:
        raise ValueError("HistorySubjectReceipt receipt_kind is invalid")


# Exact marker top-level field set (amendment §"Exact run-history registry
# encoding", lines 2616-2622).
MARKER_FIELDS = frozenset(
    {
        "schema_version",
        "module_id",
        "authorization_receipt",
        "run_identity_receipt",
        "run_id",
        "output_root",
        "nonce",
        "sequence_ordinal",
        "prior_marker_receipt",
        "event",
        "subject_kind",
        "subject_receipts",
        "subject_projection_sha256",
        "root_subject_cas",
        "worker_launch_claim",
        "cumulative_active_runtime_nanoseconds",
        "cumulative_resource_samples",
        "cumulative_resource_log_bytes",
        "created_at_utc",
        "artifact_sha256",
    }
)

SUBJECT_KINDS: tuple[str, ...] = (
    "worker_launch_admission",
    "completed_private",
    "published_paths",
)


CLAIM_FIELDS = frozenset(
    {
        "schema_version",
        "module_id",
        "authorization_receipt",
        "run_id",
        "output_root_absolute_path",
        "nonce",
        "state",
        "created_at_utc",
        "artifact_sha256",
    }
)

ADMISSION_FIELDS = frozenset(
    {
        "schema_version",
        "module_id",
        "authorization_receipt",
        "claim_receipt",
        "nonce",
        "run_id",
        "output_root_absolute_path",
        "static_input_contract",
        "maximum_run_count",
        "admission_state",
        "module_b_authorized",
        "created_at_utc",
        "artifact_sha256",
    }
)

COMPLETION_FIELDS = frozenset(
    {
        "schema_version",
        "module_id",
        "authorization_receipt",
        "claim_receipt",
        "admission_receipt",
        "nonce",
        "run_id",
        "output_root_absolute_path",
        "consumption_count",
        "state",
        "root_identity",
        "admission_identity",
        "created_at_utc",
        "artifact_sha256",
    }
)


def _verify_nonnegative_int(value: object, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} is invalid")


def verify_run_consumption_claim(payload: Mapping[str, object]) -> None:
    """Validate a `agu.task0258-module-a-v2-run-consumption-claim.v1` payload."""
    if not isinstance(payload, Mapping) or set(payload) != CLAIM_FIELDS:
        raise ValueError("claim field set is invalid")
    verify_internal_artifact_hash(payload)
    if payload["schema_version"] != CLAIM_SCHEMA or payload["module_id"] != MODULE_ID:
        raise ValueError("claim identity is invalid")
    verify_artifact_file_receipt(payload["authorization_receipt"])
    if payload["state"] != "claimed":
        raise ValueError("claim state must be claimed")
    if not is_sha256(payload["nonce"]) or not is_safe_slug(payload["run_id"]):
        raise ValueError("claim nonce/run_id is invalid")
    if not isinstance(payload["output_root_absolute_path"], str) or not payload["output_root_absolute_path"].startswith(
        "/"
    ):
        raise ValueError("claim output_root_absolute_path is invalid")
    if not is_rfc3339(payload["created_at_utc"]):
        raise ValueError("claim created_at_utc is invalid")


def verify_run_admission(payload: Mapping[str, object]) -> None:
    """Validate a `agu.task0258-module-a-v2-run-admission.v1` payload."""
    if not isinstance(payload, Mapping) or set(payload) != ADMISSION_FIELDS:
        raise ValueError("admission field set is invalid")
    verify_internal_artifact_hash(payload)
    if payload["schema_version"] != ADMISSION_SCHEMA or payload["module_id"] != MODULE_ID:
        raise ValueError("admission identity is invalid")
    verify_artifact_file_receipt(payload["authorization_receipt"])
    verify_artifact_file_receipt(payload["claim_receipt"])
    if payload["maximum_run_count"] != 1 or payload["admission_state"] != "admitted":
        raise ValueError("admission state/count is invalid")
    if payload["module_b_authorized"] is not False:
        raise ValueError("admission module_b_authorized must be False")
    if not is_sha256(payload["nonce"]) or not is_safe_slug(payload["run_id"]):
        raise ValueError("admission nonce/run_id is invalid")
    if not isinstance(payload["output_root_absolute_path"], str) or not payload["output_root_absolute_path"].startswith(
        "/"
    ):
        raise ValueError("admission output_root_absolute_path is invalid")
    if not is_rfc3339(payload["created_at_utc"]):
        raise ValueError("admission created_at_utc is invalid")
    verify_static_input_contract(payload["static_input_contract"])


def verify_run_consumption_completed(payload: Mapping[str, object]) -> None:
    """Validate a `agu.task0258-module-a-v2-run-consumption-completed.v1` payload."""
    if not isinstance(payload, Mapping) or set(payload) != COMPLETION_FIELDS:
        raise ValueError("completion field set is invalid")
    verify_internal_artifact_hash(payload)
    if payload["schema_version"] != COMPLETION_SCHEMA or payload["module_id"] != MODULE_ID:
        raise ValueError("completion identity is invalid")
    verify_artifact_file_receipt(payload["authorization_receipt"])
    verify_artifact_file_receipt(payload["claim_receipt"])
    verify_artifact_file_receipt(payload["admission_receipt"])
    if payload["consumption_count"] != 1 or payload["state"] != "completed":
        raise ValueError("completion count/state is invalid")
    if not is_sha256(payload["nonce"]) or not is_safe_slug(payload["run_id"]):
        raise ValueError("completion nonce/run_id is invalid")
    if not isinstance(payload["output_root_absolute_path"], str) or not payload["output_root_absolute_path"].startswith(
        "/"
    ):
        raise ValueError("completion output_root_absolute_path is invalid")
    if not is_rfc3339(payload["created_at_utc"]):
        raise ValueError("completion created_at_utc is invalid")
    root_identity = payload["root_identity"]
    if not isinstance(root_identity, Mapping) or set(root_identity) != {"device", "inode"}:
        raise ValueError("completion root_identity is invalid")
    _verify_nonnegative_int(root_identity["device"], "root device")
    _verify_nonnegative_int(root_identity["inode"], "root inode")
    admission_identity = payload["admission_identity"]
    if not isinstance(admission_identity, Mapping) or set(admission_identity) != {
        "device",
        "inode",
        "size_bytes",
        "internal_sha256",
        "file_sha256",
    }:
        raise ValueError("completion admission_identity is invalid")
    _verify_nonnegative_int(admission_identity["device"], "admission device")
    _verify_nonnegative_int(admission_identity["inode"], "admission inode")
    _verify_nonnegative_int(admission_identity["size_bytes"], "admission size")
    if not is_sha256(admission_identity["internal_sha256"]) or not is_sha256(admission_identity["file_sha256"]):
        raise ValueError("completion admission_identity hashes are invalid")


WORKER_LAUNCH_CLAIM_FIELDS = frozenset(
    {
        "worker_role",
        "parent_pid",
        "child_nonce",
        "launch_secret_sha256",
        "bootstrap_source_sha256",
        "worker_request_artifact_sha256",
        "provider_process_instance_id",
        "audit_prepared_artifact_sha256",
        "expected_hash_state_projection_sha256",
        "private_stage_identity",
        "source_fd",
        "liveness_fd",
        "result_fd",
        "admitted_marker_fd",
        "audit_start_gate_fd",
        "claim_envelope_fd",
    }
)

_EXACT_LAUNCH_FDS = {
    "source_fd": 4,
    "liveness_fd": 5,
    "result_fd": 6,
    "admitted_marker_fd": 7,
    "audit_start_gate_fd": 8,
    "claim_envelope_fd": 9,
}


def verify_worker_launch_claim(value: object) -> None:
    """Validate the exact one-shot ``worker_launch_claim`` object.

    Exactly the 16 fields from amendment §"Exact run-history registry encoding"
    (lines 2635-2645): role/nonce/hash scalars, ``private_stage_identity``
    ``{device,inode}``, and the six pinned fds 4..9.
    """
    if not isinstance(value, Mapping) or set(value) != WORKER_LAUNCH_CLAIM_FIELDS:
        raise ValueError("worker_launch_claim shape is invalid")
    if not is_safe_slug(value["worker_role"]):
        raise ValueError("worker_launch_claim worker_role is invalid")
    if not is_sha256(value["child_nonce"]):
        raise ValueError("worker_launch_claim child_nonce is invalid")
    for field in (
        "launch_secret_sha256",
        "bootstrap_source_sha256",
        "worker_request_artifact_sha256",
        "audit_prepared_artifact_sha256",
        "expected_hash_state_projection_sha256",
    ):
        if not is_sha256(value[field]):
            raise ValueError(f"worker_launch_claim {field} is invalid")
    for field in ("parent_pid",):
        v = value[field]
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise ValueError(f"worker_launch_claim {field} is invalid")
    if not is_sha256(value["provider_process_instance_id"]):
        raise ValueError("worker_launch_claim provider_process_instance_id is invalid")
    stage = value["private_stage_identity"]
    if not isinstance(stage, Mapping) or set(stage) != {"device", "inode"}:
        raise ValueError("worker_launch_claim private_stage_identity is invalid")
    for field, expected_fd in _EXACT_LAUNCH_FDS.items():
        if value[field] != expected_fd:
            raise ValueError(f"worker_launch_claim {field} must be {expected_fd}")


def verify_root_subject_cas_row(row: object) -> None:
    """Validate an exact ``root_subject_cas`` row.

    ``{relative_path, entry_kind, device, inode, size_bytes, file_sha256,
    artifact_sha256}`` with ``entry_kind`` ``directory`` | ``json_file`` |
    ``jsonl_file``. Directory rows have null size/hashes; json rows have all
    non-null; jsonl rows have size/file non-null and artifact null.
    """
    if not isinstance(row, Mapping) or set(row) != {
        "relative_path",
        "entry_kind",
        "device",
        "inode",
        "size_bytes",
        "file_sha256",
        "artifact_sha256",
    }:
        raise ValueError("root_subject_cas row shape is invalid")
    if not isinstance(row["relative_path"], str) or not row["relative_path"]:
        raise ValueError("root_subject_cas relative_path is invalid")
    rel = PurePosixPath(row["relative_path"])
    if "\\" in row["relative_path"] or rel.is_absolute() or any(part in ("", ".", "..") for part in rel.parts):
        raise ValueError("root_subject_cas relative_path must be safe and POSIX-relative")
    for field in ("device", "inode"):
        value = row[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"root_subject_cas {field} is invalid")
    kind = row["entry_kind"]
    if kind == "directory":
        if row["size_bytes"] is not None or row["file_sha256"] is not None or row["artifact_sha256"] is not None:
            raise ValueError("root_subject_cas directory row nullability is invalid")
    elif kind == "json_file":
        if not isinstance(row["size_bytes"], int) or isinstance(row["size_bytes"], bool) or row["size_bytes"] < 0:
            raise ValueError("root_subject_cas json_file size is invalid")
        if not is_sha256(row["file_sha256"]) or not is_sha256(row["artifact_sha256"]):
            raise ValueError("root_subject_cas json_file hashes are invalid")
    elif kind == "jsonl_file":
        if not isinstance(row["size_bytes"], int) or isinstance(row["size_bytes"], bool) or row["size_bytes"] < 0:
            raise ValueError("root_subject_cas jsonl_file size is invalid")
        if not is_sha256(row["file_sha256"]) or row["artifact_sha256"] is not None:
            raise ValueError("root_subject_cas jsonl_file nullability/hashes are invalid")
    else:
        raise ValueError("root_subject_cas entry_kind is invalid")


def verify_run_history_marker(payload: Mapping[str, object]) -> None:
    """Validate an exact run-history marker payload.

    Enforces the exact 20-field set, marker scalars (schema/id/ordinal/event),
    the three closed subject kinds, cumulative-counter invariants, subject-kind
    nullability, and root-subject-CAS row shapes. Deep ``worker_launch_claim``
    validation is layered on in a later phase.
    """
    if not isinstance(payload, Mapping) or set(payload) != MARKER_FIELDS:
        raise ValueError("marker field set is invalid")
    verify_internal_artifact_hash(payload)
    verify_marker_scalars(payload)
    verify_artifact_file_receipt(payload["authorization_receipt"])
    verify_artifact_file_receipt(payload["run_identity_receipt"])
    if not is_safe_slug(payload["run_id"]):
        raise ValueError("marker run_id invalid")
    if not isinstance(payload["output_root"], str) or not payload["output_root"].startswith("/"):
        raise ValueError("marker output_root invalid")
    if not is_sha256(payload["nonce"]):
        raise ValueError("marker nonce invalid")
    verify_artifact_file_receipt(payload["prior_marker_receipt"])
    subject_kind = payload["subject_kind"]
    if subject_kind not in SUBJECT_KINDS:
        raise ValueError("marker subject_kind is invalid")
    receipts = payload["subject_receipts"]
    if not isinstance(receipts, (list, tuple)):
        raise ValueError("marker subject_receipts must be a list")
    providers = [row["provider"] for row in receipts if isinstance(row, Mapping)]
    if len(set(providers)) != len(providers):
        raise ValueError("marker subject_receipts providers must be unique")
    if providers != sorted(providers):
        raise ValueError("marker subject_receipts rows must be sorted by provider")
    for row in receipts:
        verify_history_subject_receipt(row)
    root_cas = payload["root_subject_cas"]
    if not isinstance(root_cas, (list, tuple)) or not root_cas:
        raise ValueError("marker root_subject_cas must be a non-empty list")
    for row in root_cas:
        verify_root_subject_cas_row(row)
    if [row["relative_path"] for row in root_cas] != sorted(row["relative_path"] for row in root_cas):
        raise ValueError("marker root_subject_cas rows must be sorted by relative_path")
    projection = payload["subject_projection_sha256"]
    if projection is not None and not is_sha256(projection):
        raise ValueError("marker subject_projection_sha256 is invalid")
    launch_claim = payload["worker_launch_claim"]
    if subject_kind == "worker_launch_admission":
        if receipts or projection is not None or launch_claim is None:
            raise ValueError("worker_launch_admission marker nullability is invalid")
        if len(root_cas) != 1:
            raise ValueError("worker_launch_admission marker must have exactly one root CAS row")
        verify_worker_launch_claim(launch_claim)
    else:
        if launch_claim is not None:
            raise ValueError("non-admission marker worker_launch_claim must be null")
        if subject_kind == "completed_private":
            if not receipts or projection is None:
                raise ValueError("completed_private marker nullability is invalid")
        else:
            if projection is not None:
                raise ValueError("published_paths marker projection must be null")
            if not receipts:
                raise ValueError("published_paths marker must have non-empty subject_receipts")
    for field in (
        "cumulative_active_runtime_nanoseconds",
        "cumulative_resource_samples",
        "cumulative_resource_log_bytes",
    ):
        value = payload[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"marker {field} is invalid")
    if not is_rfc3339(payload["created_at_utc"]):
        raise ValueError("marker created_at_utc is invalid")


__all__ = [
    "MARKER_SCHEMA",
    "CLAIM_SCHEMA",
    "ADMISSION_SCHEMA",
    "COMPLETION_SCHEMA",
    "MODULE_ID",
    "MAX_SEQUENCE_ORDINAL",
    "MAX_MARKER_COUNT",
    "HISTORY_EVENTS",
    "TRANSITIONS",
    "MARKER_FIELDS",
    "SUBJECT_KINDS",
    "CLAIM_FIELDS",
    "ADMISSION_FIELDS",
    "COMPLETION_FIELDS",
    "is_history_event",
    "is_terminal_event",
    "legal_successors",
    "verify_history_transition",
    "verify_completed_successor",
    "parse_history_filename",
    "verify_marker_scalars",
    "claim_filename",
    "completion_filename",
    "history_filename",
    "verify_registry_listing",
    "registry_history_filenames",
    "replay_run_history_registry",
    "verify_history_subject_receipt",
    "verify_root_subject_cas_row",
    "verify_worker_launch_claim",
    "verify_run_history_marker",
    "verify_run_consumption_claim",
    "verify_run_admission",
    "verify_run_consumption_completed",
]
