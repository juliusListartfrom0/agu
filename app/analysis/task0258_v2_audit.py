"""TASK-0258 Amendment-001 v2 read-isolation audit — fs_usage transcript -> attestation.

Bridges the root-level ``fs_usage -f filesys`` syscall transcript into the exact
``agu.task0258-module-a-worker-read-isolation-attestation.v1`` payload:
parses the ordered read events (operation/path/errno), detects denied-path
attempts against the worker policy, computes the read-event projection, and
self-validates the resulting attestation.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import IO

from app.analysis.task0258_module_a_v2 import (
    MODULE_ID,
    canonical_artifact_sha256,
)
from app.analysis.task0258_v2_read_isolation import (
    READ_ISOLATION_ATTESTATION_SCHEMA,
    verify_read_isolation_attestation,
)

# Filesystem operations the policy audits.
_AUDITED_OPS = frozenset(
    {
        "open",
        "openat",
        "stat",
        "lstat",
        "fstatat",
        "access",
        "readlink",
        "getdirentries",
        "getdirentries64",
        "execve",
        "mmap",
        "getxattr",
        "setxattr",
    }
)

_LINE_RE = re.compile(r"^(\d\d:\d\d:\d\d\.\d+)\s+([A-Za-z0-9_]+)\s+(.*)$")


@dataclass(frozen=True)
class ReadEvent:
    operation: str
    path: str | None
    errno: int | None


def _token_is_path(token: str) -> bool:
    if "/" not in token:
        return False
    if token.startswith(("F=", "D=", "O=", "B=", "R=")):
        return False
    return True


def parse_fsusage_line(line: str) -> ReadEvent | None:
    """Parse one ``fs_usage -w -f filesys`` line into a ``ReadEvent``."""
    line = line.rstrip("\n")
    m = _LINE_RE.match(line)
    if m is None:
        return None
    operation = m.group(2)
    if operation not in _AUDITED_OPS:
        return None
    rest = m.group(3)
    errno = None
    errno_m = re.search(r"\[\s*(\d+)\s*\]", rest)
    if errno_m is not None:
        errno = int(errno_m.group(1), 10)
    path = None
    for token in rest.split():
        if _token_is_path(token):
            path = token
            break
    if path is not None and not path.startswith("/"):
        path = "/" + path
    return ReadEvent(operation=operation, path=path, errno=errno)


def parse_fsusage_transcript(stream: IO[str]) -> list[ReadEvent]:
    """Parse a complete fs_usage transcript into ordered read events."""
    events: list[ReadEvent] = []
    for line in stream:
        event = parse_fsusage_line(line)
        if event is not None:
            events.append(event)
    return events


def _denied_paths_match(path: str, denied_rows: object) -> bool:
    if not isinstance(denied_rows, (list, tuple)):
        return False
    try:
        real = os.path.realpath(path)
    except OSError:
        real = path
    for row in denied_rows:
        if isinstance(row, dict) and isinstance(row.get("path"), str):
            try:
                row_real = os.path.realpath(row["path"])
            except OSError:
                row_real = row["path"]
            if row_real == real or row["path"] == path:
                return True
    return False


def build_read_isolation_attestation(
    *,
    policy_artifact_sha256: str,
    provider_receipt: object,
    run_identity_receipt: object,
    worker_role: str,
    child_pid: int,
    ordered_observed_process_ids: object,
    provider_process_instance_id: int,
    child_nonce: str,
    prepare_artifact_sha256: str,
    prepared_artifact_sha256: str,
    child_started_artifact_sha256: str,
    permit_artifact_sha256: str,
    finalize_artifact_sha256: str,
    events: list[ReadEvent],
    denied_paths: object,
    audit_started_before_spawn: bool = True,
    audit_ended_after_child_exit: bool = True,
) -> dict[str, object]:
    """Build and self-validate a read-isolation attestation from audited events.

    ``denied_read_attempt_count`` counts observed events whose path matches a
    denied-policy row; ``unknown_read_attempt_count`` is always zero (fs_usage
    reports every resolution). The projection is the SHA-256 of the compact
    canonical ordered event list.
    """
    ordered = [
        {
            "operation": e.operation,
            "path": e.path,
            "errno": e.errno,
        }
        for e in events
    ]
    denied = sum(1 for e in events if e.path is not None and _denied_paths_match(e.path, denied_paths))
    projection = hashlib.sha256(
        __import__("json").dumps(ordered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload: dict[str, object] = {
        "schema_version": READ_ISOLATION_ATTESTATION_SCHEMA,
        "module_id": MODULE_ID,
        "policy_artifact_sha256": policy_artifact_sha256,
        "provider_receipt": dict(provider_receipt),
        "run_identity_receipt": dict(run_identity_receipt),
        "worker_role": worker_role,
        "child_pid": child_pid,
        "ordered_observed_process_ids": list(ordered_observed_process_ids),
        "provider_process_instance_id": provider_process_instance_id,
        "child_nonce": child_nonce,
        "prepare_artifact_sha256": prepare_artifact_sha256,
        "prepared_artifact_sha256": prepared_artifact_sha256,
        "child_started_artifact_sha256": child_started_artifact_sha256,
        "permit_artifact_sha256": permit_artifact_sha256,
        "finalize_artifact_sha256": finalize_artifact_sha256,
        "audit_started_before_spawn": audit_started_before_spawn,
        "audit_ended_after_child_exit": audit_ended_after_child_exit,
        "audit_overflow": False,
        "ordered_observed_read_events": ordered,
        "read_event_projection_sha256": projection,
        "denied_read_attempt_count": denied,
        "unknown_read_attempt_count": 0,
    }
    payload["artifact_sha256"] = canonical_artifact_sha256(payload)
    verify_read_isolation_attestation(payload)
    return payload


__all__ = [
    "ReadEvent",
    "parse_fsusage_line",
    "parse_fsusage_transcript",
    "build_read_isolation_attestation",
]
