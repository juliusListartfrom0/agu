"""TASK-0258 Amendment-001 v2 read-isolation diagnostic parser.

Parses a root-level ``fs_usage -f filesys`` transcript for diagnostics. Raw
text rows and caller-supplied "verified" rows cannot mint the
``agu.task0258-module-a-worker-read-isolation-attestation.v1`` payload; that
requires a future externally authenticated kernel-audit provider.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import IO

from app.analysis.task0258_v2_read_isolation import (
    MAXIMUM_READ_EVENT_BYTES,
    MAXIMUM_READ_EVENT_ROWS,
)


class ExternalKernelAuditUnavailable(PermissionError):
    """Raised until an externally authenticated kernel-audit provider is bound."""


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


def parse_fsusage_transcript(
    stream: IO[str],
    *,
    maximum_rows: int = MAXIMUM_READ_EVENT_ROWS,
    maximum_bytes: int = MAXIMUM_READ_EVENT_BYTES,
) -> list[ReadEvent]:
    """Parse a bounded fs_usage transcript into ordered diagnostic events.

    The limits protect this diagnostic-only parser from unbounded input. They
    do not turn parsed text into authenticated kernel evidence.
    """
    if isinstance(maximum_rows, bool) or not isinstance(maximum_rows, int) or maximum_rows <= 0:
        raise ValueError("maximum_rows must be a positive integer")
    if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int) or maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be a positive integer")

    events: list[ReadEvent] = []
    total_bytes = 0
    for line in stream:
        try:
            total_bytes += len(line.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ValueError("fs_usage transcript must contain UTF-8 text") from exc
        if total_bytes > maximum_bytes:
            raise ValueError("fs_usage transcript exceeds byte cap")
        event = parse_fsusage_line(line)
        if event is not None:
            if len(events) >= maximum_rows:
                raise ValueError("fs_usage transcript exceeds event-row cap")
            events.append(event)
    return events


def build_read_isolation_attestation(
    *,
    policy_artifact_sha256: str,
    provider_receipt: object,
    run_identity_receipt: object,
    worker_role: str,
    child_pid: int,
    ordered_observed_process_ids: object,
    provider_process_instance_id: str,
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
    """Reject untrusted ``fs_usage`` rows instead of minting an attestation.

    ``fs_usage`` is retained as a diagnostic parser, but its lossy text rows do
    not contain the authenticated metadata required by the production schema.
    The future externally authenticated provider adapter is the only permitted
    source of an actual attestation.
    """
    if events:
        raise ValueError("raw fs_usage rows cannot mint a read-isolation attestation")
    raise ExternalKernelAuditUnavailable(
        "read-isolation attestation requires an externally authenticated kernel-audit provider"
    )


def build_verified_read_isolation_attestation(
    *,
    policy_artifact_sha256: str,
    provider_receipt: object,
    run_identity_receipt: object,
    worker_role: str,
    child_pid: int,
    ordered_observed_process_ids: object,
    provider_process_instance_id: str,
    child_nonce: str,
    prepare_artifact_sha256: str,
    prepared_artifact_sha256: str,
    child_started_artifact_sha256: str,
    permit_artifact_sha256: str,
    finalize_artifact_sha256: str,
    verified_event_rows: list[dict[str, object]],
    denied_paths: object,
    audit_started_before_spawn: bool = True,
    audit_ended_after_child_exit: bool = True,
) -> dict[str, object]:
    """Reject direct Python row injection until a trusted provider adapter exists.

    The parameters remain part of the planned adapter contract, but no caller
    supplied mapping or list is allowed to mint a production attestation. The
    future Endpoint Security adapter must provide an opaque, externally bound
    capability before this function can be implemented.
    """
    del (
        policy_artifact_sha256,
        provider_receipt,
        run_identity_receipt,
        worker_role,
        child_pid,
        ordered_observed_process_ids,
        provider_process_instance_id,
        child_nonce,
        prepare_artifact_sha256,
        prepared_artifact_sha256,
        child_started_artifact_sha256,
        permit_artifact_sha256,
        finalize_artifact_sha256,
        verified_event_rows,
        denied_paths,
        audit_started_before_spawn,
        audit_ended_after_child_exit,
    )
    raise ExternalKernelAuditUnavailable(
        "verified read-isolation rows require an externally authenticated kernel-audit capability"
    )


__all__ = [
    "ReadEvent",
    "ExternalKernelAuditUnavailable",
    "MAXIMUM_READ_EVENT_BYTES",
    "MAXIMUM_READ_EVENT_ROWS",
    "parse_fsusage_line",
    "parse_fsusage_transcript",
    "build_read_isolation_attestation",
    "build_verified_read_isolation_attestation",
]
