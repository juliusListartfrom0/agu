"""TASK-0258 Amendment-001 v2 read-isolation diagnostic parser.

Parses a root-level ``fs_usage -f filesys`` transcript for diagnostics. Raw
text rows and caller-supplied "verified" rows cannot mint the
``agu.task0258-module-a-worker-read-isolation-attestation.v1`` payload; that
requires a future externally authenticated kernel-audit provider.
"""

from __future__ import annotations

import json
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


@dataclass(frozen=True)
class EndpointSecurityEvent:
    """One strictly validated row from the diagnostic Endpoint Security JSONL."""

    event: str
    pid: int
    pidversion: int
    ppid: int
    seq_num: int | None
    global_seq_num: int | None
    path: str | None
    result_type: str
    result_auth: str | None
    result_flags: int | None


_ENDPOINT_SECURITY_EVENTS = frozenset(
    {"open", "stat", "access", "readlink", "getdents", "exec", "mmap", "getxattr", "setxattr", "fork"}
)
_ENDPOINT_SECURITY_BASE_FIELDS = frozenset(
    {"event", "pid", "pidversion", "ppid", "seq_num", "global_seq_num", "path", "result_type"}
)
_ENDPOINT_SECURITY_FINAL_FIELDS = frozenset(
    {
        "record_type",
        "rows",
        "bytes",
        "overflow",
        "sequence_gap",
        "protocol_error",
        "timed_out",
        "target_exit_observed",
        "interrupted",
    }
)
MAXIMUM_ENDPOINT_SECURITY_ROW_BYTES = 512


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


def _reject_duplicate_json_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"JSON constant is not allowed: {value}")


def _verify_json_integer(value: object, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} is invalid")
    return value


def _verify_optional_json_integer(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _verify_json_integer(value, name)


def _parse_endpoint_security_row(payload: object, line_number: int) -> EndpointSecurityEvent:
    if not isinstance(payload, dict):
        raise ValueError(f"line {line_number} must contain a JSON object")
    fields = set(payload)
    result_type = payload.get("result_type")
    if fields not in (
        _ENDPOINT_SECURITY_BASE_FIELDS | {"result_auth"},
        _ENDPOINT_SECURITY_BASE_FIELDS | {"result_flags"},
    ):
        raise ValueError(f"line {line_number} field set is invalid")
    if result_type not in {"auth", "flags"}:
        raise ValueError(f"line {line_number} result_type is invalid")

    event = payload["event"]
    if not isinstance(event, str) or event not in _ENDPOINT_SECURITY_EVENTS:
        raise ValueError(f"line {line_number} event is invalid")
    pid = _verify_json_integer(payload["pid"], f"line {line_number} pid", minimum=1)
    pidversion = _verify_json_integer(payload["pidversion"], f"line {line_number} pidversion")
    ppid = _verify_json_integer(payload["ppid"], f"line {line_number} ppid")
    seq_num = _verify_optional_json_integer(payload["seq_num"], f"line {line_number} seq_num")
    global_seq_num = _verify_optional_json_integer(payload["global_seq_num"], f"line {line_number} global_seq_num")
    path = payload["path"]
    if path is not None and (not isinstance(path, str) or not path.startswith("/") or "\x00" in path):
        raise ValueError(f"line {line_number} path is invalid")

    result_auth: str | None = None
    result_flags: int | None = None
    if result_type == "auth":
        result_auth = payload["result_auth"]
        if result_auth not in {"allow", "deny"}:
            raise ValueError(f"line {line_number} result_auth is invalid")
    else:
        result_flags = _verify_json_integer(payload["result_flags"], f"line {line_number} result_flags")

    return EndpointSecurityEvent(
        event=event,
        pid=pid,
        pidversion=pidversion,
        ppid=ppid,
        seq_num=seq_num,
        global_seq_num=global_seq_num,
        path=path,
        result_type=result_type,
        result_auth=result_auth,
        result_flags=result_flags,
    )


def parse_endpoint_security_transcript(
    stream: IO[str],
    *,
    maximum_rows: int = MAXIMUM_READ_EVENT_ROWS,
    maximum_bytes: int = MAXIMUM_READ_EVENT_BYTES,
) -> list[EndpointSecurityEvent]:
    """Parse a bounded Endpoint Security JSONL transcript for diagnostics.

    The C client performs sequence-gap checks before writing rows and appends a
    clean finalization row after observation. This parser repeats the
    retained-row ordering check, validates the exact result projection and
    finalization counts, but it never upgrades the transcript into a provider
    receipt or a read-isolation attestation.
    """
    if isinstance(maximum_rows, bool) or not isinstance(maximum_rows, int) or maximum_rows <= 0:
        raise ValueError("maximum_rows must be a positive integer")
    if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int) or maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be a positive integer")

    events: list[EndpointSecurityEvent] = []
    total_bytes = 0
    event_bytes = 0
    last_global_seq_num: int | None = None
    last_seq_num: dict[str, int] = {}
    finalization_seen = False
    for line_number, line in enumerate(_iter_bounded_transcript_lines(stream), start=1):
        try:
            line_bytes = len(line.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ValueError("Endpoint Security transcript must contain UTF-8 text") from exc
        if line_bytes > MAXIMUM_ENDPOINT_SECURITY_ROW_BYTES:
            raise ValueError("Endpoint Security transcript exceeds row byte cap")
        total_bytes += line_bytes
        if total_bytes > maximum_bytes:
            raise ValueError("Endpoint Security transcript exceeds byte cap")
        if finalization_seen:
            raise ValueError(f"line {line_number} appears after finalization")
        if not line.strip():
            raise ValueError(f"line {line_number} is empty")
        try:
            payload = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_json_fields,
                parse_constant=_reject_json_constant,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("duplicate JSON field"):
                raise
            raise ValueError(f"line {line_number} is invalid JSON") from exc
        if isinstance(payload, dict) and "record_type" in payload:
            if set(payload) != _ENDPOINT_SECURITY_FINAL_FIELDS:
                raise ValueError(f"line {line_number} finalization field set is invalid")
            if payload["record_type"] != "final":
                raise ValueError(f"line {line_number} finalization record type is invalid")
            final_rows = _verify_json_integer(payload["rows"], f"line {line_number} finalization rows")
            final_bytes = _verify_json_integer(payload["bytes"], f"line {line_number} finalization bytes")
            for field in (
                "overflow",
                "sequence_gap",
                "protocol_error",
                "timed_out",
                "interrupted",
            ):
                if not isinstance(payload[field], bool):
                    raise ValueError(f"line {line_number} finalization {field} is invalid")
                if payload[field] is not False:
                    raise ValueError(f"line {line_number} finalization is not clean")
            if payload["target_exit_observed"] is not True:
                raise ValueError(f"line {line_number} finalization lacks target EXIT observation")
            if final_rows != len(events) or final_bytes != event_bytes:
                raise ValueError(f"line {line_number} finalization counts are invalid")
            finalization_seen = True
            continue
        if len(events) >= maximum_rows:
            raise ValueError("Endpoint Security transcript exceeds event-row cap")
        event = _parse_endpoint_security_row(payload, line_number)
        if event.seq_num is not None:
            previous = last_seq_num.get(event.event)
            if previous is not None and event.seq_num <= previous:
                raise ValueError(f"line {line_number} sequence is not increasing")
            last_seq_num[event.event] = event.seq_num
        if event.global_seq_num is not None:
            if last_global_seq_num is not None and event.global_seq_num <= last_global_seq_num:
                raise ValueError(f"line {line_number} global sequence is not increasing")
            last_global_seq_num = event.global_seq_num
        events.append(event)
        event_bytes += line_bytes
    if not finalization_seen:
        raise ValueError("Endpoint Security transcript is missing finalization")
    return events


def _iter_bounded_transcript_lines(stream: IO[str]):
    """Read JSONL in capped chunks so one physical line cannot grow memory."""
    read_limit = MAXIMUM_ENDPOINT_SECURITY_ROW_BYTES + 1
    while True:
        line = stream.readline(read_limit)
        if not line:
            return
        try:
            line_bytes = len(line.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ValueError("Endpoint Security transcript must contain UTF-8 text") from exc
        if line_bytes > MAXIMUM_ENDPOINT_SECURITY_ROW_BYTES:
            raise ValueError("Endpoint Security transcript exceeds row byte cap")
        yield line


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
    "EndpointSecurityEvent",
    "ReadEvent",
    "ExternalKernelAuditUnavailable",
    "MAXIMUM_READ_EVENT_BYTES",
    "MAXIMUM_READ_EVENT_ROWS",
    "MAXIMUM_ENDPOINT_SECURITY_ROW_BYTES",
    "parse_fsusage_line",
    "parse_fsusage_transcript",
    "parse_endpoint_security_transcript",
    "build_read_isolation_attestation",
    "build_verified_read_isolation_attestation",
]
