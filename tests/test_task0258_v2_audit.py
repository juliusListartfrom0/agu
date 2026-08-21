"""Tests for the TASK-0258 v2 read-isolation fs_usage audit -> attestation."""

from __future__ import annotations

import io

import pytest

from app.analysis.task0258_v2_audit import (
    EndpointSecurityEvent,
    ExternalKernelAuditUnavailable,
    ReadEvent,
    build_read_isolation_attestation,
    build_verified_read_isolation_attestation,
    parse_endpoint_security_transcript,
    parse_fsusage_line,
    parse_fsusage_transcript,
)


def _inputs(denied_paths=()):
    receipt = {"artifact_sha256": "0" * 64, "file_sha256": "0" * 64}
    return {
        "policy_artifact_sha256": "0" * 64,
        "provider_receipt": receipt,
        "run_identity_receipt": receipt,
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
        "denied_paths": denied_paths,
    }


def test_parse_fsusage_line():
    ok = parse_fsusage_line(
        "06:02:18.189632  open              F=3        (R__________X___)  private/tmp/readable.txt     0.000173   Python.733218"
    )
    assert ok == ReadEvent("open", "/private/tmp/readable.txt", None)
    fail = parse_fsusage_line(
        "06:02:18.495890  open                   [  2] (R__________X___)  private/tmp/missing_file.txt  0.000093   Python.733218"
    )
    assert fail == ReadEvent("open", "/private/tmp/missing_file.txt", 2)
    # non-audited ops and headers are skipped
    assert parse_fsusage_line("06:02:18.189653  fstat64           F=3  0.000006   Python.733218") is None
    assert parse_fsusage_line("ktrace_start: No such process") is None


def test_parse_fsusage_transcript():
    text = (
        "ktrace_start: No such process\n"
        "06:02:18.189632  open              F=3  (R___)  private/tmp/readable.txt  0.1  Python.1\n"
        "06:02:18.495890  open                   [  2] (R___)  private/tmp/missing.txt  0.1  Python.1\n"
    )
    events = parse_fsusage_transcript(io.StringIO(text))
    assert len(events) == 2
    assert events[1].errno == 2


def test_parse_fsusage_transcript_enforces_event_row_cap():
    text = (
        "06:02:18.189632  open  private/tmp/one.txt\n"
        "06:02:18.189633  stat  private/tmp/two.txt\n"
    )
    with pytest.raises(ValueError, match="event-row cap"):
        parse_fsusage_transcript(io.StringIO(text), maximum_rows=1)


def test_parse_fsusage_transcript_enforces_byte_cap():
    with pytest.raises(ValueError, match="byte cap"):
        parse_fsusage_transcript(io.StringIO("diagnostic header\n"), maximum_bytes=8)


def test_parse_endpoint_security_transcript_preserves_notify_results_and_sequences():
    text = (
        '{"event":"open","pid":42,"pidversion":7,"ppid":1,"seq_num":10,'
        '"global_seq_num":100,"path":"/private/tmp/input.json",'
        '"result_type":"auth","result_auth":"allow"}\n'
        '{"event":"fork","pid":43,"pidversion":8,"ppid":42,"seq_num":11,'
        '"global_seq_num":102,"path":null,"result_type":"flags","result_flags":3}\n'
    )

    assert parse_endpoint_security_transcript(io.StringIO(text)) == [
        EndpointSecurityEvent(
            event="open",
            pid=42,
            pidversion=7,
            ppid=1,
            seq_num=10,
            global_seq_num=100,
            path="/private/tmp/input.json",
            result_type="auth",
            result_auth="allow",
            result_flags=None,
        ),
        EndpointSecurityEvent(
            event="fork",
            pid=43,
            pidversion=8,
            ppid=42,
            seq_num=11,
            global_seq_num=102,
            path=None,
            result_type="flags",
            result_auth=None,
            result_flags=3,
        ),
    ]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"event":"open","pid":42,"pidversion":7,"ppid":1}', "field set"),
        (
            '{"event":"unknown","pid":42,"pidversion":7,"ppid":1,"seq_num":1,'
            '"global_seq_num":1,"path":null,"result_type":"auth","result_auth":"allow"}',
            "event",
        ),
        (
            '{"event":"open","pid":0,"pidversion":7,"ppid":1,"seq_num":1,'
            '"global_seq_num":1,"path":null,"result_type":"auth","result_auth":"allow"}',
            "pid",
        ),
        (
            '{"event":"open","pid":42,"pidversion":7,"ppid":1,"seq_num":1,'
            '"global_seq_num":1,"path":"relative/path","result_type":"auth",'
            '"result_auth":"allow"}',
            "path",
        ),
        (
            '{"event":"open","pid":42,"pidversion":7,"ppid":1,"seq_num":1,'
            '"global_seq_num":1,"path":null,"result_type":"auth","result_auth":"maybe"}',
            "result_auth",
        ),
    ],
)
def test_parse_endpoint_security_transcript_rejects_invalid_rows(text, message):
    with pytest.raises(ValueError, match=message):
        parse_endpoint_security_transcript(io.StringIO(text + "\n"))


def test_parse_endpoint_security_transcript_rejects_duplicate_keys_and_sequence_regression():
    duplicate = (
        '{"event":"open","event":"stat","pid":42,"pidversion":7,"ppid":1,'
        '"seq_num":1,"global_seq_num":1,"path":null,"result_type":"auth",'
        '"result_auth":"allow"}\n'
    )
    with pytest.raises(ValueError, match="duplicate"):
        parse_endpoint_security_transcript(io.StringIO(duplicate))

    regressed = (
        '{"event":"open","pid":42,"pidversion":7,"ppid":1,"seq_num":2,'
        '"global_seq_num":2,"path":null,"result_type":"auth","result_auth":"allow"}\n'
        '{"event":"open","pid":42,"pidversion":7,"ppid":1,"seq_num":1,'
        '"global_seq_num":3,"path":null,"result_type":"auth","result_auth":"allow"}\n'
    )
    with pytest.raises(ValueError, match="sequence"):
        parse_endpoint_security_transcript(io.StringIO(regressed))


def test_parse_endpoint_security_transcript_enforces_bounds():
    row = (
        '{"event":"open","pid":42,"pidversion":7,"ppid":1,"seq_num":1,'
        '"global_seq_num":1,"path":null,"result_type":"auth","result_auth":"allow"}\n'
    )
    with pytest.raises(ValueError, match="event-row cap"):
        parse_endpoint_security_transcript(io.StringIO(row + row), maximum_rows=1)
    with pytest.raises(ValueError, match="byte cap"):
        parse_endpoint_security_transcript(io.StringIO(row), maximum_bytes=8)


def test_attestation_allowed_only():
    events = [ReadEvent("open", "/allowed/data.json", None), ReadEvent("stat", "/allowed/data.json", None)]
    with pytest.raises(ValueError):
        build_read_isolation_attestation(**{**_inputs(), "events": events})
    with pytest.raises(ExternalKernelAuditUnavailable):
        build_read_isolation_attestation(**{**_inputs(), "events": []})
    with pytest.raises(ExternalKernelAuditUnavailable):
        build_verified_read_isolation_attestation(**{**_inputs(), "verified_event_rows": []})


def test_attestation_rejects_raw_denied_read_fail_closed():
    # Raw rows cannot be upgraded into provider evidence, regardless of policy.
    events = [ReadEvent("open", "/allowed/data.json", None), ReadEvent("open", "/producer/embeddings.json", None)]
    with pytest.raises(ValueError):
        build_read_isolation_attestation(
            **{**_inputs(denied_paths=[{"path": "/producer/embeddings.json"}]), "events": events}
        )


def test_attestation_rejects_raw_path_alias_rows():
    # fs_usage path aliases remain diagnostic-only and cannot mint evidence.
    events = [ReadEvent("open", "/private/tmp/agu_audit_producer_embeddings.json", None)]
    with pytest.raises(ValueError):
        build_read_isolation_attestation(
            **{**_inputs(denied_paths=[{"path": "/tmp/agu_audit_producer_embeddings.json"}]), "events": events}
        )
