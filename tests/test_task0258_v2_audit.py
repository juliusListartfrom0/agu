"""Tests for the TASK-0258 v2 read-isolation fs_usage audit -> attestation."""

from __future__ import annotations

import io

import pytest

from app.analysis.task0258_v2_audit import (
    ReadEvent,
    build_read_isolation_attestation,
    parse_fsusage_line,
    parse_fsusage_transcript,
)
from app.analysis.task0258_v2_read_isolation import verify_read_isolation_attestation


def _inputs(denied_paths=()):
    receipt = {"artifact_sha256": "0" * 64, "file_sha256": "0" * 64}
    return {
        "policy_artifact_sha256": "0" * 64,
        "provider_receipt": receipt,
        "run_identity_receipt": receipt,
        "worker_role": "verification",
        "child_pid": 42,
        "ordered_observed_process_ids": [42],
        "provider_process_instance_id": 1,
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


def test_attestation_allowed_only():
    events = [ReadEvent("open", "/allowed/data.json", None), ReadEvent("stat", "/allowed/data.json", None)]
    att = build_read_isolation_attestation(**{**_inputs(), "events": events})
    verify_read_isolation_attestation(att)
    assert att["denied_read_attempt_count"] == 0
    assert att["unknown_read_attempt_count"] == 0
    assert att["audit_overflow"] is False
    assert len(att["read_event_projection_sha256"]) == 64


def test_attestation_detects_denied_read_fail_closed():
    # a denied-path read makes the attestation schema-invalid (denied must be 0),
    # so the builder must fail closed instead of producing an invalid artifact
    events = [ReadEvent("open", "/allowed/data.json", None), ReadEvent("open", "/producer/embeddings.json", None)]
    with pytest.raises(ValueError):
        build_read_isolation_attestation(
            **{**_inputs(denied_paths=[{"path": "/producer/embeddings.json"}]), "events": events}
        )


def test_attestation_deterministic_projection():
    events = [ReadEvent("open", "/a", None), ReadEvent("open", "/b", 2)]
    a = build_read_isolation_attestation(**{**_inputs(), "events": events})
    b = build_read_isolation_attestation(**{**_inputs(), "events": events})
    assert a["read_event_projection_sha256"] == b["read_event_projection_sha256"]


def test_denied_path_alias_matching():
    # fs_usage reports /private/tmp/... while the policy may name /tmp/...
    events = [ReadEvent("open", "/private/tmp/agu_audit_producer_embeddings.json", None)]
    with pytest.raises(ValueError):
        build_read_isolation_attestation(
            **{**_inputs(denied_paths=[{"path": "/tmp/agu_audit_producer_embeddings.json"}]), "events": events}
        )
