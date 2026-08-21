"""Tests for bounded TASK-0258 fs_usage diagnostic capture."""

from __future__ import annotations

import os
import signal
import subprocess

import pytest

from app.analysis.task0258_v2_worker_runner import WorkerTimeoutError, sanitized_worker_env
from scripts.run_fsusage_read_audit import (
    _BoundedTextCapture,
    _drain_text_stream,
    _join_diagnostic_drains,
    _wait_for_worker,
    run_read_audit,
)


def test_bounded_text_capture_returns_text_within_byte_cap():
    capture = _BoundedTextCapture(maximum_bytes=8)
    capture.append("1234\n")
    capture.append("56\n")
    assert capture.finish() == "1234\n56\n"


def test_bounded_text_capture_drains_after_cap_without_retaining_more_data():
    capture = _BoundedTextCapture(maximum_bytes=4)
    capture.append("1234")
    capture.append("this line is over the cap")
    capture.append("and this later line is discarded")
    with pytest.raises(ValueError, match="byte cap"):
        capture.finish()


def test_bounded_text_capture_defers_malformed_input_failure_to_finish():
    capture = _BoundedTextCapture(maximum_bytes=8)
    capture.append(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="text"):
        capture.finish()


def test_drain_text_stream_records_iterator_failures_for_main_thread():
    class FailingStream:
        def __iter__(self):
            yield "first line\n"
            raise RuntimeError("decoder failed")

    capture = _BoundedTextCapture(maximum_bytes=32)
    errors = []
    _drain_text_stream(FailingStream(), capture, errors)

    assert capture.finish() == "first line\n"
    assert len(errors) == 1
    assert str(errors[0]) == "decoder failed"


def test_join_diagnostic_drains_rejects_a_stream_that_did_not_finish():
    class HangingThread:
        def join(self, timeout):
            assert timeout == 0.25

        def is_alive(self):
            return True

    with pytest.raises(RuntimeError, match="did not finish"):
        _join_diagnostic_drains([HangingThread()], timeout_seconds=0.25)


def test_run_read_audit_rejects_invalid_worker_argv_before_spawn(monkeypatch):
    def fail_popen(*args, **kwargs):
        pytest.fail("invalid worker argv reached Popen")

    monkeypatch.setattr(subprocess, "Popen", fail_popen)
    with pytest.raises(ValueError, match="worker argv"):
        run_read_audit(worker_argv="not-a-list", policy_payload={}, attestation_inputs={})  # type: ignore[arg-type]


def test_run_read_audit_uses_sanitized_worker_process_group(monkeypatch):
    calls = []

    def stop_after_worker_spawn(argv, **kwargs):
        calls.append((argv, kwargs))
        raise RuntimeError("stop after worker spawn")

    monkeypatch.setattr(subprocess, "Popen", stop_after_worker_spawn)
    with pytest.raises(RuntimeError, match="stop after worker spawn"):
        run_read_audit(worker_argv=["worker"], policy_payload={}, attestation_inputs={})

    assert calls[0][1]["env"] == sanitized_worker_env()
    assert calls[0][1]["start_new_session"] is True


def test_run_read_audit_rejects_invalid_worker_timeout_before_spawn(monkeypatch):
    def fail_popen(*args, **kwargs):
        pytest.fail("invalid worker timeout reached Popen")

    monkeypatch.setattr(subprocess, "Popen", fail_popen)
    with pytest.raises(ValueError, match="worker timeout"):
        run_read_audit(worker_argv=["worker"], worker_timeout_seconds=0, policy_payload={}, attestation_inputs={})


def test_wait_for_worker_kills_and_reaps_timed_out_process_group(monkeypatch):
    class HangingProcess:
        pid = 1234

        def __init__(self):
            self.reaped = False

        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired(cmd=["worker"], timeout=timeout)
            self.reaped = True

    kill_calls = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: kill_calls.append((pid, sig)))
    process = HangingProcess()

    with pytest.raises(WorkerTimeoutError):
        _wait_for_worker(process, ["worker"], timeout_seconds=3)

    assert kill_calls == [(1234, signal.SIGKILL)]
    assert process.reaped is True


@pytest.mark.parametrize("maximum_bytes", [0, -1, True, 1.0])
def test_bounded_text_capture_rejects_invalid_caps(maximum_bytes):
    with pytest.raises(ValueError):
        _BoundedTextCapture(maximum_bytes=maximum_bytes)
