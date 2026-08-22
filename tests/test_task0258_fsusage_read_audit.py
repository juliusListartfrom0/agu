"""Tests for bounded TASK-0258 fs_usage diagnostic capture."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

from app.analysis.task0258_v2_worker_runner import WorkerTimeoutError, sanitized_worker_env
from scripts import run_fsusage_read_audit as audit_script
from scripts.run_fsusage_read_audit import (
    _BoundedTextCapture,
    _drain_text_stream,
    _join_diagnostic_drains,
    _terminate_and_reap_process,
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


def test_run_read_audit_reaps_worker_when_fs_usage_cannot_start(monkeypatch):
    class StubbornWorker:
        pid = 4321

        def __init__(self):
            self.terminated = False
            self.reaped = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired(cmd=["worker"], timeout=timeout)
            self.reaped = True

    worker = StubbornWorker()
    popen_calls = 0

    def fail_fs_usage(argv, **kwargs):
        nonlocal popen_calls
        popen_calls += 1
        if popen_calls == 1:
            return worker
        raise OSError("fs_usage unavailable")

    kill_calls = []
    monkeypatch.setattr(subprocess, "Popen", fail_fs_usage)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: kill_calls.append((pid, sig)))

    with pytest.raises(OSError, match="fs_usage unavailable"):
        run_read_audit(worker_argv=["worker"], policy_payload={}, attestation_inputs={})

    assert worker.reaped is True
    assert kill_calls == [(worker.pid, signal.SIGTERM), (worker.pid, signal.SIGKILL)]


def test_terminate_and_reap_process_kills_worker_descendants(tmp_path):
    child_pid_file = tmp_path / "child.pid"
    child_code = "import time; time.sleep(30)"
    parent_code = (
        "import pathlib, subprocess, sys, time; "
        f"child=subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid)); time.sleep(30)"
    )
    parent = subprocess.Popen([sys.executable, "-c", parent_code], start_new_session=True)
    try:
        for _ in range(40):
            if child_pid_file.exists():
                break
            time.sleep(0.05)
        child_pid = int(child_pid_file.read_text())
        _terminate_and_reap_process(parent, timeout_seconds=1)
        for _ in range(40):
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            pytest.fail("worker descendant survived process-group cleanup")
    finally:
        if parent.poll() is None:
            os.killpg(parent.pid, signal.SIGKILL)
            parent.wait()


def test_run_read_audit_reaps_fs_usage_after_worker_finishes(monkeypatch):
    class FinishedWorker:
        pid = 1111

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    class DiagnosticProcess:
        pid = 2222
        returncode = 0

        def __init__(self):
            self.terminated = False
            self.reaped = False
            self.stdin = None
            self.stdout = [""]
            self.stderr = [""]

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.reaped = True
            return self.returncode

    worker = FinishedWorker()
    diagnostic_process = DiagnosticProcess()
    popen_calls = 0

    def fake_popen(argv, **kwargs):
        nonlocal popen_calls
        popen_calls += 1
        if popen_calls == 1:
            return worker
        diagnostic_process.stdin = kwargs.get("stdin")
        return diagnostic_process

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    sha = "0" * 64
    inputs = {
        "policy_artifact_sha256": sha,
        "provider_receipt": {},
        "run_identity_receipt": {},
        "worker_role": "verification",
        "provider_process_instance_id": 1,
        "child_nonce": sha,
        "prepare_artifact_sha256": sha,
        "prepared_artifact_sha256": sha,
        "child_started_artifact_sha256": sha,
        "permit_artifact_sha256": sha,
        "finalize_artifact_sha256": sha,
    }
    report, events = run_read_audit(worker_argv=["worker"], policy_payload={}, attestation_inputs=inputs)
    assert events == []
    assert report == {
        "schema_version": "agu.task0258-fsusage-diagnostic-projection.v1",
        "status": "external_kernel_audit_required",
        "event_count": 0,
        "evidence_class": "diagnostic_only",
        "production_capability": False,
        "p5_ready": False,
    }

    assert diagnostic_process.reaped is True


def test_run_read_audit_reaps_worker_when_diagnostic_drain_cleanup_fails(monkeypatch):
    class Worker:
        pid = 3333

        def __init__(self):
            self.terminated = False
            self.reaped = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.reaped = True
            return 0

    class DiagnosticProcess:
        pid = 4444
        returncode = 0

        def __init__(self):
            self.stdin = None
            self.stdout = [""]
            self.stderr = [""]

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    worker = Worker()
    diagnostic = DiagnosticProcess()
    popen_calls = 0

    def fake_popen(argv, **kwargs):
        nonlocal popen_calls
        popen_calls += 1
        if popen_calls == 1:
            return worker
        return diagnostic

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        "scripts.run_fsusage_read_audit._join_diagnostic_drains",
        lambda threads, timeout_seconds: (_ for _ in ()).throw(RuntimeError("drain failed")),
    )

    with pytest.raises(RuntimeError, match="drain failed"):
        run_read_audit(worker_argv=["worker"], policy_payload={}, attestation_inputs={})

    assert worker.terminated is True
    assert worker.reaped is True


def test_run_read_audit_rejects_invalid_worker_timeout_before_spawn(monkeypatch):
    def fail_popen(*args, **kwargs):
        pytest.fail("invalid worker timeout reached Popen")

    monkeypatch.setattr(subprocess, "Popen", fail_popen)
    with pytest.raises(ValueError, match="worker timeout"):
        run_read_audit(worker_argv=["worker"], worker_timeout_seconds=0, policy_payload={}, attestation_inputs={})


def test_cli_rejects_symlinked_policy_and_does_not_start_audit(monkeypatch, tmp_path):
    policy_target = tmp_path / "policy-target.json"
    policy_target.write_text("{}\n", encoding="utf-8")
    policy_link = tmp_path / "policy.json"
    policy_link.symlink_to(policy_target)
    monkeypatch.setattr(audit_script, "run_read_audit", lambda **kwargs: pytest.fail("audit started"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_fsusage_read_audit",
            "--worker-argv",
            "worker",
            "--policy",
            str(policy_link),
            "--out",
            str(tmp_path / "out.json"),
        ],
    )

    with pytest.raises(ValueError, match="without following links"):
        audit_script.main()


def test_cli_does_not_clobber_symlinked_output(monkeypatch, tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text("{}\n", encoding="utf-8")
    target = tmp_path / "target.json"
    target.write_text("sentinel\n", encoding="utf-8")
    output_link = tmp_path / "out.json"
    output_link.symlink_to(target)
    monkeypatch.setattr(
        audit_script,
        "run_read_audit",
        lambda **kwargs: ({"safe": True, "denied_read_attempt_count": 0}, []),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_fsusage_read_audit",
            "--worker-argv",
            "worker",
            "--policy",
            str(policy),
            "--out",
            str(output_link),
        ],
    )

    with pytest.raises(OSError):
        audit_script.main()
    assert target.read_text(encoding="utf-8") == "sentinel\n"


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
