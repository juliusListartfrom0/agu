"""Tests for the TASK-0258 v2 verification worker subprocess isolation."""

from __future__ import annotations

import json
import os
import sys
import time

import pytest

from app.analysis.task0258_v2_worker_runner import (
    CONTRACT_ENV,
    WorkerTimeoutError,
    run_worker_subprocess,
    sanitized_worker_env,
)


def test_sanitized_worker_env():
    env = sanitized_worker_env({"PATH": "/usr/bin"})
    for key, value in CONTRACT_ENV.items():
        assert env[key] == value
    assert env["PATH"] == "/usr/bin"
    # host variables are NOT inherited
    assert "PYTHONPATH" not in env


def test_run_worker_subprocess_env_isolation():
    probe = (
        "import json,os;"
        "print(json.dumps({k: os.environ.get(k) for k in "
        "['PYTHONHASHSEED','OMP_NUM_THREADS','PYTHONPATH','SECRET_HOST_VAR']}))"
    )
    result = run_worker_subprocess(
        argv=[sys.executable, "-c", probe],
        env=sanitized_worker_env({"PATH": os.environ.get("PATH", "")}),
    )
    assert result.exit_code == 0
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    assert observed["PYTHONHASHSEED"] == "0"
    assert observed["OMP_NUM_THREADS"] == "1"
    assert observed["PYTHONPATH"] is None
    assert observed["SECRET_HOST_VAR"] is None


def test_run_worker_subprocess_timeout():
    with pytest.raises(WorkerTimeoutError):
        run_worker_subprocess(
            argv=[sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_seconds=1,
        )


def test_timeout_kills_descendant_process_group(tmp_path):
    child_pid_file = tmp_path / "child.pid"
    code = (
        "import pathlib,subprocess,sys,time; "
        f"p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(p.pid)); time.sleep(30)"
    )
    with pytest.raises(WorkerTimeoutError):
        run_worker_subprocess(argv=[sys.executable, "-c", code], timeout_seconds=1)
    child_pid = int(child_pid_file.read_text())
    for _ in range(20):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("worker descendant survived process-group cleanup")
