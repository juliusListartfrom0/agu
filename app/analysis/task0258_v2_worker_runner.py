"""TASK-0258 Amendment-001 v2 verification worker — subprocess isolation runner.

The amendment requires the second empty-state extraction to run in a NEW process
with fresh model construction under the exact deterministic environment. This
module provides the subprocess boundary: a sanitized contract environment, a
bounded timeout with hard kill, and a structured result. The stronger
kernel-audit read-isolation attestation remains a platform-level dependency; this
runner is the process isolation layer that audit wraps.
"""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass

# Exact deterministic process environment (mirrors the parent temporal contract).
CONTRACT_ENV: dict[str, str] = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class WorkerResult:
    exit_code: int
    stdout: str
    stderr: str


class WorkerTimeoutError(RuntimeError):
    def __init__(self, argv, timeout_seconds, stdout, stderr):
        super().__init__(f"worker timed out after {timeout_seconds}s: {argv!r}")
        self.argv = argv
        self.timeout_seconds = timeout_seconds
        self.stdout = stdout
        self.stderr = stderr


def sanitized_worker_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Build the worker environment: contract vars + a minimal PATH only.

    No other caller/host variables are inherited, so the worker cannot observe
    host secrets or configuration.
    """
    env = dict(CONTRACT_ENV)
    if base is not None:
        env["PATH"] = base.get("PATH", os.environ.get("PATH", ""))
    else:
        env["PATH"] = os.environ.get("PATH", "")
    return env


def run_worker_subprocess(
    *,
    argv: list[str],
    timeout_seconds: int = 120,
    env: dict[str, str] | None = None,
) -> WorkerResult:
    """Spawn the worker in a fresh subprocess with the contract environment.

    On timeout the process group is hard-killed and ``WorkerTimeoutError`` is
    raised with the captured output.
    """
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(arg, str) or "\x00" in arg for arg in argv)
    ):
        raise ValueError("worker argv is invalid")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        raise ValueError("worker timeout must be a positive integer")
    expected_env = sanitized_worker_env()
    if env is not None and env != expected_env:
        raise ValueError("worker environment must be the exact sanitized contract environment")
    proc = subprocess.Popen(
        argv,
        env=expected_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
        raise WorkerTimeoutError(argv, timeout_seconds, stdout, stderr) from None
    return WorkerResult(proc.returncode, stdout, stderr)


__all__ = [
    "CONTRACT_ENV",
    "WorkerResult",
    "WorkerTimeoutError",
    "sanitized_worker_env",
    "run_worker_subprocess",
]
