#!/usr/bin/env python3
"""TASK-0258 v2 read-isolation diagnostic harness (root-level syscall logging).

Spawns the verification worker, concurrently captures its filesystem syscall
stream with ``sudo fs_usage -f filesys``, then returns a bounded diagnostic
projection. ``fs_usage`` is not an authenticated kernel provider and this
command never creates a production read-isolation attestation.

Requires root: credentials must be cached (``sudo -v``) before invoking; the
harness itself calls ``sudo -n``.

Usage:
  run_fsusage_read_audit.py --worker-argv <cmd>... --policy <policy.json> \
      --out <diagnostic.json> [diagnostic inputs...]
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
from pathlib import Path

from app.analysis.task0258_v2_audit import MAXIMUM_READ_EVENT_BYTES, parse_fsusage_transcript
from app.analysis.task0258_v2_fs import atomic_write_json, read_regular_file_no_follow
from app.analysis.task0258_v2_worker_runner import (
    WorkerTimeoutError,
    sanitized_worker_env,
    validate_worker_launch_inputs,
)

_MAXIMUM_DIAGNOSTIC_TRANSCRIPT_BYTES = MAXIMUM_READ_EVENT_BYTES


class _BoundedTextCapture:
    """Drain a text pipe while retaining at most a bounded UTF-8 payload."""

    def __init__(self, *, maximum_bytes: int) -> None:
        if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int) or maximum_bytes <= 0:
            raise ValueError("maximum_bytes must be a positive integer")
        self._maximum_bytes = maximum_bytes
        self._total_bytes = 0
        self._lines: list[str] = []
        self._exceeded = False
        self._error: ValueError | None = None

    def append(self, line: str) -> None:
        """Retain a line only while the complete stream remains under the cap."""
        if not isinstance(line, str):
            self._error = ValueError("diagnostic stream must contain text")
            return
        if self._exceeded or self._error is not None:
            return
        try:
            line_bytes = len(line.encode("utf-8"))
        except UnicodeEncodeError:
            self._error = ValueError("diagnostic stream must contain UTF-8 text")
            return
        if self._total_bytes + line_bytes > self._maximum_bytes:
            self._exceeded = True
            return
        self._lines.append(line)
        self._total_bytes += line_bytes

    def finish(self) -> str:
        """Return retained text or fail closed after an over-limit stream."""
        if self._error is not None:
            raise self._error
        if self._exceeded:
            raise ValueError("diagnostic transcript exceeds byte cap")
        return "".join(self._lines)


def _drain_text_stream(stream, sink: _BoundedTextCapture, errors: list[Exception]) -> None:
    """Drain one diagnostic stream while making iterator failures observable."""
    try:
        for line in stream:
            sink.append(line)
    except Exception as exc:
        errors.append(exc)


def _join_diagnostic_drains(threads, *, timeout_seconds: float) -> None:
    """Require every diagnostic drain to finish before parsing its transcript."""
    for thread in threads:
        thread.join(timeout=timeout_seconds)
    if any(thread.is_alive() for thread in threads):
        raise RuntimeError("fs_usage diagnostic stream did not finish")


def _wait_for_worker(proc, worker_argv: list[str], *, timeout_seconds: int) -> None:
    """Wait for the worker or kill and reap its dedicated process group."""
    try:
        proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        raise WorkerTimeoutError(worker_argv, timeout_seconds, "", "") from None


def _kill_process_group(proc) -> None:
    """Hard-kill and reap a process whose group was created by this harness."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def _terminate_and_reap_process(proc, *, timeout_seconds: int = 5) -> None:
    """Terminate the entire harness child group, escalate, and reap its leader."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        if proc.poll() is None:
            proc.terminate()
    try:
        proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)


def run_read_audit(
    *,
    worker_argv: list[str],
    policy_payload: dict,
    attestation_inputs: dict,
    fs_usage_extra_args: list[str] | None = None,
    sudo_password: str | None = None,
    worker_timeout_seconds: int = 120,
) -> tuple[dict, list]:
    """Run the diagnostic and return ``(diagnostic_payload, events)``.

    ``attestation_inputs`` is retained for the planned provider adapter but is
    never used to mint authority. Root is required:
    pass ``sudo_password`` (used via ``sudo -S`` on stdin) or pre-cache
    credentials so ``sudo -n`` succeeds.
    """
    import io
    import threading

    del policy_payload, attestation_inputs
    validate_worker_launch_inputs(worker_argv, worker_timeout_seconds)
    proc = subprocess.Popen(
        worker_argv,
        env=sanitized_worker_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    sudo_mode = ["sudo", "-S"] if sudo_password is not None else ["sudo", "-n"]
    fs_argv = sudo_mode + ["fs_usage", "-w", "-f", "filesys", str(proc.pid)]
    if fs_usage_extra_args:
        fs_argv = fs_argv[:3] + fs_usage_extra_args + fs_argv[3:]
    out_capture = _BoundedTextCapture(maximum_bytes=_MAXIMUM_DIAGNOSTIC_TRANSCRIPT_BYTES)
    err_capture = _BoundedTextCapture(maximum_bytes=_MAXIMUM_DIAGNOSTIC_TRANSCRIPT_BYTES)
    drain_errors: list[Exception] = []
    drain_threads = []
    fs = None
    try:
        fs = subprocess.Popen(
            fs_argv,
            stdin=subprocess.PIPE if sudo_password is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        t1 = threading.Thread(target=_drain_text_stream, args=(fs.stdout, out_capture, drain_errors), daemon=True)
        t2 = threading.Thread(target=_drain_text_stream, args=(fs.stderr, err_capture, drain_errors), daemon=True)
        drain_threads = [t1, t2]
        t1.start()
        t2.start()
        if sudo_password is not None:
            if fs.stdin is None:
                raise RuntimeError("fs_usage stdin was not opened")
            fs.stdin.write(sudo_password + "\n")
            fs.stdin.flush()
            fs.stdin.close()
        _wait_for_worker(proc, worker_argv, timeout_seconds=worker_timeout_seconds)
    finally:
        try:
            if fs is not None:
                try:
                    if fs.stdin is not None:
                        fs.stdin.close()
                except (OSError, ValueError):
                    pass
                _terminate_and_reap_process(fs)
            if drain_threads:
                _join_diagnostic_drains(drain_threads, timeout_seconds=5)
        finally:
            _terminate_and_reap_process(proc)

    if drain_errors:
        raise RuntimeError("fs_usage diagnostic stream drain failed") from drain_errors[0]
    fs_out = out_capture.finish()
    fs_err = err_capture.finish()
    if fs.returncode not in (0, -15, None) and not fs_out and fs_err:
        raise RuntimeError(f"fs_usage failed: {fs_err.strip()}")
    events = parse_fsusage_transcript(io.StringIO(fs_out))
    diagnostic = {
        "schema_version": "agu.task0258-fsusage-diagnostic-projection.v1",
        "status": "external_kernel_audit_required",
        "event_count": len(events),
        "evidence_class": "diagnostic_only",
        "production_capability": False,
        "p5_ready": False,
    }
    return diagnostic, events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-argv", nargs="+", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker-role", default="verification")
    parser.add_argument("--child-nonce", default="0" * 64)
    parser.add_argument("--provider-process-instance-id", type=int, default=1)
    parser.add_argument("--sha", default="0" * 64, help="value for the five artifact SHA fields")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    policy_raw = read_regular_file_no_follow(args.policy)
    try:
        policy = json.loads(policy_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("policy must be UTF-8 JSON") from exc
    if not isinstance(policy, dict):
        raise ValueError("policy must be a JSON object")
    receipt = {"artifact_sha256": args.sha, "file_sha256": args.sha}
    inputs = {
        "policy_artifact_sha256": args.sha,
        "provider_receipt": receipt,
        "run_identity_receipt": receipt,
        "worker_role": args.worker_role,
        "child_pid": 0,  # overridden by run_read_audit with the actual spawned pid
        "ordered_observed_process_ids": [],
        "provider_process_instance_id": args.provider_process_instance_id,
        "child_nonce": args.child_nonce,
        "prepare_artifact_sha256": args.sha,
        "prepared_artifact_sha256": args.sha,
        "child_started_artifact_sha256": args.sha,
        "permit_artifact_sha256": args.sha,
        "finalize_artifact_sha256": args.sha,
    }
    attestation, events = run_read_audit(
        worker_argv=args.worker_argv,
        policy_payload=policy,
        attestation_inputs=inputs,
        sudo_password=__import__("os").environ.get("AGU_SUDO_PASSWORD"),
    )
    atomic_write_json(args.out, attestation)
    print(f"events={len(events)}")
    print(f"status={attestation['status']}")
    print(f"diagnostic projection written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
