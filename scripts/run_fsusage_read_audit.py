#!/usr/bin/env python3
"""TASK-0258 v2 read-isolation audit harness (root-level syscall auditing).

Spawns the verification worker, concurrently captures its filesystem syscall
stream with ``sudo fs_usage -f filesys``, then builds the
``agu.task0258-module-a-worker-read-isolation-attestation.v1`` from the ordered
read events.

Requires root: credentials must be cached (``sudo -v``) before invoking; the
harness itself calls ``sudo -n``.

Usage:
  run_fsusage_read_audit.py --worker-argv <cmd>... --policy <policy.json> \
      --out <attestation.json> [attestation inputs...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from app.analysis.task0258_v2_audit import (
    MAXIMUM_READ_EVENT_BYTES,
    build_read_isolation_attestation,
    parse_fsusage_transcript,
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

    def append(self, line: str) -> None:
        """Retain a line only while the complete stream remains under the cap."""
        if not isinstance(line, str):
            raise ValueError("diagnostic stream must contain text")
        if self._exceeded:
            return
        try:
            line_bytes = len(line.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ValueError("diagnostic stream must contain UTF-8 text") from exc
        if self._total_bytes + line_bytes > self._maximum_bytes:
            self._exceeded = True
            return
        self._lines.append(line)
        self._total_bytes += line_bytes

    def finish(self) -> str:
        """Return retained text or fail closed after an over-limit stream."""
        if self._exceeded:
            raise ValueError("diagnostic transcript exceeds byte cap")
        return "".join(self._lines)


def run_read_audit(
    *,
    worker_argv: list[str],
    policy_payload: dict,
    attestation_inputs: dict,
    fs_usage_extra_args: list[str] | None = None,
    sudo_password: str | None = None,
) -> tuple[dict, list]:
    """Run the audit and return ``(attestation_payload, events)``.

    ``attestation_inputs`` carries the non-derived attestation fields
    (role/nonce/hashes); the derived fields (ordered events, denied count,
    projection) are computed from the audited transcript. Root is required:
    pass ``sudo_password`` (used via ``sudo -S`` on stdin) or pre-cache
    credentials so ``sudo -n`` succeeds.
    """
    import io
    import threading

    proc = subprocess.Popen(worker_argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
    attestation_inputs = {
        **attestation_inputs,
        "child_pid": proc.pid,
        "ordered_observed_process_ids": [proc.pid],
    }
    sudo_mode = ["sudo", "-S"] if sudo_password is not None else ["sudo", "-n"]
    fs_argv = sudo_mode + ["fs_usage", "-w", "-f", "filesys", str(proc.pid)]
    if fs_usage_extra_args:
        fs_argv = fs_argv[:3] + fs_usage_extra_args + fs_argv[3:]
    fs = subprocess.Popen(
        fs_argv,
        stdin=subprocess.PIPE if sudo_password is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if sudo_password is not None:
        fs.stdin.write(sudo_password + "\n")
        fs.stdin.flush()
        fs.stdin.close()
    out_capture = _BoundedTextCapture(maximum_bytes=_MAXIMUM_DIAGNOSTIC_TRANSCRIPT_BYTES)
    err_capture = _BoundedTextCapture(maximum_bytes=_MAXIMUM_DIAGNOSTIC_TRANSCRIPT_BYTES)

    def _drain(stream, sink):
        for line in stream:
            sink.append(line)

    t1 = threading.Thread(target=_drain, args=(fs.stdout, out_capture), daemon=True)
    t2 = threading.Thread(target=_drain, args=(fs.stderr, err_capture), daemon=True)
    t1.start()
    t2.start()
    proc.wait()
    try:
        fs.terminate()
    except ProcessLookupError:
        pass
    t1.join(timeout=5)
    t2.join(timeout=5)
    fs_out = out_capture.finish()
    fs_err = err_capture.finish()
    if fs.returncode not in (0, -15, None) and not fs_out and fs_err:
        raise RuntimeError(f"fs_usage failed: {fs_err.strip()}")
    events = parse_fsusage_transcript(io.StringIO(fs_out))
    attestation = build_read_isolation_attestation(
        **attestation_inputs,
        events=events,
        denied_paths=policy_payload.get("ordered_denied_read_rows", []),
    )
    return attestation, events


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
    policy = json.loads(args.policy.read_text())
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
    args.out.write_text(json.dumps(attestation, indent=2) + "\n")
    print(f"events={len(events)} denied={attestation['denied_read_attempt_count']}")
    print(f"projection={attestation['read_event_projection_sha256']}")
    print(f"attestation written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
