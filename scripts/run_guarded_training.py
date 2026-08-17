#!/usr/bin/env python3
"""Run a local AGU training command under sustained CPU/memory protection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.training.resource_guard import (  # noqa: E402
    TrainingResourceGuard,
    TrainingResourceThresholds,
    sample_training_resources,
)

RESOURCE_LIMIT_EXIT_CODE = 75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-memory-percent", type=float, default=90.0)
    parser.add_argument("--min-available-gib", type=float, default=2.0)
    parser.add_argument("--min-free-swap-gib", type=float, default=0.0)
    parser.add_argument("--max-cpu-percent", type=float, default=95.0)
    parser.add_argument("--consecutive-breaches", type=int, default=3)
    parser.add_argument("--sample-interval-sec", type=float, default=5.0)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a training command is required after --")
    if args.sample_interval_sec <= 0:
        parser.error("--sample-interval-sec must be positive")
    return args


def _write_event(stream: TextIO, payload: dict[str, object] | str) -> None:
    line = payload if isinstance(payload, str) else json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )
    stream.write(line + "\n")
    stream.flush()
    os.fsync(stream.fileno())
    print(line, flush=True)


def _terminate_supervised_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def main() -> int:
    args = parse_args()
    thresholds = TrainingResourceThresholds(
        max_system_memory_percent=args.max_memory_percent,
        min_available_memory_gib=args.min_available_gib,
        min_free_swap_gib=args.min_free_swap_gib,
        max_system_cpu_percent=args.max_cpu_percent,
        consecutive_breaches=args.consecutive_breaches,
    )
    args.log.parent.mkdir(parents=True, exist_ok=True)
    guard = TrainingResourceGuard(thresholds)
    with args.log.open("a", encoding="utf-8") as stream:
        process: subprocess.Popen[bytes] | None = None
        try:
            process = subprocess.Popen(args.command, start_new_session=True)
            _write_event(
                stream,
                {
                    "event": "training_resource_guard_started",
                    "command_executable": Path(args.command[0]).name,
                    "command_argv_count": len(args.command),
                    "command_sha256": hashlib.sha256(
                        json.dumps(args.command, separators=(",", ":")).encode(
                            "utf-8"
                        )
                    ).hexdigest(),
                    "process_id": process.pid,
                    "sample_interval_sec": args.sample_interval_sec,
                    "thresholds": {
                        "max_system_memory_percent": (
                            thresholds.max_system_memory_percent
                        ),
                        "min_available_memory_gib": (
                            thresholds.min_available_memory_gib
                        ),
                        "min_free_swap_gib": thresholds.min_free_swap_gib,
                        "max_system_cpu_percent": thresholds.max_system_cpu_percent,
                        "consecutive_breaches": thresholds.consecutive_breaches,
                    },
                },
            )
            try:
                while process.poll() is None:
                    try:
                        snapshot = sample_training_resources(process.pid)
                    except Exception as exc:
                        _terminate_supervised_process(process)
                        _write_event(
                            stream,
                            {
                                "event": "training_resource_guard_error",
                                "error_type": type(exc).__name__,
                                "exit_code": RESOURCE_LIMIT_EXIT_CODE,
                            },
                        )
                        return RESOURCE_LIMIT_EXIT_CODE
                    decision = guard.observe(snapshot, stage="training")
                    _write_event(stream, decision.to_json())
                    if decision.should_stop:
                        _terminate_supervised_process(process)
                        _write_event(
                            stream,
                            {
                                "event": "training_resource_guard_stopped",
                                "exit_code": RESOURCE_LIMIT_EXIT_CODE,
                                "reasons": decision.reasons,
                            },
                        )
                        return RESOURCE_LIMIT_EXIT_CODE
                    time.sleep(args.sample_interval_sec)
            except KeyboardInterrupt:
                _terminate_supervised_process(process)
                _write_event(
                    stream,
                    {
                        "event": "training_resource_guard_interrupted",
                        "exit_code": 130,
                    },
                )
                return 130
            exit_code = int(process.returncode or 0)
            _write_event(
                stream,
                {"event": "training_resource_guard_finished", "exit_code": exit_code},
            )
            return exit_code
        finally:
            if process is not None:
                _terminate_supervised_process(process)


if __name__ == "__main__":
    raise SystemExit(main())
