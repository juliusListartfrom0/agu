#!/usr/bin/env python3
"""Read-only memory guard for Codex processes and session logs.

The guard is intentionally observational.  It never kills processes, archives
files, or edits session state.  Exit code 2 means that the current task should
stop before more tool calls; a human can then close duplicate sessions, archive
oversized logs, or restart Codex.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISREG
from typing import Sequence

GIB = 1024**3
MIB = 1024**2
DEFAULT_MEMORY_THRESHOLD_BYTES = 8 * GIB
DEFAULT_SESSION_WARNING_BYTES = 100 * MIB
STOP_REQUIRED = 2
_CODEX_COMMAND_RE = re.compile(r"(?i)(?:^|[/\s])codex(?:$|[/\s])")


@dataclass(frozen=True)
class ProcessMemory:
    """One Codex process as reported by macOS ``ps``."""

    pid: int
    rss_bytes: int
    elapsed: str
    command: str


@dataclass(frozen=True)
class SessionLog:
    """A session-log size observation; file contents are never read."""

    path: str
    size_bytes: int


def _is_codex_command(command: str) -> bool:
    if "monitor_codex_memory.py" in command:
        return False
    return bool(_CODEX_COMMAND_RE.search(command))


def _command_kind(command: str) -> str:
    normalized = command.lower()
    if "renderer" in normalized:
        return "codex_renderer"
    if "computer use" in normalized:
        return "codex_computer_use"
    if "crashpad" in normalized:
        return "codex_crashpad"
    if "gpu-process" in normalized:
        return "codex_gpu"
    return "codex_service"


def parse_process_table(output: str) -> tuple[ProcessMemory, ...]:
    """Parse ``ps`` rows, retaining only Codex commands.

    The macOS ``rss`` column is expressed in KiB.  Malformed rows are ignored
    so an unrelated process-table change cannot cause a false stop decision.
    """

    processes: list[ProcessMemory] = []
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=3)
        if len(fields) != 4:
            continue
        pid_text, rss_text, elapsed, command = fields
        if not _is_codex_command(command):
            continue
        try:
            pid = int(pid_text)
            rss_kib = int(rss_text)
        except ValueError:
            continue
        if pid <= 0 or rss_kib < 0:
            continue
        processes.append(ProcessMemory(pid=pid, rss_bytes=rss_kib * 1024, elapsed=elapsed, command=command))
    return tuple(sorted(processes, key=lambda process: (-process.rss_bytes, process.pid)))


def collect_codex_processes() -> tuple[ProcessMemory, ...]:
    """Collect current Codex RSS without inspecting session contents."""

    completed = subprocess.run(
        ["ps", "-axo", "pid=,rss=,etime=,command="],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ps failed with exit code {completed.returncode}: {completed.stderr.strip()}")
    return parse_process_table(completed.stdout)


def inspect_session_logs(sessions_root: Path) -> tuple[SessionLog, ...]:
    """Stat active JSONL session logs, skipping symlinks and non-regular files."""

    sessions_root = Path(sessions_root)
    if not sessions_root.exists() or not sessions_root.is_dir() or sessions_root.is_symlink():
        return ()
    observations: list[SessionLog] = []
    for path in sessions_root.rglob("*.jsonl"):
        if path.is_symlink():
            continue
        try:
            stat_result = path.stat(follow_symlinks=False)
        except OSError:
            continue
        if S_ISREG(stat_result.st_mode):
            observations.append(SessionLog(path=str(path), size_bytes=stat_result.st_size))
    return tuple(sorted(observations, key=lambda log: (-log.size_bytes, log.path)))


def build_report(
    *,
    processes: Sequence[ProcessMemory],
    session_logs: Sequence[SessionLog],
    memory_threshold_bytes: int = DEFAULT_MEMORY_THRESHOLD_BYTES,
    session_warning_bytes: int = DEFAULT_SESSION_WARNING_BYTES,
) -> dict[str, object]:
    """Build a deterministic stop/warn report from read-only observations."""

    if memory_threshold_bytes <= 0 or session_warning_bytes <= 0:
        raise ValueError("memory thresholds must be positive")
    ordered_processes = tuple(sorted(processes, key=lambda process: (-process.rss_bytes, process.pid)))
    ordered_logs = tuple(sorted(session_logs, key=lambda log: (-log.size_bytes, log.path)))
    aggregate_rss = sum(process.rss_bytes for process in ordered_processes)
    largest_process = ordered_processes[0].rss_bytes if ordered_processes else 0
    largest_session_log = ordered_logs[0].size_bytes if ordered_logs else 0
    stop_required = any(
        value >= memory_threshold_bytes for value in (aggregate_rss, largest_process, largest_session_log)
    )
    session_warning = largest_session_log >= session_warning_bytes
    return {
        "schema_version": "agu.codex-memory-guard-report.v1",
        "status": "stop_required" if stop_required else "ok",
        "exit_code": STOP_REQUIRED if stop_required else 0,
        "memory_threshold_bytes": memory_threshold_bytes,
        "session_warning_bytes": session_warning_bytes,
        "codex_process_count": len(ordered_processes),
        "aggregate_codex_rss_bytes": aggregate_rss,
        "largest_codex_process_rss_bytes": largest_process,
        "session_log_count": len(ordered_logs),
        "largest_session_log_bytes": largest_session_log,
        "session_log_warning": session_warning,
        "processes": [
            {
                "pid": process.pid,
                "rss_bytes": process.rss_bytes,
                "elapsed": process.elapsed,
                "kind": _command_kind(process.command),
            }
            for process in ordered_processes[:20]
        ],
        "largest_session_logs": [
            {"path": Path(log.path).name, "size_bytes": log.size_bytes} for log in ordered_logs[:10]
        ],
        "recommended_action": (
            "stop_current_task_before_more_tool_calls; close duplicate sessions, archive oversized logs, or restart Codex"
            if stop_required
            else "continue_with_bounded_outputs_and_no_parallel_duplicate_sessions"
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory-threshold-gib",
        type=float,
        default=8.0,
        help="stop threshold for aggregate or largest Codex RSS (default: 8 GiB)",
    )
    parser.add_argument(
        "--session-warning-mib",
        type=float,
        default=100.0,
        help="warning threshold for one session JSONL file (default: 100 MiB)",
    )
    parser.add_argument(
        "--sessions-root",
        type=Path,
        default=Path.home() / ".codex" / "sessions",
        help="Codex session root to stat without reading (default: ~/.codex/sessions)",
    )
    parser.add_argument("--json", action="store_true", help="emit one JSON report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    memory_threshold_bytes = int(args.memory_threshold_gib * GIB)
    session_warning_bytes = int(args.session_warning_mib * MIB)
    report = build_report(
        processes=collect_codex_processes(),
        session_logs=inspect_session_logs(args.sessions_root),
        memory_threshold_bytes=memory_threshold_bytes,
        session_warning_bytes=session_warning_bytes,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        print(
            f"status={report['status']} "
            f"aggregate_codex_rss_bytes={report['aggregate_codex_rss_bytes']} "
            f"largest_session_log_bytes={report['largest_session_log_bytes']}"
        )
        print(report["recommended_action"])
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
