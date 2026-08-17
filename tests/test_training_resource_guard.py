from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from app.training.resource_guard import (
    ResourceSnapshot,
    TrainingResourceGuard,
    TrainingResourceThresholds,
    sample_training_resources,
)
from scripts import run_guarded_training as guarded_training_cli


def _guard_cli_args(*, log_path: Path, sleep_seconds: float = 30.0) -> Namespace:
    return Namespace(
        max_memory_percent=90.0,
        min_available_gib=2.0,
        min_free_swap_gib=0.0,
        max_cpu_percent=95.0,
        consecutive_breaches=3,
        sample_interval_sec=0.01,
        log=log_path,
        command=[
            sys.executable,
            "-c",
            f"import time; time.sleep({sleep_seconds!r})",
        ],
    )


def _record_spawned_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> list[subprocess.Popen[bytes]]:
    real_popen = subprocess.Popen
    processes: list[subprocess.Popen[bytes]] = []

    def recording_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(guarded_training_cli.subprocess, "Popen", recording_popen)
    return processes


def _cleanup_spawned_processes(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        process.wait(timeout=5)


def _snapshot(
    *,
    memory_percent: float = 50.0,
    available_gib: float = 8.0,
    cpu_percent: float = 40.0,
    free_swap_gib: float = 8.0,
) -> ResourceSnapshot:
    return ResourceSnapshot(
        sampled_at="2026-07-24T12:00:00Z",
        system_memory_percent=memory_percent,
        available_memory_bytes=int(available_gib * 1024**3),
        system_cpu_percent=cpu_percent,
        process_tree_rss_bytes=512 * 1024**2,
        swap_free_bytes=int(free_swap_gib * 1024**3),
        swap_percent=10.0,
    )


def test_resource_thresholds_reject_unsafe_or_nonsensical_values() -> None:
    with pytest.raises(ValueError, match="max_system_memory_percent"):
        TrainingResourceThresholds(max_system_memory_percent=100.0)
    with pytest.raises(ValueError, match="min_available_memory_gib"):
        TrainingResourceThresholds(min_available_memory_gib=-1.0)
    with pytest.raises(ValueError, match="min_free_swap_gib"):
        TrainingResourceThresholds(min_free_swap_gib=-1.0)
    with pytest.raises(ValueError, match="consecutive_breaches"):
        TrainingResourceThresholds(consecutive_breaches=0)


def test_resource_guard_stops_only_after_sustained_cpu_pressure() -> None:
    guard = TrainingResourceGuard(
        TrainingResourceThresholds(
            max_system_cpu_percent=90.0,
            consecutive_breaches=3,
        )
    )

    first = guard.observe(_snapshot(cpu_percent=96.0), stage="train")
    second = guard.observe(_snapshot(cpu_percent=97.0), stage="train")
    third = guard.observe(_snapshot(cpu_percent=98.0), stage="train")

    assert first.should_stop is False
    assert second.should_stop is False
    assert third.should_stop is True
    assert third.consecutive_breaches == 3
    assert third.reasons == ("system_cpu_percent>90.0",)


def test_resource_guard_recovery_resets_consecutive_breach_counter() -> None:
    guard = TrainingResourceGuard(
        TrainingResourceThresholds(
            max_system_memory_percent=85.0,
            consecutive_breaches=2,
        )
    )

    assert guard.observe(_snapshot(memory_percent=91.0), stage="epoch-1").consecutive_breaches == 1
    assert guard.observe(_snapshot(memory_percent=70.0), stage="epoch-1").consecutive_breaches == 0
    assert guard.observe(_snapshot(memory_percent=92.0), stage="epoch-2").should_stop is False


def test_resource_guard_treats_low_available_memory_as_a_breach() -> None:
    guard = TrainingResourceGuard(
        TrainingResourceThresholds(
            min_available_memory_gib=2.0,
            consecutive_breaches=1,
        )
    )

    decision = guard.observe(_snapshot(available_gib=1.5), stage="fit")

    assert decision.should_stop is True
    assert decision.reasons == ("available_memory_gib<2.000",)


def test_resource_guard_treats_low_free_swap_as_a_breach() -> None:
    guard = TrainingResourceGuard(
        TrainingResourceThresholds(
            min_free_swap_gib=0.75,
            consecutive_breaches=1,
        )
    )

    decision = guard.observe(_snapshot(free_swap_gib=0.5), stage="fit")

    assert decision.should_stop is True
    assert decision.reasons == ("free_swap_gib<0.750",)


def test_resource_sampling_marks_unavailable_swap_as_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import psutil

    def raise_os_error() -> None:
        raise OSError("swap metrics unavailable")

    monkeypatch.setattr(psutil, "swap_memory", raise_os_error)

    snapshot = sample_training_resources(os.getpid())

    assert snapshot.swap_free_bytes == 0
    assert snapshot.swap_percent == 100.0


def test_resource_decision_serializes_as_structured_json() -> None:
    guard = TrainingResourceGuard(
        TrainingResourceThresholds(consecutive_breaches=2)
    )

    decision = guard.observe(_snapshot(), stage="startup")
    payload = json.loads(decision.to_json())

    assert payload["event"] == "training_resource_sample"
    assert payload["stage"] == "startup"
    assert payload["should_stop"] is False
    assert payload["snapshot"]["process_tree_rss_bytes"] == 512 * 1024**2


def test_guarded_training_opens_log_before_spawning_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = tmp_path / "resources.jsonl"
    processes = _record_spawned_processes(monkeypatch)
    monkeypatch.setattr(
        guarded_training_cli,
        "parse_args",
        lambda: _guard_cli_args(log_path=log_path),
    )
    real_path_open = Path.open

    def fail_log_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == log_path:
            raise OSError("injected log open failure")
        return real_path_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_log_open)

    try:
        with pytest.raises(OSError, match="injected log open failure"):
            guarded_training_cli.main()
        assert processes == []
    finally:
        _cleanup_spawned_processes(processes)


def test_guarded_training_reaps_child_when_log_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = tmp_path / "resources.jsonl"
    processes = _record_spawned_processes(monkeypatch)
    monkeypatch.setattr(
        guarded_training_cli,
        "parse_args",
        lambda: _guard_cli_args(log_path=log_path),
    )
    real_path_open = Path.open

    class FailingWriteStream:
        def __enter__(self) -> FailingWriteStream:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def write(self, _value: str) -> int:
            raise OSError("injected log write failure")

        def flush(self) -> None:
            return None

        def fileno(self) -> int:
            raise AssertionError("fileno must not be reached after a write failure")

    def open_failing_stream(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == log_path:
            return FailingWriteStream()
        return real_path_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_failing_stream)

    try:
        with pytest.raises(OSError, match="injected log write failure"):
            guarded_training_cli.main()
        assert len(processes) == 1
        assert processes[0].poll() is not None
    finally:
        _cleanup_spawned_processes(processes)


def test_guarded_training_reaps_child_when_log_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = tmp_path / "resources.jsonl"
    processes = _record_spawned_processes(monkeypatch)
    monkeypatch.setattr(
        guarded_training_cli,
        "parse_args",
        lambda: _guard_cli_args(log_path=log_path, sleep_seconds=0.1),
    )

    def fail_fsync(_fd: int) -> None:
        raise OSError("injected log fsync failure")

    monkeypatch.setattr(guarded_training_cli.os, "fsync", fail_fsync)

    try:
        with pytest.raises(OSError, match="injected log fsync failure"):
            guarded_training_cli.main()
        assert len(processes) == 1
        assert processes[0].poll() is not None
    finally:
        _cleanup_spawned_processes(processes)


def test_guarded_training_reaps_child_when_resource_sampling_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = tmp_path / "resources.jsonl"
    processes = _record_spawned_processes(monkeypatch)
    monkeypatch.setattr(
        guarded_training_cli,
        "parse_args",
        lambda: _guard_cli_args(log_path=log_path),
    )

    def fail_sampling(_process_id: int) -> ResourceSnapshot:
        raise RuntimeError("injected resource sampling failure")

    monkeypatch.setattr(
        guarded_training_cli,
        "sample_training_resources",
        fail_sampling,
    )

    try:
        assert guarded_training_cli.main() == 75
        assert len(processes) == 1
        assert processes[0].poll() is not None
        events = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert events[-1] == {
            "error_type": "RuntimeError",
            "event": "training_resource_guard_error",
            "exit_code": 75,
        }
    finally:
        _cleanup_spawned_processes(processes)


def test_guarded_training_propagates_child_exit_and_writes_lifecycle(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "resources.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_guarded_training.py",
            "--sample-interval-sec",
            "0.01",
            "--log",
            str(log_path),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(7)",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert result.returncode == 7
    assert events[0]["event"] == "training_resource_guard_started"
    assert "command" not in events[0]
    assert len(events[0]["command_sha256"]) == 64
    assert events[-1] == {
        "event": "training_resource_guard_finished",
        "exit_code": 7,
    }


def test_guarded_training_terminates_its_child_after_memory_pressure(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "resources.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_guarded_training.py",
            "--min-available-gib",
            "1000000",
            "--consecutive-breaches",
            "1",
            "--sample-interval-sec",
            "0.01",
            "--log",
            str(log_path),
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert result.returncode == 75
    assert events[-1]["event"] == "training_resource_guard_stopped"
    assert events[-1]["exit_code"] == 75
    # An unrelated concurrent workload may also breach the CPU threshold while
    # this test forces the memory threshold. The guard must report the forced
    # memory breach; additional simultaneous reasons are valid evidence.
    assert "available_memory_gib<1000000.000" in events[-1]["reasons"]
