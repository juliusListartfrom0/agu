from __future__ import annotations

from types import SimpleNamespace

from scripts.run_resumable_guarded_command import run_with_resource_retries


def test_resumable_guard_retries_only_after_memory_recovers() -> None:
    exit_codes = iter((75, 75, 0))
    available_gib = iter((6.25, 4.0, 6.25, 7.0))
    runs: list[tuple[str, ...]] = []
    sleeps: list[float] = []

    exit_code = run_with_resource_retries(
        ("python", "worker.py"),
        max_resource_retries=3,
        retry_min_available_gib=6.0,
        retry_poll_seconds=5.0,
        runner=lambda command: (
            runs.append(tuple(command)) or SimpleNamespace(returncode=next(exit_codes))
        ),
        available_memory_gib=lambda: next(available_gib),
        sleep=sleeps.append,
    )

    assert exit_code == 0
    assert runs == [
        ("python", "worker.py"),
        ("python", "worker.py"),
        ("python", "worker.py"),
    ]
    assert sleeps == [5.0]


def test_resumable_guard_waits_for_memory_before_initial_run() -> None:
    available_gib = iter((4.5, 5.75, 6.1))
    runs: list[tuple[str, ...]] = []
    sleeps: list[float] = []

    exit_code = run_with_resource_retries(
        ("python", "worker.py"),
        max_resource_retries=0,
        retry_min_available_gib=6.0,
        retry_poll_seconds=5.0,
        runner=lambda command: (
            runs.append(tuple(command)) or SimpleNamespace(returncode=0)
        ),
        available_memory_gib=lambda: next(available_gib),
        sleep=sleeps.append,
    )

    assert exit_code == 0
    assert runs == [("python", "worker.py")]
    assert sleeps == [5.0, 5.0]


def test_resumable_guard_waits_for_swap_before_initial_run() -> None:
    free_swap_gib = iter((0.5, 0.8))
    runs: list[tuple[str, ...]] = []
    sleeps: list[float] = []

    exit_code = run_with_resource_retries(
        ("python", "worker.py"),
        max_resource_retries=0,
        retry_min_available_gib=6.0,
        retry_min_free_swap_gib=0.75,
        retry_poll_seconds=5.0,
        runner=lambda command: (
            runs.append(tuple(command)) or SimpleNamespace(returncode=0)
        ),
        available_memory_gib=lambda: 8.0,
        free_swap_gib=lambda: next(free_swap_gib),
        sleep=sleeps.append,
    )

    assert exit_code == 0
    assert runs == [("python", "worker.py")]
    assert sleeps == [5.0]


def test_resumable_guard_does_not_retry_non_resource_failure() -> None:
    runs: list[tuple[str, ...]] = []

    exit_code = run_with_resource_retries(
        ("python", "worker.py"),
        max_resource_retries=5,
        retry_min_available_gib=6.0,
        retry_poll_seconds=5.0,
        runner=lambda command: (
            runs.append(tuple(command)) or SimpleNamespace(returncode=7)
        ),
        available_memory_gib=lambda: 8.0,
        sleep=lambda _: None,
    )

    assert exit_code == 7
    assert runs == [("python", "worker.py")]


def test_resumable_guard_honors_retry_limit() -> None:
    runs: list[tuple[str, ...]] = []

    exit_code = run_with_resource_retries(
        ("python", "worker.py"),
        max_resource_retries=1,
        retry_min_available_gib=6.0,
        retry_poll_seconds=5.0,
        runner=lambda command: (
            runs.append(tuple(command)) or SimpleNamespace(returncode=75)
        ),
        available_memory_gib=lambda: 8.0,
        sleep=lambda _: None,
    )

    assert exit_code == 75
    assert runs == [("python", "worker.py"), ("python", "worker.py")]
