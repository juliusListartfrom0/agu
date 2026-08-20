#!/usr/bin/env python3
"""Retry a resource-guarded command after system memory safely recovers."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Callable, Sequence
from typing import Protocol

import psutil

RESOURCE_LIMIT_EXIT_CODE = 75


class CompletedProcessLike(Protocol):
    returncode: int


def _available_memory_gib() -> float:
    return float(psutil.virtual_memory().available) / 1024**3


def _free_swap_gib() -> float:
    try:
        return float(psutil.swap_memory().free) / 1024**3
    except (OSError, psutil.Error):
        # Match the primary training guard: unknown swap is exhausted.
        # A positive threshold therefore remains fail-closed, while the
        # explicit zero default can still supervise physical memory.
        return 0.0


def _emit(event: str, **values: object) -> None:
    print(
        json.dumps({"event": event, **values}, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def _wait_for_resources(
    *,
    retry_min_available_gib: float,
    retry_min_free_swap_gib: float,
    retry_poll_seconds: float,
    resource_retry: int,
    available_memory_gib: Callable[[], float],
    free_swap_gib: Callable[[], float],
    sleep: Callable[[float], None],
) -> tuple[float, float]:
    while True:
        current_available_gib = available_memory_gib()
        current_free_swap_gib = free_swap_gib()
        if (
            current_available_gib >= retry_min_available_gib
            and current_free_swap_gib >= retry_min_free_swap_gib
        ):
            return current_available_gib, current_free_swap_gib
        _emit(
            "resumable_guard_waiting_for_resources",
            available_memory_gib=round(current_available_gib, 3),
            required_memory_gib=retry_min_available_gib,
            free_swap_gib=round(current_free_swap_gib, 3),
            required_free_swap_gib=retry_min_free_swap_gib,
            resource_retry=resource_retry,
        )
        sleep(retry_poll_seconds)


def run_with_resource_retries(
    command: Sequence[str],
    *,
    max_resource_retries: int,
    retry_min_available_gib: float,
    retry_min_free_swap_gib: float = 0.0,
    retry_poll_seconds: float,
    runner: Callable[[Sequence[str]], CompletedProcessLike] = lambda command: subprocess.run(
        command,
        check=False,
    ),
    available_memory_gib: Callable[[], float] = _available_memory_gib,
    free_swap_gib: Callable[[], float] = _free_swap_gib,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Run command until success/non-resource failure or the retry budget is exhausted."""

    resource_retries = 0
    current_available_gib, current_free_swap_gib = _wait_for_resources(
        retry_min_available_gib=retry_min_available_gib,
        retry_min_free_swap_gib=retry_min_free_swap_gib,
        retry_poll_seconds=retry_poll_seconds,
        resource_retry=resource_retries,
        available_memory_gib=available_memory_gib,
        free_swap_gib=free_swap_gib,
        sleep=sleep,
    )
    _emit(
        "resumable_guard_starting",
        available_memory_gib=round(current_available_gib, 3),
        required_memory_gib=retry_min_available_gib,
        free_swap_gib=round(current_free_swap_gib, 3),
        required_free_swap_gib=retry_min_free_swap_gib,
    )
    while True:
        result = runner(command)
        exit_code = int(result.returncode)
        if exit_code != RESOURCE_LIMIT_EXIT_CODE:
            return exit_code
        if resource_retries >= max_resource_retries:
            _emit(
                "resumable_guard_retry_limit_reached",
                exit_code=exit_code,
                resource_retries=resource_retries,
            )
            return exit_code

        resource_retries += 1
        current_available_gib, current_free_swap_gib = _wait_for_resources(
            retry_min_available_gib=retry_min_available_gib,
            retry_min_free_swap_gib=retry_min_free_swap_gib,
            retry_poll_seconds=retry_poll_seconds,
            resource_retry=resource_retries,
            available_memory_gib=available_memory_gib,
            free_swap_gib=free_swap_gib,
            sleep=sleep,
        )

        _emit(
            "resumable_guard_restarting",
            available_memory_gib=round(current_available_gib, 3),
            required_memory_gib=retry_min_available_gib,
            free_swap_gib=round(current_free_swap_gib, 3),
            required_free_swap_gib=retry_min_free_swap_gib,
            resource_retry=resource_retries,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-resource-retries", type=int, default=1000)
    parser.add_argument("--retry-min-available-gib", type=float, default=6.0)
    parser.add_argument("--retry-min-free-swap-gib", type=float, default=0.0)
    parser.add_argument("--retry-poll-seconds", type=float, default=10.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a guarded command is required after --")
    if args.max_resource_retries < 0:
        parser.error("--max-resource-retries must be non-negative")
    if args.retry_min_available_gib <= 0:
        parser.error("--retry-min-available-gib must be positive")
    if args.retry_min_free_swap_gib < 0:
        parser.error("--retry-min-free-swap-gib must be nonnegative")
    if args.retry_poll_seconds <= 0:
        parser.error("--retry-poll-seconds must be positive")
    return args


def main() -> int:
    args = parse_args()
    return run_with_resource_retries(
        args.command,
        max_resource_retries=args.max_resource_retries,
        retry_min_available_gib=args.retry_min_available_gib,
        retry_min_free_swap_gib=args.retry_min_free_swap_gib,
        retry_poll_seconds=args.retry_poll_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
