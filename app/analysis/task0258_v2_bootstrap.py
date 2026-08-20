"""TASK-0258 Amendment-001 v2 authorized pre-import bootstrap — pure core.

The trusted runner launches
``[PYTHON, "-P", "-S", "/dev/fd/203", "--task0258-review-request-fd", "202",
"--task0258-source-fd", "203"]``; the bootstrap reads a
``agu.task0258-review-driver-request.v1`` from FD 202, constructs the exact
three-entry ``sys.path`` from the runtime snapshot, and dispatches only
``runpy.run_module(target_module, run_name="__main__", alter_sys=True)``.

This module holds the pure, testable core (flag parsing, sys.path construction,
target validation, guarded dispatch). The FD/OS plumbing lives in
``scripts/task0258_module_a_verified_bootstrap.py``.
"""

from __future__ import annotations

import runpy
from collections.abc import Mapping, Sequence


def parse_bootstrap_flags(argv: Sequence[str]) -> tuple[int, int]:
    """Parse ``--task0258-review-request-fd N --task0258-source-fd N``.

    Returns ``(request_fd, source_fd)``. Any other flag, missing value, or a
    nonpositive/duplicate fd is rejected.
    """
    if argv is None or not isinstance(argv, (list, tuple)):
        raise ValueError("bootstrap argv is invalid")
    values = list(argv)
    flags = {}
    index = 0
    while index < len(values):
        token = values[index]
        if token in ("--task0258-review-request-fd", "--task0258-source-fd"):
            if index + 1 >= len(values):
                raise ValueError(f"bootstrap flag {token} is missing its value")
            raw = values[index + 1]
            try:
                fd = int(raw, 10)
            except (TypeError, ValueError):
                raise ValueError(f"bootstrap flag {token} has a non-integer value") from None
            if fd <= 0:
                raise ValueError(f"bootstrap flag {token} must be a positive fd")
            if token in flags:
                raise ValueError(f"bootstrap flag {token} is duplicated")
            flags[token] = fd
            index += 2
        else:
            raise ValueError(f"unknown bootstrap argument: {token!r}")
    if set(flags) != {
        "--task0258-review-request-fd",
        "--task0258-source-fd",
    }:
        raise ValueError("bootstrap must receive both request-fd and source-fd")
    return flags["--task0258-review-request-fd"], flags["--task0258-source-fd"]


def build_sys_path_from_entries(
    runtime_root: str,
    entries: object,
) -> list[str]:
    """Construct the exact three-entry ``sys.path``.

    ``entries`` must be the frozen ordered rows
    ``[{ordinal=1,root_id=base_runtime,relative_path=lib/python3.11},
    {ordinal=2,root_id=base_runtime,relative_path=lib/python3.11/lib-dynload},
    {ordinal=3,root_id=venv_site_packages,relative_path=""}]`` in this order.
    Each maps to ``<runtime_root>/<root_id>/<relative_path>``.
    """
    expected = (
        (1, "base_runtime", "lib/python3.11"),
        (2, "base_runtime", "lib/python3.11/lib-dynload"),
        (3, "venv_site_packages", ""),
    )
    if not isinstance(entries, (list, tuple)) or len(entries) != 3:
        raise ValueError("runtime snapshot must have exactly three sys.path entries")
    paths: list[str] = []
    for row, (ordinal, root_id, rel) in zip(entries, expected):
        if not isinstance(row, Mapping):
            raise ValueError("sys.path entry is invalid")
        if row.get("ordinal") != ordinal:
            raise ValueError("sys.path entry ordinal is invalid")
        if row.get("root_id") != root_id or row.get("relative_path") != rel:
            raise ValueError("sys.path entry root/path is invalid")
        paths.append(runtime_root + "/" + root_id if rel == "" else f"{runtime_root}/{root_id}/{rel}")
    return paths


def validate_target_module(target_module: object) -> None:
    """Reject path-backed, suffixed, or absolute module targets."""
    if not isinstance(target_module, str) or not target_module:
        raise ValueError("target module is invalid")
    if "/" in target_module or "\\" in target_module:
        raise ValueError("target module must not be path-backed")
    if target_module.endswith((".py", ".pyc", ".pyw")) or target_module.endswith(".pyi"):
        raise ValueError("target module must not carry a file suffix")
    if target_module.startswith(".") or ":" in target_module:
        raise ValueError("target module is invalid")


def dispatch_module(
    target_module: str,
    target_argv: Sequence[str],
    *,
    run_name: str = "__main__",
) -> Mapping[str, object]:
    """Dispatch exactly ``runpy.run_module`` with the fixed target argv.

    ``target_argv`` replaces ``sys.argv[1:]`` before dispatch. ``run_name`` is
    pinned to ``__main__``; any other run name is rejected.
    """
    if run_name != "__main__":
        raise ValueError(f"run_name must be __main__, got {run_name!r}")
    validate_target_module(target_module)
    import sys

    prior = sys.argv[:]
    try:
        sys.argv = [target_module, *list(target_argv)]
        result = runpy.run_module(target_module, run_name="__main__", alter_sys=True)
    finally:
        sys.argv = prior
    return result


__all__ = [
    "parse_bootstrap_flags",
    "build_sys_path_from_entries",
    "validate_target_module",
    "dispatch_module",
]
