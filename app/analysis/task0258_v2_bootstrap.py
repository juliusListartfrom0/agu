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

from app.analysis.task0258_module_a_v2 import (
    MODULE_ID,
    canonical_artifact_sha256,
    is_sha256,
)

REVIEW_DRIVER_REQUEST_SCHEMA = "agu.task0258-review-driver-request.v1"
REVIEW_CHECK_NAMES = frozenset({"focused_pytest", "full_pytest", "ruff_check", "ruff_format_check"})
REVIEW_DRIVER_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "module_id",
        "review_execution_kind",
        "check_name",
        "target_module",
        "target_argv",
        "runtime_snapshot_receipt",
        "namespace_provider_manifest_receipt",
        "expected_runtime_read_receipts",
        "review_bootstrap_source_size_bytes",
        "review_bootstrap_source_sha256",
        "artifact_sha256",
    }
)


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


def validate_review_driver_request(request: Mapping[str, object]) -> None:
    """Validate the closed request accepted by the pre-import bootstrap."""
    if not isinstance(request, Mapping) or set(request) != REVIEW_DRIVER_REQUEST_FIELDS:
        raise ValueError("review driver request field set is invalid")
    if request["schema_version"] != REVIEW_DRIVER_REQUEST_SCHEMA or request["module_id"] != MODULE_ID:
        raise ValueError("review driver request identity is invalid")
    if request["review_execution_kind"] not in {"discovery", "evidence"}:
        raise ValueError("review driver request execution kind is invalid")
    check_name = request["check_name"]
    if check_name not in REVIEW_CHECK_NAMES:
        raise ValueError("review driver request check name is invalid")
    target_module = request["target_module"]
    validate_target_module(target_module)
    expected_module = "pytest" if check_name.endswith("pytest") else "ruff"
    if target_module != expected_module:
        raise ValueError("review driver target module does not match the fixed check")
    target_argv = request["target_argv"]
    if (
        not isinstance(target_argv, (list, tuple))
        or not target_argv
        or any(not isinstance(value, str) or not value for value in target_argv)
    ):
        raise ValueError("review driver target argv is invalid")
    if any(value in {"-c", "-m", "--exec", "--pdb"} for value in target_argv):
        raise ValueError("review driver target argv contains an executable escape")
    runtime = request["runtime_snapshot_receipt"]
    if not isinstance(runtime, Mapping):
        raise ValueError("review driver runtime snapshot is invalid")
    for field in ("namespace_provider_manifest_receipt", "expected_runtime_read_receipts"):
        value = request[field]
        if value is not None and not isinstance(value, (Mapping, list, tuple)):
            raise ValueError(f"review driver {field} is invalid")
    size = request["review_bootstrap_source_size_bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError("review bootstrap source size is invalid")
    if not is_sha256(request["review_bootstrap_source_sha256"]):
        raise ValueError("review bootstrap source hash is invalid")
    expected_hash = canonical_artifact_sha256(
        {key: value for key, value in request.items() if key != "artifact_sha256"}
    )
    if request["artifact_sha256"] != expected_hash:
        raise ValueError("review driver request artifact hash is invalid")


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
    "validate_review_driver_request",
    "dispatch_module",
]
