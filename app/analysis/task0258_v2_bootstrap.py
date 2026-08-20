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

import hashlib
import json
import os
import runpy
import stat
from collections.abc import Mapping, Sequence

from app.analysis.task0258_module_a_v2 import (
    MODULE_ID,
    canonical_artifact_sha256,
    compact_canonical_json,
    is_sha256,
)

REVIEW_DRIVER_REQUEST_SCHEMA = "agu.task0258-review-driver-request.v1"
REVIEW_CHECK_NAMES = frozenset({"focused_pytest", "full_pytest", "ruff_check", "ruff_format_check"})

_FOCUSED_PYTEST_TARGETS = (
    "tests/test_vru_causal_temporal_feature_plan.py",
    "tests/test_vru_causal_tiled_swin_embeddings.py",
    "tests/test_vru_causal_temporal_retrospective.py",
    "tests/test_vru_causal_final_evaluator.py",
    "tests/test_task0258_module_a_cli.py",
)
_RUFF_TARGETS = (
    "app/analysis/vru_causal_temporal_retrospective.py",
    "scripts/seal_vru_causal_temporal_feature_plan.py",
    "scripts/extract_vru_causal_tiled_swin_embeddings.py",
    "scripts/screen_vru_causal_temporal_retrospective.py",
    "scripts/task0258_module_a_verified_bootstrap.py",
    "tests/test_vru_causal_temporal_feature_plan.py",
    "tests/test_vru_causal_tiled_swin_embeddings.py",
    "tests/test_vru_causal_temporal_retrospective.py",
    "tests/test_vru_causal_final_evaluator.py",
    "tests/test_task0258_module_a_cli.py",
)
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
RUNTIME_SNAPSHOT_RECEIPT_FIELDS = frozenset(
    {"contract_absolute_path", "artifact_sha256", "file_sha256", "postpublication_free_bytes"}
)
RUNTIME_SNAPSHOT_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "module_id",
        "manifest_absolute_path",
        "manifest_artifact_sha256",
        "manifest_file_sha256",
        "runtime_root_absolute_path",
        "runtime_root_device",
        "runtime_root_inode",
        "ordered_mount_flags",
        "ordered_origin_root_rows",
        "ordered_python_sys_path_entries",
        "python_snapshot_location",
        "python_executable_copy_receipt",
        "bootstrap_source_copy_receipt",
        "ordered_venv_executable_symlink_receipts",
        "ordered_tree_member_receipts",
        "tree_projection_sha256",
        "ordered_subprocess_executable_receipts",
        "review_heavy_execution_policy_receipt",
        "os_system_runtime_receipt",
        "build_resource_limits",
        "build_prepublication_observation",
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
    request_fd = flags["--task0258-review-request-fd"]
    source_fd = flags["--task0258-source-fd"]
    if request_fd == source_fd:
        raise ValueError("bootstrap request-fd and source-fd must be distinct")
    if (request_fd, source_fd) != (202, 203):
        raise ValueError("bootstrap fd assignment is not the fixed 202/203 map")
    return request_fd, source_fd


def expected_review_target_argv(check_name: object) -> list[str]:
    """Return the exact target argv authorized by the parent review command."""
    if check_name == "focused_pytest":
        return ["-q", "-p", "no:cacheprovider", "--import-mode=importlib", *_FOCUSED_PYTEST_TARGETS]
    if check_name == "full_pytest":
        return ["-q", "-p", "no:cacheprovider", "--import-mode=importlib"]
    if check_name == "ruff_check":
        return ["check", *_RUFF_TARGETS]
    if check_name == "ruff_format_check":
        return ["format", "--check", *_RUFF_TARGETS]
    raise ValueError(f"review check name is not fixed: {check_name!r}")


def _validate_absolute_path_syntax(path: str) -> None:
    """Reject non-canonical absolute paths before descriptor-relative opening."""
    if not isinstance(path, str) or not path.startswith("/") or "\x00" in path:
        raise ValueError("runtime contract path is not absolute")
    if os.path.normpath(path) != path:
        raise ValueError("runtime contract path is not canonical")


def _open_no_follow_absolute_file(path: str) -> int:
    """Open an absolute regular-file path through a descriptor-relative chain."""
    _validate_absolute_path_syntax(path)
    components = path.split("/")[1:]
    if not components or any(not component for component in components):
        raise ValueError("runtime contract path contains an empty component")
    common_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags = common_flags | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    no_follow_flags = common_flags | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open("/", directory_flags)
    try:
        for component in components[:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        result_fd = os.open(components[-1], no_follow_flags, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError("runtime contract path cannot be opened without following links") from exc
    finally:
        os.close(parent_fd)
    return result_fd


def load_runtime_snapshot_contract(receipt: Mapping[str, object]) -> Mapping[str, object]:
    """Reopen and verify the standalone runtime contract bound by a small receipt."""
    if not isinstance(receipt, Mapping) or set(receipt) != RUNTIME_SNAPSHOT_RECEIPT_FIELDS:
        raise ValueError("runtime snapshot receipt is invalid")
    if not is_sha256(receipt["artifact_sha256"]) or not is_sha256(receipt["file_sha256"]):
        raise ValueError("runtime snapshot receipt hashes are invalid")
    if (
        not isinstance(receipt["postpublication_free_bytes"], int)
        or isinstance(receipt["postpublication_free_bytes"], bool)
        or receipt["postpublication_free_bytes"] < 0
    ):
        raise ValueError("runtime snapshot receipt free-byte observation is invalid")
    contract_path = receipt["contract_absolute_path"]
    if not isinstance(contract_path, str):
        raise ValueError("runtime contract path is invalid")
    try:
        fd = _open_no_follow_absolute_file(contract_path)
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("runtime contract cannot be opened without following links") from exc
    try:
        file_stat = os.fstat(fd)
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or stat.S_IMODE(file_stat.st_mode) != 0o600
            or file_stat.st_size > 67_108_864
        ):
            raise ValueError("runtime contract file kind or size is invalid")
        chunks: list[bytes] = []
        remaining = file_stat.st_size
        while remaining:
            chunk = os.read(fd, min(1 << 20, remaining))
            if not chunk:
                raise ValueError("runtime contract ended before its recorded size")
            chunks.append(chunk)
            remaining -= len(chunk)
        contract_bytes = b"".join(chunks)
    finally:
        os.close(fd)
    if hashlib.sha256(contract_bytes).hexdigest() != receipt["file_sha256"]:
        raise ValueError("runtime contract file hash does not match its receipt")
    try:
        contract = json.loads(contract_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("runtime contract is not canonical JSON") from exc
    if not isinstance(contract, Mapping) or set(contract) != RUNTIME_SNAPSHOT_CONTRACT_FIELDS:
        raise ValueError("runtime contract field set is invalid")
    if contract_bytes != compact_canonical_json(contract).encode("utf-8"):
        raise ValueError("runtime contract bytes are not compact canonical JSON")
    if contract["artifact_sha256"] != canonical_artifact_sha256(
        {key: value for key, value in contract.items() if key != "artifact_sha256"}
    ):
        raise ValueError("runtime contract artifact hash is invalid")
    if contract["artifact_sha256"] != receipt["artifact_sha256"]:
        raise ValueError("runtime contract artifact hash does not match its receipt")
    if contract["schema_version"] != "agu.task0258-runtime-snapshot-contract.v1":
        raise ValueError("runtime contract schema is invalid")
    if contract["module_id"] != MODULE_ID:
        raise ValueError("runtime contract module identity is invalid")
    for field in ("manifest_absolute_path", "runtime_root_absolute_path"):
        value = contract[field]
        if not isinstance(value, str) or not value.startswith("/") or os.path.normpath(value) != value:
            raise ValueError(f"runtime contract {field} is invalid")
    for field in ("manifest_artifact_sha256", "manifest_file_sha256", "tree_projection_sha256"):
        if not is_sha256(contract[field]):
            raise ValueError(f"runtime contract {field} is invalid")
    for field in ("runtime_root_device", "runtime_root_inode"):
        value = contract[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"runtime contract {field} is invalid")
    if contract["ordered_mount_flags"] != ["nodev", "nosuid", "read_only"]:
        raise ValueError("runtime contract mount flags are invalid")
    entries = contract["ordered_python_sys_path_entries"]
    if not isinstance(entries, list) or len(entries) != 3:
        raise ValueError("runtime contract Python sys.path entries are invalid")
    expected = (
        (1, "base_runtime", "lib/python3.11", "stdlib"),
        (2, "base_runtime", "lib/python3.11/lib-dynload", "dynload"),
        (3, "venv_site_packages", "", "site_packages"),
    )
    for row, (ordinal, root_id, relative_path, purpose) in zip(entries, expected):
        if not isinstance(row, Mapping) or set(row) != {"ordinal", "root_id", "relative_path", "purpose"}:
            raise ValueError("runtime contract Python sys.path row is invalid")
        if (row["ordinal"], row["root_id"], row["relative_path"], row["purpose"]) != (
            ordinal,
            root_id,
            relative_path,
            purpose,
        ):
            raise ValueError("runtime contract Python sys.path row is not the fixed entry")
    return contract


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
    if not isinstance(check_name, str) or check_name not in REVIEW_CHECK_NAMES:
        raise ValueError("review driver request check name is invalid")
    target_module = request["target_module"]
    validate_target_module(target_module)
    expected_module = "pytest" if check_name.endswith("pytest") else "ruff"
    if target_module != expected_module:
        raise ValueError("review driver target module does not match the fixed check")
    target_argv = request["target_argv"]
    if not isinstance(target_argv, (list, tuple)) or tuple(target_argv) != tuple(
        expected_review_target_argv(check_name)
    ):
        raise ValueError("review driver target argv is not the fixed parent command")
    runtime = request["runtime_snapshot_receipt"]
    if not isinstance(runtime, Mapping) or set(runtime) != RUNTIME_SNAPSHOT_RECEIPT_FIELDS:
        raise ValueError("review driver runtime snapshot is invalid")
    if (
        not isinstance(runtime["contract_absolute_path"], str)
        or not runtime["contract_absolute_path"].startswith("/")
        or not is_sha256(runtime["artifact_sha256"])
        or not is_sha256(runtime["file_sha256"])
        or not isinstance(runtime["postpublication_free_bytes"], int)
        or isinstance(runtime["postpublication_free_bytes"], bool)
        or runtime["postpublication_free_bytes"] < 0
    ):
        raise ValueError("review driver runtime snapshot receipt is invalid")
    namespace_receipt = request["namespace_provider_manifest_receipt"]
    expected_reads = request["expected_runtime_read_receipts"]
    if namespace_receipt is None:
        if expected_reads is not None:
            raise ValueError("runtime read receipts require a namespace manifest receipt")
    else:
        if not isinstance(namespace_receipt, Mapping) or set(namespace_receipt) != {
            "artifact_sha256",
            "file_sha256",
        }:
            raise ValueError("namespace provider manifest receipt is invalid")
        if not is_sha256(namespace_receipt["artifact_sha256"]) or not is_sha256(namespace_receipt["file_sha256"]):
            raise ValueError("namespace provider manifest receipt hashes are invalid")
        if not isinstance(expected_reads, (list, tuple)):
            raise ValueError("expected runtime read receipts must be an array")
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
    "expected_review_target_argv",
    "load_runtime_snapshot_contract",
    "build_sys_path_from_entries",
    "validate_target_module",
    "validate_review_driver_request",
    "dispatch_module",
]
