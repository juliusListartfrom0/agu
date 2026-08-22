#!/usr/bin/env python3
"""TASK-0258 Amendment-001 authorized pre-import bootstrap (FD plumbing).

Trusted-runner launch form:
``[PYTHON, "-P", "-S", "/dev/fd/203", "--task0258-review-request-fd", "202",
"--task0258-source-fd", "203"]``

This entry reads the ``agu.task0258-review-driver-request.v1`` from the request
fd, constructs the exact three-entry ``sys.path`` from the request's runtime
snapshot, and dispatches only ``runpy.run_module`` with the fixed target argv.
The pure core lives in ``app.analysis.task0258_v2_bootstrap``.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from collections.abc import Mapping

_PREIMPORT_MODULE_ID = "existing-45-temporal-retrospective"
_PREIMPORT_REQUEST_SCHEMA = "agu.task0258-review-driver-request.v1"
_PREIMPORT_REQUEST_FIELDS = {
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
_PREIMPORT_RUNTIME_RECEIPT_FIELDS = {
    "contract_absolute_path",
    "artifact_sha256",
    "file_sha256",
    "postpublication_free_bytes",
}


def _preimport_canonical_json(value: object) -> str:
    """Canonicalize the small JSON boundary before the sealed package imports."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _preimport_sha256(value: object) -> str:
    return hashlib.sha256(_preimport_canonical_json(value).encode("utf-8")).hexdigest()


def _preimport_open_absolute_file(path: object) -> int:
    if not isinstance(path, str) or not path.startswith("/") or os.path.normpath(path) != path:
        raise ValueError("pre-import path is not canonical absolute")
    components = path.split("/")[1:]
    if not components or any(not component for component in components):
        raise ValueError("pre-import path has an empty component")
    common_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags = common_flags | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = common_flags | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open("/", directory_flags)
    try:
        for component in components[:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        return os.open(components[-1], file_flags, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError("pre-import file cannot be opened without following links") from exc
    finally:
        os.close(parent_fd)


def _preimport_read_bounded_file(path: object) -> bytes:
    fd = _preimport_open_absolute_file(path)
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode) or stat.S_IMODE(file_stat.st_mode) != 0o600:
            raise ValueError("pre-import file kind or mode is invalid")
        if file_stat.st_size > 67_108_864:
            raise ValueError("pre-import file is too large")
        chunks: list[bytes] = []
        remaining = file_stat.st_size
        while remaining:
            chunk = os.read(fd, min(1 << 20, remaining))
            if not chunk:
                raise ValueError("pre-import file ended before its recorded size")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _preimport_parse_flags(argv: object) -> tuple[int, int]:
    expected = [
        "--task0258-review-request-fd",
        "202",
        "--task0258-source-fd",
        "203",
    ]
    if argv != expected:
        raise ValueError("pre-import bootstrap argv is not the fixed request/source map")
    return 202, 203


def _close_bootstrap_fds(request_fd: int | None, source_fd: int | None) -> None:
    """Close the sensitive review descriptors before dispatching target code."""
    for descriptor in (request_fd, source_fd):
        if descriptor is None:
            continue
        try:
            os.close(descriptor)
        except OSError:
            pass


def _preimport_validate_fds(request_fd: int, source_fd: int) -> None:
    request_stat = os.fstat(request_fd)
    source_stat = os.fstat(source_fd)
    if (request_stat.st_dev, request_stat.st_ino) == (source_stat.st_dev, source_stat.st_ino):
        raise ValueError("pre-import request/source descriptors alias")
    if stat.S_ISDIR(request_stat.st_mode) or stat.S_ISCHR(request_stat.st_mode):
        raise ValueError("pre-import request descriptor has an invalid kind")
    if not stat.S_ISREG(source_stat.st_mode) or source_stat.st_nlink != 1:
        raise ValueError("pre-import source descriptor is not a single-link regular file")


def _preimport_read_request(request_fd: int) -> tuple[bytes, Mapping[str, object]]:
    request_stat = os.fstat(request_fd)
    if stat.S_ISREG(request_stat.st_mode):
        if request_stat.st_size > 8_388_608:
            raise ValueError("review request is too large")
        os.lseek(request_fd, 0, os.SEEK_SET)
        request_bytes = os.read(request_fd, request_stat.st_size + 1)
    else:
        chunks: list[bytes] = []
        total = 0
        while total <= 8_388_608:
            chunk = os.read(request_fd, min(65_536, 8_388_609 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        request_bytes = b"".join(chunks)
    if not request_bytes or len(request_bytes) > 8_388_608 or not request_bytes.endswith(b"\n"):
        raise ValueError("review request framing is invalid")
    try:
        request = json.loads(request_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("review request JSON is invalid") from exc
    if not isinstance(request, Mapping) or set(request) != _PREIMPORT_REQUEST_FIELDS:
        raise ValueError("review request fields are invalid before import")
    if request["schema_version"] != _PREIMPORT_REQUEST_SCHEMA or request["module_id"] != _PREIMPORT_MODULE_ID:
        raise ValueError("review request identity is invalid before import")
    if request["artifact_sha256"] != _preimport_sha256(
        {key: value for key, value in request.items() if key != "artifact_sha256"}
    ):
        raise ValueError("review request artifact hash is invalid before import")
    if request_bytes != (_preimport_canonical_json(request) + "\n").encode("utf-8"):
        raise ValueError("review request is not canonical before import")
    return request_bytes, request


def _preimport_runtime_sys_path(request: Mapping[str, object]) -> list[str]:
    receipt = request["runtime_snapshot_receipt"]
    if not isinstance(receipt, Mapping) or set(receipt) != _PREIMPORT_RUNTIME_RECEIPT_FIELDS:
        raise ValueError("runtime snapshot receipt is invalid before import")
    contract_path = receipt["contract_absolute_path"]
    contract_bytes = _preimport_read_bounded_file(contract_path)
    if hashlib.sha256(contract_bytes).hexdigest() != receipt["file_sha256"]:
        raise ValueError("runtime contract file hash is invalid before import")
    try:
        contract = json.loads(contract_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("runtime contract JSON is invalid before import") from exc
    if not isinstance(contract, Mapping) or contract_bytes != _preimport_canonical_json(contract).encode("utf-8"):
        raise ValueError("runtime contract is not canonical before import")
    if contract.get("schema_version") != "agu.task0258-runtime-snapshot-contract.v1":
        raise ValueError("runtime contract schema is invalid before import")
    if contract.get("module_id") != _PREIMPORT_MODULE_ID:
        raise ValueError("runtime contract module identity is invalid before import")
    if contract.get("artifact_sha256") != _preimport_sha256(
        {key: value for key, value in contract.items() if key != "artifact_sha256"}
    ):
        raise ValueError("runtime contract artifact hash is invalid before import")
    runtime_root = contract.get("runtime_root_absolute_path")
    if (
        not isinstance(runtime_root, str)
        or not runtime_root.startswith("/")
        or os.path.normpath(runtime_root) != runtime_root
    ):
        raise ValueError("runtime root is invalid before import")
    entries = contract.get("ordered_python_sys_path_entries")
    expected = (
        (1, "base_runtime", "lib/python3.11"),
        (2, "base_runtime", "lib/python3.11/lib-dynload"),
        (3, "venv_site_packages", ""),
    )
    if not isinstance(entries, list) or len(entries) != len(expected):
        raise ValueError("runtime sys.path rows are invalid before import")
    paths: list[str] = []
    for row, (ordinal, root_id, relative_path) in zip(entries, expected):
        if not isinstance(row, Mapping) or (row.get("ordinal"), row.get("root_id"), row.get("relative_path")) != (
            ordinal,
            root_id,
            relative_path,
        ):
            raise ValueError("runtime sys.path row is invalid before import")
        paths.append(f"{runtime_root}/{root_id}" if not relative_path else f"{runtime_root}/{root_id}/{relative_path}")
    return paths


def main() -> int:
    request_fd: int | None = None
    source_fd: int | None = None
    try:
        request_fd, source_fd = _preimport_parse_flags(sys.argv[1:])
        _preimport_validate_fds(request_fd, source_fd)
        request_bytes, request = _preimport_read_request(request_fd)
        sys.path[:] = _preimport_runtime_sys_path(request)
    except (OSError, TypeError, ValueError):
        _close_bootstrap_fds(request_fd, source_fd)
        return 2

    try:
        from app.analysis.task0258_module_a_v2 import compact_canonical_json
        from app.analysis.task0258_v2_bootstrap import (
            build_sys_path_from_entries,
            dispatch_module,
            load_runtime_snapshot_contract,
            validate_bootstrap_fd_bindings,
            validate_review_driver_request,
        )

        validate_bootstrap_fd_bindings(request_fd, source_fd)
        validate_review_driver_request(request)
    except (ImportError, OSError, TypeError, ValueError):
        _close_bootstrap_fds(request_fd, source_fd)
        return 2

    try:
        request_stat = os.fstat(request_fd)
        if stat.S_ISREG(request_stat.st_mode):
            os.lseek(request_fd, 0, os.SEEK_SET)
            request_bytes = os.read(request_fd, request_stat.st_size + 1)
        if request_bytes != (compact_canonical_json(request) + "\n").encode("utf-8"):
            return 2
        source_stat = os.fstat(source_fd)
        if source_stat.st_size != request["review_bootstrap_source_size_bytes"]:
            return 2
        os.lseek(source_fd, 0, os.SEEK_SET)
        source_bytes = os.read(source_fd, source_stat.st_size + 1)
        if len(source_bytes) != source_stat.st_size:
            return 2
        if hashlib.sha256(source_bytes).hexdigest() != request["review_bootstrap_source_sha256"]:
            return 2
        runtime_contract = load_runtime_snapshot_contract(request["runtime_snapshot_receipt"])
        entries = runtime_contract["ordered_python_sys_path_entries"]
        runtime_root = runtime_contract["runtime_root_absolute_path"]
        sys.path[:] = build_sys_path_from_entries(runtime_root, entries)
        _close_bootstrap_fds(request_fd, source_fd)
        dispatch_module(request["target_module"], request["target_argv"])
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return 2
    finally:
        _close_bootstrap_fds(request_fd, source_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
