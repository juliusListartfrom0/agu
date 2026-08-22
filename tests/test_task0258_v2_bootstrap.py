"""Tests for the TASK-0258 v2 bootstrap pure core."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from app.analysis.task0258_module_a_v2 import MODULE_ID, canonical_artifact_sha256, compact_canonical_json
from app.analysis.task0258_v2_bootstrap import (
    build_sys_path_from_entries,
    dispatch_module,
    expected_review_target_argv,
    load_runtime_snapshot_contract,
    parse_bootstrap_flags,
    validate_bootstrap_fd_bindings,
    validate_review_driver_request,
    validate_target_module,
)
from scripts import task0258_module_a_verified_bootstrap as bootstrap_script

_BOOTSTRAP_TARGET_SOURCE_SHA256 = {
    "app/__init__.py": "7d2e2db50d693717d0a15c16767025c05f0a7e6d5a202e4603ab2d752652bc4e",
    "app/version.py": "e13df818c99497bcc976ce37d152bf978b416ae9b4f8eafdd17b72104d41994d",
    "app/analysis/__init__.py": "5c2fc5f80f83f4e146ddb460ba922cb0eab5a9d14fdc4dbdc19bd544688ae65b",
    "app/analysis/task0258_module_a_v2.py": "24ce88717acc24f5a28bde690274db1172cbe1663a8bf0900c4cdbf280d3a06b",
    "app/analysis/task0258_v2_bootstrap.py": "c50ab5a67c259fe0f464e09a006ed42728face86ebac3beb043c08c9467279e5",
    "scripts/task0258_module_a_verified_bootstrap.py": "12e539c2bb3d1c27f986376643268e880faf9fa6f64fd1459bbf18aac9b3b905",
}


def test_parse_bootstrap_flags():
    assert parse_bootstrap_flags(["--task0258-review-request-fd", "202", "--task0258-source-fd", "203"]) == (202, 203)
    with pytest.raises(ValueError):
        parse_bootstrap_flags(["--task0258-review-request-fd", "202"])
    with pytest.raises(ValueError):
        parse_bootstrap_flags(["--bogus", "1"])
    with pytest.raises(ValueError):
        parse_bootstrap_flags(["--task0258-review-request-fd", "0", "--task0258-source-fd", "203"])
    with pytest.raises(ValueError):
        parse_bootstrap_flags(["--task0258-review-request-fd", "202", "--task0258-source-fd", "202"])
    with pytest.raises(ValueError):
        parse_bootstrap_flags(["--task0258-review-request-fd", "201", "--task0258-source-fd", "203"])


def test_validate_bootstrap_fd_bindings_rejects_aliases_and_non_regular_source(tmp_path):
    request_path = tmp_path / "request"
    source_path = tmp_path / "source"
    request_path.write_bytes(b"{}")
    source_path.write_bytes(b"source")
    request_fd = os.open(request_path, os.O_RDONLY)
    source_fd = os.open(source_path, os.O_RDONLY)
    try:
        validate_bootstrap_fd_bindings(request_fd, source_fd)
        alias_fd = os.dup(source_fd)
        try:
            with pytest.raises(ValueError):
                validate_bootstrap_fd_bindings(source_fd, alias_fd)
        finally:
            os.close(alias_fd)
    finally:
        os.close(request_fd)
    os.close(source_fd)
    directory_fd = os.open(tmp_path, os.O_RDONLY)
    request_fd = os.open(request_path, os.O_RDONLY)
    try:
        with pytest.raises(ValueError):
            validate_bootstrap_fd_bindings(request_fd, directory_fd)
    finally:
        os.close(request_fd)
        os.close(directory_fd)


def test_bootstrap_closes_sensitive_fds_before_dispatch(tmp_path):
    request_path = tmp_path / "request"
    source_path = tmp_path / "source"
    request_path.write_bytes(b"request")
    source_path.write_bytes(b"source")
    request_fd = os.open(request_path, os.O_RDONLY)
    source_fd = os.open(source_path, os.O_RDONLY)
    saved: dict[int, int | None] = {}
    try:
        for target_fd in (202, 203):
            try:
                saved[target_fd] = os.dup(target_fd)
            except OSError:
                saved[target_fd] = None
        os.dup2(request_fd, 202)
        os.dup2(source_fd, 203)
        bootstrap_script._close_bootstrap_fds(202, 203)
        with pytest.raises(OSError):
            os.fstat(202)
        with pytest.raises(OSError):
            os.fstat(203)
    finally:
        for target_fd in (203, 202):
            saved_fd = saved[target_fd]
            if saved_fd is None:
                try:
                    os.close(target_fd)
                except OSError:
                    pass
            else:
                os.dup2(saved_fd, target_fd)
                os.close(saved_fd)
        os.close(request_fd)
        os.close(source_fd)


def test_bootstrap_closes_fds_when_preimport_validation_fails(monkeypatch):
    closed = []
    monkeypatch.setattr(bootstrap_script, "_preimport_parse_flags", lambda _argv: (202, 203))

    def fail_validation(_request_fd, _source_fd):
        raise ValueError("invalid pre-import descriptors")

    monkeypatch.setattr(bootstrap_script, "_preimport_validate_fds", fail_validation)
    monkeypatch.setattr(
        bootstrap_script, "_close_bootstrap_fds", lambda request, source: closed.append((request, source))
    )
    assert bootstrap_script.main() == 2
    assert closed == [(202, 203)]


def test_bootstrap_executed_source_must_match_source_fd(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    other = tmp_path / "other.py"
    source.write_text("source\n")
    other.write_text("other\n")
    source_fd = os.open(source, os.O_RDONLY)
    try:
        monkeypatch.setattr(bootstrap_script, "__file__", str(other))
        with pytest.raises(ValueError, match="not bound"):
            bootstrap_script._preimport_validate_executed_source(source_fd)
    finally:
        os.close(source_fd)


def test_build_sys_path_from_entries():
    entries = [
        {"ordinal": 1, "root_id": "base_runtime", "relative_path": "lib/python3.11"},
        {"ordinal": 2, "root_id": "base_runtime", "relative_path": "lib/python3.11/lib-dynload"},
        {"ordinal": 3, "root_id": "venv_site_packages", "relative_path": ""},
    ]
    paths = build_sys_path_from_entries("/runtime", entries)
    assert paths == [
        "/runtime/base_runtime/lib/python3.11",
        "/runtime/base_runtime/lib/python3.11/lib-dynload",
        "/runtime/venv_site_packages",
    ]
    with pytest.raises(ValueError):
        build_sys_path_from_entries("/runtime", entries[:2])
    bad = list(entries)
    bad[0] = {"ordinal": 1, "root_id": "bogus", "relative_path": "x"}
    with pytest.raises(ValueError):
        build_sys_path_from_entries("/runtime", bad)


def test_validate_target_module():
    validate_target_module("pytest")
    with pytest.raises(ValueError):
        validate_target_module("scripts/x.py")
    with pytest.raises(ValueError):
        validate_target_module("x.py")
    with pytest.raises(ValueError):
        validate_target_module("")


def test_dispatch_module_rejects_bad_run_name():
    with pytest.raises(ValueError):
        dispatch_module("pytest", [], run_name="not-main")


def test_review_target_argv_is_the_closed_parent_command():
    assert expected_review_target_argv("focused_pytest") == [
        "-q",
        "-p",
        "no:cacheprovider",
        "--import-mode=importlib",
        "tests/test_vru_causal_temporal_feature_plan.py",
        "tests/test_vru_causal_tiled_swin_embeddings.py",
        "tests/test_vru_causal_temporal_retrospective.py",
        "tests/test_vru_causal_final_evaluator.py",
        "tests/test_task0258_module_a_cli.py",
    ]
    assert expected_review_target_argv("full_pytest") == [
        "-q",
        "-p",
        "no:cacheprovider",
        "--import-mode=importlib",
    ]
    ruff_paths = [
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
    ]
    assert expected_review_target_argv("ruff_check") == ["check", *ruff_paths]
    assert expected_review_target_argv("ruff_format_check") == ["format", "--check", *ruff_paths]
    with pytest.raises(ValueError):
        expected_review_target_argv("unknown")


def test_review_driver_request_is_closed_and_target_bound():
    request = {
        "schema_version": "agu.task0258-review-driver-request.v1",
        "module_id": MODULE_ID,
        "review_execution_kind": "evidence",
        "check_name": "focused_pytest",
        "target_module": "pytest",
        "target_argv": expected_review_target_argv("focused_pytest"),
        "runtime_snapshot_receipt": {
            "contract_absolute_path": "/runtime/contract.json",
            "artifact_sha256": "0" * 64,
            "file_sha256": "0" * 64,
            "postpublication_free_bytes": 1,
        },
        "namespace_provider_manifest_receipt": None,
        "expected_runtime_read_receipts": None,
        "review_bootstrap_source_size_bytes": 1,
        "review_bootstrap_source_sha256": "0" * 64,
    }
    request["artifact_sha256"] = canonical_artifact_sha256(request)
    validate_review_driver_request(request)
    bad = dict(request)
    bad["target_module"] = "ruff"
    with pytest.raises(ValueError):
        validate_review_driver_request(bad)
    bad = dict(request)
    bad["target_argv"] = [*request["target_argv"], "--pdb"]
    bad["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in bad.items() if key != "artifact_sha256"}
    )
    with pytest.raises(ValueError):
        validate_review_driver_request(bad)


def test_runtime_snapshot_receipt_reopens_the_bound_canonical_contract(tmp_path):
    contract = {
        "schema_version": "agu.task0258-runtime-snapshot-contract.v1",
        "module_id": MODULE_ID,
        "manifest_absolute_path": "/runtime/.task0258-runtime-snapshot-manifest.json",
        "manifest_artifact_sha256": "1" * 64,
        "manifest_file_sha256": "2" * 64,
        "runtime_root_absolute_path": "/runtime",
        "runtime_root_device": 1,
        "runtime_root_inode": 2,
        "ordered_mount_flags": ["nodev", "nosuid", "read_only"],
        "ordered_origin_root_rows": [],
        "ordered_python_sys_path_entries": [
            {"ordinal": 1, "root_id": "base_runtime", "relative_path": "lib/python3.11", "purpose": "stdlib"},
            {
                "ordinal": 2,
                "root_id": "base_runtime",
                "relative_path": "lib/python3.11/lib-dynload",
                "purpose": "dynload",
            },
            {"ordinal": 3, "root_id": "venv_site_packages", "relative_path": "", "purpose": "site_packages"},
        ],
        "python_snapshot_location": {},
        "python_executable_copy_receipt": {},
        "bootstrap_source_copy_receipt": {},
        "ordered_venv_executable_symlink_receipts": [],
        "ordered_tree_member_receipts": [],
        "tree_projection_sha256": "3" * 64,
        "ordered_subprocess_executable_receipts": [],
        "review_heavy_execution_policy_receipt": {},
        "os_system_runtime_receipt": {},
        "build_resource_limits": {},
        "build_prepublication_observation": {},
    }
    contract["artifact_sha256"] = canonical_artifact_sha256(contract)
    contract_path = tmp_path / "runtime-contract.json"
    contract_bytes = compact_canonical_json(contract).encode("utf-8")
    contract_path.write_bytes(contract_bytes)
    os.chmod(contract_path, 0o600)
    receipt = {
        "contract_absolute_path": str(contract_path),
        "artifact_sha256": contract["artifact_sha256"],
        "file_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "postpublication_free_bytes": 1,
    }
    assert load_runtime_snapshot_contract(receipt) == contract

    bad = dict(receipt)
    bad["file_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        load_runtime_snapshot_contract(bad)

    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    linked_receipt = dict(receipt)
    linked_receipt["contract_absolute_path"] = str(linked_parent / contract_path.name)
    with pytest.raises(ValueError):
        load_runtime_snapshot_contract(linked_receipt)


def test_verified_bootstrap_launcher_replays_fixed_fd_contract(tmp_path):
    """Exercise the real ``-P -S /dev/fd/203`` launcher locally."""
    repository_root = Path(__file__).resolve().parents[1]
    for relative_path, expected_sha256 in _BOOTSTRAP_TARGET_SOURCE_SHA256.items():
        source = repository_root / relative_path
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_sha256
    runtime_root = tmp_path / "runtime"
    base_runtime = runtime_root / "base_runtime"
    stdlib_parent = base_runtime / "lib"
    stdlib_parent.mkdir(parents=True)
    stdlib = stdlib_parent / "python3.11"
    shutil.copytree(Path(os.__file__).resolve().parent, stdlib, symlinks=False)
    site_packages = runtime_root / "venv_site_packages"
    for path in (site_packages,):
        path.mkdir(parents=True)

    fake_app = site_packages / "app"
    (fake_app / "analysis").mkdir(parents=True)
    for relative_path in (
        "app/__init__.py",
        "app/version.py",
        "app/analysis/__init__.py",
        "app/analysis/task0258_module_a_v2.py",
        "app/analysis/task0258_v2_bootstrap.py",
    ):
        destination = site_packages / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository_root / relative_path, destination)

    marker_path = tmp_path / "launcher-marker.json"
    fake_pytest = site_packages / "pytest"
    fake_pytest.mkdir()
    (fake_pytest / "__init__.py").write_text("", encoding="utf-8")
    (fake_pytest / "__main__.py").write_text(
        "import json, pathlib, sys\n"
        f"pathlib.Path({str(marker_path)!r}).write_text(json.dumps(sys.argv), encoding='utf-8')\n",
        encoding="utf-8",
    )

    contract = {
        "schema_version": "agu.task0258-runtime-snapshot-contract.v1",
        "module_id": MODULE_ID,
        "manifest_absolute_path": str(tmp_path / "manifest.json"),
        "manifest_artifact_sha256": "1" * 64,
        "manifest_file_sha256": "2" * 64,
        "runtime_root_absolute_path": str(runtime_root),
        "runtime_root_device": runtime_root.stat().st_dev,
        "runtime_root_inode": runtime_root.stat().st_ino,
        "ordered_mount_flags": ["nodev", "nosuid", "read_only"],
        "ordered_origin_root_rows": [],
        "ordered_python_sys_path_entries": [
            {"ordinal": 1, "root_id": "base_runtime", "relative_path": "lib/python3.11", "purpose": "stdlib"},
            {
                "ordinal": 2,
                "root_id": "base_runtime",
                "relative_path": "lib/python3.11/lib-dynload",
                "purpose": "dynload",
            },
            {"ordinal": 3, "root_id": "venv_site_packages", "relative_path": "", "purpose": "site_packages"},
        ],
        "python_snapshot_location": {},
        "python_executable_copy_receipt": {},
        "bootstrap_source_copy_receipt": {},
        "ordered_venv_executable_symlink_receipts": [],
        "ordered_tree_member_receipts": [],
        "tree_projection_sha256": "3" * 64,
        "ordered_subprocess_executable_receipts": [],
        "review_heavy_execution_policy_receipt": {},
        "os_system_runtime_receipt": {},
        "build_resource_limits": {},
        "build_prepublication_observation": {},
    }
    contract["artifact_sha256"] = canonical_artifact_sha256(contract)
    contract_path = tmp_path / "runtime-contract.json"
    contract_bytes = compact_canonical_json(contract).encode("utf-8")
    contract_path.write_bytes(contract_bytes)
    os.chmod(contract_path, 0o600)
    runtime_receipt = {
        "contract_absolute_path": str(contract_path),
        "artifact_sha256": contract["artifact_sha256"],
        "file_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "postpublication_free_bytes": 1,
    }

    source_path = repository_root / "scripts/task0258_module_a_verified_bootstrap.py"
    source_bytes = source_path.read_bytes()
    request = {
        "schema_version": "agu.task0258-review-driver-request.v1",
        "module_id": MODULE_ID,
        "review_execution_kind": "evidence",
        "check_name": "focused_pytest",
        "target_module": "pytest",
        "target_argv": expected_review_target_argv("focused_pytest"),
        "runtime_snapshot_receipt": runtime_receipt,
        "namespace_provider_manifest_receipt": None,
        "expected_runtime_read_receipts": None,
        "review_bootstrap_source_size_bytes": len(source_bytes),
        "review_bootstrap_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    }
    request["artifact_sha256"] = canonical_artifact_sha256(request)
    request_path = tmp_path / "request.json"
    request_path.write_bytes((compact_canonical_json(request) + "\n").encode("utf-8"))

    request_fd = os.open(request_path, os.O_RDONLY)
    source_fd = os.open(source_path, os.O_RDONLY)
    while request_fd in (202, 203):
        duplicate = os.dup(request_fd)
        os.close(request_fd)
        request_fd = duplicate
    while source_fd in (202, 203):
        duplicate = os.dup(source_fd)
        os.close(source_fd)
        source_fd = duplicate
    saved_fds: dict[int, int | None] = {}
    try:
        for target_fd in (202, 203):
            try:
                saved_fds[target_fd] = os.dup(target_fd)
            except OSError:
                saved_fds[target_fd] = None
        os.dup2(request_fd, 202, inheritable=True)
        os.dup2(source_fd, 203, inheritable=True)
        completed = subprocess.run(
            [
                sys.executable,
                "-P",
                "-S",
                "/dev/fd/203",
                "--task0258-review-request-fd",
                "202",
                "--task0258-source-fd",
                "203",
            ],
            cwd=repository_root,
            pass_fds=(202, 203),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    finally:
        for target_fd in (203, 202):
            saved_fd = saved_fds[target_fd]
            if saved_fd is None:
                os.close(target_fd)
            else:
                os.dup2(saved_fd, target_fd, inheritable=True)
                os.close(saved_fd)
        os.close(request_fd)
        os.close(source_fd)

    assert completed.returncode == 0, completed.stderr
    observed_argv = json.loads(marker_path.read_text(encoding="utf-8"))
    assert observed_argv[1:] == request["target_argv"]
