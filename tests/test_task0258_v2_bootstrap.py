"""Tests for the TASK-0258 v2 bootstrap pure core."""

from __future__ import annotations

import pytest

from app.analysis.task0258_module_a_v2 import MODULE_ID, canonical_artifact_sha256
from app.analysis.task0258_v2_bootstrap import (
    build_sys_path_from_entries,
    dispatch_module,
    parse_bootstrap_flags,
    validate_review_driver_request,
    validate_target_module,
)


def test_parse_bootstrap_flags():
    assert parse_bootstrap_flags(["--task0258-review-request-fd", "202", "--task0258-source-fd", "203"]) == (202, 203)
    with pytest.raises(ValueError):
        parse_bootstrap_flags(["--task0258-review-request-fd", "202"])
    with pytest.raises(ValueError):
        parse_bootstrap_flags(["--bogus", "1"])
    with pytest.raises(ValueError):
        parse_bootstrap_flags(["--task0258-review-request-fd", "0", "--task0258-source-fd", "203"])


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


def test_review_driver_request_is_closed_and_target_bound():
    request = {
        "schema_version": "agu.task0258-review-driver-request.v1",
        "module_id": MODULE_ID,
        "review_execution_kind": "evidence",
        "check_name": "focused_pytest",
        "target_module": "pytest",
        "target_argv": ["-q", "tests/test_task0258_module_a_v2.py"],
        "runtime_snapshot_receipt": {},
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
    bad["target_argv"] = ["-c", "print('escape')"]
    bad["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in bad.items() if key != "artifact_sha256"}
    )
    with pytest.raises(ValueError):
        validate_review_driver_request(bad)
