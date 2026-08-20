"""Tests for the TASK-0258 v2 bootstrap pure core."""

from __future__ import annotations

import pytest

from app.analysis.task0258_v2_bootstrap import (
    build_sys_path_from_entries,
    dispatch_module,
    parse_bootstrap_flags,
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
