"""Tests for the read-only Codex memory guard."""

from __future__ import annotations

import json

from scripts import monitor_codex_memory as memory_guard


def test_parse_process_table_keeps_codex_processes_and_ignores_test_processes():
    output = "\n".join(
        [
            "101 1048576 00:01 /Applications/ChatGPT.app/Codex Framework/Codex (Renderer) --type=renderer",
            "102 2048 00:01 /Users/ppt/projects/agu/.venv/bin/pytest -q",
            "103 4096 00:01 /Users/ppt/projects/agu/scripts/monitor_codex_memory.py --json",
        ]
    )

    processes = memory_guard.parse_process_table(output)

    assert [(process.pid, process.rss_bytes) for process in processes] == [(101, 1048576 * 1024)]


def test_build_report_requires_stop_at_eight_gib_aggregate():
    processes = (
        memory_guard.ProcessMemory(pid=101, rss_bytes=5 * memory_guard.GIB, elapsed="00:01", command="Codex"),
        memory_guard.ProcessMemory(pid=102, rss_bytes=3 * memory_guard.GIB, elapsed="00:01", command="Codex"),
    )

    report = memory_guard.build_report(processes=processes, session_logs=())

    assert report["status"] == "stop_required"
    assert report["aggregate_codex_rss_bytes"] == 8 * memory_guard.GIB
    assert report["recommended_action"].startswith("stop_current_task")


def test_build_report_warns_on_large_session_without_stopping():
    processes = (memory_guard.ProcessMemory(pid=101, rss_bytes=1 * memory_guard.GIB, elapsed="00:01", command="Codex"),)
    session_logs = (memory_guard.SessionLog(path="/tmp/session.jsonl", size_bytes=101 * memory_guard.MIB),)

    report = memory_guard.build_report(processes=processes, session_logs=session_logs)

    assert report["status"] == "ok"
    assert report["session_log_warning"] is True
    assert report["largest_session_log_bytes"] == 101 * memory_guard.MIB


def test_main_emits_json_and_exit_code_for_stop_required(monkeypatch, capsys):
    monkeypatch.setattr(
        memory_guard,
        "collect_codex_processes",
        lambda: (
            memory_guard.ProcessMemory(pid=101, rss_bytes=8 * memory_guard.GIB, elapsed="00:01", command="Codex"),
        ),
    )
    monkeypatch.setattr(memory_guard, "inspect_session_logs", lambda root: ())

    exit_code = memory_guard.main(["--json"])

    assert exit_code == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "stop_required"
    assert report["exit_code"] == 2
