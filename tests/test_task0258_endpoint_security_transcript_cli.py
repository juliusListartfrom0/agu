"""Tests for the diagnostic-only Endpoint Security transcript inspector."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts import inspect_endpoint_security_transcript as inspector


def _transcript() -> str:
    event_rows = (
        '{"event":"open","pid":42,"pidversion":7,"ppid":1,"seq_num":10,'
        '"global_seq_num":100,"path":"/private/tmp/input.json",'
        '"result_type":"auth","result_auth":"allow"}\n'
    )
    finalization = {
        "record_type": "final",
        "rows": 1,
        "bytes": len(event_rows.encode("utf-8")),
        "overflow": False,
        "sequence_gap": False,
        "protocol_error": False,
        "timed_out": False,
    }
    return event_rows + json.dumps(finalization, separators=(",", ":")) + "\n"


def test_inspector_reports_valid_diagnostic_transcript(monkeypatch, capsys, tmp_path: Path):
    transcript = tmp_path / "audit.jsonl"
    transcript.write_text(_transcript(), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["inspect", "--transcript", str(transcript), "--json"])

    assert inspector.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "event_count": 1,
        "evidence_class": "diagnostic_only",
        "finalization_verified": True,
        "p5_ready": False,
        "production_capability": False,
        "schema_version": "agu.task0258-endpoint-security-diagnostic-inspection.v1",
        "status": "valid_diagnostic_transcript",
    }


def test_inspector_rejects_truncated_transcript_without_authority(monkeypatch, capsys, tmp_path: Path):
    transcript = tmp_path / "truncated.jsonl"
    transcript.write_text(_transcript().splitlines()[0] + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["inspect", "--transcript", str(transcript), "--json"])

    assert inspector.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "invalid_diagnostic_transcript"
    assert report["production_capability"] is False
    assert report["p5_ready"] is False


def test_inspector_rejects_symlinked_transcript(monkeypatch, capsys, tmp_path: Path):
    real = tmp_path / "real.jsonl"
    real.write_text(_transcript(), encoding="utf-8")
    link = tmp_path / "link.jsonl"
    link.symlink_to(real)
    monkeypatch.setattr(sys, "argv", ["inspect", "--transcript", str(link), "--json"])

    assert inspector.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "invalid_diagnostic_transcript"
    assert report["production_capability"] is False
