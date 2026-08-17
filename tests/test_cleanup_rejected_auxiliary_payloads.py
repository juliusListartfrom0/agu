from __future__ import annotations

import json
from pathlib import Path

import scripts.cleanup_rejected_auxiliary_payloads as cleanup


def test_cleanup_dry_run_preserves_payload_and_execute_removes_only_targets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "frame.jpg").write_bytes(b"pixels")
    protected = tmp_path / "protected.json"
    protected.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(cleanup, "ROOT", tmp_path)
    monkeypatch.setattr(cleanup, "TARGETS", (target,))
    monkeypatch.setattr(cleanup, "PROTECTED", (protected,))

    dry_run = cleanup.build_audit(execute=False)
    assert dry_run["deleted_bytes"] == 0
    assert target.exists()
    assert protected.exists()

    executed = cleanup.build_audit(execute=True)
    assert executed["deleted_files"] == 1
    assert executed["deleted_bytes"] == len(b"pixels")
    assert not target.exists()
    assert protected.read_text(encoding="utf-8") == "{}\n"
    assert json.loads(json.dumps(executed))["audit_sha256"] == executed["audit_sha256"]
