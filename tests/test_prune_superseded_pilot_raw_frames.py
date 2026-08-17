from __future__ import annotations

from pathlib import Path

import scripts.prune_superseded_pilot_raw_frames as cleanup


def _fixture(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path]:
    root = tmp_path / "repo"
    public = root / "analysis_outputs/public_research"
    monkeypatch.setattr(cleanup, "ROOT", root)
    monkeypatch.setattr(cleanup, "PUBLIC", public)

    old_a = public / f"{cleanup.PREFIX}2/raw_frames/a/old-a.jpg"
    old_b = public / f"{cleanup.PREFIX}22/raw_frames/old-b.jpg"
    protected = public / f"{cleanup.PREFIX}23/raw_frames/keep.jpg"
    for path, payload in (
        (old_a, b"old-a"),
        (old_b, b"old-b"),
        (protected, b"keep"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return old_a, old_b, protected


def test_dry_run_only_inventories_allowlisted_raw_frames(tmp_path: Path, monkeypatch) -> None:
    old_a, old_b, protected = _fixture(tmp_path, monkeypatch)

    audit = cleanup.build_audit(execute=False)

    assert audit["file_count"] == 2
    assert audit["bytes"] == len(b"old-a") + len(b"old-b")
    assert audit["execute"] is False
    assert old_a.exists()
    assert old_b.exists()
    assert protected.read_bytes() == b"keep"
    assert "deleted_files" not in audit


def test_execute_deletes_allowlist_and_preserves_protected_batch(
    tmp_path: Path, monkeypatch
) -> None:
    old_a, old_b, protected = _fixture(tmp_path, monkeypatch)

    audit = cleanup.build_audit(execute=True)

    assert audit["deleted_files"] == 2
    assert audit["deleted_bytes"] == len(b"old-a") + len(b"old-b")
    assert audit["protected_file_counts_before"] == audit["protected_file_counts_after"]
    assert not old_a.exists()
    assert not old_b.exists()
    assert protected.read_bytes() == b"keep"
