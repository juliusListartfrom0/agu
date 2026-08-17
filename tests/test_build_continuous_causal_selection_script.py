from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

import app.analysis.continuous_causal_selection as selection_module
import scripts.build_continuous_causal_selection as cli
from app.analysis.continuous_causal_selection import (
    SOURCE_SELECTION_PROTOCOL,
    verify_continuous_causal_selection,
)


@pytest.fixture(autouse=True)
def _matching_media_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        selection_module,
        "_probe_video_geometry",
        lambda _path: {
            "fps": 30.0,
            "frame_count": 123_753,
            "duration_seconds": 4_125.088,
        },
    )


def _manifest(tmp_path: Path) -> Path:
    video = tmp_path / "harwood.webm"
    video_bytes = b"protected source video bytes\n"
    video.write_bytes(video_bytes)
    path = tmp_path / "source.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "agu.vru-basketball-source-manifest.v1",
                "purpose": "offline_harwood_causal_review",
                "license": "CC-BY-4.0-per-file",
                "source_urls": ["https://commons.wikimedia.org/wiki/File:Harwood.webm"],
                "selection": SOURCE_SELECTION_PROTOCOL,
                "runtime_consumable": False,
                "codex_runtime_answer_used": False,
                "provenance": {
                    "provider": "Wikimedia Commons",
                    "publisher": "Hardwick Community Television (HCTV)",
                    "publisher_channel_id": "hctv",
                    "source_page_url": "https://commons.wikimedia.org/wiki/File:Harwood.webm",
                    "original_source_url": "https://www.youtube.com/watch?v=fixture",
                    "original_source_id": "fixture",
                    "title": "Boys Varsity Basketball v. Harwood",
                    "event_date": "2026-01-22",
                    "license_spdx": "CC-BY-4.0",
                    "attribution": "Hardwick Community Television (HCTV)",
                    "license_review_warning": True,
                    "production_family": "HCTV",
                },
                "videos": [
                    {
                        "location": "harwood",
                        "clip_id": "harwood_fullgame_2026",
                        "path": video.name,
                        "size_bytes": len(video_bytes),
                        "sha256": hashlib.sha256(video_bytes).hexdigest(),
                        "fps": 30.0,
                        "frame_count": 123_753,
                        "duration_seconds": 4_125.088,
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _run(monkeypatch: pytest.MonkeyPatch, manifest: Path, output: Path) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_continuous_causal_selection.py",
            "--source-manifest",
            str(manifest),
            "--source-id",
            "harwood",
            "--output",
            str(output),
        ],
    )
    return cli.main()


def test_cli_writes_verified_artifact_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "nested" / "selection.json"

    assert _run(monkeypatch, _manifest(tmp_path), output) == 0

    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert verify_continuous_causal_selection(artifact) == artifact
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


@pytest.mark.parametrize("alias_kind", ("direct", "symlink", "hardlink"))
def test_cli_rejects_output_alias_before_building(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_kind: str,
) -> None:
    manifest = _manifest(tmp_path)
    output = manifest
    if alias_kind == "symlink":
        output = tmp_path / "selection.json"
        output.symlink_to(manifest)
    elif alias_kind == "hardlink":
        output = tmp_path / "selection.json"
        output.hardlink_to(manifest)
    built = False

    def _unexpected_build(**_: object) -> dict[str, object]:
        nonlocal built
        built = True
        return {}

    monkeypatch.setattr(cli, "build_continuous_causal_selection", _unexpected_build)
    with pytest.raises(ValueError, match="alias"):
        _run(monkeypatch, manifest, output)
    assert built is False


def test_cli_rejects_casefold_output_collision_before_building(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(tmp_path)
    output = manifest.with_name(manifest.name.upper())
    if output.exists():
        pytest.skip("case-insensitive filesystem already resolves the alias")
    built = False

    def _unexpected_build(**_: object) -> dict[str, object]:
        nonlocal built
        built = True
        return {}

    monkeypatch.setattr(cli, "build_continuous_causal_selection", _unexpected_build)
    with pytest.raises(ValueError, match="alias"):
        _run(monkeypatch, manifest, output)
    assert built is False


@pytest.mark.parametrize("alias_kind", ("direct", "symlink", "hardlink", "casefold"))
def test_cli_rejects_source_video_alias_and_preserves_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_kind: str,
) -> None:
    manifest = _manifest(tmp_path)
    video = tmp_path / "harwood.webm"
    original = video.read_bytes()
    output = video
    if alias_kind == "symlink":
        output = tmp_path / "selection.json"
        output.symlink_to(video)
    elif alias_kind == "hardlink":
        output = tmp_path / "selection.json"
        output.hardlink_to(video)
    elif alias_kind == "casefold":
        output = video.with_name(video.name.upper())
        if output.exists():
            pytest.skip("case-insensitive filesystem already resolves the alias")
    built = False

    def _unexpected_build(**_: object) -> dict[str, object]:
        nonlocal built
        built = True
        return {}

    monkeypatch.setattr(cli, "build_continuous_causal_selection", _unexpected_build)
    with pytest.raises(ValueError, match="alias"):
        _run(monkeypatch, manifest, output)
    assert built is False
    assert video.read_bytes() == original


def test_atomic_replace_failure_preserves_existing_output_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "selection.json"
    output.write_text("old\n", encoding="utf-8")
    original_replace = Path.replace

    def _fail_replace(self: Path, target: Path) -> Path:
        if target == output:
            raise OSError("injected replace failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", _fail_replace)
    with pytest.raises(OSError, match="injected"):
        cli._atomic_write_json(output, {"new": True})
    assert output.read_text(encoding="utf-8") == "old\n"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))
