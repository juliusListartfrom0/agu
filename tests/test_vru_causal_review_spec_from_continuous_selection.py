from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

import app.analysis.continuous_causal_selection as selection_module
from app.analysis.continuous_causal_selection import (
    SOURCE_SELECTION_PROTOCOL,
    build_continuous_causal_selection,
)
from scripts import build_vru_causal_review as legacy_builder
from scripts import build_vru_causal_review_spec_from_continuous_selection as cli


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


def _stub_frame_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCapture:
        def isOpened(self) -> bool:
            return True

        def release(self) -> None:
            pass

    monkeypatch.setattr(legacy_builder.cv2, "VideoCapture", lambda _path: FakeCapture())
    monkeypatch.setattr(
        legacy_builder,
        "_frame_sha256s",
        lambda _capture, _path, indexes: {str(index): "a" * 64 for index in indexes},
    )


def _write_manifest(tmp_path: Path) -> Path:
    video = tmp_path / "raw" / "harwood.webm"
    video.parent.mkdir(parents=True, exist_ok=True)
    source_video_bytes = b"immutable-source-video-bytes"
    video.write_bytes(source_video_bytes)
    path = tmp_path / "source_manifest.json"
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
                        "path": "raw/harwood.webm",
                        "size_bytes": len(source_video_bytes),
                        "sha256": hashlib.sha256(source_video_bytes).hexdigest(),
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


def _write_selection(tmp_path: Path, manifest: Path) -> tuple[Path, dict[str, Any]]:
    selection = build_continuous_causal_selection(
        source_manifest_path=manifest,
        source_id="harwood",
    )
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    return path, selection


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reseal(selection: dict[str, Any]) -> None:
    unsigned = {key: value for key, value in selection.items() if key != "artifact_sha256"}
    selection["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_verified_selection_expands_to_exact_24_by_64_half_open_spec(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    selection_path, selection = _write_selection(tmp_path, manifest)

    spec = cli.build_spec_from_continuous_selection(
        selection_path=selection_path,
        expected_selection_artifact_sha256=selection["artifact_sha256"],
        source_manifest_path=manifest,
    )

    assert spec["schema_version"] == "agu.vru-causal-review-spec.v1"
    assert spec["runtime_consumable"] is False
    assert spec["codex_runtime_answer_used"] is False
    assert spec["labels_hidden_from_reviewer"] is True
    assert spec["source_selection_artifact_sha256"] == selection["artifact_sha256"]
    assert spec["expected_selection_artifact_sha256"] == selection["artifact_sha256"]
    assert spec["source_manifest_sha256"] == _file_sha256(manifest)
    assert spec["radius_seconds"] == 4.0
    assert spec["sample_period_seconds"] == 0.125
    assert spec["sample_count"] == 64
    assert spec["interval_semantics"] == "half_open"
    assert len(spec["examples"]) == 24
    for window, example in zip(selection["windows"], spec["examples"], strict=True):
        expected_indexes = [
            round((window["start_seconds"] + index / 8.0) * window["source_fps"]) for index in range(64)
        ]
        assert example == {
            "review_id": window["review_id"],
            "frame_indexes": expected_indexes,
        }
        assert expected_indexes == sorted(set(expected_indexes))
        assert expected_indexes[32] == window["anchor_frame"]
        assert expected_indexes[-1] < round(window["end_seconds"] * window["source_fps"])


def test_atomic_adapter_builds_final_plan_bound_to_selection_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_manifest(tmp_path)
    selection_path, selection = _write_selection(tmp_path, manifest)
    _stub_frame_decoder(monkeypatch)

    plan = cli.build_review_plan_from_continuous_selection(
        selection_path=selection_path,
        expected_selection_artifact_sha256=selection["artifact_sha256"],
        source_manifest_path=manifest,
    )

    assert plan["source_selection_artifact_sha256"] == selection["artifact_sha256"]
    assert plan["source_manifest_sha256"] == _file_sha256(manifest)
    assert plan["artifact_sha256"]
    assert plan["reviewer_visible_fields"] == [
        "review_id",
        "source_video_handle",
        "source_fps",
        "frame_indexes",
        "frame_sha256s",
    ]
    assert len(plan["examples"]) == 24
    assert all(len(row["frame_indexes"]) == 64 for row in plan["examples"])
    assert all("source_video_handle" in row for row in plan["examples"])
    assert all("harwood" not in row["review_id"] for row in plan["examples"])


def test_legacy_builder_rejects_mutable_continuous_adapter_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_manifest(tmp_path)
    selection_path, selection = _write_selection(tmp_path, manifest)
    spec = cli.build_spec_from_continuous_selection(
        selection_path=selection_path,
        expected_selection_artifact_sha256=selection["artifact_sha256"],
        source_manifest_path=manifest,
    )
    spec["examples"][0]["frame_indexes"] = [0, 1, 2]
    spec_path = tmp_path / "mutable_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    _stub_frame_decoder(monkeypatch)

    with pytest.raises(ValueError, match="atomic.*continuous selection"):
        legacy_builder.build_plan(manifest_path=manifest, spec_path=spec_path)


def test_legacy_builder_rejects_pre_migration_continuous_spec_with_clip_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_manifest(tmp_path)
    selection_path, selection = _write_selection(tmp_path, manifest)
    spec = cli.build_spec_from_continuous_selection(
        selection_path=selection_path,
        expected_selection_artifact_sha256=selection["artifact_sha256"],
        source_manifest_path=manifest,
    )
    spec["purpose"] = "offline_continuous_game_manual_causal_review_closure"
    for row in spec["examples"]:
        row["clip_id"] = selection["source"]["clip_id"]
    spec_path = tmp_path / "old_mutable_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    _stub_frame_decoder(monkeypatch)

    with pytest.raises(ValueError, match="atomic.*continuous selection"):
        legacy_builder.build_plan(manifest_path=manifest, spec_path=spec_path)

    stripped = dict(spec)
    stripped["purpose"] = "legacy_mutable_manual_review"
    stripped.pop("source_selection_artifact_sha256")
    stripped.pop("expected_selection_artifact_sha256")
    stripped.pop("source_manifest_sha256")
    spec_path.write_text(json.dumps(stripped), encoding="utf-8")
    with pytest.raises(ValueError, match="atomic.*continuous selection"):
        legacy_builder.build_plan(manifest_path=manifest, spec_path=spec_path)


def test_atomic_builder_reverifies_the_final_plan_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_manifest(tmp_path)
    selection_path, selection = _write_selection(tmp_path, manifest)
    spec = cli.build_spec_from_continuous_selection(
        selection_path=selection_path,
        expected_selection_artifact_sha256=selection["artifact_sha256"],
        source_manifest_path=manifest,
    )
    monkeypatch.setattr(
        legacy_builder,
        "_build_plan_from_spec",
        lambda **_kwargs: {
            "source_selection_artifact_sha256": "f" * 64,
            "source_manifest_sha256": "e" * 64,
        },
    )

    with pytest.raises(ValueError, match="final.*plan.*receipt|plan.*invalid"):
        legacy_builder.build_plan_from_verified_continuous_spec(
            manifest_path=manifest,
            spec=spec,
            source_selection_artifact_sha256=selection["artifact_sha256"],
            expected_source_manifest_sha256=_file_sha256(manifest),
            source_clip_id=selection["source"]["clip_id"],
        )


def test_external_expected_selection_sha_is_required_and_fail_closed(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    selection_path, _selection = _write_selection(tmp_path, manifest)

    with pytest.raises(ValueError, match="expected artifact SHA-256"):
        cli.build_spec_from_continuous_selection(
            selection_path=selection_path,
            expected_selection_artifact_sha256="f" * 64,
            source_manifest_path=manifest,
        )


def test_adapter_passes_external_receipt_to_the_public_selection_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _write_manifest(tmp_path)
    selection_path, selection = _write_selection(tmp_path, manifest)
    observed: list[tuple[str, Path]] = []
    public_verifier = cli.verify_continuous_causal_selection_receipt

    def _observe_verifier(
        payload: dict[str, Any],
        *,
        expected_artifact_sha256: str,
        source_manifest_path: Path,
    ) -> dict[str, Any]:
        observed.append((expected_artifact_sha256, source_manifest_path))
        return public_verifier(
            payload,
            expected_artifact_sha256=expected_artifact_sha256,
            source_manifest_path=source_manifest_path,
        )

    monkeypatch.setattr(
        cli,
        "verify_continuous_causal_selection_receipt",
        _observe_verifier,
    )

    cli.build_spec_from_continuous_selection(
        selection_path=selection_path,
        expected_selection_artifact_sha256=selection["artifact_sha256"],
        source_manifest_path=manifest,
    )

    assert observed == [(selection["artifact_sha256"], manifest)]


def test_source_manifest_file_sha_must_match_the_sealed_selection(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    selection_path, selection = _write_selection(tmp_path, manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["purpose"] = "tampered"
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source manifest file SHA-256"):
        cli.build_spec_from_continuous_selection(
            selection_path=selection_path,
            expected_selection_artifact_sha256=selection["artifact_sha256"],
            source_manifest_path=manifest,
        )


def test_sealed_source_must_exactly_match_its_manifest_row(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    selection_path, selection = _write_selection(tmp_path, manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["videos"][0]["clip_id"] = "different_clip"
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    selection["source_manifest_sha256"] = _file_sha256(manifest)
    _reseal(selection)
    selection_path.write_text(json.dumps(selection) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest/video do not match"):
        cli.build_spec_from_continuous_selection(
            selection_path=selection_path,
            expected_selection_artifact_sha256=selection["artifact_sha256"],
            source_manifest_path=manifest,
        )


def test_source_video_sha_must_match_the_manifest_receipt(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    selection_path, selection = _write_selection(tmp_path, manifest)
    video = manifest.parent / "raw" / "harwood.webm"
    tampered = bytearray(video.read_bytes())
    tampered[0] ^= 1
    video.write_bytes(tampered)

    with pytest.raises(ValueError, match="video SHA-256"):
        cli.build_spec_from_continuous_selection(
            selection_path=selection_path,
            expected_selection_artifact_sha256=selection["artifact_sha256"],
            source_manifest_path=manifest,
        )


def _run_cli(
    monkeypatch: pytest.MonkeyPatch,
    *,
    selection: Path,
    expected_sha256: str,
    manifest: Path,
    output: Path,
) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_vru_causal_review_spec_from_continuous_selection.py",
            "--selection",
            str(selection),
            "--expected-selection-artifact-sha256",
            expected_sha256,
            "--source-manifest",
            str(manifest),
            "--output",
            str(output),
        ],
    )
    return cli.main()


@pytest.mark.parametrize("input_name", ("selection", "manifest", "video"))
@pytest.mark.parametrize("alias_kind", ("direct", "symlink", "hardlink", "casefold"))
def test_cli_rejects_every_output_input_alias_before_building(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_name: str,
    alias_kind: str,
) -> None:
    manifest = _write_manifest(tmp_path)
    selection, payload = _write_selection(tmp_path, manifest)
    video = tmp_path / "raw" / "harwood.webm"
    source_video_bytes = video.read_bytes()
    target = {
        "selection": selection,
        "manifest": manifest,
        "video": video,
    }[input_name]
    output = target
    if alias_kind == "symlink":
        output = tmp_path / f"{input_name}-output-symlink.json"
        output.symlink_to(target)
    elif alias_kind == "hardlink":
        output = tmp_path / f"{input_name}-output-hardlink.json"
        output.hardlink_to(target)
    elif alias_kind == "casefold":
        output = target.with_name(target.name.upper())
    built = False

    def _unexpected_build(**_: object) -> dict[str, Any]:
        nonlocal built
        built = True
        return {}

    monkeypatch.setattr(cli, "build_spec_from_continuous_selection", _unexpected_build)
    with pytest.raises(ValueError, match="alias"):
        _run_cli(
            monkeypatch,
            selection=selection,
            expected_sha256=payload["artifact_sha256"],
            manifest=manifest,
            output=output,
        )
    assert built is False
    assert video.read_bytes() == source_video_bytes


def test_cli_stages_complete_json_in_random_same_directory_file_and_fsyncs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _write_manifest(tmp_path)
    selection, payload = _write_selection(tmp_path, manifest)
    output = tmp_path / "nested" / "review_spec.json"
    replacements: list[tuple[Path, Path]] = []
    fsync_calls: list[int] = []
    original_replace = Path.replace
    _stub_frame_decoder(monkeypatch)

    def _observe_replace(source: Path, target: Path) -> Path:
        staged = json.loads(source.read_text(encoding="utf-8"))
        assert staged["schema_version"] == "agu.vru-causal-review-plan.v1"
        assert staged["source_selection_artifact_sha256"] == payload["artifact_sha256"]
        assert len(staged["examples"]) == 24
        replacements.append((source, target))
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", _observe_replace)
    monkeypatch.setattr(cli.os, "fsync", fsync_calls.append)

    assert (
        _run_cli(
            monkeypatch,
            selection=selection,
            expected_sha256=payload["artifact_sha256"],
            manifest=manifest,
            output=output,
        )
        == 0
    )
    assert len(fsync_calls) == 1
    assert len(replacements) == 1
    temporary, target = replacements[0]
    assert temporary.parent == output.parent
    assert temporary.name.startswith(f".{output.name}.")
    assert temporary.suffix == ".tmp"
    assert target == output
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert not temporary.exists()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["artifact_sha256"] == json.loads(output.read_text(encoding="utf-8"))["artifact_sha256"]
    assert receipt["source_manifest_sha256"] == _file_sha256(manifest)


def test_atomic_replace_failure_preserves_output_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "review_spec.json"
    output.write_text("old\n", encoding="utf-8")

    def _fail_replace(_source: Path, target: Path) -> Path:
        assert target == output
        raise OSError("injected replace failure")

    monkeypatch.setattr(Path, "replace", _fail_replace)
    with pytest.raises(OSError, match="injected"):
        cli._atomic_write_json(output, {"new": True})
    assert output.read_text(encoding="utf-8") == "old\n"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))
