from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from app.analysis.official_evaluation import RawOnlyEvaluationError, seal_raw_only_predictions
from app.analysis.training_annotation import (
    seal_training_annotation_manifest,
    verify_training_annotation_manifest,
)
from scripts import seal_training_annotation_manifest as manifest_cli


def _benchmark_bundle(path: Path) -> dict:
    return seal_raw_only_predictions(
        game_id="acceptance",
        raw_video_paths=[path],
        events=[],
        config={},
    ).model_dump(mode="json")


def _manifest_cli_argv(
    *,
    source_video: Path,
    annotation: Path,
    benchmark_bundle: Path,
    output: Path,
) -> list[str]:
    return [
        "seal_training_annotation_manifest.py",
        "--producer",
        "codex",
        "--source-video",
        str(source_video),
        "--annotation",
        str(annotation),
        "--task-type",
        "shot_validity",
        "--benchmark-bundle",
        str(benchmark_bundle),
        "--output",
        str(output),
    ]


def test_codex_training_manifest_is_hash_bound_and_benchmark_disjoint(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.mov"
    training = tmp_path / "training.mov"
    labels = tmp_path / "labels.json"
    benchmark.write_bytes(b"acceptance")
    training.write_bytes(b"different game")
    labels.write_text('{"boxes": []}', encoding="utf-8")

    manifest = seal_training_annotation_manifest(
        producer="codex",
        source_video_paths=[training],
        annotation_paths=[labels],
        task_types=["player_bbox", "ball_bbox"],
        benchmark_bundles=[_benchmark_bundle(benchmark)],
    )

    verified = verify_training_annotation_manifest(manifest)
    assert verified["runtime_consumable"] is False
    assert verified["benchmark_overlap"] is False


def test_training_manifest_rejects_acceptance_video_reuse(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.mov"
    labels = tmp_path / "labels.json"
    benchmark.write_bytes(b"acceptance")
    labels.write_text('{"boxes": []}', encoding="utf-8")

    with pytest.raises(RawOnlyEvaluationError, match="overlap acceptance"):
        seal_training_annotation_manifest(
            producer="codex",
            source_video_paths=[benchmark],
            annotation_paths=[labels],
            task_types=["player_bbox"],
            benchmark_bundles=[_benchmark_bundle(benchmark)],
        )


def test_training_manifest_rejects_mutation(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.mov"
    training = tmp_path / "training.mov"
    labels = tmp_path / "labels.json"
    benchmark.write_bytes(b"acceptance")
    training.write_bytes(b"different game")
    labels.write_text('{"boxes": []}', encoding="utf-8")
    manifest = seal_training_annotation_manifest(
        producer="codex",
        source_video_paths=[training],
        annotation_paths=[labels],
        task_types=["player_bbox"],
        benchmark_bundles=[_benchmark_bundle(benchmark)],
    )
    manifest["runtime_consumable"] = True

    with pytest.raises(RawOnlyEvaluationError, match="hash mismatch"):
        verify_training_annotation_manifest(manifest)


@pytest.mark.parametrize(
    "alias_kind",
    ("direct", "symlink", "case_only", "hardlink"),
)
def test_training_manifest_cli_rejects_output_input_alias_before_any_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_kind: str,
) -> None:
    source_video = tmp_path / "training.mov"
    annotation = tmp_path / "labels.json"
    benchmark_bundle = tmp_path / "benchmark.json"
    source_video.write_bytes(b"training")
    annotation.write_text("not read\n", encoding="utf-8")
    benchmark_bundle.write_text("not read\n", encoding="utf-8")
    output = tmp_path / "manifest.json"
    if alias_kind == "direct":
        output = source_video
    elif alias_kind == "symlink":
        output.symlink_to(annotation)
    elif alias_kind == "case_only":
        output = source_video.with_name(source_video.name.upper())
    else:
        os.link(benchmark_bundle, output)

    monkeypatch.setattr(
        sys,
        "argv",
        _manifest_cli_argv(
            source_video=source_video,
            annotation=annotation,
            benchmark_bundle=benchmark_bundle,
            output=output,
        ),
    )
    reads: list[Path] = []

    def reject_read(path: Path, *_args: Any, **_kwargs: Any) -> str:
        reads.append(path)
        raise AssertionError("no input may be read before alias validation")

    monkeypatch.setattr(Path, "read_text", reject_read)

    with pytest.raises(ValueError, match="output.*input|alias"):
        manifest_cli.main()

    assert reads == []


def test_training_manifest_cli_freezes_all_resolved_paths_before_sealing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source_target = inputs / "training.mov"
    annotation_target = inputs / "labels.json"
    benchmark_target = inputs / "benchmark.json"
    source_target.write_bytes(b"training")
    annotation_target.write_text('{"labels": []}', encoding="utf-8")
    benchmark_target.write_text('{"benchmark": true}', encoding="utf-8")
    source_link = tmp_path / "source-link.mov"
    annotation_link = tmp_path / "annotation-link.json"
    benchmark_link = tmp_path / "benchmark-link.json"
    source_link.symlink_to(source_target)
    annotation_link.symlink_to(annotation_target)
    benchmark_link.symlink_to(benchmark_target)
    output = tmp_path / "nested" / "manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        _manifest_cli_argv(
            source_video=source_link,
            annotation=annotation_link,
            benchmark_bundle=benchmark_link,
            output=output,
        ),
    )
    captured: dict[str, object] = {}

    def capture_seal(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"manifest_sha256": "a" * 64}

    written: list[tuple[Path, dict[str, object]]] = []
    monkeypatch.setattr(
        manifest_cli,
        "seal_training_annotation_manifest",
        capture_seal,
    )
    monkeypatch.setattr(
        manifest_cli,
        "_write_json_atomic",
        lambda path, payload: written.append((path, payload)),
        raising=False,
    )

    assert manifest_cli.main() == 0

    assert captured["source_video_paths"] == [source_target.resolve()]
    assert captured["annotation_paths"] == [annotation_target.resolve()]
    assert captured["benchmark_bundles"] == [{"benchmark": True}]
    assert written == [(output.resolve(), {"manifest_sha256": "a" * 64})]


def test_training_manifest_atomic_writer_preserves_exact_bytes_and_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "manifest.json"
    destination.write_text("old manifest\n", encoding="utf-8")
    payload = {"producer": "人工", "manifest_sha256": "a" * 64}
    expected = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
    original_replace = Path.replace
    events: list[str] = []

    def observe_fsync(file_descriptor: int) -> None:
        assert file_descriptor >= 0
        events.append("fsync")

    def observe_replace(source: Path, target: Path) -> Path:
        assert source.parent == destination.parent
        assert source.name.startswith(f".{destination.name}.")
        assert source.suffix == ".tmp"
        assert source.read_bytes() == expected
        assert destination.read_bytes() == b"old manifest\n"
        events.append("replace")
        return original_replace(source, target)

    monkeypatch.setattr(manifest_cli.os, "fsync", observe_fsync)
    monkeypatch.setattr(Path, "replace", observe_replace)

    manifest_cli._write_json_atomic(destination, payload)

    assert events == ["fsync", "replace"]
    assert destination.read_bytes() == expected
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("failure_stage", ("fsync", "replace"))
def test_training_manifest_atomic_writer_preserves_old_target_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    destination = tmp_path / "manifest.json"
    destination.write_text("old manifest\n", encoding="utf-8")

    if failure_stage == "fsync":
        monkeypatch.setattr(
            manifest_cli.os,
            "fsync",
            lambda _descriptor: (_ for _ in ()).throw(OSError("fsync failed")),
        )
    else:
        monkeypatch.setattr(
            Path,
            "replace",
            lambda _source, _target: (_ for _ in ()).throw(
                OSError("replace failed")
            ),
        )

    with pytest.raises(OSError, match=f"{failure_stage} failed"):
        manifest_cli._write_json_atomic(destination, {"new": True})

    assert destination.read_bytes() == b"old manifest\n"
    assert list(tmp_path.iterdir()) == [destination]


def test_training_manifest_cli_bytes_match_the_existing_contract(
    tmp_path: Path,
) -> None:
    benchmark_video = tmp_path / "benchmark.mov"
    source_video = tmp_path / "training.mov"
    annotation = tmp_path / "labels.json"
    benchmark_path = tmp_path / "benchmark.json"
    output = tmp_path / "manifest.json"
    benchmark_video.write_bytes(b"acceptance")
    source_video.write_bytes(b"training")
    annotation.write_text('{"boxes": []}', encoding="utf-8")
    benchmark = _benchmark_bundle(benchmark_video)
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
    expected = seal_training_annotation_manifest(
        producer="codex",
        source_video_paths=[source_video.resolve()],
        annotation_paths=[annotation.resolve()],
        task_types=["shot_validity"],
        benchmark_bundles=[benchmark],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/seal_training_annotation_manifest.py",
            *_manifest_cli_argv(
                source_video=source_video,
                annotation=annotation,
                benchmark_bundle=benchmark_path,
                output=output,
            )[1:],
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == (
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")


def test_seal_training_manifest_script_runs_without_pythonpath() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/seal_training_annotation_manifest.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--benchmark-bundle" in result.stdout
