from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from app.analysis.official_evaluation import seal_raw_only_predictions
from app.analysis.schemas import GameEventResponse
from app.analysis.shot_validity import SHOT_VALIDITY_LABEL_SCHEMA
from app.analysis.shot_validity_sampling_window import (
    SHOT_SAMPLING_PROTOCOL,
    resolve_shot_sampling_window,
)
from app.analysis.shot_validity_video_backbone import seal_video_embedding_artifact
from app.analysis.training_annotation import seal_training_annotation_manifest
from scripts import extract_shot_validity_video_embeddings as embedding_cli

GIB = 1024**3
BACKBONE = "torchvision/mvit_v2_s/kinetics400_v1"
EMBEDDING_DIMENSION = 768


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(payload: dict[str, object]) -> str:
    return _sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _write_resealed_training_manifest(
    inputs: dict[str, Any],
    payload: dict[str, object],
) -> None:
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = _canonical_sha256(payload)
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    inputs["manifest"].write_bytes(encoded)
    inputs["manifest_sha256"] = payload["manifest_sha256"]
    inputs["manifest_file_sha256"] = _sha256(encoded)


def _event(event_id: str, *, start_frame: int = 10) -> GameEventResponse:
    return GameEventResponse(
        event_id=event_id,
        revision=1,
        event_type="field_goal_attempt",
        source_video_id="video_001",
        start_frame=start_frame,
        end_frame=start_frame + 20,
        confidence=0.5,
    )


def _valid_extraction_inputs(
    tmp_path: Path,
    *,
    event_count: int = 1,
    source_count: int = 3,
) -> dict[str, Any]:
    if source_count not in (3, 4):
        raise ValueError("fixture supports only v1/v2 source counts")
    sources: list[Path] = []
    bundle_paths: list[Path] = []
    annotation_paths: list[Path] = []
    bundles: list[Any] = []
    labels_by_source: list[dict[str, object]] = []
    events_by_source: list[list[GameEventResponse]] = []
    for source_index in range(source_count):
        source = tmp_path / f"source-{source_index}.webm"
        source.write_bytes(f"stable-video-snapshot-{source_index}".encode())
        events = (
            [_event(f"event-{index}", start_frame=10 + index * 30) for index in range(event_count)]
            if source_index == 0
            else []
        )
        bundle = seal_raw_only_predictions(
            game_id=f"game-{source_index}",
            raw_video_paths=[source],
            events=events,
            config={"fixture": True},
        )
        bundle_path = tmp_path / f"candidate-bundle-{source_index}.json"
        bundle_path.write_bytes(bundle.model_dump_json(indent=2).encode("utf-8"))
        labels = {
            "schema_version": SHOT_VALIDITY_LABEL_SCHEMA,
            "purpose": "causal_shot_validity_training_only",
            "runtime_consumable": False,
            "formal_evaluation_eligible": False,
            "source_video_sha256": bundle.raw_videos[0].sha256,
            "candidate_bundle_sha256": bundle.bundle_sha256,
            "examples": [
                {"event_id": event.event_id, "event_present": index % 2 == 0} for index, event in enumerate(events)
            ],
        }
        annotation = tmp_path / f"labels-{source_index}.json"
        annotation.write_bytes(json.dumps(labels, sort_keys=True).encode("utf-8"))
        sources.append(source)
        bundle_paths.append(bundle_path)
        annotation_paths.append(annotation)
        bundles.append(bundle)
        labels_by_source.append(labels)
        events_by_source.append(events)
    training_export = tmp_path / ("training_export_v2.json" if source_count == 4 else "training_export.json")
    training_export.write_text(
        json.dumps(
            {
                "schema_version": (
                    "agu.vru-causal-shot-validity-training-export.v2"
                    if source_count == 4
                    else "agu.vru-causal-shot-validity-training-export.v1"
                )
            }
        ),
        encoding="utf-8",
    )
    benchmark_video = tmp_path / "benchmark.webm"
    benchmark_video.write_bytes(b"independent-benchmark")
    benchmark = seal_raw_only_predictions(
        game_id="benchmark",
        raw_video_paths=[benchmark_video],
        events=[],
        config={"fixture": True},
    )
    manifest = seal_training_annotation_manifest(
        producer="codex-assisted-offline-causal-review",
        source_video_paths=sources,
        annotation_paths=[training_export, *annotation_paths],
        task_types=["shot_validity"],
        benchmark_bundles=[benchmark.model_dump(mode="json")],
    )
    manifest_path = tmp_path / "training-manifest.json"
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    backbone_checkpoint = tmp_path / "mvit.pth"
    backbone_bytes = b"immutable-backbone-snapshot"
    backbone_checkpoint.write_bytes(backbone_bytes)
    output_parent = tmp_path / "outputs"
    output_parent.mkdir()
    return {
        "manifest": manifest_path,
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": _sha256(manifest_bytes),
        "training_export": training_export,
        "candidate_bundles": bundle_paths,
        "annotations": annotation_paths,
        "videos": sources,
        "candidate_bundle": bundle_paths[0],
        "annotation": annotation_paths[0],
        "video": sources[0],
        "backbone_checkpoint": backbone_checkpoint,
        "backbone_sha256": _sha256(backbone_bytes),
        "bundles": bundles,
        "labels_by_source": labels_by_source,
        "events_by_source": events_by_source,
        "bundle": bundles[0],
        "labels": labels_by_source[0],
        "events": events_by_source[0],
        "staging": output_parent / "embeddings.resume.json",
        "output": output_parent / "embeddings.json",
    }


def _valid_extract_kwargs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest_path": inputs["manifest"],
        "candidate_bundle_paths": inputs["candidate_bundles"],
        "annotation_paths": inputs["annotations"],
        "video_paths": inputs["videos"],
        "backbone_name": BACKBONE,
        "backbone_checkpoint": inputs["backbone_checkpoint"],
        "expected_backbone_sha256": inputs["backbone_sha256"],
        "expected_training_manifest_sha256": inputs["manifest_sha256"],
        "expected_training_manifest_file_sha256": inputs["manifest_file_sha256"],
        "clip_frames": 16,
        "batch_size": 1,
        "device_name": "cpu",
        "resume_checkpoint_path": inputs["staging"],
        "output_path": inputs["output"],
        "resume": False,
    }


def _allow_resource_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        embedding_cli.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=50.0, available=4 * GIB),
    )
    monkeypatch.setattr(
        embedding_cli.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=4 * GIB),
    )


def _reject_heavy_or_write_work(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("receipt failure reached heavy work or an output write")

    monkeypatch.setattr(embedding_cli, "_resolve_device", reject)
    monkeypatch.setattr(embedding_cli, "load_video_backbone", reject)
    monkeypatch.setattr(embedding_cli, "extract_video_embeddings", reject)
    monkeypatch.setattr(embedding_cli.cv2, "VideoCapture", reject)
    monkeypatch.setattr(embedding_cli, "file_sha256", reject, raising=False)
    monkeypatch.setattr(embedding_cli.tempfile, "NamedTemporaryFile", reject)
    monkeypatch.setattr(embedding_cli, "_write_json_atomic", reject)


@pytest.mark.parametrize(
    "corruption",
    (
        "manifest_internal_receipt",
        "manifest_file_receipt",
        "backbone_receipt",
        "bundle_schema",
        "annotation_receipt",
        "video_receipt",
        "resume_schema",
    ),
)
def test_every_receipt_and_schema_failure_precedes_opencv_device_model_and_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    inputs = _valid_extraction_inputs(tmp_path)
    kwargs = _valid_extract_kwargs(inputs)
    if corruption == "manifest_internal_receipt":
        kwargs["expected_training_manifest_sha256"] = "b" * 64
    elif corruption == "manifest_file_receipt":
        kwargs["expected_training_manifest_file_sha256"] = "b" * 64
    elif corruption == "backbone_receipt":
        kwargs["expected_backbone_sha256"] = "b" * 64
    elif corruption == "bundle_schema":
        inputs["candidate_bundle"].write_bytes(b"{}")
    elif corruption == "annotation_receipt":
        inputs["annotation"].write_bytes(b'{"tampered":true}')
    elif corruption == "video_receipt":
        inputs["video"].write_bytes(b"tampered-video")
    elif corruption == "resume_schema":
        resume_bytes = b"{}"
        inputs["staging"].write_bytes(resume_bytes)
        kwargs["resume"] = True
        kwargs["expected_resume_checkpoint_sha256"] = _sha256(resume_bytes)
    else:  # pragma: no cover - exhaustive parameter list
        raise AssertionError(corruption)
    _allow_resource_preflight(monkeypatch)
    _reject_heavy_or_write_work(monkeypatch)

    with pytest.raises(ValueError):
        embedding_cli.extract_labeled_video_embeddings(**kwargs)

    assert not inputs["output"].exists()
    if corruption != "resume_schema":
        assert not inputs["staging"].exists()


@pytest.mark.parametrize(
    "missing_receipt",
    (
        "expected_training_manifest_sha256",
        "expected_training_manifest_file_sha256",
        "expected_resume_checkpoint_sha256",
    ),
)
def test_paired_external_receipts_fail_before_any_file_read_or_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_receipt: str,
) -> None:
    inputs = _valid_extraction_inputs(tmp_path)
    kwargs = _valid_extract_kwargs(inputs)
    if missing_receipt == "expected_training_manifest_sha256":
        kwargs[missing_receipt] = None
    elif missing_receipt == "expected_training_manifest_file_sha256":
        kwargs[missing_receipt] = None
    else:
        inputs["staging"].write_bytes(b"resume")
        kwargs["resume"] = True
        kwargs.pop(missing_receipt, None)
    _allow_resource_preflight(monkeypatch)
    _reject_preflight_side_effects(monkeypatch)

    with pytest.raises(ValueError, match="receipt|SHA|sha256"):
        embedding_cli.extract_labeled_video_embeddings(**kwargs)

    assert not inputs["output"].exists()


@pytest.mark.parametrize(
    "required_receipt",
    (
        "expected_training_manifest_sha256",
        "expected_training_manifest_file_sha256",
    ),
)
def test_public_api_requires_both_training_manifest_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    required_receipt: str,
) -> None:
    inputs = _valid_extraction_inputs(tmp_path)
    kwargs = _valid_extract_kwargs(inputs)
    kwargs.pop(required_receipt)
    _reject_preflight_side_effects(monkeypatch)

    with pytest.raises(TypeError, match=required_receipt):
        embedding_cli.extract_labeled_video_embeddings(**kwargs)


@pytest.mark.parametrize(
    ("receipt_name", "invalid_receipt"),
    (
        ("expected_backbone_sha256", True),
        ("expected_backbone_sha256", "A" * 64),
        ("expected_training_manifest_sha256", "short"),
        ("expected_training_manifest_file_sha256", "F" * 64),
    ),
)
def test_api_receipts_require_exact_lowercase_sha256_before_any_file_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_name: str,
    invalid_receipt: object,
) -> None:
    inputs = _valid_extraction_inputs(tmp_path)
    kwargs = _valid_extract_kwargs(inputs)
    kwargs[receipt_name] = invalid_receipt
    _allow_resource_preflight(monkeypatch)
    _reject_preflight_side_effects(monkeypatch)

    with pytest.raises(ValueError, match="SHA-256|sha256"):
        embedding_cli.extract_labeled_video_embeddings(**kwargs)

    assert not inputs["staging"].exists()
    assert not inputs["output"].exists()


def test_extractor_rejects_manifest_source_subset_before_backbone_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _valid_extraction_inputs(tmp_path)
    kwargs = _valid_extract_kwargs(inputs)
    kwargs["candidate_bundle_paths"] = inputs["candidate_bundles"][:-1]
    kwargs["annotation_paths"] = inputs["annotations"][:-1]
    kwargs["video_paths"] = inputs["videos"][:-1]
    _allow_resource_preflight(monkeypatch)
    original_open = Path.open

    def reject_backbone_read(path: Path, *args: object, **open_kwargs: object) -> object:
        if path.resolve() == inputs["backbone_checkpoint"].resolve():
            raise AssertionError("manifest coverage failure reached backbone read")
        return original_open(path, *args, **open_kwargs)

    monkeypatch.setattr(Path, "open", reject_backbone_read)
    _reject_heavy_or_write_work(monkeypatch)

    with pytest.raises(ValueError, match="manifest|source|pair|cover|count"):
        embedding_cli.extract_labeled_video_embeddings(**kwargs)


@pytest.mark.parametrize(
    "manifest_drift",
    (
        "extra_source",
        "duplicate_source",
        "missing_label",
        "extra_annotation",
        "wrong_index",
    ),
)
def test_manifest_must_exactly_equal_passed_sources_labels_plus_one_export_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest_drift: str,
) -> None:
    inputs = _valid_extraction_inputs(tmp_path)
    manifest = json.loads(inputs["manifest"].read_text(encoding="utf-8"))
    if manifest_drift == "extra_source":
        manifest["source_videos"].append({"filename": "extra.webm", "sha256": "e" * 64, "size_bytes": 1})
    elif manifest_drift == "duplicate_source":
        manifest["source_videos"].append(dict(manifest["source_videos"][0]))
    elif manifest_drift == "missing_label":
        manifest["annotation_files"].pop()
    elif manifest_drift == "extra_annotation":
        manifest["annotation_files"].append({"filename": "extra-label.json", "sha256": "e" * 64, "size_bytes": 1})
    elif manifest_drift == "wrong_index":
        manifest["annotation_files"][0]["filename"] = "unrelated-index.json"
    else:  # pragma: no cover - exhaustive parameter list
        raise AssertionError(manifest_drift)
    _write_resealed_training_manifest(inputs, manifest)
    kwargs = _valid_extract_kwargs(inputs)
    _allow_resource_preflight(monkeypatch)
    _reject_heavy_or_write_work(monkeypatch)

    with pytest.raises(ValueError, match="manifest|source|annotation|label|index|unique|cover"):
        embedding_cli.extract_labeled_video_embeddings(**kwargs)

    assert not inputs["staging"].exists()
    assert not inputs["output"].exists()


def test_four_source_v2_manifest_is_fully_consumed_and_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _valid_extraction_inputs(tmp_path, event_count=2, source_count=4)
    kwargs = _valid_extract_kwargs(inputs)
    _install_lightweight_extraction_fakes(monkeypatch, inputs)

    artifact = embedding_cli.extract_labeled_video_embeddings(**kwargs)

    assert len(artifact["source_video_sha256"]) == 4
    assert len(artifact["training_annotation_sha256"]) == 4
    assert [row["event_id"] for row in artifact["examples"]] == [
        "event-0",
        "event-1",
    ]
    assert not inputs["staging"].exists()
    assert json.loads(inputs["output"].read_text(encoding="utf-8")) == artifact


@pytest.mark.parametrize(
    "alias_kind",
    (
        "output_input",
        "staging_input",
        "output_staging_direct",
        "output_staging_hardlink",
        "output_staging_symlink",
        "output_staging_casefold",
        "output_staging_ancestor",
        "output_staging_descendant",
        "output_ancestor",
        "staging_descendant",
    ),
)
def test_output_and_staging_reject_all_input_and_cross_output_aliases_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_kind: str,
) -> None:
    inputs = _valid_extraction_inputs(tmp_path)
    kwargs = _valid_extract_kwargs(inputs)
    if alias_kind == "output_input":
        kwargs["output_path"] = inputs["video"]
    elif alias_kind == "staging_input":
        kwargs["resume_checkpoint_path"] = inputs["annotation"]
    elif alias_kind == "output_staging_direct":
        kwargs["output_path"] = inputs["staging"]
    elif alias_kind == "output_staging_hardlink":
        inputs["staging"].write_bytes(b"existing-stage")
        os.link(inputs["staging"], inputs["output"])
    elif alias_kind == "output_staging_symlink":
        inputs["output"].symlink_to(inputs["staging"])
    elif alias_kind == "output_staging_casefold":
        kwargs["output_path"] = inputs["staging"].with_name(inputs["staging"].name.swapcase())
    elif alias_kind == "output_staging_ancestor":
        kwargs["resume_checkpoint_path"] = inputs["output"].parent
    elif alias_kind == "output_staging_descendant":
        kwargs["output_path"] = inputs["staging"] / "final.json"
    elif alias_kind == "output_ancestor":
        kwargs["output_path"] = tmp_path
    elif alias_kind == "staging_descendant":
        kwargs["resume_checkpoint_path"] = inputs["video"] / "stage.json"
    else:  # pragma: no cover - exhaustive parameter list
        raise AssertionError(alias_kind)
    _reject_preflight_side_effects(monkeypatch)

    with pytest.raises(ValueError, match="alias|ancestor|descendant|overlap"):
        embedding_cli.extract_labeled_video_embeddings(**kwargs)


def test_resource_preflight_rejects_existing_non_directory_parent(
    tmp_path: Path,
) -> None:
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_bytes(b"file")

    with pytest.raises(RuntimeError, match="directory"):
        embedding_cli._nearest_existing_parent(parent_file / "nested")


def test_existing_output_directory_is_rejected_before_reads_or_resource_probes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _valid_extraction_inputs(tmp_path)
    inputs["output"].mkdir()
    kwargs = _valid_extract_kwargs(inputs)
    _reject_preflight_side_effects(monkeypatch)

    def reject_resource_probe(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("invalid output type must fail before resource probes")

    monkeypatch.setattr(embedding_cli.psutil, "virtual_memory", reject_resource_probe)
    monkeypatch.setattr(embedding_cli.shutil, "disk_usage", reject_resource_probe)

    with pytest.raises(ValueError, match="output.*regular file"):
        embedding_cli.extract_labeled_video_embeddings(**kwargs)


def _install_lightweight_extraction_fakes(
    monkeypatch: pytest.MonkeyPatch,
    inputs: dict[str, Any],
    *,
    mutate_video: bool = False,
) -> tuple[list[Path], list[str]]:
    loaded_snapshots: list[Path] = []
    extracted_event_ids: list[str] = []
    _allow_resource_preflight(monkeypatch)
    monkeypatch.setattr(embedding_cli, "_read_video_fps", lambda _path: 30.0)
    monkeypatch.setattr(embedding_cli, "_resolve_device", lambda _name: torch.device("cpu"))

    def fake_load(
        _name: str,
        checkpoint_path: Path,
        *,
        device: torch.device,
    ) -> tuple[object, object]:
        assert device == torch.device("cpu")
        assert checkpoint_path != inputs["backbone_checkpoint"]
        assert checkpoint_path.read_bytes() == b"immutable-backbone-snapshot"
        loaded_snapshots.append(checkpoint_path)
        inputs["backbone_checkpoint"].write_bytes(b"mutated-after-snapshot")
        return object(), object()

    def fake_extract(**kwargs: Any) -> list[list[float]]:
        assert kwargs["batch_size"] == 1
        assert len(kwargs["events"]) == 1
        event = kwargs["events"][0]
        extracted_event_ids.append(event.event_id)
        if mutate_video:
            inputs["video"].write_bytes(b"mutated-during-extraction")
        return [[float(len(extracted_event_ids))] * EMBEDDING_DIMENSION]

    monkeypatch.setattr(embedding_cli, "load_video_backbone", fake_load)
    monkeypatch.setattr(embedding_cli, "extract_video_embeddings", fake_extract)
    return loaded_snapshots, extracted_event_ids


def test_success_uses_single_json_and_backbone_snapshots_then_stages_each_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _valid_extraction_inputs(tmp_path, event_count=2)
    kwargs = _valid_extract_kwargs(inputs)
    loaded_snapshots, extracted_event_ids = _install_lightweight_extraction_fakes(
        monkeypatch,
        inputs,
    )
    original_open = Path.open
    snapshot_open_counts: dict[Path, int] = {}
    snapshot_inputs = {
        inputs["manifest"].resolve(),
        inputs["backbone_checkpoint"].resolve(),
        *(path.resolve() for path in inputs["candidate_bundles"]),
        *(path.resolve() for path in inputs["annotations"]),
    }

    def observe_open(path: Path, *args: object, **kwargs: object) -> object:
        resolved = path.resolve()
        mode = str(args[0] if args else kwargs.get("mode", "r"))
        if resolved in snapshot_inputs and "r" in mode:
            snapshot_open_counts[resolved] = snapshot_open_counts.get(resolved, 0) + 1
        return original_open(path, *args, **kwargs)

    writes: list[Path] = []
    original_write = embedding_cli._write_json_atomic

    def observe_write(path: Path, payload: dict[str, object]) -> None:
        writes.append(path)
        original_write(path, payload)

    monkeypatch.setattr(Path, "open", observe_open)
    monkeypatch.setattr(embedding_cli, "_write_json_atomic", observe_write)

    artifact = embedding_cli.extract_labeled_video_embeddings(**kwargs)

    for path in snapshot_inputs:
        assert snapshot_open_counts[path] == 1
    assert loaded_snapshots and all(not path.exists() for path in loaded_snapshots)
    assert extracted_event_ids == ["event-0", "event-1"]
    assert writes == [inputs["staging"], inputs["staging"], inputs["output"]]
    assert not inputs["staging"].exists()
    assert json.loads(inputs["output"].read_text(encoding="utf-8")) == artifact


def test_video_drift_after_extraction_removes_staging_and_preserves_old_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _valid_extraction_inputs(tmp_path)
    kwargs = _valid_extract_kwargs(inputs)
    inputs["output"].write_bytes(b"old-final\n")
    _install_lightweight_extraction_fakes(monkeypatch, inputs, mutate_video=True)

    with pytest.raises(ValueError, match="video.*changed|changed.*video"):
        embedding_cli.extract_labeled_video_embeddings(**kwargs)

    assert not inputs["staging"].exists()
    assert inputs["output"].read_bytes() == b"old-final\n"


def test_video_drift_immediately_before_final_rehash_is_not_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _valid_extraction_inputs(tmp_path)
    kwargs = _valid_extract_kwargs(inputs)
    inputs["output"].write_bytes(b"old-final\n")
    _install_lightweight_extraction_fakes(monkeypatch, inputs)
    original_stream_hash = embedding_cli._stream_file_sha256
    stream_calls = 0

    def drift_before_final_hash(path: Path) -> tuple[str, int]:
        nonlocal stream_calls
        stream_calls += 1
        if stream_calls == len(inputs["videos"]) + 1:
            path.write_bytes(b"mutated-before-final-rehash")
        return original_stream_hash(path)

    monkeypatch.setattr(
        embedding_cli,
        "_stream_file_sha256",
        drift_before_final_hash,
    )

    with pytest.raises(ValueError, match="video.*changed|changed.*video"):
        embedding_cli.extract_labeled_video_embeddings(**kwargs)

    assert stream_calls == len(inputs["videos"]) + 1
    assert not inputs["staging"].exists()
    assert inputs["output"].read_bytes() == b"old-final\n"


@pytest.mark.parametrize("existing_final", (False, True))
def test_incomplete_extraction_keeps_atomic_row_staging_without_publishing_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_final: bool,
) -> None:
    inputs = _valid_extraction_inputs(tmp_path, event_count=2)
    kwargs = _valid_extract_kwargs(inputs)
    if existing_final:
        inputs["output"].write_bytes(b"old-final\n")
    _loaded_snapshots, extracted_event_ids = _install_lightweight_extraction_fakes(
        monkeypatch,
        inputs,
    )

    def fail_second_row(**extract_kwargs: Any) -> list[list[float]]:
        event = extract_kwargs["events"][0]
        extracted_event_ids.append(event.event_id)
        if len(extracted_event_ids) == 2:
            raise RuntimeError("synthetic interrupted extraction")
        return [[1.0] * EMBEDDING_DIMENSION]

    monkeypatch.setattr(embedding_cli, "extract_video_embeddings", fail_second_row)

    with pytest.raises(RuntimeError, match="interrupted"):
        embedding_cli.extract_labeled_video_embeddings(**kwargs)

    staged = json.loads(inputs["staging"].read_text(encoding="utf-8"))
    assert [row["event_id"] for row in staged["examples"]] == ["event-0"]
    if existing_final:
        assert inputs["output"].read_bytes() == b"old-final\n"
    else:
        assert not inputs["output"].exists()


def test_failed_final_publication_preserves_complete_staging_and_old_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _valid_extraction_inputs(tmp_path, event_count=2)
    kwargs = _valid_extract_kwargs(inputs)
    inputs["output"].write_bytes(b"old-final\n")
    _install_lightweight_extraction_fakes(monkeypatch, inputs)
    original_write = embedding_cli._write_json_atomic

    def fail_final_write(path: Path, payload: dict[str, object]) -> None:
        if path == inputs["output"]:
            raise OSError("synthetic final publication failure")
        original_write(path, payload)

    monkeypatch.setattr(embedding_cli, "_write_json_atomic", fail_final_write)

    with pytest.raises(OSError, match="publication failure"):
        embedding_cli.extract_labeled_video_embeddings(**kwargs)

    staged = json.loads(inputs["staging"].read_text(encoding="utf-8"))
    assert [row["event_id"] for row in staged["examples"]] == [
        "event-0",
        "event-1",
    ]
    assert inputs["output"].read_bytes() == b"old-final\n"


def _write_partial_resume_checkpoint(inputs: dict[str, Any]) -> str:
    event = inputs["events"][0]
    window = resolve_shot_sampling_window(event, fps=30.0)
    spec = embedding_cli.get_video_backbone_spec(BACKBONE)
    artifact = seal_video_embedding_artifact(
        {
            "purpose": "backbone_screening_training_only",
            "producer": "agu",
            "training_manifest_sha256": inputs["manifest_sha256"],
            "training_annotation_sha256": sorted(_sha256(path.read_bytes()) for path in inputs["annotations"]),
            "source_video_sha256": sorted(_sha256(path.read_bytes()) for path in inputs["videos"]),
            "backbone": BACKBONE,
            "backbone_sha256": inputs["backbone_sha256"],
            "backbone_license": spec.license,
            "backbone_weights_url": spec.weights_url,
            "embedding_dimension": spec.embedding_dimension,
            "clip_frames": 16,
            "sampling_protocol": SHOT_SAMPLING_PROTOCOL,
            "examples": [
                {
                    "source_video_sha256": _sha256(inputs["video"].read_bytes()),
                    "candidate_bundle_sha256": inputs["bundle"].bundle_sha256,
                    "event_id": event.event_id,
                    "event_present": True,
                    "sampling_window": asdict(window),
                    "embedding": [1.0] * EMBEDDING_DIMENSION,
                }
            ],
        }
    )
    encoded = (json.dumps(artifact, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    inputs["staging"].write_bytes(encoded)
    return _sha256(encoded)


def _rewrite_resume_checkpoint(
    inputs: dict[str, Any],
    corruption: str,
) -> str:
    artifact = json.loads(inputs["staging"].read_text(encoding="utf-8"))
    row = artifact["examples"][0]
    if corruption == "extra_top_level_field":
        artifact["unexpected_top_level"] = "self-sealed extra"
    elif corruption == "extra_example_field":
        row["unexpected"] = "self-sealed extra"
    elif corruption == "missing_example_field":
        row.pop("source_video_sha256")
    elif corruption == "extra_sampling_field":
        row["sampling_window"]["unexpected"] = 1
    elif corruption == "bool_sampling_frame":
        row["sampling_window"]["start_frame"] = True
    elif corruption == "string_sampling_frame":
        row["sampling_window"]["start_frame"] = "10"
    elif corruption == "wrong_sampling_protocol":
        row["sampling_window"]["protocol"] = "wrong-protocol"
    elif corruption == "invalid_sampling_bounds":
        row["sampling_window"]["start_frame"] = 999
    elif corruption == "bool_embedding":
        row["embedding"][0] = True
    elif corruption == "string_embedding":
        row["embedding"][0] = "1.0"
    elif corruption == "wrong_embedding_dimension":
        row["embedding"].pop()
    elif corruption == "nonfinite_embedding":
        row["embedding"][0] = float("nan")
    else:  # pragma: no cover - exhaustive parameter list
        raise AssertionError(corruption)
    artifact.pop("artifact_sha256")
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    encoded = (json.dumps(artifact, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    inputs["staging"].write_bytes(encoded)
    return _sha256(encoded)


def test_verified_resume_is_loaded_before_model_and_continues_at_next_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _valid_extraction_inputs(tmp_path, event_count=2)
    kwargs = _valid_extract_kwargs(inputs)
    kwargs["resume"] = True
    kwargs["expected_resume_checkpoint_sha256"] = _write_partial_resume_checkpoint(inputs)
    _loaded_snapshots, extracted_event_ids = _install_lightweight_extraction_fakes(
        monkeypatch,
        inputs,
    )

    artifact = embedding_cli.extract_labeled_video_embeddings(**kwargs)

    assert extracted_event_ids == ["event-1"]
    assert [row["event_id"] for row in artifact["examples"]] == ["event-0", "event-1"]
    assert not inputs["staging"].exists()
    assert json.loads(inputs["output"].read_text(encoding="utf-8")) == artifact


@pytest.mark.parametrize(
    "corruption",
    ("external_receipt", "provenance", "row_prefix"),
)
def test_resume_receipt_and_provenance_fail_before_opencv_model_or_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    inputs = _valid_extraction_inputs(tmp_path, event_count=2)
    expected_resume_sha256 = _write_partial_resume_checkpoint(inputs)
    if corruption == "external_receipt":
        expected_resume_sha256 = "b" * 64
    else:
        artifact = json.loads(inputs["staging"].read_text(encoding="utf-8"))
        if corruption == "provenance":
            artifact["backbone_sha256"] = "b" * 64
        else:
            artifact["examples"][0]["event_id"] = "wrong-event"
        artifact = seal_video_embedding_artifact(
            {key: value for key, value in artifact.items() if key != "artifact_sha256"}
        )
        encoded = (json.dumps(artifact, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        inputs["staging"].write_bytes(encoded)
        expected_resume_sha256 = _sha256(encoded)
    kwargs = _valid_extract_kwargs(inputs)
    kwargs["resume"] = True
    kwargs["expected_resume_checkpoint_sha256"] = expected_resume_sha256
    _allow_resource_preflight(monkeypatch)
    _reject_heavy_or_write_work(monkeypatch)

    with pytest.raises(ValueError, match="resume|checkpoint|provenance|SHA|sha256"):
        embedding_cli.extract_labeled_video_embeddings(**kwargs)

    assert not inputs["output"].exists()


@pytest.mark.parametrize(
    "corruption",
    (
        "extra_top_level_field",
        "extra_example_field",
        "missing_example_field",
        "extra_sampling_field",
        "bool_sampling_frame",
        "string_sampling_frame",
        "wrong_sampling_protocol",
        "invalid_sampling_bounds",
        "bool_embedding",
        "string_embedding",
        "wrong_embedding_dimension",
        "nonfinite_embedding",
    ),
)
def test_resume_rows_require_exact_typed_schema_before_opencv_or_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    inputs = _valid_extraction_inputs(tmp_path, event_count=2)
    _write_partial_resume_checkpoint(inputs)
    kwargs = _valid_extract_kwargs(inputs)
    kwargs["resume"] = True
    kwargs["expected_resume_checkpoint_sha256"] = _rewrite_resume_checkpoint(
        inputs,
        corruption,
    )
    _allow_resource_preflight(monkeypatch)
    _reject_heavy_or_write_work(monkeypatch)

    with pytest.raises(
        ValueError,
        match="resume|checkpoint|example|sampling|embedding|window",
    ):
        embedding_cli.extract_labeled_video_embeddings(**kwargs)

    assert not inputs["output"].exists()


def test_fresh_run_rejects_existing_staging_before_any_input_read_or_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _valid_extraction_inputs(tmp_path)
    inputs["staging"].write_bytes(b"existing-stage")
    kwargs = _valid_extract_kwargs(inputs)
    _reject_preflight_side_effects(monkeypatch)

    with pytest.raises(ValueError, match="staging|resume"):
        embedding_cli.extract_labeled_video_embeddings(**kwargs)

    assert inputs["staging"].stat().st_size == len(b"existing-stage")
    assert not inputs["output"].exists()


def _input_paths(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "manifest": tmp_path / "training-manifest.json",
        "candidate_bundle": tmp_path / "candidate-bundle.json",
        "annotation": tmp_path / "labels.json",
        "video": tmp_path / "source.webm",
        "backbone_checkpoint": tmp_path / "mvit.pth",
    }
    for index, path in enumerate(paths.values()):
        path.write_bytes(f"input-{index}\n".encode())
    return paths


def _reject_preflight_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("preflight failure must precede file reads/OpenCV/model/device/write work")

    monkeypatch.setattr(Path, "read_text", reject)
    monkeypatch.setattr(Path, "read_bytes", reject)
    monkeypatch.setattr(Path, "open", reject)
    monkeypatch.setattr(embedding_cli, "_resolve_device", reject)
    monkeypatch.setattr(embedding_cli, "get_video_backbone_spec", reject)
    monkeypatch.setattr(embedding_cli, "load_video_backbone", reject)
    monkeypatch.setattr(embedding_cli, "extract_video_embeddings", reject)
    monkeypatch.setattr(embedding_cli.cv2, "VideoCapture", reject)
    monkeypatch.setattr(embedding_cli, "file_sha256", reject, raising=False)
    monkeypatch.setattr(embedding_cli.tempfile, "NamedTemporaryFile", reject)
    monkeypatch.setattr(embedding_cli, "_write_json_atomic", reject)


def _extract_with_preflight(
    paths: dict[str, Path],
    output_path: Path,
    **overrides: object,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "manifest_path": paths["manifest"],
        "candidate_bundle_paths": [paths["candidate_bundle"]],
        "annotation_paths": [paths["annotation"]],
        "video_paths": [paths["video"]],
        "backbone_name": "torchvision/mvit_v2_s/kinetics400_v1",
        "backbone_checkpoint": paths["backbone_checkpoint"],
        "clip_frames": 16,
        "batch_size": 1,
        "device_name": "cpu",
        "expected_backbone_sha256": "a" * 64,
        "expected_training_manifest_sha256": "c" * 64,
        "expected_training_manifest_file_sha256": "d" * 64,
        "checkpoint_path": output_path,
        "resume": True,
        "expected_resume_checkpoint_sha256": "b" * 64,
    }
    kwargs.update(overrides)
    return embedding_cli.extract_labeled_video_embeddings(**kwargs)


@pytest.mark.parametrize("batch_size", (2, 1.0, True))
def test_extraction_rejects_non_integer_or_non_unit_batch_before_heavy_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    batch_size: object,
) -> None:
    paths = _input_paths(tmp_path)
    output_path = tmp_path / "output" / "embeddings.json"
    _reject_preflight_side_effects(monkeypatch)

    def reject_resource_probe(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("batch validation must precede resource probes")

    monkeypatch.setattr(embedding_cli.psutil, "virtual_memory", reject_resource_probe)
    monkeypatch.setattr(embedding_cli.shutil, "disk_usage", reject_resource_probe)

    with pytest.raises(ValueError, match="batch size.*1"):
        _extract_with_preflight(paths, output_path, batch_size=batch_size)

    assert not output_path.exists()


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"max_system_memory_percent": 90.0001}, "max_system_memory_percent"),
        ({"max_system_memory_percent": float("nan")}, "max_system_memory_percent"),
        ({"max_system_memory_percent": True}, "max_system_memory_percent"),
        ({"min_available_memory_gib": 1.9999}, "min_available_memory_gib"),
        ({"min_available_memory_gib": float("inf")}, "min_available_memory_gib"),
        ({"min_available_memory_gib": True}, "min_available_memory_gib"),
        ({"min_output_free_gib": 1.9999}, "min_output_free_gib"),
        ({"min_output_free_gib": float("nan")}, "min_output_free_gib"),
        ({"min_output_free_gib": True}, "min_output_free_gib"),
    ),
)
def test_extraction_rejects_weaker_or_nonfinite_preflight_thresholds_before_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, float],
    message: str,
) -> None:
    paths = _input_paths(tmp_path)
    output_parent = tmp_path / "output"
    output_parent.mkdir()
    output_path = output_parent / "embeddings.json"
    _reject_preflight_side_effects(monkeypatch)

    def reject_resource_probe(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("invalid thresholds must fail before resource probes")

    monkeypatch.setattr(embedding_cli.psutil, "virtual_memory", reject_resource_probe)
    monkeypatch.setattr(embedding_cli.shutil, "disk_usage", reject_resource_probe)

    with pytest.raises(ValueError, match=message):
        _extract_with_preflight(paths, output_path, **overrides)

    assert not output_path.exists()


@pytest.mark.parametrize(
    ("memory_percent", "available_bytes", "free_bytes", "message"),
    (
        (90.01, 3 * GIB, 3 * GIB, "system memory"),
        (50.0, 2 * GIB - 1, 3 * GIB, "available memory"),
        (50.0, 3 * GIB, 2 * GIB - 1, "output filesystem"),
    ),
)
def test_extraction_resource_preflight_fails_before_json_model_or_checkpoint_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    memory_percent: float,
    available_bytes: int,
    free_bytes: int,
    message: str,
) -> None:
    paths = _input_paths(tmp_path)
    output_parent = tmp_path / "output"
    output_parent.mkdir()
    output_path = output_parent / "embeddings.json"
    disk_probe_paths: list[Path] = []
    _reject_preflight_side_effects(monkeypatch)
    monkeypatch.setattr(
        embedding_cli.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=memory_percent, available=available_bytes),
    )

    def disk_usage(path: Path) -> SimpleNamespace:
        disk_probe_paths.append(path)
        return SimpleNamespace(free=free_bytes)

    monkeypatch.setattr(embedding_cli.shutil, "disk_usage", disk_usage)

    with pytest.raises(RuntimeError, match=message):
        _extract_with_preflight(paths, output_path)

    if memory_percent <= 90.0 and available_bytes >= 2 * GIB:
        assert disk_probe_paths == [output_parent.resolve()]
    else:
        assert disk_probe_paths == []
    assert not output_path.exists()


def test_backbone_snapshot_volume_is_preflighted_without_staging_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _input_paths(tmp_path)
    disk_probe_paths: list[Path] = []
    _reject_preflight_side_effects(monkeypatch)
    monkeypatch.setattr(
        embedding_cli.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=50.0, available=3 * GIB),
    )

    def disk_usage(path: Path) -> SimpleNamespace:
        disk_probe_paths.append(path)
        return SimpleNamespace(free=2 * GIB - 1)

    monkeypatch.setattr(embedding_cli.shutil, "disk_usage", disk_usage)

    with pytest.raises(RuntimeError, match="output filesystem"):
        embedding_cli.extract_labeled_video_embeddings(
            manifest_path=paths["manifest"],
            candidate_bundle_paths=[paths["candidate_bundle"]],
            annotation_paths=[paths["annotation"]],
            video_paths=[paths["video"]],
            backbone_name=BACKBONE,
            backbone_checkpoint=paths["backbone_checkpoint"],
            expected_backbone_sha256="a" * 64,
            expected_training_manifest_sha256="b" * 64,
            expected_training_manifest_file_sha256="c" * 64,
            clip_frames=16,
            batch_size=1,
            device_name="cpu",
        )

    assert disk_probe_paths == [tmp_path.resolve()]


@pytest.mark.parametrize(
    "input_name",
    (
        "manifest",
        "candidate_bundle",
        "annotation",
        "video",
        "backbone_checkpoint",
    ),
)
@pytest.mark.parametrize(
    "alias_kind",
    (
        "direct",
        "output_symlink",
        "input_symlink",
        "output_hardlink",
        "case_variant",
        "output_ancestor",
        "output_descendant",
    ),
)
def test_extraction_rejects_checkpoint_input_alias_before_reading_or_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_name: str,
    alias_kind: str,
) -> None:
    paths = _input_paths(tmp_path)
    checkpoint_path = paths[input_name]
    if alias_kind == "output_symlink":
        checkpoint_path = tmp_path / f"{input_name}-output-alias.json"
        checkpoint_path.symlink_to(paths[input_name])
    elif alias_kind == "input_symlink":
        physical_input = tmp_path / f"{input_name}-physical"
        paths[input_name].replace(physical_input)
        paths[input_name].symlink_to(physical_input)
        checkpoint_path = physical_input
    elif alias_kind == "output_hardlink":
        checkpoint_path = tmp_path / f"{input_name}-output-hardlink.json"
        os.link(paths[input_name], checkpoint_path)
    elif alias_kind == "case_variant":
        checkpoint_path = paths[input_name].with_name(paths[input_name].name.swapcase())
    elif alias_kind == "output_ancestor":
        checkpoint_path = tmp_path
    elif alias_kind == "output_descendant":
        checkpoint_path = paths[input_name] / "nested-output.json"
    before = {name: path.resolve().read_bytes() for name, path in paths.items()}

    def reject_read(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("input alias must fail before reading JSON")

    def reject_model_load(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("input alias must fail before model loading")

    monkeypatch.setattr(Path, "read_text", reject_read)
    monkeypatch.setattr(embedding_cli, "load_video_backbone", reject_model_load)

    with pytest.raises(ValueError, match="checkpoint.*input|output.*input|alias"):
        embedding_cli.extract_labeled_video_embeddings(
            manifest_path=paths["manifest"],
            candidate_bundle_paths=[paths["candidate_bundle"]],
            annotation_paths=[paths["annotation"]],
            video_paths=[paths["video"]],
            backbone_name="torchvision/mvit_v2_s/kinetics400_v1",
            backbone_checkpoint=paths["backbone_checkpoint"],
            clip_frames=16,
            batch_size=1,
            device_name="cpu",
            expected_backbone_sha256="a" * 64,
            expected_training_manifest_sha256="c" * 64,
            expected_training_manifest_file_sha256="d" * 64,
            checkpoint_path=checkpoint_path,
            resume=True,
        )

    assert {name: path.resolve().read_bytes() for name, path in paths.items()} == before


def test_embedding_json_writer_fsyncs_then_atomically_replaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "embeddings.json"
    destination.write_text("old artifact\n", encoding="utf-8")
    original_replace = Path.replace
    events: list[str] = []

    def observe_fsync(file_descriptor: int) -> None:
        assert file_descriptor >= 0
        events.append("fsync")

    def observe_replace(source: Path, target: Path) -> Path:
        assert source.parent == destination.parent
        assert source.name.startswith(f".{destination.name}.")
        assert source.suffix == ".tmp"
        assert json.loads(source.read_text(encoding="utf-8")) == {"new": True}
        assert destination.read_text(encoding="utf-8") == "old artifact\n"
        events.append("replace")
        return original_replace(source, target)

    monkeypatch.setattr(embedding_cli.os, "fsync", observe_fsync)
    monkeypatch.setattr(Path, "replace", observe_replace)

    embedding_cli._write_json_atomic(destination, {"new": True})

    assert events == ["fsync", "replace"]
    assert json.loads(destination.read_text(encoding="utf-8")) == {"new": True}
    assert list(tmp_path.iterdir()) == [destination]


def test_embedding_json_writer_preserves_destination_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "embeddings.json"
    destination.write_text("old artifact\n", encoding="utf-8")

    def reject_replace(_source: Path, _target: Path) -> Path:
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", reject_replace)

    with pytest.raises(OSError, match="replace failed"):
        embedding_cli._write_json_atomic(destination, {"new": True})

    assert destination.read_text(encoding="utf-8") == "old artifact\n"
    assert list(tmp_path.iterdir()) == [destination]


def test_embedding_cli_separates_resume_staging_from_final_output_and_forwards_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _input_paths(tmp_path)
    output_path = tmp_path / "embeddings.json"
    staging_path = tmp_path / "embeddings.resume.json"
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_shot_validity_video_embeddings.py",
            "--manifest",
            str(paths["manifest"]),
            "--candidate-bundle",
            str(paths["candidate_bundle"]),
            "--annotation",
            str(paths["annotation"]),
            "--video",
            str(paths["video"]),
            "--backbone",
            "torchvision/mvit_v2_s/kinetics400_v1",
            "--backbone-checkpoint",
            str(paths["backbone_checkpoint"]),
            "--expected-backbone-sha256",
            "a" * 64,
            "--expected-training-manifest-sha256",
            "b" * 64,
            "--expected-training-manifest-file-sha256",
            "c" * 64,
            "--resume-checkpoint",
            str(staging_path),
            "--expected-resume-checkpoint-sha256",
            "d" * 64,
            "--output",
            str(output_path),
            "--max-memory-percent",
            "85",
            "--min-available-gib",
            "3",
            "--min-output-free-gib",
            "4",
            "--resume",
        ],
    )

    def fake_extract(**kwargs: object) -> dict[str, object]:
        captured["extract"] = kwargs
        return {
            "examples": [],
            "backbone": "torchvision/mvit_v2_s/kinetics400_v1",
            "artifact_sha256": "a" * 64,
        }

    def fake_atomic_write(path: Path, payload: dict[str, object]) -> None:
        raise AssertionError(f"main must not separately publish {path}: {payload}")

    monkeypatch.setattr(embedding_cli, "extract_labeled_video_embeddings", fake_extract)
    monkeypatch.setattr(
        embedding_cli,
        "_write_json_atomic",
        fake_atomic_write,
        raising=False,
    )

    assert embedding_cli.main() == 0
    extract_call = captured["extract"]
    assert isinstance(extract_call, dict)
    assert extract_call["resume"] is True
    assert extract_call["resume_checkpoint_path"] == staging_path.resolve()
    assert extract_call["output_path"] == output_path.resolve()
    assert extract_call["expected_backbone_sha256"] == "a" * 64
    assert extract_call["expected_training_manifest_sha256"] == "b" * 64
    assert extract_call["expected_training_manifest_file_sha256"] == "c" * 64
    assert extract_call["expected_resume_checkpoint_sha256"] == "d" * 64
    assert extract_call["max_system_memory_percent"] == 85.0
    assert extract_call["min_available_memory_gib"] == 3.0
    assert extract_call["min_output_free_gib"] == 4.0


@pytest.mark.parametrize(
    ("flag", "value"),
    (
        ("--batch-size", "2"),
        ("--batch-size", "1.0"),
        ("--max-memory-percent", "90.0001"),
        ("--max-memory-percent", "nan"),
        ("--max-memory-percent", "true"),
        ("--min-available-gib", "1.9999"),
        ("--min-available-gib", "inf"),
        ("--min-output-free-gib", "1.9999"),
        ("--min-output-free-gib", "nan"),
        ("--expected-backbone-sha256", "not-a-sha"),
        ("--expected-training-manifest-sha256", "z" * 64),
        ("--expected-training-manifest-file-sha256", "1234"),
    ),
)
def test_embedding_cli_rejects_unsafe_preflight_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    value: str,
) -> None:
    paths = _input_paths(tmp_path)
    output_path = tmp_path / "embeddings.json"
    staging_path = tmp_path / "embeddings.resume.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_shot_validity_video_embeddings.py",
            "--manifest",
            str(paths["manifest"]),
            "--candidate-bundle",
            str(paths["candidate_bundle"]),
            "--annotation",
            str(paths["annotation"]),
            "--video",
            str(paths["video"]),
            "--backbone",
            "torchvision/mvit_v2_s/kinetics400_v1",
            "--backbone-checkpoint",
            str(paths["backbone_checkpoint"]),
            "--expected-backbone-sha256",
            "a" * 64,
            "--expected-training-manifest-sha256",
            "b" * 64,
            "--expected-training-manifest-file-sha256",
            "c" * 64,
            "--resume-checkpoint",
            str(staging_path),
            "--output",
            str(output_path),
            flag,
            value,
        ],
    )

    with pytest.raises(SystemExit) as error:
        embedding_cli.parse_args()

    assert error.value.code == 2


def test_embedding_cli_requires_expected_resume_receipt_with_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _input_paths(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_shot_validity_video_embeddings.py",
            "--manifest",
            str(paths["manifest"]),
            "--candidate-bundle",
            str(paths["candidate_bundle"]),
            "--annotation",
            str(paths["annotation"]),
            "--video",
            str(paths["video"]),
            "--backbone",
            BACKBONE,
            "--backbone-checkpoint",
            str(paths["backbone_checkpoint"]),
            "--expected-backbone-sha256",
            "a" * 64,
            "--expected-training-manifest-sha256",
            "b" * 64,
            "--expected-training-manifest-file-sha256",
            "c" * 64,
            "--resume-checkpoint",
            str(tmp_path / "embeddings.resume.json"),
            "--output",
            str(tmp_path / "embeddings.json"),
            "--resume",
        ],
    )

    with pytest.raises(SystemExit) as error:
        embedding_cli.parse_args()

    assert error.value.code == 2
