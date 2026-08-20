from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from app.analysis.shot_validity_sampling_window import SHOT_SAMPLING_PROTOCOL
from app.analysis.shot_validity_video_backbone import (
    get_video_backbone_spec,
    seal_video_embedding_artifact,
)
from app.analysis.vru_causal_training import (
    verify_vru_causal_shot_validity_training_export,
)
from app.analysis.vru_causal_video_probe import (
    screen_vru_causal_video_embeddings_nested,
    verify_vru_causal_video_probe,
)
from scripts import screen_vru_causal_video_embeddings_nested as probe_cli

SOURCE_SHAS = ("a" * 64, "b" * 64, "c" * 64)
BUNDLE_SHAS = ("1" * 64, "2" * 64, "3" * 64)
SOURCE_COUNTS = ((6, 2), (4, 3), (4, 3))
CLI_EXPORT_SHA = "e" * 64


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _examples() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source_index, (positive_count, negative_count) in enumerate(SOURCE_COUNTS):
        for label, count in ((True, positive_count), (False, negative_count)):
            for row_index in range(count):
                rows.append(
                    {
                        "source_video_sha256": SOURCE_SHAS[source_index],
                        "candidate_bundle_sha256": BUNDLE_SHAS[source_index],
                        "event_id": (
                            f"source-{source_index}-"
                            f"{'shot' if label else 'nonshot'}-{row_index}"
                        ),
                        "event_present": label,
                    }
                )
    return rows


def _training_export(
    examples: list[dict[str, object]],
) -> dict[str, object]:
    source_ids = ("hazen", "randolph", "vtv")
    sources = []
    for index, source_id in enumerate(source_ids):
        rows = [
            row
            for row in examples
            if row["source_video_sha256"] == SOURCE_SHAS[index]
        ]
        positive_count = sum(bool(row["event_present"]) for row in rows)
        sources.append(
            {
                "source_id": source_id,
                "source_video_filename": f"{source_id}.webm",
                "source_video_sha256": SOURCE_SHAS[index],
                "source_video_size_bytes": 1,
                "candidate_bundle": {
                    "filename": f"{source_id}_candidate_bundle_v1.json",
                    "sha256": str(index + 4) * 64,
                    "size_bytes": 1,
                    "bundle_sha256": BUNDLE_SHAS[index],
                },
                "labels": {
                    "filename": f"{source_id}_shot_validity_labels_v1.json",
                    "sha256": str(index + 7) * 64,
                    "size_bytes": 1,
                },
                "example_count": len(rows),
                "positive_count": positive_count,
                "negative_count": len(rows) - positive_count,
            }
        )
    payload: dict[str, object] = {
        "schema_version": "agu.vru-causal-shot-validity-training-export.v1",
        "purpose": "development_diagnostic_only",
        "runtime_consumable": False,
        "training_consumable": True,
        "formal_evaluation_eligible": False,
        "codex_runtime_answer_used": False,
        "source_selection": {
            "schema_version": "agu.vru-causal-closure-selection.v1",
            "artifact_sha256": "a" * 64,
            "file_sha256": "b" * 64,
            "training_consumable": False,
            "prior_labels_used_for_stratified_selection": True,
            "prior_labels_used_as_final_8s_targets": False,
        },
        "source_review_plan": {
            "schema_version": "agu.vru-causal-review-plan.v1",
            "artifact_sha256": "c" * 64,
            "file_sha256": "d" * 64,
        },
        "source_review": {
            "schema_version": "agu.vru-causal-offline-review.v1",
            "artifact_sha256": "e" * 64,
            "file_sha256": "f" * 64,
        },
        "source_frame_manifest": {
            "schema_version": "agu.vru-causal-review-frame-manifest.v1",
            "artifact_sha256": "0" * 64,
            "file_sha256": "1" * 64,
        },
        "source_manifest": {
            "schema_version": "agu.vru-basketball-source-manifest.v1",
            "file_sha256": "2" * 64,
        },
        "mapping_policy": {
            "shot": True,
            "not_a_shot": False,
            "uncertain": "excluded",
            "outcome_used_for_shot_validity": False,
            "target_conditioned_anchor_used": False,
        },
        "training_routes": {
            "scene_representation": True,
            "video_representation": True,
            "traditional_feature_extra_trees": False,
            "runtime_inference": False,
        },
        "counts": {
            "selected": 24,
            "exported": 22,
            "positive": 14,
            "negative": 8,
            "excluded_uncertain": 2,
        },
        "excluded_review_ids": [
            "closure-randolph-0002",
            "closure-vtv-0003",
        ],
        "sources": sources,
        "examples": sorted(
            copy.deepcopy(examples),
            key=lambda row: (
                str(row["source_video_sha256"]),
                str(row["candidate_bundle_sha256"]),
                str(row["event_id"]),
            ),
        ),
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return payload


def _embedding_artifact(
    examples: list[dict[str, object]],
    *,
    manifest_sha256: str,
    training_annotation_sha256s: list[str],
    backbone: str = "torchvision/mvit_v2_s/kinetics400_v1",
) -> dict[str, object]:
    dimension = get_video_backbone_spec(backbone).embedding_dimension
    embedded_rows = []
    for index, row in enumerate(examples):
        center = -2.0 if "-nonshot-" in str(row["event_id"]) else 2.0
        source_offset = SOURCE_SHAS.index(str(row["source_video_sha256"])) * 0.05
        embedded_rows.append(
            {
                **copy.deepcopy(row),
                "sampling_window": {
                    "start_frame": index * 100,
                    "end_frame": index * 100 + 63,
                    "anchor_frame": index * 100 + 63,
                    "anchor_source": "bounded_event",
                    "protocol": SHOT_SAMPLING_PROTOCOL,
                },
                "embedding": [center + source_offset + index * 0.001] * dimension,
            }
        )
    return seal_video_embedding_artifact(
        {
            "purpose": "backbone_screening_training_only",
            "producer": "agu",
            "training_manifest_sha256": manifest_sha256,
            "training_annotation_sha256": sorted(training_annotation_sha256s),
            "source_video_sha256": sorted(SOURCE_SHAS),
            "backbone": backbone,
            "backbone_sha256": "d" * 64,
            "embedding_dimension": dimension,
            "clip_frames": 16,
            "sampling_protocol": SHOT_SAMPLING_PROTOCOL,
            "examples": embedded_rows,
        }
    )


def _write_export_and_manifest(
    tmp_path: Path,
    export: dict[str, object],
) -> tuple[Path, Path, dict[str, object]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    export_path = tmp_path / "closure-training-export.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")
    export_file_sha = hashlib.sha256(export_path.read_bytes()).hexdigest()
    manifest_payload: dict[str, object] = {
        "schema_version": "agu.training-annotation.v1",
        "purpose": "model_training_only",
        "producer": "test",
        "runtime_consumable": False,
        "benchmark_overlap": False,
        "source_videos": [
            {
                "filename": source["source_video_filename"],
                "sha256": source["source_video_sha256"],
                "size_bytes": source["source_video_size_bytes"],
            }
            for source in export["sources"]
        ],
        "annotation_files": [
            {
                "filename": export_path.name,
                "sha256": export_file_sha,
                "size_bytes": export_path.stat().st_size,
            },
            *(copy.deepcopy(source["labels"]) for source in export["sources"]),
        ],
        "task_types": ["shot_validity"],
        "benchmark_raw_sha256": ["f" * 64],
    }
    manifest_payload["manifest_sha256"] = _canonical_sha256(manifest_payload)
    manifest_path = tmp_path / "training-manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    return export_path, manifest_path, manifest_payload


def _screen(
    tmp_path: Path,
    *,
    examples: list[dict[str, object]] | None = None,
    mutate_embedding: Callable[[dict[str, object]], None] | None = None,
    mutate_manifest: Callable[[dict[str, object]], None] | None = None,
    backbones: tuple[str, ...] = ("torchvision/mvit_v2_s/kinetics400_v1",),
) -> dict[str, object]:
    rows = copy.deepcopy(examples if examples is not None else _examples())
    export = _training_export(rows)
    export_path, manifest_path, manifest = _write_export_and_manifest(tmp_path, export)
    if mutate_manifest is not None:
        manifest.pop("manifest_sha256")
        mutate_manifest(manifest)
        manifest["manifest_sha256"] = _canonical_sha256(manifest)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    embedding_paths = []
    for index, backbone in enumerate(backbones):
        embedding = _embedding_artifact(
            copy.deepcopy(export["examples"]),
            manifest_sha256=str(manifest["manifest_sha256"]),
            training_annotation_sha256s=[
                str(source["labels"]["sha256"]) for source in export["sources"]
            ],
            backbone=backbone,
        )
        if mutate_embedding is not None and index == 0:
            embedding.pop("artifact_sha256")
            mutate_embedding(embedding)
            embedding = seal_video_embedding_artifact(embedding)
        embedding_path = tmp_path / f"video-embeddings-{index}.json"
        embedding_path.write_text(json.dumps(embedding), encoding="utf-8")
        embedding_paths.append(embedding_path)

    return screen_vru_causal_video_embeddings_nested(
        embedding_paths=embedding_paths,
        training_export_path=export_path,
        training_manifest_path=manifest_path,
        expected_training_export_sha256=str(export["artifact_sha256"]),
    )


def test_nested_probe_is_export_bound_and_diagnostic_only(
    tmp_path: Path,
) -> None:
    export = _training_export(_examples())
    assert verify_vru_causal_shot_validity_training_export(export) == export
    result = _screen(tmp_path)

    assert result["schema_version"] == "agu.vru-causal-video-representation-probe.v1"
    assert result["runtime_consumable"] is False
    assert result["formal_evaluation_eligible"] is False
    assert result["promotion_eligible"] is False
    assert result["promoted"] is False
    assert result["regularization_c"] == 0.01
    assert result["row_count"] == 22
    assert result["source_count"] == 3
    assert result["training_export_artifact_sha256"] == export["artifact_sha256"]
    assert result["source_selection"]["prior_stratified"] is True
    assert result["source_selection"]["limitation"]
    assert len(result["outer_folds"]) == 3
    for fold in result["outer_folds"]:
        held = fold["held_source_video_sha256"]
        assert held not in fold["inner_selection_source_video_sha256s"]
        assert held not in fold["fit_source_video_sha256s"]
        assert fold["regularization_c"] == 0.01
        assert fold["oof_predictions"]
        assert all("event_present" not in row for row in fold["oof_predictions"])


def test_probe_artifact_is_hash_bound_and_rejects_policy_tampering(
    tmp_path: Path,
) -> None:
    result = _screen(tmp_path)

    assert verify_vru_causal_video_probe(result)["artifact_sha256"] == result[
        "artifact_sha256"
    ]
    result["formal_evaluation_eligible"] = True

    with pytest.raises(ValueError, match="hash mismatch|formal evaluation"):
        verify_vru_causal_video_probe(result)


def test_probe_verifier_rejects_rehashed_noncanonical_closure_shape(
    tmp_path: Path,
) -> None:
    result = _screen(tmp_path)
    result.pop("artifact_sha256")
    result["row_count"] = 21
    result["artifact_sha256"] = _canonical_sha256(result)

    with pytest.raises(ValueError, match="22-row|22 row|row count"):
        verify_vru_causal_video_probe(result)


def test_probe_verifier_rejects_rehashed_threshold_policy_drift(
    tmp_path: Path,
) -> None:
    result = _screen(tmp_path)
    result.pop("artifact_sha256")
    result["minimum_precision"] = 0.0
    result["artifact_sha256"] = _canonical_sha256(result)

    with pytest.raises(ValueError, match="minimum precision|threshold policy"):
        verify_vru_causal_video_probe(result)


def test_probe_verifier_rejects_rehashed_outer_fold_leakage(
    tmp_path: Path,
) -> None:
    result = _screen(tmp_path)
    result.pop("artifact_sha256")
    fold = result["outer_folds"][0]
    fold["fit_source_video_sha256s"].append(fold["held_source_video_sha256"])
    result["artifact_sha256"] = _canonical_sha256(result)

    with pytest.raises(ValueError, match="held.*fit|outer fold.*disjoint|leak"):
        verify_vru_causal_video_probe(result)


@pytest.mark.parametrize("duplicate_field", ("backbone", "artifact_sha256"))
def test_probe_verifier_rejects_rehashed_duplicate_embedding_receipt_identity(
    tmp_path: Path,
    duplicate_field: str,
) -> None:
    result = _screen(
        tmp_path,
        backbones=(
            "torchvision/mvit_v2_s/kinetics400_v1",
            "torchvision/swin3d_t/kinetics400_v1",
        ),
    )
    result.pop("artifact_sha256")
    receipts = result["video_embedding_artifacts"]
    receipts[1][duplicate_field] = receipts[0][duplicate_field]
    receipts.sort(key=lambda row: (row["backbone"], row["artifact_sha256"]))
    result["artifact_sha256"] = _canonical_sha256(result)

    with pytest.raises(ValueError, match="backbone.*unique|artifact.*unique|receipts.*unique"):
        verify_vru_causal_video_probe(result)


def _metrics_from_counts(*, tp: int, fp: int, fn: int, tn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def test_probe_verifier_rejects_rehashed_metrics_that_contradict_decisions(
    tmp_path: Path,
) -> None:
    result = _screen(tmp_path)
    result.pop("artifact_sha256")
    fold = result["outer_folds"][0]
    original = fold["held_metrics"]
    positive_count = original["tp"] + original["fn"]
    negative_count = original["fp"] + original["tn"]
    decision_count = sum(row["decision"] for row in fold["oof_predictions"])
    if decision_count == 0:
        forged = _metrics_from_counts(
            tp=positive_count,
            fp=negative_count,
            fn=0,
            tn=0,
        )
    else:
        forged = _metrics_from_counts(
            tp=0,
            fp=0,
            fn=positive_count,
            tn=negative_count,
        )
    fold["held_metrics"] = forged
    held_source = fold["held_source_video_sha256"]
    result["observed_metrics"]["per_source"][held_source] = forged
    fold_metrics = [row["held_metrics"] for row in result["outer_folds"]]
    result["observed_metrics"]["pooled"] = _metrics_from_counts(
        tp=sum(row["tp"] for row in fold_metrics),
        fp=sum(row["fp"] for row in fold_metrics),
        fn=sum(row["fn"] for row in fold_metrics),
        tn=sum(row["tn"] for row in fold_metrics),
    )
    result["artifact_sha256"] = _canonical_sha256(result)

    with pytest.raises(ValueError, match="decision.*metrics|confusion"):
        verify_vru_causal_video_probe(result)


def test_outer_held_label_flip_does_not_change_that_fold_threshold_or_predictions(
    tmp_path: Path,
) -> None:
    original = _screen(tmp_path / "original")
    flipped_rows = _examples()
    held_rows = [
        row for row in flipped_rows if row["source_video_sha256"] == SOURCE_SHAS[0]
    ]
    positive = next(row for row in held_rows if row["event_present"] is True)
    negative = next(row for row in held_rows if row["event_present"] is False)
    positive["event_present"] = False
    negative["event_present"] = True
    flipped = _screen(tmp_path / "flipped", examples=flipped_rows)

    original_fold = next(
        fold
        for fold in original["outer_folds"]
        if fold["held_source_video_sha256"] == SOURCE_SHAS[0]
    )
    flipped_fold = next(
        fold
        for fold in flipped["outer_folds"]
        if fold["held_source_video_sha256"] == SOURCE_SHAS[0]
    )

    assert original_fold["threshold"] == pytest.approx(flipped_fold["threshold"])
    assert original_fold["oof_predictions"] == flipped_fold["oof_predictions"]
    assert original_fold["held_metrics"] != flipped_fold["held_metrics"]


def test_probe_rejects_embedding_without_export_bound_extractor_provenance(
    tmp_path: Path,
) -> None:
    def strip_provenance(payload: dict[str, object]) -> None:
        for field in (
            "producer",
            "training_annotation_sha256",
            "source_video_sha256",
            "sampling_protocol",
        ):
            payload.pop(field)
        for row in payload["examples"]:
            row.pop("sampling_window")

    with pytest.raises(ValueError, match="provenance|sampling|annotation"):
        _screen(tmp_path, mutate_embedding=strip_provenance)


def test_probe_rejects_target_conditioned_sampling_even_when_resealed(
    tmp_path: Path,
) -> None:
    def expose_release_anchor(payload: dict[str, object]) -> None:
        sampling_window = payload["examples"][0]["sampling_window"]
        sampling_window["anchor_source"] = "release_frame"

    with pytest.raises(ValueError, match="label-hidden|bounded-event"):
        _screen(tmp_path, mutate_embedding=expose_release_anchor)


def test_probe_rejects_embedding_label_mismatch_with_verified_export(
    tmp_path: Path,
) -> None:
    def flip_embedding_label(payload: dict[str, object]) -> None:
        row = payload["examples"][0]
        row["event_present"] = not row["event_present"]

    with pytest.raises(ValueError, match="labels.*verified training export"):
        _screen(tmp_path, mutate_embedding=flip_embedding_label)


def test_probe_accepts_multiple_aligned_video_representations(tmp_path: Path) -> None:
    result = _screen(
        tmp_path,
        backbones=(
            "torchvision/mvit_v2_s/kinetics400_v1",
            "torchvision/swin3d_t/kinetics400_v1",
        ),
    )

    assert result["representation_dimension"] == 1536
    assert len(result["video_embedding_artifacts"]) == 2


def test_probe_aligns_shuffled_embedding_rows_by_exact_triple_key(
    tmp_path: Path,
) -> None:
    baseline = _screen(tmp_path / "baseline")

    def reverse_rows(payload: dict[str, object]) -> None:
        payload["examples"] = list(reversed(payload["examples"]))

    shuffled = _screen(
        tmp_path / "shuffled",
        mutate_embedding=reverse_rows,
    )

    def fold_projection(result: dict[str, object]) -> list[tuple[object, ...]]:
        return [
            (
                fold["held_source_video_sha256"],
                fold["threshold"],
                fold["oof_predictions"],
            )
            for fold in result["outer_folds"]
        ]

    assert fold_projection(shuffled) == fold_projection(baseline)
    assert shuffled["observed_metrics"] == baseline["observed_metrics"]


def test_probe_rejects_export_file_not_sha_bound_by_manifest(tmp_path: Path) -> None:
    def detach_export(manifest: dict[str, object]) -> None:
        manifest["annotation_files"][0]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="export file.*SHA-bound"):
        _screen(tmp_path, mutate_manifest=detach_export)


def test_probe_rejects_manifest_that_omits_export_label_assets(tmp_path: Path) -> None:
    def omit_labels(manifest: dict[str, object]) -> None:
        manifest["annotation_files"] = manifest["annotation_files"][:1]

    with pytest.raises(ValueError, match="label.*manifest|annotation.*exact"):
        _screen(tmp_path, mutate_manifest=omit_labels)


def test_probe_rejects_manifest_source_filename_drift(tmp_path: Path) -> None:
    def rename_source(manifest: dict[str, object]) -> None:
        manifest["source_videos"][0]["filename"] = "wrong-source.webm"

    with pytest.raises(ValueError, match="source.*manifest|source.*exact"):
        _screen(tmp_path, mutate_manifest=rename_source)


def test_probe_rejects_embedding_manifest_mismatch(tmp_path: Path) -> None:
    def rebind_embedding(payload: dict[str, object]) -> None:
        payload["training_manifest_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="embedding training manifest"):
        _screen(tmp_path, mutate_embedding=rebind_embedding)


def test_probe_rejects_export_path_swap_between_parse_and_file_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _examples()
    export = _training_export(rows)
    export_path, manifest_path, manifest = _write_export_and_manifest(
        tmp_path,
        export,
    )
    replacement_bytes = b'{"different":"manifest-bound-file"}\n'
    manifest.pop("manifest_sha256")
    manifest["annotation_files"][0] = {
        "filename": export_path.name,
        "sha256": hashlib.sha256(replacement_bytes).hexdigest(),
        "size_bytes": len(replacement_bytes),
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    embedding = _embedding_artifact(
        copy.deepcopy(export["examples"]),
        manifest_sha256=str(manifest["manifest_sha256"]),
        training_annotation_sha256s=[
            str(source["labels"]["sha256"]) for source in export["sources"]
        ],
    )
    embedding_path = tmp_path / "video-embeddings.json"
    embedding_path.write_text(json.dumps(embedding), encoding="utf-8")

    original_read_text = Path.read_text
    original_read_bytes = Path.read_bytes
    swapped = False

    def swap_after_read(path: Path) -> None:
        nonlocal swapped
        if path == export_path and not swapped:
            swapped = True
            path.write_bytes(replacement_bytes)

    def read_text_then_swap(path: Path, *args: object, **kwargs: object) -> str:
        value = original_read_text(path, *args, **kwargs)
        swap_after_read(path)
        return value

    def read_bytes_then_swap(path: Path) -> bytes:
        value = original_read_bytes(path)
        swap_after_read(path)
        return value

    monkeypatch.setattr(Path, "read_text", read_text_then_swap)
    monkeypatch.setattr(Path, "read_bytes", read_bytes_then_swap)

    with pytest.raises(ValueError, match="SHA-bound|snapshot|changed"):
        screen_vru_causal_video_embeddings_nested(
            embedding_paths=[embedding_path],
            training_export_path=export_path,
            training_manifest_path=manifest_path,
            expected_training_export_sha256=str(export["artifact_sha256"]),
        )


@pytest.mark.parametrize("input_name", ("embedding", "export", "manifest"))
@pytest.mark.parametrize(
    "alias_kind",
    (
        "direct",
        "output_symlink",
        "input_symlink",
        "output_hardlink",
        "case_variant",
    ),
)
def test_probe_cli_rejects_output_input_alias_before_screening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_name: str,
    alias_kind: str,
) -> None:
    paths = {
        "embedding": tmp_path / "embedding.json",
        "export": tmp_path / "export.json",
        "manifest": tmp_path / "manifest.json",
    }
    for path in paths.values():
        path.write_text("not read\n", encoding="utf-8")
    output = paths[input_name]
    if alias_kind == "output_symlink":
        output = tmp_path / f"{input_name}-output-alias.json"
        output.symlink_to(paths[input_name])
    elif alias_kind == "input_symlink":
        physical_input = tmp_path / f"{input_name}-physical.json"
        paths[input_name].replace(physical_input)
        paths[input_name].symlink_to(physical_input)
        output = physical_input
    elif alias_kind == "output_hardlink":
        output = tmp_path / f"{input_name}-output-hardlink.json"
        os.link(paths[input_name], output)
    elif alias_kind == "case_variant":
        output = paths[input_name].with_name(paths[input_name].name.swapcase())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "screen_vru_causal_video_embeddings_nested.py",
            "--embeddings",
            str(paths["embedding"]),
            "--training-export",
            str(paths["export"]),
            "--training-manifest",
            str(paths["manifest"]),
            "--expected-training-export-sha256",
            CLI_EXPORT_SHA,
            "--output",
            str(output),
        ],
    )

    def reject_screen(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("input/output alias must fail before screening")

    monkeypatch.setattr(
        probe_cli,
        "screen_vru_causal_video_embeddings_nested",
        reject_screen,
    )

    with pytest.raises(ValueError, match="output.*input"):
        probe_cli.main()


def test_probe_cli_writes_a_successful_diagnostic_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_path = tmp_path / "embedding.json"
    export_path = tmp_path / "export.json"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "probe.json"
    for path in (embedding_path, export_path, manifest_path):
        path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "screen_vru_causal_video_embeddings_nested.py",
            "--embeddings",
            str(embedding_path),
            "--training-export",
            str(export_path),
            "--training-manifest",
            str(manifest_path),
            "--expected-training-export-sha256",
            CLI_EXPORT_SHA,
            "--output",
            str(output_path),
        ],
    )

    def fake_screen(**kwargs: object) -> dict[str, object]:
        assert kwargs == {
            "embedding_paths": [embedding_path.resolve()],
            "training_export_path": export_path.resolve(),
            "training_manifest_path": manifest_path.resolve(),
            "expected_training_export_sha256": CLI_EXPORT_SHA,
        }
        return {
            "schema_version": "agu.vru-causal-video-representation-probe.v1",
            "formal_evaluation_eligible": False,
            "runtime_consumable": False,
            "promotion_eligible": False,
            "promoted": False,
            "row_count": 22,
            "source_count": 3,
            "artifact_sha256": "9" * 64,
        }

    monkeypatch.setattr(
        probe_cli,
        "screen_vru_causal_video_embeddings_nested",
        fake_screen,
    )

    assert probe_cli.main() == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["promoted"] is False


def test_probe_json_writer_fsyncs_then_atomically_replaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "probe.json"
    destination.write_text("old artifact\n", encoding="utf-8")
    original_replace = Path.replace
    events: list[str] = []

    def observe_fsync(file_descriptor: int) -> None:
        assert file_descriptor >= 0
        events.append("fsync")

    def observe_replace(source: Path, target: Path) -> Path:
        assert source.parent == destination.parent
        assert source.suffix == ".tmp"
        assert json.loads(source.read_text(encoding="utf-8")) == {"new": True}
        assert destination.read_text(encoding="utf-8") == "old artifact\n"
        events.append("replace")
        return original_replace(source, target)

    monkeypatch.setattr(probe_cli.os, "fsync", observe_fsync)
    monkeypatch.setattr(Path, "replace", observe_replace)

    probe_cli._write_json_atomic(destination, {"new": True})

    assert events == ["fsync", "replace"]
    assert json.loads(destination.read_text(encoding="utf-8")) == {"new": True}
    assert not list(tmp_path.glob("*.tmp"))
