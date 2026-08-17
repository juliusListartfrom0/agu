from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

import pytest

from app.analysis import vru_causal_video_probe_v2 as probe_module
from app.analysis.shot_validity_sampling_window import SHOT_SAMPLING_PROTOCOL
from app.analysis.shot_validity_video_backbone import seal_video_embedding_artifact
from app.analysis.vru_causal_source_groups import (
    SOURCE_GROUP_SPEC_SCHEMA,
    seal_vru_causal_source_groups,
)
from app.analysis.vru_causal_training import encode_json_artifact
from app.analysis.vru_causal_training_v2 import (
    EXPECTED_EXCLUDED_REVIEW_IDS,
    EXPORT_PURPOSE,
    FROZEN_HARWOOD_ARTIFACT_SHA256,
    FROZEN_HARWOOD_FILE_SHA256,
    FROZEN_HARWOOD_VIDEO_SHA256,
    FROZEN_HARWOOD_VIDEO_SIZE_BYTES,
    FROZEN_PARENT_ARTIFACT_SHA256,
    FROZEN_PARENT_CHILD_SHA256,
    FROZEN_PARENT_FILE_SHA256,
    HARWOOD_CANDIDATE_FILENAME,
    HARWOOD_LABEL_FILENAME,
    TRAINING_EXPORT_V2_SCHEMA,
    ContinuousCausalTrainingChain,
    verify_vru_causal_shot_validity_training_extension_index,
)
from app.analysis.vru_causal_video_probe_v2 import (
    VIDEO_PROBE_PLAN_V2_SCHEMA,
    screen_vru_causal_video_embeddings_nested_v2,
    seal_vru_causal_video_probe_plan_v2,
    verify_vru_causal_video_probe_plan_v2,
    verify_vru_causal_video_probe_v2,
)
from scripts import screen_vru_causal_video_embeddings_nested_v2 as probe_cli

MVIT = "torchvision/mvit_v2_s/kinetics400_v1"
SWIN = "torchvision/swin3d_t/kinetics400_v1"
GAME_ORDER = ("hazen", "randolph", "vtv", "harwood")
SOURCE_SHAS = ("1" * 64, "2" * 64, "3" * 64, FROZEN_HARWOOD_VIDEO_SHA256)
BUNDLE_SHAS = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)
SOURCE_COUNTS = ((6, 2), (4, 3), (4, 3), (3, 20))
GAME_FAMILIES = ("hctv", "hctv", "vtv", "hctv")
ACCEPTANCE_SHA256 = "9651a64355839780dffac2d404fa4715a08668b192a254cf87cbaba2d70c7f2e"
LABEL_ASSETS = (
    {
        "filename": "hazen_shot_validity_labels_v1.json",
        "sha256": FROZEN_PARENT_CHILD_SHA256["hazen_shot_validity_labels_v1.json"],
        "size_bytes": 101,
    },
    {
        "filename": "randolph_shot_validity_labels_v1.json",
        "sha256": FROZEN_PARENT_CHILD_SHA256["randolph_shot_validity_labels_v1.json"],
        "size_bytes": 102,
    },
    {
        "filename": "vtv_shot_validity_labels_v1.json",
        "sha256": FROZEN_PARENT_CHILD_SHA256["vtv_shot_validity_labels_v1.json"],
        "size_bytes": 103,
    },
    {
        "filename": HARWOOD_LABEL_FILENAME,
        "sha256": "9" * 64,
        "size_bytes": 104,
    },
)


def _plan_spec() -> dict[str, object]:
    return {
        "training_export_artifact_sha256": "1" * 64,
        "training_export_file_sha256": "2" * 64,
        "training_manifest_sha256": "3" * 64,
        "training_manifest_file_sha256": "4" * 64,
        "source_groups_artifact_sha256": "5" * 64,
        "source_groups_file_sha256": "6" * 64,
        "embedding_sources": [
            {
                "backbone": MVIT,
                "backbone_sha256": "7" * 64,
                "embedding_dimension": 768,
                "clip_frames": 16,
                "sampling_protocol": SHOT_SAMPLING_PROTOCOL,
            },
            {
                "backbone": SWIN,
                "backbone_sha256": "8" * 64,
                "embedding_dimension": 768,
                "clip_frames": 16,
                "sampling_protocol": SHOT_SAMPLING_PROTOCOL,
            },
        ],
        "representation_variants": [
            {
                "name": "swin3d_t",
                "backbones": [SWIN],
                "input_dimension": 768,
            },
            {
                "name": "mvit_v2_s+swin3d_t",
                "backbones": [MVIT, SWIN],
                "input_dimension": 1536,
            },
        ],
    }


def _canonical_sha256(payload: dict[str, object]) -> str:
    unsigned = copy.deepcopy(payload)
    unsigned.pop("artifact_sha256", None)
    return hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _reseal(payload: dict[str, object]) -> dict[str, object]:
    result = copy.deepcopy(payload)
    result.pop("artifact_sha256", None)
    result["artifact_sha256"] = _canonical_sha256(result)
    return result


def _reseal_result_with_export_index(
    payload: dict[str, object],
    *,
    export_file_sha256: str | None = None,
    export_size_bytes: int | None = None,
) -> dict[str, object]:
    result = copy.deepcopy(payload)
    index = result["training_export_index"]
    index["artifact_sha256"] = _canonical_sha256(index)
    encoded_index = encode_json_artifact(index)
    file_sha256 = export_file_sha256 or hashlib.sha256(encoded_index).hexdigest()
    size_bytes = export_size_bytes or len(encoded_index)
    export_receipt = result["training_export_receipt"]
    export_receipt["artifact_sha256"] = index["artifact_sha256"]
    export_receipt["file_sha256"] = file_sha256
    export_receipt["size_bytes"] = size_bytes
    manifest_receipt = result["training_manifest_receipt"]
    manifest = manifest_receipt["manifest"]
    manifest["annotation_files"][0] = {
        "filename": export_receipt["filename"],
        "sha256": file_sha256,
        "size_bytes": size_bytes,
    }
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    encoded_manifest = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    manifest_receipt["file_sha256"] = hashlib.sha256(encoded_manifest).hexdigest()
    plan = result["probe_plan"]
    plan["training_export_artifact_sha256"] = index["artifact_sha256"]
    plan["training_export_file_sha256"] = file_sha256
    plan["training_manifest_sha256"] = manifest["manifest_sha256"]
    plan["training_manifest_file_sha256"] = manifest_receipt["file_sha256"]
    result["probe_plan"] = _reseal(plan)
    return _reseal(result)


def _examples() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for game_index, (positive_count, negative_count) in enumerate(SOURCE_COUNTS):
        for label, count in ((True, positive_count), (False, negative_count)):
            for row_index in range(count):
                rows.append(
                    {
                        "source_video_sha256": SOURCE_SHAS[game_index],
                        "candidate_bundle_sha256": BUNDLE_SHAS[game_index],
                        "event_id": (f"{GAME_ORDER[game_index]}-{'shot' if label else 'nonshot'}-{row_index:02d}"),
                        "event_present": label,
                    }
                )
    game_positions = {source: index for index, source in enumerate(SOURCE_SHAS)}
    return sorted(
        rows,
        key=lambda row: (
            game_positions[str(row["source_video_sha256"])],
            str(row["event_id"]),
        ),
    )


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cli_paths(tmp_path: Path) -> dict[str, object]:
    paths: dict[str, object] = {
        "embeddings": [
            tmp_path / "mvit-embeddings.json",
            tmp_path / "swin-embeddings.json",
        ],
        "training_export": tmp_path / "training-export-v2.json",
        "extension_asset_root": tmp_path / "extension-assets",
        "parent_export": tmp_path / "parent-export.json",
        "parent_asset_root": tmp_path / "parent-assets",
        "harwood_selection": tmp_path / "harwood-selection.json",
        "harwood_review_plan": tmp_path / "harwood-review-plan.json",
        "harwood_sealed_review": tmp_path / "harwood-sealed-review.json",
        "harwood_raw_frame_manifest": tmp_path / "harwood-raw-frames.json",
        "harwood_source_manifest": tmp_path / "harwood-source.json",
        "source_groups": tmp_path / "source-groups.json",
        "training_manifest": tmp_path / "training-manifest.json",
        "plan": tmp_path / "probe-plan.json",
    }
    for directory_key in ("extension_asset_root", "parent_asset_root"):
        path = paths[directory_key]
        assert isinstance(path, Path)
        path.mkdir()
    for key, value in paths.items():
        if key in {"extension_asset_root", "parent_asset_root"}:
            continue
        path_values = value if isinstance(value, list) else [value]
        for path in path_values:
            assert isinstance(path, Path)
            path.write_text("{}\n", encoding="utf-8")
    return paths


def _cli_argv(
    paths: dict[str, object],
    *,
    output: Path,
    export_file_sha256: str,
) -> list[str]:
    sha = "1" * 64
    embeddings = paths["embeddings"]
    plan_path = paths["plan"]
    assert isinstance(embeddings, list)
    assert isinstance(plan_path, Path)
    plan_file_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    return [
        "screen_vru_causal_video_embeddings_nested_v2.py",
        "--embeddings",
        str(embeddings[0]),
        "--embeddings",
        str(embeddings[1]),
        "--expected-embedding-artifact-sha256",
        sha,
        "--expected-embedding-artifact-sha256",
        "2" * 64,
        "--expected-embedding-file-sha256",
        "3" * 64,
        "--expected-embedding-file-sha256",
        "4" * 64,
        "--training-export",
        str(paths["training_export"]),
        "--expected-training-export-artifact-sha256",
        "5" * 64,
        "--expected-training-export-file-sha256",
        export_file_sha256,
        "--extension-asset-root",
        str(paths["extension_asset_root"]),
        "--parent-export",
        str(paths["parent_export"]),
        "--expected-parent-artifact-sha256",
        "6" * 64,
        "--expected-parent-file-sha256",
        "7" * 64,
        "--parent-asset-root",
        str(paths["parent_asset_root"]),
        "--harwood-selection",
        str(paths["harwood_selection"]),
        "--expected-harwood-selection-artifact-sha256",
        "8" * 64,
        "--harwood-review-plan",
        str(paths["harwood_review_plan"]),
        "--expected-harwood-review-plan-artifact-sha256",
        "9" * 64,
        "--harwood-sealed-review",
        str(paths["harwood_sealed_review"]),
        "--expected-harwood-sealed-review-artifact-sha256",
        "a" * 64,
        "--harwood-raw-frame-manifest",
        str(paths["harwood_raw_frame_manifest"]),
        "--expected-harwood-raw-frame-manifest-artifact-sha256",
        "b" * 64,
        "--harwood-source-manifest",
        str(paths["harwood_source_manifest"]),
        "--source-groups",
        str(paths["source_groups"]),
        "--expected-source-groups-artifact-sha256",
        "c" * 64,
        "--expected-source-groups-file-sha256",
        "d" * 64,
        "--training-manifest",
        str(paths["training_manifest"]),
        "--expected-training-manifest-sha256",
        "e" * 64,
        "--expected-training-manifest-file-sha256",
        "f" * 64,
        "--plan",
        str(paths["plan"]),
        "--expected-plan-artifact-sha256",
        sha,
        "--expected-plan-file-sha256",
        plan_file_sha256,
        "--output",
        str(output),
    ]


def _public_screen_kwargs(
    paths: dict[str, object],
    *,
    plan: dict[str, object],
    plan_file_sha256: str,
) -> dict[str, object]:
    return {
        "embedding_paths": paths["embeddings"],
        "expected_embedding_artifact_sha256s": ["1" * 64, "2" * 64],
        "expected_embedding_file_sha256s": ["3" * 64, "4" * 64],
        "training_export_path": paths["training_export"],
        "expected_training_export_artifact_sha256": plan["training_export_artifact_sha256"],
        "expected_training_export_file_sha256": plan["training_export_file_sha256"],
        "extension_asset_root": paths["extension_asset_root"],
        "parent_export_path": paths["parent_export"],
        "expected_parent_artifact_sha256": "6" * 64,
        "expected_parent_file_sha256": "7" * 64,
        "parent_asset_root": paths["parent_asset_root"],
        "harwood_selection_path": paths["harwood_selection"],
        "expected_harwood_selection_artifact_sha256": "8" * 64,
        "harwood_review_plan_path": paths["harwood_review_plan"],
        "expected_harwood_review_plan_artifact_sha256": "9" * 64,
        "harwood_sealed_review_path": paths["harwood_sealed_review"],
        "expected_harwood_sealed_review_artifact_sha256": "a" * 64,
        "harwood_raw_frame_manifest_path": paths["harwood_raw_frame_manifest"],
        "expected_harwood_raw_frame_manifest_artifact_sha256": "b" * 64,
        "harwood_source_manifest_path": paths["harwood_source_manifest"],
        "source_groups_path": paths["source_groups"],
        "expected_source_groups_artifact_sha256": plan["source_groups_artifact_sha256"],
        "expected_source_groups_file_sha256": plan["source_groups_file_sha256"],
        "training_manifest_path": paths["training_manifest"],
        "expected_training_manifest_sha256": plan["training_manifest_sha256"],
        "expected_training_manifest_file_sha256": plan["training_manifest_file_sha256"],
        "probe_plan_path": paths["plan"],
        "expected_probe_plan_sha256": plan["artifact_sha256"],
        "expected_probe_plan_file_sha256": plan_file_sha256,
    }


def _source_groups() -> dict[str, object]:
    return seal_vru_causal_source_groups(
        {
            "schema_version": SOURCE_GROUP_SPEC_SCHEMA,
            "games": [
                {
                    "game_id": game_id,
                    "source_video_sha256": source_sha256,
                    "production_family": production_family,
                }
                for game_id, source_sha256, production_family in zip(
                    GAME_ORDER,
                    SOURCE_SHAS,
                    GAME_FAMILIES,
                    strict=True,
                )
            ],
        }
    )


def _manifest(
    *,
    export_filename: str,
    export_file_sha256: str,
    export_size_bytes: int = 123,
    label_assets: tuple[dict[str, object], ...] = LABEL_ASSETS,
    benchmark_raw_sha256: str = ACCEPTANCE_SHA256,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "agu.training-annotation.v1",
        "purpose": "model_training_only",
        "producer": "codex-assisted-offline-causal-review-v2",
        "runtime_consumable": False,
        "benchmark_overlap": False,
        "source_videos": [
            {
                "filename": f"{game_id}.webm",
                "sha256": source_sha256,
                "size_bytes": index + 1,
            }
            for index, (game_id, source_sha256) in enumerate(zip(GAME_ORDER, SOURCE_SHAS, strict=True))
        ],
        "annotation_files": [
            {
                "filename": export_filename,
                "sha256": export_file_sha256,
                "size_bytes": export_size_bytes,
            },
            *copy.deepcopy(label_assets),
        ],
        "task_types": ["shot_validity"],
        "benchmark_raw_sha256": [benchmark_raw_sha256],
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    return payload


def _training_chain(
    *,
    source_groups: dict[str, object],
    source_groups_file_sha256: str,
    rows: list[dict[str, object]],
) -> ContinuousCausalTrainingChain:
    index: dict[str, object] = {
        "schema_version": TRAINING_EXPORT_V2_SCHEMA,
        "purpose": EXPORT_PURPOSE,
        "runtime_consumable": False,
        "training_consumable": True,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
        "codex_runtime_answer_used": False,
        "parent": {
            "schema_version": "agu.vru-causal-shot-validity-training-export.v1",
            "artifact_sha256": FROZEN_PARENT_ARTIFACT_SHA256,
            "file_sha256": FROZEN_PARENT_FILE_SHA256,
            "sources": [
                {
                    "source_id": game_id,
                    "source_video_sha256": SOURCE_SHAS[position],
                    "candidate_bundle": {
                        "filename": f"{game_id}_candidate_bundle_v1.json",
                        "sha256": FROZEN_PARENT_CHILD_SHA256[f"{game_id}_candidate_bundle_v1.json"],
                        "size_bytes": 201 + position,
                        "bundle_sha256": BUNDLE_SHAS[position],
                    },
                    "labels": copy.deepcopy(LABEL_ASSETS[position]),
                }
                for position, game_id in enumerate(GAME_ORDER[:3])
            ],
        },
        "harwood_evidence": {
            "source_manifest": {
                "schema_version": "agu.vru-basketball-source-manifest.v1",
                "file_sha256": FROZEN_HARWOOD_FILE_SHA256["source_manifest"],
            },
            "source_video": {
                "filename": "hctv_harwood_2026.webm",
                "sha256": FROZEN_HARWOOD_VIDEO_SHA256,
                "size_bytes": FROZEN_HARWOOD_VIDEO_SIZE_BYTES,
            },
            "source_selection": {
                "schema_version": "agu.continuous-causal-review-selection.v1",
                "artifact_sha256": FROZEN_HARWOOD_ARTIFACT_SHA256["selection"],
                "file_sha256": FROZEN_HARWOOD_FILE_SHA256["selection"],
            },
            "source_review_plan": {
                "schema_version": "agu.vru-causal-review-plan.v1",
                "artifact_sha256": FROZEN_HARWOOD_ARTIFACT_SHA256["review_plan"],
                "file_sha256": FROZEN_HARWOOD_FILE_SHA256["review_plan"],
            },
            "source_frame_manifest": {
                "schema_version": "agu.vru-causal-review-frame-manifest.v1",
                "artifact_sha256": FROZEN_HARWOOD_ARTIFACT_SHA256["raw_frame_manifest"],
                "file_sha256": FROZEN_HARWOOD_FILE_SHA256["raw_frame_manifest"],
                "jpeg_count": 1536,
                "jpeg_set_sha256": "f" * 64,
            },
            "source_review": {
                "schema_version": "agu.vru-causal-offline-review.v1",
                "artifact_sha256": FROZEN_HARWOOD_ARTIFACT_SHA256["sealed_review"],
                "file_sha256": FROZEN_HARWOOD_FILE_SHA256["sealed_review"],
            },
        },
        "extension": {
            "source_id": "harwood",
            "game_id": "harwood",
            "source_video_sha256": FROZEN_HARWOOD_VIDEO_SHA256,
            "candidate_bundle": {
                "filename": HARWOOD_CANDIDATE_FILENAME,
                "sha256": "8" * 64,
                "size_bytes": 204,
                "bundle_sha256": BUNDLE_SHAS[3],
            },
            "labels": copy.deepcopy(LABEL_ASSETS[3]),
            "selected_count": 24,
            "example_count": 23,
            "positive_count": 3,
            "negative_count": 20,
            "excluded_uncertain": 1,
        },
        "source_groups": {
            "schema_version": source_groups["schema_version"],
            "artifact_sha256": source_groups["artifact_sha256"],
            "file_sha256": source_groups_file_sha256,
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
            "selected": 48,
            "exported": 45,
            "positive": 17,
            "negative": 28,
            "excluded_uncertain": 3,
            "game_groups": 4,
            "production_families": 2,
        },
        "excluded_review_ids": list(EXPECTED_EXCLUDED_REVIEW_IDS),
    }
    index["artifact_sha256"] = _canonical_sha256(index)
    index = verify_vru_causal_shot_validity_training_extension_index(index)
    return ContinuousCausalTrainingChain(
        index=index,
        extension_candidate_bundles={},
        extension_label_files={},
        source_groups=copy.deepcopy(source_groups),
        resolved_examples=tuple(rows),
        validated_input_paths=(),
    )


def _embedding_artifact(
    *,
    rows: list[dict[str, object]],
    backbone: str,
    backbone_sha256: str,
    manifest: dict[str, object],
) -> dict[str, object]:
    embedded = []
    for index, row in enumerate(rows):
        center = -2.0 if "-nonshot-" in str(row["event_id"]) else 2.0
        if backbone == SWIN:
            center *= 0.9
        embedded.append(
            {
                **copy.deepcopy(row),
                "sampling_window": {
                    "start_frame": index * 100,
                    "end_frame": index * 100 + 63,
                    "anchor_frame": index * 100 + 63,
                    "anchor_source": "bounded_event",
                    "protocol": SHOT_SAMPLING_PROTOCOL,
                },
                "embedding": [center + index * 0.0001] * 768,
            }
        )
    return seal_video_embedding_artifact(
        {
            "purpose": "backbone_screening_training_only",
            "producer": "agu",
            "training_manifest_sha256": manifest["manifest_sha256"],
            "training_annotation_sha256": sorted(row["sha256"] for row in manifest["annotation_files"][1:]),
            "source_video_sha256": sorted(SOURCE_SHAS),
            "backbone": backbone,
            "backbone_sha256": backbone_sha256,
            "backbone_license": "test-only fixture",
            "backbone_weights_url": "https://example.invalid/fixture",
            "embedding_dimension": 768,
            "clip_frames": 16,
            "sampling_protocol": SHOT_SAMPLING_PROTOCOL,
            "examples": embedded,
        }
    )


def _screen(
    tmp_path: Path,
    *,
    examples: list[dict[str, object]] | None = None,
    embedding_drift: str | None = None,
    external_receipt_drift: str | None = None,
    manifest_drift: str | None = None,
) -> dict[str, object]:
    rows = copy.deepcopy(examples if examples is not None else _examples())
    source_groups = _source_groups()
    source_groups_path = tmp_path / "source-groups.json"
    source_groups_file_sha256 = _write_json(source_groups_path, source_groups)
    chain = _training_chain(
        source_groups=source_groups,
        source_groups_file_sha256=source_groups_file_sha256,
        rows=rows,
    )
    export_artifact_sha256 = str(chain.index["artifact_sha256"])
    export_bytes = encode_json_artifact(chain.index)
    export_file_sha256 = hashlib.sha256(export_bytes).hexdigest()
    export_size_bytes = len(export_bytes)
    manifest = _manifest(
        export_filename="training-export-v2.json",
        export_file_sha256=export_file_sha256,
        export_size_bytes=export_size_bytes,
    )
    if manifest_drift == "arbitrary_labels":
        manifest["annotation_files"][1:] = [
            {
                "filename": f"attacker-{index}.json",
                "sha256": str(index + 5) * 64,
                "size_bytes": index + 1,
            }
            for index in range(4)
        ]
    elif manifest_drift == "benchmark":
        manifest["benchmark_raw_sha256"] = ["0" * 64]
    elif manifest_drift == "label_order":
        manifest["annotation_files"][1:3] = reversed(manifest["annotation_files"][1:3])
    elif manifest_drift == "producer":
        manifest["producer"] = "codex-assisted-offline-causal-review"
    elif manifest_drift is not None:  # pragma: no cover - test-helper contract
        raise AssertionError(manifest_drift)
    if manifest_drift is not None:
        manifest.pop("manifest_sha256")
        manifest["manifest_sha256"] = _canonical_sha256(manifest)
    manifest_path = tmp_path / "training-manifest.json"
    manifest_file_sha256 = _write_json(manifest_path, manifest)
    spec = _plan_spec()
    spec.update(
        {
            "training_export_artifact_sha256": export_artifact_sha256,
            "training_export_file_sha256": export_file_sha256,
            "training_manifest_sha256": manifest["manifest_sha256"],
            "training_manifest_file_sha256": manifest_file_sha256,
            "source_groups_artifact_sha256": source_groups["artifact_sha256"],
            "source_groups_file_sha256": source_groups_file_sha256,
        }
    )
    plan = seal_vru_causal_video_probe_plan_v2(spec)
    plan_path = tmp_path / "probe-plan.json"
    plan_file_sha256 = _write_json(plan_path, plan)
    embedding_paths = []
    embedding_artifact_sha256s = []
    embedding_file_sha256s = []
    for index, source in enumerate(plan["embedding_sources"]):
        artifact = _embedding_artifact(
            rows=rows,
            backbone=source["backbone"],
            backbone_sha256=source["backbone_sha256"],
            manifest=manifest,
        )
        if index == 0 and embedding_drift is not None:
            if embedding_drift == "label":
                artifact["examples"][0]["event_present"] = not artifact["examples"][0]["event_present"]
            elif embedding_drift == "checkpoint":
                artifact["backbone_sha256"] = "0" * 64
            elif embedding_drift == "sampling_anchor":
                artifact["examples"][0]["sampling_window"]["anchor_source"] = "release_frame"
            elif embedding_drift == "duplicate_key":
                artifact["examples"][1]["event_id"] = artifact["examples"][0]["event_id"]
            else:  # pragma: no cover - test-helper contract
                raise AssertionError(embedding_drift)
            artifact = seal_video_embedding_artifact(artifact)
        path = tmp_path / f"embedding-{index}.json"
        embedding_paths.append(path)
        embedding_artifact_sha256s.append(artifact["artifact_sha256"])
        embedding_file_sha256s.append(_write_json(path, artifact))
    return probe_module._screen_verified_chain(
        embedding_paths=embedding_paths,
        training_chain=chain,
        training_export_asset={
            "filename": "training-export-v2.json",
            "sha256": export_file_sha256,
            "size_bytes": export_size_bytes,
        },
        source_groups_path=source_groups_path,
        expected_source_groups_artifact_sha256=str(source_groups["artifact_sha256"]),
        expected_source_groups_file_sha256=(
            "0" * 64 if external_receipt_drift == "source_groups_file" else source_groups_file_sha256
        ),
        training_manifest_path=manifest_path,
        expected_training_manifest_sha256=str(manifest["manifest_sha256"]),
        expected_training_manifest_file_sha256=(
            "0" * 64 if external_receipt_drift == "manifest_file" else manifest_file_sha256
        ),
        probe_plan=plan,
        probe_plan_file_sha256=plan_file_sha256,
        expected_embedding_artifact_sha256s=embedding_artifact_sha256s,
        expected_embedding_file_sha256s=(
            ["0" * 64, *embedding_file_sha256s[1:]]
            if external_receipt_drift == "embedding_file"
            else embedding_file_sha256s
        ),
    )


@pytest.fixture(scope="module")
def sealed_probe_result(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    return _screen(tmp_path_factory.mktemp("sealed-probe-v2"))


def test_v2_probe_public_interfaces_are_additive() -> None:
    assert callable(screen_vru_causal_video_embeddings_nested_v2)
    assert callable(verify_vru_causal_video_probe_v2)
    parameters = inspect.signature(screen_vru_causal_video_embeddings_nested_v2).parameters
    assert "training_chain" not in parameters
    assert "training_export_path" in parameters
    assert "expected_probe_plan_file_sha256" in parameters


def test_public_probe_snapshots_plan_before_replaying_the_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _cli_paths(tmp_path)
    export_path = paths["training_export"]
    plan_path = paths["plan"]
    assert isinstance(export_path, Path)
    assert isinstance(plan_path, Path)
    export_payload = {"fixture": "verified by the mocked upstream boundary"}
    export_file_sha256 = _write_json(export_path, export_payload)
    spec = _plan_spec()
    spec["training_export_file_sha256"] = export_file_sha256
    plan = seal_vru_causal_video_probe_plan_v2(spec)
    plan_file_sha256 = _write_json(plan_path, plan)
    chain = _training_chain(
        source_groups=_source_groups(),
        source_groups_file_sha256=str(plan["source_groups_file_sha256"]),
        rows=_examples(),
    )
    reads: list[Path] = []
    replay_calls: list[tuple[dict[str, object], dict[str, object]]] = []
    screen_calls: list[dict[str, object]] = []
    real_reader = probe_module._read_json_snapshot

    def tracking_reader(path: Path) -> tuple[dict[str, object], str, int]:
        reads.append(path)
        return real_reader(path)

    def fake_replay(
        payload: dict[str, object],
        **kwargs: object,
    ) -> ContinuousCausalTrainingChain:
        replay_calls.append((payload, kwargs))
        return chain

    sentinel = {"verified": True}

    def fake_screen(**kwargs: object) -> dict[str, object]:
        screen_calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(probe_module, "_read_json_snapshot", tracking_reader)
    monkeypatch.setattr(
        probe_module,
        "verify_vru_causal_shot_validity_training_extension_export",
        fake_replay,
    )
    monkeypatch.setattr(probe_module, "_screen_verified_chain", fake_screen)

    result = screen_vru_causal_video_embeddings_nested_v2(
        **_public_screen_kwargs(
            paths,
            plan=plan,
            plan_file_sha256=plan_file_sha256,
        )
    )

    assert result is sentinel
    assert reads == [plan_path, export_path]
    assert replay_calls[0][0] == export_payload
    assert screen_calls[0]["training_chain"] is chain
    assert screen_calls[0]["probe_plan"] == plan
    assert screen_calls[0]["probe_plan_file_sha256"] == plan_file_sha256
    assert screen_calls[0]["training_export_asset"] == {
        "filename": export_path.name,
        "sha256": export_file_sha256,
        "size_bytes": len(export_path.read_bytes()),
    }


def test_public_probe_rejects_a_self_resealed_bad_plan_before_export_or_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _cli_paths(tmp_path)
    plan = seal_vru_causal_video_probe_plan_v2(_plan_spec())
    plan["runtime_consumable"] = True
    plan = _reseal(plan)
    plan_path = paths["plan"]
    assert isinstance(plan_path, Path)
    plan_file_sha256 = _write_json(plan_path, plan)
    reads: list[Path] = []
    real_reader = probe_module._read_json_snapshot

    def plan_only_reader(path: Path) -> tuple[dict[str, object], str, int]:
        reads.append(path)
        if path != plan_path:
            raise AssertionError("bad plan reached export, label, or review reads")
        return real_reader(path)

    def reject_upstream(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("bad plan reached upstream chain replay")

    monkeypatch.setattr(probe_module, "_read_json_snapshot", plan_only_reader)
    monkeypatch.setattr(
        probe_module,
        "verify_vru_causal_shot_validity_training_extension_export",
        reject_upstream,
        raising=False,
    )

    with pytest.raises(ValueError, match="runtime_consumable.*false"):
        screen_vru_causal_video_embeddings_nested_v2(
            **_public_screen_kwargs(
                paths,
                plan=plan,
                plan_file_sha256=plan_file_sha256,
            )
        )
    assert reads == [plan_path]


def test_probe_plan_seals_the_exact_nonpromotable_nested_contract() -> None:
    plan = seal_vru_causal_video_probe_plan_v2(_plan_spec())

    assert plan["schema_version"] == VIDEO_PROBE_PLAN_V2_SCHEMA
    assert plan["purpose"] == "development_diagnostic_only"
    assert plan["runtime_consumable"] is False
    assert plan["formal_evaluation_eligible"] is False
    assert plan["promotion_eligible"] is False
    assert plan["promoted"] is False
    assert plan["regularization_c"] == 0.01
    assert plan["preprocessing"] == {
        "normalization": "standard_scaler_fit_split_train_only",
        "pca_enabled": False,
        "pca_components": None,
    }
    assert plan["threshold_selection"]["threshold_grid"] == [index / 20 for index in range(21)]
    assert (
        verify_vru_causal_video_probe_plan_v2(
            plan,
            expected_artifact_sha256=str(plan["artifact_sha256"]),
        )
        == plan
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("runtime_consumable", True),
        ("formal_evaluation_eligible", True),
        ("promotion_eligible", True),
        ("promoted", True),
        ("regularization_c", 1.0),
    ),
)
def test_probe_plan_rejects_resealed_policy_drift(
    field: str,
    value: object,
) -> None:
    plan = seal_vru_causal_video_probe_plan_v2(_plan_spec())
    plan[field] = value
    plan = _reseal(plan)

    with pytest.raises(ValueError, match="false|regularization|policy"):
        verify_vru_causal_video_probe_plan_v2(plan)


def test_probe_plan_rejects_resealed_variant_order_or_dimension_drift() -> None:
    plan = seal_vru_causal_video_probe_plan_v2(_plan_spec())
    plan["representation_variants"][0]["input_dimension"] = 1536
    plan = _reseal(plan)

    with pytest.raises(ValueError, match="variant|dimension"):
        verify_vru_causal_video_probe_plan_v2(plan)


@pytest.mark.parametrize(
    "drift",
    ("float_dimension", "boolean_threshold", "integer_pca_flag"),
)
def test_probe_plan_rejects_json_type_drift_even_when_values_compare_equal(
    drift: str,
) -> None:
    plan = seal_vru_causal_video_probe_plan_v2(_plan_spec())
    if drift == "float_dimension":
        plan["embedding_sources"][0]["embedding_dimension"] = 768.0
    elif drift == "boolean_threshold":
        plan["threshold_selection"]["threshold_grid"][-1] = True
    else:
        plan["preprocessing"]["pca_enabled"] = 0
    plan = _reseal(plan)

    with pytest.raises(ValueError, match="dimension|threshold|preprocessing|type"):
        verify_vru_causal_video_probe_plan_v2(plan)


def test_four_game_probe_is_nested_receipt_bound_and_nonpromotable(
    tmp_path: Path,
) -> None:
    result = _screen(tmp_path)

    assert result["schema_version"] == "agu.vru-causal-video-representation-probe.v2"
    assert result["runtime_consumable"] is False
    assert result["formal_evaluation_eligible"] is False
    assert result["promotion_eligible"] is False
    assert result["promoted"] is False
    assert result["row_count"] == 45
    assert result["positive_count"] == 17
    assert result["negative_count"] == 28
    assert result["game_group_count"] == 4
    assert result["production_family_count"] == 2
    assert result["production_family_disjoint_all"] is False
    assert [fold["held_game_id"] for fold in result["outer_folds"]] == list(GAME_ORDER)
    assert [fold["production_family_disjoint"] for fold in result["outer_folds"]] == [False, False, True, False]
    for fold in result["outer_folds"]:
        assert len(fold["fit_game_ids"]) == 3
        assert len(fold["threshold_selection"]["inner_folds"]) == 3
        assert fold["held_game_id"] not in fold["fit_game_ids"]
        assert fold["selected_settings"]["regularization_c"] == 0.01
        assert fold["selected_settings"]["pca_enabled"] is False
    assert result["observed_metrics"]["pooled"]["support"] == 45
    assert [row["production_family"] for row in result["observed_metrics"]["per_production_family"]] == ["hctv", "vtv"]
    assert all(
        row["descriptive_only"] is True and row["production_held_generalization"] is False
        for row in result["observed_metrics"]["per_production_family"]
    )
    assert verify_vru_causal_video_probe_v2(result) == result


def test_probe_v2_is_deterministic_and_accepts_an_independent_result_receipt(
    tmp_path: Path,
) -> None:
    first = _screen(tmp_path / "first")
    second = _screen(tmp_path / "second")

    assert first == second
    assert (
        verify_vru_causal_video_probe_v2(
            first,
            expected_artifact_sha256=str(first["artifact_sha256"]),
        )
        == first
    )
    with pytest.raises(ValueError, match="expected artifact"):
        verify_vru_causal_video_probe_v2(
            first,
            expected_artifact_sha256="0" * 64,
        )


def test_outer_held_labels_cannot_change_that_fold_selection_or_probabilities(
    tmp_path: Path,
) -> None:
    original = _screen(tmp_path / "original")
    rows = _examples()
    held = [row for row in rows if row["source_video_sha256"] == SOURCE_SHAS[0]]
    positive = next(row for row in held if row["event_present"] is True)
    negative = next(row for row in held if row["event_present"] is False)
    positive["event_present"] = False
    negative["event_present"] = True
    relabeled = _screen(tmp_path / "relabeled", examples=rows)

    original_fold = original["outer_folds"][0]
    relabeled_fold = relabeled["outer_folds"][0]
    assert original_fold["selected_settings"] == relabeled_fold["selected_settings"]
    assert original_fold["threshold_selection"] == relabeled_fold["threshold_selection"]
    assert [
        (row["source_video_sha256"], row["event_id"], row["probability"], row["decision"])
        for row in original_fold["oof_predictions"]
    ] == [
        (row["source_video_sha256"], row["event_id"], row["probability"], row["decision"])
        for row in relabeled_fold["oof_predictions"]
    ]
    assert original_fold["held_metrics"] != relabeled_fold["held_metrics"]


@pytest.mark.parametrize(
    "field",
    (
        "runtime_consumable",
        "formal_evaluation_eligible",
        "promotion_eligible",
        "promoted",
    ),
)
def test_probe_verifier_rejects_resealed_eligibility_drift(
    sealed_probe_result: dict[str, object],
    field: str,
) -> None:
    tampered = copy.deepcopy(sealed_probe_result)
    tampered[field] = True
    tampered = _reseal(tampered)

    with pytest.raises(ValueError, match="must be false"):
        verify_vru_causal_video_probe_v2(tampered)


def test_probe_verifier_recomputes_inner_candidate_metrics_from_predictions(
    sealed_probe_result: dict[str, object],
) -> None:
    tampered = copy.deepcopy(sealed_probe_result)
    candidate = tampered["outer_folds"][0]["threshold_selection"]["candidate_results"][0]
    candidate["inner_oof_metrics"]["tp"] += 1
    candidate["inner_oof_metrics"]["support"] += 1
    tampered = _reseal(tampered)

    with pytest.raises(ValueError, match="candidate metrics|arithmetic"):
        verify_vru_causal_video_probe_v2(tampered)


def test_probe_verifier_rejects_resealed_outer_key_or_decision_drift(
    sealed_probe_result: dict[str, object],
) -> None:
    tampered = copy.deepcopy(sealed_probe_result)
    prediction = tampered["outer_folds"][0]["oof_predictions"][0]
    prediction["decision"] = not prediction["decision"]
    tampered = _reseal(tampered)

    with pytest.raises(ValueError, match="decision.*threshold"):
        verify_vru_causal_video_probe_v2(tampered)


def test_probe_verifier_rejects_resealed_duplicate_embedding_file_receipt(
    sealed_probe_result: dict[str, object],
) -> None:
    tampered = copy.deepcopy(sealed_probe_result)
    receipts = tampered["video_embedding_artifacts"]
    receipts[1]["file_sha256"] = receipts[0]["file_sha256"]
    tampered = _reseal(tampered)

    with pytest.raises(ValueError, match="embedding receipts.*unique"):
        verify_vru_causal_video_probe_v2(tampered)


def test_probe_verifier_rejects_unknown_field_even_when_resealed(
    sealed_probe_result: dict[str, object],
) -> None:
    tampered = copy.deepcopy(sealed_probe_result)
    tampered["unexpected"] = True
    tampered = _reseal(tampered)

    with pytest.raises(ValueError, match="fields.*canonical"):
        verify_vru_causal_video_probe_v2(tampered)


def test_probe_verifier_rejects_resealed_harwood_label_lineage_drift(
    sealed_probe_result: dict[str, object],
) -> None:
    tampered = copy.deepcopy(sealed_probe_result)
    manifest_receipt = tampered["training_manifest_receipt"]
    manifest = manifest_receipt["manifest"]
    manifest["annotation_files"][4]["sha256"] = "0" * 64
    manifest["annotation_files"][4]["size_bytes"] += 1
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    manifest_receipt["file_sha256"] = "e" * 64
    plan = tampered["probe_plan"]
    plan["training_manifest_sha256"] = manifest["manifest_sha256"]
    plan["training_manifest_file_sha256"] = manifest_receipt["file_sha256"]
    tampered["probe_plan"] = _reseal(plan)
    tampered = _reseal(tampered)

    with pytest.raises(ValueError, match="Harwood|label lineage|export index"):
        verify_vru_causal_video_probe_v2(tampered)


@pytest.mark.parametrize("drift", ("promotion_flags", "unknown_field"))
def test_probe_verifier_replays_exact_embedded_export_index(
    sealed_probe_result: dict[str, object],
    drift: str,
) -> None:
    tampered = copy.deepcopy(sealed_probe_result)
    index = tampered["training_export_index"]
    if drift == "promotion_flags":
        for field in (
            "runtime_consumable",
            "formal_evaluation_eligible",
            "promotion_eligible",
            "promoted",
        ):
            index[field] = True
    else:
        index["unexpected"] = True
    tampered = _reseal_result_with_export_index(tampered)

    with pytest.raises(ValueError, match="fields|consumption|flags|runtime"):
        verify_vru_causal_video_probe_v2(tampered)


def test_probe_verifier_binds_export_index_to_canonical_file_receipt(
    sealed_probe_result: dict[str, object],
) -> None:
    original_size = sealed_probe_result["training_export_receipt"]["size_bytes"]
    tampered = _reseal_result_with_export_index(
        sealed_probe_result,
        export_file_sha256="e" * 64,
        export_size_bytes=original_size + 1,
    )

    with pytest.raises(ValueError, match="index bytes|file receipt"):
        verify_vru_causal_video_probe_v2(tampered)


def test_probe_verifier_joins_export_index_sources_to_source_groups(
    sealed_probe_result: dict[str, object],
) -> None:
    tampered = copy.deepcopy(sealed_probe_result)
    tampered["training_export_index"]["parent"]["sources"][0]["source_video_sha256"] = "e" * 64
    tampered = _reseal_result_with_export_index(tampered)

    with pytest.raises(ValueError, match="index.*source groups"):
        verify_vru_causal_video_probe_v2(tampered)


def test_probe_verifier_binds_outer_and_inner_keys_to_export_bundles(
    sealed_probe_result: dict[str, object],
) -> None:
    tampered = copy.deepcopy(sealed_probe_result)
    prediction = tampered["outer_folds"][0]["oof_predictions"][0]
    source_sha256 = prediction["source_video_sha256"]
    event_id = prediction["event_id"]
    previous_bundle = prediction["candidate_bundle_sha256"]
    replacement_bundle = "e" * 64
    prediction["candidate_bundle_sha256"] = replacement_bundle
    inner_replacements = 0
    for fold in tampered["outer_folds"]:
        variants = fold["threshold_selection"]["variant_inner_oof_predictions"]
        for variant in variants:
            for row in variant["predictions"]:
                if (
                    row["source_video_sha256"] == source_sha256
                    and row["event_id"] == event_id
                    and row["candidate_bundle_sha256"] == previous_bundle
                ):
                    row["candidate_bundle_sha256"] = replacement_bundle
                    inner_replacements += 1
    assert inner_replacements == 6
    tampered = _reseal(tampered)

    with pytest.raises(ValueError, match="export.*bundle|held-game"):
        verify_vru_causal_video_probe_v2(tampered)


@pytest.mark.parametrize(
    "drift",
    (
        "source_count",
        "pca_flag",
        "candidate_threshold",
        "inner_count",
        "probability_int",
        "metric_rate_int",
    ),
)
def test_probe_verifier_rejects_resealed_json_type_drift(
    sealed_probe_result: dict[str, object],
    drift: str,
) -> None:
    tampered = copy.deepcopy(sealed_probe_result)
    if drift == "source_count":
        tampered["source_selection"]["parent_prior_stratified_rows"] = 22.0
    elif drift == "pca_flag":
        tampered["outer_folds"][0]["selected_settings"]["pca_enabled"] = 0
    elif drift == "candidate_threshold":
        tampered["outer_folds"][0]["threshold_selection"]["candidate_results"][20]["threshold"] = True
    elif drift == "inner_count":
        tampered["outer_folds"][0]["threshold_selection"]["inner_folds"][0]["row_count"] = 8.0
    elif drift == "probability_int":
        prediction = tampered["outer_folds"][0]["oof_predictions"][0]
        prediction["probability"] = 0
        prediction["decision"] = False
    else:
        tampered["outer_folds"][0]["held_metrics"]["precision"] = 0
    tampered = _reseal(tampered)

    with pytest.raises(
        ValueError,
        match="source|settings|candidate|inner|probability|metric|type|float",
    ):
        verify_vru_causal_video_probe_v2(tampered)


@pytest.mark.parametrize(
    ("drift", "message"),
    (
        ("source_groups_file", "source-group file"),
        ("manifest_file", "training manifest file"),
        ("embedding_file", "embedding file"),
    ),
)
def test_probe_v2_rejects_wrong_external_file_receipts(
    tmp_path: Path,
    drift: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _screen(tmp_path, external_receipt_drift=drift)


@pytest.mark.parametrize(
    ("drift", "message"),
    (
        ("label", "labels.*training chain"),
        ("checkpoint", "checkpoint.*plan"),
        ("sampling_anchor", "sampling.*label-hidden"),
        ("duplicate_key", "duplicate three-part keys"),
    ),
)
def test_probe_v2_rejects_resealed_embedding_semantic_drift(
    tmp_path: Path,
    drift: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _screen(tmp_path, embedding_drift=drift)


@pytest.mark.parametrize(
    ("drift", "message"),
    (
        ("arbitrary_labels", "annotation.*exact|label.*lineage"),
        ("benchmark", "acceptance benchmark|benchmark"),
        ("label_order", "annotation.*order|label.*lineage"),
        ("producer", "acceptance benchmark|producer|contract"),
    ),
)
def test_probe_v2_rejects_resealed_manifest_lineage_or_benchmark_drift(
    tmp_path: Path,
    drift: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _screen(tmp_path, manifest_drift=drift)


@pytest.mark.parametrize(
    "alias_kind",
    (
        "direct",
        "output_symlink",
        "input_symlink",
        "hardlink",
        "casefold",
        "descendant",
        "ancestor",
    ),
)
def test_probe_v2_cli_rejects_output_aliases_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_kind: str,
) -> None:
    paths = _cli_paths(tmp_path)
    embeddings = paths["embeddings"]
    assert isinstance(embeddings, list)
    protected = embeddings[0]
    assert isinstance(protected, Path)
    output = protected
    if alias_kind == "output_symlink":
        output = tmp_path / "output-alias.json"
        output.symlink_to(protected)
    elif alias_kind == "input_symlink":
        physical = tmp_path / "physical-embedding.json"
        protected.replace(physical)
        protected.symlink_to(physical)
        output = physical
    elif alias_kind == "hardlink":
        output = tmp_path / "output-hardlink.json"
        os.link(protected, output)
    elif alias_kind == "casefold":
        output = protected.with_name(protected.name.swapcase())
    elif alias_kind == "descendant":
        extension_root = paths["extension_asset_root"]
        assert isinstance(extension_root, Path)
        output = extension_root / "probe.json"
    elif alias_kind == "ancestor":
        output = tmp_path / "future-output"
        nested_input = output / "embedding.json"
        nested_input.parent.mkdir()
        nested_input.write_text("{}\n", encoding="utf-8")
        embeddings[0] = nested_input
    monkeypatch.setattr(
        sys,
        "argv",
        _cli_argv(paths, output=output, export_file_sha256="0" * 64),
    )

    def reject_read(_path: Path) -> tuple[dict[str, object], str]:
        raise AssertionError("aliases must fail before any input is read")

    monkeypatch.setattr(probe_cli, "_read_json_snapshot", reject_read)

    with pytest.raises(ValueError, match="regular file|alias|nest"):
        probe_cli.main()


def test_probe_v2_cli_passes_all_paths_and_receipts_to_public_screen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _cli_paths(tmp_path)
    training_export = paths["training_export"]
    assert isinstance(training_export, Path)
    export_file_sha256 = hashlib.sha256(training_export.read_bytes()).hexdigest()
    output = tmp_path / "results" / "probe.json"
    monkeypatch.setattr(
        sys,
        "argv",
        _cli_argv(
            paths,
            output=output,
            export_file_sha256=export_file_sha256,
        ),
    )
    screen_calls: list[dict[str, object]] = []

    def fake_screen(**kwargs: object) -> dict[str, object]:
        screen_calls.append(kwargs)
        return {
            "artifact_sha256": "9" * 64,
            "formal_evaluation_eligible": False,
            "game_group_count": 4,
            "production_family_count": 2,
            "promoted": False,
            "promotion_eligible": False,
            "row_count": 45,
            "runtime_consumable": False,
        }

    monkeypatch.setattr(
        probe_cli,
        "screen_vru_causal_video_embeddings_nested_v2",
        fake_screen,
    )

    assert probe_cli.main() == 0
    assert len(screen_calls) == 1
    assert screen_calls[0]["training_export_path"] == training_export.resolve()
    assert screen_calls[0]["expected_training_export_file_sha256"] == export_file_sha256
    assert screen_calls[0]["probe_plan_path"] == Path(paths["plan"]).resolve()
    assert (
        screen_calls[0]["expected_probe_plan_file_sha256"]
        == hashlib.sha256(Path(paths["plan"]).read_bytes()).hexdigest()
    )
    assert screen_calls[0]["parent_export_path"] == Path(paths["parent_export"]).resolve()
    assert screen_calls[0]["harwood_sealed_review_path"] == Path(paths["harwood_sealed_review"]).resolve()
    assert screen_calls[0]["embedding_paths"] == [path.resolve() for path in paths["embeddings"]]
    assert json.loads(output.read_text(encoding="utf-8"))["row_count"] == 45


def test_probe_v2_cli_rejects_malformed_receipts_before_input_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _cli_paths(tmp_path)
    argv = _cli_argv(
        paths,
        output=tmp_path / "probe.json",
        export_file_sha256="0" * 64,
    )
    receipt_position = argv.index("--expected-plan-artifact-sha256") + 1
    argv[receipt_position] = "NOT-A-SHA"
    monkeypatch.setattr(sys, "argv", argv)

    def reject_read(_path: Path) -> tuple[dict[str, object], str]:
        raise AssertionError("malformed receipts must fail before reads")

    monkeypatch.setattr(probe_cli, "_read_json_snapshot", reject_read)

    with pytest.raises(ValueError, match="SHA-256"):
        probe_cli.main()


def test_probe_v2_atomic_writer_fsyncs_file_then_replace_then_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "probe.json"
    destination.write_text("old artifact\n", encoding="utf-8")
    real_replace = os.replace
    events: list[str] = []

    def observe_fsync(file_descriptor: int) -> None:
        assert file_descriptor >= 0
        events.append("fsync")

    def observe_replace(source: Path, target: Path) -> None:
        assert source.parent == destination.parent
        assert source.name.startswith(f".{destination.name}.")
        assert source.suffix == ".tmp"
        assert destination.read_text(encoding="utf-8") == "old artifact\n"
        events.append("replace")
        real_replace(source, target)

    monkeypatch.setattr(probe_cli.os, "fsync", observe_fsync)
    monkeypatch.setattr(probe_cli.os, "replace", observe_replace)

    probe_cli._write_json_atomic(destination, {"new": True})

    assert events == ["fsync", "replace", "fsync"]
    assert json.loads(destination.read_text(encoding="utf-8")) == {"new": True}
    assert not list(tmp_path.glob(".probe.json.*.tmp"))


def test_probe_v2_atomic_writer_preserves_old_file_and_cleans_temp_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "probe.json"
    destination.write_text("old artifact\n", encoding="utf-8")

    def reject_replace(_source: Path, _target: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(probe_cli.os, "replace", reject_replace)

    with pytest.raises(OSError, match="injected"):
        probe_cli._write_json_atomic(destination, {"new": True})

    assert destination.read_text(encoding="utf-8") == "old artifact\n"
    assert not list(tmp_path.glob(".probe.json.*.tmp"))


@pytest.mark.parametrize(
    "reader",
    (probe_module._read_json_snapshot, probe_cli._read_json_snapshot),
)
def test_probe_v2_json_readers_reject_duplicate_object_keys(
    tmp_path: Path,
    reader: object,
) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"receipt":"first","receipt":"second"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON.*key"):
        reader(path)


@pytest.mark.parametrize(
    ("reader", "limit"),
    (
        (probe_module._read_json_snapshot, probe_module._MAX_JSON_BYTES),
        (probe_cli._read_json_snapshot, probe_cli._MAX_EXPORT_JSON_BYTES),
    ),
)
def test_probe_v2_json_readers_bound_their_single_read_before_allocating(
    reader: object,
    limit: int,
) -> None:
    requested_sizes: list[int] = []

    class FakeStream:
        def __enter__(self) -> FakeStream:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, size: int) -> bytes:
            requested_sizes.append(size)
            return b"{}"

    class FakePath:
        name = "bounded.json"

        def open(self, mode: str) -> FakeStream:
            assert mode == "rb"
            return FakeStream()

    payload, file_sha256, size_bytes = reader(FakePath())

    assert payload == {}
    assert file_sha256 == hashlib.sha256(b"{}").hexdigest()
    assert size_bytes == 2
    assert requested_sizes == [limit + 1]


@pytest.mark.parametrize(
    "reader",
    (probe_module._read_json_snapshot, probe_cli._read_json_snapshot),
)
@pytest.mark.parametrize("token", ("NaN", "Infinity", "-Infinity", "1e999"))
def test_probe_v2_json_readers_reject_every_nonfinite_number(
    tmp_path: Path,
    reader: object,
    token: str,
) -> None:
    path = tmp_path / "nonfinite.json"
    path.write_text(f'{{"value":{token}}}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite|finite JSON"):
        reader(path)
