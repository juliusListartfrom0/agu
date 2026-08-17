"""Four-game nested diagnostics for additive causal video training evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from app.analysis.shot_validity_sampling_window import SHOT_SAMPLING_PROTOCOL
from app.analysis.shot_validity_video_backbone import (
    verify_video_embedding_artifact,
)
from app.analysis.training_annotation import verify_training_annotation_manifest
from app.analysis.vru_causal_source_groups import (
    SOURCE_GROUP_SCHEMA,
    verify_vru_causal_source_groups,
)
from app.analysis.vru_causal_training import encode_json_artifact
from app.analysis.vru_causal_training_v2 import (
    FROZEN_PARENT_CHILD_SHA256,
    HARWOOD_LABEL_FILENAME,
    ContinuousCausalTrainingChain,
    verify_vru_causal_shot_validity_training_extension_export,
    verify_vru_causal_shot_validity_training_extension_index,
)

VIDEO_PROBE_V2_SCHEMA = "agu.vru-causal-video-representation-probe.v2"
VIDEO_PROBE_PLAN_V2_SCHEMA = "agu.vru-causal-video-probe-plan.v2"
VIDEO_PROBE_PURPOSE = "development_diagnostic_only"
SELECTION_PROTOCOL = "nested_outer_game_held_inner_logo_v2"
THRESHOLD_PROTOCOL = "inner_logo_frozen_grid_v2"
SELECTION_METRIC = "balanced_accuracy_then_f1_then_precision_then_recall"
SELECTION_TIE_BREAK = "representation_variant_order_then_threshold_desc"
REGULARIZATION_C = 0.01
THRESHOLD_GRID = tuple(index / 20 for index in range(21))
MVIT_BACKBONE = "torchvision/mvit_v2_s/kinetics400_v1"
SWIN_BACKBONE = "torchvision/swin3d_t/kinetics400_v1"
EXPECTED_BACKBONES = (MVIT_BACKBONE, SWIN_BACKBONE)
EXPECTED_VARIANTS = (
    ("swin3d_t", (SWIN_BACKBONE,), 768),
    ("mvit_v2_s+swin3d_t", (MVIT_BACKBONE, SWIN_BACKBONE), 1536),
)
EXPECTED_GAME_ORDER = ("hazen", "randolph", "vtv", "harwood")
EXPECTED_GAME_COUNTS = {
    "hazen": (6, 2),
    "randolph": (4, 3),
    "vtv": (4, 3),
    "harwood": (3, 20),
}
EXPECTED_ROW_COUNT = 45
EXPECTED_POSITIVE_COUNT = 17
EXPECTED_NEGATIVE_COUNT = 28
_MAX_JSON_BYTES = 64 * 1024 * 1024
ACCEPTANCE_BENCHMARK_RAW_SHA256 = "9651a64355839780dffac2d404fa4715a08668b192a254cf87cbaba2d70c7f2e"
TRAINING_MANIFEST_PRODUCER = "codex-assisted-offline-causal-review-v2"
EXPECTED_LABEL_FILENAMES = (
    "hazen_shot_validity_labels_v1.json",
    "randolph_shot_validity_labels_v1.json",
    "vtv_shot_validity_labels_v1.json",
    HARWOOD_LABEL_FILENAME,
)
SOURCE_SELECTION_LIMITATION = (
    "The resolved view combines 22 parent rows selected with prior-label strata "
    "and 23 Harwood rows from a geometry-only label-hidden selection. Both are "
    "development evidence across four games but only two production families, "
    "not continuous exhaustive truth, a formal generalization estimate, or "
    "evidence for the 85% acceptance gate."
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_EVENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,255}")
_PLAN_SPEC_FIELDS = {
    "embedding_sources",
    "representation_variants",
    "source_groups_artifact_sha256",
    "source_groups_file_sha256",
    "training_export_artifact_sha256",
    "training_export_file_sha256",
    "training_manifest_file_sha256",
    "training_manifest_sha256",
}
_PLAN_FIELDS = {
    *_PLAN_SPEC_FIELDS,
    "artifact_sha256",
    "formal_evaluation_eligible",
    "preprocessing",
    "promoted",
    "promotion_eligible",
    "purpose",
    "regularization_c",
    "runtime_consumable",
    "schema_version",
    "selection_protocol",
    "threshold_selection",
}
_EMBEDDING_SOURCE_FIELDS = {
    "backbone",
    "backbone_sha256",
    "clip_frames",
    "embedding_dimension",
    "sampling_protocol",
}
_VARIANT_FIELDS = {"backbones", "input_dimension", "name"}
_PREPROCESSING = {
    "normalization": "standard_scaler_fit_split_train_only",
    "pca_enabled": False,
    "pca_components": None,
}
_THRESHOLD_SELECTION = {
    "protocol": THRESHOLD_PROTOCOL,
    "selection_metric": SELECTION_METRIC,
    "tie_break": SELECTION_TIE_BREAK,
    "threshold_grid": list(THRESHOLD_GRID),
}
_TRAINING_EXAMPLE_FIELDS = {
    "candidate_bundle_sha256",
    "event_id",
    "event_present",
    "source_video_sha256",
}
_EMBEDDING_ROOT_FIELDS = {
    "artifact_sha256",
    "backbone",
    "backbone_license",
    "backbone_sha256",
    "backbone_weights_url",
    "clip_frames",
    "embedding_dimension",
    "examples",
    "producer",
    "purpose",
    "runtime_consumable",
    "sampling_protocol",
    "schema_version",
    "source_video_sha256",
    "training_annotation_sha256",
    "training_manifest_sha256",
}
_EMBEDDING_EXAMPLE_FIELDS = {
    *_TRAINING_EXAMPLE_FIELDS,
    "embedding",
    "sampling_window",
}
_MANIFEST_FIELDS = {
    "annotation_files",
    "benchmark_overlap",
    "benchmark_raw_sha256",
    "manifest_sha256",
    "producer",
    "purpose",
    "runtime_consumable",
    "schema_version",
    "source_videos",
    "task_types",
}
_ASSET_FIELDS = {"filename", "sha256", "size_bytes"}
_PROBE_FIELDS = {
    "artifact_sha256",
    "formal_evaluation_eligible",
    "game_group_count",
    "negative_count",
    "observed_metrics",
    "outer_folds",
    "positive_count",
    "probe_plan",
    "probe_plan_file_sha256",
    "promoted",
    "promotion_eligible",
    "production_family_count",
    "production_family_disjoint_all",
    "purpose",
    "row_count",
    "runtime_consumable",
    "schema_version",
    "selection_protocol",
    "source_groups",
    "source_groups_file_sha256",
    "source_selection",
    "training_export_receipt",
    "training_export_index",
    "training_manifest_receipt",
    "video_embedding_artifacts",
}
_EXPORT_RECEIPT_FIELDS = {
    "artifact_sha256",
    "file_sha256",
    "filename",
    "size_bytes",
}
_MANIFEST_RECEIPT_FIELDS = {"file_sha256", "manifest"}
_SOURCE_GROUP_REF_FIELDS = {
    "artifact_sha256",
    "file_sha256",
    "schema_version",
}
_EMBEDDING_RECEIPT_FIELDS = {
    "artifact_sha256",
    "backbone",
    "backbone_sha256",
    "clip_frames",
    "embedding_dimension",
    "file_sha256",
    "sampling_protocol",
}
_FOLD_FIELDS = {
    "fit_game_ids",
    "held_game_id",
    "held_metrics",
    "held_production_family",
    "held_source_video_sha256",
    "inner_selection_game_ids",
    "oof_predictions",
    "production_family_disjoint",
    "selected_settings",
    "threshold_selection",
}
_SELECTED_SETTINGS_FIELDS = {
    "backbones",
    "input_dimension",
    "normalization",
    "pca_components",
    "pca_enabled",
    "regularization_c",
    "representation_variant",
    "threshold",
}
_THRESHOLD_RESULT_FIELDS = {
    "candidate_results",
    "inner_folds",
    "protocol",
    "selection_metric",
    "tie_break",
    "variant_inner_oof_predictions",
}
_CANDIDATE_RESULT_FIELDS = {
    "inner_oof_metrics",
    "input_dimension",
    "representation_variant",
    "threshold",
}
_INNER_FOLD_FIELDS = {
    "fit_game_ids",
    "held_game_id",
    "negative_count",
    "positive_count",
    "row_count",
}
_PREDICTION_FIELDS = {
    "candidate_bundle_sha256",
    "decision",
    "event_id",
    "event_present",
    "probability",
    "source_video_sha256",
}
_VARIANT_INNER_FIELDS = {
    "input_dimension",
    "predictions",
    "representation_variant",
}
_INNER_PREDICTION_FIELDS = {
    "candidate_bundle_sha256",
    "event_id",
    "event_present",
    "probability",
    "source_video_sha256",
}
_METRIC_FIELDS = {
    "balanced_accuracy",
    "f1",
    "fn",
    "fp",
    "negative_support",
    "positive_support",
    "precision",
    "recall",
    "specificity",
    "support",
    "tn",
    "tp",
}


def seal_vru_causal_video_probe_plan_v2(
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the label-independent configuration for the four-game probe."""

    if not isinstance(spec, Mapping) or set(spec) != _PLAN_SPEC_FIELDS:
        raise ValueError("causal video probe plan spec fields are not canonical")
    plan: dict[str, Any] = copy.deepcopy(dict(spec))
    plan.update(
        {
            "schema_version": VIDEO_PROBE_PLAN_V2_SCHEMA,
            "purpose": VIDEO_PROBE_PURPOSE,
            "runtime_consumable": False,
            "formal_evaluation_eligible": False,
            "promotion_eligible": False,
            "promoted": False,
            "selection_protocol": SELECTION_PROTOCOL,
            "regularization_c": REGULARIZATION_C,
            "preprocessing": dict(_PREPROCESSING),
            "threshold_selection": copy.deepcopy(_THRESHOLD_SELECTION),
        }
    )
    _validate_probe_plan(plan)
    plan["artifact_sha256"] = _canonical_sha256(plan)
    return plan


def verify_vru_causal_video_probe_plan_v2(
    payload: Mapping[str, Any],
    *,
    expected_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify the exact frozen plan and an optional external receipt."""

    if not isinstance(payload, Mapping) or set(payload) != _PLAN_FIELDS:
        raise ValueError("causal video probe plan fields are not canonical")
    plan = copy.deepcopy(dict(payload))
    claimed_sha256 = _require_sha256(
        plan.pop("artifact_sha256", None),
        field="causal video probe plan artifact",
    )
    if claimed_sha256 != _canonical_sha256(plan):
        raise ValueError("causal video probe plan artifact hash mismatch")
    if expected_artifact_sha256 is not None:
        expected = _require_sha256(
            expected_artifact_sha256,
            field="expected causal video probe plan artifact",
        )
        if claimed_sha256 != expected:
            raise ValueError("causal video probe plan does not match expected artifact")
    _validate_probe_plan(plan)
    plan["artifact_sha256"] = claimed_sha256
    return plan


def _validate_probe_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != VIDEO_PROBE_PLAN_V2_SCHEMA:
        raise ValueError("unsupported causal video probe plan schema")
    if plan.get("purpose") != VIDEO_PROBE_PURPOSE:
        raise ValueError("causal video probe plan purpose is invalid")
    for field in (
        "runtime_consumable",
        "formal_evaluation_eligible",
        "promotion_eligible",
        "promoted",
    ):
        if plan.get(field) is not False:
            raise ValueError(f"causal video probe plan {field} must be false")
    if plan.get("selection_protocol") != SELECTION_PROTOCOL:
        raise ValueError("causal video probe plan selection policy is invalid")
    if type(plan.get("regularization_c")) is not float or plan.get("regularization_c") != REGULARIZATION_C:
        raise ValueError("causal video probe plan regularization policy is invalid")
    _validate_preprocessing_policy(plan.get("preprocessing"))
    _validate_threshold_policy(plan.get("threshold_selection"))
    for field in (
        "training_export_artifact_sha256",
        "training_export_file_sha256",
        "training_manifest_sha256",
        "training_manifest_file_sha256",
        "source_groups_artifact_sha256",
        "source_groups_file_sha256",
    ):
        _require_sha256(plan.get(field), field=field.replace("_", " "))
    _validate_embedding_sources(plan.get("embedding_sources"))
    _validate_representation_variants(plan.get("representation_variants"))


def _validate_embedding_sources(value: object) -> None:
    if not isinstance(value, list) or len(value) != len(EXPECTED_BACKBONES):
        raise ValueError("causal video probe plan requires the two frozen backbones")
    observed_backbones = []
    for row in value:
        if not isinstance(row, Mapping) or set(row) != _EMBEDDING_SOURCE_FIELDS:
            raise ValueError("causal video probe embedding source fields are invalid")
        observed_backbones.append(row.get("backbone"))
        _require_sha256(
            row.get("backbone_sha256"),
            field="causal video probe backbone checkpoint",
        )
        if type(row.get("embedding_dimension")) is not int or row.get("embedding_dimension") != 768:
            raise ValueError("causal video probe embedding dimension is invalid")
        if type(row.get("clip_frames")) is not int or row.get("clip_frames") != 16:
            raise ValueError("causal video probe clip frame count is invalid")
        if row.get("sampling_protocol") != SHOT_SAMPLING_PROTOCOL:
            raise ValueError("causal video probe sampling protocol is invalid")
    if tuple(observed_backbones) != EXPECTED_BACKBONES:
        raise ValueError("causal video probe backbones are not in canonical order")


def _validate_representation_variants(value: object) -> None:
    if not isinstance(value, list) or len(value) != len(EXPECTED_VARIANTS):
        raise ValueError("causal video probe requires the two frozen variants")
    observed = []
    for row in value:
        if not isinstance(row, Mapping) or set(row) != _VARIANT_FIELDS:
            raise ValueError("causal video probe variant fields are invalid")
        backbones = row.get("backbones")
        if not isinstance(backbones, list):
            raise ValueError("causal video probe variant backbones are invalid")
        if type(row.get("input_dimension")) is not int:
            raise ValueError("causal video probe variant input dimension type is invalid")
        observed.append((row.get("name"), tuple(backbones), row.get("input_dimension")))
    if tuple(observed) != EXPECTED_VARIANTS:
        raise ValueError("causal video probe variant order or dimension is invalid")


def _validate_preprocessing_policy(value: object) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != set(_PREPROCESSING)
        or value.get("normalization") != _PREPROCESSING["normalization"]
        or value.get("pca_enabled") is not False
        or value.get("pca_components") is not None
    ):
        raise ValueError("causal video probe plan preprocessing policy is invalid")


def _validate_threshold_policy(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != set(_THRESHOLD_SELECTION):
        raise ValueError("causal video probe plan threshold policy is invalid")
    if (
        value.get("protocol") != THRESHOLD_PROTOCOL
        or value.get("selection_metric") != SELECTION_METRIC
        or value.get("tie_break") != SELECTION_TIE_BREAK
    ):
        raise ValueError("causal video probe plan threshold policy is invalid")
    grid = value.get("threshold_grid")
    if (
        not isinstance(grid, list)
        or len(grid) != len(THRESHOLD_GRID)
        or any(
            type(actual) is not float or actual != expected
            for actual, expected in zip(grid, THRESHOLD_GRID, strict=True)
        )
    ):
        raise ValueError("causal video probe plan threshold grid type or value is invalid")


def screen_vru_causal_video_embeddings_nested_v2(
    *,
    embedding_paths: Sequence[Path],
    expected_embedding_artifact_sha256s: Sequence[str],
    expected_embedding_file_sha256s: Sequence[str],
    training_export_path: Path,
    expected_training_export_artifact_sha256: str,
    expected_training_export_file_sha256: str,
    extension_asset_root: Path,
    parent_export_path: Path,
    expected_parent_artifact_sha256: str,
    expected_parent_file_sha256: str,
    parent_asset_root: Path,
    harwood_selection_path: Path,
    expected_harwood_selection_artifact_sha256: str,
    harwood_review_plan_path: Path,
    expected_harwood_review_plan_artifact_sha256: str,
    harwood_sealed_review_path: Path,
    expected_harwood_sealed_review_artifact_sha256: str,
    harwood_raw_frame_manifest_path: Path,
    expected_harwood_raw_frame_manifest_artifact_sha256: str,
    harwood_source_manifest_path: Path,
    source_groups_path: Path,
    expected_source_groups_artifact_sha256: str,
    expected_source_groups_file_sha256: str,
    training_manifest_path: Path,
    expected_training_manifest_sha256: str,
    expected_training_manifest_file_sha256: str,
    probe_plan_path: Path,
    expected_probe_plan_sha256: str,
    expected_probe_plan_file_sha256: str,
) -> dict[str, object]:
    """Replay the v2 chain, then run its plan-first four-game diagnostic."""

    expected_plan_file = _require_sha256(
        expected_probe_plan_file_sha256,
        field="expected causal video probe plan file",
    )
    plan_payload, plan_file_sha256, _plan_size_bytes = _read_json_snapshot(probe_plan_path)
    if plan_file_sha256 != expected_plan_file:
        raise ValueError("causal video probe plan file does not match its external receipt")
    plan = verify_vru_causal_video_probe_plan_v2(
        plan_payload,
        expected_artifact_sha256=expected_probe_plan_sha256,
    )
    _verify_plan_external_receipts(
        plan,
        expected_training_export_artifact_sha256=(expected_training_export_artifact_sha256),
        expected_training_export_file_sha256=expected_training_export_file_sha256,
        expected_source_groups_artifact_sha256=(expected_source_groups_artifact_sha256),
        expected_source_groups_file_sha256=expected_source_groups_file_sha256,
        expected_training_manifest_sha256=expected_training_manifest_sha256,
        expected_training_manifest_file_sha256=(expected_training_manifest_file_sha256),
    )

    export_payload, export_file_sha256, export_size_bytes = _read_json_snapshot(training_export_path)
    if export_file_sha256 != plan["training_export_file_sha256"]:
        raise ValueError("training export file does not match the frozen probe plan")
    training_chain = verify_vru_causal_shot_validity_training_extension_export(
        export_payload,
        expected_artifact_sha256=expected_training_export_artifact_sha256,
        extension_asset_root=extension_asset_root,
        parent_export_path=parent_export_path,
        expected_parent_artifact_sha256=expected_parent_artifact_sha256,
        expected_parent_file_sha256=expected_parent_file_sha256,
        parent_asset_root=parent_asset_root,
        harwood_selection_path=harwood_selection_path,
        expected_harwood_selection_artifact_sha256=(expected_harwood_selection_artifact_sha256),
        harwood_review_plan_path=harwood_review_plan_path,
        expected_harwood_review_plan_artifact_sha256=(expected_harwood_review_plan_artifact_sha256),
        harwood_sealed_review_path=harwood_sealed_review_path,
        expected_harwood_sealed_review_artifact_sha256=(expected_harwood_sealed_review_artifact_sha256),
        harwood_raw_frame_manifest_path=harwood_raw_frame_manifest_path,
        expected_harwood_raw_frame_manifest_artifact_sha256=(expected_harwood_raw_frame_manifest_artifact_sha256),
        harwood_source_manifest_path=harwood_source_manifest_path,
        source_groups_path=source_groups_path,
        expected_source_groups_artifact_sha256=(expected_source_groups_artifact_sha256),
    )
    return _screen_verified_chain(
        embedding_paths=embedding_paths,
        training_chain=training_chain,
        training_export_asset={
            "filename": training_export_path.name,
            "sha256": export_file_sha256,
            "size_bytes": export_size_bytes,
        },
        source_groups_path=source_groups_path,
        expected_source_groups_artifact_sha256=(expected_source_groups_artifact_sha256),
        expected_source_groups_file_sha256=expected_source_groups_file_sha256,
        training_manifest_path=training_manifest_path,
        expected_training_manifest_sha256=expected_training_manifest_sha256,
        expected_training_manifest_file_sha256=(expected_training_manifest_file_sha256),
        probe_plan=plan,
        probe_plan_file_sha256=plan_file_sha256,
        expected_embedding_artifact_sha256s=expected_embedding_artifact_sha256s,
        expected_embedding_file_sha256s=expected_embedding_file_sha256s,
    )


def _verify_plan_external_receipts(
    plan: Mapping[str, Any],
    *,
    expected_training_export_artifact_sha256: str,
    expected_training_export_file_sha256: str,
    expected_source_groups_artifact_sha256: str,
    expected_source_groups_file_sha256: str,
    expected_training_manifest_sha256: str,
    expected_training_manifest_file_sha256: str,
) -> None:
    expected = {
        "training_export_artifact_sha256": expected_training_export_artifact_sha256,
        "training_export_file_sha256": expected_training_export_file_sha256,
        "source_groups_artifact_sha256": expected_source_groups_artifact_sha256,
        "source_groups_file_sha256": expected_source_groups_file_sha256,
        "training_manifest_sha256": expected_training_manifest_sha256,
        "training_manifest_file_sha256": expected_training_manifest_file_sha256,
    }
    for field, value in expected.items():
        receipt = _require_sha256(value, field=f"expected {field.replace('_', ' ')}")
        if plan[field] != receipt:
            raise ValueError(f"causal video probe plan does not bind {field}")


def _screen_verified_chain(
    *,
    embedding_paths: Sequence[Path],
    training_chain: ContinuousCausalTrainingChain,
    training_export_asset: Mapping[str, object],
    source_groups_path: Path,
    expected_source_groups_artifact_sha256: str,
    expected_source_groups_file_sha256: str,
    training_manifest_path: Path,
    expected_training_manifest_sha256: str,
    expected_training_manifest_file_sha256: str,
    probe_plan: Mapping[str, Any],
    probe_plan_file_sha256: str,
    expected_embedding_artifact_sha256s: Sequence[str],
    expected_embedding_file_sha256s: Sequence[str],
) -> dict[str, object]:
    """Compute only after the public boundary has replayed the complete chain."""

    if not isinstance(training_chain, ContinuousCausalTrainingChain):
        raise TypeError("training_chain must be a verified ContinuousCausalTrainingChain")
    export_assets = _strict_asset_rows(
        [training_export_asset],
        field="verified training export",
        expected_count=1,
    )
    export_asset = export_assets[0]
    expected_export_file_sha256 = str(export_asset["sha256"])
    plan = verify_vru_causal_video_probe_plan_v2(probe_plan)
    plan_file_sha256 = _require_sha256(
        probe_plan_file_sha256,
        field="verified causal video probe plan file",
    )
    export_artifact_sha256 = _require_sha256(
        training_chain.index.get("artifact_sha256"),
        field="verified training export artifact",
    )
    if (
        plan["training_export_artifact_sha256"] != export_artifact_sha256
        or plan["training_export_file_sha256"] != expected_export_file_sha256
    ):
        raise ValueError("probe plan does not bind the verified training export receipts")
    training_export_index = _verify_training_export_index_receipt(
        training_chain.index,
        plan=plan,
        training_export_asset=export_asset,
    )

    source_group_payload, source_groups_file_sha256, _source_group_size = _read_json_snapshot(source_groups_path)
    expected_source_group_file = _require_sha256(
        expected_source_groups_file_sha256,
        field="expected source-group file",
    )
    if source_groups_file_sha256 != expected_source_group_file:
        raise ValueError("source-group file SHA-256 does not match its external receipt")
    source_groups = verify_vru_causal_source_groups(
        source_group_payload,
        expected_artifact_sha256=expected_source_groups_artifact_sha256,
    )
    if (
        plan["source_groups_artifact_sha256"] != source_groups["artifact_sha256"]
        or plan["source_groups_file_sha256"] != source_groups_file_sha256
    ):
        raise ValueError("probe plan does not bind the source-group receipts")
    if _canonical_json(training_chain.source_groups) != _canonical_json(source_groups):
        raise ValueError("verified training chain source groups do not match the probe input")
    expected_bundle_by_source = _training_index_bundle_by_source(
        training_export_index,
        source_groups=source_groups,
    )

    manifest_payload, manifest_file_sha256, _manifest_size = _read_json_snapshot(training_manifest_path)
    expected_manifest_file = _require_sha256(
        expected_training_manifest_file_sha256,
        field="expected training manifest file",
    )
    if manifest_file_sha256 != expected_manifest_file:
        raise ValueError("training manifest file SHA-256 does not match its external receipt")
    manifest = verify_training_annotation_manifest(manifest_payload)
    expected_manifest_sha256 = _require_sha256(
        expected_training_manifest_sha256,
        field="expected training manifest",
    )
    if manifest.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("training manifest does not match its external receipt")
    if (
        plan["training_manifest_sha256"] != expected_manifest_sha256
        or plan["training_manifest_file_sha256"] != manifest_file_sha256
    ):
        raise ValueError("probe plan does not bind the training manifest receipts")
    manifest_receipt = _verify_training_manifest_v2_bindings(
        manifest,
        training_chain=training_chain,
        source_groups=source_groups,
        training_export_asset=export_asset,
        manifest_file_sha256=manifest_file_sha256,
    )

    examples = _strict_resolved_examples(
        training_chain.resolved_examples,
        source_groups=source_groups,
        expected_bundle_by_source=expected_bundle_by_source,
    )
    expected_keys = [_example_key(row) for row in examples]
    expected_labels = [bool(row["event_present"]) for row in examples]
    embeddings, embedding_receipts = _load_verified_embeddings(
        embedding_paths=embedding_paths,
        expected_artifact_sha256s=expected_embedding_artifact_sha256s,
        expected_file_sha256s=expected_embedding_file_sha256s,
        plan=plan,
        manifest_receipt=manifest_receipt,
        expected_keys=expected_keys,
        expected_labels=expected_labels,
    )
    matrices = {str(artifact["backbone"]): matrix for artifact, matrix in embeddings}
    variants = [
        (
            str(row["name"]),
            tuple(str(backbone) for backbone in row["backbones"]),
            np.concatenate(
                [matrices[str(backbone)] for backbone in row["backbones"]],
                axis=1,
            ),
        )
        for row in plan["representation_variants"]
    ]
    for variant_plan, (_name, _backbones, matrix) in zip(
        plan["representation_variants"],
        variants,
        strict=True,
    ):
        if matrix.shape != (EXPECTED_ROW_COUNT, variant_plan["input_dimension"]):
            raise ValueError("video representation matrix does not match the frozen plan")

    target = np.asarray(expected_labels, dtype=np.int64)
    source_to_game = {str(row["source_video_sha256"]): str(row["game_id"]) for row in source_groups["games"]}
    game_to_family = {str(row["game_id"]): str(row["production_family"]) for row in source_groups["games"]}
    games = np.asarray(
        [source_to_game[str(row["source_video_sha256"])] for row in examples],
        dtype=str,
    )
    outer_folds: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []
    for held_game in EXPECTED_GAME_ORDER:
        fit_games = [game for game in EXPECTED_GAME_ORDER if game != held_game]
        development_mask = np.isin(games, fit_games)
        held_mask = games == held_game
        candidate_results: list[dict[str, object]] = []
        variant_inner_oof_predictions: list[dict[str, object]] = []
        for variant_name, _variant_backbones, matrix in variants:
            inner_probabilities = np.full(len(target), np.nan, dtype=np.float64)
            for inner_held_game in fit_games:
                inner_held_mask = development_mask & (games == inner_held_game)
                inner_fit_mask = development_mask & (games != inner_held_game)
                inner_probabilities[inner_held_mask] = _fit_probability_model(
                    matrix=matrix,
                    target=target,
                    fit_mask=inner_fit_mask,
                    predict_mask=inner_held_mask,
                )
            if not np.isfinite(inner_probabilities[development_mask]).all():
                raise ValueError("nested inner game-held predictions are incomplete")
            development_examples = [
                row
                for row, is_development in zip(
                    examples,
                    development_mask.tolist(),
                    strict=True,
                )
                if is_development
            ]
            variant_inner_oof_predictions.append(
                {
                    "representation_variant": variant_name,
                    "input_dimension": int(matrix.shape[1]),
                    "predictions": [
                        {
                            "source_video_sha256": str(row["source_video_sha256"]),
                            "candidate_bundle_sha256": str(row["candidate_bundle_sha256"]),
                            "event_id": str(row["event_id"]),
                            "event_present": bool(row["event_present"]),
                            "probability": float(probability),
                        }
                        for row, probability in zip(
                            development_examples,
                            inner_probabilities[development_mask],
                            strict=True,
                        )
                    ],
                }
            )
            for threshold in THRESHOLD_GRID:
                candidate_results.append(
                    {
                        "representation_variant": variant_name,
                        "input_dimension": int(matrix.shape[1]),
                        "threshold": threshold,
                        "inner_oof_metrics": _binary_metrics(
                            target[development_mask],
                            inner_probabilities[development_mask],
                            threshold,
                        ),
                    }
                )
        selected_candidate = _select_candidate(candidate_results, plan=plan)
        selected_variant_name = str(selected_candidate["representation_variant"])
        selected_variant = next(row for row in variants if row[0] == selected_variant_name)
        _name, selected_backbones, selected_matrix = selected_variant
        threshold = float(selected_candidate["threshold"])
        held_probabilities = _fit_probability_model(
            matrix=selected_matrix,
            target=target,
            fit_mask=development_mask,
            predict_mask=held_mask,
        )
        held_decisions = held_probabilities >= threshold
        held_examples = [row for row, is_held in zip(examples, held_mask.tolist(), strict=True) if is_held]
        predictions = [
            {
                "source_video_sha256": str(row["source_video_sha256"]),
                "candidate_bundle_sha256": str(row["candidate_bundle_sha256"]),
                "event_id": str(row["event_id"]),
                "event_present": bool(row["event_present"]),
                "probability": float(probability),
                "decision": bool(decision),
            }
            for row, probability, decision in zip(
                held_examples,
                held_probabilities,
                held_decisions,
                strict=True,
            )
        ]
        all_predictions.extend(predictions)
        held_family = game_to_family[held_game]
        fit_families = {game_to_family[game] for game in fit_games}
        outer_folds.append(
            {
                "held_game_id": held_game,
                "held_source_video_sha256": str(
                    source_groups["games"][EXPECTED_GAME_ORDER.index(held_game)]["source_video_sha256"]
                ),
                "held_production_family": held_family,
                "fit_game_ids": fit_games,
                "inner_selection_game_ids": fit_games,
                "production_family_disjoint": held_family not in fit_families,
                "selected_settings": {
                    "representation_variant": selected_variant_name,
                    "backbones": list(selected_backbones),
                    "input_dimension": int(selected_matrix.shape[1]),
                    "normalization": _PREPROCESSING["normalization"],
                    "pca_enabled": False,
                    "pca_components": None,
                    "regularization_c": REGULARIZATION_C,
                    "threshold": threshold,
                },
                "threshold_selection": {
                    "protocol": THRESHOLD_PROTOCOL,
                    "selection_metric": SELECTION_METRIC,
                    "tie_break": SELECTION_TIE_BREAK,
                    "inner_folds": _inner_fold_rows(fit_games),
                    "variant_inner_oof_predictions": (variant_inner_oof_predictions),
                    "candidate_results": candidate_results,
                },
                "held_metrics": _binary_metrics(
                    target[held_mask],
                    held_probabilities,
                    threshold,
                ),
                "oof_predictions": predictions,
            }
        )

    observed_metrics = _observed_metrics(
        all_predictions,
        source_groups=source_groups,
    )
    payload: dict[str, object] = {
        "schema_version": VIDEO_PROBE_V2_SCHEMA,
        "purpose": VIDEO_PROBE_PURPOSE,
        "runtime_consumable": False,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
        "selection_protocol": SELECTION_PROTOCOL,
        "source_selection": {
            "parent_prior_stratified_rows": 22,
            "harwood_geometry_only_label_hidden_rows": 23,
            "continuous_exhaustive_truth": False,
            "limitation": SOURCE_SELECTION_LIMITATION,
        },
        "training_export_receipt": {
            "artifact_sha256": export_artifact_sha256,
            "file_sha256": expected_export_file_sha256,
            "filename": export_asset["filename"],
            "size_bytes": export_asset["size_bytes"],
        },
        "training_export_index": training_export_index,
        "source_groups": copy.deepcopy(source_groups),
        "source_groups_file_sha256": source_groups_file_sha256,
        "training_manifest_receipt": manifest_receipt,
        "probe_plan": plan,
        "probe_plan_file_sha256": plan_file_sha256,
        "video_embedding_artifacts": embedding_receipts,
        "row_count": EXPECTED_ROW_COUNT,
        "positive_count": EXPECTED_POSITIVE_COUNT,
        "negative_count": EXPECTED_NEGATIVE_COUNT,
        "game_group_count": 4,
        "production_family_count": 2,
        "production_family_disjoint_all": all(bool(fold["production_family_disjoint"]) for fold in outer_folds),
        "outer_folds": outer_folds,
        "observed_metrics": observed_metrics,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return verify_vru_causal_video_probe_v2(payload)


def _strict_resolved_examples(
    rows: object,
    *,
    source_groups: Mapping[str, Any],
    expected_bundle_by_source: Mapping[str, str],
) -> list[dict[str, object]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("verified training chain examples must be a sequence")
    source_rows = list(source_groups["games"])
    source_positions = {str(row["source_video_sha256"]): index for index, row in enumerate(source_rows)}
    source_to_game = {str(row["source_video_sha256"]): str(row["game_id"]) for row in source_rows}
    examples: list[dict[str, object]] = []
    keys: list[tuple[str, str, str]] = []
    order_keys: list[tuple[int, str]] = []
    labels_by_game: dict[str, list[bool]] = {game_id: [] for game_id in EXPECTED_GAME_ORDER}
    bundles_by_game: dict[str, set[str]] = {game_id: set() for game_id in EXPECTED_GAME_ORDER}
    for raw_row in rows:
        if not isinstance(raw_row, Mapping) or set(raw_row) != _TRAINING_EXAMPLE_FIELDS:
            raise ValueError("resolved training example fields are not canonical")
        row = dict(raw_row)
        key = _example_key(row)
        source_sha256, bundle_sha256, event_id = key
        if source_sha256 not in source_positions:
            raise ValueError("resolved training example source is not in source groups")
        if bundle_sha256 != expected_bundle_by_source.get(source_sha256):
            raise ValueError("resolved training example bundle differs from the verified export index")
        if not isinstance(row.get("event_present"), bool):
            raise ValueError("resolved training examples require boolean labels")
        game_id = source_to_game[source_sha256]
        labels_by_game[game_id].append(bool(row["event_present"]))
        bundles_by_game[game_id].add(bundle_sha256)
        examples.append(row)
        keys.append(key)
        order_keys.append((source_positions[source_sha256], event_id))
    if len(keys) != len(set(keys)):
        raise ValueError("resolved training examples contain duplicate three-part keys")
    if order_keys != sorted(order_keys):
        raise ValueError("resolved training examples are not in canonical game/event order")
    for game_id, (positive_count, negative_count) in EXPECTED_GAME_COUNTS.items():
        labels = labels_by_game[game_id]
        if sum(labels) != positive_count or len(labels) - sum(labels) != negative_count:
            raise ValueError("resolved training examples do not retain per-game class counts")
        if len(bundles_by_game[game_id]) != 1:
            raise ValueError("each game must use exactly one candidate bundle receipt")
    if len(examples) != EXPECTED_ROW_COUNT:
        raise ValueError("resolved training chain must contain exactly 45 rows")
    return examples


def _example_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    source_sha256 = _require_sha256(
        row.get("source_video_sha256"),
        field="example source video",
    )
    bundle_sha256 = _require_sha256(
        row.get("candidate_bundle_sha256"),
        field="example candidate bundle",
    )
    event_id = row.get("event_id")
    if not isinstance(event_id, str) or _SAFE_EVENT_ID.fullmatch(event_id) is None:
        raise ValueError("example event ID is unsafe")
    return source_sha256, bundle_sha256, event_id


def _verify_training_manifest_v2_bindings(
    manifest: Mapping[str, Any],
    *,
    training_chain: ContinuousCausalTrainingChain,
    source_groups: Mapping[str, Any],
    training_export_asset: Mapping[str, object],
    manifest_file_sha256: str,
) -> dict[str, object]:
    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("training manifest fields are not canonical")
    _verify_manifest_policy_and_sources(manifest, source_groups=source_groups)
    annotation_assets = _strict_asset_rows(
        manifest.get("annotation_files"),
        field="training manifest annotation",
        expected_count=5,
    )
    normalized_export = _strict_asset_rows(
        [training_export_asset],
        field="verified training export",
        expected_count=1,
    )[0]
    expected_annotations = [
        normalized_export,
        *_expected_label_assets_from_index(training_chain.index),
    ]
    if annotation_assets != expected_annotations:
        raise ValueError(
            "training manifest annotation order and exact label lineage do not match the verified v2 chain"
        )
    return {
        "file_sha256": manifest_file_sha256,
        "manifest": copy.deepcopy(dict(manifest)),
    }


def _verify_manifest_policy_and_sources(
    manifest: Mapping[str, Any],
    *,
    source_groups: Mapping[str, Any],
) -> None:
    if (
        manifest.get("purpose") != "model_training_only"
        or manifest.get("producer") != TRAINING_MANIFEST_PRODUCER
        or manifest.get("runtime_consumable") is not False
        or manifest.get("benchmark_overlap") is not False
        or manifest.get("task_types") != ["shot_validity"]
        or manifest.get("benchmark_raw_sha256") != [ACCEPTANCE_BENCHMARK_RAW_SHA256]
    ):
        raise ValueError("training manifest must retain the exact shot-validity acceptance benchmark contract")
    source_assets = _strict_asset_rows(
        manifest.get("source_videos"),
        field="training manifest source video",
        expected_count=4,
    )
    expected_source_sha256s = [str(row["source_video_sha256"]) for row in source_groups["games"]]
    if [str(row["sha256"]) for row in source_assets] != expected_source_sha256s:
        raise ValueError("training manifest source videos do not match source groups")
    if ACCEPTANCE_BENCHMARK_RAW_SHA256 in expected_source_sha256s:
        raise ValueError("training manifest source videos overlap the acceptance benchmark")


def _verify_training_export_index_receipt(
    value: object,
    *,
    plan: Mapping[str, Any],
    training_export_asset: Mapping[str, object],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("verified training export index must be an object")
    index = verify_vru_causal_shot_validity_training_extension_index(
        value,
        expected_artifact_sha256=str(plan["training_export_artifact_sha256"]),
    )
    normalized_asset = _strict_asset_rows(
        [training_export_asset],
        field="verified training export",
        expected_count=1,
    )[0]
    encoded_index = encode_json_artifact(index)
    if (
        hashlib.sha256(encoded_index).hexdigest() != normalized_asset["sha256"]
        or len(encoded_index) != normalized_asset["size_bytes"]
    ):
        raise ValueError("verified training export index bytes do not match its file receipt")
    source_group_ref = index.get("source_groups")
    if (
        not isinstance(source_group_ref, Mapping)
        or set(source_group_ref) != _SOURCE_GROUP_REF_FIELDS
        or source_group_ref.get("schema_version") != SOURCE_GROUP_SCHEMA
        or source_group_ref.get("artifact_sha256") != plan["source_groups_artifact_sha256"]
        or source_group_ref.get("file_sha256") != plan["source_groups_file_sha256"]
    ):
        raise ValueError("verified training export index source-group receipt is invalid")
    _expected_label_assets_from_index(index)
    return index


def _expected_label_assets_from_index(
    index: Mapping[str, Any],
) -> list[dict[str, object]]:
    parent = index.get("parent")
    extension = index.get("extension")
    if not isinstance(parent, Mapping) or not isinstance(extension, Mapping):
        raise ValueError("verified training chain lacks canonical label lineage")
    sources = parent.get("sources")
    if not isinstance(sources, list) or len(sources) != 3:
        raise ValueError("verified training chain parent label lineage is incomplete")
    raw_assets: list[object] = []
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("verified training chain parent label lineage is invalid")
        raw_assets.append(source.get("labels"))
    raw_assets.append(extension.get("labels"))
    assets = _strict_asset_rows(
        raw_assets,
        field="verified training chain label",
        expected_count=4,
    )
    if [row["filename"] for row in assets] != list(EXPECTED_LABEL_FILENAMES):
        raise ValueError("verified training chain label filenames are not canonical")
    for row in assets[:3]:
        if row["sha256"] != FROZEN_PARENT_CHILD_SHA256[str(row["filename"])]:
            raise ValueError("verified training chain parent label receipt is not frozen")
    return assets


def _training_index_bundle_by_source(
    index: Mapping[str, Any],
    *,
    source_groups: Mapping[str, Any],
) -> dict[str, str]:
    parent = index.get("parent")
    extension = index.get("extension")
    if not isinstance(parent, Mapping) or not isinstance(extension, Mapping):
        raise ValueError("verified training export index source lineage is invalid")
    parent_sources = parent.get("sources")
    group_games = source_groups.get("games")
    if (
        not isinstance(parent_sources, list)
        or len(parent_sources) != 3
        or not isinstance(group_games, list)
        or len(group_games) != len(EXPECTED_GAME_ORDER)
    ):
        raise ValueError("verified training export index source lineage is incomplete")
    index_sources = [*parent_sources, extension]
    bundle_by_source: dict[str, str] = {}
    for game_id, group_row, index_row in zip(
        EXPECTED_GAME_ORDER,
        group_games,
        index_sources,
        strict=True,
    ):
        if not isinstance(group_row, Mapping) or not isinstance(index_row, Mapping):
            raise ValueError("verified training export index source lineage is invalid")
        source_sha256 = _require_sha256(
            group_row.get("source_video_sha256"),
            field="verified source-group video",
        )
        candidate = index_row.get("candidate_bundle")
        if (
            group_row.get("game_id") != game_id
            or index_row.get("source_id") != game_id
            or index_row.get("source_video_sha256") != source_sha256
            or (game_id == "harwood" and index_row.get("game_id") != game_id)
            or not isinstance(candidate, Mapping)
        ):
            raise ValueError("verified training export index does not match source groups")
        bundle_by_source[source_sha256] = _require_sha256(
            candidate.get("bundle_sha256"),
            field="verified training export candidate bundle",
        )
    return bundle_by_source


def _strict_asset_rows(
    value: object,
    *,
    field: str,
    expected_count: int,
) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != expected_count:
        raise ValueError(f"{field} assets must contain exactly {expected_count} rows")
    rows: list[dict[str, object]] = []
    names: set[str] = set()
    hashes: set[str] = set()
    for raw_row in value:
        if not isinstance(raw_row, Mapping) or set(raw_row) != _ASSET_FIELDS:
            raise ValueError(f"{field} asset fields are not canonical")
        filename = raw_row.get("filename")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or "/" in filename
            or "\\" in filename
        ):
            raise ValueError(f"{field} filename is unsafe")
        sha256 = _require_sha256(raw_row.get("sha256"), field=field)
        size_bytes = raw_row.get("size_bytes")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 1:
            raise ValueError(f"{field} size must be a positive integer")
        if filename.casefold() in names or sha256 in hashes:
            raise ValueError(f"{field} assets must have unique names and receipts")
        names.add(filename.casefold())
        hashes.add(sha256)
        rows.append({"filename": filename, "sha256": sha256, "size_bytes": size_bytes})
    return rows


def _load_verified_embeddings(
    *,
    embedding_paths: Sequence[Path],
    expected_artifact_sha256s: Sequence[str],
    expected_file_sha256s: Sequence[str],
    plan: Mapping[str, Any],
    manifest_receipt: Mapping[str, object],
    expected_keys: Sequence[tuple[str, str, str]],
    expected_labels: Sequence[bool],
) -> tuple[list[tuple[dict[str, Any], np.ndarray]], list[dict[str, object]]]:
    if not (
        len(embedding_paths) == len(expected_artifact_sha256s) == len(expected_file_sha256s) == len(EXPECTED_BACKBONES)
    ):
        raise ValueError("embedding paths and external receipts must cover two backbones")
    loaded: dict[str, tuple[dict[str, Any], np.ndarray, str]] = {}
    seen_artifact_sha256s: set[str] = set()
    seen_file_sha256s: set[str] = set()
    plan_sources = {str(row["backbone"]): row for row in plan["embedding_sources"]}
    manifest = manifest_receipt.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("verified training manifest receipt is incomplete")
    manifest_sha256 = _require_sha256(
        manifest.get("manifest_sha256"),
        field="verified training manifest",
    )
    annotation_sha256s = sorted(str(row["sha256"]) for row in manifest["annotation_files"][1:])
    source_sha256s = sorted(str(row["sha256"]) for row in manifest["source_videos"])
    for path, expected_artifact, expected_file in zip(
        embedding_paths,
        expected_artifact_sha256s,
        expected_file_sha256s,
        strict=True,
    ):
        expected_artifact = _require_sha256(
            expected_artifact,
            field="expected video embedding artifact",
        )
        expected_file = _require_sha256(
            expected_file,
            field="expected video embedding file",
        )
        payload, file_sha256, _embedding_size = _read_json_snapshot(path)
        if file_sha256 != expected_file:
            raise ValueError("video embedding file does not match its external receipt")
        if set(payload) != _EMBEDDING_ROOT_FIELDS:
            raise ValueError("video embedding artifact fields are not canonical")
        artifact = verify_video_embedding_artifact(payload)
        artifact_sha256 = _require_sha256(
            artifact.get("artifact_sha256"),
            field="video embedding artifact",
        )
        if artifact_sha256 != expected_artifact:
            raise ValueError("video embedding artifact does not match its external receipt")
        backbone = str(artifact.get("backbone"))
        if backbone not in plan_sources or backbone in loaded:
            raise ValueError("video embedding backbones must exactly match the frozen plan")
        source_spec = plan_sources[backbone]
        if any(
            artifact.get(field) != source_spec[field]
            for field in (
                "backbone_sha256",
                "embedding_dimension",
                "clip_frames",
                "sampling_protocol",
            )
        ):
            raise ValueError("video embedding checkpoint or sampling does not match the plan")
        if type(artifact.get("embedding_dimension")) is not int or type(artifact.get("clip_frames")) is not int:
            raise ValueError("video embedding dimension and clip frames must be integers")
        if (
            artifact.get("purpose") != "backbone_screening_training_only"
            or artifact.get("producer") != "agu"
            or artifact.get("runtime_consumable") is not False
        ):
            raise ValueError("video embedding provenance is invalid")
        if artifact.get("training_manifest_sha256") != manifest_sha256:
            raise ValueError("video embedding training manifest receipt is invalid")
        if artifact.get("training_annotation_sha256") != annotation_sha256s:
            raise ValueError("video embedding annotation receipts are invalid")
        if artifact.get("source_video_sha256") != source_sha256s:
            raise ValueError("video embedding source receipts are invalid")
        raw_rows = artifact.get("examples")
        if not isinstance(raw_rows, list) or len(raw_rows) != EXPECTED_ROW_COUNT:
            raise ValueError("video embedding artifact must contain exactly 45 examples")
        lookup: dict[tuple[str, str, str], Mapping[str, object]] = {}
        for row in raw_rows:
            if not isinstance(row, Mapping) or set(row) != _EMBEDDING_EXAMPLE_FIELDS:
                raise ValueError("video embedding example fields are not canonical")
            key = _example_key(row)
            if key in lookup:
                raise ValueError("video embedding examples contain duplicate three-part keys")
            sampling_window = row.get("sampling_window")
            if (
                not isinstance(sampling_window, Mapping)
                or set(sampling_window)
                != {
                    "anchor_frame",
                    "anchor_source",
                    "end_frame",
                    "protocol",
                    "start_frame",
                }
                or sampling_window.get("anchor_source") != "bounded_event"
                or sampling_window.get("anchor_frame") != sampling_window.get("end_frame")
                or sampling_window.get("protocol") != SHOT_SAMPLING_PROTOCOL
                or any(
                    type(sampling_window.get(field)) is not int
                    for field in ("start_frame", "end_frame", "anchor_frame")
                )
            ):
                raise ValueError("video embedding sampling must remain label-hidden")
            vector = row.get("embedding")
            if (
                not isinstance(vector, list)
                or len(vector) != source_spec["embedding_dimension"]
                or any(
                    isinstance(component, bool)
                    or not isinstance(component, (int, float))
                    or not math.isfinite(float(component))
                    for component in vector
                )
            ):
                raise ValueError("video embedding vector values are invalid")
            lookup[key] = row
        if set(lookup) != set(expected_keys):
            raise ValueError("video embedding three-part keys do not match the training chain")
        aligned = [lookup[key] for key in expected_keys]
        if [bool(row.get("event_present")) for row in aligned] != list(expected_labels):
            raise ValueError("video embedding labels do not match the training chain")
        matrix = np.asarray([row["embedding"] for row in aligned], dtype=np.float64)
        if matrix.shape != (EXPECTED_ROW_COUNT, source_spec["embedding_dimension"]):
            raise ValueError("video embedding matrix dimension is invalid")
        if not np.isfinite(matrix).all():
            raise ValueError("video embedding matrix must contain finite values")
        if artifact_sha256 in seen_artifact_sha256s or file_sha256 in seen_file_sha256s:
            raise ValueError("video embedding receipts must be unique")
        seen_artifact_sha256s.add(artifact_sha256)
        seen_file_sha256s.add(file_sha256)
        loaded[backbone] = (artifact, matrix, file_sha256)
    if tuple(backbone for backbone in EXPECTED_BACKBONES if backbone in loaded) != EXPECTED_BACKBONES:
        raise ValueError("video embedding backbones do not cover the frozen plan")
    ordered = [(loaded[backbone][0], loaded[backbone][1]) for backbone in EXPECTED_BACKBONES]
    receipts = [
        {
            "backbone": backbone,
            "artifact_sha256": loaded[backbone][0]["artifact_sha256"],
            "file_sha256": loaded[backbone][2],
            "backbone_sha256": loaded[backbone][0]["backbone_sha256"],
            "embedding_dimension": loaded[backbone][0]["embedding_dimension"],
            "clip_frames": loaded[backbone][0]["clip_frames"],
            "sampling_protocol": loaded[backbone][0]["sampling_protocol"],
        }
        for backbone in EXPECTED_BACKBONES
    ]
    return ordered, receipts


def _fit_probability_model(
    *,
    matrix: np.ndarray,
    target: np.ndarray,
    fit_mask: np.ndarray,
    predict_mask: np.ndarray,
) -> np.ndarray:
    if fit_mask.shape != target.shape or predict_mask.shape != target.shape:
        raise ValueError("nested model masks must align with labels")
    if not fit_mask.any() or not predict_mask.any() or np.any(fit_mask & predict_mask):
        raise ValueError("nested model fit and prediction splits must be disjoint")
    fit_target = target[fit_mask]
    if set(fit_target.tolist()) != {0, 1}:
        raise ValueError("nested model fitting requires both classes")
    scaler = StandardScaler().fit(matrix[fit_mask])
    classifier = LogisticRegression(
        C=REGULARIZATION_C,
        class_weight="balanced",
        max_iter=5000,
        random_state=0,
    ).fit(scaler.transform(matrix[fit_mask]), fit_target)
    probabilities = classifier.predict_proba(scaler.transform(matrix[predict_mask]))[:, 1]
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("nested model produced invalid probabilities")
    return probabilities


def _binary_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    decisions = probabilities >= threshold
    truth = target.astype(bool)
    tp = int(np.sum(decisions & truth))
    fp = int(np.sum(decisions & ~truth))
    fn = int(np.sum(~decisions & truth))
    tn = int(np.sum(~decisions & ~truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "support": tp + fp + fn + tn,
        "positive_support": tp + fn,
        "negative_support": fp + tn,
    }


def _select_candidate(
    candidates: Sequence[Mapping[str, object]],
    *,
    plan: Mapping[str, Any],
) -> Mapping[str, object]:
    variant_positions = {str(row["name"]): index for index, row in enumerate(plan["representation_variants"])}

    def score(row: Mapping[str, object]) -> tuple[float, float, float, float, int, float]:
        metrics = row["inner_oof_metrics"]
        return (
            float(metrics["balanced_accuracy"]),
            float(metrics["f1"]),
            float(metrics["precision"]),
            float(metrics["recall"]),
            -variant_positions[str(row["representation_variant"])],
            float(row["threshold"]),
        )

    if not candidates:
        raise ValueError("nested probe requires threshold candidates")
    return max(candidates, key=score)


def _inner_fold_rows(fit_games: Sequence[str]) -> list[dict[str, object]]:
    rows = []
    for held_game in fit_games:
        positive_count, negative_count = EXPECTED_GAME_COUNTS[held_game]
        rows.append(
            {
                "held_game_id": held_game,
                "fit_game_ids": [game for game in fit_games if game != held_game],
                "row_count": positive_count + negative_count,
                "positive_count": positive_count,
                "negative_count": negative_count,
            }
        )
    return rows


def _observed_metrics(
    predictions: Sequence[Mapping[str, object]],
    *,
    source_groups: Mapping[str, Any],
) -> dict[str, object]:
    source_to_game = {str(row["source_video_sha256"]): str(row["game_id"]) for row in source_groups["games"]}
    game_to_family = {str(row["game_id"]): str(row["production_family"]) for row in source_groups["games"]}

    def metrics_for(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
        return _binary_metrics(
            np.asarray([int(bool(row["event_present"])) for row in rows]),
            np.asarray([float(bool(row["decision"])) for row in rows]),
            0.5,
        )

    per_game = []
    for game_id in EXPECTED_GAME_ORDER:
        rows = [row for row in predictions if source_to_game[str(row["source_video_sha256"])] == game_id]
        per_game.append({"game_id": game_id, "metrics": metrics_for(rows)})
    per_family = []
    for family in ("hctv", "vtv"):
        rows = [row for row in predictions if game_to_family[source_to_game[str(row["source_video_sha256"])]] == family]
        per_family.append(
            {
                "production_family": family,
                "descriptive_only": True,
                "production_held_generalization": False,
                "metrics": metrics_for(rows),
            }
        )
    return {
        "pooled": metrics_for(predictions),
        "per_game": per_game,
        "per_production_family": per_family,
    }


def verify_vru_causal_video_probe_v2(
    payload: Mapping[str, Any],
    *,
    expected_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify a self-contained, permanently non-promotable v2 probe result."""

    if not isinstance(payload, Mapping) or set(payload) != _PROBE_FIELDS:
        raise ValueError("causal video probe v2 fields are not canonical")
    artifact = copy.deepcopy(dict(payload))
    claimed_sha256 = _require_sha256(
        artifact.pop("artifact_sha256", None),
        field="causal video probe v2 artifact",
    )
    if claimed_sha256 != _canonical_sha256(artifact):
        raise ValueError("causal video probe v2 artifact hash mismatch")
    if expected_artifact_sha256 is not None:
        expected = _require_sha256(
            expected_artifact_sha256,
            field="expected causal video probe v2 artifact",
        )
        if claimed_sha256 != expected:
            raise ValueError("causal video probe v2 does not match expected artifact")
    if artifact.get("schema_version") != VIDEO_PROBE_V2_SCHEMA:
        raise ValueError("unsupported causal video probe v2 schema")
    if artifact.get("purpose") != VIDEO_PROBE_PURPOSE:
        raise ValueError("causal video probe v2 purpose is invalid")
    for field in (
        "runtime_consumable",
        "formal_evaluation_eligible",
        "promotion_eligible",
        "promoted",
    ):
        if artifact.get(field) is not False:
            raise ValueError(f"causal video probe v2 {field} must be false")
    if artifact.get("selection_protocol") != SELECTION_PROTOCOL:
        raise ValueError("causal video probe v2 selection protocol is invalid")
    expected_counts = {
        "row_count": EXPECTED_ROW_COUNT,
        "positive_count": EXPECTED_POSITIVE_COUNT,
        "negative_count": EXPECTED_NEGATIVE_COUNT,
        "game_group_count": 4,
        "production_family_count": 2,
    }
    for field, expected_count in expected_counts.items():
        if type(artifact.get(field)) is not int or artifact.get(field) != expected_count:
            raise ValueError("causal video probe v2 count shape is invalid")
    _verify_source_selection(artifact.get("source_selection"))

    plan_value = artifact.get("probe_plan")
    if not isinstance(plan_value, Mapping):
        raise ValueError("causal video probe v2 requires its frozen plan")
    plan = verify_vru_causal_video_probe_plan_v2(plan_value)
    _require_sha256(
        artifact.get("probe_plan_file_sha256"),
        field="causal video probe plan file",
    )
    export_receipt = artifact.get("training_export_receipt")
    if not isinstance(export_receipt, Mapping) or set(export_receipt) != _EXPORT_RECEIPT_FIELDS:
        raise ValueError("causal video probe training export receipt is invalid")
    export_artifact_sha256 = _require_sha256(
        export_receipt.get("artifact_sha256"),
        field="probe training export artifact",
    )
    export_file_sha256 = _require_sha256(
        export_receipt.get("file_sha256"),
        field="probe training export file",
    )
    export_asset = _strict_asset_rows(
        [
            {
                "filename": export_receipt.get("filename"),
                "sha256": export_file_sha256,
                "size_bytes": export_receipt.get("size_bytes"),
            }
        ],
        field="probe training export",
        expected_count=1,
    )[0]
    if (
        export_artifact_sha256 != plan["training_export_artifact_sha256"]
        or export_file_sha256 != plan["training_export_file_sha256"]
    ):
        raise ValueError("causal video probe training export receipt differs from plan")
    training_export_index = _verify_training_export_index_receipt(
        artifact.get("training_export_index"),
        plan=plan,
        training_export_asset=export_asset,
    )

    source_groups_value = artifact.get("source_groups")
    if not isinstance(source_groups_value, Mapping):
        raise ValueError("causal video probe requires source groups")
    source_groups = verify_vru_causal_source_groups(
        source_groups_value,
        expected_artifact_sha256=str(plan["source_groups_artifact_sha256"]),
    )
    source_groups_file_sha256 = _require_sha256(
        artifact.get("source_groups_file_sha256"),
        field="probe source-group file",
    )
    if source_groups_file_sha256 != plan["source_groups_file_sha256"]:
        raise ValueError("causal video probe source-group file differs from plan")
    expected_bundle_by_source = _training_index_bundle_by_source(
        training_export_index,
        source_groups=source_groups,
    )

    manifest_receipt = _verify_manifest_receipt(
        artifact.get("training_manifest_receipt"),
        plan=plan,
        source_groups=source_groups,
        training_export_asset=export_asset,
        training_export_index=training_export_index,
    )
    _verify_embedding_receipts(
        artifact.get("video_embedding_artifacts"),
        plan=plan,
        manifest_receipt=manifest_receipt,
    )
    predictions, production_family_disjoint_all = _verify_outer_folds(
        artifact.get("outer_folds"),
        plan=plan,
        source_groups=source_groups,
        expected_bundle_by_source=expected_bundle_by_source,
    )
    if artifact.get("production_family_disjoint_all") is not (production_family_disjoint_all):
        raise ValueError("production-family-disjoint aggregate is inconsistent")
    if artifact.get("production_family_disjoint_all") is not False:
        raise ValueError("four games do not provide all-fold production disjointness")
    expected_observed = _observed_metrics(
        predictions,
        source_groups=source_groups,
    )
    if _canonical_json(artifact.get("observed_metrics")) != _canonical_json(expected_observed):
        raise ValueError("observed metrics do not exactly match outer predictions")
    artifact["artifact_sha256"] = claimed_sha256
    return artifact


def _verify_source_selection(value: object) -> None:
    expected_fields = {
        "parent_prior_stratified_rows",
        "harwood_geometry_only_label_hidden_rows",
        "continuous_exhaustive_truth",
        "limitation",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("causal video probe v2 source limitation fields are invalid")
    if (
        type(value.get("parent_prior_stratified_rows")) is not int
        or value.get("parent_prior_stratified_rows") != 22
        or type(value.get("harwood_geometry_only_label_hidden_rows")) is not int
        or value.get("harwood_geometry_only_label_hidden_rows") != 23
        or value.get("continuous_exhaustive_truth") is not False
        or value.get("limitation") != SOURCE_SELECTION_LIMITATION
    ):
        raise ValueError("causal video probe v2 source limitation is invalid")


def _verify_manifest_receipt(
    value: object,
    *,
    plan: Mapping[str, Any],
    source_groups: Mapping[str, Any],
    training_export_asset: Mapping[str, object],
    training_export_index: Mapping[str, Any],
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _MANIFEST_RECEIPT_FIELDS:
        raise ValueError("causal video probe training manifest receipt is invalid")
    file_sha256 = _require_sha256(
        value.get("file_sha256"),
        field="probe training manifest file",
    )
    raw_manifest = value.get("manifest")
    if not isinstance(raw_manifest, Mapping) or set(raw_manifest) != _MANIFEST_FIELDS:
        raise ValueError("causal video probe embedded training manifest is invalid")
    manifest = verify_training_annotation_manifest(raw_manifest)
    manifest_sha256 = _require_sha256(
        manifest.get("manifest_sha256"),
        field="probe training manifest",
    )
    if manifest_sha256 != plan["training_manifest_sha256"] or file_sha256 != plan["training_manifest_file_sha256"]:
        raise ValueError("causal video probe training manifest differs from plan")
    _verify_manifest_policy_and_sources(manifest, source_groups=source_groups)
    annotations = _strict_asset_rows(
        manifest.get("annotation_files"),
        field="training manifest annotation",
        expected_count=5,
    )
    expected_annotations = [
        dict(training_export_asset),
        *_expected_label_assets_from_index(training_export_index),
    ]
    if annotations != expected_annotations:
        raise ValueError("training manifest label lineage differs from the verified export index")
    return {"file_sha256": file_sha256, "manifest": manifest}


def _verify_embedding_receipts(
    value: object,
    *,
    plan: Mapping[str, Any],
    manifest_receipt: Mapping[str, object],
) -> None:
    if not isinstance(value, list) or len(value) != len(EXPECTED_BACKBONES):
        raise ValueError("causal video probe requires two embedding receipts")
    artifact_sha256s: set[str] = set()
    file_sha256s: set[str] = set()
    for receipt, source_spec in zip(
        value,
        plan["embedding_sources"],
        strict=True,
    ):
        if not isinstance(receipt, Mapping) or set(receipt) != _EMBEDDING_RECEIPT_FIELDS:
            raise ValueError("causal video probe embedding receipt fields are invalid")
        if any(
            receipt.get(field) != source_spec[field]
            for field in (
                "backbone",
                "backbone_sha256",
                "embedding_dimension",
                "clip_frames",
                "sampling_protocol",
            )
        ):
            raise ValueError("causal video probe embedding receipt differs from plan")
        artifact_sha256 = _require_sha256(
            receipt.get("artifact_sha256"),
            field="probe video embedding artifact",
        )
        file_sha256 = _require_sha256(
            receipt.get("file_sha256"),
            field="probe video embedding file",
        )
        if artifact_sha256 in artifact_sha256s or file_sha256 in file_sha256s:
            raise ValueError("causal video probe embedding receipts must be unique")
        artifact_sha256s.add(artifact_sha256)
        file_sha256s.add(file_sha256)
    manifest = manifest_receipt.get("manifest")
    if not isinstance(manifest, Mapping) or not manifest.get("annotation_files"):
        raise ValueError("embedding receipts require training annotation provenance")


def _verify_outer_folds(
    value: object,
    *,
    plan: Mapping[str, Any],
    source_groups: Mapping[str, Any],
    expected_bundle_by_source: Mapping[str, str],
) -> tuple[list[dict[str, object]], bool]:
    if not isinstance(value, list) or len(value) != len(EXPECTED_GAME_ORDER):
        raise ValueError("causal video probe requires exactly four outer game folds")
    game_rows = {str(row["game_id"]): row for row in source_groups["games"]}
    game_predictions: dict[str, list[dict[str, object]]] = {}
    all_keys: set[tuple[str, str, str]] = set()
    disjoint_flags: list[bool] = []
    for raw_fold, held_game in zip(value, EXPECTED_GAME_ORDER, strict=True):
        if not isinstance(raw_fold, Mapping) or set(raw_fold) != _FOLD_FIELDS:
            raise ValueError("causal video probe outer fold fields are invalid")
        fold = dict(raw_fold)
        held_source = str(game_rows[held_game]["source_video_sha256"])
        held_family = str(game_rows[held_game]["production_family"])
        fit_games = [game for game in EXPECTED_GAME_ORDER if game != held_game]
        if (
            fold.get("held_game_id") != held_game
            or fold.get("held_source_video_sha256") != held_source
            or fold.get("held_production_family") != held_family
            or fold.get("fit_game_ids") != fit_games
            or fold.get("inner_selection_game_ids") != fit_games
        ):
            raise ValueError("outer held game must be disjoint from fit and selection games")
        expected_family_disjoint = held_family not in {str(game_rows[game]["production_family"]) for game in fit_games}
        if fold.get("production_family_disjoint") is not expected_family_disjoint:
            raise ValueError("outer fold production-family disjoint flag is invalid")
        disjoint_flags.append(expected_family_disjoint)
        selected_settings = fold.get("selected_settings")
        threshold = _verify_selected_settings_shape(selected_settings)
        predictions_value = fold.get("oof_predictions")
        expected_positive, expected_negative = EXPECTED_GAME_COUNTS[held_game]
        expected_rows = expected_positive + expected_negative
        if not isinstance(predictions_value, list) or len(predictions_value) != expected_rows:
            raise ValueError("outer fold prediction count is invalid")
        predictions: list[dict[str, object]] = []
        event_ids: list[str] = []
        for raw_prediction in predictions_value:
            if not isinstance(raw_prediction, Mapping) or set(raw_prediction) != _PREDICTION_FIELDS:
                raise ValueError("outer fold prediction fields are invalid")
            prediction = dict(raw_prediction)
            key = _example_key(prediction)
            if key[0] != held_source or key[1] != expected_bundle_by_source.get(held_source) or key in all_keys:
                raise ValueError("outer prediction keys must be unique and held-game bound")
            all_keys.add(key)
            event_ids.append(key[2])
            if not isinstance(prediction.get("event_present"), bool):
                raise ValueError("outer predictions require boolean labels")
            probability = _probability(
                prediction.get("probability"),
                field="outer prediction probability",
            )
            if not isinstance(prediction.get("decision"), bool) or prediction["decision"] is not (
                probability >= threshold
            ):
                raise ValueError("outer prediction decision differs from threshold")
            predictions.append(prediction)
        if event_ids != sorted(event_ids):
            raise ValueError("outer predictions are not in canonical event order")
        positive_count = sum(bool(row["event_present"]) for row in predictions)
        if positive_count != expected_positive or len(predictions) - positive_count != expected_negative:
            raise ValueError("outer predictions do not retain per-game class counts")
        expected_metrics = _metrics_from_prediction_rows(predictions)
        _verify_metric_payload(
            fold.get("held_metrics"),
            expected=expected_metrics,
            field="outer held metrics",
        )
        game_predictions[held_game] = predictions

    if len(all_keys) != EXPECTED_ROW_COUNT:
        raise ValueError("outer folds must contain exactly 45 unique prediction keys")
    for raw_fold, held_game in zip(value, EXPECTED_GAME_ORDER, strict=True):
        fit_games = [game for game in EXPECTED_GAME_ORDER if game != held_game]
        _verify_threshold_selection_result(
            raw_fold["threshold_selection"],
            selected_settings=raw_fold["selected_settings"],
            fit_games=fit_games,
            game_predictions=game_predictions,
            plan=plan,
        )
    predictions = [prediction for game_id in EXPECTED_GAME_ORDER for prediction in game_predictions[game_id]]
    return predictions, all(disjoint_flags)


def _verify_selected_settings_shape(value: object) -> float:
    if not isinstance(value, Mapping) or set(value) != _SELECTED_SETTINGS_FIELDS:
        raise ValueError("outer fold selected settings are invalid")
    backbones = value.get("backbones")
    if (
        not isinstance(value.get("representation_variant"), str)
        or not isinstance(backbones, list)
        or not all(isinstance(backbone, str) for backbone in backbones)
        or type(value.get("input_dimension")) is not int
        or value.get("normalization") != _PREPROCESSING["normalization"]
        or value.get("pca_enabled") is not False
        or value.get("pca_components") is not None
        or type(value.get("regularization_c")) is not float
        or value.get("regularization_c") != REGULARIZATION_C
        or type(value.get("threshold")) is not float
    ):
        raise ValueError("outer fold selected settings types or policy are invalid")
    return _probability(
        value.get("threshold"),
        field="outer selected threshold",
    )


def _verify_threshold_selection_result(
    value: object,
    *,
    selected_settings: Mapping[str, object],
    fit_games: Sequence[str],
    game_predictions: Mapping[str, Sequence[Mapping[str, object]]],
    plan: Mapping[str, Any],
) -> None:
    if not isinstance(value, Mapping) or set(value) != _THRESHOLD_RESULT_FIELDS:
        raise ValueError("outer threshold selection fields are invalid")
    if (
        value.get("protocol") != THRESHOLD_PROTOCOL
        or value.get("selection_metric") != SELECTION_METRIC
        or value.get("tie_break") != SELECTION_TIE_BREAK
    ):
        raise ValueError("outer threshold selection policy or inner folds are invalid")
    _verify_inner_fold_payload(value.get("inner_folds"), fit_games=fit_games)
    expected_training_rows = [row for game_id in fit_games for row in game_predictions[game_id]]
    expected_keys = [_example_key(row) for row in expected_training_rows]
    expected_labels = [bool(row["event_present"]) for row in expected_training_rows]
    variants_value = value.get("variant_inner_oof_predictions")
    if not isinstance(variants_value, list) or len(variants_value) != len(EXPECTED_VARIANTS):
        raise ValueError("outer threshold selection requires both variant predictions")
    probabilities_by_variant: dict[str, list[float]] = {}
    for raw_variant, plan_variant in zip(
        variants_value,
        plan["representation_variants"],
        strict=True,
    ):
        if not isinstance(raw_variant, Mapping) or set(raw_variant) != _VARIANT_INNER_FIELDS:
            raise ValueError("inner variant prediction fields are invalid")
        if (
            raw_variant.get("representation_variant") != plan_variant["name"]
            or type(raw_variant.get("input_dimension")) is not int
            or raw_variant.get("input_dimension") != plan_variant["input_dimension"]
        ):
            raise ValueError("inner variant predictions differ from the frozen plan")
        predictions = raw_variant.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != len(expected_keys):
            raise ValueError("inner variant prediction count is invalid")
        observed_keys = []
        observed_labels = []
        probabilities = []
        for row in predictions:
            if not isinstance(row, Mapping) or set(row) != _INNER_PREDICTION_FIELDS:
                raise ValueError("inner OOF prediction fields are invalid")
            observed_keys.append(_example_key(row))
            if not isinstance(row.get("event_present"), bool):
                raise ValueError("inner OOF predictions require boolean labels")
            observed_labels.append(bool(row["event_present"]))
            probabilities.append(_probability(row.get("probability"), field="inner OOF probability"))
        if observed_keys != expected_keys or observed_labels != expected_labels:
            raise ValueError("inner OOF predictions do not match outer-training rows")
        probabilities_by_variant[str(plan_variant["name"])] = probabilities

    candidate_value = value.get("candidate_results")
    expected_candidate_count = len(EXPECTED_VARIANTS) * len(THRESHOLD_GRID)
    if not isinstance(candidate_value, list) or len(candidate_value) != expected_candidate_count:
        raise ValueError("outer threshold candidate count is invalid")
    verified_candidates: list[dict[str, object]] = []
    position = 0
    target = np.asarray(expected_labels, dtype=np.int64)
    for plan_variant in plan["representation_variants"]:
        variant_name = str(plan_variant["name"])
        probabilities = np.asarray(
            probabilities_by_variant[variant_name],
            dtype=np.float64,
        )
        for threshold in THRESHOLD_GRID:
            raw_candidate = candidate_value[position]
            position += 1
            if not isinstance(raw_candidate, Mapping) or set(raw_candidate) != _CANDIDATE_RESULT_FIELDS:
                raise ValueError("outer threshold candidate fields are invalid")
            if (
                raw_candidate.get("representation_variant") != variant_name
                or type(raw_candidate.get("input_dimension")) is not int
                or raw_candidate.get("input_dimension") != plan_variant["input_dimension"]
                or type(raw_candidate.get("threshold")) is not float
                or raw_candidate.get("threshold") != threshold
            ):
                raise ValueError("outer threshold candidates are not canonically ordered")
            expected_metrics = _binary_metrics(target, probabilities, threshold)
            _verify_metric_payload(
                raw_candidate.get("inner_oof_metrics"),
                expected=expected_metrics,
                field="inner OOF candidate metrics",
            )
            verified_candidates.append(dict(raw_candidate))
    selected = _select_candidate(verified_candidates, plan=plan)
    selected_variant_name = str(selected["representation_variant"])
    selected_variant = next(row for row in plan["representation_variants"] if row["name"] == selected_variant_name)
    expected_settings = {
        "representation_variant": selected_variant_name,
        "backbones": list(selected_variant["backbones"]),
        "input_dimension": selected_variant["input_dimension"],
        "normalization": _PREPROCESSING["normalization"],
        "pca_enabled": False,
        "pca_components": None,
        "regularization_c": REGULARIZATION_C,
        "threshold": selected["threshold"],
    }
    if dict(selected_settings) != expected_settings:
        raise ValueError("outer selected settings do not match inner-only selection")


def _verify_inner_fold_payload(
    value: object,
    *,
    fit_games: Sequence[str],
) -> None:
    expected = _inner_fold_rows(fit_games)
    if not isinstance(value, list) or len(value) != len(expected):
        raise ValueError("outer threshold inner folds are invalid")
    for raw_row, expected_row in zip(value, expected, strict=True):
        if not isinstance(raw_row, Mapping) or set(raw_row) != _INNER_FOLD_FIELDS:
            raise ValueError("outer threshold inner fold fields are invalid")
        if (
            raw_row.get("held_game_id") != expected_row["held_game_id"]
            or raw_row.get("fit_game_ids") != expected_row["fit_game_ids"]
        ):
            raise ValueError("outer threshold inner game split is invalid")
        for field in ("row_count", "positive_count", "negative_count"):
            if type(raw_row.get(field)) is not int or raw_row.get(field) != expected_row[field]:
                raise ValueError("outer threshold inner fold count type is invalid")


def _metrics_from_prediction_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, float | int]:
    return _binary_metrics(
        np.asarray([int(bool(row["event_present"])) for row in rows]),
        np.asarray([float(bool(row["decision"])) for row in rows]),
        0.5,
    )


def _verify_metric_payload(
    value: object,
    *,
    expected: Mapping[str, float | int],
    field: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != _METRIC_FIELDS:
        raise ValueError(f"{field} fields are invalid")
    for name in ("tp", "fp", "fn", "tn", "support", "positive_support", "negative_support"):
        actual = value.get(name)
        if type(actual) is not int or actual < 0 or actual != expected[name]:
            raise ValueError(f"{field} count arithmetic is invalid")
    for name in (
        "precision",
        "recall",
        "f1",
        "specificity",
        "balanced_accuracy",
    ):
        raw_actual = value.get(name)
        if type(raw_actual) is not float:
            raise ValueError(f"{field} {name} must be a float")
        actual = _finite_number(raw_actual, field=f"{field} {name}")
        if not math.isclose(
            actual,
            float(expected[name]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{field} {name} arithmetic is invalid")


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _probability(value: object, *, field: str) -> float:
    if type(value) is not float:
        raise ValueError(f"{field} must be a float")
    number = _finite_number(value, field=field)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be within [0, 1]")
    return number


def _read_json_snapshot(path: Path) -> tuple[dict[str, Any], str, int]:
    with path.open("rb") as handle:
        encoded = handle.read(_MAX_JSON_BYTES + 1)
    if len(encoded) > _MAX_JSON_BYTES:
        raise ValueError(f"JSON input is too large: {path.name}")
    try:
        payload = json.loads(
            encoded,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"expected valid JSON: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    _require_finite_json(payload)
    return payload, hashlib.sha256(encoded).hexdigest(), len(encoded)


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {token}")


def _require_finite_json(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number is forbidden")
    if isinstance(value, Mapping):
        for child in value.values():
            _require_finite_json(child)
    elif isinstance(value, list):
        for child in value:
            _require_finite_json(child)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON object key: {key}")
        payload[key] = value
    return payload


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256", None)
    encoded = json.dumps(
        unsigned,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
