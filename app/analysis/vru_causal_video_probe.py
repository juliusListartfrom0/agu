"""Nested source-held diagnostics for causal-closure video embeddings."""

from __future__ import annotations

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

from app.analysis.shot_validity_finetune import (
    binary_metrics,
    select_precision_threshold,
)
from app.analysis.shot_validity_sampling_window import SHOT_SAMPLING_PROTOCOL
from app.analysis.shot_validity_video_backbone import (
    verify_video_embedding_artifact,
)
from app.analysis.training_annotation import verify_training_annotation_manifest
from app.analysis.vru_causal_training import (
    verify_vru_causal_shot_validity_training_export,
)

VIDEO_PROBE_SCHEMA = "agu.vru-causal-video-representation-probe.v1"
VIDEO_PROBE_PURPOSE = "causal_closure_video_representation_diagnostic_only"
SELECTION_PROTOCOL = "nested_outer_source_held_inner_logo_v1"
THRESHOLD_PROTOCOL = "inner_logo_precision_safe_v1"
REGULARIZATION_C = 0.01
MINIMUM_PRECISION = 0.95
EXPECTED_ROW_COUNT = 22
EXPECTED_SOURCE_COUNT = 3
EXPECTED_POSITIVE_COUNT = 14
EXPECTED_NEGATIVE_COUNT = 8
SOURCE_SELECTION_LIMITATION = (
    "The 24-window closure source was selected with prior-label strata; "
    "these 22 resolved rows support a development diagnostic only, not a "
    "formal generalization estimate."
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PROBE_FIELDS = {
    "artifact_sha256",
    "formal_evaluation_eligible",
    "minimum_precision",
    "negative_count",
    "observed_metrics",
    "outer_folds",
    "positive_count",
    "promoted",
    "promotion_eligible",
    "purpose",
    "regularization_c",
    "representation_dimension",
    "row_count",
    "runtime_consumable",
    "schema_version",
    "selection_protocol",
    "source_count",
    "source_selection",
    "threshold_protocol",
    "training_export_artifact_sha256",
    "training_export_file_sha256",
    "training_manifest_sha256",
    "video_embedding_artifacts",
}
_FOLD_FIELDS = {
    "fit_source_video_sha256s",
    "held_metrics",
    "held_source_video_sha256",
    "inner_selection_source_video_sha256s",
    "oof_predictions",
    "regularization_c",
    "threshold",
    "threshold_selection",
}
_METRIC_FIELDS = {"tp", "fp", "fn", "tn", "precision", "recall", "f1"}


def screen_vru_causal_video_embeddings_nested(
    *,
    embedding_paths: Sequence[Path],
    training_export_path: Path,
    training_manifest_path: Path,
    expected_training_export_sha256: str,
) -> dict[str, object]:
    """Run a fixed-C nested LOGO probe that can never authorize promotion."""

    _require_sha256(
        expected_training_export_sha256,
        field="expected training export artifact",
    )
    if not embedding_paths:
        raise ValueError("at least one video embedding artifact is required")

    export_payload, export_file_sha256, export_size_bytes = _read_json_with_receipt(
        training_export_path
    )
    training_export = verify_vru_causal_shot_validity_training_export(
        export_payload,
        expected_artifact_sha256=expected_training_export_sha256,
    )
    manifest = verify_training_annotation_manifest(_read_json(training_manifest_path))
    if "shot_validity" not in manifest.get("task_types", []):
        raise ValueError("training manifest does not authorize shot_validity")
    export_asset = (
        training_export_path.name,
        export_file_sha256,
        export_size_bytes,
    )
    _verify_manifest_bindings(
        training_export=training_export,
        manifest=manifest,
        export_asset=export_asset,
    )

    embeddings = sorted(
        (
            verify_video_embedding_artifact(_read_json(path))
            for path in embedding_paths
        ),
        key=lambda row: (str(row["backbone"]), str(row["artifact_sha256"])),
    )
    if len({str(row["backbone"]) for row in embeddings}) != len(embeddings):
        raise ValueError("video embedding artifacts require unique backbones")
    if len({str(row["artifact_sha256"]) for row in embeddings}) != len(embeddings):
        raise ValueError("video embedding artifacts must not be duplicated")
    manifest_sha256 = str(manifest["manifest_sha256"])
    if any(
        row.get("training_manifest_sha256") != manifest_sha256
        for row in embeddings
    ):
        raise ValueError("video embedding training manifest does not match the verified manifest")

    export_examples = _strict_examples(
        training_export.get("examples"),
        artifact_name="training export",
    )
    _verify_fixed_closure_shape(export_examples)
    _verify_embedding_provenance(
        embeddings=embeddings,
        training_export=training_export,
        export_examples=export_examples,
    )
    matrices = []
    expected_keys = [_example_key(row) for row in export_examples]
    expected_labels = [bool(row["event_present"]) for row in export_examples]
    for embedding in embeddings:
        embedding_examples = _strict_examples(
            embedding.get("examples"),
            artifact_name=f"video embedding artifact {embedding['backbone']}",
        )
        embedding_lookup = {
            _example_key(row): row for row in embedding_examples
        }
        if set(embedding_lookup) != set(expected_keys):
            raise ValueError(
                "video embedding three-part key set does not match the verified training export"
            )
        aligned_examples = [embedding_lookup[key] for key in expected_keys]
        if [bool(row["event_present"]) for row in aligned_examples] != expected_labels:
            raise ValueError("video embedding labels do not match the verified training export")
        matrices.append(
            np.asarray([row["embedding"] for row in aligned_examples], dtype=np.float64)
        )

    matrix = np.concatenate(matrices, axis=1)
    target = np.asarray(expected_labels, dtype=np.int64)
    groups = np.asarray(
        [str(row["source_video_sha256"]) for row in export_examples],
        dtype=str,
    )
    unique_groups = sorted(set(groups.tolist()))
    outer_probabilities = np.full(len(target), np.nan, dtype=np.float64)
    outer_decisions = np.zeros(len(target), dtype=bool)
    outer_folds = []
    for held_group in unique_groups:
        development_groups = [group for group in unique_groups if group != held_group]
        development_mask = np.isin(groups, development_groups)
        inner_probabilities = np.full(len(target), np.nan, dtype=np.float64)
        inner_folds = []
        for inner_held_group in development_groups:
            inner_held_mask = development_mask & (groups == inner_held_group)
            inner_fit_groups = [
                group for group in development_groups if group != inner_held_group
            ]
            inner_fit_mask = np.isin(groups, inner_fit_groups)
            inner_probabilities[inner_held_mask] = _fit_probability_model(
                matrix=matrix,
                target=target,
                fit_mask=inner_fit_mask,
                predict_mask=inner_held_mask,
            )
            inner_folds.append(
                {
                    "held_source_video_sha256": inner_held_group,
                    "fit_source_video_sha256s": inner_fit_groups,
                    "row_count": int(inner_held_mask.sum()),
                }
            )
        if not np.isfinite(inner_probabilities[development_mask]).all():
            raise ValueError("nested inner OOF predictions are incomplete")
        threshold, threshold_metrics = select_precision_threshold(
            target[development_mask].astype(bool).tolist(),
            inner_probabilities[development_mask].tolist(),
            minimum_precision=MINIMUM_PRECISION,
        )
        held_mask = groups == held_group
        held_probabilities = _fit_probability_model(
            matrix=matrix,
            target=target,
            fit_mask=development_mask,
            predict_mask=held_mask,
        )
        held_decisions = held_probabilities >= threshold
        outer_probabilities[held_mask] = held_probabilities
        outer_decisions[held_mask] = held_decisions
        held_examples = [
            row
            for row, is_held in zip(export_examples, held_mask.tolist(), strict=True)
            if is_held
        ]
        outer_folds.append(
            {
                "held_source_video_sha256": held_group,
                "inner_selection_source_video_sha256s": development_groups,
                "fit_source_video_sha256s": development_groups,
                "regularization_c": REGULARIZATION_C,
                "threshold": float(threshold),
                "threshold_selection": {
                    "protocol": THRESHOLD_PROTOCOL,
                    "minimum_precision": MINIMUM_PRECISION,
                    "precision_requirement_met": (
                        float(threshold_metrics["precision"]) >= MINIMUM_PRECISION
                    ),
                    "inner_oof_metrics": threshold_metrics,
                    "inner_folds": inner_folds,
                },
                "held_metrics": binary_metrics(
                    target[held_mask].astype(bool).tolist(),
                    held_probabilities.tolist(),
                    float(threshold),
                ),
                "oof_predictions": [
                    {
                        "source_video_sha256": str(row["source_video_sha256"]),
                        "candidate_bundle_sha256": str(row["candidate_bundle_sha256"]),
                        "event_id": str(row["event_id"]),
                        "probability": float(probability),
                        "decision": bool(decision),
                    }
                    for row, probability, decision in zip(
                        held_examples,
                        held_probabilities,
                        held_decisions,
                        strict=True,
                    )
                ],
            }
        )

    if not np.isfinite(outer_probabilities).all():
        raise ValueError("nested outer OOF predictions are incomplete")
    source_selection = {
        "prior_stratified": True,
        "limitation": SOURCE_SELECTION_LIMITATION,
    }
    payload: dict[str, object] = {
        "schema_version": VIDEO_PROBE_SCHEMA,
        "purpose": VIDEO_PROBE_PURPOSE,
        "runtime_consumable": False,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
        "selection_protocol": SELECTION_PROTOCOL,
        "threshold_protocol": THRESHOLD_PROTOCOL,
        "regularization_c": REGULARIZATION_C,
        "minimum_precision": MINIMUM_PRECISION,
        "source_selection": source_selection,
        "training_export_artifact_sha256": training_export["artifact_sha256"],
        "training_export_file_sha256": export_file_sha256,
        "training_manifest_sha256": manifest_sha256,
        "video_embedding_artifacts": [
            {
                "backbone": row["backbone"],
                "artifact_sha256": row["artifact_sha256"],
                "backbone_sha256": row["backbone_sha256"],
                "embedding_dimension": row["embedding_dimension"],
            }
            for row in embeddings
        ],
        "representation_dimension": int(matrix.shape[1]),
        "row_count": len(target),
        "source_count": len(unique_groups),
        "positive_count": int(target.sum()),
        "negative_count": int((target == 0).sum()),
        "outer_folds": outer_folds,
        "observed_metrics": {
            "pooled": binary_metrics(
                target.astype(bool).tolist(),
                outer_decisions.astype(float).tolist(),
                0.5,
            ),
            "per_source": {
                group: binary_metrics(
                    target[groups == group].astype(bool).tolist(),
                    outer_decisions[groups == group].astype(float).tolist(),
                    0.5,
                )
                for group in unique_groups
            },
        },
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return payload


def verify_vru_causal_video_probe(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the immutable diagnostic-only probe artifact."""

    if set(payload) != _PROBE_FIELDS:
        raise ValueError("causal video probe fields are not canonical")
    artifact = dict(payload)
    claimed_sha256 = _require_sha256(
        artifact.pop("artifact_sha256", None),
        field="causal video probe artifact",
    )
    if claimed_sha256 != _canonical_sha256(artifact):
        raise ValueError("causal video probe artifact hash mismatch")
    if artifact.get("schema_version") != VIDEO_PROBE_SCHEMA:
        raise ValueError("unsupported causal video probe schema")
    if artifact.get("purpose") != VIDEO_PROBE_PURPOSE:
        raise ValueError("causal video probe purpose is invalid")
    if artifact.get("runtime_consumable") is not False:
        raise ValueError("causal video probe must not be runtime-consumable")
    if artifact.get("formal_evaluation_eligible") is not False:
        raise ValueError("causal video probe is not eligible for formal evaluation")
    if artifact.get("promotion_eligible") is not False or artifact.get("promoted") is not False:
        raise ValueError("causal video probe can never authorize promotion")
    if artifact.get("selection_protocol") != SELECTION_PROTOCOL:
        raise ValueError("unsupported causal video probe selection protocol")
    if artifact.get("threshold_protocol") != THRESHOLD_PROTOCOL:
        raise ValueError("unsupported causal video probe threshold protocol")
    if artifact.get("regularization_c") != REGULARIZATION_C:
        raise ValueError("causal video probe regularization C is not frozen")
    if artifact.get("minimum_precision") != MINIMUM_PRECISION:
        raise ValueError("causal video probe minimum precision policy is not frozen")
    expected_counts = {
        "row_count": EXPECTED_ROW_COUNT,
        "source_count": EXPECTED_SOURCE_COUNT,
        "positive_count": EXPECTED_POSITIVE_COUNT,
        "negative_count": EXPECTED_NEGATIVE_COUNT,
    }
    if any(artifact.get(field) != expected for field, expected in expected_counts.items()):
        raise ValueError("causal video probe must retain the exact 22-row, three-source, 14/8 shape")
    for field in (
        "training_export_artifact_sha256",
        "training_export_file_sha256",
        "training_manifest_sha256",
    ):
        _require_sha256(artifact.get(field), field=field.replace("_", " "))
    source_selection = artifact.get("source_selection")
    if (
        not isinstance(source_selection, Mapping)
        or set(source_selection) != {"prior_stratified", "limitation"}
        or source_selection.get("prior_stratified") is not True
        or source_selection.get("limitation") != SOURCE_SELECTION_LIMITATION
    ):
        raise ValueError("causal video probe must disclose prior-stratified selection")
    _verify_video_embedding_receipts(
        artifact.get("video_embedding_artifacts"),
        representation_dimension=artifact.get("representation_dimension"),
    )
    fold_metrics = _verify_outer_folds(artifact.get("outer_folds"))
    _verify_observed_metrics(
        artifact.get("observed_metrics"),
        fold_metrics=fold_metrics,
    )
    artifact["artifact_sha256"] = claimed_sha256
    return artifact


def _verify_video_embedding_receipts(
    value: object,
    *,
    representation_dimension: object,
) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError("causal video probe requires embedding artifact receipts")
    receipts = []
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {
            "artifact_sha256",
            "backbone",
            "backbone_sha256",
            "embedding_dimension",
        }:
            raise ValueError("video embedding artifact receipt is invalid")
        backbone = row.get("backbone")
        if not isinstance(backbone, str) or not backbone:
            raise ValueError("video embedding artifact backbone is missing")
        artifact_sha256 = _require_sha256(
            row.get("artifact_sha256"),
            field="video embedding artifact",
        )
        _require_sha256(
            row.get("backbone_sha256"),
            field="video embedding backbone checkpoint",
        )
        dimension = _positive_int(
            row.get("embedding_dimension"),
            field="video embedding dimension",
        )
        receipts.append((backbone, artifact_sha256, dimension))
    if len(receipts) != len({backbone for backbone, _sha, _dim in receipts}):
        raise ValueError("video embedding artifact backbones must be unique")
    if len(receipts) != len({sha for _backbone, sha, _dim in receipts}):
        raise ValueError("video embedding artifact SHAs must be unique")
    if receipts != sorted(receipts):
        raise ValueError("video embedding artifact receipts are not canonically ordered")
    if _positive_int(
        representation_dimension,
        field="video representation dimension",
    ) != sum(dimension for _backbone, _sha, dimension in receipts):
        raise ValueError("video representation dimension does not match its receipts")


def _verify_outer_folds(value: object) -> dict[str, dict[str, float | int]]:
    if not isinstance(value, list) or len(value) != EXPECTED_SOURCE_COUNT:
        raise ValueError("causal video probe requires exactly three outer source-held folds")
    held_sources = []
    for fold in value:
        if not isinstance(fold, Mapping) or set(fold) != _FOLD_FIELDS:
            raise ValueError("causal video probe outer fold fields are invalid")
        held_sources.append(
            _require_sha256(
                fold.get("held_source_video_sha256"),
                field="outer held source",
            )
        )
    if held_sources != sorted(set(held_sources)):
        raise ValueError("causal video probe outer held sources must be unique and ordered")

    held_metrics: dict[str, dict[str, float | int]] = {}
    prediction_keys: set[tuple[str, str, str]] = set()
    held_row_counts: dict[str, int] = {}
    inner_rows_by_fold: list[tuple[list[Mapping[str, object]], set[str]]] = []
    held_source_set = set(held_sources)
    for fold, held_source in zip(value, held_sources, strict=True):
        expected_fit_sources = sorted(held_source_set - {held_source})
        fit_sources = fold.get("fit_source_video_sha256s")
        inner_selection_sources = fold.get("inner_selection_source_video_sha256s")
        if fit_sources != expected_fit_sources or inner_selection_sources != expected_fit_sources:
            raise ValueError("outer held source must be disjoint from selection and fit sources")
        if fold.get("regularization_c") != REGULARIZATION_C:
            raise ValueError("outer fold regularization C is not frozen")
        threshold = _probability(fold.get("threshold"), field="outer fold threshold")
        predictions = fold.get("oof_predictions")
        if not isinstance(predictions, list) or not predictions:
            raise ValueError("outer fold requires held OOF predictions")
        for prediction in predictions:
            if not isinstance(prediction, Mapping) or set(prediction) != {
                "candidate_bundle_sha256",
                "decision",
                "event_id",
                "probability",
                "source_video_sha256",
            }:
                raise ValueError("outer OOF prediction fields are invalid")
            if prediction.get("source_video_sha256") != held_source:
                raise ValueError("outer OOF prediction is not from its held source")
            bundle_sha256 = _require_sha256(
                prediction.get("candidate_bundle_sha256"),
                field="outer OOF candidate bundle",
            )
            event_id = prediction.get("event_id")
            if not isinstance(event_id, str) or not event_id:
                raise ValueError("outer OOF prediction event ID is invalid")
            key = (held_source, bundle_sha256, event_id)
            if key in prediction_keys:
                raise ValueError("outer OOF prediction three-part keys must be unique")
            prediction_keys.add(key)
            probability = _probability(
                prediction.get("probability"),
                field="outer OOF probability",
            )
            decision = prediction.get("decision")
            if not isinstance(decision, bool) or decision != (probability >= threshold):
                raise ValueError("outer OOF decision does not match its fold threshold")
        held_row_counts[held_source] = len(predictions)
        metrics = _verify_binary_metrics(
            fold.get("held_metrics"),
            expected_rows=len(predictions),
            field="outer held metrics",
        )
        predicted_positive_count = sum(
            int(prediction["decision"]) for prediction in predictions
        )
        predicted_negative_count = len(predictions) - predicted_positive_count
        if (
            int(metrics["tp"]) + int(metrics["fp"]) != predicted_positive_count
            or int(metrics["fn"]) + int(metrics["tn"]) != predicted_negative_count
        ):
            raise ValueError(
                "outer OOF decision counts contradict held confusion metrics"
            )
        held_metrics[held_source] = metrics
        inner_folds = _verify_threshold_selection(
            fold.get("threshold_selection"),
            development_sources=set(expected_fit_sources),
        )
        inner_rows_by_fold.append((inner_folds, set(expected_fit_sources)))

    if len(prediction_keys) != EXPECTED_ROW_COUNT:
        raise ValueError("causal video probe must contain exactly 22 unique OOF predictions")
    positive_count = sum(
        int(metrics["tp"]) + int(metrics["fn"])
        for metrics in held_metrics.values()
    )
    negative_count = sum(
        int(metrics["fp"]) + int(metrics["tn"])
        for metrics in held_metrics.values()
    )
    if positive_count != EXPECTED_POSITIVE_COUNT or negative_count != EXPECTED_NEGATIVE_COUNT:
        raise ValueError("outer held metrics do not retain the 14/8 label shape")
    for inner_folds, development_sources in inner_rows_by_fold:
        for inner_fold in inner_folds:
            inner_held = str(inner_fold["held_source_video_sha256"])
            if int(inner_fold["row_count"]) != held_row_counts[inner_held]:
                raise ValueError("inner LOGO row count does not match its held source")
        if {str(row["held_source_video_sha256"]) for row in inner_folds} != development_sources:
            raise ValueError("inner LOGO folds do not exactly cover outer-training sources")
    return held_metrics


def _verify_threshold_selection(
    value: object,
    *,
    development_sources: set[str],
) -> list[Mapping[str, object]]:
    if not isinstance(value, Mapping) or set(value) != {
        "inner_folds",
        "inner_oof_metrics",
        "minimum_precision",
        "precision_requirement_met",
        "protocol",
    }:
        raise ValueError("outer fold threshold selection fields are invalid")
    if (
        value.get("protocol") != THRESHOLD_PROTOCOL
        or value.get("minimum_precision") != MINIMUM_PRECISION
    ):
        raise ValueError("outer fold threshold policy is not frozen")
    inner_folds = value.get("inner_folds")
    if not isinstance(inner_folds, list) or len(inner_folds) != len(development_sources):
        raise ValueError("outer fold requires exact inner LOGO folds")
    total_rows = 0
    for inner_fold in inner_folds:
        if not isinstance(inner_fold, Mapping) or set(inner_fold) != {
            "fit_source_video_sha256s",
            "held_source_video_sha256",
            "row_count",
        }:
            raise ValueError("inner LOGO fold fields are invalid")
        inner_held = _require_sha256(
            inner_fold.get("held_source_video_sha256"),
            field="inner held source",
        )
        expected_fit = sorted(development_sources - {inner_held})
        if inner_held not in development_sources or inner_fold.get("fit_source_video_sha256s") != expected_fit:
            raise ValueError("inner held source must be disjoint from its fit sources")
        total_rows += _positive_int(inner_fold.get("row_count"), field="inner fold row count")
    metrics = _verify_binary_metrics(
        value.get("inner_oof_metrics"),
        expected_rows=total_rows,
        field="inner OOF metrics",
    )
    if value.get("precision_requirement_met") is not (
        float(metrics["precision"]) >= MINIMUM_PRECISION
    ):
        raise ValueError("inner precision requirement flag is inconsistent")
    return inner_folds


def _verify_observed_metrics(
    value: object,
    *,
    fold_metrics: Mapping[str, Mapping[str, float | int]],
) -> None:
    if not isinstance(value, Mapping) or set(value) != {"per_source", "pooled"}:
        raise ValueError("causal video probe observed metrics are invalid")
    per_source = value.get("per_source")
    if not isinstance(per_source, Mapping) or set(per_source) != set(fold_metrics):
        raise ValueError("observed per-source metrics do not match outer folds")
    for source, metrics in fold_metrics.items():
        if per_source[source] != metrics:
            raise ValueError("observed per-source metrics differ from held metrics")
    pooled_counts = {
        name: sum(int(metrics[name]) for metrics in fold_metrics.values())
        for name in ("tp", "fp", "fn", "tn")
    }
    pooled = _verify_binary_metrics(
        value.get("pooled"),
        expected_rows=EXPECTED_ROW_COUNT,
        field="pooled observed metrics",
    )
    if any(int(pooled[name]) != count for name, count in pooled_counts.items()):
        raise ValueError("pooled observed metrics do not aggregate outer folds")


def _verify_binary_metrics(
    value: object,
    *,
    expected_rows: int,
    field: str,
) -> dict[str, float | int]:
    if not isinstance(value, Mapping) or set(value) != _METRIC_FIELDS:
        raise ValueError(f"{field} fields are invalid")
    counts = {}
    for name in ("tp", "fp", "fn", "tn"):
        raw = value.get(name)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(f"{field} counts are invalid")
        counts[name] = raw
    if sum(counts.values()) != expected_rows:
        raise ValueError(f"{field} row count is invalid")
    expected_precision = (
        counts["tp"] / (counts["tp"] + counts["fp"])
        if counts["tp"] + counts["fp"]
        else 0.0
    )
    expected_recall = (
        counts["tp"] / (counts["tp"] + counts["fn"])
        if counts["tp"] + counts["fn"]
        else 0.0
    )
    expected_f1 = (
        2 * expected_precision * expected_recall / (expected_precision + expected_recall)
        if expected_precision + expected_recall
        else 0.0
    )
    for name, expected in (
        ("precision", expected_precision),
        ("recall", expected_recall),
        ("f1", expected_f1),
    ):
        actual = _finite_number(value.get(name), field=f"{field} {name}")
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{field} {name} is inconsistent with its counts")
    return dict(value)


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _probability(value: object, *, field: str) -> float:
    number = _finite_number(value, field=field)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be within [0, 1]")
    return number


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
        raise ValueError("nested model fit and prediction splits must be non-empty and disjoint")
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
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("nested model produced invalid probabilities")
    return probabilities


def _verify_embedding_provenance(
    *,
    embeddings: Sequence[Mapping[str, object]],
    training_export: Mapping[str, object],
    export_examples: Sequence[Mapping[str, object]],
) -> None:
    expected_source_sha256s = sorted(
        {str(row["source_video_sha256"]) for row in export_examples}
    )
    expected_annotation_sha256s = sorted(
        str(source["labels"]["sha256"])
        for source in training_export["sources"]
    )
    for embedding in embeddings:
        if (
            embedding.get("purpose") != "backbone_screening_training_only"
            or embedding.get("producer") != "agu"
        ):
            raise ValueError("video embedding producer provenance is invalid")
        if embedding.get("training_annotation_sha256") != expected_annotation_sha256s:
            raise ValueError(
                "video embedding annotation provenance does not match the training export"
            )
        if embedding.get("source_video_sha256") != expected_source_sha256s:
            raise ValueError(
                "video embedding source provenance does not match the training export"
            )
        if embedding.get("sampling_protocol") != SHOT_SAMPLING_PROTOCOL:
            raise ValueError("video embedding sampling provenance is required")
        _require_sha256(
            embedding.get("backbone_sha256"),
            field="video embedding backbone checkpoint",
        )
        for row in embedding["examples"]:
            sampling_window = row.get("sampling_window")
            if not isinstance(sampling_window, Mapping):
                raise ValueError("video embedding requires a sampling window")
            if (
                sampling_window.get("anchor_source") != "bounded_event"
                or sampling_window.get("anchor_frame")
                != sampling_window.get("end_frame")
            ):
                raise ValueError(
                    "causal closure embeddings require label-hidden bounded-event sampling"
                )


def _verify_manifest_bindings(
    *,
    training_export: Mapping[str, object],
    manifest: Mapping[str, object],
    export_asset: tuple[str, str, int],
) -> None:
    expected_sources = {
        (
            str(source["source_video_filename"]),
            str(source["source_video_sha256"]),
            int(source["source_video_size_bytes"]),
        )
        for source in training_export["sources"]
    }
    manifest_sources = _manifest_asset_tuples(manifest.get("source_videos"))
    if manifest_sources != expected_sources:
        raise ValueError(
            "training manifest source assets do not exactly match the training export"
        )
    expected_annotations = {
        export_asset,
        *(
            (
                str(source["labels"]["filename"]),
                str(source["labels"]["sha256"]),
                int(source["labels"]["size_bytes"]),
            )
            for source in training_export["sources"]
        ),
    }
    manifest_annotations = _manifest_asset_tuples(
        manifest.get("annotation_files")
    )
    if export_asset not in manifest_annotations:
        raise ValueError("training export file is not SHA-bound by the training manifest")
    if manifest_annotations != expected_annotations:
        raise ValueError(
            "training manifest annotation assets do not exactly match export labels"
        )


def _manifest_asset_tuples(value: object) -> set[tuple[str, str, int]]:
    if not isinstance(value, list):
        raise ValueError("training manifest assets must be a list")
    rows = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("training manifest assets must be objects")
        rows.append(
            (
                str(item.get("filename")),
                str(item.get("sha256")),
                int(item.get("size_bytes", -1)),
            )
        )
    if len(rows) != len(set(rows)):
        raise ValueError("training manifest assets must be unique")
    return set(rows)


def _strict_examples(
    rows: object,
    *,
    artifact_name: str,
) -> list[Mapping[str, object]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{artifact_name} requires examples")
    examples = []
    keys = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{artifact_name} examples must be objects")
        key = _example_key(row)
        if not isinstance(row.get("event_present"), bool):
            raise ValueError(f"{artifact_name} examples require boolean labels")
        examples.append(row)
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{artifact_name} contains duplicate three-part keys")
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
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("example event ID must be a non-empty string")
    return source_sha256, bundle_sha256, event_id


def _verify_fixed_closure_shape(examples: Sequence[Mapping[str, object]]) -> None:
    groups = {str(row["source_video_sha256"]) for row in examples}
    positive_count = sum(bool(row["event_present"]) for row in examples)
    if (
        len(examples) != EXPECTED_ROW_COUNT
        or len(groups) != EXPECTED_SOURCE_COUNT
        or positive_count != EXPECTED_POSITIVE_COUNT
        or len(examples) - positive_count != EXPECTED_NEGATIVE_COUNT
    ):
        raise ValueError("video probe requires the exact 22-row, three-source, 14/8 closure export")
    for group in groups:
        labels = {
            bool(row["event_present"])
            for row in examples
            if row["source_video_sha256"] == group
        }
        if labels != {False, True}:
            raise ValueError("every closure source must contain both binary classes")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return payload


def _read_json_with_receipt(path: Path) -> tuple[dict[str, Any], str, int]:
    encoded = path.read_bytes()
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"expected valid JSON: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return payload, hashlib.sha256(encoded).hexdigest(), len(encoded)


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
