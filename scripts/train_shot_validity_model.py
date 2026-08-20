#!/usr/bin/env python3
"""Train a sealed traditional real-shot gate from disjoint annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.official_evaluation import verify_raw_only_bundle  # noqa: E402
from app.analysis.schemas import GameEventResponse  # noqa: E402
from app.analysis.shot_validity import (  # noqa: E402
    SHOT_VALIDITY_FEATURES,
    SHOT_VALIDITY_LABEL_SCHEMA,
    extract_shot_validity_features,
    seal_shot_validity_model,
)
from app.analysis.training_annotation import (  # noqa: E402
    verify_training_annotation_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-bundle", type=Path, nargs="+", required=True)
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tree-count", type=int, default=200)
    parser.add_argument("--tree-max-depth", type=int, default=6)
    parser.add_argument("--tree-min-samples-leaf", type=int, default=4)
    parser.add_argument("--minimum-precision", type=float, default=0.95)
    parser.add_argument("--minimum-per-video-recall", type=float, default=0.85)
    return parser.parse_args()


def train_shot_validity_model(
    *,
    manifest_path: Path,
    candidate_bundle_paths: list[Path],
    annotation_paths: list[Path],
    tree_count: int = 200,
    tree_max_depth: int = 6,
    tree_min_samples_leaf: int = 4,
    minimum_precision: float = 0.95,
    minimum_per_video_recall: float = 0.85,
) -> dict[str, Any]:
    if min(tree_count, tree_max_depth, tree_min_samples_leaf) <= 0:
        raise ValueError("Extra Trees parameters must be positive")
    if not 0 < minimum_precision <= 1:
        raise ValueError("minimum_precision must be in (0,1]")
    if not 0 < minimum_per_video_recall <= 1:
        raise ValueError("minimum_per_video_recall must be in (0,1]")
    manifest = verify_training_annotation_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    if "shot_validity" not in manifest.get("task_types", []):
        raise ValueError("training manifest does not authorize shot_validity")
    allowed_sources = {
        str(item["sha256"]) for item in manifest.get("source_videos", [])
    }
    allowed_annotations = {
        (str(item["filename"]), str(item["sha256"]))
        for item in manifest.get("annotation_files", [])
    }

    candidates: dict[tuple[str, str, str], GameEventResponse] = {}
    bundle_hashes: set[tuple[str, str]] = set()
    for path in candidate_bundle_paths:
        bundle = verify_raw_only_bundle(json.loads(path.read_text(encoding="utf-8")))
        provenance = {
            str(key).strip().lower(): str(value).strip().lower()
            for key, value in bundle.model_provenance.items()
        }
        if (
            provenance.get("traditional_feature_training_eligible") == "false"
            or provenance.get("training_geometry") == "label_hidden_window_only"
        ):
            raise ValueError(
                "candidate bundle is not eligible for traditional feature training: "
                f"{path.name}"
            )
        if len(bundle.raw_videos) != 1:
            raise ValueError("shot-validity training bundle must bind exactly one raw video")
        source_hash = bundle.raw_videos[0].sha256
        if source_hash not in allowed_sources:
            raise ValueError(f"candidate source is not bound by training manifest: {path.name}")
        bundle_key = (source_hash, bundle.bundle_sha256)
        if bundle_key in bundle_hashes:
            raise ValueError("duplicate candidate bundle")
        bundle_hashes.add(bundle_key)
        for event in bundle.events:
            if event.event_type == "field_goal_attempt":
                candidates[(source_hash, bundle.bundle_sha256, event.event_id)] = event

    rows: list[list[float]] = []
    labels: list[int] = []
    groups: list[str] = []
    seen_examples: set[tuple[str, str, str]] = set()
    annotation_hashes: list[str] = []
    for path in annotation_paths:
        digest = _file_sha256(path)
        if (path.name, digest) not in allowed_annotations:
            raise ValueError(f"annotation is not SHA-bound by training manifest: {path.name}")
        annotation_hashes.append(digest)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SHOT_VALIDITY_LABEL_SCHEMA:
            raise ValueError(f"unsupported shot validity labels: {path.name}")
        source_hash = str(payload.get("source_video_sha256") or "")
        if source_hash not in allowed_sources:
            raise ValueError(f"annotation source is not bound by training manifest: {path.name}")
        bundle_hash = str(payload.get("candidate_bundle_sha256") or "")
        if (source_hash, bundle_hash) not in bundle_hashes:
            raise ValueError(f"annotation candidate bundle hash mismatch: {path.name}")
        for example in payload.get("examples", []):
            event_id = str(example.get("event_id") or "")
            key = (source_hash, bundle_hash, event_id)
            if key in seen_examples:
                raise ValueError(f"duplicate labeled candidate: {event_id}")
            if key not in candidates:
                raise ValueError(f"labeled candidate is absent from sealed bundle: {event_id}")
            event_present = example.get("event_present")
            if not isinstance(event_present, bool):
                raise ValueError(f"event_present must be boolean: {event_id}")
            features = extract_shot_validity_features(candidates[key])
            rows.append([features[name] for name in SHOT_VALIDITY_FEATURES])
            labels.append(int(event_present))
            groups.append(source_hash)
            seen_examples.add(key)

    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        raise ValueError("leave-one-video-out calibration requires at least two games")
    target = np.asarray(labels, dtype=np.int64)
    matrix = np.asarray(rows, dtype=np.float64)
    if len(set(labels)) < 2:
        raise ValueError("shot-validity training requires both positive and negative labels")

    oof_probabilities = np.zeros(len(target), dtype=np.float64)
    for held_group in unique_groups:
        train_indexes = np.asarray(
            [index for index, group in enumerate(groups) if group != held_group]
        )
        held_indexes = np.asarray(
            [index for index, group in enumerate(groups) if group == held_group]
        )
        if len(set(target[train_indexes].tolist())) < 2:
            raise ValueError(
                "each leave-one-video-out training fold needs positive and negative labels"
            )
        fold = _classifier(
            tree_count=tree_count,
            max_depth=tree_max_depth,
            min_samples_leaf=tree_min_samples_leaf,
        ).fit(matrix[train_indexes], target[train_indexes])
        oof_probabilities[held_indexes] = _positive_probabilities(
            fold, matrix[held_indexes]
        )

    threshold, calibration = _calibrate_group_threshold(
        target,
        oof_probabilities,
        np.asarray(groups),
        minimum_precision=minimum_precision,
        minimum_per_video_recall=minimum_per_video_recall,
    )
    ranking = {
        group: {
            "roc_auc": float(roc_auc_score(target[np.asarray(groups) == group], oof_probabilities[np.asarray(groups) == group])),
            "average_precision": float(average_precision_score(target[np.asarray(groups) == group], oof_probabilities[np.asarray(groups) == group])),
            "best_precision_at_recall_0_85": _best_precision_at_recall(
                target[np.asarray(groups) == group],
                oof_probabilities[np.asarray(groups) == group],
                minimum_recall=0.85,
            ),
        }
        for group in unique_groups
    }
    classifier = _classifier(
        tree_count=tree_count,
        max_depth=tree_max_depth,
        min_samples_leaf=tree_min_samples_leaf,
    ).fit(matrix, target)
    return seal_shot_validity_model(
        {
            "training_manifest_sha256": manifest["manifest_sha256"],
            "training_annotation_producer": str(manifest["producer"]),
            "training_benchmark_overlap": bool(manifest["benchmark_overlap"]),
            "training_annotation_sha256": sorted(annotation_hashes),
            "threshold": threshold,
            "tree_count": tree_count,
            "max_depth": tree_max_depth,
            "min_samples_leaf": tree_min_samples_leaf,
            "class_weight": "balanced",
            "random_state": 0,
            "trees": [_serialize_tree(estimator) for estimator in classifier.estimators_],
            "metrics": {
                "training_rows": len(rows),
                "training_video_count": len(unique_groups),
                "positive_count": int(target.sum()),
                "negative_count": int(len(target) - target.sum()),
                "calibration": "leave_one_video_out",
                "minimum_precision": minimum_precision,
                "leave_one_video_out_precision": calibration["pooled"]["precision"],
                "leave_one_video_out_recall": calibration["pooled"]["recall"],
                "leave_one_video_out_f1": calibration["pooled"]["f1"],
                "per_video": calibration["per_video"],
                "per_video_ranking": ranking,
                "minimum_per_video_recall": minimum_per_video_recall,
                "minimum_precision_gate": calibration["promoted"],
                "promoted": calibration["promoted"],
            },
        }
    )


def _classifier(
    *, tree_count: int, max_depth: int, min_samples_leaf: int
) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=tree_count,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        max_features=None,
        random_state=0,
        n_jobs=-1,
    )


def _positive_probabilities(
    classifier: ExtraTreesClassifier, matrix: np.ndarray
) -> np.ndarray:
    index = list(classifier.classes_).index(1)
    return classifier.predict_proba(matrix)[:, index]


def _calibrate_threshold(
    truth: np.ndarray, probabilities: np.ndarray, *, minimum_precision: float
) -> tuple[float, float, float, float]:
    candidates = sorted({0.0, 1.0, *(float(value) for value in probabilities)})
    eligible: list[tuple[float, float, float, float]] = []
    for threshold in candidates:
        predicted = (probabilities >= threshold).astype(np.int64)
        precision = float(precision_score(truth, predicted, zero_division=0))
        recall = float(recall_score(truth, predicted, zero_division=0))
        f1 = float(f1_score(truth, predicted, zero_division=0))
        if precision >= minimum_precision:
            eligible.append((threshold, precision, recall, f1))
    if not eligible:
        return 1.0, 0.0, 0.0, 0.0
    return max(eligible, key=lambda item: (item[2], item[3], -item[0]))


def _calibrate_group_threshold(
    truth: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    *,
    minimum_precision: float,
    minimum_per_video_recall: float,
) -> tuple[float, dict[str, Any]]:
    eligible: list[tuple[float, float, float, dict[str, Any]]] = []
    for threshold in sorted({0.0, 1.0, *(float(value) for value in probabilities)}):
        pooled = _metrics_at_threshold(truth, probabilities, threshold)
        per_video = {
            str(group): _metrics_at_threshold(
                truth[groups == group], probabilities[groups == group], threshold
            )
            for group in sorted(set(groups.tolist()))
        }
        promoted = (
            pooled["precision"] >= minimum_precision
            and all(row["precision"] >= minimum_precision for row in per_video.values())
            and all(
                row["recall"] >= minimum_per_video_recall
                for row in per_video.values()
            )
        )
        if promoted:
            result = {"promoted": True, "pooled": pooled, "per_video": per_video}
            eligible.append((pooled["recall"], pooled["f1"], -threshold, result))
    if eligible:
        _recall, _f1, negative_threshold, result = max(eligible)
        return -negative_threshold, result
    threshold = 1.0
    return threshold, {
        "promoted": False,
        "pooled": _metrics_at_threshold(truth, probabilities, threshold),
        "per_video": {
            str(group): _metrics_at_threshold(
                truth[groups == group], probabilities[groups == group], threshold
            )
            for group in sorted(set(groups.tolist()))
        },
    }


def _metrics_at_threshold(
    truth: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    predicted = (probabilities >= threshold).astype(np.int64)
    return {
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
    }


def _best_precision_at_recall(
    truth: np.ndarray, probabilities: np.ndarray, *, minimum_recall: float
) -> dict[str, float]:
    candidates = []
    for threshold in sorted({0.0, 1.0, *(float(value) for value in probabilities)}):
        metrics = _metrics_at_threshold(truth, probabilities, threshold)
        if metrics["recall"] >= minimum_recall:
            candidates.append(
                (metrics["precision"], metrics["f1"], threshold, metrics)
            )
    if not candidates:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "threshold": 1.0}
    _precision, _f1, threshold, metrics = max(candidates)
    return {**metrics, "threshold": threshold}


def _serialize_tree(estimator: Any) -> dict[str, list[Any]]:
    tree = estimator.tree_
    positive_class_index = list(estimator.classes_).index(1)
    probabilities = []
    for node_values in tree.value:
        values = node_values[0]
        total = float(np.sum(values))
        probabilities.append(
            float(values[positive_class_index] / total) if total else 0.0
        )
    return {
        "children_left": tree.children_left.astype(int).tolist(),
        "children_right": tree.children_right.astype(int).tolist(),
        "feature": tree.feature.astype(int).tolist(),
        "threshold": tree.threshold.astype(float).tolist(),
        "positive_probability": probabilities,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    artifact = train_shot_validity_model(
        manifest_path=args.manifest,
        candidate_bundle_paths=args.candidate_bundle,
        annotation_paths=args.annotations,
        tree_count=args.tree_count,
        tree_max_depth=args.tree_max_depth,
        tree_min_samples_leaf=args.tree_min_samples_leaf,
        minimum_precision=args.minimum_precision,
        minimum_per_video_recall=args.minimum_per_video_recall,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"model_sha256": artifact["model_sha256"], **artifact["metrics"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
