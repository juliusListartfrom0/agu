#!/usr/bin/env python3
"""Train AGU's traditional action-owner model from sealed, disjoint labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.action_ownership import (  # noqa: E402
    ACTION_OWNER_FEATURES,
    extract_action_owner_features,
    seal_action_owner_model,
    seal_extra_trees_action_owner_model,
)
from app.analysis.training_annotation import verify_training_annotation_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regularization", type=float, default=1.0)
    parser.add_argument(
        "--model-type", choices=("linear", "extra_trees"), default="linear"
    )
    parser.add_argument("--tree-count", type=int, default=100)
    parser.add_argument("--tree-max-depth", type=int, default=3)
    parser.add_argument("--tree-min-samples-leaf", type=int, default=2)
    return parser.parse_args()


def train_action_owner_model(
    *,
    manifest_path: Path,
    annotation_paths: list[Path],
    regularization: float = 1.0,
    model_type: str = "linear",
    tree_count: int = 100,
    tree_max_depth: int = 3,
    tree_min_samples_leaf: int = 2,
) -> dict[str, Any]:
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    if model_type not in {"linear", "extra_trees"}:
        raise ValueError("unsupported action-owner model type")
    if min(tree_count, tree_max_depth, tree_min_samples_leaf) <= 0:
        raise ValueError("Extra Trees parameters must be positive")
    manifest = verify_training_annotation_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    allowed_annotations = {
        (str(item["filename"]), str(item["sha256"]))
        for item in manifest.get("annotation_files", [])
    }
    allowed_sources = {
        str(item["sha256"]) for item in manifest.get("source_videos", [])
    }
    examples: list[dict[str, Any]] = []
    example_count = 0
    for path in annotation_paths:
        digest = _file_sha256(path)
        if (path.name, digest) not in allowed_annotations:
            raise ValueError(f"annotation is not SHA-bound by training manifest: {path.name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "agu.action-ownership-labels.v1":
            raise ValueError(f"unsupported action ownership labels: {path.name}")
        source_hash = str(payload.get("source_video_sha256") or "")
        if source_hash not in allowed_sources:
            raise ValueError(f"annotation source is not bound by training manifest: {path.name}")
        for example_index, example in enumerate(payload.get("examples", [])):
            positive = str(example.get("positive_player_id") or "")
            anchor_frame = int(example["anchor_frame"])
            by_player: dict[str, list[dict[str, Any]]] = {}
            for observation in example.get("candidate_player_observations", []):
                player_id = str(observation.get("player_id") or "")
                if player_id:
                    by_player.setdefault(player_id, []).append(observation)
            if positive not in by_player or len(by_player) < 2:
                raise ValueError("each training example needs a positive and at least one negative player")
            feature_rows: dict[str, list[float]] = {}
            for player_id, observations in sorted(by_player.items()):
                features = extract_action_owner_features(observations, anchor_frame=anchor_frame)
                feature_rows[player_id] = [features[name] for name in ACTION_OWNER_FEATURES]
            examples.append(
                {
                    "group": source_hash,
                    "action": f"{source_hash}:{example_index}",
                    "positive": positive,
                    "features": feature_rows,
                }
            )
            example_count += 1
    if example_count < 2:
        raise ValueError("at least two labeled actions with positive and negative candidates are required")
    annotation_hashes = sorted(_file_sha256(path) for path in annotation_paths)
    if model_type == "extra_trees":
        return _train_extra_trees_model(
            examples,
            manifest_sha256=manifest["manifest_sha256"],
            training_annotation_producer=str(manifest["producer"]),
            training_benchmark_overlap=bool(manifest["benchmark_overlap"]),
            annotation_hashes=annotation_hashes,
            tree_count=tree_count,
            max_depth=tree_max_depth,
            min_samples_leaf=tree_min_samples_leaf,
        )
    rows, labels, groups = _pairwise_rows(examples)
    matrix = np.asarray(rows, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    scaler = StandardScaler().fit(matrix)
    normalized = scaler.transform(matrix)
    classifier = LogisticRegression(
        C=regularization,
        class_weight="balanced",
        max_iter=2000,
        random_state=0,
    ).fit(normalized, target)
    metrics: dict[str, Any] = {
        "training_objective": "within_action_pairwise_ranking",
        "training_rows": sum(len(example["features"]) for example in examples),
        "training_pairwise_rows": len(rows),
        "training_actions": example_count,
        "training_video_count": len(set(groups)),
    }
    if len(set(groups)) >= 2:
        pairwise_truth: list[int] = []
        pairwise_predictions: list[int] = []
        correct_actions = 0
        for held_group in sorted(set(groups)):
            train_indexes = [index for index, group in enumerate(groups) if group != held_group]
            if not train_indexes:
                continue
            fold_scaler = StandardScaler().fit(matrix[train_indexes])
            fold_classifier = LogisticRegression(
                C=regularization,
                class_weight="balanced",
                max_iter=2000,
                random_state=0,
            ).fit(fold_scaler.transform(matrix[train_indexes]), target[train_indexes])
            held_indexes = [index for index, group in enumerate(groups) if group == held_group]
            held_predictions = fold_classifier.predict(
                fold_scaler.transform(matrix[held_indexes])
            )
            pairwise_truth.extend(target[held_indexes].tolist())
            pairwise_predictions.extend(held_predictions.tolist())
            for example in examples:
                if example["group"] != held_group:
                    continue
                candidates = sorted(example["features"])
                candidate_matrix = np.asarray(
                    [example["features"][player_id] for player_id in candidates],
                    dtype=np.float64,
                )
                scores = fold_classifier.decision_function(
                    fold_scaler.transform(candidate_matrix)
                )
                selected = candidates[int(np.argmax(scores))]
                correct_actions += int(selected == example["positive"])
        metrics["leave_one_video_out_pairwise_f1"] = float(
            f1_score(pairwise_truth, pairwise_predictions)
        )
        metrics["leave_one_video_out_top1_accuracy"] = correct_actions / len(examples)
    return seal_action_owner_model(
        {
            "training_manifest_sha256": manifest["manifest_sha256"],
            "training_annotation_sha256": annotation_hashes,
            "coefficients": classifier.coef_[0].tolist(),
            "intercept": float(classifier.intercept_[0]),
            "feature_mean": scaler.mean_.tolist(),
            "feature_scale": scaler.scale_.tolist(),
            "regularization": regularization,
            "metrics": metrics,
        }
    )


def _train_extra_trees_model(
    examples: list[dict[str, Any]],
    *,
    manifest_sha256: str,
    training_annotation_producer: str,
    training_benchmark_overlap: bool,
    annotation_hashes: list[str],
    tree_count: int,
    max_depth: int,
    min_samples_leaf: int,
) -> dict[str, Any]:
    matrix, target, groups = _pointwise_rows(examples)
    classifier = _extra_trees_classifier(
        tree_count=tree_count,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    ).fit(matrix, target)
    metrics: dict[str, Any] = {
        "training_objective": "within_action_pointwise_classification",
        "training_rows": len(matrix),
        "training_actions": len(examples),
        "training_video_count": len(set(groups)),
    }
    if len(set(groups)) >= 2:
        candidate_truth: list[int] = []
        candidate_predictions: list[int] = []
        correct_actions = 0
        for held_group in sorted(set(groups)):
            train_indexes = [index for index, group in enumerate(groups) if group != held_group]
            held_indexes = [index for index, group in enumerate(groups) if group == held_group]
            fold_classifier = _extra_trees_classifier(
                tree_count=tree_count,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
            ).fit(matrix[train_indexes], target[train_indexes])
            probabilities = fold_classifier.predict_proba(matrix[held_indexes])[:, 1]
            candidate_truth.extend(target[held_indexes].tolist())
            candidate_predictions.extend((probabilities >= 0.5).astype(np.int64).tolist())
            for example in examples:
                if example["group"] != held_group:
                    continue
                candidates = sorted(example["features"])
                candidate_matrix = np.asarray(
                    [example["features"][player_id] for player_id in candidates],
                    dtype=np.float64,
                )
                scores = fold_classifier.predict_proba(candidate_matrix)[:, 1]
                selected = candidates[int(np.argmax(scores))]
                correct_actions += int(selected == example["positive"])
        metrics["leave_one_video_out_candidate_f1"] = float(
            f1_score(candidate_truth, candidate_predictions)
        )
        metrics["leave_one_video_out_top1_accuracy"] = correct_actions / len(examples)
    return seal_extra_trees_action_owner_model(
        {
            "training_manifest_sha256": manifest_sha256,
            "training_annotation_producer": training_annotation_producer,
            "training_benchmark_overlap": str(training_benchmark_overlap).lower(),
            "training_annotation_sha256": annotation_hashes,
            "tree_count": tree_count,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "class_weight": "balanced",
            "random_state": 0,
            "trees": [_serialize_tree(estimator) for estimator in classifier.estimators_],
            "metrics": metrics,
        }
    )


def _extra_trees_classifier(
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


def _pointwise_rows(
    examples: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows: list[list[float]] = []
    labels: list[int] = []
    groups: list[str] = []
    for example in examples:
        for player_id, features in sorted(example["features"].items()):
            rows.append(features)
            labels.append(int(player_id == example["positive"]))
            groups.append(example["group"])
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        groups,
    )


def _serialize_tree(estimator: Any) -> dict[str, list[Any]]:
    tree = estimator.tree_
    positive_class_index = list(estimator.classes_).index(1)
    probabilities = []
    for node_values in tree.value:
        values = node_values[0]
        total = float(np.sum(values))
        probabilities.append(float(values[positive_class_index] / total) if total else 0.0)
    return {
        "children_left": tree.children_left.astype(int).tolist(),
        "children_right": tree.children_right.astype(int).tolist(),
        "feature": tree.feature.astype(int).tolist(),
        "threshold": tree.threshold.astype(float).tolist(),
        "positive_probability": probabilities,
    }


def _pairwise_rows(
    examples: list[dict[str, Any]],
) -> tuple[list[list[float]], list[int], list[str]]:
    rows: list[list[float]] = []
    labels: list[int] = []
    groups: list[str] = []
    for example in examples:
        positive = example["features"][example["positive"]]
        for player_id, negative in sorted(example["features"].items()):
            if player_id == example["positive"]:
                continue
            difference = [left - right for left, right in zip(positive, negative)]
            rows.extend([difference, [-value for value in difference]])
            labels.extend([1, 0])
            groups.extend([example["group"], example["group"]])
    return rows, labels, groups


def main() -> int:
    args = parse_args()
    artifact = train_action_owner_model(
        manifest_path=args.manifest,
        annotation_paths=args.annotations,
        regularization=args.regularization,
        model_type=args.model_type,
        tree_count=args.tree_count,
        tree_max_depth=args.tree_max_depth,
        tree_min_samples_leaf=args.tree_min_samples_leaf,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model_sha256": artifact["model_sha256"], **artifact["metrics"]}, indent=2))
    return 0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
