#!/usr/bin/env python3
"""Screen Basketball-51 embeddings with source-game-disjoint folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.basketball51 import (  # noqa: E402
    BASKETBALL51_LABELS,
    verify_basketball51_embedding_artifact,
)


def screen_basketball51_embeddings(
    embedding_path: Path,
    *,
    fold_count: int = 5,
    regularization_values: Sequence[float] = (0.0001, 0.001, 0.01, 0.1),
) -> dict[str, Any]:
    artifact = verify_basketball51_embedding_artifact(
        json.loads(embedding_path.read_text(encoding="utf-8"))
    )
    if fold_count < 3 or not regularization_values:
        raise ValueError("Basketball-51 screening requires at least three folds")
    if any(value <= 0 for value in regularization_values):
        raise ValueError("regularization values must be positive")
    examples = artifact["examples"]
    groups = np.asarray([str(row["source_group"]) for row in examples])
    labels = np.asarray([str(row["label"]) for row in examples])
    if len(set(groups)) < fold_count or set(labels) != set(BASKETBALL51_LABELS):
        raise ValueError(
            "Basketball-51 screening requires all labels and enough source groups"
        )
    matrix = np.asarray([row["embedding"] for row in examples], dtype=np.float64)
    tasks = {
        "field_goal_vs_free_throw": _screen_binary_task(
            matrix,
            labels,
            groups,
            target=np.asarray(
                [not label.startswith("ft") for label in labels],
                dtype=np.int64,
            ),
            fold_count=fold_count,
            regularization_values=regularization_values,
        ),
        "made_vs_missed": _screen_binary_task(
            matrix,
            labels,
            groups,
            target=np.asarray(
                [label.endswith("1") for label in labels],
                dtype=np.int64,
            ),
            fold_count=fold_count,
            regularization_values=regularization_values,
        ),
        "eight_class": _screen_multiclass_task(
            matrix,
            labels,
            groups,
            fold_count=fold_count,
            regularization_values=regularization_values,
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": "agu.basketball51-video-screen.v1",
        "purpose": "backbone_screening_training_only",
        "runtime_consumable": False,
        "promotion_eligible": False,
        "promotion_exclusion_reason": (
            "source-domain pretraining screen; requires target-domain disjoint validation"
        ),
        "embedding_artifact_sha256": artifact["artifact_sha256"],
        "backbone": artifact["backbone"],
        "backbone_sha256": artifact["backbone_sha256"],
        "clip_count": len(examples),
        "source_group_count": len(set(groups)),
        "fold_count": fold_count,
        "tasks": tasks,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return payload


def _screen_binary_task(
    matrix: np.ndarray,
    stratification_labels: np.ndarray,
    groups: np.ndarray,
    *,
    target: np.ndarray,
    fold_count: int,
    regularization_values: Sequence[float],
) -> dict[str, Any]:
    screens = []
    for value in regularization_values:
        predictions = np.zeros(len(target), dtype=np.int64)
        folds = []
        splitter = StratifiedGroupKFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=0,
        )
        for fold_index, (train, held) in enumerate(
            splitter.split(matrix, stratification_labels, groups),
            start=1,
        ):
            classifier = _classifier(value).fit(matrix[train], target[train])
            held_predictions = classifier.predict(matrix[held]).astype(np.int64)
            predictions[held] = held_predictions
            folds.append(
                {
                    "fold": fold_index,
                    "source_groups": sorted(set(groups[held].tolist())),
                    **_binary_metrics(target[held], held_predictions),
                }
            )
        pooled = _binary_metrics(target, predictions)
        screens.append(
            {
                "regularization_c": value,
                "pooled": pooled,
                "folds": folds,
                "worst_fold_precision": min(row["precision"] for row in folds),
                "worst_fold_recall": min(row["recall"] for row in folds),
                "worst_fold_f1": min(row["f1"] for row in folds),
            }
        )
    best = max(
        screens,
        key=lambda row: (
            row["worst_fold_f1"],
            row["pooled"]["f1"],
            -row["regularization_c"],
        ),
    )
    return {
        "selection": "maximum_worst_source_group_fold_f1",
        "minimum_pretraining_gate": {
            "worst_fold_precision": 0.85,
            "worst_fold_recall": 0.85,
        },
        "pretraining_gate_passed": (
            best["worst_fold_precision"] >= 0.85
            and best["worst_fold_recall"] >= 0.85
        ),
        "best": best,
        "screens": screens,
    }


def _screen_multiclass_task(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    fold_count: int,
    regularization_values: Sequence[float],
) -> dict[str, Any]:
    class_labels = sorted(BASKETBALL51_LABELS)
    screens = []
    for value in regularization_values:
        predictions = np.empty(len(labels), dtype=object)
        folds = []
        splitter = StratifiedGroupKFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=0,
        )
        for fold_index, (train, held) in enumerate(
            splitter.split(matrix, labels, groups),
            start=1,
        ):
            classifier = _classifier(value).fit(matrix[train], labels[train])
            held_predictions = classifier.predict(matrix[held])
            predictions[held] = held_predictions
            folds.append(
                {
                    "fold": fold_index,
                    "source_groups": sorted(set(groups[held].tolist())),
                    "accuracy": float(
                        accuracy_score(labels[held], held_predictions)
                    ),
                    "macro_f1": float(
                        f1_score(
                            labels[held],
                            held_predictions,
                            labels=class_labels,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                }
            )
        screens.append(
            {
                "regularization_c": value,
                "pooled_accuracy": float(accuracy_score(labels, predictions)),
                "pooled_macro_f1": float(
                    f1_score(
                        labels,
                        predictions,
                        labels=class_labels,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "folds": folds,
                "worst_fold_accuracy": min(row["accuracy"] for row in folds),
                "worst_fold_macro_f1": min(row["macro_f1"] for row in folds),
            }
        )
    best = max(
        screens,
        key=lambda row: (
            row["worst_fold_macro_f1"],
            row["pooled_macro_f1"],
            -row["regularization_c"],
        ),
    )
    return {
        "selection": "maximum_worst_source_group_fold_macro_f1",
        "pretraining_gate_passed": best["worst_fold_macro_f1"] >= 0.85,
        "best": best,
        "screens": screens,
    }


def _classifier(regularization_c: float) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=regularization_c,
            class_weight="balanced",
            max_iter=5000,
            random_state=0,
        ),
    )


def _binary_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float | int]:
    return {
        "rows": len(target),
        "positive_count": int(target.sum()),
        "precision": float(
            precision_score(target, prediction, zero_division=0)
        ),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(target, prediction)),
    }


def _canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument(
        "--regularization-c",
        type=float,
        action="append",
        dest="regularization_values",
    )
    args = parser.parse_args()
    result = screen_basketball51_embeddings(
        args.embeddings,
        fold_count=args.fold_count,
        regularization_values=(
            tuple(args.regularization_values)
            if args.regularization_values
            else (0.0001, 0.001, 0.01, 0.1)
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "artifact_sha256": result["artifact_sha256"],
                "tasks": {
                    task: details["best"]
                    for task, details in result["tasks"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
