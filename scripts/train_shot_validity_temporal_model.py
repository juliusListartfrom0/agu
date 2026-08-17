#!/usr/bin/env python3
"""Train a sealed logistic temporal shot-validity head with leave-one-game-out calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.shot_validity_temporal import (  # noqa: E402
    seal_temporal_shot_model,
    verify_temporal_embedding_artifact,
)


def train_temporal_model(
    *,
    embedding_path: Path,
    regularization_c: float = 0.01,
    minimum_precision: float = 0.95,
    minimum_per_video_recall: float = 0.85,
) -> dict[str, object]:
    if (
        regularization_c <= 0
        or not 0 < minimum_precision <= 1
        or not 0 <= minimum_per_video_recall <= 1
    ):
        raise ValueError("invalid temporal model training parameters")
    artifact = verify_temporal_embedding_artifact(
        json.loads(embedding_path.read_text(encoding="utf-8"))
    )
    examples = artifact["examples"]
    matrix = np.asarray([row["embedding"] for row in examples], dtype=np.float64)
    target = np.asarray([int(row["event_present"]) for row in examples], dtype=np.int64)
    groups = np.asarray([str(row["source_video_sha256"]) for row in examples])
    if len(set(groups)) < 2 or len(set(target.tolist())) < 2:
        raise ValueError("temporal training requires two games and both classes")
    probabilities = np.zeros(len(target), dtype=np.float64)
    for held_group in sorted(set(groups)):
        train = groups != held_group
        held = ~train
        if len(set(target[train].tolist())) < 2:
            raise ValueError("each temporal leave-one-game-out fold needs both classes")
        scaler = StandardScaler().fit(matrix[train])
        classifier = LogisticRegression(
            C=regularization_c,
            class_weight="balanced",
            max_iter=5000,
            random_state=0,
        ).fit(scaler.transform(matrix[train]), target[train])
        probabilities[held] = classifier.predict_proba(scaler.transform(matrix[held]))[:, 1]
    threshold, precision, recall, f1 = _calibrate(
        target, probabilities, minimum_precision=minimum_precision
    )
    per_video = _per_video_metrics(target, probabilities, groups, threshold=threshold)
    minimum_observed_recall = min(
        float(row["recall"]) for row in per_video.values()
    )
    every_video_precision_gate = all(
        float(row["precision"]) >= minimum_precision for row in per_video.values()
    )
    scaler = StandardScaler().fit(matrix)
    classifier = LogisticRegression(
        C=regularization_c,
        class_weight="balanced",
        max_iter=5000,
        random_state=0,
    ).fit(scaler.transform(matrix), target)
    return seal_temporal_shot_model(
        {
            "training_manifest_sha256": artifact["training_manifest_sha256"],
            "training_embedding_artifact_sha256": artifact["artifact_sha256"],
            "training_annotation_sha256": artifact["training_annotation_sha256"],
            "backbone_sha256": artifact["backbone_sha256"],
            "clip_frames": artifact["clip_frames"],
            "regularization_c": regularization_c,
            "threshold": threshold,
            "coefficients": classifier.coef_[0].astype(float).tolist(),
            "intercept": float(classifier.intercept_[0]),
            "feature_mean": scaler.mean_.astype(float).tolist(),
            "feature_scale": scaler.scale_.astype(float).tolist(),
            "metrics": {
                "training_rows": len(target),
                "training_video_count": len(set(groups)),
                "positive_count": int(target.sum()),
                "negative_count": int(len(target) - target.sum()),
                "calibration": "leave_one_video_out",
                "minimum_precision": minimum_precision,
                "minimum_per_video_recall": minimum_per_video_recall,
                "leave_one_video_out_precision": precision,
                "leave_one_video_out_recall": recall,
                "leave_one_video_out_f1": f1,
                "minimum_precision_gate": precision >= minimum_precision,
                "leave_one_video_out_per_video": per_video,
                "minimum_observed_per_video_recall": minimum_observed_recall,
                "every_video_precision_gate": every_video_precision_gate,
                "promotion_gate": (
                    precision >= minimum_precision
                    and every_video_precision_gate
                    and minimum_observed_recall >= minimum_per_video_recall
                ),
            },
        }
    )


def _per_video_metrics(
    truth: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    *,
    threshold: float,
) -> dict[str, dict[str, float | int]]:
    predicted = probabilities >= threshold
    output: dict[str, dict[str, float | int]] = {}
    for group in sorted(set(groups)):
        selected = groups == group
        output[str(group)] = {
            "rows": int(selected.sum()),
            "positive_count": int(truth[selected].sum()),
            "predicted_positive_count": int(predicted[selected].sum()),
            "precision": float(
                precision_score(truth[selected], predicted[selected], zero_division=0)
            ),
            "recall": float(
                recall_score(truth[selected], predicted[selected], zero_division=0)
            ),
            "f1": float(f1_score(truth[selected], predicted[selected], zero_division=0)),
        }
    return output


def _calibrate(
    truth: np.ndarray,
    probabilities: np.ndarray,
    *,
    minimum_precision: float,
) -> tuple[float, float, float, float]:
    eligible = []
    for threshold in sorted({0.0, 1.0, *(float(value) for value in probabilities)}):
        predicted = probabilities >= threshold
        precision = float(precision_score(truth, predicted, zero_division=0))
        recall = float(recall_score(truth, predicted, zero_division=0))
        f1 = float(f1_score(truth, predicted, zero_division=0))
        if precision >= minimum_precision:
            eligible.append((threshold, precision, recall, f1))
    if not eligible:
        return 1.0, 0.0, 0.0, 0.0
    return max(eligible, key=lambda row: (row[2], row[3], -row[0]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regularization-c", type=float, default=0.01)
    parser.add_argument("--minimum-precision", type=float, default=0.95)
    parser.add_argument("--minimum-per-video-recall", type=float, default=0.85)
    args = parser.parse_args()
    artifact = train_temporal_model(
        embedding_path=args.embeddings,
        regularization_c=args.regularization_c,
        minimum_precision=args.minimum_precision,
        minimum_per_video_recall=args.minimum_per_video_recall,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model_sha256": artifact["model_sha256"], **artifact["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
