#!/usr/bin/env python3
"""Screen a video embedding with game-level OOF logistic calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.shot_validity_finetune import binary_metrics  # noqa: E402
from app.analysis.shot_validity_video_backbone import (  # noqa: E402
    verify_video_embedding_artifact,
)


def screen_video_embeddings(
    embedding_path: Path | Sequence[Path],
    *,
    regularization_c: float = 0.01,
) -> dict[str, object]:
    if regularization_c <= 0:
        raise ValueError("regularization C must be positive")
    embedding_paths = (
        [embedding_path] if isinstance(embedding_path, Path) else list(embedding_path)
    )
    if not embedding_paths:
        raise ValueError("at least one embedding artifact is required")
    artifacts = [
        verify_video_embedding_artifact(json.loads(path.read_text(encoding="utf-8")))
        for path in embedding_paths
    ]
    examples, matrix = _align_embedding_artifacts(artifacts)
    target = np.asarray([int(row["event_present"]) for row in examples], dtype=np.int64)
    groups = np.asarray([str(row["source_video_sha256"]) for row in examples])
    if len(set(groups)) < 3 or len(set(target.tolist())) < 2:
        raise ValueError("video screening requires at least three games and both classes")
    probabilities = np.zeros(len(target), dtype=np.float64)
    folds = []
    for group in sorted(set(groups)):
        train = groups != group
        held = ~train
        scaler = StandardScaler().fit(matrix[train])
        classifier = LogisticRegression(
            C=regularization_c,
            class_weight="balanced",
            max_iter=5000,
            random_state=0,
        ).fit(scaler.transform(matrix[train]), target[train])
        held_probabilities = classifier.predict_proba(
            scaler.transform(matrix[held])
        )[:, 1]
        probabilities[held] = held_probabilities
        folds.append(
            {
                "held_game_sha256": group,
                "rows": int(held.sum()),
                "positive_count": int(target[held].sum()),
                "roc_auc": float(roc_auc_score(target[held], held_probabilities)),
                "average_precision": float(
                    average_precision_score(target[held], held_probabilities)
                ),
                "best_precision_at_recall_0_85": _best_precision_at_recall(
                    target[held], held_probabilities, minimum_recall=0.85
                ),
            }
        )
    threshold, gate = _select_strict_threshold(target, probabilities, groups)
    payload: dict[str, object] = {
        "schema_version": "agu.shot-validity-video-screen.v1",
        "purpose": (
            "backbone_screening_training_only"
            if len(artifacts) == 1
            else "backbone_fusion_screening_training_only"
        ),
        "runtime_consumable": False,
        "embedding_artifact_sha256s": [row["artifact_sha256"] for row in artifacts],
        "backbones": [row["backbone"] for row in artifacts],
        "backbone_sha256s": [row["backbone_sha256"] for row in artifacts],
        "regularization_c": regularization_c,
        "threshold": threshold,
        "promotion_requirements": {
            "minimum_pooled_precision": 0.95,
            "minimum_per_game_precision": 0.95,
            "minimum_per_game_recall": 0.85,
        },
        "gate": gate,
        "folds": folds,
        "oof_predictions": [
            {
                "source_video_sha256": str(row["source_video_sha256"]),
                "event_id": str(row["event_id"]),
                "event_present": bool(row["event_present"]),
                "probability": float(probability),
            }
            for row, probability in zip(examples, probabilities, strict=True)
        ],
    }
    if len(artifacts) == 1:
        payload.update(
            {
                "embedding_artifact_sha256": artifacts[0]["artifact_sha256"],
                "backbone": artifacts[0]["backbone"],
                "backbone_sha256": artifacts[0]["backbone_sha256"],
            }
        )
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return payload


def _align_embedding_artifacts(
    artifacts: Sequence[dict[str, object]],
) -> tuple[list[dict[str, object]], np.ndarray]:
    manifests = {str(row["training_manifest_sha256"]) for row in artifacts}
    if len(manifests) != 1:
        raise ValueError("embedding artifacts must share one training manifest")

    def key(row: dict[str, object]) -> tuple[str, str, str]:
        return (
            str(row["source_video_sha256"]),
            str(row.get("candidate_bundle_sha256", "")),
            str(row["event_id"]),
        )

    examples = list(artifacts[0]["examples"])
    expected_keys = [key(row) for row in examples]
    matrices = []
    for artifact in artifacts:
        lookup = {key(row): row for row in artifact["examples"]}
        if set(lookup) != set(expected_keys):
            raise ValueError("embedding artifacts must contain identical examples")
        aligned = [lookup[row_key] for row_key in expected_keys]
        if any(
            bool(row["event_present"]) != bool(expected["event_present"])
            for row, expected in zip(aligned, examples, strict=True)
        ):
            raise ValueError("embedding artifacts disagree on example labels")
        matrices.append(
            np.asarray([row["embedding"] for row in aligned], dtype=np.float64)
        )
    return examples, np.concatenate(matrices, axis=1)


def _select_strict_threshold(
    target: np.ndarray, probabilities: np.ndarray, groups: np.ndarray
) -> tuple[float, dict[str, object]]:
    candidates = sorted({0.0, 1.0, *(float(value) for value in probabilities)})
    eligible = []
    best_gate: dict[str, object] | None = None
    for threshold in candidates:
        pooled = binary_metrics(target.astype(bool).tolist(), probabilities.tolist(), threshold)
        per_game = {}
        for group in sorted(set(groups)):
            selected = groups == group
            per_game[group] = binary_metrics(
                target[selected].astype(bool).tolist(),
                probabilities[selected].tolist(),
                threshold,
            )
        promoted = (
            float(pooled["precision"]) >= 0.95
            and all(float(row["precision"]) >= 0.95 for row in per_game.values())
            and all(float(row["recall"]) >= 0.85 for row in per_game.values())
        )
        gate = {"promoted": promoted, "pooled": pooled, "per_game": per_game}
        if promoted:
            eligible.append((float(pooled["recall"]), float(pooled["f1"]), -threshold, threshold, gate))
        if best_gate is None or _diagnostic_gate_score(gate) > _diagnostic_gate_score(
            best_gate
        ):
            best_gate = gate
    if eligible:
        _recall, _f1, _negative_threshold, threshold, gate = max(eligible)
        return threshold, gate
    return 1.0, {
        "promoted": False,
        "pooled": binary_metrics(target.astype(bool).tolist(), probabilities.tolist(), 1.0),
        "per_game": {
            group: binary_metrics(
                target[groups == group].astype(bool).tolist(),
                probabilities[groups == group].tolist(),
                1.0,
            )
            for group in sorted(set(groups))
        },
        "best_non_promoted_gate": best_gate,
    }


def _diagnostic_gate_score(gate: dict[str, object]) -> tuple[float, ...]:
    per_game = list(gate["per_game"].values())
    passing_games = sum(
        float(row["precision"]) >= 0.95 and float(row["recall"]) >= 0.85
        for row in per_game
    )
    return (
        float(passing_games),
        min(float(row["recall"]) for row in per_game),
        min(float(row["precision"]) for row in per_game),
        float(gate["pooled"]["f1"]),
    )


def _best_precision_at_recall(
    target: np.ndarray, probabilities: np.ndarray, *, minimum_recall: float
) -> dict[str, float | int]:
    candidates = []
    for threshold in sorted({0.0, 1.0, *(float(value) for value in probabilities)}):
        metrics = binary_metrics(
            target.astype(bool).tolist(), probabilities.tolist(), threshold
        )
        if float(metrics["recall"]) >= minimum_recall:
            candidates.append((float(metrics["precision"]), float(metrics["f1"]), threshold, metrics))
    if not candidates:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "threshold": 1.0}
    _precision, _f1, threshold, metrics = max(candidates)
    return {**metrics, "threshold": threshold}


def _canonical_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regularization-c", type=float, default=0.01)
    args = parser.parse_args()
    result = screen_video_embeddings(
        args.embeddings, regularization_c=args.regularization_c
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"backbones": result["backbones"], **result["gate"]}, indent=2))
    return 0 if result["gate"]["promoted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
