"""Training-only evaluation for Basketball-51 free-throw transfer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate_free_throw_transfer(
    rows: Sequence[Mapping[str, Any]],
    probabilities: Sequence[float],
    *,
    threshold: float,
    minimum_precision: float = 0.85,
    minimum_recall: float = 0.85,
) -> dict[str, Any]:
    """Evaluate a frozen source-domain classifier without target tuning."""

    if len(rows) != len(probabilities) or not rows:
        raise ValueError("transfer rows and predictions must align and be non-empty")
    if not 0.0 < threshold < 1.0:
        raise ValueError("transfer threshold must be between zero and one")
    if any(not 0.0 <= float(value) <= 1.0 for value in probabilities):
        raise ValueError("transfer probabilities must be between zero and one")

    truth = [bool(row["free_throw"]) for row in rows]
    prediction = [float(value) >= threshold for value in probabilities]
    per_game = []
    for source_sha in sorted({str(row["source_video_sha256"]) for row in rows}):
        indexes = [
            index
            for index, row in enumerate(rows)
            if str(row["source_video_sha256"]) == source_sha
        ]
        metrics = _metrics(
            [truth[index] for index in indexes],
            [prediction[index] for index in indexes],
        )
        metrics["source_video_sha256"] = source_sha
        per_game.append(metrics)

    overall = _metrics(truth, prediction)
    all_games_eligible = all(
        row["positive_count"] > 0
        and row["precision"] >= minimum_precision
        and row["recall"] >= minimum_recall
        for row in per_game
    )
    return {
        "threshold": threshold,
        "minimum_precision": minimum_precision,
        "minimum_recall": minimum_recall,
        "overall": overall,
        "per_game": per_game,
        "promotion_eligible": (
            overall["precision"] >= minimum_precision
            and overall["recall"] >= minimum_recall
            and all_games_eligible
        ),
    }


def _metrics(truth: Sequence[bool], prediction: Sequence[bool]) -> dict[str, Any]:
    return {
        "rows": len(truth),
        "positive_count": sum(truth),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
    }
