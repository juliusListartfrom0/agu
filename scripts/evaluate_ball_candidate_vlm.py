#!/usr/bin/env python3
"""Evaluate sealed independent-VLM basketball-candidate predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.analysis.ball_candidate_review import seal_artifact, verify_artifact
from app.analysis.ball_candidate_vlm import evaluate_ball_candidate_vlm


def _round_metrics(value: dict[str, Any]) -> dict[str, Any]:
    rounded = dict(value)
    rounded["metrics"] = {
        bound: {
            key: round(float(metric), 6)
            for key, metric in metrics.items()
        }
        for bound, metrics in value["metrics"].items()
    }
    return rounded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perception", type=Path, required=True)
    parser.add_argument("--review-plan", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-precision", type=float, default=0.85)
    parser.add_argument("--minimum-recall", type=float, default=0.85)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    perception = verify_artifact(json.loads(args.perception.read_text()))
    plan = verify_artifact(json.loads(args.review_plan.read_text()))
    review = verify_artifact(json.loads(args.review.read_text()))
    predictions = verify_artifact(json.loads(args.predictions.read_text()))
    evaluation = _round_metrics(
        evaluate_ball_candidate_vlm(
            perception=perception,
            plan=plan,
            review=review,
            predictions=predictions,
        )
    )
    lower = evaluation["metrics"]["lower"]
    accepted = bool(
        lower["precision"] >= args.minimum_precision
        and lower["recall"] >= args.minimum_recall
    )
    artifact = seal_artifact(
        {
            "schema_version": "agu.ball-candidate-independent-vlm-evaluation.v1",
            "purpose": "offline_development_model_screening",
            "runtime_consumable": False,
            "codex_runtime_answer_used": False,
            "perception_sha256": perception["artifact_sha256"],
            "review_plan_sha256": plan["artifact_sha256"],
            "review_sha256": review["artifact_sha256"],
            "predictions_sha256": predictions["artifact_sha256"],
            "thresholds": {
                "minimum_precision": args.minimum_precision,
                "minimum_recall": args.minimum_recall,
                "bound": "lower",
            },
            **evaluation,
            "accepted": accepted,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "accepted": accepted,
                "metrics": artifact["metrics"],
                "artifact_sha256": artifact["artifact_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
