#!/usr/bin/env python3
"""Evaluate frozen formation predictions against a sealed offline review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.causal_shot_phase_review import (  # noqa: E402
    verify_causal_shot_phase_review,
)
from app.analysis.independent_formation_vlm import (  # noqa: E402
    evaluate_independent_formation_vlm,
    verify_independent_formation_vlm_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--phase-plan", type=Path, required=True)
    parser.add_argument("--phase-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = verify_independent_formation_vlm_plan(
        json.loads(args.plan.read_text(encoding="utf-8"))
    )
    phase_plan = json.loads(args.phase_plan.read_text(encoding="utf-8"))
    phase_review = verify_causal_shot_phase_review(
        json.loads(args.phase_review.read_text(encoding="utf-8")),
        plan=phase_plan,
    )
    truth_by_identity = {
        (
            str(row["source_video_sha256"]),
            str(row["candidate_bundle_sha256"]),
            str(row["event_id"]),
        ): str(row["formation_state"])
        for row in phase_review["reviews"]
    }
    required = {
        (
            str(row["source_video_sha256"]),
            str(row["candidate_bundle_sha256"]),
            str(row["event_id"]),
        )
        for row in plan["examples"]
    }
    evaluation = evaluate_independent_formation_vlm(
        plan=plan,
        predictions=json.loads(args.predictions.read_text(encoding="utf-8")),
        truth={key: truth_by_identity[key] for key in required},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evaluation, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "artifact_sha256": evaluation["artifact_sha256"],
                "metrics": evaluation["metrics"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
