#!/usr/bin/env python3
"""Seal a compact, manually authored MUVY ball review."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.analysis.muvy_ball_review import (
    build_muvy_ball_review_decisions,
    seal_muvy_ball_review,
    verify_muvy_ball_review_plan,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--ranges", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = verify_muvy_ball_review_plan(_read_json(args.plan))
    review_ranges = _read_json(args.ranges)
    candidate_ids = [
        str(candidate["candidate_id"]) for candidate in plan["candidates"]
    ]
    decisions = build_muvy_ball_review_decisions(
        candidate_ids,
        valid_ranges=review_ranges.get("valid_ranges", []),
        uncertain_ranges=review_ranges.get("uncertain_ranges", []),
    )
    review = seal_muvy_ball_review(
        {
            "plan_sha256": plan["artifact_sha256"],
            "reviewer": review_ranges.get("reviewer"),
            "review_method": review_ranges.get("review_method"),
            "decisions": decisions,
        },
        plan=plan,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(review, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)

    counts = Counter(row["decision"] for row in decisions)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "candidate_count": len(decisions),
                "decision_counts": dict(sorted(counts.items())),
                "artifact_sha256": review["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
