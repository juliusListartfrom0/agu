#!/usr/bin/env python3
"""Validate and seal offline ball-candidate review decisions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from app.analysis.ball_candidate_review import seal_artifact, verify_artifact

ALLOWED = {"valid_ball", "false_positive", "uncertain"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = verify_artifact(json.loads(args.plan.read_text()))
    with args.decisions.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = [str(row["candidate_id"]) for row in plan["candidates"]]
    actual = [str(row.get("candidate_id", "")) for row in rows]
    if actual != expected or len(set(actual)) != len(actual):
        raise ValueError("review decisions do not exactly match plan order and IDs")
    if any(str(row.get("decision", "")) not in ALLOWED for row in rows):
        raise ValueError("review contains an unsupported decision")
    counts = Counter(str(row["decision"]) for row in rows)
    artifact = seal_artifact(
        {
            "schema_version": "agu.broadcast-ball-offline-review.v1",
            "purpose": plan["purpose"],
            "runtime_consumable": False,
            "codex_runtime_answer_used": False,
            "plan_sha256": plan["artifact_sha256"],
            "reviewer": "codex_offline_source_annotation",
            "review_method": "single-box raw-frame context plus enlarged local inset",
            "scope_confirmation": plan["review_scope"],
            "counts": dict(sorted(counts.items())),
            "decisions": [
                {
                    "candidate_id": str(row["candidate_id"]),
                    "decision": str(row["decision"]),
                    "notes": "",
                }
                for row in rows
            ],
        }
    )
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"counts": artifact["counts"], "artifact_sha256": artifact["artifact_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
