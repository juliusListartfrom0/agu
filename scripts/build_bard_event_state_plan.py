#!/usr/bin/env python3
"""Build a hash-bound, game-grouped BARD event-state training plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from app.analysis.bard_event_state import (
    build_bard_event_state_plan,
    parse_bard_dataset_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-csv-sha256", required=True)
    parser.add_argument("--per-class", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = args.source_csv.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != args.source_csv_sha256:
        raise ValueError("BARD source CSV SHA-256 mismatch")
    rows = parse_bard_dataset_csv(payload.decode("utf-8"))
    plan = build_bard_event_state_plan(
        rows,
        source_revision=args.source_revision,
        source_csv_sha256=actual_sha256,
        per_class=args.per_class,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "examples": len(plan["examples"]),
                "games": len(
                    {row["game_id"] for row in plan["examples"]}
                ),
                "plan_sha256": plan["plan_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
