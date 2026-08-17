#!/usr/bin/env python3
"""Derive a sealed two-frame formation plan from a label-hidden phase plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.independent_formation_vlm import (  # noqa: E402
    derive_independent_formation_vlm_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-plan", type=Path, required=True)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = derive_independent_formation_vlm_plan(
        json.loads(args.phase_plan.read_text(encoding="utf-8")),
        image_width=args.image_width,
        max_examples=args.max_examples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "plan_sha256": plan["plan_sha256"],
                "source_phase_plan_sha256": plan[
                    "source_phase_plan_sha256"
                ],
                "example_count": len(plan["examples"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
