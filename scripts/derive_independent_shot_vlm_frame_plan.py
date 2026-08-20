#!/usr/bin/env python3
"""Derive a sealed frame-sampled VLM plan from a frozen label-free plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.independent_shot_vlm import (  # noqa: E402
    derive_frame_sampled_shot_vlm_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--image-width", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = derive_frame_sampled_shot_vlm_plan(
        json.loads(args.source_plan.read_text(encoding="utf-8")),
        max_frames=args.max_frames,
        image_width=args.image_width,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "plan_sha256": plan["plan_sha256"],
                "source_plan_sha256": plan["source_plan_sha256"],
                "example_count": len(plan["examples"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
