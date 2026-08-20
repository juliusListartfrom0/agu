#!/usr/bin/env python3
"""Prepare a truth-free handoff after a public benchmark pair is downloaded."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.public_research_datasets import (  # noqa: E402
    build_public_game_pair_handoff,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--download-state", type=Path, required=True)
    parser.add_argument("--benchmark-slug", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    state = json.loads(args.download_state.read_text(encoding="utf-8"))
    handoff = build_public_game_pair_handoff(
        plan,
        state,
        benchmark_slug=args.benchmark_slug,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(handoff, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "handoff_sha256": handoff["handoff_sha256"],
                "benchmark": handoff["benchmark_inference_media"]["slug"],
                "enrollment_count": len(handoff["face_enrollment_media"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
