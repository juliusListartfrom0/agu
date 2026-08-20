#!/usr/bin/env python3
"""Seal one raw video as a truth-free empty AGU bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.official_evaluation import seal_raw_only_predictions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = seal_raw_only_predictions(
        game_id=args.game_id,
        raw_video_paths=[args.video],
        events=[],
        config={"purpose": "truth_free_raw_video_reservation"},
        model_provenance={
            "producer": "agu",
            "inference_mode": "traditional_cv+vlm",
            "candidate_backend": "none",
            "semantic_backend": "none",
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "bundle_sha256": bundle.bundle_sha256}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
