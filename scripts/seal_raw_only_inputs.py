#!/usr/bin/env python3
"""Seal raw acceptance video identities without adding predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.analysis.official_evaluation import seal_raw_only_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--video", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = seal_raw_only_predictions(
        game_id=args.game_id,
        raw_video_paths=args.video,
        events=[],
        config={"purpose": "acceptance_input_hash_firewall"},
        model_provenance={"artifact_role": "acceptance_input_only"},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(bundle.bundle_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
