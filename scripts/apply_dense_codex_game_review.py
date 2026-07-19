#!/usr/bin/env python3
"""Combine exhaustive dense reviews for all raw videos in one basketball game."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.review import seal_dense_game_reviews  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--package-dir", type=Path, action="append", required=True)
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-team-points", default="{}")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.package_dir) != len(args.video):
        raise ValueError("--package-dir and --video counts must match")
    expected = json.loads(args.expected_team_points)
    if not isinstance(expected, dict):
        raise ValueError("--expected-team-points must be a JSON object")
    result = seal_dense_game_reviews(
        game_id=args.game_id,
        package_video_pairs=list(zip(args.package_dir, args.video)),
        config={"pipeline": "dense_codex_raw_game_review_v1"},
        expected_team_points={str(key): int(value) for key, value in expected.items()},
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "prediction_bundle.json", result.bundle.model_dump(mode="json"))
    _write_json(args.output_dir / "official_box_score.json", result.box_score.model_dump(mode="json"))
    _write_json(
        args.output_dir / "game_review_audit.json",
        {
            "bundle_sha256": result.bundle.bundle_sha256,
            "manifest_sha256_by_video_id": result.manifest_sha256_by_video_id,
            "reviewed_window_count": result.reviewed_window_count,
            "total_window_count": result.total_window_count,
            "complete_video_coverage": result.complete_video_coverage,
            "event_count": len(result.events),
            "status": result.box_score.status,
        },
    )
    print(json.dumps({"bundle_sha256": result.bundle.bundle_sha256, "status": result.box_score.status}, indent=2))
    return 0


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
