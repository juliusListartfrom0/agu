#!/usr/bin/env python3
"""Validate dense raw-video review decisions and emit a sealed box-score bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.review import import_dense_review, seal_dense_review_result  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-team-points", default="{}", help='JSON object, for example {"dark":42}')
    parser.add_argument("--allow-incomplete-windows", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected = json.loads(args.expected_team_points)
    if not isinstance(expected, dict):
        raise ValueError("--expected-team-points must be a JSON object")
    result = import_dense_review(
        package_dir=args.package_dir,
        raw_video_path=args.video,
        require_all_windows=not args.allow_incomplete_windows,
    )
    bundle, score = seal_dense_review_result(
        game_id=args.game_id,
        raw_video_path=args.video,
        result=result,
        config={"pipeline": "dense_codex_raw_review_v1"},
        expected_team_points={str(key): int(value) for key, value in expected.items()},
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "prediction_bundle.json", bundle.model_dump(mode="json"))
    _write_json(args.output_dir / "official_box_score.json", score.model_dump(mode="json"))
    _write_json(
        args.output_dir / "import_audit.json",
        {
            "manifest_sha256": result.manifest_sha256,
            "reviewed_window_count": result.reviewed_window_count,
            "total_window_count": result.total_window_count,
            "complete_video_coverage": result.complete_video_coverage,
            "event_count": len(result.events),
            "bundle_sha256": bundle.bundle_sha256,
        },
    )
    print(json.dumps({"bundle_sha256": bundle.bundle_sha256, "status": score.status}, indent=2))
    return 0


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
