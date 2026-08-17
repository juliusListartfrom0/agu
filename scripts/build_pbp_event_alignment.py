#!/usr/bin/env python3
"""Build an offline-only official PBP to video-frame alignment artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.pbp_event_alignment import (  # noqa: E402
    build_pbp_event_alignment_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--play-by-play", type=Path, required=True)
    parser.add_argument("--scoreboard-timeline", type=Path, required=True)
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--home-team", required=True)
    parser.add_argument("--away-team", required=True)
    parser.add_argument("--video-id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sealed-blind-video-sha256",
        action="append",
        default=[],
        help="Hash guard(s) for sealed blind videos; repeat for multiple hashes.",
    )
    return parser.parse_args()


def run_alignment(
    *,
    play_by_play_path: Path,
    scoreboard_timeline_path: Path,
    game_id: str,
    home_team: str,
    away_team: str,
    output_path: Path,
    video_id: str | None = None,
    sealed_blind_video_sha256s: list[str] | None = None,
) -> dict[str, Any]:
    raw_pbp = play_by_play_path.read_bytes()
    rows = [
        json.loads(line)
        for line in raw_pbp.decode("utf-8").splitlines()
        if line.strip()
    ]
    timeline = json.loads(scoreboard_timeline_path.read_text(encoding="utf-8"))
    artifact = build_pbp_event_alignment_artifact(
        play_by_play_rows=rows,
        scoreboard_timeline=timeline,
        game_id=game_id,
        home_team_id=home_team,
        away_team_id=away_team,
        source_play_by_play_sha256=hashlib.sha256(raw_pbp).hexdigest(),
        video_id=video_id,
        sealed_blind_video_sha256s=sealed_blind_video_sha256s or [],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return artifact


def main() -> int:
    args = parse_args()
    artifact = run_alignment(
        play_by_play_path=args.play_by_play,
        scoreboard_timeline_path=args.scoreboard_timeline,
        game_id=args.game_id,
        home_team=args.home_team,
        away_team=args.away_team,
        output_path=args.output,
        video_id=args.video_id,
        sealed_blind_video_sha256s=args.sealed_blind_video_sha256,
    )
    print(
        json.dumps(
            {
                "artifact_sha256": artifact["artifact_sha256"],
                "event_count": artifact["coverage"]["event_count"],
                "mapped_event_count": artifact["coverage"]["mapped_event_count"],
                "score_anchor_count": artifact["coverage"]["score_anchor_count"],
                "runtime_consumable": artifact["runtime_consumable"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
