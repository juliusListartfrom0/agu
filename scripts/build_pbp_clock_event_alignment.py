#!/usr/bin/env python3
"""Build an offline-only PBP-to-broadcast-clock alignment artifact."""

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

from app.analysis.pbp_clock_event_alignment import (  # noqa: E402
    build_pbp_clock_event_alignment_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--play-by-play", type=Path, required=True)
    parser.add_argument("--clock-artifact", type=Path, required=True)
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--home-team", required=True)
    parser.add_argument("--away-team", required=True)
    parser.add_argument("--video-id")
    parser.add_argument("--maximum-clock-delta-seconds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"invalid JSONL row {line_number}")
        rows.append(row)
    if not rows:
        raise ValueError("play-by-play JSONL is empty")
    return raw, rows


def build_alignment(
    *,
    play_by_play_path: Path,
    clock_artifact_path: Path,
    game_id: str,
    home_team: str,
    away_team: str,
    output_path: Path,
    video_id: str | None = None,
    maximum_clock_delta_seconds: int = 3,
) -> dict[str, Any]:
    pbp_bytes, rows = load_jsonl(play_by_play_path)
    artifact = build_pbp_clock_event_alignment_artifact(
        play_by_play_rows=rows,
        clock_artifact=json.loads(clock_artifact_path.read_text(encoding="utf-8")),
        game_id=game_id,
        home_team_id=home_team,
        away_team_id=away_team,
        source_play_by_play_sha256=hashlib.sha256(pbp_bytes).hexdigest(),
        video_id=video_id,
        maximum_clock_delta_seconds=maximum_clock_delta_seconds,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return artifact


if __name__ == "__main__":
    args = parse_args()
    artifact = build_alignment(
        play_by_play_path=args.play_by_play,
        clock_artifact_path=args.clock_artifact,
        game_id=args.game_id,
        home_team=args.home_team,
        away_team=args.away_team,
        video_id=args.video_id,
        maximum_clock_delta_seconds=args.maximum_clock_delta_seconds,
        output_path=args.output,
    )
    print(json.dumps({"artifact_sha256": artifact["artifact_sha256"], "coverage": artifact["coverage"]}, indent=2))
