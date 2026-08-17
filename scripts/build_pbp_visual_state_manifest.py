#!/usr/bin/env python3
"""Build a sealed, training-only PBP-to-video visual-state manifest."""

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

from app.analysis.pbp_visual_state import (  # noqa: E402
    build_pbp_visual_state_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-artifact", type=Path, required=True)
    parser.add_argument("--play-by-play", type=Path, required=True)
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--source-slug", required=True)
    parser.add_argument(
        "--sealed-blind-acquisition",
        type=Path,
        required=True,
        help="Acquisition manifest whose blind video hashes must be rejected.",
    )
    parser.add_argument("--allowed-clock-delta-seconds", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL row {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"invalid JSONL row {line_number}")
        rows.append(row)
    if not rows:
        raise ValueError("play-by-play JSONL is empty")
    return rows


def load_sealed_blind_hashes(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("answers_opened") is not False
        or payload.get("truth_artifacts_included") is not False
    ):
        raise ValueError("sealed blind acquisition manifest is not closed")
    hashes = tuple(
        str(row.get("sha256") or "")
        for row in payload.get("media", ())
        if isinstance(row, dict) and row.get("role") == "blind_benchmark"
    )
    if not hashes:
        raise ValueError("sealed blind acquisition has no benchmark hashes")
    return hashes


def main() -> int:
    args = parse_args()
    pbp_bytes = args.play_by_play.read_bytes()
    artifact = build_pbp_visual_state_manifest(
        clock_artifact=json.loads(args.clock_artifact.read_text(encoding="utf-8")),
        pbp_rows=load_jsonl(args.play_by_play),
        pbp_sha256=hashlib.sha256(pbp_bytes).hexdigest(),
        pbp_game_id=args.game_id,
        source_slug=args.source_slug,
        allowed_clock_delta_seconds=args.allowed_clock_delta_seconds,
        sealed_blind_video_sha256s=load_sealed_blind_hashes(
            args.sealed_blind_acquisition
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
    temporary.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "artifact_sha256": artifact["artifact_sha256"],
                "summary": artifact["summary"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
