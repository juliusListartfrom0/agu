#!/usr/bin/env python3
"""Seal an offline integrity audit for the local NBA Games metadata mirror."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.analysis.nba_games_local_mirror import build_nba_games_local_mirror_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--generated-on", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = build_nba_games_local_mirror_audit(
        args.index,
        args.mirror_root,
        generated_on=args.generated_on,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "audit_sha256": artifact["audit_sha256"],
                "index_game_count": artifact["index_game_count"],
                "play_by_play_rows": artifact["play_by_play_rows"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
