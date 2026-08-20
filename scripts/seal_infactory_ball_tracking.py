#!/usr/bin/env python3
"""Seal the metadata-referenced Infactory tiny-ball subset for offline research."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.analysis.infactory_ball_tracking import (
    build_infactory_ball_tracking_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-clip-count", type=int, default=2)
    parser.add_argument("--test-clip-count", type=int, default=2)
    parser.add_argument("--split-seed", default="infactory-ball-tracking-v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_infactory_ball_tracking_manifest(
        args.root,
        source_revision=args.source_revision,
        validation_clip_count=args.validation_clip_count,
        test_clip_count=args.test_clip_count,
        split_seed=args.split_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "artifact_sha256": manifest["artifact_sha256"],
                "rows": sum(manifest["row_counts"].values()),
                "box_count": manifest["box_count"],
                "split_counts": manifest["split_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
