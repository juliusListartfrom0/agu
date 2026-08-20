#!/usr/bin/env python3
"""Seal browser-resolved BARD event MP4 URLs against a frozen plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.analysis.bard_event_state import (
    seal_bard_video_resolution_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--video-urls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    video_urls = json.loads(args.video_urls.read_text(encoding="utf-8"))
    if not isinstance(video_urls, dict):
        raise ValueError("BARD video URL input must be an object")
    manifest = seal_bard_video_resolution_manifest(
        plan=plan,
        video_urls=video_urls,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "clips": len(manifest["clips"]),
                "resolution_sha256": manifest["resolution_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
