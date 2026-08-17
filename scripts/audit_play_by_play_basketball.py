#!/usr/bin/env python3
"""Seal an offline audit of the basketball annotations in Play by Play."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from app.analysis.play_by_play_dataset import build_play_by_play_basketball_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    digest = hashlib.md5(args.source_archive.read_bytes()).hexdigest()
    audit = build_play_by_play_basketball_audit(
        args.annotations,
        source_url=args.source_url,
        source_archive_md5=digest,
        source_archive_bytes=args.source_archive.stat().st_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "clip_count": audit["clip_count"],
                "frame_count": audit["frame_count"],
                "artifact_sha256": audit["artifact_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
