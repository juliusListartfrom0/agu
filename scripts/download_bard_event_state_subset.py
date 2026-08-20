#!/usr/bin/env python3
"""Download a bounded, hash-sealed BARD event-state research subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

from app.analysis.bard_event_state import (
    materialize_bard_event_state_subset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--resolution-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument(
        "--max-clip-bytes",
        type=int,
        default=32 * 1024 * 1024,
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()
    if args.max_clip_bytes <= 0 or args.timeout_seconds <= 0:
        raise ValueError("BARD download limits must be positive")

    def fetch_video(url: str) -> bytes:
        request = Request(
            url,
            headers={
                "Accept": "video/mp4,video/*;q=0.9,*/*;q=0.1",
                "Referer": "https://www.nba.com/",
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 Chrome/138.0.0.0 Safari/537.36"
                ),
            },
        )
        with urlopen(request, timeout=args.timeout_seconds) as response:
            declared = int(response.headers.get("Content-Length") or 0)
            if declared > args.max_clip_bytes:
                raise ValueError("BARD clip exceeds the declared size limit")
            payload = response.read(args.max_clip_bytes + 1)
        if len(payload) > args.max_clip_bytes:
            raise ValueError("BARD clip exceeds the download size limit")
        return payload

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    resolution = json.loads(
        args.resolution_manifest.read_text(encoding="utf-8")
    )
    subset = materialize_bard_event_state_subset(
        plan=plan,
        resolution_manifest=resolution,
        output_dir=args.output_dir,
        fetch_video=fetch_video,
    )
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest_output.with_suffix(
        args.manifest_output.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(subset, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.manifest_output)
    print(
        json.dumps(
            {
                "output": str(args.manifest_output),
                "clips": len(subset["clips"]),
                "bytes": sum(row["size_bytes"] for row in subset["clips"]),
                "subset_sha256": subset["subset_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
