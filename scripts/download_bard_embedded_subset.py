#!/usr/bin/env python3
"""Download exact Git blobs from a frozen BARD embedded-video plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

from app.analysis.bard_event_state import materialize_bard_embedded_subset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument(
        "--max-clip-bytes",
        type=int,
        default=16 * 1024 * 1024,
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if args.max_clip_bytes <= 0 or args.timeout_seconds <= 0:
        raise ValueError("BARD embedded download limits must be positive")

    def fetch_video(url: str) -> bytes:
        request = Request(
            url,
            headers={
                "Accept": "application/octet-stream",
                "User-Agent": "AGU-offline-research/1.0",
            },
        )
        with urlopen(request, timeout=args.timeout_seconds) as response:
            declared = int(response.headers.get("Content-Length") or 0)
            if declared > args.max_clip_bytes:
                raise ValueError(
                    "BARD embedded clip exceeds the declared size limit"
                )
            payload = response.read(args.max_clip_bytes + 1)
        if len(payload) > args.max_clip_bytes:
            raise ValueError(
                "BARD embedded clip exceeds the download size limit"
            )
        return payload

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    subset = materialize_bard_embedded_subset(
        plan=plan,
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
