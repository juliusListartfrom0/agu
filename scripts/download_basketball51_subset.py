#!/usr/bin/env python3
"""Download a bounded, game-grouped Basketball-51 subset via ZIP byte ranges."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.basketball51 import (  # noqa: E402
    BASKETBALL51_LABELS,
    locate_zip_central_directory,
    materialize_basketball51_subset,
    parse_zip_central_directory,
    select_balanced_entries,
)

DEFAULT_ARCHIVE_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "sarbagyashakya/basketball-51-dataset"
)
_CONTENT_RANGE = re.compile(r"bytes\s+\d+-\d+/(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--archive-url", default=DEFAULT_ARCHIVE_URL)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--maximum-download-bytes",
        type=int,
        default=512 * 1024 * 1024,
    )
    return parser.parse_args()


def inspect_remote_archive(
    archive_url: str,
    *,
    timeout_seconds: float,
    retries: int,
) -> tuple[str, int, str, bytes]:
    """Resolve the signed archive URL and fetch only its central directory."""

    _prefix, headers, resolved_url = read_remote_range(
        archive_url,
        0,
        0,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    content_range = str(headers.get("Content-Range") or "")
    match = _CONTENT_RANGE.fullmatch(content_range)
    if match is None:
        raise ValueError("remote archive did not return a bounded Content-Range")
    archive_size = int(match.group(1))
    archive_etag = str(
        headers.get("ETag")
        or headers.get("X-Goog-Generation")
        or ""
    ).strip('"')
    if not archive_etag:
        raise ValueError("remote archive does not expose an immutable identity")
    tail_size = min(65536, archive_size)
    tail_start = archive_size - tail_size
    tail, _headers, _resolved = read_remote_range(
        resolved_url,
        tail_start,
        archive_size - 1,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    central_offset, central_size, entry_count = locate_zip_central_directory(
        tail,
        tail_start=tail_start,
    )
    central, _headers, _resolved = read_remote_range(
        resolved_url,
        central_offset,
        central_offset + central_size - 1,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    entries = parse_zip_central_directory(central)
    if len(entries) != entry_count:
        raise ValueError("remote ZIP entry count does not match its directory")
    return resolved_url, archive_size, archive_etag, central


def read_remote_range(
    url: str,
    start: int,
    end: int,
    *,
    timeout_seconds: float,
    retries: int,
) -> tuple[bytes, dict[str, str], str]:
    """Read exactly one HTTP byte range and fail closed if Range is ignored."""

    if start < 0 or end < start or timeout_seconds <= 0 or retries < 0:
        raise ValueError("invalid remote byte-range request")
    expected = end - start + 1
    request = Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "agu-public-research/1.0",
        },
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                if response.status != 206:
                    raise ValueError(
                        "remote server ignored Range; refusing an unbounded download"
                    )
                payload = response.read(expected + 1)
                if len(payload) != expected:
                    raise ValueError("remote byte-range length mismatch")
                return payload, dict(response.headers.items()), response.geturl()
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(2**attempt, 8))
    raise ValueError(f"remote byte-range request failed: {last_error}")


def main() -> int:
    args = parse_args()
    if (
        args.per_class <= 0
        or args.maximum_download_bytes <= 0
        or args.retries < 0
    ):
        raise ValueError("download limits must be positive")
    resolved_url, archive_size, archive_etag, central = inspect_remote_archive(
        args.archive_url,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
    )
    entries = parse_zip_central_directory(central)
    labels = {entry.label for entry in entries}
    if labels != set(BASKETBALL51_LABELS):
        raise ValueError("Basketball-51 archive does not contain all eight labels")
    selected = select_balanced_entries(
        entries,
        per_class=args.per_class,
        seed=args.seed,
    )
    selected_bytes = sum(entry.uncompressed_size for entry in selected)
    if selected_bytes > args.maximum_download_bytes:
        raise ValueError(
            f"selected subset exceeds byte cap: {selected_bytes} > "
            f"{args.maximum_download_bytes}"
        )
    request_count = 0
    transferred_bytes = 0

    def read_range(start: int, end: int) -> bytes:
        nonlocal request_count, transferred_bytes
        payload, _headers, _resolved = read_remote_range(
            resolved_url,
            start,
            end,
            timeout_seconds=args.timeout_seconds,
            retries=args.retries,
        )
        request_count += 1
        transferred_bytes += len(payload)
        if request_count % 32 == 0:
            print(
                f"range requests={request_count} "
                f"transferred={transferred_bytes / (1024 * 1024):.1f}MiB",
                flush=True,
            )
        return payload

    manifest = materialize_basketball51_subset(
        selected,
        output_dir=args.output_dir,
        read_range=read_range,
        archive_size=archive_size,
        archive_etag=archive_etag,
        per_class=args.per_class,
        seed=args.seed,
    )
    manifest_path = args.output_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "artifact_sha256": manifest["artifact_sha256"],
                "clip_count": manifest["clip_count"],
                "source_group_count": manifest["source_group_count"],
                "selected_bytes": selected_bytes,
                "range_requests": request_count,
                "transferred_bytes": transferred_bytes,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
