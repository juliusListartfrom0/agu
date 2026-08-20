#!/usr/bin/env python3
"""Download only the licensed MUVY basketball videos and annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.remote_zip_subset import (  # noqa: E402
    locate_remote_zip_directory,
    parse_remote_zip_entries,
    read_remote_zip_entry,
    subset_relative_path,
)

RECORD_ID = 13_883_315
RECORD_URL = f"https://zenodo.org/records/{RECORD_ID}"
ARCHIVE_URL = (
    f"https://zenodo.org/api/records/{RECORD_ID}/files/"
    "MUVY_v01.zip/content"
)
ARCHIVE_SIZE = 7_758_643_869
ARCHIVE_MD5 = "607051346013728c1e14dcdd94f319c1"
ARCHIVE_PREFIX = "MUVY_v01/sport_events/basketball/"
TAIL_BYTES = 65_536


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def download_subset(*, output: Path) -> dict[str, Any]:
    observed_size = _remote_size(ARCHIVE_URL)
    if observed_size != ARCHIVE_SIZE:
        raise ValueError("MUVY archive size changed from the audited record")
    tail_start = max(0, observed_size - TAIL_BYTES)
    tail = _fetch_range(ARCHIVE_URL, tail_start, observed_size - 1)
    directory = locate_remote_zip_directory(
        tail,
        tail_start=tail_start,
        archive_size=observed_size,
    )
    central = _fetch_range(
        ARCHIVE_URL,
        directory.offset,
        directory.offset + directory.size - 1,
    )
    entries = parse_remote_zip_entries(
        central,
        expected_count=directory.entry_count,
    )
    selected = []
    for entry in entries:
        relative = subset_relative_path(
            entry.name,
            prefix=ARCHIVE_PREFIX,
            allowed_names={"detections_info.txt", "video_info.txt"},
            allowed_suffixes={".mp4"},
        )
        if relative is not None:
            selected.append((entry, relative))
    if (
        len(selected) != 39
        or sum(relative.suffix == ".mp4" for _, relative in selected) != 13
    ):
        raise ValueError("MUVY basketball subset membership changed")

    files = []
    for index, (entry, relative) in enumerate(selected, start=1):
        payload = read_remote_zip_entry(
            entry,
            fetch_range=lambda start, end: _fetch_range(
                ARCHIVE_URL,
                start,
                end,
            ),
        )
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(payload).hexdigest()
        if target.exists():
            if (
                target.stat().st_size == len(payload)
                and _file_sha256(target) == digest
            ):
                status = "verified_existing"
            else:
                raise ValueError(f"MUVY subset file conflicts: {relative}")
        else:
            temporary = target.with_suffix(f"{target.suffix}.part")
            temporary.write_bytes(payload)
            temporary.replace(target)
            status = "downloaded"
        files.append(
            {
                "path": relative.as_posix(),
                "size_bytes": len(payload),
                "sha256": digest,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "muvy_basketball_subset",
                    "file": index,
                    "file_count": len(selected),
                    "path": relative.as_posix(),
                    "status": status,
                }
            ),
            flush=True,
        )
    artifact: dict[str, Any] = {
        "schema_version": "agu.muvy-basketball-subset.v1",
        "purpose": "licensed_small_ball_detection_pretraining_audit",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "source": {
            "record_id": RECORD_ID,
            "record_url": RECORD_URL,
            "license": "CC-BY-4.0",
            "archive_url": ARCHIVE_URL,
            "archive_filename": "MUVY_v01.zip",
            "archive_size_bytes": ARCHIVE_SIZE,
            "archive_md5": ARCHIVE_MD5,
            "archive_prefix": ARCHIVE_PREFIX,
            "central_directory_size_bytes": directory.size,
            "central_directory_entry_count": directory.entry_count,
        },
        "subset": {
            "file_count": len(files),
            "video_count": sum(
                row["path"].lower().endswith(".mp4") for row in files
            ),
            "annotation_count": sum(
                row["path"].endswith("detections_info.txt") for row in files
            ),
            "metadata_count": sum(
                row["path"].endswith("video_info.txt") for row in files
            ),
            "size_bytes": sum(row["size_bytes"] for row in files),
        },
        "files": sorted(files, key=lambda row: row["path"]),
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    manifest = output / "manifest.json"
    temporary_manifest = manifest.with_suffix(".json.part")
    temporary_manifest.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest)
    return artifact


def _remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        value = response.headers.get("Content-Length")
    if value is None:
        raise ValueError("remote MUVY archive omitted Content-Length")
    return int(value)


def _fetch_range(url: str, start: int, end: int) -> bytes:
    if start < 0 or end < start:
        raise ValueError("invalid MUVY byte range")
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes={start}-{end}"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or not content_range.startswith(
            f"bytes {start}-{end}/"
        ):
            raise ValueError("MUVY server did not honor the exact byte range")
    if len(payload) != end - start + 1:
        raise ValueError("MUVY byte-range response is truncated")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    args = parse_args()
    artifact = download_subset(output=args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "file_count": artifact["subset"]["file_count"],
                "size_bytes": artifact["subset"]["size_bytes"],
                "artifact_sha256": artifact["artifact_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
