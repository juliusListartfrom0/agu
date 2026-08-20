#!/usr/bin/env python3
"""Download and seal the tiny NSVA basketball video-text metadata subset.

Only the two generated Parquet tables are fetched.  NSVA's original NBA clips
are not downloaded here: the dataset card points to the upstream collection
tools and the underlying broadcast rights are not granted by this metadata
snapshot.  The resulting audit is for event-ontology and VLM-contract design
only, never AGU runtime, blind truth, or training-media promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from app.analysis.nsva_event_text import (
    build_nsva_event_text_audit,
    verify_nsva_event_text_audit,
)

SOURCE_URL = "https://huggingface.co/datasets/sportsvision/nsva_subset"
SOURCE_REVISION = "97c211f85beec54209b44faea2bff73544a16125"
LICENSE = "fair-noncommercial-research-license"
SCHEMA_VERSION = "agu.nsva-subset-download.v1"
PARQUET_PATHS = (
    "data/train-00000-of-00001-79578115335a4f50.parquet",
    "data/validation-00000-of-00001-893230b2272b263b.parquet",
)


def _canonical_json(payload: Mapping[str, Any], *, drop: str | None = None) -> bytes:
    normalized = dict(payload)
    if drop is not None:
        normalized.pop(drop, None)
    return json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def resolve_source_url(revision: str, source_path: str, *, base_url: str = SOURCE_URL) -> str:
    """Build a traversal-safe pinned Hub resolve URL."""

    relative = PurePosixPath(source_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"unsafe NSVA source path: {source_path}")
    encoded = urllib.parse.quote(str(relative), safe="/")
    return f"{base_url.rstrip('/')}/resolve/{revision}/{encoded}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(
    url: str,
    destination: Path,
    *,
    max_bytes: int,
    expected_size: int | None = None,
) -> tuple[int, str]:
    """Download atomically with a hard byte ceiling and a resumable target."""

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        size = destination.stat().st_size
        if size <= max_bytes and (expected_size is None or size == expected_size):
            return size, _sha256(destination)
        destination.unlink()

    for attempt in range(6):
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
            ) as handle:
                temporary = Path(handle.name)
                request = urllib.request.Request(
                    url, headers={"User-Agent": "agu-nsva-metadata-audit/1.0"}
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    content_length = response.headers.get("Content-Length")
                    if content_length is not None and int(content_length) > max_bytes:
                        raise ValueError(f"NSVA payload exceeds byte ceiling: {content_length}")
                    digest = hashlib.sha256()
                    size = 0
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > max_bytes:
                            raise ValueError("NSVA payload exceeded byte ceiling while reading")
                        handle.write(chunk)
                        digest.update(chunk)
                if expected_size is not None and size != expected_size:
                    raise ValueError(f"NSVA size mismatch: {size} != {expected_size}")
                handle.flush()
                os.fsync(handle.fileno())
                temporary.replace(destination)
                return size, digest.hexdigest()
        except (OSError, urllib.error.URLError):
            if attempt == 5:
                raise
            time.sleep(2**attempt)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise AssertionError("unreachable")


def seal_download_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(payload)
    manifest["schema_version"] = SCHEMA_VERSION
    manifest["manifest_sha256"] = hashlib.sha256(_canonical_json(manifest)).hexdigest()
    return verify_download_manifest(manifest)


def verify_download_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(payload)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported NSVA download manifest schema")
    digest = manifest.get("manifest_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("NSVA manifest hash is missing or malformed")
    expected = hashlib.sha256(_canonical_json(manifest, drop="manifest_sha256")).hexdigest()
    if digest != expected:
        raise ValueError("NSVA manifest hash mismatch")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("NSVA manifest requires at least one file")
    return manifest


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("polars is required to read NSVA Parquet metadata") from exc
    frame = pl.read_parquet(path)
    required = {"video_path", "intent"}
    if not required.issubset(set(frame.columns)):
        raise ValueError(f"NSVA table missing columns: {sorted(required - set(frame.columns))}")
    return [
        {"video_path": str(row["video_path"]), "intent": str(row["intent"])}
        for row in frame.select(["video_path", "intent"]).to_dicts()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("dataset/public_sources/nsva_subset_v1"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset/public_sources/nsva_subset_v1/source-manifest.json"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("analysis_outputs/public_research/nsva_event_text_audit_v1.json"),
    )
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--base-url", default=SOURCE_URL)
    parser.add_argument("--max-bytes", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.max_bytes <= 0:
        raise SystemExit("--max-bytes must be positive")

    file_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    total_bytes = 0
    for source_path in PARQUET_PATHS:
        destination = args.output_root / Path(source_path)
        url = resolve_source_url(args.source_revision, source_path, base_url=args.base_url)
        size, digest = _download_file(url, destination, max_bytes=args.max_bytes)
        total_bytes += size
        rows = _read_rows(destination)
        all_rows.extend(rows)
        file_rows.append({"path": source_path, "bytes": size, "sha256": digest, "row_count": len(rows)})

    manifest = seal_download_manifest(
        {
            "source_url": SOURCE_URL,
            "source_revision": args.source_revision,
            "license": LICENSE,
            "files": file_rows,
            "total_bytes": total_bytes,
            "row_count": len(all_rows),
            "media_downloaded": False,
            "runtime_consumable": False,
            "training_media_eligible": False,
            "decision": "metadata_only_no_raw_video_or_runtime_import",
        }
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    audit = build_nsva_event_text_audit(
        all_rows, source_url=SOURCE_URL, source_revision=args.source_revision
    )
    audit["download_manifest"] = str(args.manifest)
    audit["download_manifest_sha256"] = manifest["manifest_sha256"]
    audit.pop("audit_sha256")
    audit["audit_sha256"] = hashlib.sha256(_canonical_json(audit)).hexdigest()
    verify_nsva_event_text_audit(audit)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "audit_output": str(args.audit_output),
                "row_count": len(all_rows),
                "total_bytes": total_bytes,
                "audit_sha256": audit["audit_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
