#!/usr/bin/env python3
"""Download and seal a bounded BasketEvent valid-trajectory audit slice."""

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
from pathlib import Path
from typing import Any

from app.analysis.basketevent_dataset import build_basketevent_trajectory_audit

SOURCE_URL = "https://huggingface.co/datasets/zaywas/BasketEvent"
API_URL = "https://huggingface.co/api/datasets/zaywas/BasketEvent"
SOURCE_REVISION = "85aaa3ce62bc096e3995c39ecfa6773fcc9fe5e1"


def _get_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "agu-basketevent-audit/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def _list_split_files(revision: str, split: str) -> list[dict[str, Any]]:
    if split not in {"valid", "test"}:
        raise ValueError("BasketEvent split must be valid or test")
    base = f"{API_URL}/tree/{revision}/{split}"
    games = _get_json(f"{base}?recursive=false&expand=false&limit=1000")
    files: list[dict[str, Any]] = []
    for game in games:
        if game.get("type") != "directory":
            continue
        rows = _get_json(
            f"{base}/{game['path'].rsplit('/', 1)[-1]}?recursive=false&expand=false&limit=1000"
        )
        files.extend(row for row in rows if row.get("type") == "file" and row.get("path", "").endswith(".json"))
    return sorted(files, key=lambda row: str(row["path"]))


def _select_bounded_files(files: list[dict[str, Any]], max_files: int) -> list[dict[str, Any]]:
    if len(files) <= max_files:
        return files
    by_game: dict[str, list[dict[str, Any]]] = {}
    for row in files:
        parts = str(row["path"]).split("/")
        game = parts[1] if len(parts) > 2 else "unknown"
        by_game.setdefault(game, []).append(row)
    selected: list[dict[str, Any]] = []
    while len(selected) < max_files:
        progressed = False
        for game in sorted(by_game):
            rows = by_game[game]
            if rows and len(selected) < max_files:
                selected.append(rows.pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def _download_file(url: str, destination: Path, expected_size: int) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == expected_size:
        digest = hashlib.sha256()
        with destination.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            for attempt in range(8):
                digest = hashlib.sha256()
                size = 0
                handle.seek(0)
                handle.truncate()
                try:
                    request = urllib.request.Request(
                        url, headers={"User-Agent": "agu-basketevent-audit/1.0"}
                    )
                    with urllib.request.urlopen(request, timeout=180) as response:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            handle.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
                    break
                except (OSError, urllib.error.URLError):
                    if attempt == 7:
                        raise
                    time.sleep(2**attempt)
            if size != expected_size:
                raise RuntimeError(f"size mismatch for {destination}: {size} != {expected_size}")
            handle.flush()
            os.fsync(handle.fileno())
            temporary.replace(destination)
            return digest.hexdigest()
        finally:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--max-files", type=int, default=700)
    parser.add_argument("--max-bytes", type=int, default=120_000_000)
    args = parser.parse_args()

    files = _select_bounded_files(
        _list_split_files(args.source_revision, args.split), args.max_files
    )
    total_bytes = sum(int(row.get("size") or 0) for row in files)
    if not files:
        raise SystemExit(f"BasketEvent {args.split} split has no JSON files")
    if total_bytes > args.max_bytes:
        raise SystemExit(f"refusing {total_bytes} bytes; max is {args.max_bytes}")

    file_manifest: list[dict[str, Any]] = []
    for index, row in enumerate(files, start=1):
        relative = Path(str(row["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(f"unsafe source path: {relative}")
        target = args.output_root / relative
        raw_url = f"{SOURCE_URL}/resolve/{args.source_revision}/{urllib.parse.quote(str(relative), safe='/')}"
        sha256 = _download_file(raw_url, target, int(row.get("size") or 0))
        file_manifest.append(
            {"path": str(relative), "bytes": int(row.get("size") or 0), "sha256": sha256}
        )
        if index % 50 == 0 or index == len(files):
            print(json.dumps({"downloaded": index, "total": len(files)}, sort_keys=True), flush=True)

    manifest: dict[str, Any] = {
        "schema_version": "agu.basketevent-download-manifest.v1",
        "source_url": SOURCE_URL,
        "source_api_url": API_URL,
        "source_revision": args.source_revision,
        "split": args.split,
        "file_count": len(file_manifest),
        "total_bytes": total_bytes,
        "files": file_manifest,
        "media_included": False,
        "runtime_consumable": False,
        "training_media_eligible": False,
        "decision": "bounded_annotation_only_no_raw_video_or_runtime_import",
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    audit = build_basketevent_trajectory_audit(
        args.output_root,
        source_url=SOURCE_URL,
        source_revision=args.source_revision,
        source_license=None,
    )
    audit["download_manifest"] = str(args.manifest)
    audit["download_manifest_sha256"] = manifest["manifest_sha256"]
    # Recompute the artifact hash after adding manifest provenance.
    audit.pop("artifact_sha256")
    encoded = json.dumps(audit, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    audit["artifact_sha256"] = hashlib.sha256(encoded).hexdigest()
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "audit_output": str(args.audit_output),
                "file_count": len(files),
                "total_bytes": total_bytes,
                "artifact_sha256": audit["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
