#!/usr/bin/env python3
"""Download a bounded, label-balanced Basketball Events shot audit subset.

The upstream release is research-only.  This script deliberately produces a
label-bearing offline audit bundle; it must not be wired into AGU runtime or
training manifests without a separate rights and causal-evidence review.
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
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

SOURCE_URL = "https://huggingface.co/datasets/saveerjain/basketball-events"
API_URL = "https://huggingface.co/api/datasets/saveerjain/basketball-events"
SOURCE_REVISION = "26d3775286f542b41daf4a94a53190440d426111"
SCHEMA_VERSION = "agu.basketball-events-shot-subset.v1"
_OUTCOME_ORDER = ("make", "miss")


def _canonical_json(payload: Mapping[str, Any], *, drop: str | None = None) -> bytes:
    normalized = dict(payload)
    if drop is not None:
        normalized.pop(drop, None)
    return json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def seal_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Add the deterministic manifest digest without mutating the input."""

    sealed = dict(payload)
    sealed["schema_version"] = SCHEMA_VERSION
    sealed["manifest_sha256"] = hashlib.sha256(
        _canonical_json(sealed, drop="manifest_sha256")
    ).hexdigest()
    return verify_manifest(sealed)


def verify_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Verify schema and the self-excluding manifest digest."""

    manifest = dict(payload)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Basketball Events shot subset schema")
    digest = manifest.get("manifest_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("manifest digest is missing or malformed")
    expected = hashlib.sha256(
        _canonical_json(manifest, drop="manifest_sha256")
    ).hexdigest()
    if digest != expected:
        raise ValueError("manifest hash mismatch")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("shot subset manifest requires files")
    return manifest


def resolve_source_url(revision: str, source_path: str) -> str:
    """Build a traversal-safe Hub resolve URL for one repository path."""

    relative = PurePosixPath(source_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"unsafe source path: {source_path}")
    encoded = urllib.parse.quote(str(relative), safe="/")
    return f"{SOURCE_URL}/resolve/{revision}/{encoded}"


def _source_size(row: Mapping[str, Any]) -> int:
    value = row.get("size", row.get("size_bytes"))
    if value is None or int(value) < 0:
        raise ValueError("source file has no valid size")
    return int(value)


def _shot_row(game_id: str, clip: Mapping[str, Any]) -> dict[str, Any] | None:
    events = clip.get("events")
    if not isinstance(events, list):
        return None
    shots = [
        event
        for event in events
        if isinstance(event, Mapping)
        and event.get("event") == "shot"
        and event.get("result") in _OUTCOME_ORDER
    ]
    # A single labeled shot gives the VLM audit a non-ambiguous target.  A
    # single shot may still carry assist/rebound events from the same clip.
    if len(shots) != 1:
        return None
    start = float(clip.get("clip_start", -1.0))
    end = float(clip.get("clip_end", -1.0))
    clip_file = str(clip.get("clip_file", ""))
    if start < 0 or end <= start or not clip_file:
        return None
    shot = shots[0]
    return {
        "game_id": game_id,
        "clip_file": clip_file,
        "clip_start": start,
        "clip_end": end,
        "duration": float(clip.get("duration", end - start)),
        "shot_result": str(shot["result"]),
        "shot_type": str(shot.get("shot_type") or "unknown"),
        "events": [dict(event) for event in events if isinstance(event, Mapping)],
    }


def _iter_annotation_rows(
    annotation_payloads: Iterable[Mapping[str, Any]],
) -> Iterable[dict[str, Any]]:
    for payload in annotation_payloads:
        game_id = str(payload.get("game_id", ""))
        clips = payload.get("clips")
        if not game_id or not isinstance(clips, list):
            continue
        for clip in clips:
            if isinstance(clip, Mapping):
                row = _shot_row(game_id, clip)
                if row is not None:
                    yield row


def select_balanced_shot_clips(
    annotation_payloads: Sequence[Mapping[str, Any]],
    source_files: Sequence[Mapping[str, Any]],
    *,
    per_game: int = 4,
    excluded_paths: set[str] | None = None,
    max_bytes: int | None = None,
) -> list[dict[str, Any]]:
    """Select non-overlapping make/miss clips evenly across games.

    ``per_game`` is split evenly between make and miss.  Selection is sorted
    by game, outcome and source start time, so the same pinned annotations and
    Hub tree always produce the same rows.  A byte budget is a hard upper
    bound; it may yield fewer rows than the requested quota.
    """

    if per_game <= 0 or per_game % 2:
        raise ValueError("per_game must be a positive even number")
    if max_bytes is not None and max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    excluded = set(excluded_paths or ())
    file_by_path: dict[str, Mapping[str, Any]] = {}
    for file_row in source_files:
        path = str(file_row.get("path", ""))
        if path.endswith(".mp4") and path and ".." not in PurePosixPath(path).parts:
            file_by_path[path] = file_row

    candidates: list[dict[str, Any]] = []
    for row in _iter_annotation_rows(annotation_payloads):
        source_path = f"clips/{row['game_id']}/{row['clip_file']}"
        file_row = file_by_path.get(source_path)
        if file_row is None or source_path in excluded:
            continue
        candidate = dict(row)
        candidate["source_path"] = source_path
        candidate["size_bytes"] = _source_size(file_row)
        candidate["source_oid"] = str(
            (file_row.get("lfs") or {}).get("oid") or file_row.get("oid") or ""
        )
        candidates.append(candidate)

    selected: list[dict[str, Any]] = []
    total_bytes = 0
    target_each = per_game // 2
    games = sorted({str(row["game_id"]) for row in candidates})
    for game_id in games:
        game_rows = [row for row in candidates if row["game_id"] == game_id]
        game_selected: list[dict[str, Any]] = []
        for outcome in _OUTCOME_ORDER:
            options = sorted(
                (row for row in game_rows if row["shot_result"] == outcome),
                key=lambda row: (float(row["clip_start"]), str(row["clip_file"])),
            )
            for row in options:
                if len([item for item in game_selected if item["shot_result"] == outcome]) >= target_each:
                    break
                if any(
                    float(row["clip_start"]) < float(item["clip_end"])
                    and float(item["clip_start"]) < float(row["clip_end"])
                    for item in game_selected
                ):
                    continue
                size = int(row["size_bytes"])
                if max_bytes is not None and total_bytes + size > max_bytes:
                    continue
                game_selected.append(row)
                selected.append(row)
                total_bytes += size
    return selected


def _get_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "agu-basketball-events-audit/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def list_source_files(revision: str) -> list[dict[str, Any]]:
    """List the pinned repository tree and keep only video files."""

    url = f"{API_URL}/tree/{revision}?recursive=true&expand=false&limit=1000"
    rows = _get_json(url)
    if not isinstance(rows, list):
        raise ValueError("Hub tree response is not a list")
    return sorted(
        [
            dict(row)
            for row in rows
            if isinstance(row, Mapping)
            and row.get("type") == "file"
            and str(row.get("path", "")).endswith(".mp4")
        ],
        key=lambda row: str(row["path"]),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str | None = None,
) -> str:
    """Download atomically, retry transient errors, and verify size/hash."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == expected_size:
        digest = _file_sha256(destination)
        if expected_sha256 and digest != expected_sha256:
            destination.unlink()
        else:
            return digest

    for attempt in range(6):
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
            ) as handle:
                temporary = Path(handle.name)
                digest = hashlib.sha256()
                size = 0
                request = urllib.request.Request(
                    url, headers={"User-Agent": "agu-basketball-events-audit/1.0"}
                )
                with urllib.request.urlopen(request, timeout=300) as response:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if size != expected_size:
                raise ValueError(f"size mismatch: {size} != {expected_size}")
            result = digest.hexdigest()
            if expected_sha256 and result != expected_sha256:
                raise ValueError(f"sha256 mismatch: {result} != {expected_sha256}")
            assert temporary is not None
            temporary.replace(destination)
            return result
        except (OSError, urllib.error.URLError, TimeoutError, ValueError):
            if attempt == 5:
                raise
            time.sleep(2**attempt)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise AssertionError("unreachable")


def _load_payloads(annotation_root: Path) -> list[dict[str, Any]]:
    payloads = []
    for path in sorted(annotation_root.glob("*_annotations.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"annotation file is not an object: {path}")
        payloads.append(payload)
    if not payloads:
        raise ValueError(f"no annotation files under {annotation_root}")
    return payloads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-revision", default=SOURCE_REVISION)
    parser.add_argument("--per-game", type=int, default=4)
    parser.add_argument("--max-bytes", type=int, default=360_000_000)
    parser.add_argument(
        "--exclude-root",
        type=Path,
        action="append",
        default=[],
        help="directory whose existing .mp4 basenames should not be downloaded",
    )
    args = parser.parse_args()

    payloads = _load_payloads(args.annotation_root)
    files = list_source_files(args.source_revision)
    excluded: set[str] = set()
    excluded_names: set[str] = set()
    for root in args.exclude_root:
        excluded_names.update(path.name for path in root.rglob("*.mp4"))
    if excluded_names:
        excluded.update(
            str(row["path"])
            for row in files
            if Path(str(row.get("path", ""))).name in excluded_names
        )
    selected = select_balanced_shot_clips(
        payloads,
        files,
        per_game=args.per_game,
        excluded_paths=excluded,
        max_bytes=args.max_bytes,
    )
    if not selected:
        raise SystemExit("no clips satisfy the balanced selection and byte budget")

    manifest_files: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        relative = Path("media") / str(row["game_id"]) / str(row["clip_file"])
        destination = args.output_root / relative
        expected_sha = row["source_oid"] or None
        sha256 = download_file(
            resolve_source_url(args.source_revision, str(row["source_path"])),
            destination,
            expected_size=int(row["size_bytes"]),
            expected_sha256=expected_sha,
        )
        manifest_files.append(
            {
                "source_path": row["source_path"],
                "output": str(relative),
                "game_id": row["game_id"],
                "clip_file": row["clip_file"],
                "clip_start": row["clip_start"],
                "clip_end": row["clip_end"],
                "duration": row["duration"],
                "shot_result": row["shot_result"],
                "shot_type": row["shot_type"],
                "events": row["events"],
                "size_bytes": int(row["size_bytes"]),
                "sha256": sha256,
                "source_oid": row["source_oid"],
            }
        )
        print(json.dumps({"downloaded": index, "total": len(selected), "file": str(relative)}), flush=True)

    manifest = seal_manifest(
        {
            "source_url": SOURCE_URL,
            "source_api_url": API_URL,
            "source_revision": args.source_revision,
            "license_terms": "For research purposes only (upstream README); no OSI or Creative Commons license is stated.",
            "purpose": "offline_external_vlm_shot_outcome_audit",
            "runtime_consumable": False,
            "training_media_eligible": False,
            "codex_runtime_answer_used": False,
            "selection": {
                "strategy": "single-shot_non-overlapping_make_miss_round_robin_v1",
                "per_game_requested": args.per_game,
                "max_bytes": args.max_bytes,
                "file_count": len(manifest_files),
                "total_bytes": sum(int(row["size_bytes"]) for row in manifest_files),
            },
            "files": manifest_files,
            "decision": "retain_offline_only_no_runtime_or_training_import",
        }
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "files": len(manifest_files),
                "bytes": manifest["selection"]["total_bytes"],
                "manifest_sha256": manifest["manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
