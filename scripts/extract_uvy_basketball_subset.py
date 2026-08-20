#!/usr/bin/env python3
"""Extract only UVY basketball images and MOT annotations by HTTP Range.

The Zenodo release is a 3.27 GB ZIP containing several sports.  This command
reads its central directory, fetches only the four basketball image/annotation
prefixes, verifies every entry against the ZIP CRC and writes no source MP4.
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import os
import re
import struct
import subprocess
import tempfile
import time
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from app.analysis.uvy_dataset import UVY_MANIFEST_SCHEMA

SOURCE_URL = "https://zenodo.org/api/records/21303900/files/UVY.zip/content"
SOURCE_RECORD_URL = "https://zenodo.org/records/21303900"
SOURCE_REVISION = "10.5281/zenodo.21303900"
SOURCE_LICENSE = "CC-BY-4.0"
SOURCE_ARCHIVE_BYTES = 3_274_165_269
SOURCE_ARCHIVE_MD5 = "f99594a1bd9f627ebe21219db86317a6"
_CENTRAL_SIGNATURE = b"PK\x01\x02"
_LOCAL_SIGNATURE = b"PK\x03\x04"
_SEQUENCE_RE = re.compile(r"^UVY/(basketball_V0[1-4])(?:/|$)")
_IMAGE_RE = re.compile(r"^UVY/(basketball_V0[1-4])/(img|img1)/\d+\.jpg$", re.IGNORECASE)
_SELECTED_SUFFIXES = {"video_info.txt", "gt/labels.txt", "gt/gt.txt"}


@dataclass(frozen=True)
class ZipEntry:
    name: str
    compressed_bytes: int
    uncompressed_bytes: int
    method: int
    crc32: int
    local_offset: int
    filename_bytes: int
    extra_bytes: int

    @property
    def data_start(self) -> int:
        return self.local_offset + 30 + self.filename_bytes + self.extra_bytes

    @property
    def data_end(self) -> int:
        # Zenodo's local headers carry a 32-byte extra field while the central
        # directory carries a 24-byte field.  Add a bounded margin so the last
        # selected payload in a Range also includes the local-header delta (and
        # the optional data descriptor); the parser still trusts central sizes.
        return self.data_start + self.compressed_bytes + 1_024


def parse_central_directory(data: bytes) -> list[ZipEntry]:
    """Parse a classic ZIP central directory captured from a range request."""

    entries: list[ZipEntry] = []
    offset = 0
    while offset < len(data):
        if data[offset : offset + 4] != _CENTRAL_SIGNATURE:
            raise ValueError(f"unexpected central-directory signature at {offset}")
        if offset + 46 > len(data):
            raise ValueError("truncated central-directory header")
        values = struct.unpack_from("<4s6H3L5H2L", data, offset)
        filename_bytes, extra_bytes, comment_bytes = values[10:13]
        end = offset + 46 + filename_bytes + extra_bytes + comment_bytes
        if end > len(data):
            raise ValueError("truncated central-directory entry")
        name = data[offset + 46 : offset + 46 + filename_bytes].decode("utf-8")
        if not name or "\x00" in name:
            raise ValueError("invalid ZIP entry name")
        entries.append(
            ZipEntry(
                name=name,
                compressed_bytes=int(values[8]),
                uncompressed_bytes=int(values[9]),
                method=int(values[4]),
                crc32=int(values[7]),
                local_offset=int(values[16]),
                filename_bytes=int(filename_bytes),
                extra_bytes=int(extra_bytes),
            )
        )
        offset = end
    if not entries:
        raise ValueError("central directory is empty")
    return entries


def _selected(entry: ZipEntry) -> tuple[str, str] | None:
    match = _SEQUENCE_RE.match(entry.name)
    if match is None:
        return None
    sequence = match.group(1)
    relative = entry.name.removeprefix("UVY/")
    if any(relative.endswith(f"/{suffix}") for suffix in _SELECTED_SUFFIXES):
        return sequence, "annotation"
    image = _IMAGE_RE.match(entry.name)
    if image is not None:
        return sequence, "image"
    return None


def _safe_relative(name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe ZIP path: {name}")
    return Path(*path.parts)


def _range_groups(entries: list[ZipEntry], *, max_gap: int) -> list[tuple[int, int, list[ZipEntry]]]:
    ordered = sorted(entries, key=lambda item: item.local_offset)
    groups: list[tuple[int, int, list[ZipEntry]]] = []
    if not ordered:
        return groups
    start = ordered[0].local_offset
    end = ordered[0].data_end - 1
    group = [ordered[0]]
    for entry in ordered[1:]:
        gap = entry.local_offset - end - 1
        if gap > max_gap:
            groups.append((start, end, group))
            start = entry.local_offset
            end = entry.data_end - 1
            group = [entry]
            continue
        group.append(entry)
        end = max(end, entry.data_end - 1)
    groups.append((start, end, group))
    return groups


def _split_range_groups(
    groups: list[tuple[int, int, list[ZipEntry]]], *, max_bytes: int
) -> list[tuple[int, int, list[ZipEntry]]]:
    """Split large groups only between selected ZIP entries."""

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    result: list[tuple[int, int, list[ZipEntry]]] = []
    for start, _end, entries in groups:
        chunk_start = start
        chunk_end = entries[0].data_end - 1
        chunk_entries = [entries[0]]
        for entry in entries[1:]:
            candidate_end = entry.data_end - 1
            if chunk_entries and candidate_end - chunk_start + 1 > max_bytes:
                result.append((chunk_start, chunk_end, chunk_entries))
                chunk_start = entry.local_offset
                chunk_end = candidate_end
                chunk_entries = [entry]
            else:
                chunk_end = candidate_end
                chunk_entries.append(entry)
        result.append((chunk_start, chunk_end, chunk_entries))
    return result


def _fetch_range(
    url: str,
    start: int,
    end: int,
    destination: Path,
    *,
    timeout_seconds: float,
    attempts: int = 6,
) -> int:
    expected = end - start + 1
    for attempt in range(attempts):
        header_path = destination.with_suffix(destination.suffix + ".headers")
        try:
            completed = subprocess.run(
                [
                    "curl",
                    "-L",
                    "--fail",
                    "--retry",
                    "3",
                    "--retry-delay",
                    "2",
                    "--connect-timeout",
                    "20",
                    "--max-time",
                    str(max(30.0, timeout_seconds)),
                    "-sS",
                    "-D",
                    str(header_path),
                    "-A",
                    "AGU-UVY-subset/1.0",
                    "-H",
                    f"Range: bytes={start}-{end}",
                    url,
                    "-o",
                    str(destination),
                ],
                check=False,
                capture_output=True,
            )
            if completed.returncode != 0:
                detail = completed.stderr.decode("utf-8", "replace").strip()
                raise RuntimeError(f"curl failed ({completed.returncode}): {detail[-500:]}")
            headers = header_path.read_text(encoding="utf-8", errors="replace")
            ranges = re.findall(r"(?im)^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)\s*$", headers)
            if not ranges:
                raise RuntimeError("UVY range response has no Content-Range header")
            actual_start, actual_end, _total = (int(value) for value in ranges[-1])
            if (actual_start, actual_end) != (start, end):
                raise RuntimeError(f"UVY range response mismatch: {actual_start}-{actual_end}")
            total = destination.stat().st_size if destination.is_file() else 0
            if total != expected:
                raise RuntimeError(f"UVY range size mismatch: {total} != {expected}")
            return total
        except (OSError, RuntimeError):
            if attempt + 1 >= attempts:
                raise
            time.sleep(min(60.0, 2.0**attempt))
        finally:
            header_path.unlink(missing_ok=True)
    raise AssertionError("unreachable")


def _existing_valid(path: Path, entry: ZipEntry) -> bool:
    if not path.is_file() or path.stat().st_size != entry.uncompressed_bytes:
        return False
    crc = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            crc = binascii.crc32(chunk, crc)
    return (crc & 0xFFFFFFFF) == entry.crc32


def _decode_and_write(entry: ZipEntry, compressed: bytes, destination: Path) -> str:
    if entry.method == 8:
        decoded = zlib.decompress(compressed, -15)
    elif entry.method == 0:
        decoded = compressed
    else:
        raise ValueError(f"unsupported UVY ZIP method {entry.method} for {entry.name}")
    if len(decoded) != entry.uncompressed_bytes:
        raise ValueError(f"UVY uncompressed size mismatch for {entry.name}")
    crc = binascii.crc32(decoded) & 0xFFFFFFFF
    if crc != entry.crc32:
        raise ValueError(f"UVY CRC mismatch for {entry.name}")
    digest = hashlib.sha256(decoded).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.uvy-tmp")
    try:
        temporary.write_bytes(decoded)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _parse_group(
    group_file: BinaryIO,
    group_start: int,
    group_end: int,
    *,
    entries_by_offset: dict[int, ZipEntry],
    selected_by_name: dict[str, ZipEntry],
    output_root: Path,
) -> set[str]:
    """Parse local entries while skipping data descriptors via central offsets."""

    offsets = sorted(offset for offset in entries_by_offset if group_start <= offset <= group_end)
    processed: set[str] = set()
    offset_index = 0
    while offset_index < len(offsets):
        local_offset = offsets[offset_index]
        if local_offset < group_start:
            offset_index += 1
            continue
        group_file.seek(local_offset - group_start)
        header = group_file.read(30)
        if len(header) != 30:
            raise ValueError(f"truncated UVY local header at {local_offset}")
        signature, _, _, _method, _, _, _, _, _, filename_bytes, extra_bytes = struct.unpack(
            "<4s5H3L2H", header
        )
        if signature != _LOCAL_SIGNATURE:
            raise ValueError(f"unexpected UVY local signature at {local_offset}")
        name_bytes = group_file.read(filename_bytes)
        group_file.seek(extra_bytes, os.SEEK_CUR)
        name = name_bytes.decode("utf-8")
        entry = entries_by_offset.get(local_offset)
        if entry is None or entry.name != name:
            raise ValueError(f"UVY local/central entry mismatch at {local_offset}")
        selected = selected_by_name.get(name)
        payload_end = group_start + group_file.tell() + entry.compressed_bytes
        if payload_end > group_end + 1:
            # A neighboring excluded MP4 may begin inside the bounded range,
            # or the next selected entry may begin at the final local header
            # of this chunk.  Its complete payload is handled by the next
            # chunk; stop before reading beyond this range.
            break
        compressed = group_file.read(entry.compressed_bytes)
        if len(compressed) != entry.compressed_bytes:
            raise ValueError(f"truncated UVY payload for {name}")
        if selected is not None:
            destination = output_root / _safe_relative(name)
            if not _existing_valid(destination, selected):
                _decode_and_write(selected, compressed, destination)
            processed.add(name)
        offset_index += 1
    return processed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_manifest(
    *,
    selected: list[ZipEntry],
    output_root: Path,
    central_path: Path,
    downloaded_range_bytes: int,
    range_request_count: int,
) -> dict[str, object]:
    sequence_entries: dict[str, list[ZipEntry]] = {}
    for entry in selected:
        match = _SEQUENCE_RE.match(entry.name)
        if match is None:
            raise ValueError(f"selected entry has no sequence: {entry.name}")
        sequence_entries.setdefault(match.group(1), []).append(entry)
    records: list[dict[str, object]] = []
    for entry in sorted(selected, key=lambda item: item.name):
        destination = output_root / _safe_relative(entry.name)
        if not destination.is_file() or destination.stat().st_size != entry.uncompressed_bytes:
            raise ValueError(f"UVY extracted file is missing or truncated: {destination}")
        records.append(
            {
                "path": entry.name,
                "bytes": entry.uncompressed_bytes,
                "sha256": _sha256_file(destination),
                "source_zip_crc32": f"{entry.crc32:08x}",
                "source_zip_compressed_bytes": entry.compressed_bytes,
                "source_zip_offset": entry.local_offset,
            }
        )
    sequences = []
    for sequence in sorted(sequence_entries):
        rows = sequence_entries[sequence]
        image_rows = [row for row in rows if _selected(row) == (sequence, "image")]
        sequences.append(
            {
                "sequence": sequence,
                "entry_count": len(rows),
                "image_file_count": len(image_rows),
                "compressed_bytes": sum(row.compressed_bytes for row in rows),
                "uncompressed_bytes": sum(row.uncompressed_bytes for row in rows),
                "image_compressed_bytes": sum(row.compressed_bytes for row in image_rows),
                "image_uncompressed_bytes": sum(row.uncompressed_bytes for row in image_rows),
            }
        )
    manifest: dict[str, object] = {
        "schema_version": UVY_MANIFEST_SCHEMA,
        "source_url": SOURCE_URL,
        "source_record_url": SOURCE_RECORD_URL,
        "source_revision": SOURCE_REVISION,
        "source_license": SOURCE_LICENSE,
        "source_archive": {
            "name": "UVY.zip",
            "bytes": SOURCE_ARCHIVE_BYTES,
            "md5": SOURCE_ARCHIVE_MD5,
            "full_archive_downloaded": False,
        },
        "central_directory_sha256": _sha256_file(central_path),
        "payload_downloads_performed": 1 if range_request_count else 0,
        "range_request_count": range_request_count,
        "range_bytes_downloaded": downloaded_range_bytes,
        "full_archive_downloaded": False,
        "media_included": True,
        "runtime_consumable": False,
        "training_media_eligible": True,
        "training_scope": "auxiliary_detector_and_hard_negative_only",
        "causal_truth_eligible": False,
        "excluded_payloads": ["all .mp4 files", "all non-basketball sequences"],
        "entry_count": len(records),
        "image_file_count": sum(len([row for row in rows if _selected(row) == (sequence, "image")]) for sequence, rows in sequence_entries.items()),
        "selected_compressed_bytes": sum(entry.compressed_bytes for entry in selected),
        "selected_uncompressed_bytes": sum(entry.uncompressed_bytes for entry in selected),
        "sequences": sequences,
        "entries": records,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--central-directory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--max-bytes", type=int, default=600_000_000)
    parser.add_argument("--max-gap", type=int, default=1_000_000)
    parser.add_argument("--chunk-bytes", type=int, default=50_000_000)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()
    if args.max_bytes <= 0 or args.max_gap < 0 or args.chunk_bytes <= 0:
        raise SystemExit("--max-bytes/--chunk-bytes and --max-gap must be positive/non-negative")
    entries = parse_central_directory(args.central_directory.read_bytes())
    selected = [entry for entry in entries if _selected(entry) is not None]
    if not selected:
        raise SystemExit("UVY central directory has no basketball entries")
    selected_by_name = {entry.name: entry for entry in selected}
    entries_by_offset = {entry.local_offset: entry for entry in entries}
    if len(entries_by_offset) != len(entries):
        raise SystemExit("UVY central directory has duplicate local offsets")
    pending = [
        entry
        for entry in selected
        if not _existing_valid(args.output_root / _safe_relative(entry.name), entry)
    ]
    groups = _split_range_groups(
        _range_groups(pending, max_gap=args.max_gap), max_bytes=args.chunk_bytes
    )
    estimated = sum(end - start + 1 for start, end, _ in groups)
    if estimated > args.max_bytes:
        raise SystemExit(f"refusing {estimated} range bytes; max is {args.max_bytes}")
    downloaded = 0
    processed: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="agu_uvy_ranges_") as temp_dir:
        for index, (start, end, _group_entries) in enumerate(groups, start=1):
            range_path = Path(temp_dir) / f"range_{index:03d}.bin"
            size = _fetch_range(
                args.source_url,
                start,
                end,
                range_path,
                timeout_seconds=args.timeout_seconds,
            )
            downloaded += size
            with range_path.open("rb") as handle:
                processed.update(
                    _parse_group(
                        handle,
                        start,
                        end,
                        entries_by_offset=entries_by_offset,
                        selected_by_name=selected_by_name,
                        output_root=args.output_root,
                    )
                )
            print(
                json.dumps(
                    {
                        "range": index,
                        "ranges": len(groups),
                        "start": start,
                        "end": end,
                        "bytes": size,
                        "selected_entries_seen": len(processed),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if set(entry.name for entry in pending) - processed:
        missing = sorted(set(entry.name for entry in pending) - processed)
        raise SystemExit(f"UVY selected entries were not processed: {missing[:3]}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = _build_manifest(
        selected=selected,
        output_root=args.output_root,
        central_path=args.central_directory,
        downloaded_range_bytes=downloaded,
        range_request_count=len(groups),
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.manifest)
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "entry_count": manifest["entry_count"],
                "image_file_count": manifest["image_file_count"],
                "range_request_count": len(groups),
                "range_bytes_downloaded": downloaded,
                "manifest_sha256": manifest["manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
