#!/usr/bin/env python3
"""Freeze and download a bounded TrackID3x3 Google Drive video subset."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
from pathlib import Path, PurePath
from typing import Any, Callable
from urllib.request import Request, urlopen

SCHEMA_VERSION = "agu.trackid3x3-drive-manifest.v1"
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
DEFAULT_ROOT_FOLDER_ID = "1aWqMwQKr5xKMjqms7-raYluSlxPsGvwX"
DEFAULT_REPOSITORY_REVISION = "9822a3dde59fc80fbb5eaacd443e79e8b059d94e"
DEFAULT_SUBSET_PATH = ("videos", "Indoor", "raw")
DEFAULT_EXPECTED_FILE_COUNT = 42
DEFAULT_EXPECTED_SIZE_BYTES = 69_690_612
DEFAULT_RESERVE_BYTES = 2 * 1024**3
USER_AGENT = "Mozilla/5.0 (compatible; AGU-Research-Audit/1.0)"
MAXIMUM_FOLDER_HTML_BYTES = 5 * 1024**2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-folder-id", default=DEFAULT_ROOT_FOLDER_ID)
    parser.add_argument(
        "--subset-path",
        default="/".join(DEFAULT_SUBSET_PATH),
        help="Exact folder path below the public Drive root.",
    )
    parser.add_argument(
        "--repository-revision",
        default=DEFAULT_REPOSITORY_REVISION,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/public_sources/trackid3x3/Indoor/raw"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "dataset/public_sources/trackid3x3/Indoor/source-manifest.json"
        ),
    )
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument(
        "--expected-file-count",
        type=int,
        default=DEFAULT_EXPECTED_FILE_COUNT,
    )
    parser.add_argument(
        "--expected-size-bytes",
        type=int,
        default=DEFAULT_EXPECTED_SIZE_BYTES,
    )
    parser.add_argument(
        "--reserve-bytes",
        type=int,
        default=DEFAULT_RESERVE_BYTES,
        help="Free disk space retained after all missing downloads.",
    )
    return parser.parse_args()


def parse_drive_folder_records(html: str) -> list[dict[str, Any]]:
    """Decode the public folder metadata embedded by Google Drive."""

    match = re.search(
        r"""window\['_DRIVE_ivd'\]\s*=\s*"""
        r"""('(?:\\.|[^'])*'|"(?:\\.|[^"])*")""",
        html,
    )
    if match is None:
        raise ValueError("Google Drive folder page lacks _DRIVE_ivd metadata")
    try:
        encoded = ast.literal_eval(match.group(1))
        payload = json.loads(encoded)
        raw_records = payload[0]
    except (IndexError, TypeError, ValueError, SyntaxError, json.JSONDecodeError) as exc:
        raise ValueError("Google Drive folder metadata is malformed") from exc
    if not isinstance(raw_records, list):
        raise ValueError("Google Drive folder metadata has no file list")

    records: list[dict[str, Any]] = []
    for raw in raw_records:
        if not isinstance(raw, list) or len(raw) < 14:
            raise ValueError("Google Drive file record is malformed")
        file_id, name, mime_type, size_bytes = raw[0], raw[2], raw[3], raw[13]
        if not all(isinstance(value, str) and value for value in (file_id, name, mime_type)):
            raise ValueError("Google Drive file record lacks identity metadata")
        if size_bytes is not None and (
            not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0
        ):
            raise ValueError("Google Drive file record has an invalid size")
        records.append(
            {
                "file_id": file_id,
                "name": name,
                "mime_type": mime_type,
                "size_bytes": size_bytes,
            }
        )
    return records


def fetch_drive_folder_html(folder_id: str) -> str:
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        body = response.read(MAXIMUM_FOLDER_HTML_BYTES + 1)
    if len(body) > MAXIMUM_FOLDER_HTML_BYTES:
        raise RuntimeError("Google Drive folder page exceeded the audit size limit")
    return body.decode("utf-8", "replace")


def build_subset_manifest(
    *,
    root_folder_id: str,
    subset_path: tuple[str, ...],
    repository_revision: str,
    fetch_html: Callable[[str], str] = fetch_drive_folder_html,
) -> dict[str, Any]:
    """Resolve one exact public folder and freeze its video file metadata."""

    if not root_folder_id.strip() or not repository_revision.strip() or not subset_path:
        raise ValueError("root folder, subset path and repository revision are required")
    folder_id = root_folder_id
    for component in subset_path:
        if not component or PurePath(component).name != component:
            raise ValueError(f"unsafe subset path component: {component!r}")
        records = parse_drive_folder_records(fetch_html(folder_id))
        matches = [
            record
            for record in records
            if record["name"] == component
            and record["mime_type"] == FOLDER_MIME_TYPE
        ]
        if len(matches) != 1:
            raise ValueError(
                f"subset component {component!r} must resolve to exactly one folder"
            )
        folder_id = str(matches[0]["file_id"])

    records = parse_drive_folder_records(fetch_html(folder_id))
    if not records:
        raise ValueError("TrackID3x3 subset folder is empty")
    if any(not str(record["mime_type"]).startswith("video/") for record in records):
        raise ValueError("TrackID3x3 subset contains a non-video item")

    names: set[str] = set()
    files: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: str(item["name"])):
        name = str(record["name"])
        if PurePath(name).name != name or name in {".", ".."}:
            raise ValueError(f"unsafe TrackID3x3 file name: {name!r}")
        if name in names:
            raise ValueError(f"duplicate TrackID3x3 file name: {name}")
        names.add(name)
        size_bytes = record["size_bytes"]
        if not isinstance(size_bytes, int) or size_bytes <= 0:
            raise ValueError(f"TrackID3x3 video lacks a positive declared size: {name}")
        files.append(
            {
                "file_id": record["file_id"],
                "name": name,
                "mime_type": record["mime_type"],
                "declared_size_bytes": size_bytes,
                "sha256": None,
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "trackid3x3",
        "source_url": (
            f"https://drive.google.com/drive/folders/{root_folder_id}"
        ),
        "source_repository": "https://github.com/open-starlab/TrackID3x3",
        "repository_revision": repository_revision,
        "declared_data_license": "CC-BY-4.0",
        "subset_path": "/".join(subset_path),
        "subset_folder_id": folder_id,
        "training_only": True,
        "runtime_consumable": False,
        "acceptance_media_eligible": False,
        "file_count": len(files),
        "declared_size_bytes": sum(
            int(item["declared_size_bytes"]) for item in files
        ),
        "files": files,
    }
    return seal_manifest(manifest)


def seal_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload["manifest_sha256"] = digest
    return payload


def verify_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = dict(manifest)
    claimed_hash = str(payload.pop("manifest_sha256", ""))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported TrackID3x3 manifest schema")
    expected_hash = seal_manifest(payload)["manifest_sha256"]
    if claimed_hash != expected_hash:
        raise ValueError("TrackID3x3 manifest hash mismatch")
    payload["manifest_sha256"] = claimed_hash
    return payload


def preserve_completed_file_seals(
    current: dict[str, Any],
    previous: dict[str, Any],
) -> dict[str, Any]:
    """Keep verified local hashes when a refreshed public listing is identical."""

    if previous.get("download_complete") is not True:
        return current
    previous = verify_manifest(previous)
    identity_keys = ("schema_version", "repository_revision", "subset_folder_id")
    if any(current.get(key) != previous.get(key) for key in identity_keys):
        raise ValueError("completed TrackID3x3 manifest does not match current source")
    current_files = current.get("files")
    previous_files = previous.get("files")
    if not isinstance(current_files, list) or not isinstance(previous_files, list):
        raise ValueError("completed TrackID3x3 manifest does not match current source")
    current_identity = [
        (item.get("file_id"), item.get("name"), item.get("declared_size_bytes"))
        for item in current_files
    ]
    previous_identity = [
        (item.get("file_id"), item.get("name"), item.get("declared_size_bytes"))
        for item in previous_files
    ]
    if current_identity != previous_identity:
        raise ValueError("completed TrackID3x3 manifest does not match current source")

    refreshed = dict(current)
    refreshed_files = []
    for current_item, previous_item in zip(
        current_files,
        previous_files,
        strict=True,
    ):
        sha256 = previous_item.get("sha256")
        local_size = previous_item.get("local_size_bytes")
        if (
            not isinstance(sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
            or local_size != current_item["declared_size_bytes"]
        ):
            raise ValueError("completed TrackID3x3 manifest has an invalid file seal")
        refreshed_files.append(
            {
                **current_item,
                "sha256": sha256,
                "local_size_bytes": local_size,
            }
        )
    refreshed["files"] = refreshed_files
    refreshed["download_complete"] = True
    return seal_manifest(refreshed)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(seal_manifest(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_manifest_files(
    manifest: dict[str, Any],
    *,
    output_dir: Path,
    manifest_path: Path,
    reserve_bytes: int,
) -> dict[str, Any]:
    manifest = verify_manifest(manifest)
    if reserve_bytes < 0:
        raise ValueError("reserve bytes must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_bytes = 0
    for item in manifest["files"]:
        target = output_dir / item["name"]
        if target.exists():
            if target.stat().st_size != item["declared_size_bytes"]:
                raise ValueError(f"existing TrackID3x3 video has wrong size: {target}")
        else:
            missing_bytes += int(item["declared_size_bytes"])
    free_bytes = shutil.disk_usage(output_dir).free
    if free_bytes < missing_bytes + reserve_bytes:
        raise RuntimeError(
            "insufficient disk space for TrackID3x3 subset: "
            f"free={free_bytes}, required={missing_bytes + reserve_bytes}"
        )

    for index, item in enumerate(manifest["files"], start=1):
        target = output_dir / item["name"]
        if not target.exists():
            _download_drive_file(
                file_id=str(item["file_id"]),
                target=target,
                expected_size=int(item["declared_size_bytes"]),
            )
        actual_sha256 = sha256_file(target)
        expected_sha256 = item.get("sha256")
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise ValueError(f"TrackID3x3 video hash mismatch: {target}")
        item["sha256"] = actual_sha256
        item["local_size_bytes"] = target.stat().st_size
        manifest = seal_manifest(manifest)
        write_manifest(manifest_path, manifest)
        print(
            f"[{index}/{manifest['file_count']}] sealed {target.name} "
            f"{target.stat().st_size} bytes",
            flush=True,
        )
    manifest["download_complete"] = True
    manifest = seal_manifest(manifest)
    write_manifest(manifest_path, manifest)
    return manifest


def _download_drive_file(*, file_id: str, target: Path, expected_size: int) -> None:
    part = target.with_suffix(f"{target.suffix}.part")
    existing = part.stat().st_size if part.exists() else 0
    if existing == expected_size:
        os.replace(part, target)
        return
    if existing > expected_size:
        raise RuntimeError(
            f"partial download exceeds declared size for {target.name}"
        )
    headers = {"User-Agent": USER_AGENT}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    url = (
        "https://drive.usercontent.google.com/download"
        f"?id={file_id}&export=download&confirm=t"
    )
    request = Request(url, headers=headers)
    with urlopen(request, timeout=60) as response:
        append = existing > 0 and getattr(response, "status", None) == 206
        mode = "ab" if append else "wb"
        with part.open(mode) as handle:
            maximum_bytes = expected_size - existing if append else expected_size
            copy_response_bounded(
                response,
                handle,
                maximum_bytes=maximum_bytes,
            )
    actual_size = part.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"download size mismatch for {target.name}: "
            f"expected={expected_size}, actual={actual_size}"
        )
    os.replace(part, target)


def copy_response_bounded(
    response: Any,
    destination: Any,
    *,
    maximum_bytes: int,
) -> int:
    if maximum_bytes < 0:
        raise ValueError("maximum response bytes must be non-negative")
    copied = 0
    while True:
        block = response.read(min(1024 * 1024, maximum_bytes - copied + 1))
        if not block:
            return copied
        if copied + len(block) > maximum_bytes:
            raise RuntimeError("download response exceeded its declared size")
        destination.write(block)
        copied += len(block)


def main() -> int:
    args = parse_args()
    subset_path = tuple(part for part in args.subset_path.split("/") if part)
    manifest = build_subset_manifest(
        root_folder_id=args.root_folder_id,
        subset_path=subset_path,
        repository_revision=args.repository_revision,
    )
    if (
        manifest["file_count"] != args.expected_file_count
        or manifest["declared_size_bytes"] != args.expected_size_bytes
    ):
        raise RuntimeError(
            "TrackID3x3 public subset drifted: "
            f"files={manifest['file_count']}, "
            f"bytes={manifest['declared_size_bytes']}"
        )
    if args.manifest.is_file():
        previous = json.loads(args.manifest.read_text(encoding="utf-8"))
        manifest = preserve_completed_file_seals(manifest, previous)
    write_manifest(args.manifest, manifest)
    if args.manifest_only:
        print(
            f"manifest sealed: {manifest['file_count']} files, "
            f"{manifest['declared_size_bytes']} bytes",
            flush=True,
        )
        return 0
    download_manifest_files(
        manifest,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        reserve_bytes=args.reserve_bytes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
