#!/usr/bin/env python3
"""Remove explicitly rejected auxiliary pixel payloads with a hash-bound audit.

The allowlist is intentionally narrow: only source pixels/derived YOLO files
from already rejected auxiliary screens may be removed.  Manifests, source
metadata, licenses, central directories, audit outputs, runtime assets and
the three Wikimedia originals are outside the target set.  A dry run is the
default; ``--execute`` is required for deletion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

TARGETS = (
    ROOT / "dataset/public_sources/infactory_ball_tracking_v1/data",
    ROOT / "dataset/public_sources/infactory_ball_tracking_v1/yolo/images",
    ROOT / "dataset/public_sources/infactory_ball_tracking_v1/yolo/labels",
    ROOT / "dataset/public_sources/apidis_ball_metadata_v1/raw",
    ROOT / "dataset/public_sources/uvy_v1/UVY",
    ROOT / "dataset/public_sources/uvy_v1/yolo_aux_v1",
)

PROTECTED = (
    ROOT / "dataset/public_sources/infactory_ball_tracking_v1/manifest.json",
    ROOT / "dataset/public_sources/infactory_ball_tracking_v1/metadata.csv",
    ROOT / "dataset/public_sources/infactory_ball_tracking_v1/yolo/manifest.json",
    ROOT / "dataset/public_sources/apidis_ball_metadata_v1/manifest.json",
    ROOT / "dataset/public_sources/uvy_v1/manifest.json",
    ROOT / "dataset/public_sources/uvy_v1/central-directory.bin",
    ROOT / "dataset/public_sources/wikimedia_hctv_fullgame_v1/raw",
    ROOT / "dataset/public_sources/wikimedia_hctv_randolph_v1/raw",
    ROOT / "dataset/public_sources/wikimedia_vtv_fullgame_v1/raw",
    ROOT / "dataset/public_sources/open_models/ebqwen25_vl_3b_native",
    ROOT / "dataset/public_sources/open_models/ebqwen25_vl_3b_mlx_4bit",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"symlink is not an allowed cleanup target: {path}")
    if not path.exists():
        return {
            "path": path.relative_to(ROOT).as_posix(),
            "exists": False,
            "file_count": 0,
            "bytes": 0,
            "inventory_sha256": hashlib.sha256(b"").hexdigest(),
        }
    files = [
        child
        for child in sorted(path.rglob("*"))
        if child.is_file()
    ] if path.is_dir() else [path]
    digest = hashlib.sha256()
    total_bytes = 0
    for child in files:
        relative = child.relative_to(ROOT).as_posix()
        size = child.stat().st_size
        child_sha = _sha256(child)
        digest.update(f"{relative}\t{size}\t{child_sha}\n".encode())
        total_bytes += size
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "exists": True,
        "file_count": len(files),
        "bytes": total_bytes,
        "inventory_sha256": digest.hexdigest(),
    }


def _protected_inventory() -> list[dict[str, Any]]:
    return [_inventory(path) for path in PROTECTED]


def _remove_target(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    if path.is_symlink():
        raise ValueError(f"symlink is not an allowed cleanup target: {path}")
    files = [child for child in path.rglob("*") if child.is_file()]
    deleted_files = 0
    deleted_bytes = 0
    for child in files:
        deleted_bytes += child.stat().st_size
        child.unlink()
        deleted_files += 1
    for directory in sorted(
        (child for child in path.rglob("*") if child.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        directory.rmdir()
    path.rmdir()
    return deleted_files, deleted_bytes


def build_audit(*, execute: bool) -> dict[str, Any]:
    before = [_inventory(path) for path in TARGETS]
    protected_before = _protected_inventory()
    deleted_files = 0
    deleted_bytes = 0
    if execute:
        for path in TARGETS:
            files, size = _remove_target(path)
            deleted_files += files
            deleted_bytes += size
        after = [_inventory(path) for path in TARGETS]
        if any(item["exists"] for item in after):
            raise RuntimeError("an allowlisted auxiliary payload remains")
        protected_after = _protected_inventory()
        if protected_after != protected_before:
            raise RuntimeError("a protected source/model changed during cleanup")
    else:
        after = before
        protected_after = protected_before
    audit: dict[str, Any] = {
        "schema_version": "agu.rejected-auxiliary-payload-cleanup.v1",
        "created_on": datetime.now(timezone.utc).isoformat(),
        "purpose": "remove_rejected_auxiliary_pixels_keep_provenance_and_runtime_assets",
        "authorization": "user_authorized_online_download_and_deletion_of_non_helpful_agu_data",
        "runtime_consumable": False,
        "training_consumable": False,
        "execute": execute,
        "targets": [item["path"] for item in before],
        "before": before,
        "after": after,
        "deleted_files": deleted_files,
        "deleted_bytes": deleted_bytes,
        "protected_before": protected_before,
        "protected_after": protected_after,
        "retained": [
            "manifests, source metadata, licenses and UVY central directory",
            "formal audit/results/resource traces and code/tests",
            "TeamTrack/VRU/NBA baseline media and all Wikimedia source originals",
            "EBQwen native and MLX-4bit model trees",
        ],
        "recovery": (
            "Each source is re-downloadable or re-extractable from its retained URL, "
            "revision and central-directory/manifest evidence; deleted pixels are not "
            "used by runtime or promoted training."
        ),
    }
    canonical = dict(audit)
    audit["audit_sha256"] = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = build_audit(execute=args.execute)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "execute": args.execute, "deleted_bytes": audit["deleted_bytes"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
