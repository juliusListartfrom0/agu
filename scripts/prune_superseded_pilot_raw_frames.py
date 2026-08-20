#!/usr/bin/env python3
"""Prune superseded derived pilot frames while keeping review provenance.

Only the ``raw_frames`` children of the explicitly enumerated v2--v22 pilot
artifacts are eligible.  The source videos, review decisions, manifests,
summaries, and v23+ pilots are never targets.  The command requires
``--execute`` so an accidental invocation can only produce a read-only plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "analysis_outputs/public_research"
PREFIX = "wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v"
TARGET_TAGS = tuple(str(value) for value in range(2, 23))
PROTECTED_TAGS = ("23", "24_rv", "25_rv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="delete the exact allowlisted derived raw-frame files",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_audit(*, execute: bool) -> dict[str, Any]:
    targets = [PUBLIC / f"{PREFIX}{tag}" / "raw_frames" for tag in TARGET_TAGS]
    protected = [PUBLIC / f"{PREFIX}{tag}" / "raw_frames" for tag in PROTECTED_TAGS]
    if any(path.is_symlink() for path in [*targets, *protected]):
        raise ValueError("pilot raw-frame roots must not be symlinks")
    if any(path.exists() and not path.is_dir() for path in targets):
        raise ValueError("an allowlisted raw-frame root is not a directory")
    if any(path.exists() and not path.is_dir() for path in protected):
        raise ValueError("a protected raw-frame root is not a directory")

    rows: list[dict[str, Any]] = []
    total_files = 0
    total_bytes = 0
    for directory in targets:
        files = sorted(
            path for path in directory.rglob("*")
            if path.is_file() and not path.is_symlink()
        ) if directory.is_dir() else []
        inventory = hashlib.sha256()
        directory_bytes = 0
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            digest = _file_sha256(path)
            size = path.stat().st_size
            inventory.update(f"{relative}\t{size}\t{digest}\n".encode())
            directory_bytes += size
        rows.append(
            {
                "path": directory.relative_to(ROOT).as_posix(),
                "file_count": len(files),
                "bytes": directory_bytes,
                "inventory_sha256": inventory.hexdigest(),
            }
        )
        total_files += len(files)
        total_bytes += directory_bytes

    protected_counts = {
        directory.relative_to(ROOT).as_posix(): sum(
            1 for path in directory.rglob("*") if path.is_file()
        ) if directory.is_dir() else 0
        for directory in protected
    }
    audit: dict[str, Any] = {
        "schema_version": "agu.superseded-pilot-raw-frame-cleanup.v1",
        "created_on": datetime.now(timezone.utc).isoformat(),
        "purpose": "delete_derived_frames_superseded_by_sealed_review_metadata",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "training_consumable": False,
        "execute": execute,
        "allowlist": [f"{PREFIX}{tag}/raw_frames" for tag in TARGET_TAGS],
        "protected": [f"{PREFIX}{tag}/raw_frames" for tag in PROTECTED_TAGS],
        "directories": rows,
        "file_count": total_files,
        "bytes": total_bytes,
        "protected_file_counts_before": protected_counts,
        "recovery": (
            "Derived JPEGs can be re-materialized from retained hash-bound source videos and review plans; "
            "review decisions, manifests, summaries, source videos, and v23-v25 raw frames remain retained."
        ),
    }
    if execute:
        deleted_files = 0
        deleted_bytes = 0
        for directory in targets:
            files = sorted(
                path for path in directory.rglob("*")
                if path.is_file() and not path.is_symlink()
            ) if directory.is_dir() else []
            for path in files:
                size = path.stat().st_size
                path.unlink()
                deleted_files += 1
                deleted_bytes += size
        remaining = [
            path.relative_to(ROOT).as_posix()
            for directory in targets
            if directory.is_dir()
            for path in directory.rglob("*")
            if path.is_file()
        ]
        if remaining:
            raise RuntimeError(f"allowlisted raw-frame files remain: {remaining[:3]}")
        protected_counts_after = {
            directory.relative_to(ROOT).as_posix(): sum(
                1 for path in directory.rglob("*") if path.is_file()
            ) if directory.is_dir() else 0
            for directory in protected
        }
        if protected_counts_after != protected_counts:
            raise RuntimeError("protected pilot files changed during cleanup")
        audit["deleted_files"] = deleted_files
        audit["deleted_bytes"] = deleted_bytes
        audit["protected_file_counts_after"] = protected_counts_after
        if deleted_files != total_files or deleted_bytes != total_bytes:
            raise RuntimeError("deletion totals do not match the preflight inventory")
    audit["audit_sha256"] = _canonical_sha256(audit)
    return audit


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    value = dict(payload)
    value.pop("audit_sha256", None)
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    args = parse_args()
    audit = build_audit(execute=args.execute)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "files": audit["file_count"], "bytes": audit["bytes"], "execute": args.execute}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
