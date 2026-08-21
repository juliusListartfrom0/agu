#!/usr/bin/env python3
"""Inspect a local Endpoint Security JSONL transcript without minting evidence.

The inspector is a read-only handoff tool for a future externally authorized
platform run. It validates the C client's bounded JSONL projection and clean
finalization record, but it never creates an attestation, approval, or product
result.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path
from typing import TextIO

from app.analysis.task0258_v2_audit import parse_endpoint_security_transcript

SCHEMA_VERSION = "agu.task0258-endpoint-security-diagnostic-inspection.v1"


def _open_regular_transcript(path: Path) -> TextIO:
    """Open an absolute, regular, non-symlink transcript for bounded reading."""
    raw_path = str(path)
    if not path.is_absolute():
        raise ValueError("transcript path must be absolute")
    if raw_path.startswith("//") or "//" in raw_path:
        raise ValueError("transcript path must be canonical absolute path")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise OSError("platform does not provide no-follow directory opening")
    common_flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        common_flags |= os.O_CLOEXEC
    directory_flags = common_flags | os.O_DIRECTORY
    parts = path.parts
    if len(parts) < 2 or parts[0] != "/" or any(part in {"", ".", ".."} for part in parts[1:]):
        raise ValueError("transcript path contains an invalid component")
    directory_fd = os.open("/", directory_flags)
    try:
        for component in parts[1:-1]:
            next_directory_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_directory_fd
        descriptor = os.open(parts[-1], common_flags, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("transcript path is not a regular file")
        return os.fdopen(descriptor, "r", encoding="utf-8", newline="")
    except Exception:
        os.close(descriptor)
        raise


def _error_report(message: str) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "invalid_diagnostic_transcript",
        "error": message,
        "evidence_class": "diagnostic_only",
        "production_capability": False,
        "p5_ready": False,
    }


def inspect_transcript(path: Path) -> dict[str, object]:
    """Return a non-authorizing report for a validated diagnostic transcript."""
    with _open_regular_transcript(path) as stream:
        events = parse_endpoint_security_transcript(stream)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "valid_diagnostic_transcript",
        "event_count": len(events),
        "finalization_verified": True,
        "evidence_class": "diagnostic_only",
        "production_capability": False,
        "p5_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="emit the complete report as JSON")
    args = parser.parse_args()
    try:
        report = inspect_transcript(args.transcript)
    except (OSError, UnicodeError, ValueError) as exc:
        report = _error_report(str(exc))
        exit_code = 1
    else:
        exit_code = 0
    if args.json:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        print(report["status"])
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
