#!/usr/bin/env python3
"""Seal a PBP/OCR-clock coverage audit without promoting causal labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.pbp_causal_coverage import (  # noqa: E402
    build_pbp_causal_coverage_audit,
)


def _load(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"alignment must be an object: {path}")
    payload["source_file_sha256"] = hashlib.sha256(raw).hexdigest()
    return raw, payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment", action="append", type=Path, required=True)
    parser.add_argument("--generated-on", default="2026-08-08")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs/public_research/pbp_causal_coverage_audit_2026-08-08.json"),
    )
    args = parser.parse_args()
    alignments = []
    source_files: list[dict[str, str]] = []
    for path in args.alignment:
        _, payload = _load(path)
        alignments.append(payload)
        source_files.append(
            {
                "path": str(path),
                "file_sha256": payload.pop("source_file_sha256"),
                "alignment_artifact_sha256": str(payload.get("artifact_sha256") or ""),
            }
        )
    audit = build_pbp_causal_coverage_audit(alignments, generated_on=args.generated_on)
    audit["source_files"] = source_files
    # Recompute the hash after adding immutable source-file provenance.
    import hashlib as _hashlib

    audit.pop("audit_sha256", None)
    audit["audit_sha256"] = _hashlib.sha256(
        json.dumps(audit, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "games": len(audit["games"]),
                "pooled": audit["pooled"],
                "audit_sha256": audit["audit_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
