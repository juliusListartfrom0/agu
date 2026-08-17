#!/usr/bin/env python3
"""Seal the local UVY basketball detector-auxiliary audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.analysis.uvy_dataset import build_uvy_basketball_audit

SOURCE_URL = "https://zenodo.org/api/records/21303900/files/UVY.zip/content"
SOURCE_RECORD_URL = "https://zenodo.org/records/21303900"
SOURCE_REVISION = "10.5281/zenodo.21303900"
SOURCE_LICENSE = "CC-BY-4.0"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    audit = build_uvy_basketball_audit(
        args.source_root,
        manifest,
        source_url=SOURCE_URL,
        source_record_url=SOURCE_RECORD_URL,
        source_revision=SOURCE_REVISION,
        source_license=SOURCE_LICENSE,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sequence_count": audit["sequence_count"],
                "image_frame_count": audit["image_frame_count"],
                "box_count": audit["box_count"],
                "class_counts": audit["class_counts"],
                "training_media_eligible": audit["training_media_eligible"],
                "causal_truth_eligible": audit["causal_truth_eligible"],
                "artifact_sha256": audit["artifact_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
