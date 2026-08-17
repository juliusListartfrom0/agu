#!/usr/bin/env python3
"""Materialize hash-bound UVY MOT boxes for auxiliary YOLO experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.analysis.uvy_yolo import (
    assign_uvy_sequence_splits,
    materialize_uvy_yolo_subset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-sequence", default="basketball_V02")
    parser.add_argument("--test-sequence", default="basketball_V04")
    parser.add_argument(
        "--ignore-sequence",
        action="append",
        default=["basketball_V03"],
        help="Sequence to exclude after the duplicate-GT audit (repeatable).",
    )
    args = parser.parse_args()
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    ignored = {str(value).strip() for value in args.ignore_sequence if str(value).strip()}
    sequences = [
        str(item["sequence"])
        for item in source_manifest.get("sequences", [])
        if str(item["sequence"]) not in ignored
    ]
    split_map = assign_uvy_sequence_splits(
        sequences,
        validation_sequence=args.validation_sequence,
        test_sequence=args.test_sequence,
    )
    manifest = materialize_uvy_yolo_subset(
        args.source_root,
        args.output_root,
        source_manifest_sha256=str(source_manifest.get("manifest_sha256") or ""),
        sequence_splits=split_map,
        ignored_sequences=sorted(ignored),
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "split_counts": manifest["split_counts"],
                "split_box_counts": manifest["split_box_counts"],
                "class_counts": manifest["class_counts"],
                "link_modes": manifest["link_modes"],
                "manifest_sha256": manifest["manifest_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
