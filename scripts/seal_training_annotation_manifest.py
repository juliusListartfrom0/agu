#!/usr/bin/env python3
"""Seal Codex/human training labels while excluding acceptance videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.analysis.training_annotation import seal_training_annotation_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", required=True)
    parser.add_argument("--source-video", action="append", type=Path, required=True)
    parser.add_argument("--annotation", action="append", type=Path, required=True)
    parser.add_argument("--task-type", action="append", required=True)
    parser.add_argument("--benchmark-bundle", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    benchmark_bundles = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.benchmark_bundle
    ]
    manifest = seal_training_annotation_manifest(
        producer=args.producer,
        source_video_paths=args.source_video,
        annotation_paths=args.annotation,
        task_types=args.task_type,
        benchmark_bundles=benchmark_bundles,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "manifest_sha256": manifest["manifest_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
