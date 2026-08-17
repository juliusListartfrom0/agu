#!/usr/bin/env python3
"""Seal Codex/human training labels while excluding acceptance videos."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.training_annotation import seal_training_annotation_manifest  # noqa: E402


def _resolved_disjoint_paths(
    *,
    source_videos: Sequence[Path],
    annotations: Sequence[Path],
    benchmark_bundles: Sequence[Path],
    output: Path,
) -> tuple[list[Path], list[Path], list[Path], Path]:
    resolved_sources = [path.resolve() for path in source_videos]
    resolved_annotations = [path.resolve() for path in annotations]
    resolved_benchmarks = [path.resolve() for path in benchmark_bundles]
    resolved_output = output.resolve()
    resolved_inputs = [
        *resolved_sources,
        *resolved_annotations,
        *resolved_benchmarks,
    ]
    input_paths = set(resolved_inputs)
    input_casefolds = {
        path.as_posix().casefold() for path in resolved_inputs
    }
    input_identities = {
        identity
        for path in resolved_inputs
        if (identity := _existing_path_identity(path)) is not None
    }
    if (
        resolved_output in input_paths
        or resolved_output.as_posix().casefold() in input_casefolds
        or _existing_path_identity(resolved_output) in input_identities
    ):
        raise ValueError("output path must not alias an input path")
    return (
        resolved_sources,
        resolved_annotations,
        resolved_benchmarks,
        resolved_output,
    )


def _existing_path_identity(path: Path) -> tuple[int, int] | None:
    try:
        status = path.stat()
    except FileNotFoundError:
        return None
    return status.st_dev, status.st_ino


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", required=True)
    parser.add_argument("--source-video", action="append", type=Path, required=True)
    parser.add_argument("--annotation", action="append", type=Path, required=True)
    parser.add_argument("--task-type", action="append", required=True)
    parser.add_argument("--benchmark-bundle", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    (
        source_video_paths,
        annotation_paths,
        benchmark_bundle_paths,
        output_path,
    ) = _resolved_disjoint_paths(
        source_videos=args.source_video,
        annotations=args.annotation,
        benchmark_bundles=args.benchmark_bundle,
        output=args.output,
    )
    benchmark_bundles = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in benchmark_bundle_paths
    ]
    manifest = seal_training_annotation_manifest(
        producer=args.producer,
        source_video_paths=source_video_paths,
        annotation_paths=annotation_paths,
        task_types=args.task_type,
        benchmark_bundles=benchmark_bundles,
    )
    _write_json_atomic(output_path, manifest)
    print(json.dumps({"output": str(args.output), "manifest_sha256": manifest["manifest_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
