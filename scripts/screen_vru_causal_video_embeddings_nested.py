#!/usr/bin/env python3
"""Run the non-promotable nested causal-closure video representation probe."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.vru_causal_video_probe import (  # noqa: E402
    screen_vru_causal_video_embeddings_nested,
)


def _require_output_disjoint_from_inputs(
    output_path: Path,
    *,
    input_paths: Sequence[Path],
) -> tuple[Path, list[Path]]:
    resolved_output = output_path.resolve()
    resolved_inputs = [path.resolve() for path in input_paths]
    input_casefolds = {path.as_posix().casefold() for path in resolved_inputs}
    input_identities = {
        identity
        for path in resolved_inputs
        if (identity := _existing_path_identity(path)) is not None
    }
    if (
        resolved_output in set(resolved_inputs)
        or resolved_output.as_posix().casefold() in input_casefolds
        or _existing_path_identity(resolved_output) in input_identities
    ):
        raise ValueError("output path must not alias an input path")
    return resolved_output, resolved_inputs


def _existing_path_identity(path: Path) -> tuple[int, int] | None:
    try:
        status = path.stat()
    except FileNotFoundError:
        return None
    return status.st_dev, status.st_ino


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", action="append", type=Path, required=True)
    parser.add_argument("--training-export", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-training-export-sha256",
        required=True,
        help="frozen internal artifact SHA-256 of the verified training export",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output_path, resolved_inputs = _require_output_disjoint_from_inputs(
        args.output,
        input_paths=[
            *args.embeddings,
            args.training_export,
            args.training_manifest,
        ],
    )
    embedding_count = len(args.embeddings)
    embedding_paths = resolved_inputs[:embedding_count]
    training_export_path = resolved_inputs[embedding_count]
    training_manifest_path = resolved_inputs[embedding_count + 1]
    result = screen_vru_causal_video_embeddings_nested(
        embedding_paths=embedding_paths,
        training_export_path=training_export_path,
        training_manifest_path=training_manifest_path,
        expected_training_export_sha256=args.expected_training_export_sha256,
    )
    _write_json_atomic(output_path, result)
    print(
        json.dumps(
            {
                "formal_evaluation_eligible": result["formal_evaluation_eligible"],
                "runtime_consumable": result["runtime_consumable"],
                "promotion_eligible": result["promotion_eligible"],
                "promoted": result["promoted"],
                "row_count": result["row_count"],
                "source_count": result["source_count"],
                "artifact_sha256": result["artifact_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
