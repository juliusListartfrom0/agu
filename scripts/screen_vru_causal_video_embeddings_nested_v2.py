#!/usr/bin/env python3
"""Run the receipt-bound four-game causal video representation probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.vru_causal_video_probe_v2 import (  # noqa: E402
    screen_vru_causal_video_embeddings_nested_v2,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_EXPORT_JSON_BYTES = 4 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", action="append", type=Path, required=True)
    parser.add_argument(
        "--expected-embedding-artifact-sha256",
        action="append",
        required=True,
    )
    parser.add_argument(
        "--expected-embedding-file-sha256",
        action="append",
        required=True,
    )
    parser.add_argument("--training-export", type=Path, required=True)
    parser.add_argument("--expected-training-export-artifact-sha256", required=True)
    parser.add_argument("--expected-training-export-file-sha256", required=True)
    parser.add_argument("--extension-asset-root", type=Path, required=True)
    parser.add_argument("--parent-export", type=Path, required=True)
    parser.add_argument("--expected-parent-artifact-sha256", required=True)
    parser.add_argument("--expected-parent-file-sha256", required=True)
    parser.add_argument("--parent-asset-root", type=Path, required=True)
    parser.add_argument("--harwood-selection", type=Path, required=True)
    parser.add_argument(
        "--expected-harwood-selection-artifact-sha256",
        required=True,
    )
    parser.add_argument("--harwood-review-plan", type=Path, required=True)
    parser.add_argument(
        "--expected-harwood-review-plan-artifact-sha256",
        required=True,
    )
    parser.add_argument("--harwood-sealed-review", type=Path, required=True)
    parser.add_argument(
        "--expected-harwood-sealed-review-artifact-sha256",
        required=True,
    )
    parser.add_argument("--harwood-raw-frame-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-harwood-raw-frame-manifest-artifact-sha256",
        required=True,
    )
    parser.add_argument("--harwood-source-manifest", type=Path, required=True)
    parser.add_argument("--source-groups", type=Path, required=True)
    parser.add_argument("--expected-source-groups-artifact-sha256", required=True)
    parser.add_argument("--expected-source-groups-file-sha256", required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--expected-training-manifest-sha256", required=True)
    parser.add_argument("--expected-training-manifest-file-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-artifact-sha256", required=True)
    parser.add_argument("--expected-plan-file-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _input_paths(args: argparse.Namespace) -> list[Path]:
    return [
        *args.embeddings,
        args.training_export,
        args.extension_asset_root,
        args.parent_export,
        args.parent_asset_root,
        args.harwood_selection,
        args.harwood_review_plan,
        args.harwood_sealed_review,
        args.harwood_raw_frame_manifest,
        args.harwood_source_manifest,
        args.source_groups,
        args.training_manifest,
        args.plan,
    ]


def _preflight_output(
    output_path: Path,
    *,
    input_paths: Sequence[Path],
) -> tuple[Path, list[Path]]:
    if output_path.is_symlink() or (output_path.exists() and not output_path.is_file()):
        raise ValueError("probe output must be a regular file path")
    unresolved_output = output_path.absolute()
    resolved_output = output_path.resolve()
    resolved_inputs = [path.resolve() for path in input_paths]
    output_casefold = unresolved_output.as_posix().casefold()
    input_casefolds = {path.absolute().as_posix().casefold() for path in input_paths} | {
        path.as_posix().casefold() for path in resolved_inputs
    }
    input_identities = {identity for path in resolved_inputs if (identity := _existing_identity(path)) is not None}
    if (
        output_casefold in input_casefolds
        or resolved_output.as_posix().casefold() in input_casefolds
        or _existing_identity(resolved_output) in input_identities
        or any(_paths_related_casefold(resolved_output, path) for path in resolved_inputs)
    ):
        raise ValueError("probe output path must not alias or nest an input path")
    return resolved_output, resolved_inputs


def _paths_related_casefold(left: Path, right: Path) -> bool:
    left_parts = tuple(part.casefold() for part in left.parts)
    right_parts = tuple(part.casefold() for part in right.parts)
    shorter = min(len(left_parts), len(right_parts))
    return left_parts[:shorter] == right_parts[:shorter]


def _existing_identity(path: Path) -> tuple[int, int] | None:
    try:
        status = path.stat()
    except FileNotFoundError:
        return None
    return status.st_dev, status.st_ino


def _read_json_snapshot(path: Path) -> tuple[dict[str, Any], str, int]:
    with path.open("rb") as handle:
        encoded = handle.read(_MAX_EXPORT_JSON_BYTES + 1)
    if len(encoded) > _MAX_EXPORT_JSON_BYTES:
        raise ValueError(f"JSON input is too large: {path.name}")
    try:
        payload = json.loads(
            encoded,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"expected valid JSON: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    _require_finite_json(payload)
    return payload, hashlib.sha256(encoded).hexdigest(), len(encoded)


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {token}")


def _require_finite_json(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number is forbidden")
    if isinstance(value, Mapping):
        for child in value.values():
            _require_finite_json(child)
    elif isinstance(value, list):
        for child in value:
            _require_finite_json(child)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON object key: {key}")
        payload[key] = value
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary_path.read_bytes() != encoded:
            raise ValueError("staged probe output bytes failed verification")
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _validate_cli_receipts(args: argparse.Namespace) -> None:
    repeated = {
        "expected embedding artifact": args.expected_embedding_artifact_sha256,
        "expected embedding file": args.expected_embedding_file_sha256,
    }
    for field, values in repeated.items():
        for value in values:
            _require_sha256(value, field=field)
    scalar_names = (
        "expected_training_export_artifact_sha256",
        "expected_training_export_file_sha256",
        "expected_parent_artifact_sha256",
        "expected_parent_file_sha256",
        "expected_harwood_selection_artifact_sha256",
        "expected_harwood_review_plan_artifact_sha256",
        "expected_harwood_sealed_review_artifact_sha256",
        "expected_harwood_raw_frame_manifest_artifact_sha256",
        "expected_source_groups_artifact_sha256",
        "expected_source_groups_file_sha256",
        "expected_training_manifest_sha256",
        "expected_training_manifest_file_sha256",
        "expected_plan_artifact_sha256",
        "expected_plan_file_sha256",
    )
    for name in scalar_names:
        _require_sha256(
            getattr(args, name),
            field=name.replace("_", " "),
        )


def main() -> int:
    args = parse_args()
    if not (
        len(args.embeddings)
        == len(args.expected_embedding_artifact_sha256)
        == len(args.expected_embedding_file_sha256)
        == 2
    ):
        raise ValueError("two embedding paths and two external receipt pairs are required")
    _validate_cli_receipts(args)
    output_path, resolved_inputs = _preflight_output(
        args.output,
        input_paths=_input_paths(args),
    )
    cursor = 0
    embedding_count = len(args.embeddings)
    embedding_paths = resolved_inputs[cursor : cursor + embedding_count]
    cursor += embedding_count
    (
        training_export_path,
        extension_asset_root,
        parent_export_path,
        parent_asset_root,
        harwood_selection_path,
        harwood_review_plan_path,
        harwood_sealed_review_path,
        harwood_raw_frame_manifest_path,
        harwood_source_manifest_path,
        source_groups_path,
        training_manifest_path,
        plan_path,
    ) = resolved_inputs[cursor:]
    result = screen_vru_causal_video_embeddings_nested_v2(
        embedding_paths=embedding_paths,
        expected_embedding_artifact_sha256s=(args.expected_embedding_artifact_sha256),
        expected_embedding_file_sha256s=args.expected_embedding_file_sha256,
        training_export_path=training_export_path,
        expected_training_export_artifact_sha256=(args.expected_training_export_artifact_sha256),
        expected_training_export_file_sha256=(args.expected_training_export_file_sha256),
        extension_asset_root=extension_asset_root,
        parent_export_path=parent_export_path,
        expected_parent_artifact_sha256=args.expected_parent_artifact_sha256,
        expected_parent_file_sha256=args.expected_parent_file_sha256,
        parent_asset_root=parent_asset_root,
        harwood_selection_path=harwood_selection_path,
        expected_harwood_selection_artifact_sha256=(args.expected_harwood_selection_artifact_sha256),
        harwood_review_plan_path=harwood_review_plan_path,
        expected_harwood_review_plan_artifact_sha256=(args.expected_harwood_review_plan_artifact_sha256),
        harwood_sealed_review_path=harwood_sealed_review_path,
        expected_harwood_sealed_review_artifact_sha256=(args.expected_harwood_sealed_review_artifact_sha256),
        harwood_raw_frame_manifest_path=harwood_raw_frame_manifest_path,
        expected_harwood_raw_frame_manifest_artifact_sha256=(args.expected_harwood_raw_frame_manifest_artifact_sha256),
        harwood_source_manifest_path=harwood_source_manifest_path,
        source_groups_path=source_groups_path,
        expected_source_groups_artifact_sha256=(args.expected_source_groups_artifact_sha256),
        expected_source_groups_file_sha256=args.expected_source_groups_file_sha256,
        training_manifest_path=training_manifest_path,
        expected_training_manifest_sha256=args.expected_training_manifest_sha256,
        expected_training_manifest_file_sha256=(args.expected_training_manifest_file_sha256),
        probe_plan_path=plan_path,
        expected_probe_plan_sha256=args.expected_plan_artifact_sha256,
        expected_probe_plan_file_sha256=args.expected_plan_file_sha256,
    )
    _write_json_atomic(output_path, result)
    print(
        json.dumps(
            {
                "artifact_sha256": result["artifact_sha256"],
                "formal_evaluation_eligible": result["formal_evaluation_eligible"],
                "game_group_count": result["game_group_count"],
                "production_family_count": result["production_family_count"],
                "promoted": result["promoted"],
                "promotion_eligible": result["promotion_eligible"],
                "row_count": result["row_count"],
                "runtime_consumable": result["runtime_consumable"],
            },
            allow_nan=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
