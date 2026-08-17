#!/usr/bin/env python3
"""Export the sealed VRU causal closure as training-only shot labels."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.vru_causal_training import (  # noqa: E402
    VruCausalShotValidityTrainingArtifacts,
    build_vru_causal_shot_validity_training_export,
    encode_json_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--expected-selection-artifact-sha256",
        required=True,
        help="externally frozen internal SHA-256 for the causal selection",
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-review-plan-artifact-sha256", required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--expected-sealed-review-artifact-sha256", required=True)
    parser.add_argument("--raw-frame-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-raw-frame-manifest-artifact-sha256",
        required=True,
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="export index path; per-source artifacts are written beside it",
    )
    return parser.parse_args()


def _resolved_disjoint_paths(
    *,
    selection: Path,
    plan: Path,
    review: Path,
    raw_frame_manifest: Path,
    source_manifest: Path,
    output: Path,
) -> tuple[dict[str, Path], Path, dict[str, Path]]:
    inputs = {
        "selection": selection.resolve(),
        "plan": plan.resolve(),
        "review": review.resolve(),
        "raw_frame_manifest": raw_frame_manifest.resolve(),
        "source_manifest": source_manifest.resolve(),
    }
    output_path = output.resolve()
    child_names = tuple(
        f"{source_id}_{suffix}_v1.json"
        for source_id in ("hazen", "randolph", "vtv")
        for suffix in ("candidate_bundle", "shot_validity_labels")
    )
    child_paths = {
        name: (output_path.parent / name).resolve() for name in child_names
    }
    _require_output_paths_disjoint(
        (output_path, *child_paths.values()),
        inputs.values(),
    )
    return inputs, output_path, child_paths


def _write_artifacts(
    *,
    output_path: Path,
    child_paths: Mapping[str, Path],
    artifacts: VruCausalShotValidityTrainingArtifacts,
) -> None:
    _require_output_paths_disjoint(
        (output_path, *child_paths.values()),
        artifacts.validated_input_paths,
    )
    expected_names = set(artifacts.candidate_bundles) | set(artifacts.label_files)
    if set(child_paths) != expected_names:
        raise ValueError("resolved child output set does not match the export")
    for filename in sorted(artifacts.candidate_bundles):
        _write_json_atomic(child_paths[filename], artifacts.candidate_bundles[filename])
    for filename in sorted(artifacts.label_files):
        _write_json_atomic(child_paths[filename], artifacts.label_files[filename])
    _write_json_atomic(output_path, artifacts.index)


def _require_output_paths_disjoint(
    output_paths: Iterable[Path],
    input_paths: Iterable[Path],
) -> None:
    outputs = tuple(output_paths)
    inputs = tuple(input_paths)
    output_casefolds = [path.as_posix().casefold() for path in outputs]
    if len(output_casefolds) != len(set(output_casefolds)):
        raise ValueError("output artifact paths must be casefold-unique and disjoint")
    input_casefolds = {path.as_posix().casefold() for path in inputs}
    if any(path.casefold() in input_casefolds for path in output_casefolds):
        raise ValueError("output artifact paths must not alias input paths")

    input_identities = {
        identity
        for path in inputs
        if (identity := _existing_path_identity(path)) is not None
    }
    output_identities: set[tuple[int, int]] = set()
    for path in outputs:
        identity = _existing_path_identity(path)
        if identity is None:
            continue
        if identity in input_identities:
            raise ValueError("output artifact paths must not alias input paths")
        if identity in output_identities:
            raise ValueError("output artifact paths must be mutually disjoint")
        output_identities.add(identity)


def _existing_path_identity(path: Path) -> tuple[int, int] | None:
    try:
        status = path.stat()
    except FileNotFoundError:
        return None
    return status.st_dev, status.st_ino


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = encode_json_artifact(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    inputs, output_path, child_paths = _resolved_disjoint_paths(
        selection=args.selection,
        plan=args.plan,
        review=args.review,
        raw_frame_manifest=args.raw_frame_manifest,
        source_manifest=args.source_manifest,
        output=args.output,
    )
    artifacts = build_vru_causal_shot_validity_training_export(
        selection_path=inputs["selection"],
        expected_selection_artifact_sha256=(
            args.expected_selection_artifact_sha256
        ),
        review_plan_path=inputs["plan"],
        expected_review_plan_artifact_sha256=(
            args.expected_review_plan_artifact_sha256
        ),
        sealed_review_path=inputs["review"],
        expected_sealed_review_artifact_sha256=(
            args.expected_sealed_review_artifact_sha256
        ),
        raw_frame_manifest_path=inputs["raw_frame_manifest"],
        expected_raw_frame_manifest_artifact_sha256=(
            args.expected_raw_frame_manifest_artifact_sha256
        ),
        source_manifest_path=inputs["source_manifest"],
    )
    _write_artifacts(
        output_path=output_path,
        child_paths=child_paths,
        artifacts=artifacts,
    )
    print(
        json.dumps(
            {
                "output": output_path.as_posix(),
                "artifact_sha256": artifacts.index["artifact_sha256"],
                "examples": artifacts.index["counts"]["exported"],
                "positive": artifacts.index["counts"]["positive"],
                "negative": artifacts.index["counts"]["negative"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
