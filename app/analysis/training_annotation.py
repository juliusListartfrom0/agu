"""Hash-bound governance for Codex-assisted model training annotations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from app.analysis.official_evaluation import RawOnlyEvaluationError, verify_raw_only_bundle

TRAINING_ANNOTATION_SCHEMA = "agu.training-annotation.v1"


def seal_training_annotation_manifest(
    *,
    producer: str,
    source_video_paths: Sequence[str | Path],
    annotation_paths: Sequence[str | Path],
    task_types: Iterable[str],
    benchmark_bundles: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Seal training inputs after proving they do not reuse acceptance videos."""

    if not source_video_paths or not annotation_paths:
        raise RawOnlyEvaluationError("training manifest requires source videos and annotations")
    benchmark_hashes = {
        asset.sha256
        for bundle in benchmark_bundles
        for asset in verify_raw_only_bundle(bundle).raw_videos
    }
    if not benchmark_hashes:
        raise RawOnlyEvaluationError("at least one sealed benchmark bundle is required")
    sources = [_asset(Path(path)) for path in source_video_paths]
    overlap = sorted({asset["sha256"] for asset in sources} & benchmark_hashes)
    if overlap:
        raise RawOnlyEvaluationError(
            f"training annotations overlap acceptance raw-video hashes: {overlap}"
        )
    payload: dict[str, Any] = {
        "schema_version": TRAINING_ANNOTATION_SCHEMA,
        "purpose": "model_training_only",
        "producer": producer,
        "runtime_consumable": False,
        "benchmark_overlap": False,
        "source_videos": sources,
        "annotation_files": [_asset(Path(path)) for path in annotation_paths],
        "task_types": sorted(set(task_types)),
        "benchmark_raw_sha256": sorted(benchmark_hashes),
    }
    payload["manifest_sha256"] = _json_sha256(payload)
    return payload


def verify_training_annotation_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Verify immutability and the non-runtime contract of a training manifest."""

    payload = dict(manifest)
    claimed_hash = str(payload.pop("manifest_sha256", ""))
    if _json_sha256(payload) != claimed_hash:
        raise RawOnlyEvaluationError("training annotation manifest hash mismatch")
    if payload.get("schema_version") != TRAINING_ANNOTATION_SCHEMA:
        raise RawOnlyEvaluationError("unsupported training annotation manifest schema")
    if payload.get("purpose") != "model_training_only" or payload.get("runtime_consumable") is not False:
        raise RawOnlyEvaluationError("training annotations must not be runtime-consumable")
    if payload.get("benchmark_overlap") is not False:
        raise RawOnlyEvaluationError("training annotation manifest must declare benchmark_overlap=false")
    source_hashes = {item["sha256"] for item in payload.get("source_videos", [])}
    benchmark_hashes = set(payload.get("benchmark_raw_sha256", []))
    if source_hashes & benchmark_hashes:
        raise RawOnlyEvaluationError("training annotation manifest contains benchmark video overlap")
    payload["manifest_sha256"] = claimed_hash
    return payload


def _asset(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RawOnlyEvaluationError(f"annotation manifest asset does not exist: {path}")
    return {"filename": path.name, "sha256": _file_sha256(path), "size_bytes": path.stat().st_size}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
