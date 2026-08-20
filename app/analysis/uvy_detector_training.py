"""Contracts for a bounded, auxiliary-only UVY detector training screen.

The UVY labels are useful for detector adaptation and hard-negative studies,
but they do not contain shot outcomes or causal ball--hand--rim truth.  This
module makes that boundary impossible to omit from a training plan/result.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

UVY_TRAINING_SCHEMA = "agu.uvy-detector-training-screen.v1"
UVY_YOLO_SCHEMA = "agu.uvy-yolo-auxiliary.v1"
UVY_TRAINING_SCOPE = "auxiliary_detector_and_hard_negative_only"


def build_uvy_training_plan(
    manifest: Mapping[str, Any],
    *,
    model_sha256: str,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
    output_dir: str | Path,
    max_memory_percent: float,
    min_available_memory_gib: float,
) -> dict[str, Any]:
    """Build a validated plan for one bounded UVY training experiment."""

    if manifest.get("schema_version") != UVY_YOLO_SCHEMA:
        raise ValueError("UVY training requires a UVY YOLO manifest")
    if manifest.get("runtime_consumable") is not False:
        raise ValueError("UVY training manifest must remain non-runtime")
    if manifest.get("training_media_eligible") is not True:
        raise ValueError("UVY training requires explicitly eligible media")
    if manifest.get("training_scope") != UVY_TRAINING_SCOPE:
        raise ValueError("UVY training scope is too broad")
    if manifest.get("causal_truth_eligible") is not False:
        raise ValueError("UVY training cannot use causal truth")
    splits = manifest.get("sequence_splits")
    if not isinstance(splits, Mapping) or set(splits.values()) != {"train", "val", "test"}:
        raise ValueError("UVY training requires train, val and test sequence splits")
    _require_sha(model_sha256, "model_sha256")
    _require_sha(str(manifest.get("manifest_sha256") or ""), "manifest_sha256")
    _require_sha(str(manifest.get("source_manifest_sha256") or ""), "source_manifest_sha256")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or not 1 <= epochs <= 20:
        raise ValueError("epochs must be an integer in [1, 20]")
    if not isinstance(imgsz, int) or isinstance(imgsz, bool) or not 128 <= imgsz <= 640 or imgsz % 32:
        raise ValueError("imgsz must be a multiple of 32 in [128, 640]")
    if not isinstance(batch, int) or isinstance(batch, bool) or not 1 <= batch <= 16:
        raise ValueError("batch must be an integer in [1, 16]")
    if not isinstance(workers, int) or isinstance(workers, bool) or not 0 <= workers <= 4:
        raise ValueError("workers must be an integer in [0, 4]")
    normalized_device = str(device).strip().lower()
    if normalized_device not in {"cpu", "mps"}:
        raise ValueError("device must be cpu or mps")
    if not 0 < float(max_memory_percent) < 100:
        raise ValueError("max_memory_percent must be in (0, 100)")
    if float(min_available_memory_gib) <= 0:
        raise ValueError("min_available_memory_gib must be positive")

    plan: dict[str, Any] = {
        "schema_version": UVY_TRAINING_SCHEMA,
        "purpose": "offline_uvy_auxiliary_detector_training_screen",
        "source_yolo_manifest_sha256": str(manifest["manifest_sha256"]),
        "source_manifest_sha256": str(manifest["source_manifest_sha256"]),
        "model_sha256": model_sha256,
        "sequence_splits": dict(sorted((str(key), str(value)) for key, value in splits.items())),
        "training_scope": UVY_TRAINING_SCOPE,
        "parameters": {
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "workers": workers,
            "device": normalized_device,
            "seed": 42,
            "deterministic": True,
            "cache": False,
            "amp": False,
        },
        "resource_thresholds": {
            "max_system_memory_percent": float(max_memory_percent),
            "min_available_memory_gib": float(min_available_memory_gib),
            "max_system_cpu_percent": 90.0,
            "consecutive_breaches": 2,
        },
        "output_dir": str(Path(output_dir)),
        "runtime_consumable": False,
        "causal_truth_eligible": False,
        "promotion_eligible": False,
        "codex_runtime_answer_used": False,
    }
    plan["plan_sha256"] = _canonical_sha256(plan)
    return plan


def seal_uvy_training_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Seal one training/screen result while preserving fail-closed flags."""

    artifact = dict(payload)
    artifact.setdefault("schema_version", UVY_TRAINING_SCHEMA)
    artifact.setdefault("purpose", "offline_uvy_auxiliary_detector_training_screen")
    artifact.setdefault("runtime_consumable", False)
    artifact.setdefault("causal_truth_eligible", False)
    artifact.setdefault("promotion_eligible", False)
    artifact.setdefault("training_scope", UVY_TRAINING_SCOPE)
    artifact.setdefault("codex_runtime_answer_used", False)
    artifact.setdefault("checkpoint_promoted", False)
    artifact.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_uvy_training_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a sealed result and reject any attempt to promote it."""

    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    if artifact.get("schema_version") != UVY_TRAINING_SCHEMA:
        raise ValueError("invalid UVY training schema")
    if artifact.get("runtime_consumable") is not False:
        raise ValueError("UVY training result cannot be runtime consumable")
    if artifact.get("causal_truth_eligible") is not False:
        raise ValueError("UVY training result cannot be causal truth")
    if artifact.get("promotion_eligible") is not False:
        raise ValueError("UVY training result cannot be promotion eligible")
    if artifact.get("checkpoint_promoted") is not False:
        raise ValueError("UVY training checkpoint cannot be promoted")
    if artifact.get("training_scope") != UVY_TRAINING_SCOPE:
        raise ValueError("UVY training result scope is too broad")
    if not claimed or claimed != _canonical_sha256(artifact):
        raise ValueError("UVY training result hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _require_sha(value: str, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
