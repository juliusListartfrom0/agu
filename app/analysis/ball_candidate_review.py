"""Offline-only review contracts for broadcast basketball candidates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def select_stratified_candidates(
    detections: Sequence[Mapping[str, Any]],
    *,
    bands: Sequence[tuple[float, float]],
    samples_per_band: int,
) -> list[dict[str, Any]]:
    """Select frame-ordered, evenly spaced rows from every confidence band."""
    if samples_per_band <= 0:
        raise ValueError("samples per band must be positive")
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for low, high in bands:
        if not 0 <= low < high:
            raise ValueError("invalid confidence band")
        candidates = sorted(
            (
                dict(row)
                for row in detections
                if low <= float(row["confidence"]) < high
            ),
            key=lambda row: (int(row["frame"]), str(row["detection_id"])),
        )
        if len(candidates) < samples_per_band:
            raise ValueError(f"underfilled confidence band: {low}..{high}")
        if samples_per_band == 1:
            indices = [len(candidates) // 2]
        else:
            indices = [
                round(index * (len(candidates) - 1) / (samples_per_band - 1))
                for index in range(samples_per_band)
            ]
        for candidate_index in indices:
            row = candidates[candidate_index]
            detection_id = str(row["detection_id"])
            if detection_id in seen_ids:
                raise ValueError("duplicate stratified candidate")
            seen_ids.add(detection_id)
            row["confidence_band"] = [float(low), float(high)]
            selected.append(row)
    return selected


def seal_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    sealed = dict(artifact)
    sealed.pop("artifact_sha256", None)
    sealed["artifact_sha256"] = canonical_sha256(sealed)
    return sealed


def verify_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    verified = dict(artifact)
    claimed = str(verified.pop("artifact_sha256", ""))
    if not claimed or claimed != canonical_sha256(verified):
        raise ValueError("ball candidate review artifact hash mismatch")
    verified["artifact_sha256"] = claimed
    return verified
