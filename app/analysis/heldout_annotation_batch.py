"""Deterministic held-out sampling from a label-hidden continuous-game queue."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from app.analysis.continuous_game_annotation_queue import (
    QUEUE_SCHEMA,
    canonical_sha256,
    verify_continuous_game_annotation_queue,
)

BATCH_SCHEMA = "agu.continuous-game-annotation-batch.v1"
TEMPORAL_BUCKET_FIELD = "temporal_3way_v1"
TEMPORAL_BUCKETS = ("early", "middle", "late")


def build_heldout_annotation_batch(
    queue: Mapping[str, Any],
    *,
    per_source: int,
    seed: int,
    source_ids: Sequence[str] | None = None,
    excluded_anchors: Sequence[Mapping[str, Any]] = (),
    exclusion_radius_seconds: float = 20.0,
    temporal_bucket_quotas: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    """Select a deterministic source-balanced batch outside prior anchors.

    Anchors are source/center pairs, not labels.  A queue window is excluded
    when its center is within ``exclusion_radius_seconds`` of any prior anchor
    from the same source, preventing the next review from silently reusing a
    previously inspected causal neighborhood.
    """

    verified = verify_continuous_game_annotation_queue(queue)
    if isinstance(per_source, bool) or not isinstance(per_source, int) or per_source <= 0:
        raise ValueError("per_source must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    radius = float(exclusion_radius_seconds)
    if radius < 0.0 or radius != radius or radius == float("inf"):
        raise ValueError("exclusion_radius_seconds must be finite and non-negative")

    anchors: list[dict[str, Any]] = []
    for anchor in excluded_anchors:
        if not isinstance(anchor, Mapping):
            raise ValueError("excluded anchors must be mappings")
        source_id = str(anchor.get("source_id") or "")
        center = float(anchor.get("center_seconds"))
        if not source_id or center != center or center in (float("inf"), float("-inf")):
            raise ValueError("excluded anchors require finite source_id and center_seconds")
        anchors.append({"source_id": source_id, "center_seconds": round(center, 6)})

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    available_source_ids = sorted(str(row["source_id"]) for row in verified["sources"])
    selected_source_ids = _normalize_source_ids(source_ids, available_source_ids)
    normalized_quotas = _normalize_temporal_bucket_quotas(
        temporal_bucket_quotas,
        source_ids=selected_source_ids,
        per_source=per_source,
    )
    source_by_id = {str(row["source_id"]): row for row in verified["sources"]}
    for source_id in selected_source_ids:
        rows = [row for row in verified["windows"] if str(row["source_id"]) == source_id]
        eligible = [
            row
            for row in rows
            if all(
                abs(_center_seconds(row) - float(anchor["center_seconds"])) > radius
                for anchor in anchors
                if str(anchor["source_id"]) == source_id
            )
        ]
        if len(eligible) < per_source:
            raise ValueError(f"source {source_id} has too few held-out queue windows")
        if normalized_quotas is None:
            selected.extend(rng.sample(eligible, per_source))
            continue
        duration = float(source_by_id[source_id]["duration_seconds"])
        by_bucket = {
            bucket: [
                row
                for row in eligible
                if temporal_bucket(row, duration) == bucket
            ]
            for bucket in TEMPORAL_BUCKETS
        }
        for bucket in TEMPORAL_BUCKETS:
            quota = normalized_quotas[source_id][bucket]
            if len(by_bucket[bucket]) < quota:
                raise ValueError(
                    f"source {source_id} has too few held-out queue windows in {bucket} bucket"
                )
            selected.extend(rng.sample(by_bucket[bucket], quota))

    artifact: dict[str, Any] = {
        "schema_version": BATCH_SCHEMA,
        "purpose": "offline_continuous_game_heldout_annotation_batch",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "labels_hidden_from_reviewer": True,
        "queue_schema_version": QUEUE_SCHEMA,
        "queue_sha256": str(verified["artifact_sha256"]),
        "selection": {
            "seed": seed,
            "per_source": per_source,
            "source_ids": selected_source_ids,
            "exclusion_radius_seconds": radius,
            "excluded_anchors": anchors,
        },
        "windows": sorted(
            [dict(row) for row in selected],
            key=lambda row: (str(row["source_id"]), str(row["window_id"])),
        ),
    }
    if normalized_quotas is not None:
        artifact["selection"]["temporal_bucket_field"] = TEMPORAL_BUCKET_FIELD
        artifact["selection"]["temporal_bucket_quotas"] = normalized_quotas
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return verify_heldout_annotation_batch(artifact, queue=verified)


def verify_heldout_annotation_batch(
    payload: Mapping[str, Any], *, queue: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify batch provenance and enforce the no-label contract."""

    verified_queue = verify_continuous_game_annotation_queue(queue)
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    if artifact.get("schema_version") != BATCH_SCHEMA:
        raise ValueError("unsupported held-out annotation batch schema")
    if artifact.get("purpose") != "offline_continuous_game_heldout_annotation_batch":
        raise ValueError("held-out annotation batch must remain offline-only")
    if artifact.get("runtime_consumable") is not False:
        raise ValueError("held-out annotation batch cannot be runtime consumable")
    if artifact.get("codex_runtime_answer_used") is not False:
        raise ValueError("held-out annotation batch cannot be a runtime answer channel")
    if artifact.get("labels_hidden_from_reviewer") is not True:
        raise ValueError("held-out annotation batch must hide labels")
    if artifact.get("queue_schema_version") != QUEUE_SCHEMA:
        raise ValueError("held-out annotation batch queue schema mismatch")
    if artifact.get("queue_sha256") != verified_queue["artifact_sha256"]:
        raise ValueError("held-out annotation batch queue binding mismatch")
    selection = artifact.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("held-out annotation batch selection is required")
    per_source = selection.get("per_source")
    if isinstance(per_source, bool) or not isinstance(per_source, int) or per_source <= 0:
        raise ValueError("held-out annotation batch per_source is invalid")
    available_source_ids = sorted(str(row["source_id"]) for row in verified_queue["sources"])
    selected_source_ids = _normalize_source_ids(
        selection.get("source_ids"), available_source_ids
    )
    temporal_quotas = selection.get("temporal_bucket_quotas")
    if temporal_quotas is not None:
        if selection.get("temporal_bucket_field") != TEMPORAL_BUCKET_FIELD:
            raise ValueError("held-out annotation batch temporal bucket field is invalid")
        temporal_quotas = _normalize_temporal_bucket_quotas(
            temporal_quotas,
            source_ids=selected_source_ids,
            per_source=per_source,
        )
    rows = artifact.get("windows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("held-out annotation batch requires windows")
    queue_by_id = {str(row["window_id"]): row for row in verified_queue["windows"]}
    seen: set[str] = set()
    counts: dict[str, int] = {}
    bucket_counts: dict[str, dict[str, int]] = {}
    source_by_id = {str(row["source_id"]): row for row in verified_queue["sources"]}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("held-out annotation batch rows must be mappings")
        forbidden = {
            "event_present",
            "ground_truth",
            "label",
            "outcome",
            "review_note",
            "shot_sequence",
            "target",
        }
        if forbidden.intersection(row):
            raise ValueError("label-bearing held-out batch row is forbidden")
        window_id = str(row.get("window_id") or "")
        if not window_id or window_id in seen or window_id not in queue_by_id:
            raise ValueError("held-out annotation batch window provenance is invalid")
        if dict(row) != dict(queue_by_id[window_id]):
            raise ValueError("held-out annotation batch row differs from the queue")
        seen.add(window_id)
        source_id = str(row["source_id"])
        counts[source_id] = counts.get(source_id, 0) + 1
        if temporal_quotas is not None:
            bucket = temporal_bucket(row, float(source_by_id[source_id]["duration_seconds"]))
            per_source_counts = bucket_counts.setdefault(
                source_id, {name: 0 for name in TEMPORAL_BUCKETS}
            )
            per_source_counts[bucket] += 1
    if any(count != per_source for count in counts.values()):
        raise ValueError("held-out annotation batch must be source-balanced")
    if set(counts) != set(selected_source_ids):
        raise ValueError("held-out annotation batch must cover selected source_ids exactly")
    if temporal_quotas is not None:
        if bucket_counts != temporal_quotas:
            raise ValueError("held-out annotation batch temporal bucket quotas mismatch")
    if not claimed or claimed != canonical_sha256(artifact):
        raise ValueError("held-out annotation batch hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _center_seconds(row: Mapping[str, Any]) -> float:
    return (float(row["start_seconds"]) + float(row["end_seconds"])) / 2.0


def temporal_bucket(row: Mapping[str, Any], duration_seconds: float) -> str:
    """Return a deterministic label-free early/middle/late time bucket."""

    duration = float(duration_seconds)
    if duration <= 0.0 or duration != duration or duration == float("inf"):
        raise ValueError("duration_seconds must be finite and positive")
    fraction = _center_seconds(row) / duration
    if fraction < 1.0 / 3.0:
        return "early"
    if fraction < 2.0 / 3.0:
        return "middle"
    return "late"


def _normalize_temporal_bucket_quotas(
    quotas: Mapping[str, Mapping[str, int]] | None,
    *,
    source_ids: Sequence[str],
    per_source: int,
) -> dict[str, dict[str, int]] | None:
    if quotas is None:
        return None
    if not isinstance(quotas, Mapping):
        raise ValueError("temporal bucket quotas must be a mapping")
    expected_sources = set(source_ids)
    if set(str(source_id) for source_id in quotas) != expected_sources:
        raise ValueError("temporal bucket quotas must cover every source exactly")
    normalized: dict[str, dict[str, int]] = {}
    for source_id in source_ids:
        value = quotas.get(source_id)
        if not isinstance(value, Mapping) or set(str(bucket) for bucket in value) != set(TEMPORAL_BUCKETS):
            raise ValueError("temporal bucket quotas must contain early, middle, and late")
        row: dict[str, int] = {}
        for bucket in TEMPORAL_BUCKETS:
            quota = value.get(bucket)
            if isinstance(quota, bool) or not isinstance(quota, int) or quota < 0:
                raise ValueError("temporal bucket quotas must be non-negative integers")
            row[bucket] = quota
        if sum(row.values()) != per_source:
            raise ValueError("temporal bucket quotas must sum to per_source")
        normalized[source_id] = row
    return normalized


def _normalize_source_ids(
    source_ids: Sequence[str] | None,
    available_source_ids: Sequence[str],
) -> list[str]:
    """Normalize an optional source subset while preserving old all-source behavior."""

    available = sorted(str(source_id) for source_id in available_source_ids)
    if source_ids is None:
        return available
    if isinstance(source_ids, (str, bytes)):
        raise ValueError("source_ids must be a non-empty sequence of source IDs")
    try:
        values = [str(source_id) for source_id in source_ids]
    except TypeError as exc:
        raise ValueError("source_ids must be a non-empty sequence of source IDs") from exc
    if not values or any(not value for value in values):
        raise ValueError("source_ids must be a non-empty sequence of source IDs")
    if len(set(values)) != len(values):
        raise ValueError("source_ids must not contain duplicates")
    unknown = sorted(set(values) - set(available))
    if unknown:
        raise ValueError(f"source_ids contains unknown source IDs: {unknown}")
    return sorted(values)
