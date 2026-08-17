#!/usr/bin/env python3
"""Freeze a deterministic 24-window causal-closure review selection.

Prior four-second dispositions are used only to stratify the candidate pool;
they are never exported to the reviewer or reused as the final eight-second
causal labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.vru_causal_review import (  # noqa: E402
    verify_vru_causal_review,
    verify_vru_causal_review_plan,
)

SCHEMA_VERSION = "agu.vru-causal-closure-selection.v1"
PURPOSE = "offline_causal_closure_selection_freeze"
RANKING_NAMESPACE = "agu.causal-closure-pilot.v1|2026-08-15|8s|8fps"
ANNOTATION_PREFIX = "wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_"
ANNOTATION_PATTERN = re.compile(
    rf"{re.escape(ANNOTATION_PREFIX)}v(?P<version>[0-9]+)(?P<suffix>_rv|_smoke)?"
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
NEUTRAL_ID_PATTERN = re.compile(r"closure-(hazen|randolph|vtv)-[0-9]{4}")
SOURCE_IDS = ("hazen", "randolph", "vtv")
TEMPORAL_BUCKETS = ("early", "middle", "late")
TEMPORAL_BUCKET_QUOTAS = {"early": 3, "middle": 2, "late": 3}
PRIOR_POOL_QUOTA_PAIRS = {
    "hazen": (4, 4),
    "randolph": (3, 5),
    "vtv": (4, 4),
}
WINDOW_SECONDS = 8.0
SAMPLE_RATE_HZ = 8.0
SAMPLE_COUNT = 64
MIN_VERSION = 3
FROZEN_MAX_VERSION = 41
FROZEN_INCLUDED_INPUT_IDS = tuple(
    [f"v{version}" for version in range(MIN_VERSION, 24)]
    + [f"v{version}_rv" for version in range(24, FROZEN_MAX_VERSION + 1)]
)
FORBIDDEN_FIELDS = {
    "confidence",
    "event_present",
    "evidence",
    "ground_truth",
    "label",
    "labels",
    "notes",
    "outcome",
    "prior_disposition",
    "release_frame",
    "release_position",
    "review_note",
    "rim_frame",
    "rim_position",
    "shot_sequence",
    "target",
    "training_eligible",
}
WINDOW_FIELDS = {
    "anchor_frame",
    "center_seconds",
    "clip_id",
    "end_seconds",
    "frame_count",
    "ranking_sha256",
    "review_id",
    "source_fps",
    "source_id",
    "source_video_filename",
    "source_video_sha256",
    "start_seconds",
    "temporal_bucket",
}
TOP_LEVEL_FIELDS = {
    "artifact_sha256",
    "audit",
    "codex_runtime_answer_used",
    "formal_evaluation_eligible",
    "inputs",
    "labels_hidden_from_reviewer",
    "prior_labels_used_as_final_8s_targets",
    "prior_labels_used_for_stratified_selection",
    "purpose",
    "runtime_consumable",
    "schema_version",
    "selection",
    "selection_provenance_verification",
    "source_manifest_sha256",
    "training_consumable",
    "windows",
}
AUDIT_FIELDS = {
    "candidate_physical_window_count",
    "collision_group_count",
    "conflicting_physical_window_count",
    "excluded_input_ids",
    "excluded_inputs",
    "included_input_count",
    "observation_count",
    "physical_window_count",
    "selectable_physical_window_count",
}


@dataclass(frozen=True)
class _Source:
    source_id: str
    clip_id: str
    video_sha256: str
    video_filename: str
    fps: float
    frame_count: int
    duration_seconds: float


@dataclass(frozen=True)
class _Observation:
    version: int
    input_id: str
    plan_sha256: str
    sealed_review_sha256: str
    prior_review_id: str
    source: _Source
    anchor_frame: int
    prior_first: str
    prior_second: str


@dataclass(frozen=True)
class _Candidate:
    observation: _Observation
    pool_index: int
    temporal_bucket: str
    ranking_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--max-version", type=int, default=FROZEN_MAX_VERSION)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_causal_closure_selection(
    *,
    annotation_root: Path,
    source_manifest_path: Path,
    max_version: int = FROZEN_MAX_VERSION,
) -> dict[str, Any]:
    """Build the fixed three-source closure pilot without exporting prior answers."""

    if (
        isinstance(max_version, bool)
        or not isinstance(max_version, int)
        or max_version != FROZEN_MAX_VERSION
    ):
        raise ValueError(f"selection freeze requires max_version={FROZEN_MAX_VERSION}")
    sources, source_manifest_sha256 = _load_sources(source_manifest_path)
    included, excluded = _discover_inputs(annotation_root, max_version=max_version)

    observations: list[_Observation] = []
    inputs: list[dict[str, str]] = []
    for version, input_id, directory in included:
        rows, provenance = _load_input(
            directory,
            version=version,
            input_id=input_id,
            sources=sources,
        )
        observations.extend(rows)
        inputs.append(provenance)

    excluded_inputs: list[dict[str, str]] = []
    for version, input_id, directory in excluded:
        _, provenance = _load_input(
            directory,
            version=version,
            input_id=input_id,
            sources=sources,
        )
        excluded_inputs.append(provenance)

    by_physical_key: dict[tuple[str, int], list[_Observation]] = defaultdict(list)
    for row in observations:
        by_physical_key[(row.source.video_sha256, row.anchor_frame)].append(row)
    for rows in by_physical_key.values():
        rows.sort(key=lambda row: (row.version, row.input_id, row.prior_review_id))

    latest = [rows[-1] for rows in by_physical_key.values()]
    candidate_physical_window_count = sum(
        _prior_pool_index(row) is not None for row in latest
    )
    candidates = _build_candidates(latest)
    windows = _select_windows(candidates)
    conflicting_physical_window_count = sum(
        len({(row.prior_first, row.prior_second) for row in rows}) > 1
        for rows in by_physical_key.values()
    )

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": PURPOSE,
        "runtime_consumable": False,
        "training_consumable": False,
        "codex_runtime_answer_used": False,
        "formal_evaluation_eligible": False,
        "labels_hidden_from_reviewer": True,
        "prior_labels_used_for_stratified_selection": True,
        "prior_labels_used_as_final_8s_targets": False,
        "selection_provenance_verification": (
            "structural_only_without_source_inputs"
        ),
        "source_manifest_sha256": source_manifest_sha256,
        "selection": {
            "ranking_namespace": RANKING_NAMESPACE,
            "ranking_direction": "ascending_sha256",
            "cell_selection_order": (
                "prior_pool_then_available_count_then_early_middle_late"
            ),
            "physical_window_supersession": (
                "highest_formal_version_then_input_id_then_review_id"
            ),
            "source_ids": list(SOURCE_IDS),
            "per_source": 8,
            "window_seconds": WINDOW_SECONDS,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "sample_count": SAMPLE_COUNT,
            "interval_semantics": "half_open",
            "prior_pool_quota_pairs": {
                source_id: list(PRIOR_POOL_QUOTA_PAIRS[source_id])
                for source_id in SOURCE_IDS
            },
            "temporal_bucket_quotas": {
                source_id: dict(TEMPORAL_BUCKET_QUOTAS) for source_id in SOURCE_IDS
            },
        },
        "inputs": inputs,
        "audit": {
            "included_input_count": len(inputs),
            "excluded_input_ids": [row["input_id"] for row in excluded_inputs],
            "excluded_inputs": excluded_inputs,
            "observation_count": len(observations),
            "physical_window_count": len(by_physical_key),
            "collision_group_count": sum(
                len(rows) > 1 for rows in by_physical_key.values()
            ),
            "conflicting_physical_window_count": (
                conflicting_physical_window_count
            ),
            "candidate_physical_window_count": candidate_physical_window_count,
            "selectable_physical_window_count": len(candidates),
        },
        "windows": windows,
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return verify_causal_closure_selection(artifact)


def verify_causal_closure_selection(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the sealed, reviewer-label-hidden balanced window contract."""

    _reject_label_fields(payload)
    if set(payload) != TOP_LEVEL_FIELDS:
        raise ValueError("causal closure selection top-level fields are not canonical")
    artifact = dict(payload)
    claimed = _require_sha(
        artifact.pop("artifact_sha256", None),
        "causal closure selection artifact",
    )
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported causal closure selection schema")
    if artifact.get("purpose") != PURPOSE:
        raise ValueError("causal closure selection must remain offline-only")
    if artifact.get("runtime_consumable") is not False:
        raise ValueError("causal closure selection cannot be runtime consumable")
    if artifact.get("training_consumable") is not False:
        raise ValueError("causal closure selection cannot be training consumable")
    if artifact.get("codex_runtime_answer_used") is not False:
        raise ValueError("causal closure selection cannot be a runtime answer channel")
    if artifact.get("formal_evaluation_eligible") is not False:
        raise ValueError(
            "causal closure selection alone is not formal evaluation evidence"
        )
    if artifact.get("labels_hidden_from_reviewer") is not True:
        raise ValueError("causal closure selection must hide prior answers")
    if artifact.get("prior_labels_used_for_stratified_selection") is not True:
        raise ValueError("causal closure selection must disclose prior-label stratification")
    if artifact.get("prior_labels_used_as_final_8s_targets") is not False:
        raise ValueError("prior dispositions cannot become final eight-second targets")
    if (
        artifact.get("selection_provenance_verification")
        != "structural_only_without_source_inputs"
    ):
        raise ValueError("causal closure selection provenance claim is invalid")
    _require_sha(artifact.get("source_manifest_sha256"), "source manifest")
    _verify_selection_policy(artifact.get("selection"))
    _verify_inputs(
        artifact.get("inputs"),
        expected_ids=FROZEN_INCLUDED_INPUT_IDS,
    )
    _verify_audit(artifact.get("audit"))
    _verify_windows(artifact.get("windows"))
    if not claimed or claimed != canonical_sha256(artifact):
        raise ValueError("causal closure selection hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("artifact_sha256", None)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_sources(path: Path) -> tuple[dict[str, _Source], str]:
    manifest = _read_mapping(path)
    if manifest.get("schema_version") != "agu.vru-basketball-source-manifest.v1":
        raise ValueError("unsupported VRU source manifest")
    if manifest.get("runtime_consumable") is not False:
        raise ValueError("VRU source manifest must remain offline-only")
    if manifest.get("codex_runtime_answer_used") is not False:
        raise ValueError("VRU source manifest cannot be a runtime answer channel")
    videos = manifest.get("videos")
    if not isinstance(videos, list):
        raise ValueError("VRU source manifest requires videos")
    sources: dict[str, _Source] = {}
    for value in videos:
        if not isinstance(value, Mapping):
            raise ValueError("VRU source manifest video rows must be mappings")
        source_id = str(value.get("location") or "")
        fps = _finite_float(value.get("fps"), "source FPS")
        frame_count = _positive_int(value.get("frame_count"), "source frame count")
        duration = _finite_float(value.get("duration_seconds"), "source duration")
        if duration <= 0.0:
            raise ValueError("source duration must be positive")
        source = _Source(
            source_id=source_id,
            clip_id=str(value.get("clip_id") or ""),
            video_sha256=_require_sha(value.get("sha256"), "source video"),
            video_filename=Path(str(value.get("path") or "")).name,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration,
        )
        if (
            not source.source_id
            or not source.clip_id
            or not source.video_filename
            or source.source_id in sources
        ):
            raise ValueError("VRU source manifest source fields must be non-empty and unique")
        sources[source.source_id] = source
    if tuple(sorted(sources)) != tuple(sorted(SOURCE_IDS)):
        raise ValueError("causal closure pilot requires the Hazen, Randolph and VTV sources")
    return sources, hashlib.sha256(path.read_bytes()).hexdigest()


def _discover_inputs(
    annotation_root: Path,
    *,
    max_version: int,
) -> tuple[list[tuple[int, str, Path]], list[tuple[int, str, Path]]]:
    if not annotation_root.is_dir():
        raise ValueError("annotation_root must be an existing directory")
    included: list[tuple[int, str, Path]] = []
    excluded: list[tuple[int, str, Path]] = []
    versions: set[int] = set()
    for directory in annotation_root.iterdir():
        if not directory.is_dir():
            continue
        match = ANNOTATION_PATTERN.fullmatch(directory.name)
        if match is None:
            continue
        version = int(match.group("version"))
        if version < MIN_VERSION or version > max_version:
            continue
        suffix = str(match.group("suffix") or "")
        input_id = f"v{version}{suffix}"
        if suffix == "_smoke":
            excluded.append((version, input_id, directory))
            continue
        if version in versions:
            raise ValueError(f"duplicate formal annotation version: v{version}")
        versions.add(version)
        included.append((version, input_id, directory))
    missing = sorted(set(range(MIN_VERSION, max_version + 1)) - versions)
    if missing:
        raise ValueError(f"missing formal annotation versions: {missing}")
    included.sort(key=lambda row: row[0])
    excluded.sort(key=lambda row: (row[0], row[1]))
    if [row[1] for row in excluded] != ["v22_smoke"]:
        raise ValueError("causal closure input exclusion must be exactly v22_smoke")
    return included, excluded


def _load_input(
    directory: Path,
    *,
    version: int,
    input_id: str,
    sources: Mapping[str, _Source],
) -> tuple[list[_Observation], dict[str, str]]:
    plan = verify_vru_causal_review_plan(_read_mapping(directory / "review_plan.json"))
    review = verify_vru_causal_review(
        _read_mapping(directory / "review_sealed.json"),
        plan=plan,
    )
    plan_rows = {str(row["review_id"]): row for row in plan["examples"]}
    review_rows = review.get("reviews")
    if not isinstance(review_rows, list):
        raise ValueError("sealed causal review requires rows")
    review_ids = [
        str(row.get("review_id") or "")
        for row in review_rows
        if isinstance(row, Mapping)
    ]
    if len(review_ids) != len(review_rows) or set(review_ids) != set(plan_rows):
        raise ValueError("sealed causal review must exactly cover its plan")
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("sealed causal review IDs must be unique")

    source_by_sha = {source.video_sha256: source for source in sources.values()}
    observations: list[_Observation] = []
    for value in review_rows:
        assert isinstance(value, Mapping)
        review_id = str(value["review_id"])
        planned = plan_rows[review_id]
        source_sha = _require_sha(planned.get("source_video_sha256"), "planned source video")
        source = source_by_sha.get(source_sha)
        if source is None:
            raise ValueError("planned source video is absent from the canonical manifest")
        frame_indexes = planned.get("frame_indexes")
        if not isinstance(frame_indexes, list) or len(frame_indexes) < 3:
            raise ValueError("prior causal windows must contain at least three frames")
        anchor_frame = _non_negative_int(
            frame_indexes[len(frame_indexes) // 2], "prior anchor frame"
        )
        if anchor_frame >= source.frame_count:
            raise ValueError("prior anchor frame is outside the canonical source")
        if not math.isclose(float(planned.get("source_fps")), source.fps, abs_tol=1e-9):
            raise ValueError("prior causal source FPS differs from the canonical manifest")
        if int(planned.get("frame_count")) != source.frame_count:
            raise ValueError("prior causal frame count differs from the canonical manifest")
        observations.append(
            _Observation(
                version=version,
                input_id=input_id,
                plan_sha256=str(plan["artifact_sha256"]),
                sealed_review_sha256=str(review["artifact_sha256"]),
                prior_review_id=review_id,
                source=source,
                anchor_frame=anchor_frame,
                prior_first=str(value.get("shot_sequence") or ""),
                prior_second=str(value.get("outcome") or ""),
            )
        )
    provenance = {
        "input_id": input_id,
        "plan_sha256": str(plan["artifact_sha256"]),
        "sealed_review_sha256": str(review["artifact_sha256"]),
    }
    return observations, provenance


def _build_candidates(observations: Sequence[_Observation]) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for row in observations:
        pool_index = _prior_pool_index(row)
        if pool_index is None:
            continue
        center_seconds = row.anchor_frame / row.source.fps
        radius = WINDOW_SECONDS / 2.0
        if center_seconds - radius < 0.0 or center_seconds + radius > row.source.duration_seconds:
            continue
        candidates.append(
            _Candidate(
                observation=row,
                pool_index=pool_index,
                temporal_bucket=_temporal_bucket(
                    center_seconds,
                    duration_seconds=row.source.duration_seconds,
                ),
                ranking_sha256=_ranking_sha256(
                    row.source.video_sha256,
                    row.anchor_frame,
                ),
            )
        )
    return candidates


def _prior_pool_index(row: _Observation) -> int | None:
    if row.prior_first == "shot" and row.prior_second == "unknown":
        return 0
    if row.prior_first == "uncertain":
        return 1
    return None


def _select_windows(candidates: Sequence[_Candidate]) -> list[dict[str, Any]]:
    selected_windows: list[dict[str, Any]] = []
    for source_id in SOURCE_IDS:
        source_candidates = [
            row for row in candidates if row.observation.source.source_id == source_id
        ]
        quotas = _cell_quotas(source_candidates, source_id=source_id)
        cells = [
            (pool_index, bucket, quota)
            for (pool_index, bucket), quota in quotas.items()
            if quota
        ]
        cells.sort(
            key=lambda cell: (
                cell[0],
                sum(
                    row.pool_index == cell[0] and row.temporal_bucket == cell[1]
                    for row in source_candidates
                ),
                TEMPORAL_BUCKETS.index(cell[1]),
            )
        )
        selected: list[_Candidate] = []
        for pool_index, bucket, quota in cells:
            ranked = sorted(
                (
                    row
                    for row in source_candidates
                    if row.pool_index == pool_index and row.temporal_bucket == bucket
                ),
                key=lambda row: row.ranking_sha256,
            )
            accepted = 0
            for row in ranked:
                if any(_overlaps(row, other) for other in selected):
                    continue
                selected.append(row)
                accepted += 1
                if accepted == quota:
                    break
            if accepted != quota:
                raise ValueError(
                    f"source {source_id} cannot satisfy its non-overlapping closure quota"
                )
        if len(selected) != 8:
            raise ValueError(f"source {source_id} must select exactly eight windows")
        selected.sort(key=lambda row: row.observation.anchor_frame)
        for index, row in enumerate(selected, start=1):
            observation = row.observation
            source = observation.source
            center = observation.anchor_frame / source.fps
            radius = WINDOW_SECONDS / 2.0
            selected_windows.append(
                {
                    "review_id": f"closure-{source_id}-{index:04d}",
                    "source_id": source_id,
                    "clip_id": source.clip_id,
                    "source_video_sha256": source.video_sha256,
                    "source_video_filename": source.video_filename,
                    "source_fps": source.fps,
                    "frame_count": source.frame_count,
                    "anchor_frame": observation.anchor_frame,
                    "center_seconds": round(center, 6),
                    "start_seconds": round(center - radius, 6),
                    "end_seconds": round(center + radius, 6),
                    "temporal_bucket": row.temporal_bucket,
                    "ranking_sha256": row.ranking_sha256,
                }
            )
    return selected_windows


def _cell_quotas(
    candidates: Sequence[_Candidate],
    *,
    source_id: str,
) -> dict[tuple[int, str], int]:
    first_total, second_total = PRIOR_POOL_QUOTA_PAIRS[source_id]
    availability = Counter((row.pool_index, row.temporal_bucket) for row in candidates)
    first = {bucket: 1 for bucket in TEMPORAL_BUCKETS}
    if first_total < len(TEMPORAL_BUCKETS):
        raise ValueError("first prior pool must cover every temporal bucket")
    if any(availability[(0, bucket)] < 1 for bucket in TEMPORAL_BUCKETS):
        raise ValueError(f"source {source_id} lacks first-pool temporal coverage")
    for _ in range(first_total - len(TEMPORAL_BUCKETS)):
        choices = [
            bucket
            for bucket in TEMPORAL_BUCKETS
            if first[bucket] < availability[(0, bucket)]
            and first[bucket] < TEMPORAL_BUCKET_QUOTAS[bucket]
        ]
        if not choices:
            raise ValueError(f"source {source_id} lacks first-pool quota capacity")
        bucket = max(
            choices,
            key=lambda value: (
                availability[(0, value)] - first[value],
                -TEMPORAL_BUCKETS.index(value),
            ),
        )
        first[bucket] += 1
    second = {
        bucket: TEMPORAL_BUCKET_QUOTAS[bucket] - first[bucket]
        for bucket in TEMPORAL_BUCKETS
    }
    if sum(second.values()) != second_total:
        raise ValueError(f"source {source_id} prior-pool and temporal quotas disagree")
    if any(availability[(1, bucket)] < second[bucket] for bucket in TEMPORAL_BUCKETS):
        raise ValueError(f"source {source_id} lacks second-pool temporal capacity")
    return {
        **{(0, bucket): first[bucket] for bucket in TEMPORAL_BUCKETS},
        **{(1, bucket): second[bucket] for bucket in TEMPORAL_BUCKETS},
    }


def _verify_selection_policy(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("causal closure selection policy is required")
    expected = {
        "ranking_namespace": RANKING_NAMESPACE,
        "ranking_direction": "ascending_sha256",
        "cell_selection_order": (
            "prior_pool_then_available_count_then_early_middle_late"
        ),
        "physical_window_supersession": (
            "highest_formal_version_then_input_id_then_review_id"
        ),
        "source_ids": list(SOURCE_IDS),
        "per_source": 8,
        "window_seconds": WINDOW_SECONDS,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "sample_count": SAMPLE_COUNT,
        "interval_semantics": "half_open",
        "prior_pool_quota_pairs": {
            source_id: list(PRIOR_POOL_QUOTA_PAIRS[source_id])
            for source_id in SOURCE_IDS
        },
        "temporal_bucket_quotas": {
            source_id: dict(TEMPORAL_BUCKET_QUOTAS) for source_id in SOURCE_IDS
        },
    }
    if dict(value) != expected:
        raise ValueError("causal closure selection policy is not the frozen v1 policy")


def _verify_inputs(value: Any, *, expected_ids: Sequence[str]) -> None:
    if not isinstance(value, list) or len(value) != len(expected_ids):
        raise ValueError("causal closure input provenance is incomplete")
    ids: list[str] = []
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {
            "input_id",
            "plan_sha256",
            "sealed_review_sha256",
        }:
            raise ValueError("causal closure input provenance row is invalid")
        input_id = row.get("input_id")
        if not isinstance(input_id, str):
            raise ValueError("causal closure input ID must be a scalar string")
        ids.append(input_id)
        _require_sha(row["plan_sha256"], "input plan")
        _require_sha(row["sealed_review_sha256"], "input sealed review")
    if ids != list(expected_ids):
        raise ValueError("causal closure input IDs are not the frozen ordered set")


def _verify_audit(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("causal closure audit is required")
    if set(value) != AUDIT_FIELDS:
        raise ValueError("causal closure audit fields are not canonical")
    if value.get("included_input_count") != 39:
        raise ValueError("causal closure audit included-input count is invalid")
    if value.get("excluded_input_ids") != ["v22_smoke"]:
        raise ValueError("causal closure audit must exclude v22_smoke exactly")
    excluded = value.get("excluded_inputs")
    _verify_inputs(excluded, expected_ids=("v22_smoke",))
    for field in (
        "observation_count",
        "physical_window_count",
        "collision_group_count",
        "conflicting_physical_window_count",
        "candidate_physical_window_count",
        "selectable_physical_window_count",
    ):
        _positive_int(value.get(field), f"audit {field}")


def _verify_windows(value: Any) -> None:
    if not isinstance(value, list) or len(value) != 24:
        raise ValueError("causal closure selection requires exactly 24 windows")
    if any(not isinstance(row, Mapping) for row in value):
        raise ValueError("causal closure window rows must be mappings")
    expected_order = sorted(
        value,
        key=lambda row: (str(row.get("source_id")), float(row.get("center_seconds"))),
    )
    if value != expected_order:
        raise ValueError("causal closure windows must use canonical source/time order")
    counts: Counter[str] = Counter()
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    physical_keys: set[tuple[str, int]] = set()
    for row in value:
        if not isinstance(row, Mapping) or set(row) != WINDOW_FIELDS:
            raise ValueError("causal closure window fields are invalid")
        review_id = row.get("review_id")
        source_id = row.get("source_id")
        clip_id = row.get("clip_id")
        source_filename = row.get("source_video_filename")
        if (
            not isinstance(review_id, str)
            or not isinstance(source_id, str)
            or not isinstance(clip_id, str)
            or not clip_id
            or not isinstance(source_filename, str)
            or not source_filename
            or Path(source_filename).name != source_filename
        ):
            raise ValueError(
                "causal closure window IDs and filename must be non-empty scalars"
            )
        if NEUTRAL_ID_PATTERN.fullmatch(review_id) is None or source_id not in SOURCE_IDS:
            raise ValueError("causal closure review IDs must be neutral and source-bound")
        source_sha = _require_sha(row.get("source_video_sha256"), "window source video")
        ranking_sha = _require_sha(row.get("ranking_sha256"), "window ranking")
        fps = _finite_float(row.get("source_fps"), "window source FPS")
        frame_count = _positive_int(row.get("frame_count"), "window frame count")
        anchor_frame = _non_negative_int(row.get("anchor_frame"), "window anchor frame")
        if anchor_frame >= frame_count:
            raise ValueError("causal closure anchor frame is outside its source")
        center = _finite_float(row.get("center_seconds"), "window center")
        start = _finite_non_negative_float(
            row.get("start_seconds"), "window start"
        )
        end = _finite_float(row.get("end_seconds"), "window end")
        if not math.isclose(center, round(anchor_frame / fps, 6), abs_tol=1e-6):
            raise ValueError("causal closure center is not anchor-frame bound")
        if not math.isclose(start, round(center - WINDOW_SECONDS / 2.0, 6), abs_tol=1e-6):
            raise ValueError("causal closure start is invalid")
        if not math.isclose(end, round(center + WINDOW_SECONDS / 2.0, 6), abs_tol=1e-6):
            raise ValueError("causal closure end is invalid")
        if ranking_sha != _ranking_sha256(source_sha, anchor_frame):
            raise ValueError("causal closure ranking SHA is invalid")
        physical_key = (source_sha, anchor_frame)
        if physical_key in physical_keys:
            raise ValueError("causal closure physical windows must be globally unique")
        physical_keys.add(physical_key)
        sample_frames = [
            round((start + index / SAMPLE_RATE_HZ) * fps)
            for index in range(SAMPLE_COUNT)
        ]
        if (
            sample_frames != sorted(set(sample_frames))
            or sample_frames[0] < 0
            or sample_frames[-1] >= frame_count
            or sample_frames[SAMPLE_COUNT // 2] != anchor_frame
            or sample_frames[-1] >= round(end * fps)
        ):
            raise ValueError(
                "causal closure strict samples must be unique, bounded, "
                "half-open and anchor-aligned"
            )
        bucket = row.get("temporal_bucket")
        if not isinstance(bucket, str):
            raise ValueError("causal closure temporal bucket must be a scalar")
        if bucket not in TEMPORAL_BUCKETS:
            raise ValueError("causal closure temporal bucket is invalid")
        counts[source_id] += 1
        buckets[source_id][bucket] += 1
        by_source[source_id].append(row)
    if counts != Counter({source_id: 8 for source_id in SOURCE_IDS}):
        raise ValueError("causal closure selection must contain eight windows per source")
    for source_id in SOURCE_IDS:
        if buckets[source_id] != Counter(TEMPORAL_BUCKET_QUOTAS):
            raise ValueError("causal closure temporal quotas are invalid")
        expected_ids = [f"closure-{source_id}-{index:04d}" for index in range(1, 9)]
        if [str(row["review_id"]) for row in by_source[source_id]] != expected_ids:
            raise ValueError("causal closure neutral IDs are not canonical")
        for left, right in zip(by_source[source_id], by_source[source_id][1:], strict=False):
            if float(left["end_seconds"]) > float(right["start_seconds"]):
                raise ValueError("causal closure half-open windows must not overlap")


def _reject_label_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = FORBIDDEN_FIELDS.intersection(str(key) for key in value)
        if forbidden:
            raise ValueError(f"label-bearing causal closure fields are forbidden: {sorted(forbidden)}")
        for child in value.values():
            _reject_label_fields(child)
    elif isinstance(value, list):
        for child in value:
            _reject_label_fields(child)


def _ranking_sha256(source_sha256: str, anchor_frame: int) -> str:
    value = f"{RANKING_NAMESPACE}\0{source_sha256}\0{anchor_frame}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _temporal_bucket(center_seconds: float, *, duration_seconds: float) -> str:
    fraction = center_seconds / duration_seconds
    if fraction < 1.0 / 3.0:
        return "early"
    if fraction < 2.0 / 3.0:
        return "middle"
    return "late"


def _overlaps(left: _Candidate, right: _Candidate) -> bool:
    left_center = left.observation.anchor_frame / left.observation.source.fps
    right_center = right.observation.anchor_frame / right.observation.source.fps
    radius = WINDOW_SECONDS / 2.0
    return left_center - radius < right_center + radius and right_center - radius < left_center + radius


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read causal closure input: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"causal closure input must be an object: {path}")
    return value


def _require_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return number


def _finite_non_negative_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite and non-negative")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


def _require_output_disjoint_from_inputs(
    *,
    annotation_root: Path,
    source_manifest_path: Path,
    output_path: Path,
    max_version: int,
) -> tuple[Path, Path, Path]:
    resolved_annotation_root = annotation_root.resolve()
    resolved_source_manifest = source_manifest_path.resolve()
    resolved_output = output_path.resolve()
    resolved_inputs = {
        resolved_annotation_root,
        resolved_source_manifest,
    }
    if resolved_annotation_root.is_dir():
        for directory in resolved_annotation_root.iterdir():
            if not directory.is_dir():
                continue
            match = ANNOTATION_PATTERN.fullmatch(directory.name)
            if match is None:
                continue
            version = int(match.group("version"))
            if version < MIN_VERSION or version > max_version:
                continue
            resolved_directory = directory.resolve()
            resolved_inputs.update(
                {
                    resolved_directory,
                    (resolved_directory / "review_plan.json").resolve(),
                    (resolved_directory / "review_sealed.json").resolve(),
                }
            )
    if resolved_output in resolved_inputs:
        raise ValueError(
            "output path must not alias an annotation or source manifest input path"
        )
    return resolved_annotation_root, resolved_source_manifest, resolved_output


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
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
    args = parse_args()
    annotation_root, source_manifest_path, output_path = (
        _require_output_disjoint_from_inputs(
            annotation_root=args.annotation_root,
            source_manifest_path=args.source_manifest,
            output_path=args.output,
            max_version=args.max_version,
        )
    )
    artifact = build_causal_closure_selection(
        annotation_root=annotation_root,
        source_manifest_path=source_manifest_path,
        max_version=args.max_version,
    )
    _write_json_atomic(output_path, artifact)
    print(
        json.dumps(
            {
                "artifact_sha256": artifact["artifact_sha256"],
                "output": output_path.as_posix(),
                "windows": len(artifact["windows"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
