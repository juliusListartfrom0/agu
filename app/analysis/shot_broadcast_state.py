"""Label-free continuous broadcast-clock state evidence for shot screening."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from app.analysis.broadcast_clock import verify_broadcast_clock_artifact
from app.analysis.shot_validity_scene_state import verify_scene_embedding_artifact

BROADCAST_STATE_EVIDENCE_SCHEMA = "agu.shot-broadcast-state-evidence.v1"
FEATURE_NAMES = (
    "sample_count",
    "read_count",
    "read_fraction",
    "pre_anchor_read_count",
    "post_anchor_read_count",
    "distinct_clock_count",
    "distinct_period_count",
    "dominant_clock_fraction",
    "adjacent_comparable_count",
    "frozen_adjacent_fraction",
    "clock_elapsed_seconds",
    "observed_offset_span_seconds",
    "clock_elapsed_to_offset_ratio",
    "monotonic_violation_count",
    "mean_absolute_clock_step_seconds",
    "maximum_absolute_clock_step_seconds",
    "leading_missing_fraction",
    "trailing_missing_fraction",
    "longest_missing_run_fraction",
    "frozen_clock_flag",
    "disappearance_after_read_flag",
)
_FORBIDDEN_LABEL_FIELDS = {
    "event_present",
    "ground_truth",
    "label",
    "review_note",
    "target",
}


def extract_broadcast_state_features(event: Mapping[str, Any]) -> dict[str, float]:
    """Summarize OCR clock continuity without labels or review decisions."""

    samples = event.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("broadcast state features require clock samples")
    ordered = sorted(samples, key=lambda row: float(row["offset_seconds"]))
    offsets = [float(row["offset_seconds"]) for row in ordered]
    if (
        len(set(offsets)) != len(offsets)
        or not all(math.isfinite(value) for value in offsets)
    ):
        raise ValueError("broadcast state sample offsets must be finite and unique")

    reads: list[tuple[float, int, int]] = []
    missing: list[bool] = []
    for sample in ordered:
        read = sample.get("read")
        missing.append(read is None)
        if read is None:
            continue
        if not isinstance(read, Mapping):
            raise ValueError("broadcast state clock read must be an object")
        period = int(read.get("period", 0))
        seconds = int(read.get("clock_seconds", -1))
        if period not in {1, 2, 3, 4, 5} or not 0 <= seconds <= 12 * 60:
            raise ValueError("broadcast state clock read is invalid")
        reads.append((float(sample["offset_seconds"]), period, seconds))

    clock_counts = Counter((period, seconds) for _, period, seconds in reads)
    comparable = [
        (first, second)
        for first, second in zip(reads, reads[1:], strict=False)
        if first[1] == second[1]
    ]
    clock_steps = [first[2] - second[2] for first, second in comparable]
    observed_spans = [second[0] - first[0] for first, second in comparable]
    elapsed = float(sum(max(0, step) for step in clock_steps))
    observed_span = float(sum(observed_spans))
    sample_count = len(ordered)
    read_count = len(reads)
    first_read_index = next(
        (index for index, value in enumerate(missing) if not value),
        sample_count,
    )
    last_read_index = next(
        (
            index
            for index in range(sample_count - 1, -1, -1)
            if not missing[index]
        ),
        -1,
    )
    trailing_missing_count = sample_count - last_read_index - 1
    longest_missing_run = _longest_true_run(missing)
    dominant_clock_fraction = (
        max(clock_counts.values()) / read_count if read_count else 0.0
    )
    frozen_clock = read_count >= 2 and dominant_clock_fraction == 1.0
    values = (
        float(sample_count),
        float(read_count),
        read_count / sample_count,
        float(sum(read is not None for read, offset in _reads_with_offsets(ordered) if offset < 0.0)),
        float(sum(read is not None for read, offset in _reads_with_offsets(ordered) if offset >= 0.0)),
        float(len(clock_counts)),
        float(len({period for _, period, _ in reads})),
        float(dominant_clock_fraction),
        float(len(comparable)),
        (
            sum(step == 0 for step in clock_steps) / len(clock_steps)
            if clock_steps
            else 0.0
        ),
        elapsed,
        observed_span,
        elapsed / observed_span if observed_span > 0.0 else 0.0,
        float(sum(step < 0 for step in clock_steps)),
        (
            sum(abs(step) for step in clock_steps) / len(clock_steps)
            if clock_steps
            else 0.0
        ),
        float(max((abs(step) for step in clock_steps), default=0)),
        first_read_index / sample_count,
        trailing_missing_count / sample_count,
        longest_missing_run / sample_count,
        float(frozen_clock),
        float(read_count > 0 and trailing_missing_count > 0),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("broadcast state features must be finite")
    return dict(zip(FEATURE_NAMES, values, strict=True))


def build_broadcast_state_evidence_artifact(
    *,
    scene_artifact: Mapping[str, Any],
    clock_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Align one or more raw clock artifacts to an exact scene-example set."""

    scene = verify_scene_embedding_artifact(scene_artifact)
    if not clock_artifacts:
        raise ValueError("clock evidence must exactly cover scene examples")
    scene_rows = {_example_key(row): row for row in scene["examples"]}
    if len(scene_rows) != len(scene["examples"]):
        raise ValueError("scene examples must be unique")

    clock_rows: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    clock_hashes = []
    for raw_artifact in clock_artifacts:
        artifact = verify_broadcast_clock_artifact(raw_artifact)
        source_sha = str(artifact.get("raw_video_sha256") or "")
        bundle_sha = str(artifact.get("candidate_bundle_sha256") or "")
        if not source_sha or not bundle_sha:
            raise ValueError("clock evidence requires source and candidate binding")
        clock_hashes.append(str(artifact["artifact_sha256"]))
        for event in artifact["events"]:
            key = (source_sha, bundle_sha, str(event["event_id"]))
            if key in clock_rows:
                raise ValueError("clock evidence example is duplicated")
            clock_rows[key] = event
    if set(clock_rows) != set(scene_rows):
        raise ValueError("clock evidence must exactly cover scene examples")

    examples = []
    for row in scene["examples"]:
        key = _example_key(row)
        feature_map = extract_broadcast_state_features(clock_rows[key])
        examples.append(
            {
                "source_video_sha256": key[0],
                "candidate_bundle_sha256": key[1],
                "event_id": key[2],
                "features": [feature_map[name] for name in FEATURE_NAMES],
            }
        )
    return seal_broadcast_state_evidence_artifact(
        {
            "source_scene_artifact_sha256": scene["artifact_sha256"],
            "source_clock_artifact_sha256s": sorted(clock_hashes),
            "examples": examples,
        }
    )


def seal_broadcast_state_evidence_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["schema_version"] = BROADCAST_STATE_EVIDENCE_SCHEMA
    artifact["purpose"] = "offline_label_free_broadcast_state_screening"
    artifact["runtime_consumable"] = False
    artifact["codex_runtime_answer_used"] = False
    artifact["feature_names"] = list(FEATURE_NAMES)
    _validate_artifact(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_broadcast_state_evidence_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_artifact(artifact)
    if not claimed or claimed != _canonical_sha256(artifact):
        raise ValueError("broadcast state evidence artifact hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _validate_artifact(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != BROADCAST_STATE_EVIDENCE_SCHEMA:
        raise ValueError("unsupported broadcast state evidence schema")
    if (
        artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("broadcast state evidence must remain offline and label-free")
    if tuple(artifact.get("feature_names") or ()) != FEATURE_NAMES:
        raise ValueError("broadcast state feature contract mismatch")
    if not artifact.get("source_scene_artifact_sha256") or not artifact.get(
        "source_clock_artifact_sha256s"
    ):
        raise ValueError("broadcast state evidence requires source provenance")
    examples = artifact.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("broadcast state evidence requires examples")
    seen: set[tuple[str, str, str]] = set()
    for row in examples:
        if not isinstance(row, Mapping):
            raise ValueError("broadcast state evidence examples must be objects")
        if _FORBIDDEN_LABEL_FIELDS.intersection(row):
            raise ValueError("broadcast state evidence examples must not contain labels")
        key = _example_key(row)
        if key in seen:
            raise ValueError("broadcast state evidence example keys must be unique")
        seen.add(key)
        features = row.get("features")
        if (
            not isinstance(features, list)
            or len(features) != len(FEATURE_NAMES)
            or not all(math.isfinite(float(value)) for value in features)
        ):
            raise ValueError("broadcast state evidence feature vector is invalid")


def _example_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    key = (
        str(row.get("source_video_sha256") or ""),
        str(row.get("candidate_bundle_sha256") or ""),
        str(row.get("event_id") or ""),
    )
    if not all(key):
        raise ValueError("broadcast state evidence example key is incomplete")
    return key


def _reads_with_offsets(
    samples: Sequence[Mapping[str, Any]],
) -> list[tuple[Any, float]]:
    return [(sample.get("read"), float(sample["offset_seconds"])) for sample in samples]


def _longest_true_run(values: Sequence[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
