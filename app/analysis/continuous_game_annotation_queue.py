"""Label-hidden, hash-bound coverage queues for manual full-game annotation.

The queue is an offline worklist, not a model input.  It records deterministic
time/frame windows over retained continuous originals so manual review can be
expanded from a small pilot to full-game coverage without silently adding
labels, target answers, or runtime dependencies.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from typing import Any

QUEUE_SCHEMA = "agu.continuous-game-annotation-queue.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_LABEL_FIELDS = {
    "event_present",
    "ground_truth",
    "label",
    "outcome",
    "review_note",
    "shot_sequence",
    "target",
}


def build_continuous_game_annotation_queue(
    sources: Sequence[Mapping[str, Any]],
    *,
    stride_seconds: float = 10.0,
    window_seconds: float = 2.0,
) -> dict[str, Any]:
    """Build deterministic coverage windows over every supplied source.

    ``stride_seconds`` controls the first-pass coverage grid.  Later visual
    review may expand selected windows with a denser causal plan; this function
    deliberately never emits labels or reads a review artifact.
    """

    stride = _positive_float(stride_seconds, "stride_seconds")
    window = _positive_float(window_seconds, "window_seconds")
    if window > stride:
        raise ValueError("window_seconds cannot exceed stride_seconds")
    if not sources:
        raise ValueError("annotation queue requires at least one source")

    normalized_sources: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for value in sources:
        if not isinstance(value, Mapping):
            raise ValueError("annotation queue sources must be mappings")
        if _FORBIDDEN_LABEL_FIELDS.intersection(value):
            raise ValueError("label-bearing source metadata is forbidden")
        source_id = str(value.get("source_id") or "")
        filename = str(value.get("source_video_filename") or "")
        source_sha = _require_sha(value.get("source_video_sha256"), "source video")
        fps = _positive_float(value.get("source_fps"), "source_fps")
        frame_count = _positive_int(value.get("frame_count"), "frame_count")
        duration = _positive_float(value.get("duration_seconds"), "duration_seconds")
        if not source_id or source_id in seen_ids:
            raise ValueError("source IDs must be unique")
        if not filename:
            raise ValueError("source_video_filename is required")
        if source_sha in seen_hashes:
            raise ValueError("source video hashes must be unique")
        if duration * fps > frame_count + 2.0:
            raise ValueError("duration and frame_count are inconsistent")
        seen_ids.add(source_id)
        seen_hashes.add(source_sha)
        normalized_sources.append(
            {
                "source_id": source_id,
                "source_video_filename": filename,
                "source_video_sha256": source_sha,
                "source_fps": fps,
                "frame_count": frame_count,
                "duration_seconds": duration,
            }
        )

    normalized_sources.sort(key=lambda row: str(row["source_id"]))
    windows: list[dict[str, Any]] = []
    for source in normalized_sources:
        duration = float(source["duration_seconds"])
        source_fps = float(source["source_fps"])
        max_start = max(0.0, duration - window)
        index = 0
        start_seconds = 0.0
        while start_seconds <= max_start + 1e-9:
            end_seconds = min(duration, start_seconds + window)
            start_frame = min(
                int(source["frame_count"]) - 1,
                max(0, int(round(start_seconds * source_fps))),
            )
            end_frame = min(
                int(source["frame_count"]),
                max(start_frame + 1, int(round(end_seconds * source_fps))),
            )
            windows.append(
                {
                    "window_id": f"{source['source_id']}-coverage-{index:06d}",
                    "source_id": source["source_id"],
                    "source_video_filename": source["source_video_filename"],
                    "source_video_sha256": source["source_video_sha256"],
                    "source_fps": source_fps,
                    "start_seconds": round(start_seconds, 6),
                    "end_seconds": round(end_seconds, 6),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "layer": "uniform_coverage",
                }
            )
            index += 1
            start_seconds += stride

    artifact: dict[str, Any] = {
        "schema_version": QUEUE_SCHEMA,
        "purpose": "offline_continuous_game_manual_annotation_queue",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "labels_hidden_from_reviewer": True,
        "reviewer_visible_fields": [
            "window_id",
            "source_id",
            "source_video_filename",
            "source_video_sha256",
            "source_fps",
            "start_seconds",
            "end_seconds",
            "start_frame",
            "end_frame",
            "layer",
        ],
        "sampling": {
            "stride_seconds": stride,
            "window_seconds": window,
            "coverage": "uniform_start_grid",
        },
        "source_video_sha256s": sorted(seen_hashes),
        "sources": normalized_sources,
        "windows": windows,
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return verify_continuous_game_annotation_queue(artifact)


def select_annotation_batch(
    queue: Mapping[str, Any], *, per_source: int, seed: int
) -> list[dict[str, Any]]:
    """Select a deterministic, source-balanced review batch without labels."""

    verified = verify_continuous_game_annotation_queue(queue)
    if isinstance(per_source, bool) or not isinstance(per_source, int) or per_source <= 0:
        raise ValueError("per_source must be a positive integer")
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for source_id in sorted(str(row["source_id"]) for row in verified["sources"]):
        rows = [row for row in verified["windows"] if row["source_id"] == source_id]
        if len(rows) < per_source:
            raise ValueError(f"source {source_id} has too few queue windows")
        selected.extend(rng.sample(rows, per_source))
    return sorted(selected, key=lambda row: (str(row["source_id"]), str(row["window_id"])))


def verify_continuous_game_annotation_queue(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify hashes, source/window provenance and absence of target labels."""

    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    if artifact.get("schema_version") != QUEUE_SCHEMA:
        raise ValueError("unsupported continuous annotation queue schema")
    if artifact.get("purpose") != "offline_continuous_game_manual_annotation_queue":
        raise ValueError("annotation queue must remain offline-only")
    if artifact.get("runtime_consumable") is not False:
        raise ValueError("annotation queue cannot be runtime consumable")
    if artifact.get("codex_runtime_answer_used") is not False:
        raise ValueError("annotation queue cannot be a runtime answer channel")
    if artifact.get("labels_hidden_from_reviewer") is not True:
        raise ValueError("annotation queue must hide labels")
    sources = artifact.get("sources")
    windows = artifact.get("windows")
    if not isinstance(sources, list) or not sources or not isinstance(windows, list) or not windows:
        raise ValueError("annotation queue requires sources and windows")
    source_by_id: dict[str, Mapping[str, Any]] = {}
    source_hashes: set[str] = set()
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("annotation queue source rows must be mappings")
        if _FORBIDDEN_LABEL_FIELDS.intersection(source):
            raise ValueError("label-bearing source metadata is forbidden")
        source_id = str(source.get("source_id") or "")
        source_sha = _require_sha(source.get("source_video_sha256"), "source video")
        if not source_id or source_id in source_by_id or source_sha in source_hashes:
            raise ValueError("annotation queue source provenance is not unique")
        _positive_float(source.get("source_fps"), "source_fps")
        _positive_int(source.get("frame_count"), "frame_count")
        _positive_float(source.get("duration_seconds"), "duration_seconds")
        source_by_id[source_id] = source
        source_hashes.add(source_sha)
    if sorted(source_hashes) != artifact.get("source_video_sha256s"):
        raise ValueError("annotation queue source hash coverage mismatch")

    seen_windows: set[str] = set()
    for row in windows:
        if not isinstance(row, Mapping):
            raise ValueError("annotation queue window rows must be mappings")
        if _FORBIDDEN_LABEL_FIELDS.intersection(row):
            raise ValueError("label-bearing queue window is forbidden")
        window_id = str(row.get("window_id") or "")
        source_id = str(row.get("source_id") or "")
        if not window_id or window_id in seen_windows or source_id not in source_by_id:
            raise ValueError("annotation queue window provenance is invalid")
        source = source_by_id[source_id]
        start_frame = _nonnegative_int(row.get("start_frame"), "start_frame")
        end_frame = _positive_int(row.get("end_frame"), "end_frame")
        frame_count = int(source["frame_count"])
        if start_frame >= end_frame or end_frame > frame_count:
            raise ValueError("annotation queue window frame bounds are invalid")
        start_seconds = _nonnegative_float(row.get("start_seconds"), "start_seconds")
        end_seconds = _positive_float(row.get("end_seconds"), "end_seconds")
        if end_seconds <= start_seconds or end_seconds > float(source["duration_seconds"]) + 1e-5:
            raise ValueError("annotation queue window time bounds are invalid")
        if row.get("source_video_sha256") != source["source_video_sha256"]:
            raise ValueError("annotation queue window source hash mismatch")
        if row.get("layer") != "uniform_coverage":
            raise ValueError("unsupported annotation queue layer")
        seen_windows.add(window_id)
    if not claimed or claimed != canonical_sha256(artifact):
        raise ValueError("continuous annotation queue hash mismatch")
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


def _require_sha(value: Any, field: str) -> str:
    text = str(value or "")
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return text


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _positive_float(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{field} must be positive")
    return parsed


def _nonnegative_float(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return parsed
