"""Hash-bound full causal windows for development-only basketball detection."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from app.analysis.causal_shot_phase_review import (
    CAUSAL_PHASE_OFFSETS_SECONDS,
    verify_causal_shot_phase_review_plan,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


def causal_ball_windows(
    examples: Sequence[Mapping[str, Any]],
    *,
    source_video_sha256: str,
    merge_gap_frames: int,
) -> tuple[list[tuple[int, int]], list[str]]:
    """Return merged exact `-4s..+4s` windows for one planned source."""

    source = _require_sha256(source_video_sha256)
    if merge_gap_frames < 0:
        raise ValueError("causal ball merge gap must be non-negative")
    selected = [
        row for row in examples if row.get("source_video_sha256") == source
    ]
    if not selected:
        raise ValueError("causal plan has no events for source video")
    windows = []
    review_ids = []
    seen = set()
    for row in selected:
        review_id = str(row.get("phase_review_id") or "")
        indexes = row.get("frame_indexes")
        if (
            not review_id
            or review_id in seen
            or not isinstance(indexes, list)
            or len(indexes) != len(CAUSAL_PHASE_OFFSETS_SECONDS)
            or any(int(value) < 0 for value in indexes)
        ):
            raise ValueError("invalid causal ball plan example")
        seen.add(review_id)
        review_ids.append(review_id)
        windows.append((min(map(int, indexes)), max(map(int, indexes)) + 1))
    return _merge_windows(windows, maximum_gap=merge_gap_frames), sorted(
        review_ids
    )


def sampled_frame_count(
    windows: Sequence[tuple[int, int]],
    *,
    stride_frames: int,
) -> int:
    if stride_frames < 1:
        raise ValueError("causal ball stride must be positive")
    return sum(
        len(range(int(start), int(end), stride_frames))
        for start, end in windows
    )


def verify_causal_ball_perception(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify one complete plan-only ball-perception artifact."""

    verified_plan = verify_causal_shot_phase_review_plan(plan)
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    if artifact.get("schema_version") != "agu.official-perception.v1":
        raise ValueError("unsupported causal ball perception schema")
    raw_video = artifact.get("raw_video")
    detector = artifact.get("detector")
    sampling = artifact.get("sampling")
    causal_source = artifact.get("causal_source")
    detections = artifact.get("detections")
    tracks = artifact.get("ball_tracks")
    counts = artifact.get("counts")
    if not all(
        isinstance(value, Mapping)
        for value in (raw_video, detector, sampling, causal_source, counts)
    ) or not isinstance(detections, list) or not isinstance(tracks, list):
        raise ValueError("causal ball perception payload is invalid")

    source = _require_sha256(raw_video.get("sha256"))
    if (
        source not in verified_plan["source_video_sha256s"]
        or source in verified_plan["sealed_blind_video_sha256s"]
    ):
        raise ValueError("sealed blind or unplanned causal ball source")
    source_fps = float(raw_video.get("source_fps", 0.0))
    frame_count = int(raw_video.get("frame_count", 0))
    requested_fps = float(sampling.get("requested_fps", 0.0))
    stride = int(sampling.get("stride_frames", 0))
    merge_gap_sec = float(causal_source.get("merge_gap_sec", -1.0))
    if (
        source_fps <= 0
        or frame_count < 1
        or requested_fps <= 0
        or stride != max(1, round(source_fps / requested_fps))
        or merge_gap_sec < 0
    ):
        raise ValueError("causal ball sampling metadata is invalid")
    windows, review_ids = causal_ball_windows(
        verified_plan["examples"],
        source_video_sha256=source,
        merge_gap_frames=round(merge_gap_sec * source_fps),
    )
    observed_windows = sampling.get("windows")
    expected_windows = [
        {"start_frame": start, "end_frame": end} for start, end in windows
    ]
    if observed_windows != expected_windows:
        raise ValueError("causal ball sampling windows mismatch")
    expected_samples = sampled_frame_count(windows, stride_frames=stride)
    if (
        int(sampling.get("start_frame", -1)) != windows[0][0]
        or int(sampling.get("end_frame", -1)) != windows[-1][1]
        or int(sampling.get("decoded_frame_count", -1))
        != sum(end - start for start, end in windows)
        or int(sampling.get("sample_count", -1)) != expected_samples
    ):
        raise ValueError("causal ball sample count or bounds mismatch")
    if (
        causal_source.get("review_plan_sha256")
        != verified_plan["artifact_sha256"]
        or causal_source.get("phase_review_ids") != review_ids
        or int(causal_source.get("event_count", -1)) != len(review_ids)
        or causal_source.get("complete_run") is not True
    ):
        raise ValueError("causal ball run is not complete or plan-bound")
    _require_sha256(detector.get("model_sha256"))
    if detector.get("object_type_allowlist") != ["basketball"]:
        raise ValueError("causal ball detector allowlist is invalid")
    sampled_frames = {
        frame
        for start, end in windows
        for frame in range(start, end, stride)
    }
    if any(
        row.get("object_type") != "basketball"
        or int(row.get("frame", -1)) not in sampled_frames
        for row in detections
    ) or int(counts.get("basketball", -1)) != len(detections):
        raise ValueError("causal ball detections are invalid")
    if claimed != _canonical_sha256(artifact):
        raise ValueError("causal ball perception hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _merge_windows(
    windows: Sequence[tuple[int, int]],
    *,
    maximum_gap: int,
) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(windows):
        if end <= start:
            raise ValueError("causal ball window is empty")
        if merged and start <= merged[-1][1] + maximum_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _require_sha256(value: object) -> str:
    text = str(value or "")
    if _SHA256.fullmatch(text) is None:
        raise ValueError("invalid causal ball source SHA-256")
    return text


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
