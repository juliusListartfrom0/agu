"""Deterministic, hash-bound sampling for offline cross-game shot reviews."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Mapping
from typing import Any

SHOT_WINDOW_SPEC_SCHEMA = "agu.cross-game-shot-window-spec.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def select_uniform_shot_windows(
    *,
    frame_count: int,
    fps: float,
    count: int,
    window_frames: int,
    minimum_center_spacing_frames: int,
    seed: int,
) -> list[dict[str, int | float | str]]:
    """Select deterministic, non-overlapping windows without using labels.

    The sampler first reserves the requested spacing, then distributes the
    remaining slack with a seeded pseudo-random composition.  This avoids a
    potentially enormous list of every frame while retaining reproducibility.
    """

    if frame_count <= 0 or fps <= 0:
        raise ValueError("frame_count and fps must be positive")
    if count <= 0 or window_frames <= 1 or minimum_center_spacing_frames <= 0:
        raise ValueError("count, window_frames and spacing must be positive")
    half = window_frames // 2
    lower = half
    upper = frame_count - (window_frames - half) - 1
    if upper < lower:
        raise ValueError("window_frames exceed the source frame count")
    required_span = (count - 1) * minimum_center_spacing_frames
    available_span = upper - lower
    if required_span > available_span:
        raise ValueError("cannot place requested windows with the configured spacing")

    rng = random.Random(seed)
    slack = available_span - required_span
    # Spread slack over the leading margin, every inter-window gap and the
    # trailing margin.  A sequential ``randrange`` composition tends to put
    # most of the remaining span in its final gap, which can silently bias a
    # nominally uniform review toward one part of a game.
    slot_count = count + 1
    if slack:
        cuts = sorted(rng.sample(range(slack + slot_count - 1), slot_count - 1))
        boundaries = [-1, *cuts, slack + slot_count - 1]
        extras = [
            boundaries[index + 1] - boundaries[index] - 1
            for index in range(slot_count)
        ]
    else:
        extras = [0] * slot_count
    start_offset = extras[0]
    gaps = [
        minimum_center_spacing_frames + extras[index + 1]
        for index in range(count - 1)
    ]

    centers: list[int] = []
    center = lower + start_offset
    for index in range(count):
        centers.append(center)
        if index < len(gaps):
            center += gaps[index]

    rows: list[dict[str, int | float | str]] = []
    for index, anchor in enumerate(centers, 1):
        start = anchor - half
        end = start + window_frames - 1
        rows.append(
            {
                "event_id": f"raw-shot-{index:03d}",
                "start_frame": start,
                "end_frame": end,
                "anchor_frame": anchor,
                "source_fps": float(fps),
            }
        )
    return rows


def seal_shot_window_spec(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Attach a canonical SHA-256 and validate an offline window spec."""

    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return verify_shot_window_spec(artifact)


def verify_shot_window_spec(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a sealed window spec and its candidate geometry."""

    artifact = dict(payload)
    if artifact.get("schema_version") != SHOT_WINDOW_SPEC_SCHEMA:
        raise ValueError("unsupported shot window spec schema")
    if artifact.get("purpose") != "offline_training_annotation_only":
        raise ValueError("shot window specs must remain training-only")
    if artifact.get("runtime_consumable") is not False:
        raise ValueError("shot window specs cannot be runtime-consumable")
    source_sha = str(artifact.get("source_video_sha256") or "")
    if not _SHA256_RE.fullmatch(source_sha):
        raise ValueError("source_video_sha256 must be a lowercase SHA-256")
    frame_count = int(artifact.get("video_frame_count", -1))
    fps = float(artifact.get("video_fps", 0.0))
    window_frames = int(artifact.get("window_frames", 0))
    spacing = int(artifact.get("minimum_center_spacing_frames", 0))
    if frame_count <= 0 or fps <= 0 or window_frames <= 1 or spacing <= 0:
        raise ValueError("shot window geometry is invalid")
    candidates = artifact.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("shot window spec requires candidates")
    seen_ids: set[str] = set()
    centers: list[int] = []
    half = window_frames // 2
    for row in candidates:
        if not isinstance(row, Mapping):
            raise ValueError("shot window candidate must be an object")
        event_id = str(row.get("event_id") or "")
        start = int(row.get("start_frame", -1))
        end = int(row.get("end_frame", -1))
        anchor = int(row.get("anchor_frame", -1))
        if (
            not event_id
            or event_id in seen_ids
            or start < 0
            or start >= end
            or end >= frame_count
            or not start <= anchor <= end
        ):
            raise ValueError("invalid or duplicated shot window candidate")
        if anchor - start != half or end - anchor != window_frames - half - 1:
            raise ValueError("candidate does not match the declared window geometry")
        seen_ids.add(event_id)
        centers.append(anchor)
    if any(second - first < spacing for first, second in zip(centers, centers[1:])):
        raise ValueError("candidate centers violate the declared spacing")
    claimed = str(artifact.pop("artifact_sha256", ""))
    if not claimed or claimed != _canonical_sha256(artifact):
        raise ValueError("shot window spec hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
