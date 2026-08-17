"""Deterministic, training-only windows around official PBP-aligned events."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any


def select_pbp_event_windows(
    *,
    alignment: Mapping[str, Any],
    frame_count: int,
    fps: float,
    count: int,
    window_frames: int,
    minimum_center_spacing_frames: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select balanced field-goal and hard-negative windows for training.

    The PBP event type is intentionally used for positive enrichment.  This
    function is therefore only suitable for offline supervision and must not be
    used to construct an independent evaluation set or a runtime answer.
    """

    if frame_count <= 0 or fps <= 0:
        raise ValueError("frame_count and fps must be positive")
    if count <= 0:
        raise ValueError("count must be positive")
    if window_frames <= 1:
        raise ValueError("window_frames must be greater than one")
    if minimum_center_spacing_frames <= 0:
        raise ValueError("minimum_center_spacing_frames must be positive")
    events = alignment.get("events")
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        raise ValueError("alignment events must be a sequence")

    half = window_frames // 2
    min_anchor = half
    max_anchor = frame_count - (window_frames - half) - 1
    if max_anchor < min_anchor:
        raise ValueError("window_frames exceed the source frame count")

    field_goals: list[dict[str, Any]] = []
    hard_negatives: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        raw_frame = event.get("video_frame")
        if raw_frame is None:
            continue
        try:
            frame = int(raw_frame)
        except (TypeError, ValueError):
            continue
        if not min_anchor <= frame <= max_anchor:
            continue
        is_field_goal = bool(event.get("is_field_goal"))
        action_type = str(event.get("action_type") or "").strip()
        if is_field_goal:
            field_goals.append(dict(event))
        elif action_type not in {"", "period", "Jump Ball"}:
            hard_negatives.append(dict(event))
    if not field_goals or not hard_negatives:
        raise ValueError("not enough mapped events in both event strata")

    rng = random.Random(seed)
    target_field_goals = min(len(field_goals), max(1, count // 2))
    target_negatives = min(len(hard_negatives), count - target_field_goals)
    if target_field_goals + target_negatives < count:
        # Backfill the short stratum while retaining the other stratum.
        if len(field_goals) - target_field_goals >= count - target_field_goals - target_negatives:
            target_field_goals = count - target_negatives
        else:
            target_negatives = count - target_field_goals
    if target_field_goals + target_negatives < count:
        raise ValueError("not enough mapped events for the requested count")

    sampled = [
        *_sample_evenly(field_goals, target_field_goals, rng),
        *_sample_evenly(hard_negatives, target_negatives, rng),
    ]
    sampled_ids = {int(row.get("source_index", -1)) for row in sampled}
    remaining = [
        row
        for row in [*field_goals, *hard_negatives]
        if int(row.get("source_index", -1)) not in sampled_ids
    ]
    sampled.sort(key=lambda row: int(row["video_frame"]))
    chosen = _greedy_spacing(sampled, minimum_center_spacing_frames)
    if len(chosen) < count:
        chosen_ids = {int(row.get("source_index", -1)) for row in chosen}
        remaining.sort(key=lambda row: (int(row["video_frame"]), int(row.get("source_index", -1))))
        for row in remaining:
            if len(chosen) >= count:
                break
            if int(row.get("source_index", -1)) in chosen_ids:
                continue
            frame = int(row["video_frame"])
            if all(abs(frame - int(other["video_frame"])) >= minimum_center_spacing_frames for other in chosen):
                chosen.append(row)
                chosen_ids.add(int(row.get("source_index", -1)))
    if len(chosen) < count:
        raise ValueError("not enough mapped events after spacing constraints")

    chosen.sort(key=lambda row: (int(row["video_frame"]), int(row.get("source_index", -1))))
    result: list[dict[str, Any]] = []
    for ordinal, event in enumerate(chosen[:count], 1):
        anchor = int(event["video_frame"])
        start = anchor - half
        end = start + window_frames - 1
        result.append(
            {
                "event_id": f"pbp-window-{ordinal:03d}",
                "start_frame": start,
                "end_frame": end,
                "anchor_frame": anchor,
                "source_fps": float(fps),
                "selection_kind": "field_goal" if bool(event.get("is_field_goal")) else "hard_negative",
                "source_index": int(event.get("source_index", -1)),
                "period": int(event.get("period", 0)),
                "clock": str(event.get("clock") or ""),
                "alignment_status": str(event.get("alignment_status") or ""),
                "frame_uncertainty_frames": event.get("frame_uncertainty_frames"),
            }
        )
    return result


def _sample_evenly(rows: Sequence[Mapping[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if count >= len(rows):
        return [dict(row) for row in rows]
    result: list[dict[str, Any]] = []
    for index in range(count):
        lower = (index * len(rows)) // count
        upper = ((index + 1) * len(rows)) // count - 1
        result.append(dict(rows[rng.randint(lower, max(lower, upper))]))
    return result


def _greedy_spacing(rows: Sequence[Mapping[str, Any]], spacing: int) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for row in rows:
        frame = int(row["video_frame"])
        if all(abs(frame - int(other["video_frame"])) >= spacing for other in chosen):
            chosen.append(dict(row))
    return chosen
