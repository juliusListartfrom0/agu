"""Deterministic raw-evidence sampling windows for training-only shot screening."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from app.analysis.schemas import GameEventResponse

SHOT_SAMPLING_PROTOCOL = "agu.raw-evidence-atomic-shot-window.v1"


@dataclass(frozen=True)
class ShotSamplingWindow:
    start_frame: int
    end_frame: int
    anchor_frame: int
    anchor_source: str
    protocol: str = SHOT_SAMPLING_PROTOCOL


def validate_sampling_window_payload(payload: Mapping[str, object]) -> None:
    """Validate persisted window provenance before accepting an embedding row."""

    if payload.get("protocol") != SHOT_SAMPLING_PROTOCOL:
        raise ValueError("unsupported shot sampling protocol")
    try:
        start_frame = int(payload["start_frame"])
        end_frame = int(payload["end_frame"])
        anchor_frame = int(payload["anchor_frame"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("shot sampling window requires integer frames") from exc
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("invalid shot sampling window bounds")
    if not start_frame <= anchor_frame <= end_frame:
        raise ValueError("shot sampling anchor is outside its window")
    if not str(payload.get("anchor_source") or ""):
        raise ValueError("shot sampling window requires an anchor source")


def video_frame_indexes(
    window: ShotSamplingWindow,
    *,
    clip_frames: int,
) -> tuple[int, ...]:
    """Uniformly sample only inside a previously resolved evidence window."""

    if clip_frames < 2:
        raise ValueError("video clip requires at least two frames")
    span = window.end_frame - window.start_frame
    return tuple(
        int(round(window.start_frame + span * index / (clip_frames - 1)))
        for index in range(clip_frames)
    )


def resolve_shot_sampling_window(
    event: GameEventResponse,
    *,
    fps: float,
    pre_anchor_seconds: float = 1.5,
    post_anchor_seconds: float = 2.5,
    scoreboard_lookback_seconds: float = 12.0,
) -> ShotSamplingWindow:
    """Resolve a bounded window without consulting a label or reference answer."""

    if event.event_type != "field_goal_attempt":
        raise ValueError("shot sampling requires a field_goal_attempt")
    if fps <= 0:
        raise ValueError("shot sampling fps must be positive")
    if pre_anchor_seconds < 0 or post_anchor_seconds < 0:
        raise ValueError("shot sampling anchor offsets must be non-negative")
    if scoreboard_lookback_seconds <= 0:
        raise ValueError("scoreboard lookback must be positive")

    anchor_frame: int | None = event.release_frame
    anchor_source = "release_frame"
    if anchor_frame is None:
        anchor_frame = _candidate_event_frame(event)
        anchor_source = "candidate_event_frame"
    if anchor_frame is not None:
        if not event.start_frame <= anchor_frame <= event.end_frame:
            raise ValueError("shot sampling anchor is outside the event window")
        return ShotSamplingWindow(
            start_frame=max(
                event.start_frame,
                anchor_frame - int(round(pre_anchor_seconds * fps)),
            ),
            end_frame=min(
                event.end_frame,
                anchor_frame + int(round(post_anchor_seconds * fps)),
            ),
            anchor_frame=anchor_frame,
            anchor_source=anchor_source,
        )

    scoreboard_present = any(
        evidence.kind == "scoreboard_delta" for evidence in event.evidence
    )
    maximum_unanchored_frames = int(round(scoreboard_lookback_seconds * fps))
    if scoreboard_present:
        return ShotSamplingWindow(
            start_frame=max(
                event.start_frame,
                event.end_frame - maximum_unanchored_frames,
            ),
            end_frame=event.end_frame,
            anchor_frame=event.end_frame,
            anchor_source="scoreboard_after_frame",
        )
    if event.end_frame - event.start_frame <= maximum_unanchored_frames:
        return ShotSamplingWindow(
            start_frame=event.start_frame,
            end_frame=event.end_frame,
            anchor_frame=event.end_frame,
            anchor_source="bounded_event",
        )
    raise ValueError("long event has no reliable sampling anchor")


def _candidate_event_frame(event: GameEventResponse) -> int | None:
    for evidence in event.evidence:
        details = evidence.details
        if not isinstance(details, Mapping):
            continue
        value = details.get("candidate_event_frame")
        if value is None or isinstance(value, bool):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None
