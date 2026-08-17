"""Training-only alignment of official PBP events to a scoreboard video timeline.

This module deliberately keeps the alignment artifact outside the AGU runtime.  It
uses official play-by-play score states to reject OCR outliers and to anchor the
game-clock-to-video-frame interpolation.  The resulting artifact is useful for
offline supervision and audit only; it is not an autonomous prediction source.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from bisect import bisect_right
from statistics import median
from typing import Any, Mapping, Sequence

from app.analysis.shot_overlay_state import verify_scoreboard_timeline_artifact

PBP_ALIGNMENT_SCHEMA = "agu.pbp-event-alignment.v1"
_CLOCK_RE = re.compile(
    r"^PT(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?$",
    re.IGNORECASE,
)
_FRAME_STATUSES = {
    "score_anchor",
    "interpolated",
    "unmapped_before_first_anchor",
    "unmapped_after_last_anchor",
    "unmapped_no_anchor",
}


def game_elapsed_seconds(
    period: int | str,
    clock: str,
    *,
    regulation_period_seconds: int = 12 * 60,
    overtime_period_seconds: int = 5 * 60,
) -> float:
    """Convert an NBA PBP period/clock pair to elapsed game seconds."""

    try:
        period_number = int(period)
    except (TypeError, ValueError) as exc:
        raise ValueError("period must be a positive integer") from exc
    if period_number <= 0:
        raise ValueError("period must be a positive integer")
    if regulation_period_seconds <= 0 or overtime_period_seconds <= 0:
        raise ValueError("period lengths must be positive")
    match = _CLOCK_RE.fullmatch(str(clock).strip())
    if match is None or not (match.group("minutes") or match.group("seconds")):
        raise ValueError(f"invalid NBA game clock: {clock!r}")
    minutes = int(match.group("minutes") or 0)
    seconds = float(match.group("seconds") or 0.0)
    if not math.isfinite(seconds) or seconds < 0.0 or seconds >= 60.0:
        raise ValueError(f"invalid NBA game clock seconds: {clock!r}")
    remaining = minutes * 60.0 + seconds
    period_length = (
        regulation_period_seconds
        if period_number <= 4
        else overtime_period_seconds
    )
    if remaining < 0.0 or remaining > period_length:
        raise ValueError(f"game clock exceeds period length: {clock!r}")
    if period_number <= 4:
        return float((period_number - 1) * regulation_period_seconds) + (
            float(regulation_period_seconds) - remaining
        )
    return float(4 * regulation_period_seconds) + float(
        (period_number - 5) * overtime_period_seconds
    ) + (float(overtime_period_seconds) - remaining)


def interpolate_video_frame(
    game_elapsed: float,
    anchors: Sequence[Mapping[str, Any]],
) -> int | None:
    """Interpolate a frame between bounded score anchors, failing closed outside."""

    result = _interpolate_with_status(game_elapsed, anchors)
    return result[0]


def build_pbp_event_alignment_artifact(
    *,
    play_by_play_rows: Sequence[Mapping[str, Any]],
    scoreboard_timeline: Mapping[str, Any],
    game_id: str,
    home_team_id: str,
    away_team_id: str,
    source_play_by_play_sha256: str,
    video_id: str | None = None,
    sealed_blind_video_sha256s: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a signed, offline-only PBP/event-to-frame training artifact."""

    if not str(game_id).strip():
        raise ValueError("game_id is required")
    if not str(source_play_by_play_sha256).strip():
        raise ValueError("source_play_by_play_sha256 is required")
    home = _normalize_team(home_team_id)
    away = _normalize_team(away_team_id)
    if not home or not away or home == away:
        raise ValueError("home and away team IDs must be distinct")
    timeline = verify_scoreboard_timeline_artifact(scoreboard_timeline)
    raw_video_sha256 = str(timeline["raw_video_sha256"])
    sealed_hashes = {
        str(value).strip().lower()
        for value in sealed_blind_video_sha256s
        if str(value).strip()
    }
    if raw_video_sha256.lower() in sealed_hashes:
        raise ValueError("raw video is a sealed blind video")
    timeline_teams = {_normalize_team(value) for value in timeline["team_ids"]}
    if {home, away} != timeline_teams:
        raise ValueError("PBP teams must exactly match scoreboard timeline teams")

    events = _parse_events(play_by_play_rows, home=home, away=away)
    official_states = {
        tuple(event["score_state"])
        for event in events
        if event["score_state"] is not None
    }
    if not official_states:
        raise ValueError("play-by-play has no usable official score states")
    reads, rejected_not_official, rejected_decrease = _accepted_timeline_reads(
        timeline,
        official_states=official_states,
        home=home,
        away=away,
    )
    state_samples: dict[tuple[int, int], dict[str, Any]] = {}
    for sample in reads:
        state = sample["score_state"]
        state_samples.setdefault(state, sample)
    stride_frames = _timeline_stride_frames(timeline, reads)
    previous_state: tuple[int, int] | None = None
    anchors: list[dict[str, Any]] = []
    state_anchor_by_event_index: dict[int, dict[str, Any]] = {}
    for event in events:
        state = event["score_state"]
        transition = state is not None and state != previous_state
        event["score_transition"] = bool(transition)
        if state is not None:
            previous_state = state
        if not transition or state not in state_samples:
            continue
        sample = state_samples[state]
        anchor = {
            "source_index": event["source_index"],
            "action_id": event["action_id"],
            "period": event["period"],
            "clock": event["clock"],
            "game_elapsed_seconds": event["game_elapsed_seconds"],
            "score_home": state[0],
            "score_away": state[1],
            "video_frame": int(sample["frame"]),
            "frame_uncertainty_frames": stride_frames,
            "scoreboard_confidence": float(sample["confidence"]),
        }
        anchors.append(anchor)
        state_anchor_by_event_index[event["source_index"]] = anchor

    for event in events:
        direct = state_anchor_by_event_index.get(event["source_index"])
        if direct is not None:
            event["video_frame"] = int(direct["video_frame"])
            event["frame_uncertainty_frames"] = int(
                direct["frame_uncertainty_frames"]
            )
            event["alignment_status"] = "score_anchor"
            continue
        frame, status = _interpolate_with_status(
            event["game_elapsed_seconds"],
            anchors,
            source_index=event["source_index"],
        )
        event["video_frame"] = frame
        event["frame_uncertainty_frames"] = stride_frames if frame is not None else None
        event["alignment_status"] = status

    mapped_count = sum(event["video_frame"] is not None for event in events)
    transition_count = sum(bool(event["score_transition"]) for event in events)
    artifact: dict[str, Any] = {
        "schema_version": PBP_ALIGNMENT_SCHEMA,
        "purpose": "offline_training_only_official_pbp_to_video_alignment",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "game_id": str(game_id),
        "video_id": str(video_id) if video_id else None,
        "home_team_id": home,
        "away_team_id": away,
        "raw_video_sha256": raw_video_sha256,
        "source_play_by_play_sha256": str(source_play_by_play_sha256),
        "source_scoreboard_timeline_sha256": str(timeline["artifact_sha256"]),
        "period_lengths_seconds": {
            "regulation": 12 * 60,
            "overtime": 5 * 60,
        },
        "label_policy": {
            "official_pbp_score_states_used_to_reconcile_ocr": True,
            "external_video_distribution": "not_permitted_without_media_rights",
            "blind_video_hash_guard_count": len(sealed_hashes),
        },
        "scoreboard_reconciliation": {
            "timeline_read_count": len(timeline["reads"]),
            "accepted_read_count": len(reads),
            "rejected_not_in_official_pbp_count": rejected_not_official,
            "rejected_score_decrease_count": rejected_decrease,
            "accepted_state_count": len(state_samples),
            "stride_frames": stride_frames,
        },
        "coverage": {
            "event_count": len(events),
            "mapped_event_count": mapped_count,
            "mapped_event_fraction": round(mapped_count / len(events), 6)
            if events
            else 0.0,
            "score_transition_count": transition_count,
            "score_anchor_count": len(anchors),
            "unmapped_event_count": len(events) - mapped_count,
        },
        "anchors": anchors,
        "events": events,
    }
    _validate_pbp_alignment(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_pbp_event_alignment_artifact(
    payload: Mapping[str, Any],
    *,
    sealed_blind_video_sha256s: Sequence[str] = (),
) -> dict[str, Any]:
    """Verify the artifact hash and its offline/blind-data contract."""

    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_pbp_alignment(artifact)
    sealed_hashes = {
        str(value).strip().lower()
        for value in sealed_blind_video_sha256s
        if str(value).strip()
    }
    if str(artifact["raw_video_sha256"]).lower() in sealed_hashes:
        raise ValueError("raw video is a sealed blind video")
    if not claimed or claimed != _canonical_sha256(artifact):
        raise ValueError("PBP alignment artifact hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _parse_events(
    rows: Sequence[Mapping[str, Any]],
    *,
    home: str,
    away: str,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    running_score: tuple[int, int] | None = None
    for source_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("play-by-play rows must be objects")
        period = _positive_int(row.get("period"), "period")
        clock = str(row.get("clock") or "").strip()
        elapsed = game_elapsed_seconds(period, clock)
        state = _forward_filled_score_state(row, running_score)
        if state is not None:
            running_score = state
        action_id = _optional_int(row.get("actionId"))
        if action_id is None:
            action_id = _optional_int(row.get("actionNumber"))
        event = {
            "source_index": source_index,
            "action_id": action_id,
            "period": period,
            "clock": clock,
            "game_elapsed_seconds": round(elapsed, 3),
            "team_id": _normalize_team(row.get("teamTricode"))
            or str(row.get("teamId") or ""),
            "person_id": str(row.get("personId") or ""),
            "player_name": str(row.get("playerName") or ""),
            "action_type": str(row.get("actionType") or ""),
            "sub_type": str(row.get("subType") or ""),
            "description": str(row.get("description") or ""),
            "shot_result": str(row.get("shotResult") or ""),
            "shot_value": _optional_int(row.get("shotValue")),
            "is_field_goal": bool(row.get("isFieldGoal")),
            "score_home": state[0] if state is not None else None,
            "score_away": state[1] if state is not None else None,
            "score_state": state,
            "score_transition": False,
            "video_frame": None,
            "frame_uncertainty_frames": None,
            "alignment_status": "unmapped_no_anchor",
        }
        events.append(event)
    return events


def _forward_filled_score_state(
    row: Mapping[str, Any],
    previous: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """Read cumulative PBP score fields without treating zero placeholders as resets.

    NBA feeds commonly put ``0,0`` on non-scoring actions (misses, rebounds,
    fouls, and turnovers) even after the game has started.  Those values are
    placeholders rather than a scoreboard reset.  Keep the last monotonic
    cumulative state in that case; a genuinely lower non-zero state is handled
    the same way so a malformed row cannot create a backwards score anchor.
    """

    candidate = _score_state(row)
    if candidate is None:
        return previous
    if previous is None:
        return candidate
    if candidate == (0, 0) and previous != (0, 0):
        return previous
    if candidate[0] < previous[0] or candidate[1] < previous[1]:
        return previous
    return candidate


def _accepted_timeline_reads(
    timeline: Mapping[str, Any],
    *,
    official_states: set[tuple[int, int]],
    home: str,
    away: str,
) -> tuple[list[dict[str, Any]], int, int]:
    reads: list[dict[str, Any]] = []
    rejected_not_official = 0
    rejected_decrease = 0
    current: tuple[int, int] | None = None
    for row in sorted(timeline["reads"], key=lambda item: int(item["frame"])):
        scores = row["scores"]
        normalized_scores = {
            _normalize_team(team): int(score) for team, score in scores.items()
        }
        try:
            state = (int(normalized_scores[home]), int(normalized_scores[away]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("scoreboard timeline teams do not match PBP teams") from exc
        if state not in official_states:
            rejected_not_official += 1
            continue
        if current is not None and (
            state[0] < current[0] or state[1] < current[1]
        ):
            rejected_decrease += 1
            continue
        current = state
        reads.append(
            {
                "frame": int(row["frame"]),
                "score_state": state,
                "confidence": float(row["confidence"]),
            }
        )
    return reads, rejected_not_official, rejected_decrease


def _timeline_stride_frames(
    timeline: Mapping[str, Any],
    reads: Sequence[Mapping[str, Any]],
) -> int:
    configured = _optional_int(timeline.get("stride_frames"))
    if configured is not None and configured > 0:
        return configured
    gaps = [
        int(second["frame"]) - int(first["frame"])
        for first, second in zip(reads, reads[1:], strict=False)
        if int(second["frame"]) > int(first["frame"])
    ]
    return max(1, int(round(median(gaps)))) if gaps else 1


def _interpolate_with_status(
    game_elapsed: float,
    anchors: Sequence[Mapping[str, Any]],
    *,
    source_index: int | None = None,
) -> tuple[int | None, str]:
    if not math.isfinite(float(game_elapsed)):
        raise ValueError("game_elapsed must be finite")
    points_by_time: dict[float, list[tuple[int, float]]] = {}
    for anchor in anchors:
        time = float(anchor["game_elapsed_seconds"])
        frame = float(anchor["video_frame"])
        anchor_source_index = _optional_int(anchor.get("source_index"))
        if anchor_source_index is None:
            anchor_source_index = -1
        if not math.isfinite(time) or not math.isfinite(frame):
            raise ValueError("alignment anchors must be finite")
        points_by_time.setdefault(time, []).append((anchor_source_index, frame))
    if not points_by_time:
        return None, "unmapped_no_anchor"
    # When several official score changes share one game-clock value (most
    # commonly consecutive free throws), the left side of an interval should
    # use the last score at that clock and the right side the first score.  A
    # source-row tie break handles fouls/substitutions before or after those
    # scores without inventing a non-monotonic frame sequence.
    point_bounds = sorted(
        (
            time,
            int(min(frame for _, frame in rows)),
            int(max(frame for _, frame in rows)),
            rows,
        )
        for time, rows in points_by_time.items()
    )
    points = [(time, first_frame) for time, first_frame, _, _ in point_bounds]
    if game_elapsed < points[0][0]:
        return None, "unmapped_before_first_anchor"
    if game_elapsed > points[-1][0]:
        return None, "unmapped_after_last_anchor"
    times = [point[0] for point in points]
    right = bisect_right(times, game_elapsed)
    if right == 0:
        return None, "unmapped_before_first_anchor"
    if right == len(points) or points[right - 1][0] == game_elapsed:
        time, first_frame, last_frame, rows = point_bounds[right - 1]
        if source_index is None:
            return first_frame, "score_anchor"
        before = [frame for row_index, frame in rows if row_index <= source_index]
        after = [frame for row_index, frame in rows if row_index > source_index]
        if before:
            return int(max(before)), "score_anchor"
        if after:
            return int(min(after)), "score_anchor"
        return last_frame, "score_anchor"
    left_time, _, left_last_frame, _ = point_bounds[right - 1]
    right_time, right_first_frame, _, _ = point_bounds[right]
    left_frame = left_last_frame
    right_frame = right_first_frame
    if right_time <= left_time:
        return left_frame, "score_anchor"
    ratio = (game_elapsed - left_time) / (right_time - left_time)
    return int(round(left_frame + ratio * (right_frame - left_frame))), "interpolated"


def _score_state(row: Mapping[str, Any]) -> tuple[int, int] | None:
    home = row.get("scoreHome")
    away = row.get("scoreAway")
    if home in (None, "") or away in (None, ""):
        return None
    home_score = _optional_int(home)
    away_score = _optional_int(away)
    if home_score is None or away_score is None or home_score < 0 or away_score < 0:
        raise ValueError("play-by-play score state is invalid")
    return home_score, away_score


def _normalize_team(value: Any) -> str:
    return str(value or "").strip().upper()


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_int(value: Any, name: str) -> int:
    parsed = _optional_int(value)
    if parsed is None or parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _validate_pbp_alignment(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != PBP_ALIGNMENT_SCHEMA:
        raise ValueError("unsupported PBP alignment schema")
    if (
        artifact.get("purpose") != "offline_training_only_official_pbp_to_video_alignment"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("PBP alignment must remain offline and training-only")
    for key in (
        "game_id",
        "raw_video_sha256",
        "source_play_by_play_sha256",
        "source_scoreboard_timeline_sha256",
    ):
        if not str(artifact.get(key) or "").strip():
            raise ValueError(f"PBP alignment requires {key}")
    events = artifact.get("events")
    anchors = artifact.get("anchors")
    coverage = artifact.get("coverage")
    if not isinstance(events, list) or not isinstance(anchors, list):
        raise ValueError("PBP alignment events and anchors must be lists")
    if not isinstance(coverage, Mapping):
        raise ValueError("PBP alignment coverage is invalid")
    if int(coverage.get("event_count", -1)) != len(events):
        raise ValueError("PBP alignment event count mismatch")
    if int(coverage.get("score_anchor_count", -1)) != len(anchors):
        raise ValueError("PBP alignment anchor count mismatch")
    seen_indices: set[int] = set()
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("PBP alignment events must be objects")
        source_index = int(event.get("source_index", -1))
        if source_index < 0 or source_index in seen_indices:
            raise ValueError("PBP alignment event source indices must be unique")
        seen_indices.add(source_index)
        frame = event.get("video_frame")
        if frame is not None and int(frame) < 0:
            raise ValueError("PBP alignment video frame is invalid")
        if str(event.get("alignment_status")) not in _FRAME_STATUSES:
            raise ValueError("PBP alignment frame status is invalid")
    previous_elapsed = -math.inf
    previous_frame = -1
    for anchor in anchors:
        if not isinstance(anchor, Mapping):
            raise ValueError("PBP alignment anchors must be objects")
        elapsed = float(anchor.get("game_elapsed_seconds", math.nan))
        frame = int(anchor.get("video_frame", -1))
        if not math.isfinite(elapsed) or frame < 0:
            raise ValueError("PBP alignment anchor is invalid")
        if elapsed < previous_elapsed or frame < previous_frame:
            raise ValueError("PBP alignment anchors must be monotonic")
        previous_elapsed = elapsed
        previous_frame = frame


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
