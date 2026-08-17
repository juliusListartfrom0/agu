"""Training-only PBP alignment using broadcast game-clock OCR reads."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from app.analysis.broadcast_clock import verify_broadcast_clock_artifact
from app.analysis.pbp_event_alignment import _parse_events

PBP_CLOCK_ALIGNMENT_SCHEMA = "agu.pbp-clock-event-alignment.v1"
_CLOCK_RE = re.compile(
    r"^PT(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?$",
    re.IGNORECASE,
)
_STATUSES = {"clock_read", "unmapped_no_clock_read"}


def build_pbp_clock_event_alignment_artifact(
    *,
    play_by_play_rows: Sequence[Mapping[str, Any]],
    clock_artifact: Mapping[str, Any],
    game_id: str,
    home_team_id: str,
    away_team_id: str,
    source_play_by_play_sha256: str,
    video_id: str | None = None,
    maximum_clock_delta_seconds: int = 3,
    sealed_blind_video_sha256s: Sequence[str] = (),
) -> dict[str, Any]:
    """Bind each PBP row to the nearest same-period OCR clock read."""

    if not str(game_id).strip() or not str(source_play_by_play_sha256).strip():
        raise ValueError("game_id and source_play_by_play_sha256 are required")
    if maximum_clock_delta_seconds < 0:
        raise ValueError("maximum_clock_delta_seconds must be non-negative")
    clock = verify_broadcast_clock_artifact(clock_artifact)
    raw_video_sha = str(clock["raw_video_sha256"])
    blind = {str(value).strip().lower() for value in sealed_blind_video_sha256s if str(value).strip()}
    if raw_video_sha.lower() in blind:
        raise ValueError("sealed blind video cannot be used for PBP training")
    events = _parse_events(
        play_by_play_rows,
        home=str(home_team_id).strip().upper(),
        away=str(away_team_id).strip().upper(),
    )
    reads = _clock_reads(clock)
    for event in events:
        target = _clock_seconds(str(event["clock"]))
        candidates = [
            row
            for row in reads
            if row["period"] == int(event["period"])
            and abs(float(row["clock_seconds"]) - target) <= maximum_clock_delta_seconds
        ]
        if not candidates:
            event["video_frame"] = None
            event["frame_uncertainty_frames"] = None
            event["alignment_status"] = "unmapped_no_clock_read"
            event["matched_clock_seconds"] = None
            event["clock_delta_seconds"] = None
            continue
        matched = min(
            candidates,
            key=lambda row: (
                abs(float(row["clock_seconds"]) - target),
                int(row["frame"]),
            ),
        )
        delta = abs(float(matched["clock_seconds"]) - target)
        event["video_frame"] = int(matched["frame"])
        event["frame_uncertainty_frames"] = None
        event["alignment_status"] = "clock_read"
        event["matched_clock_seconds"] = int(matched["clock_seconds"])
        event["clock_delta_seconds"] = round(delta, 3)

    mapped = sum(event["video_frame"] is not None for event in events)
    artifact: dict[str, Any] = {
        "schema_version": PBP_CLOCK_ALIGNMENT_SCHEMA,
        "purpose": "offline_training_only_official_pbp_to_broadcast_clock_alignment",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "game_id": str(game_id),
        "video_id": str(video_id) if video_id else None,
        "home_team_id": str(home_team_id).strip().upper(),
        "away_team_id": str(away_team_id).strip().upper(),
        "raw_video_sha256": raw_video_sha,
        "source_play_by_play_sha256": str(source_play_by_play_sha256),
        "source_clock_artifact_sha256": str(clock["artifact_sha256"]),
        "maximum_clock_delta_seconds": maximum_clock_delta_seconds,
        "label_policy": {
            "official_pbp_used_for_training_alignment_only": True,
            "external_video_distribution": "not_permitted_without_media_rights",
            "blind_video_hash_guard_count": len(blind),
        },
        "coverage": {
            "event_count": len(events),
            "mapped_event_count": mapped,
            "mapped_event_fraction": round(mapped / len(events), 6) if events else 0.0,
            "unmapped_event_count": len(events) - mapped,
            "clock_read_count": len(reads),
        },
        "events": events,
    }
    _validate(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_pbp_clock_event_alignment_artifact(
    payload: Mapping[str, Any],
    *,
    sealed_blind_video_sha256s: Sequence[str] = (),
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate(artifact)
    blind = {str(value).strip().lower() for value in sealed_blind_video_sha256s if str(value).strip()}
    if str(artifact["raw_video_sha256"]).lower() in blind:
        raise ValueError("sealed blind video cannot be used for PBP training")
    if not claimed or claimed != _canonical_sha256(artifact):
        raise ValueError("PBP clock alignment artifact hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _clock_reads(clock: Mapping[str, Any]) -> list[dict[str, Any]]:
    reads: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for event in clock["events"]:
        for sample in event["samples"]:
            read = sample.get("read")
            if not isinstance(read, Mapping):
                continue
            period = int(read["period"])
            seconds = int(read["clock_seconds"])
            frame = int(read["frame"])
            key = (period, seconds, frame)
            if key in seen:
                continue
            seen.add(key)
            reads.append(
                {
                    "period": period,
                    "clock_seconds": seconds,
                    "frame": frame,
                    "confidence": float(read["confidence"]),
                }
            )
    reads.sort(key=lambda row: (int(row["period"]), -int(row["clock_seconds"]), int(row["frame"])))
    return reads


def _clock_seconds(clock: str) -> float:
    match = _CLOCK_RE.fullmatch(clock.strip())
    if match is None or not (match.group("minutes") or match.group("seconds")):
        raise ValueError(f"invalid NBA game clock: {clock!r}")
    minutes = int(match.group("minutes") or 0)
    seconds = float(match.group("seconds") or 0.0)
    if not math.isfinite(seconds) or not 0.0 <= seconds < 60.0:
        raise ValueError(f"invalid NBA game clock: {clock!r}")
    return float(minutes * 60) + seconds


def _validate(artifact: Mapping[str, Any]) -> None:
    if (
        artifact.get("schema_version") != PBP_CLOCK_ALIGNMENT_SCHEMA
        or artifact.get("purpose")
        != "offline_training_only_official_pbp_to_broadcast_clock_alignment"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("PBP clock alignment must remain offline and training-only")
    for key in (
        "game_id",
        "raw_video_sha256",
        "source_play_by_play_sha256",
        "source_clock_artifact_sha256",
    ):
        if not str(artifact.get(key) or "").strip():
            raise ValueError(f"PBP clock alignment requires {key}")
    events = artifact.get("events")
    coverage = artifact.get("coverage")
    if not isinstance(events, list) or not isinstance(coverage, Mapping):
        raise ValueError("PBP clock alignment events and coverage are invalid")
    if int(coverage.get("event_count", -1)) != len(events):
        raise ValueError("PBP clock alignment event count mismatch")
    seen: set[int] = set()
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("PBP clock alignment events must be objects")
        source_index = int(event.get("source_index", -1))
        if source_index < 0 or source_index in seen:
            raise ValueError("PBP clock alignment source indices must be unique")
        seen.add(source_index)
        if str(event.get("alignment_status")) not in _STATUSES:
            raise ValueError("PBP clock alignment status is invalid")
        frame = event.get("video_frame")
        if frame is not None and int(frame) < 0:
            raise ValueError("PBP clock alignment video frame is invalid")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
