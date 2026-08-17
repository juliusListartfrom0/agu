"""Deterministic, offline-only APIDIS video/annotation time alignment helpers.

APIDIS stores manual ball centres in local ``HHMMSS.FFF`` camera time while
the pseudo-synchronised videos and whole-game event XML use UTC epoch seconds.
This module keeps that conversion explicit and deliberately has no runtime
detector/VLM dependencies.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

_EPOCH_RE = re.compile(r"^\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*s?\s*$", re.IGNORECASE)
_LOCAL_TIME_RE = re.compile(r"^\s*(\d{2})(\d{2})(\d{2})(?:\.(\d+))?\s*$")


def parse_apidis_epoch_seconds(value: str | float | int) -> float:
    """Parse APIDIS epoch strings such as ``1,207,759,620.5s``."""

    if isinstance(value, bool):
        raise ValueError("APIDIS epoch must be numeric")
    if isinstance(value, (float, int)):
        result = float(value)
    else:
        match = _EPOCH_RE.fullmatch(str(value))
        if match is None:
            raise ValueError(f"invalid APIDIS epoch: {value!r}")
        result = float(match.group(1).replace(",", ""))
    if not math.isfinite(result):
        raise ValueError(f"invalid APIDIS epoch: {value!r}")
    return result


def _local_seconds(value: str) -> float:
    match = _LOCAL_TIME_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid APIDIS local timestamp: {value!r}")
    hour, minute, second = (int(match.group(index)) for index in (1, 2, 3))
    if hour > 23 or minute > 59 or second > 59:
        raise ValueError(f"invalid APIDIS local timestamp: {value!r}")
    fraction = float(f"0.{match.group(4)}") if match.group(4) else 0.0
    return hour * 3600.0 + minute * 60.0 + second + fraction


def frame_index_for_local_timestamp(
    timestamp: str,
    clip_start_timestamp: str,
    *,
    fps: float,
    frame_count: int,
) -> int | None:
    """Map a local camera timestamp to a nearest frame, or ``None`` if outside.

    The APIDIS ball files use local ``+02`` time; the pseudo-synchronised file
    names identify the same instant as UTC ``Z``.  The two values therefore
    share a clock-of-day after the fixed timezone conversion already encoded by
    the caller's matching ``184700``/``164700`` pair.  A one-day wrap is
    handled defensively for clips around midnight.
    """

    if not math.isfinite(float(fps)) or float(fps) <= 0.0:
        raise ValueError("fps must be positive")
    if int(frame_count) != frame_count or frame_count <= 0:
        raise ValueError("frame_count must be a positive integer")
    delta = _local_seconds(timestamp) - _local_seconds(clip_start_timestamp)
    if delta < -43200.0:
        delta += 86400.0
    elif delta > 43200.0:
        delta -= 86400.0
    if delta < 0.0 or delta >= float(frame_count) / float(fps):
        return None
    frame = int(math.floor(delta * float(fps) + 0.5))
    return frame if 0 <= frame < frame_count else None


def parse_ball_position_rows(path: Path) -> list[dict[str, Any]]:
    """Parse manual APIDIS ball centres without silently dropping bad rows."""

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.lower().startswith("timestamp"):
            continue
        fields = stripped.split()
        if len(fields) < 3:
            raise ValueError(f"invalid ball annotation at {path}:{line_number}")
        try:
            timestamp = str(fields[0])
            _local_seconds(timestamp)
            x, y = (float(fields[index]) for index in (1, 2))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid ball annotation at {path}:{line_number}") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"invalid ball annotation at {path}:{line_number}")
        rows.append({"timestamp": timestamp, "x": x, "y": y})
    return rows


def _tag_name(element: ET.Element) -> str:
    return str(element.tag).rsplit("}", 1)[-1]


def events_in_clip(
    event_paths: Iterable[Path],
    *,
    clip_start_epoch: float,
    clip_duration_seconds: float,
    fps: float,
    frame_count: int,
) -> list[dict[str, Any]]:
    """Extract nested APIDIS action types whose event timestamp is in a clip."""

    start = parse_apidis_epoch_seconds(clip_start_epoch)
    duration = float(clip_duration_seconds)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("clip duration must be positive")
    if not math.isfinite(float(fps)) or float(fps) <= 0.0:
        raise ValueError("fps must be positive")
    if int(frame_count) != frame_count or frame_count <= 0:
        raise ValueError("frame_count must be a positive integer")

    rows: list[dict[str, Any]] = []
    for path in sorted((Path(value) for value in event_paths), key=lambda value: str(value)):
        root = ET.parse(path).getroot()
        for source_index, element in enumerate(root.iter()):
            if _tag_name(element) != "Clock-event":
                continue
            timestamp_value = element.attrib.get("Timestamp") or element.attrib.get("Start-time")
            if not timestamp_value:
                continue
            timestamp = parse_apidis_epoch_seconds(timestamp_value)
            if timestamp < start or timestamp >= start + duration:
                continue
            frame = int(math.floor((timestamp - start) * float(fps) + 0.5))
            if not 0 <= frame < frame_count:
                continue
            action_types = sorted(
                {
                    _tag_name(child)
                    for child in list(element)
                    if _tag_name(child) not in {"Clock-event", "Ball-possession-period"}
                }
            )
            rows.append(
                {
                    "source_file": path.name,
                    "source_index": source_index,
                    "timestamp_epoch": timestamp,
                    "frame": frame,
                    "action_types": action_types,
                    "attributes": {
                        key: str(element.attrib[key])
                        for key in sorted(element.attrib)
                        if key not in {"Timestamp", "Start-time", "End-time"}
                    },
                }
            )
    rows.sort(key=lambda row: (float(row["timestamp_epoch"]), str(row["source_file"]), int(row["source_index"])))
    return rows


def summarize_event_types(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Count nested event action tags in a deterministic order."""

    counts: Counter[str] = Counter()
    for row in rows:
        values = row.get("action_types", [])
        if not isinstance(values, list):
            raise ValueError("event action_types must be a list")
        counts.update(str(value) for value in values)
    return dict(sorted(counts.items()))
