"""Reference-assisted basketball event ledger and reconciliation helpers.

This module treats edited clips and exported statistics as review evidence. It
does not promote them to an independent model benchmark and does not alter the
v3 action-model preprocessing contract.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

ORIGIN_TIME_RE = re.compile(r"^\s*(?P<period>\d+)\s*-\s*(?P<minute>\d+)\s*:\s*(?P<second>\d+)\s*$")

COUNTING_FIELDS = (
    "points",
    "assists",
    "rebounds",
    "offensive_rebounds",
    "defensive_rebounds",
    "steals",
    "blocks",
    "blocked_shots",
    "two_pt_made",
    "two_pt_attempted",
    "three_pt_made",
    "three_pt_attempted",
    "free_throw_made",
    "free_throw_attempted",
    "fouls",
    "turnovers",
)

SUMMARY_COLUMN_MAP = {
    "得分": "points",
    "助攻": "assists",
    "篮板": "rebounds",
    "前场篮板": "offensive_rebounds",
    "后场篮板": "defensive_rebounds",
    "抢断": "steals",
    "盖帽": "blocks",
    "被盖": "blocked_shots",
    "2分球命中": "two_pt_made",
    "2分球出手": "two_pt_attempted",
    "3分球命中": "three_pt_made",
    "3分球出手": "three_pt_attempted",
    "罚球命中": "free_throw_made",
    "罚球出手": "free_throw_attempted",
    "犯规": "fouls",
    "失误": "turnovers",
}


@dataclass(frozen=True)
class NormalizedEvent:
    event_id: str
    source_row: int
    period: int
    elapsed_seconds: int
    team: str
    event_type: str
    primary_player: str
    secondary_player: str | None = None
    shot_value: int | None = None
    outcome: str | None = None
    rebound_type: str | None = None
    derivation: str = "direct"


@dataclass(frozen=True)
class FrameDescriptorIndex:
    descriptors: object
    source_paths: tuple[str, ...]
    source_indices: object
    source_times: object
    sample_fps: float


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read UTF-8-SIG CSV input without changing the reference file."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {key: (value or "").strip() for key, value in row.items() if key is not None}
            for row in csv.DictReader(handle)
        ]


def parse_origin_time(value: str) -> tuple[int, int]:
    match = ORIGIN_TIME_RE.match(value)
    if not match:
        raise ValueError(f"Invalid OriginTime: {value!r}")
    return int(match.group("period")), int(match.group("minute")) * 60 + int(match.group("second"))


def _event(
    *,
    source_row: int,
    suffix: str,
    period: int,
    elapsed_seconds: int,
    team: str,
    event_type: str,
    primary_player: str,
    secondary_player: str | None = None,
    shot_value: int | None = None,
    outcome: str | None = None,
    rebound_type: str | None = None,
    derivation: str = "direct",
) -> NormalizedEvent:
    return NormalizedEvent(
        event_id=f"row_{source_row:04d}_{suffix}",
        source_row=source_row,
        period=period,
        elapsed_seconds=elapsed_seconds,
        team=team,
        event_type=event_type,
        primary_player=primary_player,
        secondary_player=secondary_player or None,
        shot_value=shot_value,
        outcome=outcome,
        rebound_type=rebound_type,
        derivation=derivation,
    )


def normalize_detail_rows(rows: Iterable[Mapping[str, str]]) -> list[NormalizedEvent]:
    """Expand JUSHOOP detail rows into atomic, auditable basketball events."""

    events: list[NormalizedEvent] = []
    for source_row, row in enumerate(rows, start=2):
        event_name = (row.get("Event") or "").strip()
        player = (row.get("Player") or "").strip()
        team = (row.get("Team") or "").strip()
        info = (row.get("Info") or "").strip()
        obj = (row.get("Object") or "").strip()
        origin_time = (row.get("OriginTime") or "").strip()
        if not origin_time or event_name in {"换人", "计时开始"}:
            continue
        period, elapsed_seconds = parse_origin_time(origin_time)

        if event_name in {"2分球出手", "3分球出手", "罚球出手"}:
            shot_value = 1 if event_name == "罚球出手" else int(event_name[0])
            event_type = "free_throw_attempt" if shot_value == 1 else "field_goal_attempt"
            events.append(
                _event(
                    source_row=source_row,
                    suffix="shot",
                    period=period,
                    elapsed_seconds=elapsed_seconds,
                    team=team,
                    event_type=event_type,
                    primary_player=player,
                    shot_value=shot_value,
                    outcome="made" if info == "进球" else "missed",
                )
            )
        elif event_name == "篮板":
            events.append(
                _event(
                    source_row=source_row,
                    suffix="rebound",
                    period=period,
                    elapsed_seconds=elapsed_seconds,
                    team=team,
                    event_type="rebound",
                    primary_player=player,
                    rebound_type="offensive" if info == "前场" else "defensive",
                )
            )
        elif event_name == "助攻":
            shot_value = 3 if info.startswith("3") else 2
            events.append(
                _event(
                    source_row=source_row,
                    suffix="assist",
                    period=period,
                    elapsed_seconds=elapsed_seconds,
                    team=team,
                    event_type="assist",
                    primary_player=player,
                    secondary_player=obj,
                    shot_value=shot_value,
                    outcome="made",
                )
            )
            events.append(
                _event(
                    source_row=source_row,
                    suffix="implied_made_shot",
                    period=period,
                    elapsed_seconds=elapsed_seconds,
                    team=team,
                    event_type="field_goal_attempt",
                    primary_player=obj,
                    secondary_player=player,
                    shot_value=shot_value,
                    outcome="made",
                    derivation="implied_by_assist",
                )
            )
        elif event_name == "盖帽":
            shot_value = 3 if info.startswith("3") else 2
            events.append(
                _event(
                    source_row=source_row,
                    suffix="block",
                    period=period,
                    elapsed_seconds=elapsed_seconds,
                    team=team,
                    event_type="block",
                    primary_player=player,
                    secondary_player=obj,
                    shot_value=shot_value,
                )
            )
            events.append(
                _event(
                    source_row=source_row,
                    suffix="implied_blocked_shot",
                    period=period,
                    elapsed_seconds=elapsed_seconds,
                    team="白队" if team == "黑队" else "黑队",
                    event_type="field_goal_attempt",
                    primary_player=obj,
                    secondary_player=player,
                    shot_value=shot_value,
                    outcome="missed",
                    derivation="implied_by_block",
                )
            )
        elif event_name == "抢断":
            events.append(
                _event(
                    source_row=source_row,
                    suffix="steal",
                    period=period,
                    elapsed_seconds=elapsed_seconds,
                    team=team,
                    event_type="steal",
                    primary_player=player,
                    secondary_player=obj,
                )
            )
            events.append(
                _event(
                    source_row=source_row,
                    suffix="implied_turnover",
                    period=period,
                    elapsed_seconds=elapsed_seconds,
                    team="白队" if team == "黑队" else "黑队",
                    event_type="turnover",
                    primary_player=obj,
                    secondary_player=player,
                    derivation="implied_by_steal",
                )
            )
        elif event_name == "失误":
            events.append(
                _event(
                    source_row=source_row,
                    suffix="turnover",
                    period=period,
                    elapsed_seconds=elapsed_seconds,
                    team=team,
                    event_type="turnover",
                    primary_player=player,
                    outcome=info or None,
                )
            )
        elif event_name == "犯规":
            events.append(
                _event(
                    source_row=source_row,
                    suffix="foul",
                    period=period,
                    elapsed_seconds=elapsed_seconds,
                    team=team,
                    event_type="foul",
                    primary_player=player,
                    secondary_player=obj,
                    outcome=info or None,
                )
            )
            if info == "进攻犯规":
                events.append(
                    _event(
                        source_row=source_row,
                        suffix="implied_turnover",
                        period=period,
                        elapsed_seconds=elapsed_seconds,
                        team=team,
                        event_type="turnover",
                        primary_player=player,
                        derivation="implied_by_offensive_foul",
                    )
                )
    return events


def aggregate_player_stats(events: Iterable[NormalizedEvent]) -> dict[tuple[str, str], dict[str, int]]:
    stats: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {field: 0 for field in COUNTING_FIELDS})
    for event in events:
        if not event.primary_player:
            continue
        row = stats[(event.team, event.primary_player)]
        if event.event_type == "field_goal_attempt":
            prefix = "two_pt" if event.shot_value == 2 else "three_pt"
            row[f"{prefix}_attempted"] += 1
            if event.outcome == "made":
                row[f"{prefix}_made"] += 1
                row["points"] += int(event.shot_value or 0)
            if event.derivation == "implied_by_block":
                row["blocked_shots"] += 1
        elif event.event_type == "free_throw_attempt":
            row["free_throw_attempted"] += 1
            if event.outcome == "made":
                row["free_throw_made"] += 1
                row["points"] += 1
        elif event.event_type == "assist":
            row["assists"] += 1
        elif event.event_type == "rebound":
            row["rebounds"] += 1
            row[f"{event.rebound_type}_rebounds"] += 1
        elif event.event_type == "steal":
            row["steals"] += 1
        elif event.event_type == "block":
            row["blocks"] += 1
        elif event.event_type == "foul":
            row["fouls"] += 1
        elif event.event_type == "turnover":
            row["turnovers"] += 1
    return dict(stats)


def load_summary_stats(paths: Iterable[Path]) -> tuple[dict[tuple[str, str], dict[str, int]], dict[str, dict[str, str]]]:
    expected: dict[tuple[str, str], dict[str, int]] = {}
    source_rows: dict[str, dict[str, str]] = {}
    for path in paths:
        team = path.stem.removesuffix("数据")
        for row in read_csv_rows(path):
            player = row.get("姓名", "")
            if not player or player == team:
                continue
            expected[(team, player)] = {
                target: int(float(row[source])) for source, target in SUMMARY_COLUMN_MAP.items()
            }
            source_rows[f"{team}:{player}"] = dict(row)
    return expected, source_rows


def reconcile_player_stats(
    actual: Mapping[tuple[str, str], Mapping[str, int]],
    expected: Mapping[tuple[str, str], Mapping[str, int]],
) -> dict[str, object]:
    comparisons: list[dict[str, object]] = []
    matched = 0
    for key in sorted(set(actual) | set(expected)):
        team, player = key
        for field in COUNTING_FIELDS:
            actual_value = int(actual.get(key, {}).get(field, 0))
            expected_value = int(expected.get(key, {}).get(field, 0))
            is_match = actual_value == expected_value
            matched += int(is_match)
            comparisons.append(
                {
                    "team": team,
                    "player": player,
                    "field": field,
                    "actual": actual_value,
                    "expected": expected_value,
                    "match": is_match,
                }
            )
    total = len(comparisons)
    return {
        "matched_fields": matched,
        "total_fields": total,
        "field_accuracy": matched / total if total else 0.0,
        "mismatches": [item for item in comparisons if not item["match"]],
        "comparisons": comparisons,
    }


def extract_dhash_descriptors(
    video_path: Path,
    *,
    sample_fps: float,
    ffmpeg_bin: str = "ffmpeg",
) -> object:
    """Extract compact difference hashes through a replaceable FFmpeg adapter."""

    import numpy as np

    width, height = 17, 16
    command = [
        ffmpeg_bin,
        "-v",
        "error",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-vf",
        f"fps={sample_fps},scale={width}:{height}:flags=area,format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_size = width * height
    usable = len(result.stdout) - (len(result.stdout) % frame_size)
    pixels = np.frombuffer(result.stdout[:usable], dtype=np.uint8).reshape(-1, height, width)
    bits = pixels[:, :, 1:] > pixels[:, :, :-1]
    return np.packbits(bits.reshape(len(pixels), -1), axis=1)


def build_raw_frame_index(
    raw_paths: Iterable[Path],
    *,
    sample_fps: float = 4.0,
    ffmpeg_bin: str = "ffmpeg",
) -> FrameDescriptorIndex:
    import numpy as np

    paths = tuple(str(path) for path in raw_paths)
    descriptor_parts = []
    source_indices = []
    source_times = []
    for source_index, raw_path in enumerate(paths):
        descriptors = extract_dhash_descriptors(
            Path(raw_path), sample_fps=sample_fps, ffmpeg_bin=ffmpeg_bin
        )
        descriptor_parts.append(descriptors)
        source_indices.append(np.full(len(descriptors), source_index, dtype=np.int16))
        source_times.append(np.arange(len(descriptors), dtype=np.float32) / sample_fps)
    return FrameDescriptorIndex(
        descriptors=np.concatenate(descriptor_parts, axis=0),
        source_paths=paths,
        source_indices=np.concatenate(source_indices),
        source_times=np.concatenate(source_times),
        sample_fps=sample_fps,
    )


def match_reference_clip(
    reference_path: Path,
    raw_index: FrameDescriptorIndex,
    *,
    reference_fps: float = 1.0,
    max_hamming_distance: float = 56.0,
    minimum_margin: float = 3.0,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    scene_threshold: float = 0.15,
    minimum_scene_seconds: float = 0.5,
) -> dict[str, object]:
    """Locate sampled reference frames in raw video and group linear runs."""

    import cv2

    query = extract_dhash_descriptors(
        reference_path, sample_fps=reference_fps, ffmpeg_bin=ffmpeg_bin
    )
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = matcher.knnMatch(query, raw_index.descriptors, k=2)
    samples: list[dict[str, object]] = []
    for query_index, candidates in enumerate(pairs):
        if len(candidates) < 2:
            continue
        best, second = candidates
        margin = float(second.distance - best.distance)
        accepted = best.distance <= max_hamming_distance and margin >= minimum_margin
        source_index = int(raw_index.source_indices[best.trainIdx])
        samples.append(
            {
                "reference_time": query_index / reference_fps,
                "raw_video": raw_index.source_paths[source_index],
                "raw_time": float(raw_index.source_times[best.trainIdx]),
                "distance": float(best.distance),
                "margin": margin,
                "accepted": accepted,
            }
        )

    scene_boundaries = detect_reference_scene_boundaries(
        reference_path,
        threshold=scene_threshold,
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
        minimum_scene_seconds=minimum_scene_seconds,
    )
    segments = localize_reference_scenes(samples, scene_boundaries)
    accepted_count = sum(bool(sample["accepted"]) for sample in samples)
    localized_count = sum(segment["status"] == "localized" for segment in segments)
    return {
        "reference_video": str(reference_path),
        "sample_count": len(samples),
        "accepted_sample_count": accepted_count,
        "accepted_sample_rate": accepted_count / len(samples) if samples else 0.0,
        "scene_count": len(segments),
        "localized_scene_count": localized_count,
        "localized_scene_rate": localized_count / len(segments) if segments else 0.0,
        "segments": segments,
        "samples": samples,
    }


def detect_reference_scene_boundaries(
    video_path: Path,
    *,
    threshold: float = 0.15,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    minimum_scene_seconds: float = 0.5,
) -> list[float]:
    """Return `[0, cut..., duration]` for hard-cut reference compilations."""

    duration_result = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    duration = float(duration_result.stdout.strip())
    scene_result = subprocess.run(
        [
            ffmpeg_bin,
            "-v",
            "info",
            "-i",
            str(video_path),
            "-vf",
            f"scale=320:-1,select='gt(scene,{threshold})',showinfo",
            "-an",
            "-f",
            "null",
            "-",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    cuts = sorted(
        {
            float(value)
            for value in re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", scene_result.stderr)
            if 0.25 < float(value) < duration - 0.25
        }
    )
    return suppress_short_scene_boundaries(
        [0.0, *cuts, duration], minimum_scene_seconds=minimum_scene_seconds
    )


def suppress_short_scene_boundaries(
    boundaries: Iterable[float],
    *,
    minimum_scene_seconds: float = 0.5,
) -> list[float]:
    """Collapse consecutive transition-frame cuts into one edit boundary."""

    values = list(boundaries)
    if len(values) <= 2:
        return values
    filtered = [values[0]]
    for value in values[1:-1]:
        if value - filtered[-1] >= minimum_scene_seconds:
            filtered.append(value)
    duration = values[-1]
    if duration - filtered[-1] < minimum_scene_seconds and len(filtered) > 1:
        filtered.pop()
    filtered.append(duration)
    return filtered


def localize_reference_scenes(
    samples: Iterable[Mapping[str, object]],
    boundaries: Iterable[float],
    *,
    maximum_offset_error: float = 1.5,
    minimum_samples: int = 2,
    single_sample_max_distance: float = 8.0,
    single_sample_minimum_margin: float = 3.0,
) -> list[dict[str, object]]:
    """Use frame matches to assign every edit scene to one raw-video interval."""

    import statistics

    sample_list = [sample for sample in samples if sample["accepted"]]
    boundary_list = list(boundaries)
    segments: list[dict[str, object]] = []
    for start, end in zip(boundary_list, boundary_list[1:]):
        candidates = [
            sample
            for sample in sample_list
            if start <= float(sample["reference_time"]) < end
        ]
        offset_buckets: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
        for sample in candidates:
            offset = float(sample["raw_time"]) - float(sample["reference_time"])
            offset_buckets[(str(sample["raw_video"]), round(offset))].append(sample)
        dominant = max(
            offset_buckets.values(),
            key=lambda items: (len(items), -min(float(item["distance"]) for item in items)),
            default=[],
        )
        if dominant:
            offsets = [float(item["raw_time"]) - float(item["reference_time"]) for item in dominant]
            median_offset = statistics.median(offsets)
            dominant_video = str(dominant[0]["raw_video"])
            inliers = [
                item
                for item in candidates
                if str(item["raw_video"]) == dominant_video
                if abs(
                    (float(item["raw_time"]) - float(item["reference_time"]))
                    - median_offset
                )
                <= maximum_offset_error
            ]
        else:
            median_offset = 0.0
            inliers = []
        localized = len(inliers) >= minimum_samples
        if not localized and candidates:
            strongest = min(candidates, key=lambda item: float(item["distance"]))
            if (
                float(strongest["distance"]) <= single_sample_max_distance
                and float(strongest["margin"]) >= single_sample_minimum_margin
            ):
                inliers = [strongest]
                median_offset = float(strongest["raw_time"]) - float(strongest["reference_time"])
                localized = True
        segments.append(
            {
                "reference_start": start,
                "reference_end": end,
                "raw_video": str(inliers[0]["raw_video"]) if localized else None,
                "raw_start": start + median_offset if localized else None,
                "raw_end": end + median_offset if localized else None,
                "matched_samples": len(inliers),
                "mean_distance": (
                    sum(float(item["distance"]) for item in inliers) / len(inliers)
                    if inliers
                    else None
                ),
                "mean_margin": (
                    sum(float(item["margin"]) for item in inliers) / len(inliers)
                    if inliers
                    else None
                ),
                "status": "localized" if localized else "needs_review",
            }
        )
    return segments


def group_reference_match_samples(
    samples: Iterable[Mapping[str, object]],
    *,
    maximum_gap_seconds: float = 4.0,
    minimum_samples: int = 2,
) -> list[dict[str, object]]:
    """Merge sparse accepted frame matches into stable source-time runs."""

    runs: list[list[Mapping[str, object]]] = []
    current: list[dict[str, object]] = []
    for sample in (item for item in samples if item["accepted"]):
        if current:
            previous = current[-1]
            ref_step = float(sample["reference_time"]) - float(previous["reference_time"])
            raw_step = float(sample["raw_time"]) - float(previous["raw_time"])
            previous_offset = float(previous["raw_time"]) - float(previous["reference_time"])
            current_offset = float(sample["raw_time"]) - float(sample["reference_time"])
            continuous = (
                sample["raw_video"] == previous["raw_video"]
                and ref_step <= maximum_gap_seconds
                and 0.0 < raw_step <= ref_step + 1.25
                and abs(current_offset - previous_offset) <= 1.25
            )
            if not continuous:
                if len(current) >= minimum_samples:
                    runs.append(current)
                current = []
        current.append(sample)
    if len(current) >= minimum_samples:
        runs.append(current)

    return [
        {
            "reference_start": run[0]["reference_time"],
            "reference_end": run[-1]["reference_time"],
            "raw_video": run[0]["raw_video"],
            "raw_start": run[0]["raw_time"],
            "raw_end": run[-1]["raw_time"],
            "matched_samples": len(run),
            "mean_distance": sum(float(item["distance"]) for item in run) / len(run),
            "mean_margin": sum(float(item["margin"]) for item in run) / len(run),
        }
        for run in runs
    ]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, events: Iterable[NormalizedEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
