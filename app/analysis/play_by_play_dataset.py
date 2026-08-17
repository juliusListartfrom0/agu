"""Audit the offline basketball slice of the Play by Play dataset."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

AUDIT_SCHEMA = "agu.play-by-play-basketball-audit.v1"
_MD5 = re.compile(r"[0-9a-f]{32}")


def build_play_by_play_basketball_audit(
    annotation_root: str | Path,
    *,
    source_url: str,
    source_archive_md5: str,
    source_archive_bytes: int,
) -> dict[str, Any]:
    """Summarize standardized basketball trajectories without exposing answers."""

    root = Path(annotation_root) / "play-by-play-dataset" / "basketball"
    if not root.is_dir():
        raise ValueError("Play by Play basketball annotation root is missing")
    if not _MD5.fullmatch(source_archive_md5):
        raise ValueError("source archive MD5 must be a lowercase hex digest")
    if source_archive_bytes <= 0:
        raise ValueError("source archive byte count must be positive")

    clip_paths = sorted(
        path for path in root.rglob("*.json") if path.name != "info.json"
    )
    if not clip_paths:
        raise ValueError("Play by Play basketball annotations contain no clips")

    label_counts: Counter[str] = Counter()
    match_ids: set[str] = set()
    frame_lengths: list[int] = []
    player_observation_count = 0
    ball_observation_count = 0
    ball_observation_frame_count = 0
    last_ball_observation_count = 0
    last_ball_zero_placeholder_count = 0
    stopped_ball_observation_count = 0
    coordinate_min = {"x": math.inf, "y": math.inf}
    coordinate_max = {"x": -math.inf, "y": -math.inf}
    field_keys: Counter[str] = Counter()
    for path in clip_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        label = str(payload.get("label") or "")
        frames = payload.get("frames")
        if not label or not isinstance(frames, list) or not frames:
            raise ValueError(f"invalid Play by Play clip: {path}")
        label_counts[label] += 1
        match_ids.add(path.parent.parent.name)
        frame_lengths.append(len(frames))
        for key in payload.get("field", {}):
            field_keys[str(key)] += 1
        for frame in frames:
            if not isinstance(frame, Mapping):
                raise ValueError(f"invalid Play by Play frame: {path}")
            players = _list(frame, "players", path)
            balls = _list(frame, "balls", path)
            last_balls = _list(frame, "last_balls", path)
            stopped_balls = _list(frame, "stopped_balls", path)
            player_observation_count += len(players)
            ball_observation_count += len(balls)
            ball_observation_frame_count += int(bool(balls))
            last_ball_observation_count += len(last_balls)
            stopped_ball_observation_count += len(stopped_balls)
            if last_balls and _is_zero_point(last_balls[0]):
                last_ball_zero_placeholder_count += 1
            for observations in (players, balls, last_balls, stopped_balls):
                for observation in observations:
                    _accumulate_coordinates(
                        observation,
                        coordinate_min=coordinate_min,
                        coordinate_max=coordinate_max,
                        path=path,
                    )

    info_path = root / "info.json"
    split = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    test_match_ids = [str(value) for value in split.get("test_set", [])]
    missing_test_match_ids = sorted(set(test_match_ids) - match_ids)
    artifact: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "purpose": "offline_standardized_ball_player_trajectory_source_screening",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "source_url": source_url,
        "license": "cc-by-nc-4.0",
        "source_archive_md5": source_archive_md5,
        "source_archive_bytes": int(source_archive_bytes),
        "annotation_root": "play-by-play-dataset/basketball",
        "clip_count": len(clip_paths),
        "frame_count": sum(frame_lengths),
        "match_count": len(match_ids),
        "label_counts": dict(sorted(label_counts.items())),
        "test_match_ids": sorted(test_match_ids),
        "missing_test_match_ids": missing_test_match_ids,
        "frame_length": {
            "min": min(frame_lengths),
            "max": max(frame_lengths),
            "mean": sum(frame_lengths) / len(frame_lengths),
        },
        "field_keys": dict(sorted(field_keys.items())),
        "coordinate_range": {
            "x": {"min": coordinate_min["x"], "max": coordinate_max["x"]},
            "y": {"min": coordinate_min["y"], "max": coordinate_max["y"]},
        },
        "player_observation_count": player_observation_count,
        "ball_observation_count": ball_observation_count,
        "ball_observation_frame_count": ball_observation_frame_count,
        "last_ball_observation_count": last_ball_observation_count,
        "last_ball_zero_placeholder_count": last_ball_zero_placeholder_count,
        "stopped_ball_observation_count": stopped_ball_observation_count,
        "media_included": False,
        "decision": (
            "retain_annotation_only_as_noncommercial_standardized_trajectory_auxiliary; "
            "no_raw_video_hand_rim_ground_truth_or_runtime_promotion"
        ),
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_play_by_play_basketball_audit(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    if artifact.get("schema_version") != AUDIT_SCHEMA:
        raise ValueError("unsupported Play by Play basketball audit schema")
    if artifact.get("runtime_consumable") is not False:
        raise ValueError("Play by Play audit cannot be runtime consumable")
    if artifact.get("codex_runtime_answer_used") is not False:
        raise ValueError("Play by Play audit cannot be a runtime answer channel")
    if not claimed or claimed != _canonical_sha256(artifact):
        raise ValueError("Play by Play basketball audit hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _list(frame: Mapping[str, Any], key: str, path: Path) -> list[Mapping[str, Any]]:
    value = frame.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Play by Play frame field {key} is invalid: {path}")
    if not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"Play by Play frame field {key} contains invalid points: {path}")
    return list(value)


def _accumulate_coordinates(
    observation: Mapping[str, Any],
    *,
    coordinate_min: dict[str, float],
    coordinate_max: dict[str, float],
    path: Path,
) -> None:
    for key in ("x", "y"):
        value = observation.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"Play by Play point has invalid {key}: {path}")
        coordinate_min[key] = min(coordinate_min[key], float(value))
        coordinate_max[key] = max(coordinate_max[key], float(value))


def _is_zero_point(point: Mapping[str, Any]) -> bool:
    return float(point.get("x", 1.0)) == 0.0 and float(point.get("y", 1.0)) == 0.0


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
