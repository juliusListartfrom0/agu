"""Audit hash-bound player/ball trajectories from the BasketEvent release."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

AUDIT_SCHEMA = "agu.basketevent-trajectory-audit.v1"


def build_basketevent_trajectory_audit(
    annotation_root: str | Path,
    *,
    source_url: str,
    source_revision: str,
    source_license: str | None,
) -> dict[str, Any]:
    """Summarize local BasketEvent JSON without importing media or runtime output."""

    root = Path(annotation_root)
    if not root.is_dir():
        raise ValueError("BasketEvent annotation root is missing")
    if not source_url or not source_revision:
        raise ValueError("BasketEvent source identity is required")

    clip_paths = sorted(root.rglob("*.json"))
    if not clip_paths:
        raise ValueError("BasketEvent annotation root contains no JSON clips")

    action_types: Counter[str] = Counter()
    subtype_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    trajectory_lengths: list[int] = []
    player_track_count = 0
    player_observation_count = 0
    player_observation_frame_count = 0
    ball_track_count = 0
    missing_ball_track_clip_count = 0
    ball_observation_count = 0
    ball_observation_frame_count = 0
    event_clip_count = 0

    for path in clip_paths:
        payload = _load_clip(path)
        split = path.relative_to(root).parts[0] if path.relative_to(root).parts else "unknown"
        split_counts[split] += 1
        tracks = [
            (key, value)
            for key, value in payload.items()
            if isinstance(key, str) and key.startswith("player_")
        ]
        ball = payload.get("ball")
        ball_trajectory: list[list[float] | None] | None = None
        if isinstance(ball, Mapping):
            ball_trajectory = _trajectory(ball.get("trajectory"), path, "ball")
            ball_track_count += 1
            ball_observation_count += sum(item is not None for item in ball_trajectory)
            ball_observation_frame_count += sum(item is not None for item in ball_trajectory)
        else:
            missing_ball_track_clip_count += 1

        clip_has_event = False
        player_trajectories: list[list[list[float] | None]] = []
        for key, value in tracks:
            if not isinstance(value, Mapping):
                raise ValueError(f"BasketEvent player track is invalid: {path}")
            trajectory = _trajectory(value.get("trajectory"), path, key)
            if ball_trajectory is not None and len(trajectory) != len(ball_trajectory):
                raise ValueError(f"BasketEvent trajectories have different lengths: {path}")
            player_track_count += 1
            player_trajectories.append(trajectory)
            player_observation_count += sum(item is not None for item in trajectory)
            event = value.get("event")
            if event is None:
                continue
            if not isinstance(event, Mapping):
                raise ValueError(f"BasketEvent player event is invalid: {path}")
            action_type = str(event.get("actionType") or "")
            subtype = str(event.get("subType") or "")
            if not action_type:
                raise ValueError(f"BasketEvent event has no actionType: {path}")
            action_types[action_type] += 1
            if subtype:
                subtype_counts[subtype] += 1
            clip_has_event = True
        if not player_trajectories:
            raise ValueError(f"BasketEvent clip has no player tracks: {path}")
        clip_length = len(ball_trajectory) if ball_trajectory is not None else len(player_trajectories[0])
        if any(len(trajectory) != clip_length for trajectory in player_trajectories):
            raise ValueError(f"BasketEvent player trajectories have different lengths: {path}")
        if ball_trajectory is None:
            trajectory_lengths.append(clip_length)
        else:
            trajectory_lengths.append(len(ball_trajectory))
        player_observation_frame_count += sum(
            any(trajectory[index] is not None for trajectory in player_trajectories)
            for index in range(clip_length)
        )
        event_clip_count += int(clip_has_event)

    artifact: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "purpose": "offline_player_ball_trajectory_and_event_semantics_screening",
        "runtime_consumable": False,
        "training_media_eligible": False,
        "codex_runtime_answer_used": False,
        "source_url": source_url,
        "source_revision": source_revision,
        "source_license": source_license,
        "annotation_root": root.name,
        "clip_count": len(clip_paths),
        "game_count": len({path.parent.name for path in clip_paths}),
        "split_counts": dict(sorted(split_counts.items())),
        "event_clip_count": event_clip_count,
        "event_action_type_counts": dict(sorted(action_types.items())),
        "event_subtype_counts": dict(sorted(subtype_counts.items())),
        "player_track_count": player_track_count,
        "player_observation_count": player_observation_count,
        "player_observation_frame_count": player_observation_frame_count,
        "ball_track_count": ball_track_count,
        "missing_ball_track_clip_count": missing_ball_track_clip_count,
        "ball_observation_count": ball_observation_count,
        "ball_observation_frame_count": ball_observation_frame_count,
        "trajectory_length": {
            "min": min(trajectory_lengths),
            "max": max(trajectory_lengths),
            "mean": sum(trajectory_lengths) / len(trajectory_lengths),
        },
        "media_included": False,
        "decision": (
            "retain_bounded_trajectory_annotations_for_offline_structured_auxiliary_only; "
            "no_raw_video_or_media_rights_and_no_runtime_promotion"
        ),
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_basketevent_trajectory_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the sealed BasketEvent audit hash and non-runtime boundary."""

    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    if artifact.get("schema_version") != AUDIT_SCHEMA:
        raise ValueError("unsupported BasketEvent audit schema")
    if artifact.get("runtime_consumable") is not False:
        raise ValueError("BasketEvent audit cannot be runtime consumable")
    if artifact.get("training_media_eligible") is not False:
        raise ValueError("BasketEvent audit cannot be media-training eligible")
    if not claimed or claimed != _canonical_sha256(artifact):
        raise ValueError("BasketEvent audit hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _load_clip(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid BasketEvent JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"BasketEvent clip must be an object: {path}")
    return value


def _trajectory(value: Any, path: Path, key: str) -> list[list[float] | None]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"BasketEvent {key} trajectory is invalid: {path}")
    result: list[list[float] | None] = []
    for item in value:
        if item is None:
            result.append(None)
            continue
        if not isinstance(item, list) or len(item) != 4:
            raise ValueError(f"BasketEvent trajectory box is invalid: {path}")
        box = [float(number) for number in item]
        if not all(math.isfinite(number) for number in box):
            raise ValueError(f"BasketEvent trajectory box is invalid: {path}")
        result.append(box)
    return result


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
