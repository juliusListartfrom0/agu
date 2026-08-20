"""Permutation-invariant entity relations for dense causal basketball phases."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from app.analysis.causal_shot_phase_review import (
    CAUSAL_PHASE_OFFSETS_SECONDS,
)

DENSE_ENTITY_RELATION_SCHEMA = "agu.causal-phase-entity-relations.v1"
ENTITY_RELATION_FEATURE_NAMES = (
    "player_visible",
    "rim_visible",
    "ball_visible",
    "player_count",
    "player_height_scale",
    "team_count_min",
    "team_count_max",
    "team_count_difference",
    "team_centroid_distance",
    "player_pair_distance_min",
    "player_pair_distance_q25",
    "player_pair_distance_median",
    "player_pair_distance_q75",
    "player_pair_distance_max",
    "player_pair_distance_mean",
    "player_pair_distance_std",
    "within_team_distance_min",
    "within_team_distance_median",
    "within_team_distance_max",
    "between_team_distance_min",
    "between_team_distance_median",
    "between_team_distance_max",
    "player_rim_distance_min",
    "player_rim_distance_q25",
    "player_rim_distance_median",
    "player_rim_distance_q75",
    "player_rim_distance_max",
    "player_rim_distance_mean",
    "player_rim_distance_std",
    "player_rim_abs_x_q25",
    "player_rim_abs_x_median",
    "player_rim_abs_x_q75",
    "player_rim_y_q25",
    "player_rim_y_median",
    "player_rim_y_q75",
    "players_near_rim_count",
    "lane_player_count",
    "team_lane_count_min",
    "team_lane_count_max",
    "team_lane_count_difference",
    "player_ball_distance_min",
    "player_ball_distance_q25",
    "player_ball_distance_median",
    "player_ball_distance_max",
    "player_ball_possession_gap",
    "team_ball_distance_min",
    "team_ball_distance_max",
    "team_ball_distance_gap",
    "ball_rim_abs_x",
    "ball_rim_y",
    "ball_rim_distance",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def extract_entity_relation_frame_features(
    *,
    players: Sequence[Mapping[str, Any]],
    rims: Sequence[Mapping[str, Any]],
    balls: Sequence[Mapping[str, Any]],
    frame_width: int,
    frame_height: int,
) -> list[float]:
    """Summarize individual player/ball/rim relations without ordering leakage."""

    if frame_width < 1 or frame_height < 1:
        raise ValueError("entity relation frame dimensions must be positive")
    normalized_players = [
        _normalized_box(row, frame_width, frame_height) for row in players
    ]
    rim = _best_box(rims, frame_width, frame_height)
    ball = _best_box(balls, frame_width, frame_height)
    if not normalized_players and rim is None and ball is None:
        return [0.0] * len(ENTITY_RELATION_FEATURE_NAMES)

    heights = [row["height"] for row in normalized_players if row["height"] > 0]
    scale = float(np.median(heights)) if heights else 0.0
    distance_scale = max(scale, 1e-6)
    points = [(row["x"], row["y"]) for row in normalized_players]

    teams: dict[str, list[dict[str, float]]] = defaultdict(list)
    for source, normalized in zip(players, normalized_players, strict=True):
        team = str(source.get("team_id") or "")
        if team:
            teams[team].append(normalized)
    team_groups = [values for values in teams.values() if values]
    team_counts = sorted(len(values) for values in team_groups)
    team_centroids = [
        (
            float(np.mean([row["x"] for row in values])),
            float(np.mean([row["y"] for row in values])),
        )
        for values in team_groups
    ]
    centroid_distances = _pairwise_distances(team_centroids, distance_scale)

    pair_distances = _pairwise_distances(points, distance_scale)
    within_distances = [
        distance
        for values in team_groups
        for distance in _pairwise_distances(
            [(row["x"], row["y"]) for row in values],
            distance_scale,
        )
    ]
    between_distances: list[float] = []
    for first_index, first_team in enumerate(team_groups):
        for second_team in team_groups[first_index + 1 :]:
            between_distances.extend(
                _cross_distances(first_team, second_team, distance_scale)
            )

    rim_distances: list[float] = []
    rim_abs_x: list[float] = []
    rim_relative_y: list[float] = []
    lane_flags: list[bool] = []
    team_lane_counts: list[int] = []
    if rim is not None:
        for row in normalized_players:
            dx = (row["x"] - rim["x"]) / distance_scale
            dy = (row["y"] - rim["y"]) / distance_scale
            rim_abs_x.append(abs(dx))
            rim_relative_y.append(dy)
            rim_distances.append(math.hypot(dx, dy))
            lane_flags.append(abs(dx) <= 1.5 and 0.0 <= dy <= 5.0)
        for values in team_groups:
            team_lane_counts.append(
                sum(
                    abs((row["x"] - rim["x"]) / distance_scale) <= 1.5
                    and 0.0
                    <= (row["y"] - rim["y"]) / distance_scale
                    <= 5.0
                    for row in values
                )
            )

    ball_distances: list[float] = []
    team_ball_distances: list[float] = []
    if ball is not None:
        ball_distances = sorted(
            math.hypot(row["x"] - ball["x"], row["y"] - ball["y"])
            / distance_scale
            for row in normalized_players
        )
        team_ball_distances = sorted(
            min(
                math.hypot(row["x"] - ball["x"], row["y"] - ball["y"])
                / distance_scale
                for row in values
            )
            for values in team_groups
        )

    ball_rim_abs_x = ball_rim_y = ball_rim_distance = 0.0
    if ball is not None and rim is not None:
        ball_rim_abs_x = abs(ball["x"] - rim["x"]) / distance_scale
        ball_rim_y = (ball["y"] - rim["y"]) / distance_scale
        ball_rim_distance = math.hypot(ball_rim_abs_x, ball_rim_y)

    values = [
        float(bool(normalized_players)),
        float(rim is not None),
        float(ball is not None),
        float(len(normalized_players)),
        scale,
        float(team_counts[0]) if team_counts else 0.0,
        float(team_counts[-1]) if team_counts else 0.0,
        float(team_counts[-1] - team_counts[0]) if team_counts else 0.0,
        _mean(centroid_distances),
        _quantile(pair_distances, 0.0),
        _quantile(pair_distances, 0.25),
        _quantile(pair_distances, 0.5),
        _quantile(pair_distances, 0.75),
        _quantile(pair_distances, 1.0),
        _mean(pair_distances),
        _std(pair_distances),
        _quantile(within_distances, 0.0),
        _quantile(within_distances, 0.5),
        _quantile(within_distances, 1.0),
        _quantile(between_distances, 0.0),
        _quantile(between_distances, 0.5),
        _quantile(between_distances, 1.0),
        _quantile(rim_distances, 0.0),
        _quantile(rim_distances, 0.25),
        _quantile(rim_distances, 0.5),
        _quantile(rim_distances, 0.75),
        _quantile(rim_distances, 1.0),
        _mean(rim_distances),
        _std(rim_distances),
        _quantile(rim_abs_x, 0.25),
        _quantile(rim_abs_x, 0.5),
        _quantile(rim_abs_x, 0.75),
        _quantile(rim_relative_y, 0.25),
        _quantile(rim_relative_y, 0.5),
        _quantile(rim_relative_y, 0.75),
        float(sum(distance <= 3.0 for distance in rim_distances)),
        float(sum(lane_flags)),
        float(min(team_lane_counts)) if team_lane_counts else 0.0,
        float(max(team_lane_counts)) if team_lane_counts else 0.0,
        float(max(team_lane_counts) - min(team_lane_counts))
        if team_lane_counts
        else 0.0,
        _quantile(ball_distances, 0.0),
        _quantile(ball_distances, 0.25),
        _quantile(ball_distances, 0.5),
        _quantile(ball_distances, 1.0),
        (
            ball_distances[1] - ball_distances[0]
            if len(ball_distances) >= 2
            else 0.0
        ),
        team_ball_distances[0] if team_ball_distances else 0.0,
        team_ball_distances[-1] if team_ball_distances else 0.0,
        (
            team_ball_distances[-1] - team_ball_distances[0]
            if len(team_ball_distances) >= 2
            else 0.0
        ),
        ball_rim_abs_x,
        ball_rim_y,
        ball_rim_distance,
    ]
    if len(values) != len(ENTITY_RELATION_FEATURE_NAMES) or not np.isfinite(
        values
    ).all():
        raise ValueError("entity relation extraction produced invalid features")
    return [float(value) for value in values]


def prepare_entity_relation_perception(
    artifact: Mapping[str, Any],
    *,
    expected_object_type: str,
) -> dict[str, Any]:
    """Index one official perception artifact for repeated event extraction."""

    if artifact.get("schema_version") != "agu.official-perception.v1":
        raise ValueError("unsupported entity relation perception schema")
    detections = artifact.get("detections")
    sampling = artifact.get("sampling")
    if not isinstance(detections, list) or not isinstance(sampling, Mapping):
        raise ValueError("entity relation perception payload is invalid")
    by_frame: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for detection in detections:
        if detection.get("object_type") == expected_object_type:
            by_frame[int(detection["frame"])].append(detection)
    return {"by_frame": by_frame, "sampling": sampling}


def extract_prepared_entity_relation_features(
    *,
    frame_indexes: Sequence[int],
    frame_width: int,
    frame_height: int,
    player_perception: Mapping[str, Any],
    rim_perception: Mapping[str, Any],
    ball_perception: Mapping[str, Any],
) -> list[list[float]]:
    """Extract all 24 relation rows from pre-indexed perception payloads."""

    if len(frame_indexes) != len(CAUSAL_PHASE_OFFSETS_SECONDS):
        raise ValueError("dense entity relations require 24 frame indexes")
    rows = []
    for frame_index in frame_indexes:
        groups = []
        for payload in (player_perception, rim_perception, ball_perception):
            sampled_frame = _nearest_sample_frame(
                payload["sampling"],
                int(frame_index),
            )
            groups.append(
                []
                if sampled_frame is None
                else payload["by_frame"].get(sampled_frame, [])
            )
        rows.append(
            extract_entity_relation_frame_features(
                players=groups[0],
                rims=groups[1],
                balls=groups[2],
                frame_width=frame_width,
                frame_height=frame_height,
            )
        )
    return rows


def seal_dense_entity_relation_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["schema_version"] = DENSE_ENTITY_RELATION_SCHEMA
    artifact["purpose"] = "training_only_dense_entity_relations"
    artifact["runtime_consumable"] = False
    artifact["codex_runtime_answer_used"] = False
    artifact["feature_names"] = list(ENTITY_RELATION_FEATURE_NAMES)
    _validate_artifact(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_dense_entity_relation_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_artifact(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("dense entity relation hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def attach_dense_entity_relation_features(
    examples: Sequence[Mapping[str, Any]],
    *,
    relation_artifact: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    relation = verify_dense_entity_relation_artifact(relation_artifact)
    relation_by_key = {
        _event_key(row): row for row in relation["examples"]
    }
    example_keys = {_event_key(row) for row in examples}
    if not example_keys.issubset(relation_by_key):
        raise ValueError("dense entity relations must cover all temporal examples")
    rows = []
    for row in examples:
        relation_row = relation_by_key[_event_key(row)]
        if relation_row["phase_review_id"] != row["phase_review_id"]:
            raise ValueError("dense entity relation review binding mismatch")
        fused = [
            [
                *[float(value) for value in visual],
                *[float(value) for value in relation_values],
            ]
            for visual, relation_values in zip(
                row["embeddings"],
                relation_row["features"],
                strict=True,
            )
        ]
        rows.append({**row, "embeddings": fused})
    return rows, {
        "entity_relation_artifact_sha256": relation["artifact_sha256"],
        "entity_relation_feature_names": list(ENTITY_RELATION_FEATURE_NAMES),
        "entity_relation_feature_dimension": len(ENTITY_RELATION_FEATURE_NAMES),
        "entity_relation_source_artifact_sha256s": list(
            relation["source_artifact_sha256s"]
        ),
        "entity_relation_example_count": len(relation_by_key),
        "excluded_unresolved_entity_relation_count": (
            len(relation_by_key) - len(example_keys)
        ),
    }


def _normalized_box(
    row: Mapping[str, Any],
    frame_width: int,
    frame_height: int,
) -> dict[str, float]:
    box = row.get("bbox")
    if not isinstance(box, Mapping):
        raise ValueError("entity relation detection bbox is invalid")
    x1 = float(box["x1"]) / frame_width
    y1 = float(box["y1"]) / frame_height
    x2 = float(box["x2"]) / frame_width
    y2 = float(box["y2"]) / frame_height
    values = {
        "x": (x1 + x2) / 2.0,
        "y": (y1 + y2) / 2.0,
        "width": max(0.0, x2 - x1),
        "height": max(0.0, y2 - y1),
        "confidence": float(row.get("confidence", 0.0)),
    }
    if not np.isfinite(list(values.values())).all():
        raise ValueError("entity relation detection values are invalid")
    return values


def _best_box(
    rows: Sequence[Mapping[str, Any]],
    frame_width: int,
    frame_height: int,
) -> dict[str, float] | None:
    if not rows:
        return None
    return _normalized_box(
        max(rows, key=lambda row: float(row.get("confidence", 0.0))),
        frame_width,
        frame_height,
    )


def _nearest_sample_frame(
    sampling: Mapping[str, Any],
    target_frame: int,
) -> int | None:
    stride = int(sampling.get("stride_frames", 0))
    if stride < 1:
        raise ValueError("entity relation sampling stride is invalid")
    windows = sampling.get("windows")
    ranges = (
        windows
        if isinstance(windows, list)
        else [
            {
                "start_frame": sampling.get("start_frame", 0),
                "end_frame": sampling.get("end_frame", -1),
            }
        ]
    )
    candidates = []
    for window in ranges:
        start = int(window["start_frame"])
        end = int(window["end_frame"])
        if start <= target_frame <= end:
            offset = round((target_frame - start) / stride)
            candidates.append(min(end, max(start, start + offset * stride)))
    if not candidates:
        return None
    return min(candidates, key=lambda value: abs(value - target_frame))


def _pairwise_distances(
    points: Sequence[tuple[float, float]],
    scale: float,
) -> list[float]:
    return [
        math.hypot(first[0] - second[0], first[1] - second[1]) / scale
        for first_index, first in enumerate(points)
        for second in points[first_index + 1 :]
    ]


def _cross_distances(
    first: Sequence[Mapping[str, float]],
    second: Sequence[Mapping[str, float]],
    scale: float,
) -> list[float]:
    return [
        math.hypot(left["x"] - right["x"], left["y"] - right["y"]) / scale
        for left in first
        for right in second
    ]


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), fraction))


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _validate_artifact(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != DENSE_ENTITY_RELATION_SCHEMA:
        raise ValueError("unsupported dense entity relation schema")
    if (
        artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or tuple(artifact.get("feature_names") or ())
        != ENTITY_RELATION_FEATURE_NAMES
    ):
        raise ValueError("dense entity relation policy is invalid")
    _require_sha256(artifact.get("review_plan_sha256"), "review plan")
    sources = set(_hash_list(artifact.get("source_video_sha256s"), "source"))
    blind = set(
        _hash_list(
            artifact.get("sealed_blind_video_sha256s"),
            "blind source",
        )
    )
    _hash_list(artifact.get("source_artifact_sha256s"), "source artifact")
    if sources & blind:
        raise ValueError("sealed blind source entered dense entity relations")
    examples = artifact.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("dense entity relation examples are required")
    seen = set()
    observed_sources = set()
    for row in examples:
        key = _event_key(row)
        features = np.asarray(row.get("features"), dtype=np.float64)
        if (
            key in seen
            or not str(row.get("phase_review_id") or "")
            or features.shape
            != (
                len(CAUSAL_PHASE_OFFSETS_SECONDS),
                len(ENTITY_RELATION_FEATURE_NAMES),
            )
            or not np.isfinite(features).all()
        ):
            raise ValueError("invalid or duplicate dense entity relation example")
        seen.add(key)
        observed_sources.add(key[0])
    if observed_sources != sources:
        raise ValueError("dense entity relation source coverage mismatch")


def _event_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    source = _require_sha256(row.get("source_video_sha256"), "source")
    bundle = _require_sha256(
        row.get("candidate_bundle_sha256"),
        "candidate bundle",
    )
    event_id = str(row.get("event_id") or "")
    if not event_id:
        raise ValueError("dense entity relation event ID is required")
    return source, bundle, event_id


def _hash_list(value: object, field: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} hashes are required")
    return [_require_sha256(item, field) for item in value]


def _require_sha256(value: object, field: str) -> str:
    text = str(value or "")
    if not _SHA256.fullmatch(text):
        raise ValueError(f"invalid {field} SHA-256")
    return text


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
