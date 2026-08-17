"""Per-frame normalized geometry for dense causal basketball phases."""

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

DENSE_CAUSAL_GEOMETRY_SCHEMA = "agu.causal-phase-dense-geometry.v1"
GEOMETRY_FEATURE_NAMES = (
    "player_sample_available",
    "player_count",
    "player_confidence_mean",
    "player_center_x_mean",
    "player_center_y_mean",
    "player_center_x_std",
    "player_center_y_std",
    "player_area_mean",
    "player_area_max",
    "team_count_min",
    "team_count_max",
    "team_centroid_distance",
    "rim_sample_available",
    "rim_visible",
    "rim_confidence",
    "rim_center_x",
    "rim_center_y",
    "rim_width",
    "rim_height",
    "ball_sample_available",
    "ball_visible",
    "ball_confidence",
    "ball_center_x",
    "ball_center_y",
    "ball_width",
    "ball_height",
    "ball_rim_dx",
    "ball_rim_dy",
    "ball_rim_distance",
    "nearest_player_ball_distance",
    "player_rim_distance_mean",
    "player_rim_distance_std",
    "players_near_rim_count",
    "player_rim_relative_x_mean",
    "player_rim_relative_y_mean",
    "ball_motion_x",
    "ball_motion_y",
    "ball_motion_speed",
    "player_centroid_motion_x",
    "player_centroid_motion_y",
    "player_centroid_motion_speed",
    "rim_motion_x",
    "rim_motion_y",
    "rim_motion_speed",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def extract_dense_geometry_features(
    *,
    frame_indexes: Sequence[int],
    frame_width: int,
    frame_height: int,
    player_artifact: Mapping[str, Any],
    rim_artifact: Mapping[str, Any],
    ball_artifact: Mapping[str, Any],
) -> list[list[float]]:
    """Build normalized per-position layout and adjacent motion features."""

    return extract_prepared_dense_geometry_features(
        frame_indexes=frame_indexes,
        frame_width=frame_width,
        frame_height=frame_height,
        player_perception=prepare_dense_geometry_perception(
            player_artifact,
            expected_object_type="player",
        ),
        rim_perception=prepare_dense_geometry_perception(
            rim_artifact,
            expected_object_type="rim",
        ),
        ball_perception=prepare_dense_geometry_perception(
            ball_artifact,
            expected_object_type="basketball",
        ),
    )


def prepare_dense_geometry_perception(
    artifact: Mapping[str, Any],
    *,
    expected_object_type: str,
) -> dict[str, Any]:
    """Index one verified perception payload once for many event windows."""

    return _prepare_perception(
        artifact,
        expected_object_type=expected_object_type,
    )


def prepare_dense_geometry_ball_tracks(
    artifact: Mapping[str, Any],
    *,
    minimum_visible_points: int = 2,
) -> dict[str, Any]:
    """Index supported visible/interpolated ball-track centers by frame."""

    if (
        artifact.get("schema_version") != "agu.official-perception.v1"
        or minimum_visible_points < 2
    ):
        raise ValueError("unsupported dense geometry ball-track payload")
    tracks = artifact.get("ball_tracks")
    sampling = artifact.get("sampling")
    if not isinstance(tracks, list) or not isinstance(sampling, Mapping):
        raise ValueError("dense geometry ball tracks are invalid")
    windows = sampling.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("dense geometry ball tracks require sampling windows")
    allowed_ranges = [
        (int(window["start_frame"]), int(window["end_frame"]))
        for window in windows
    ]
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for track in tracks:
        points = track.get("points")
        if not isinstance(points, list):
            raise ValueError("dense geometry ball track points are invalid")
        visible_count = sum(
            point.get("visible") is True and point.get("predicted") is False
            for point in points
        )
        if visible_count < minimum_visible_points:
            continue
        for point in points:
            frame = int(point.get("frame", -1))
            if not any(start <= frame < end for start, end in allowed_ranges):
                continue
            center = point.get("center")
            if not isinstance(center, Mapping):
                raise ValueError("dense geometry ball track center is invalid")
            x = float(center["x"])
            y = float(center["y"])
            confidence = float(point.get("confidence", 0.0))
            if not np.isfinite((x, y, confidence)).all():
                raise ValueError("dense geometry ball track values are invalid")
            by_frame[frame].append(
                {
                    "frame": frame,
                    "object_type": "basketball",
                    "confidence": confidence,
                    "bbox": {"x1": x, "y1": y, "x2": x, "y2": y},
                }
            )
    track_sampling = dict(sampling)
    track_sampling["stride_frames"] = 1
    return {"by_frame": by_frame, "sampling": track_sampling}


def extract_prepared_dense_geometry_features(
    *,
    frame_indexes: Sequence[int],
    frame_width: int,
    frame_height: int,
    player_perception: Mapping[str, Any],
    rim_perception: Mapping[str, Any],
    ball_perception: Mapping[str, Any],
) -> list[list[float]]:
    """Build dense geometry from pre-indexed perception artifacts."""

    if (
        len(frame_indexes) != len(CAUSAL_PHASE_OFFSETS_SECONDS)
        or frame_width < 1
        or frame_height < 1
    ):
        raise ValueError("invalid dense geometry event frame contract")
    payloads = {
        "player": player_perception,
        "rim": rim_perception,
        "ball": ball_perception,
    }
    width = float(frame_width)
    height = float(frame_height)
    base_rows: list[list[float]] = []
    motion_state: list[dict[str, tuple[float, float] | None]] = []
    for target_frame in frame_indexes:
        player_frame = _nearest_sample_frame(
            payloads["player"]["sampling"],
            int(target_frame),
        )
        rim_frame = _nearest_sample_frame(
            payloads["rim"]["sampling"],
            int(target_frame),
        )
        ball_frame = _nearest_sample_frame(
            payloads["ball"]["sampling"],
            int(target_frame),
        )
        players = (
            payloads["player"]["by_frame"].get(player_frame, [])
            if player_frame is not None
            else []
        )
        rims = (
            payloads["rim"]["by_frame"].get(rim_frame, [])
            if rim_frame is not None
            else []
        )
        balls = (
            payloads["ball"]["by_frame"].get(ball_frame, [])
            if ball_frame is not None
            else []
        )
        player_geometry = _player_geometry(players, width=width, height=height)
        rim = _best_box(rims, width=width, height=height)
        ball = _best_box(balls, width=width, height=height)
        relative = _relative_geometry(
            player_centers=player_geometry["centers"],
            rim=rim,
            ball=ball,
        )
        base_rows.append(
            [
                float(player_frame is not None),
                float(len(players)),
                player_geometry["confidence_mean"],
                player_geometry["center_x_mean"],
                player_geometry["center_y_mean"],
                player_geometry["center_x_std"],
                player_geometry["center_y_std"],
                player_geometry["area_mean"],
                player_geometry["area_max"],
                player_geometry["team_count_min"],
                player_geometry["team_count_max"],
                player_geometry["team_centroid_distance"],
                float(rim_frame is not None),
                float(rim is not None),
                0.0 if rim is None else rim["confidence"],
                0.0 if rim is None else rim["x"],
                0.0 if rim is None else rim["y"],
                0.0 if rim is None else rim["width"],
                0.0 if rim is None else rim["height"],
                float(ball_frame is not None),
                float(ball is not None),
                0.0 if ball is None else ball["confidence"],
                0.0 if ball is None else ball["x"],
                0.0 if ball is None else ball["y"],
                0.0 if ball is None else ball["width"],
                0.0 if ball is None else ball["height"],
                relative["ball_rim_dx"],
                relative["ball_rim_dy"],
                relative["ball_rim_distance"],
                relative["nearest_player_ball_distance"],
                relative["player_rim_distance_mean"],
                relative["player_rim_distance_std"],
                relative["players_near_rim_count"],
                relative["player_rim_relative_x_mean"],
                relative["player_rim_relative_y_mean"],
            ]
        )
        motion_state.append(
            {
                "ball": None if ball is None else (ball["x"], ball["y"]),
                "player": player_geometry["centroid"],
                "rim": None if rim is None else (rim["x"], rim["y"]),
            }
        )

    rows = []
    for index, base in enumerate(base_rows):
        previous = motion_state[index - 1] if index else {}
        current = motion_state[index]
        motion = []
        for name in ("ball", "player", "rim"):
            prior_point = previous.get(name)
            current_point = current.get(name)
            if prior_point is None or current_point is None:
                dx = dy = 0.0
            else:
                dx = current_point[0] - prior_point[0]
                dy = current_point[1] - prior_point[1]
            motion.extend((dx, dy, math.hypot(dx, dy)))
        values = [*base, *motion]
        if len(values) != len(GEOMETRY_FEATURE_NAMES) or not np.isfinite(
            values
        ).all():
            raise ValueError("dense geometry produced invalid features")
        rows.append(values)
    return rows


def seal_dense_geometry_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["schema_version"] = DENSE_CAUSAL_GEOMETRY_SCHEMA
    artifact["purpose"] = "training_only_dense_causal_geometry"
    artifact["runtime_consumable"] = False
    artifact["codex_runtime_answer_used"] = False
    artifact["feature_names"] = list(GEOMETRY_FEATURE_NAMES)
    _validate_artifact(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_dense_geometry_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_artifact(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("dense causal geometry hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def attach_dense_geometry_features(
    examples: Sequence[Mapping[str, Any]],
    *,
    geometry_artifact: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    geometry = verify_dense_geometry_artifact(geometry_artifact)
    geometry_by_key = {
        _event_key(row): row for row in geometry["examples"]
    }
    example_keys = {_event_key(row) for row in examples}
    if not example_keys.issubset(geometry_by_key):
        raise ValueError(
            "dense geometry must cover all temporal examples"
        )
    rows = []
    for row in examples:
        geometry_row = geometry_by_key[_event_key(row)]
        if str(geometry_row.get("phase_review_id") or "") != str(
            row.get("phase_review_id") or ""
        ):
            raise ValueError("dense geometry review binding mismatch")
        fused = [
            [*[float(value) for value in visual], *geometry_values]
            for visual, geometry_values in zip(
                row["embeddings"],
                geometry_row["features"],
                strict=True,
            )
        ]
        rows.append({**row, "embeddings": fused})
    return rows, {
        "geometry_artifact_sha256": geometry["artifact_sha256"],
        "geometry_feature_names": list(GEOMETRY_FEATURE_NAMES),
        "geometry_feature_dimension": len(GEOMETRY_FEATURE_NAMES),
        "geometry_source_artifact_sha256s": list(
            geometry["source_artifact_sha256s"]
        ),
        "geometry_example_count": len(geometry_by_key),
        "excluded_unresolved_geometry_count": (
            len(geometry_by_key) - len(example_keys)
        ),
    }


def _prepare_perception(
    artifact: Mapping[str, Any],
    *,
    expected_object_type: str,
) -> dict[str, Any]:
    if artifact.get("schema_version") != "agu.official-perception.v1":
        raise ValueError("unsupported dense geometry perception schema")
    detections = artifact.get("detections")
    sampling = artifact.get("sampling")
    if not isinstance(detections, list) or not isinstance(sampling, Mapping):
        raise ValueError("dense geometry perception payload is invalid")
    by_frame: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for detection in detections:
        if detection.get("object_type") == expected_object_type:
            by_frame[int(detection["frame"])].append(detection)
    return {"by_frame": by_frame, "sampling": sampling}


def _nearest_sample_frame(
    sampling: Mapping[str, Any],
    target_frame: int,
) -> int | None:
    stride = int(sampling.get("stride_frames", 0))
    if stride < 1:
        raise ValueError("dense geometry sampling stride is invalid")
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


def _player_geometry(
    detections: Sequence[Mapping[str, Any]],
    *,
    width: float,
    height: float,
) -> dict[str, Any]:
    centers = []
    confidences = []
    areas = []
    teams: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in detections:
        box = _normalized_box(row, width=width, height=height)
        centers.append((box["x"], box["y"]))
        confidences.append(box["confidence"])
        areas.append(box["width"] * box["height"])
        team = str(row.get("team_id") or "")
        if team:
            teams[team].append((box["x"], box["y"]))
    center_array = np.asarray(centers, dtype=np.float64)
    area_array = np.asarray(areas, dtype=np.float64)
    confidence_array = np.asarray(confidences, dtype=np.float64)
    team_values = sorted(len(values) for values in teams.values())
    team_centroids = [
        np.asarray(values, dtype=np.float64).mean(axis=0)
        for values in teams.values()
        if values
    ]
    team_distance = (
        float(np.linalg.norm(team_centroids[0] - team_centroids[1]))
        if len(team_centroids) == 2
        else 0.0
    )
    centroid = (
        None
        if not centers
        else (float(center_array[:, 0].mean()), float(center_array[:, 1].mean()))
    )
    return {
        "centers": centers,
        "centroid": centroid,
        "confidence_mean": _mean(confidence_array),
        "center_x_mean": 0.0 if centroid is None else centroid[0],
        "center_y_mean": 0.0 if centroid is None else centroid[1],
        "center_x_std": _std(center_array[:, 0] if centers else center_array),
        "center_y_std": _std(center_array[:, 1] if centers else center_array),
        "area_mean": _mean(area_array),
        "area_max": _max(area_array),
        "team_count_min": float(team_values[0]) if team_values else 0.0,
        "team_count_max": float(team_values[-1]) if team_values else 0.0,
        "team_centroid_distance": team_distance,
    }


def _best_box(
    detections: Sequence[Mapping[str, Any]],
    *,
    width: float,
    height: float,
) -> dict[str, float] | None:
    if not detections:
        return None
    return _normalized_box(
        max(detections, key=lambda row: float(row.get("confidence", 0.0))),
        width=width,
        height=height,
    )


def _normalized_box(
    row: Mapping[str, Any],
    *,
    width: float,
    height: float,
) -> dict[str, float]:
    bbox = row.get("bbox")
    if not isinstance(bbox, Mapping):
        raise ValueError("dense geometry detection bbox is invalid")
    x1 = float(bbox["x1"]) / width
    y1 = float(bbox["y1"]) / height
    x2 = float(bbox["x2"]) / width
    y2 = float(bbox["y2"]) / height
    values = {
        "x": (x1 + x2) / 2.0,
        "y": (y1 + y2) / 2.0,
        "width": max(0.0, x2 - x1),
        "height": max(0.0, y2 - y1),
        "confidence": float(row.get("confidence", 0.0)),
    }
    if not np.isfinite(list(values.values())).all():
        raise ValueError("dense geometry detection values are invalid")
    return values


def _relative_geometry(
    *,
    player_centers: Sequence[tuple[float, float]],
    rim: Mapping[str, float] | None,
    ball: Mapping[str, float] | None,
) -> dict[str, float]:
    ball_rim_dx = ball_rim_dy = ball_rim_distance = 0.0
    if ball is not None and rim is not None:
        ball_rim_dx = ball["x"] - rim["x"]
        ball_rim_dy = ball["y"] - rim["y"]
        ball_rim_distance = math.hypot(ball_rim_dx, ball_rim_dy)
    nearest_player_ball = 0.0
    if ball is not None and player_centers:
        nearest_player_ball = min(
            math.hypot(x - ball["x"], y - ball["y"])
            for x, y in player_centers
        )
    rim_distances = []
    relative_x = []
    relative_y = []
    if rim is not None:
        for x, y in player_centers:
            relative_x.append(x - rim["x"])
            relative_y.append(y - rim["y"])
            rim_distances.append(math.hypot(x - rim["x"], y - rim["y"]))
    return {
        "ball_rim_dx": ball_rim_dx,
        "ball_rim_dy": ball_rim_dy,
        "ball_rim_distance": ball_rim_distance,
        "nearest_player_ball_distance": nearest_player_ball,
        "player_rim_distance_mean": _mean(rim_distances),
        "player_rim_distance_std": _std(rim_distances),
        "players_near_rim_count": float(
            sum(distance <= 0.25 for distance in rim_distances)
        ),
        "player_rim_relative_x_mean": _mean(relative_x),
        "player_rim_relative_y_mean": _mean(relative_y),
    }


def _validate_artifact(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != DENSE_CAUSAL_GEOMETRY_SCHEMA:
        raise ValueError("unsupported dense causal geometry schema")
    if (
        artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or tuple(artifact.get("feature_names") or ())
        != GEOMETRY_FEATURE_NAMES
        or artifact.get("ball_geometry_source", "detections")
        not in {"detections", "supported_tracks_min_visible_2"}
    ):
        raise ValueError("dense causal geometry policy is invalid")
    _require_sha256(artifact.get("review_plan_sha256"), field="review plan")
    sources = set(_hash_list(artifact.get("source_video_sha256s"), field="source"))
    blind = set(
        _hash_list(
            artifact.get("sealed_blind_video_sha256s"),
            field="blind source",
        )
    )
    _hash_list(
        artifact.get("source_artifact_sha256s"),
        field="source artifact",
    )
    if sources & blind:
        raise ValueError("sealed blind source entered dense geometry")
    examples = artifact.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("dense geometry examples are required")
    seen = set()
    observed_sources = set()
    for row in examples:
        key = _event_key(row)
        values = np.asarray(row.get("features"), dtype=np.float64)
        if (
            key in seen
            or not str(row.get("phase_review_id") or "")
            or values.shape
            != (
                len(CAUSAL_PHASE_OFFSETS_SECONDS),
                len(GEOMETRY_FEATURE_NAMES),
            )
            or not np.isfinite(values).all()
        ):
            raise ValueError("invalid or duplicate dense geometry example")
        seen.add(key)
        observed_sources.add(key[0])
    if observed_sources != sources:
        raise ValueError("dense geometry source coverage mismatch")


def _event_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    source = _require_sha256(row.get("source_video_sha256"), field="source")
    bundle = _require_sha256(
        row.get("candidate_bundle_sha256"),
        field="candidate bundle",
    )
    event_id = str(row.get("event_id") or "")
    if not event_id:
        raise ValueError("dense geometry event ID is required")
    return source, bundle, event_id


def _hash_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} hashes are required")
    hashes = [_require_sha256(item, field=field) for item in value]
    if len(hashes) != len(set(hashes)):
        raise ValueError(f"duplicate {field} hash")
    return hashes


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value or "")
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"invalid {field} SHA-256")
    return text


def _mean(values: Sequence[float] | np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def _std(values: Sequence[float] | np.ndarray) -> float:
    return float(np.std(values)) if len(values) else 0.0


def _max(values: Sequence[float] | np.ndarray) -> float:
    return float(np.max(values)) if len(values) else 0.0


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
