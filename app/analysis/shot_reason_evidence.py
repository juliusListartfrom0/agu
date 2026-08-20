"""Training-only continuous evidence for shot-reason screening."""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from app.analysis.shot_validity_scene_state import verify_scene_embedding_artifact

REASON_EVIDENCE_SCHEMA = "agu.shot-reason-evidence.v2"
FEATURE_NAMES = (
    "sampled_frame_count",
    "player_count_mean",
    "player_count_std",
    "player_count_min",
    "player_count_max",
    "bbox_area_mean",
    "bbox_area_std",
    "bbox_area_p90",
    "bbox_area_max",
    "large_player_fraction",
    "center_x_spread_mean",
    "center_y_spread_mean",
    "raised_wrist_fraction",
    "keypoint_completeness_mean",
    "frame_centroid_shift_mean",
    "frame_centroid_shift_max",
    "player_count_delta_mean",
    "player_count_delta_max",
    "player_pair_distance_mean",
    "player_pair_distance_p10",
    "vertical_alignment_fraction",
    "horizontal_alignment_fraction",
    "formation_anisotropy_mean",
    "nearest_player_motion_mean",
    "ball_rim_hit_count",
    "min_ball_rim_distance",
    "approach_point_count",
    "approach_rise_height_ratio",
    "ball_rim_distance_range",
    "ball_rim_distance_last_minus_first",
    "ball_path_displacement",
    "candidate_observation_count",
    "candidate_player_count",
    "min_wrist_ball_distance",
    "wrist_ball_distance_range",
    "wrist_ball_distance_last_minus_first",
    "max_stable_control_score",
    "max_stable_contact_count",
    "candidate_team_count",
)
_FORBIDDEN_LABEL_FIELDS = {
    "event_present",
    "label",
    "ground_truth",
    "target",
    "review_note",
    "review_notes",
}


def extract_reason_evidence_features(
    *,
    detections: Sequence[Mapping[str, Any]],
    start_frame: int,
    end_frame: int,
    frame_width: int | float,
    frame_height: int | float,
    event: Mapping[str, Any],
) -> dict[str, float]:
    """Aggregate pose/formation and causal candidate evidence inside one window."""
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("reason evidence frame window is invalid")
    width = float(frame_width)
    height = float(frame_height)
    if width <= 0 or height <= 0:
        raise ValueError("reason evidence frame dimensions must be positive")

    by_frame: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for detection in detections:
        frame = int(detection.get("frame", -1))
        if (
            start_frame <= frame <= end_frame
            and detection.get("object_type") == "player"
        ):
            by_frame[frame].append(detection)

    counts: list[float] = []
    areas: list[float] = []
    large_players: list[float] = []
    x_spreads: list[float] = []
    y_spreads: list[float] = []
    raised_wrists: list[float] = []
    keypoint_completeness: list[float] = []
    centroids: list[tuple[float, float]] = []
    frame_centers: list[list[tuple[float, float]]] = []
    pair_distances: list[float] = []
    vertical_alignments: list[float] = []
    horizontal_alignments: list[float] = []
    formation_anisotropies: list[float] = []
    for frame in sorted(by_frame):
        players = by_frame[frame]
        counts.append(float(len(players)))
        centers_x: list[float] = []
        centers_y: list[float] = []
        for player in players:
            bbox = player.get("bbox")
            if not isinstance(bbox, Mapping):
                continue
            x1, y1 = float(bbox["x1"]), float(bbox["y1"])
            x2, y2 = float(bbox["x2"]), float(bbox["y2"])
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / (width * height)
            areas.append(area)
            large_players.append(float(area > 0.1))
            centers_x.append((x1 + x2) / (2.0 * width))
            centers_y.append((y1 + y2) / (2.0 * height))
            keypoints = player.get("keypoints")
            keypoints = keypoints if isinstance(keypoints, Mapping) else {}
            keypoint_completeness.append(min(1.0, len(keypoints) / 17.0))
            raised_wrists.append(float(_has_raised_wrist(keypoints)))
        x_spreads.append(float(np.std(centers_x)) if len(centers_x) > 1 else 0.0)
        y_spreads.append(float(np.std(centers_y)) if len(centers_y) > 1 else 0.0)
        centroids.append(
            (
                float(np.mean(centers_x)) if centers_x else 0.0,
                float(np.mean(centers_y)) if centers_y else 0.0,
            )
        )
        centers = list(zip(centers_x, centers_y, strict=True))
        frame_centers.append(centers)
        for index, first in enumerate(centers):
            for second in centers[index + 1 :]:
                dx = abs(first[0] - second[0])
                dy = abs(first[1] - second[1])
                pair_distances.append(math.hypot(dx, dy))
                vertical_alignments.append(float(dx <= 0.08 and dy >= 0.05))
                horizontal_alignments.append(float(dy <= 0.08 and dx >= 0.05))
        formation_anisotropies.append(_formation_anisotropy(centers))

    centroid_shifts = [
        math.dist(first, second)
        for first, second in zip(centroids, centroids[1:], strict=False)
    ]
    count_deltas = [
        abs(first - second)
        for first, second in zip(counts, counts[1:], strict=False)
    ]
    nearest_player_motions = [
        min(math.dist(current, previous) for previous in first)
        for first, second in zip(frame_centers, frame_centers[1:], strict=False)
        if first and second
        for current in second
    ]
    details = _ball_rim_details(event)
    observations = details.get("candidate_player_observations")
    observations = observations if isinstance(observations, list) else []
    wrist_distances = _finite_values(observations, "wrist_ball_distance")
    wrist_distance_series = _longest_player_series(
        observations, "wrist_ball_distance"
    )
    control_scores = _finite_values(observations, "stable_control_score")
    contact_counts = _finite_values(observations, "stable_contact_count")
    review_observations = details.get("review_rim_observations")
    review_observations = (
        review_observations if isinstance(review_observations, list) else []
    )
    rim_distance_series, ball_center_series = _ball_rim_series(
        review_observations,
        frame_width=width,
        frame_height=height,
    )

    values = (
        float(len(by_frame)),
        _mean(counts),
        _std(counts),
        min(counts, default=0.0),
        max(counts, default=0.0),
        _mean(areas),
        _std(areas),
        _quantile(areas, 0.9),
        max(areas, default=0.0),
        _mean(large_players),
        _mean(x_spreads),
        _mean(y_spreads),
        _mean(raised_wrists),
        _mean(keypoint_completeness),
        _mean(centroid_shifts),
        max(centroid_shifts, default=0.0),
        _mean(count_deltas),
        max(count_deltas, default=0.0),
        _mean(pair_distances),
        _quantile(pair_distances, 0.1),
        _mean(vertical_alignments),
        _mean(horizontal_alignments),
        _mean(formation_anisotropies),
        _mean(nearest_player_motions),
        _finite_or(details.get("hit_count"), 0.0),
        min(_finite_or(details.get("minimum_normalized_distance"), 9.0), 9.0),
        _finite_or(details.get("maximum_approach_point_count"), 0.0),
        _finite_or(details.get("maximum_approach_rise_px"), 0.0) / height,
        _range(rim_distance_series),
        _last_minus_first(rim_distance_series),
        (
            math.dist(ball_center_series[0], ball_center_series[-1])
            if len(ball_center_series) > 1
            else 0.0
        ),
        float(len(observations)),
        float(len({str(row.get("player_id")) for row in observations})),
        min(wrist_distances, default=2.0),
        _range(wrist_distance_series),
        _last_minus_first(wrist_distance_series),
        max(control_scores, default=0.0),
        max(contact_counts, default=0.0),
        float(
            len(
                {
                    str(row.get("team_id"))
                    for row in observations
                    if row.get("team_id")
                }
            )
        ),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("reason evidence features must be finite")
    return dict(zip(FEATURE_NAMES, values, strict=True))


def build_reason_evidence_artifact(
    *,
    scene_artifact: Mapping[str, Any],
    pose_artifacts: Sequence[Mapping[str, Any]],
    pose_source_sha256s: Sequence[str],
    candidate_bundles: Sequence[Mapping[str, Any]],
    video_dimensions: Mapping[str, tuple[int | float, int | float]],
) -> dict[str, Any]:
    """Align existing pose frames and candidate evidence to a scene example set."""
    scene = verify_scene_embedding_artifact(scene_artifact)
    if len(pose_artifacts) != len(pose_source_sha256s):
        raise ValueError("pose artifacts and source hashes must align")

    by_video_frame: dict[
        str, dict[int, dict[tuple[float, float, float, float], Mapping[str, Any]]]
    ] = defaultdict(lambda: defaultdict(dict))
    for artifact in pose_artifacts:
        if artifact.get("schema_version") != "agu.official-pose.v1":
            raise ValueError("unsupported pose evidence schema")
        raw = artifact.get("raw_video")
        if not isinstance(raw, Mapping) or not raw.get("sha256"):
            raise ValueError("pose evidence requires raw-video provenance")
        source_sha = str(raw["sha256"])
        for detection in artifact.get("detections") or []:
            if not isinstance(detection, Mapping):
                continue
            bbox = detection.get("bbox")
            if not isinstance(bbox, Mapping):
                continue
            signature = tuple(
                round(float(bbox[name]), 3)
                for name in ("x1", "y1", "x2", "y2")
            )
            by_video_frame[source_sha][int(detection["frame"])][signature] = detection

    frame_indexes = {
        source_sha: sorted(frame_rows)
        for source_sha, frame_rows in by_video_frame.items()
    }
    events: dict[tuple[str, str], Mapping[str, Any]] = {}
    bundle_hashes: list[str] = []
    for bundle in candidate_bundles:
        bundle_sha = str(bundle.get("bundle_sha256") or "")
        if not bundle_sha:
            raise ValueError("candidate evidence requires bundle provenance")
        bundle_hashes.append(bundle_sha)
        for event in bundle.get("events") or []:
            if isinstance(event, Mapping):
                events[(bundle_sha, str(event.get("event_id") or ""))] = event

    examples = []
    for row in scene["examples"]:
        source_sha = str(row["source_video_sha256"])
        bundle_sha = str(row["candidate_bundle_sha256"])
        event_id = str(row["event_id"])
        event = events.get((bundle_sha, event_id))
        if event is None:
            raise ValueError("candidate evidence does not cover every scene example")
        dimensions = video_dimensions.get(source_sha)
        if dimensions is None:
            raise ValueError("video dimensions do not cover every scene example")
        sampling_window = row.get("sampling_window")
        if isinstance(sampling_window, Mapping):
            start_frame = int(sampling_window["start_frame"])
            end_frame = int(sampling_window["end_frame"])
        else:
            start_frame = int(event["start_frame"])
            end_frame = int(event["end_frame"])
        indexes = frame_indexes.get(source_sha, [])
        start = bisect_left(indexes, start_frame)
        end = bisect_right(indexes, end_frame)
        detections = [
            detection
            for frame in indexes[start:end]
            for detection in by_video_frame[source_sha][frame].values()
        ]
        feature_map = extract_reason_evidence_features(
            detections=detections,
            start_frame=start_frame,
            end_frame=end_frame,
            frame_width=dimensions[0],
            frame_height=dimensions[1],
            event=event,
        )
        examples.append(
            {
                "source_video_sha256": source_sha,
                "candidate_bundle_sha256": bundle_sha,
                "event_id": event_id,
                "features": [feature_map[name] for name in FEATURE_NAMES],
            }
        )
    return seal_reason_evidence_artifact(
        {
            "source_scene_artifact_sha256": scene["artifact_sha256"],
            "pose_source_sha256s": sorted(set(pose_source_sha256s)),
            "candidate_bundle_sha256s": sorted(set(bundle_hashes)),
            "examples": examples,
        }
    )


def seal_reason_evidence_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["schema_version"] = REASON_EVIDENCE_SCHEMA
    artifact["purpose"] = "offline_shot_reason_evidence_screening"
    artifact["runtime_consumable"] = False
    artifact["codex_runtime_answer_used"] = False
    artifact["feature_names"] = list(FEATURE_NAMES)
    _validate_artifact(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_reason_evidence_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_artifact(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("reason evidence artifact hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _validate_artifact(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != REASON_EVIDENCE_SCHEMA:
        raise ValueError("unsupported reason evidence schema")
    if artifact.get("runtime_consumable") is not False:
        raise ValueError("reason evidence must remain training-only")
    if artifact.get("codex_runtime_answer_used") is not False:
        raise ValueError("Codex must not provide runtime reason evidence")
    if tuple(artifact.get("feature_names") or ()) != FEATURE_NAMES:
        raise ValueError("reason evidence feature contract mismatch")
    for field in (
        "source_scene_artifact_sha256",
        "pose_source_sha256s",
        "candidate_bundle_sha256s",
    ):
        if not artifact.get(field):
            raise ValueError(f"reason evidence requires {field}")
    examples = artifact.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("reason evidence requires examples")
    seen: set[tuple[str, str, str]] = set()
    for row in examples:
        if not isinstance(row, Mapping):
            raise ValueError("reason evidence examples must be objects")
        if _FORBIDDEN_LABEL_FIELDS.intersection(row):
            raise ValueError("reason evidence examples must not contain labels")
        key = (
            str(row.get("source_video_sha256") or ""),
            str(row.get("candidate_bundle_sha256") or ""),
            str(row.get("event_id") or ""),
        )
        if not all(key) or key in seen:
            raise ValueError("reason evidence example keys must be complete and unique")
        seen.add(key)
        features = row.get("features")
        if (
            not isinstance(features, list)
            or len(features) != len(FEATURE_NAMES)
            or not all(math.isfinite(float(value)) for value in features)
        ):
            raise ValueError("reason evidence feature vector is invalid")


def _has_raised_wrist(keypoints: Mapping[str, Any]) -> bool:
    for side in ("left", "right"):
        wrist = keypoints.get(f"{side}_wrist")
        shoulder = keypoints.get(f"{side}_shoulder")
        if (
            isinstance(wrist, Mapping)
            and isinstance(shoulder, Mapping)
            and float(wrist["y"]) < float(shoulder["y"])
        ):
            return True
    return False


def _ball_rim_details(event: Mapping[str, Any]) -> Mapping[str, Any]:
    for evidence in event.get("evidence") or []:
        if (
            isinstance(evidence, Mapping)
            and evidence.get("kind") == "ball_rim_proximity_cluster"
        ):
            details = evidence.get("details")
            return details if isinstance(details, Mapping) else {}
    return {}


def _finite_values(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _formation_anisotropy(centers: Sequence[tuple[float, float]]) -> float:
    if len(centers) < 2:
        return 0.0
    covariance = np.cov(np.asarray(centers, dtype=np.float64), rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    total = float(np.sum(eigenvalues))
    return float(max(eigenvalues) / total) if total > 0.0 else 0.0


def _longest_player_series(
    rows: Sequence[Mapping[str, Any]], field: str
) -> list[float]:
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        value = _optional_finite(row.get(field))
        if value is None:
            continue
        grouped[str(row.get("player_id") or "")].append(
            (int(row.get("frame") or 0), value)
        )
    if not grouped:
        return []
    selected = min(
        grouped.values(),
        key=lambda series: (-len(series), min(value for _, value in series)),
    )
    return [value for _, value in sorted(selected)]


def _ball_rim_series(
    rows: Sequence[Mapping[str, Any]],
    *,
    frame_width: float,
    frame_height: float,
) -> tuple[list[float], list[tuple[float, float]]]:
    diagonal = math.hypot(frame_width, frame_height)
    samples: list[tuple[int, float, tuple[float, float]]] = []
    for row in rows:
        ball_center = _bbox_center(row.get("ball_bbox"))
        rim_center = _bbox_center(row.get("rim_bbox"))
        if ball_center is None or rim_center is None:
            continue
        normalized_ball = (
            ball_center[0] / diagonal,
            ball_center[1] / diagonal,
        )
        samples.append(
            (
                int(row.get("frame") or 0),
                math.dist(ball_center, rim_center) / diagonal,
                normalized_ball,
            )
        )
    samples.sort(key=lambda sample: sample[0])
    return (
        [distance for _, distance, _ in samples],
        [center for _, _, center in samples],
    )


def _bbox_center(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Mapping):
        return None
    coordinates = [_optional_finite(value.get(name)) for name in ("x1", "y1", "x2", "y2")]
    if any(coordinate is None for coordinate in coordinates):
        return None
    x1, y1, x2, y2 = (float(coordinate) for coordinate in coordinates)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _optional_finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _range(values: Sequence[float]) -> float:
    return max(values) - min(values) if values else 0.0


def _last_minus_first(values: Sequence[float]) -> float:
    return values[-1] - values[0] if len(values) > 1 else 0.0


def _finite_or(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _quantile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(values, q)) if values else 0.0


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
