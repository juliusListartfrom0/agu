"""Traditional temporal features and sealed models for action ownership."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ACTION_OWNER_MODEL_SCHEMA = "agu.action-owner-model.v2"
ACTION_OWNER_EXTRA_TREES_MODEL_SCHEMA = "agu.action-owner-extra-trees-model.v1"
ACTION_OWNER_FEATURES = (
    "minimum_ball_distance",
    "mean_ball_distance",
    "ball_distance_exit_slope",
    "minimum_wrist_ball_distance",
    "mean_wrist_ball_distance",
    "wrist_ball_exit_slope",
    "wrist_ball_contact_coverage",
    "maximum_wrist_above_shoulder",
    "maximum_arm_extension",
    "maximum_wrist_upward_velocity",
    "maximum_wrist_speed",
    "pose_coverage",
    "temporal_coverage",
)


@dataclass(frozen=True)
class ActionOwnerPrediction:
    player_id: str
    probability: float
    features: Mapping[str, float]


def extract_action_owner_features(
    observations: Sequence[Mapping[str, Any]],
    *,
    anchor_frame: int,
) -> dict[str, float]:
    """Summarize one tracked player around a release/control anchor."""

    ordered = sorted(observations, key=lambda item: int(item.get("frame", 0)))
    if not ordered:
        return {name: 0.0 for name in ACTION_OWNER_FEATURES}
    distances = [
        float(item["ball_player_distance"]) for item in ordered if _finite_number(item.get("ball_player_distance"))
    ]
    wrist_above: list[float] = []
    arm_extensions: list[float] = []
    wrist_ball_samples: list[tuple[int, float]] = []
    wrist_samples: dict[str, list[tuple[int, float, float, float]]] = {
        "left": [],
        "right": [],
    }
    pose_frames = 0
    for item in ordered:
        box = item.get("bbox") or {}
        height = max(1.0, float(box.get("y2", 0.0)) - float(box.get("y1", 0.0)))
        width = max(1.0, float(box.get("x2", 0.0)) - float(box.get("x1", 0.0)))
        scale = max(height, width)
        keypoints = item.get("keypoints") or {}
        ball_center = _point_value(item.get("ball_center"))
        frame_has_pose = False
        frame_wrist_ball_distances: list[float] = []
        for side in ("left", "right"):
            shoulder = _point(keypoints, f"{side}_shoulder")
            elbow = _point(keypoints, f"{side}_elbow")
            wrist = _point(keypoints, f"{side}_wrist")
            if shoulder is not None and wrist is not None:
                wrist_above.append((shoulder[1] - wrist[1]) / height)
                frame_has_pose = True
            if shoulder is not None and elbow is not None and wrist is not None:
                extension = (math.dist(shoulder, elbow) + math.dist(elbow, wrist)) / scale
                arm_extensions.append(extension)
            if wrist is not None:
                wrist_samples[side].append((int(item.get("frame", 0)), wrist[0], wrist[1], scale))
                frame_has_pose = True
                if ball_center is not None:
                    frame_wrist_ball_distances.append(math.dist(wrist, ball_center) / max(1.0, height))
        if frame_has_pose:
            pose_frames += 1
        if frame_wrist_ball_distances:
            wrist_ball_samples.append((int(item.get("frame", 0)), min(frame_wrist_ball_distances)))
    upward_velocity: list[float] = []
    wrist_speed: list[float] = []
    for samples in wrist_samples.values():
        for previous, current in zip(samples, samples[1:]):
            frame_gap = current[0] - previous[0]
            if frame_gap <= 0:
                continue
            scale = max(1.0, (previous[3] + current[3]) / 2.0)
            upward_velocity.append((previous[2] - current[2]) / scale / frame_gap)
            wrist_speed.append(math.dist((previous[1], previous[2]), (current[1], current[2])) / scale / frame_gap)
    return {
        "minimum_ball_distance": min(distances, default=4.0),
        "mean_ball_distance": sum(distances) / len(distances) if distances else 4.0,
        "ball_distance_exit_slope": _exit_slope(ordered, anchor_frame=anchor_frame),
        "minimum_wrist_ball_distance": min((distance for _, distance in wrist_ball_samples), default=4.0),
        "mean_wrist_ball_distance": (
            sum(distance for _, distance in wrist_ball_samples) / len(wrist_ball_samples) if wrist_ball_samples else 4.0
        ),
        "wrist_ball_exit_slope": _sample_exit_slope(wrist_ball_samples, anchor_frame=anchor_frame),
        "wrist_ball_contact_coverage": len(wrist_ball_samples) / len(ordered),
        "maximum_wrist_above_shoulder": max(wrist_above, default=-1.0),
        "maximum_arm_extension": max(arm_extensions, default=0.0),
        "maximum_wrist_upward_velocity": max(upward_velocity, default=0.0),
        "maximum_wrist_speed": max(wrist_speed, default=0.0),
        "pose_coverage": pose_frames / len(ordered),
        "temporal_coverage": min(1.0, len({int(item.get("frame", 0)) for item in ordered}) / 8.0),
    }


class ActionOwnerModel:
    """Dependency-light inference for a hash-sealed action-owner model."""

    def __init__(self, artifact: Mapping[str, Any]) -> None:
        self.artifact = verify_action_owner_model(artifact)

    @classmethod
    def from_path(cls, path: str | Path) -> ActionOwnerModel:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def rank(
        self,
        observations: Sequence[Mapping[str, Any]],
        *,
        anchor_frame: int,
        maximum_frame_distance: int = 45,
    ) -> list[ActionOwnerPrediction]:
        if maximum_frame_distance <= 0:
            raise ValueError("maximum_frame_distance must be positive")
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for item in observations:
            player_id = str(item.get("player_id") or "")
            try:
                frame = int(item.get("frame", 0))
            except (TypeError, ValueError):
                continue
            if player_id and abs(frame - anchor_frame) <= maximum_frame_distance:
                grouped.setdefault(player_id, []).append(item)
        predictions = []
        for player_id, player_observations in grouped.items():
            features = extract_action_owner_features(player_observations, anchor_frame=anchor_frame)
            if self.artifact["schema_version"] == ACTION_OWNER_MODEL_SCHEMA:
                probability = _linear_probability(self.artifact, features)
            else:
                probability = _extra_trees_probability(self.artifact, features)
            predictions.append(ActionOwnerPrediction(player_id, probability, features))
        return sorted(predictions, key=lambda item: (-item.probability, item.player_id))


def seal_action_owner_model(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    artifact["schema_version"] = ACTION_OWNER_MODEL_SCHEMA
    artifact["feature_names"] = list(ACTION_OWNER_FEATURES)
    artifact.pop("model_sha256", None)
    _validate_model_shape(artifact)
    artifact["model_sha256"] = _json_sha256(artifact)
    return artifact


def seal_extra_trees_action_owner_model(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Seal a sklearn-independent Extra Trees representation for AGU runtime use."""

    artifact = dict(payload)
    artifact["schema_version"] = ACTION_OWNER_EXTRA_TREES_MODEL_SCHEMA
    artifact["feature_names"] = list(ACTION_OWNER_FEATURES)
    artifact.pop("model_sha256", None)
    _validate_extra_trees_model(artifact)
    artifact["model_sha256"] = _json_sha256(artifact)
    return artifact


def verify_action_owner_model(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    claimed_hash = str(artifact.pop("model_sha256", ""))
    schema_version = artifact.get("schema_version")
    if schema_version not in {
        ACTION_OWNER_MODEL_SCHEMA,
        ACTION_OWNER_EXTRA_TREES_MODEL_SCHEMA,
    }:
        raise ValueError("unsupported action-owner model schema")
    if artifact.get("feature_names") != list(ACTION_OWNER_FEATURES):
        raise ValueError("action-owner model feature contract mismatch")
    if schema_version == ACTION_OWNER_MODEL_SCHEMA:
        _validate_model_shape(artifact)
    else:
        _validate_extra_trees_model(artifact)
    if _json_sha256(artifact) != claimed_hash:
        raise ValueError("action-owner model hash mismatch")
    artifact["model_sha256"] = claimed_hash
    return artifact


def _validate_model_shape(artifact: Mapping[str, Any]) -> None:
    feature_count = len(ACTION_OWNER_FEATURES)
    for field in ("coefficients", "feature_mean", "feature_scale"):
        values = artifact.get(field)
        if not isinstance(values, list) or len(values) != feature_count:
            raise ValueError(f"action-owner model {field} must contain {feature_count} values")
        if not all(_finite_number(value) for value in values):
            raise ValueError(f"action-owner model {field} must contain finite numbers")
    if not _finite_number(artifact.get("intercept")):
        raise ValueError("action-owner model intercept must be finite")
    if not artifact.get("training_manifest_sha256"):
        raise ValueError("action-owner model requires training manifest provenance")


def _validate_extra_trees_model(artifact: Mapping[str, Any]) -> None:
    trees = artifact.get("trees")
    if not isinstance(trees, list) or not trees:
        raise ValueError("action-owner Extra Trees model requires at least one tree")
    if not artifact.get("training_manifest_sha256"):
        raise ValueError("action-owner model requires training manifest provenance")
    for tree in trees:
        if not isinstance(tree, Mapping):
            raise ValueError("action-owner Extra Trees entries must be objects")
        fields = {
            name: tree.get(name)
            for name in (
                "children_left",
                "children_right",
                "feature",
                "threshold",
                "positive_probability",
            )
        }
        lengths = {len(value) for value in fields.values() if isinstance(value, list)}
        if len(fields) != sum(isinstance(value, list) for value in fields.values()) or lengths == {0}:
            raise ValueError("action-owner Extra Trees arrays must be non-empty lists")
        if len(lengths) != 1:
            raise ValueError("action-owner Extra Trees arrays must have equal length")
        node_count = next(iter(lengths))
        for node in range(node_count):
            left = fields["children_left"][node]
            right = fields["children_right"][node]
            feature = fields["feature"][node]
            threshold = fields["threshold"][node]
            probability = fields["positive_probability"][node]
            if not all(isinstance(value, int) and not isinstance(value, bool) for value in (left, right, feature)):
                raise ValueError("action-owner Extra Trees indexes must be integers")
            is_leaf = left == -1 and right == -1
            if not is_leaf and not (0 <= left < node_count and 0 <= right < node_count):
                raise ValueError("action-owner Extra Trees child index is invalid")
            if not is_leaf and not (0 <= feature < len(ACTION_OWNER_FEATURES)):
                raise ValueError("action-owner Extra Trees feature index is invalid")
            if not _finite_number(threshold) or not _finite_number(probability):
                raise ValueError("action-owner Extra Trees values must be finite")
            if not 0.0 <= float(probability) <= 1.0:
                raise ValueError("action-owner Extra Trees probabilities must be within [0, 1]")


def _linear_probability(artifact: Mapping[str, Any], features: Mapping[str, float]) -> float:
    score = float(artifact["intercept"])
    for index, name in enumerate(ACTION_OWNER_FEATURES):
        standardized = (features[name] - float(artifact["feature_mean"][index])) / max(
            float(artifact["feature_scale"][index]), 1e-9
        )
        score += standardized * float(artifact["coefficients"][index])
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, score))))


def _extra_trees_probability(artifact: Mapping[str, Any], features: Mapping[str, float]) -> float:
    feature_vector = [features[name] for name in ACTION_OWNER_FEATURES]
    probabilities = []
    for tree in artifact["trees"]:
        node = 0
        while tree["children_left"][node] != -1:
            feature = tree["feature"][node]
            if feature_vector[feature] <= tree["threshold"][node]:
                node = tree["children_left"][node]
            else:
                node = tree["children_right"][node]
        probabilities.append(float(tree["positive_probability"][node]))
    return sum(probabilities) / len(probabilities)


def _exit_slope(observations: Sequence[Mapping[str, Any]], *, anchor_frame: int) -> float:
    samples = [
        (int(item.get("frame", 0)), float(item["ball_player_distance"]))
        for item in observations
        if _finite_number(item.get("ball_player_distance"))
    ]
    return _sample_exit_slope(samples, anchor_frame=anchor_frame)


def _sample_exit_slope(samples: Sequence[tuple[int, float]], *, anchor_frame: int) -> float:
    before = [item for item in samples if item[0] <= anchor_frame]
    after = [item for item in samples if item[0] > anchor_frame]
    if not before or not after:
        return 0.0
    left = min(before, key=lambda item: (abs(item[0] - anchor_frame), -item[0]))
    right = min(after, key=lambda item: (abs(item[0] - anchor_frame), item[0]))
    return (right[1] - left[1]) / max(1, right[0] - left[0])


def _point(keypoints: Mapping[str, Any], name: str) -> tuple[float, float] | None:
    value = keypoints.get(name)
    if not isinstance(value, Mapping):
        return None
    if not _finite_number(value.get("x")) or not _finite_number(value.get("y")):
        return None
    return float(value["x"]), float(value["y"])


def _point_value(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, Mapping):
        return None
    if not _finite_number(value.get("x")) or not _finite_number(value.get("y")):
        return None
    return float(value["x"]), float(value["y"])


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
