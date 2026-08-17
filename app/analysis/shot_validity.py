"""Traditional hard-negative gate for raw-only field-goal candidates."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.analysis.action_ownership import extract_action_owner_features
from app.analysis.schemas import EventEvidenceResponse, GameEventResponse

SHOT_VALIDITY_MODEL_SCHEMA_V1 = "agu.shot-validity-extra-trees-model.v1"
SHOT_VALIDITY_MODEL_SCHEMA_V2 = "agu.shot-validity-extra-trees-model.v2"
SHOT_VALIDITY_MODEL_SCHEMA = "agu.shot-validity-extra-trees-model.v3"
SHOT_VALIDITY_LABEL_SCHEMA = "agu.shot-validity-labels.v1"
SHOT_VALIDITY_FEATURES_V2 = (
    "candidate_confidence",
    "hit_count",
    "minimum_normalized_distance",
    "maximum_approach_point_count",
    "normalized_approach_rise",
    "trajectory_confidence",
    "trajectory_resolved",
    "cluster_span_frames",
    "candidate_window_frames",
    "ball_frame_count",
    "player_candidate_count",
    "minimum_ball_player_distance",
    "best_pose_coverage",
    "best_wrist_ball_contact_coverage",
    "best_wrist_above_shoulder",
    "best_arm_extension",
    "best_wrist_upward_velocity",
    "best_temporal_coverage",
    "best_minimum_wrist_ball_distance",
    "best_ball_distance_exit_slope",
    "best_wrist_ball_exit_slope",
    "best_wrist_speed",
)
SHOT_VALIDITY_FEATURES_V1 = SHOT_VALIDITY_FEATURES_V2[:-4]
SHOT_VALIDITY_FEATURES = SHOT_VALIDITY_FEATURES_V2 + (
    "window_causal_present",
    "window_minimum_normalized_distance",
    "window_maximum_approach_point_count",
    "window_normalized_approach_rise",
    "window_ball_frame_count",
    "window_player_candidate_count",
    "window_best_pose_coverage",
    "window_best_wrist_ball_contact_coverage",
    "window_best_ball_distance_exit_slope",
    "window_best_wrist_ball_exit_slope",
    "window_best_wrist_speed",
)


@dataclass(frozen=True)
class ShotValidityPrediction:
    probability: float
    threshold: float
    accepted: bool
    features: Mapping[str, float]


class ShotValidityModel:
    """Dependency-light inference for a sealed Extra Trees shot gate."""

    def __init__(self, artifact: Mapping[str, Any]) -> None:
        self.artifact = verify_shot_validity_model(artifact)

    def predict(self, event: GameEventResponse, *, threshold: float | None = None) -> ShotValidityPrediction:
        features = extract_shot_validity_features(event)
        vector = [features[name] for name in self.artifact["feature_names"]]
        probabilities = [_tree_probability(tree, vector) for tree in self.artifact["trees"]]
        probability = sum(probabilities) / len(probabilities)
        gate = float(self.artifact["threshold"] if threshold is None else threshold)
        if not 0 <= gate <= 1:
            raise ValueError("shot validity threshold must be in [0,1]")
        return ShotValidityPrediction(
            probability=probability,
            threshold=gate,
            accepted=probability >= gate,
            features=features,
        )


def extract_shot_validity_features(event: GameEventResponse) -> dict[str, float]:
    """Summarize raw-derived candidate evidence without any reference answer."""

    if event.event_type != "field_goal_attempt":
        raise ValueError("shot validity features require a field_goal_attempt")
    causal_evidence = [
        item
        for item in event.evidence
        if item.kind == "ball_rim_proximity_cluster"
    ]
    window_evidence = next(
        (
            item
            for item in reversed(causal_evidence)
            if item.details.get("causal_evidence_version")
            == "agu_window_ball_rim_pose_v1"
        ),
        None,
    )
    evidence = next(
        (
            item
            for item in causal_evidence
            if item.details.get("causal_evidence_version")
            != "agu_window_ball_rim_pose_v1"
        ),
        event.evidence[0] if event.evidence else None,
    )
    details = evidence.details if evidence is not None else {}
    frames = [_safe_int(value) for value in details.get("frames") or []]
    frames = [value for value in frames if value is not None]
    observations = [
        item
        for item in details.get("candidate_player_observations") or []
        if isinstance(item, Mapping)
    ]
    heights = [
        max(
            1.0,
            _safe_float((item.get("bbox") or {}).get("y2"))
            - _safe_float((item.get("bbox") or {}).get("y1")),
        )
        for item in observations
        if isinstance(item.get("bbox"), Mapping)
    ]
    median_height = sorted(heights)[len(heights) // 2] if heights else 1.0
    by_player: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in observations:
        player_id = str(item.get("player_id") or "")
        if player_id:
            by_player[player_id].append(item)
    anchor_frame = event.release_frame or _safe_int(details.get("candidate_event_frame")) or event.end_frame
    ownership = [
        extract_action_owner_features(items, anchor_frame=anchor_frame)
        for items in by_player.values()
    ]
    trajectory_outcome = str(details.get("trajectory_outcome") or "unknown")
    ball_distances = [
        _safe_float(item.get("ball_player_distance"), default=math.inf)
        for item in observations
    ]
    finite_distances = [value for value in ball_distances if math.isfinite(value)]
    features = {
        "candidate_confidence": float(event.confidence),
        "hit_count": _safe_float(details.get("hit_count")),
        "minimum_normalized_distance": _safe_float(
            details.get("minimum_normalized_distance"), default=4.0
        ),
        "maximum_approach_point_count": _safe_float(
            details.get("maximum_approach_point_count")
        ),
        "normalized_approach_rise": _safe_float(
            details.get("maximum_approach_rise_px")
        )
        / median_height,
        "trajectory_confidence": _safe_float(details.get("trajectory_confidence")),
        "trajectory_resolved": float(trajectory_outcome in {"made", "missed"}),
        "cluster_span_frames": float(max(frames) - min(frames)) if frames else 0.0,
        "candidate_window_frames": float(max(0, event.end_frame - event.start_frame)),
        "ball_frame_count": float(len(set(frames))),
        "player_candidate_count": float(len(by_player)),
        "minimum_ball_player_distance": min(finite_distances, default=4.0),
        "best_pose_coverage": _maximum_feature(ownership, "pose_coverage"),
        "best_wrist_ball_contact_coverage": _maximum_feature(
            ownership, "wrist_ball_contact_coverage"
        ),
        "best_wrist_above_shoulder": _maximum_feature(
            ownership, "maximum_wrist_above_shoulder", default=-1.0
        ),
        "best_arm_extension": _maximum_feature(ownership, "maximum_arm_extension"),
        "best_wrist_upward_velocity": _maximum_feature(
            ownership, "maximum_wrist_upward_velocity"
        ),
        "best_temporal_coverage": _maximum_feature(ownership, "temporal_coverage"),
        "best_minimum_wrist_ball_distance": _minimum_feature(
            ownership, "minimum_wrist_ball_distance", default=4.0
        ),
        "best_ball_distance_exit_slope": _maximum_feature(
            ownership, "ball_distance_exit_slope"
        ),
        "best_wrist_ball_exit_slope": _maximum_feature(
            ownership, "wrist_ball_exit_slope"
        ),
        "best_wrist_speed": _maximum_feature(ownership, "maximum_wrist_speed"),
    }
    features.update(_extract_window_causal_features(event, window_evidence))
    return features


def _extract_window_causal_features(
    event: GameEventResponse,
    evidence: EventEvidenceResponse | None,
) -> dict[str, float]:
    if evidence is None:
        return {
            "window_causal_present": 0.0,
            "window_minimum_normalized_distance": 4.0,
            "window_maximum_approach_point_count": 0.0,
            "window_normalized_approach_rise": 0.0,
            "window_ball_frame_count": 0.0,
            "window_player_candidate_count": 0.0,
            "window_best_pose_coverage": 0.0,
            "window_best_wrist_ball_contact_coverage": 0.0,
            "window_best_ball_distance_exit_slope": 0.0,
            "window_best_wrist_ball_exit_slope": 0.0,
            "window_best_wrist_speed": 0.0,
        }
    details = evidence.details
    frames = [_safe_int(value) for value in details.get("frames") or []]
    frames = [value for value in frames if value is not None]
    observations = [
        item
        for item in details.get("candidate_player_observations") or []
        if isinstance(item, Mapping)
    ]
    heights = [
        max(
            1.0,
            _safe_float((item.get("bbox") or {}).get("y2"))
            - _safe_float((item.get("bbox") or {}).get("y1")),
        )
        for item in observations
        if isinstance(item.get("bbox"), Mapping)
    ]
    median_height = sorted(heights)[len(heights) // 2] if heights else 1.0
    by_player: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in observations:
        player_id = str(item.get("player_id") or "")
        if player_id:
            by_player[player_id].append(item)
    anchor_frame = (
        event.release_frame
        or _safe_int(details.get("candidate_event_frame"))
        or event.end_frame
    )
    ownership = [
        extract_action_owner_features(items, anchor_frame=anchor_frame)
        for items in by_player.values()
    ]
    return {
        "window_causal_present": 1.0,
        "window_minimum_normalized_distance": _safe_float(
            details.get("minimum_normalized_distance"), default=4.0
        ),
        "window_maximum_approach_point_count": _safe_float(
            details.get("maximum_approach_point_count")
        ),
        "window_normalized_approach_rise": _safe_float(
            details.get("maximum_approach_rise_px")
        )
        / median_height,
        "window_ball_frame_count": float(len(set(frames))),
        "window_player_candidate_count": float(len(by_player)),
        "window_best_pose_coverage": _maximum_feature(ownership, "pose_coverage"),
        "window_best_wrist_ball_contact_coverage": _maximum_feature(
            ownership, "wrist_ball_contact_coverage"
        ),
        "window_best_ball_distance_exit_slope": _maximum_feature(
            ownership, "ball_distance_exit_slope"
        ),
        "window_best_wrist_ball_exit_slope": _maximum_feature(
            ownership, "wrist_ball_exit_slope"
        ),
        "window_best_wrist_speed": _maximum_feature(
            ownership, "maximum_wrist_speed"
        ),
    }


def apply_shot_validity_gate(
    events: Sequence[GameEventResponse],
    *,
    model: ShotValidityModel,
    threshold: float | None = None,
) -> list[GameEventResponse]:
    """Drop rejected shots and their synthetic rebound children."""

    rejected_shot_ids: set[str] = set()
    scored: list[GameEventResponse] = []
    for event in events:
        if event.event_type != "field_goal_attempt":
            scored.append(event)
            continue
        prediction = model.predict(event, threshold=threshold)
        if not prediction.accepted:
            rejected_shot_ids.add(event.event_id)
            continue
        evidence = EventEvidenceResponse(
            evidence_id=f"{event.event_id}:shot-validity",
            kind="traditional_shot_validity_model",
            source_video_id=event.source_video_id,
            start_frame=event.start_frame,
            end_frame=event.end_frame,
            confidence=prediction.probability,
            details={
                "model_sha256": model.artifact["model_sha256"],
                "probability": prediction.probability,
                "threshold": prediction.threshold,
                "features": dict(prediction.features),
            },
        )
        scored.append(
            event.model_copy(
                update={
                    "confidence": min(1.0, max(event.confidence, prediction.probability)),
                    "evidence": [*event.evidence, evidence],
                    "reason": f"{event.reason}; traditional shot-validity gate accepted candidate",
                }
            )
        )
    return [
        event
        for event in scored
        if not rejected_shot_ids.intersection(event.related_event_ids)
    ]


def seal_shot_validity_model(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    artifact["schema_version"] = SHOT_VALIDITY_MODEL_SCHEMA
    artifact["feature_names"] = list(SHOT_VALIDITY_FEATURES)
    artifact.pop("model_sha256", None)
    _validate_model(artifact)
    artifact["model_sha256"] = _json_sha256(artifact)
    return artifact


def verify_shot_validity_model(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("model_sha256", ""))
    _validate_model(artifact)
    if _json_sha256(artifact) != claimed:
        raise ValueError("shot validity model hash mismatch")
    artifact["model_sha256"] = claimed
    return artifact


def _validate_model(artifact: Mapping[str, Any]) -> None:
    schema_version = artifact.get("schema_version")
    if schema_version not in {
        SHOT_VALIDITY_MODEL_SCHEMA_V1,
        SHOT_VALIDITY_MODEL_SCHEMA_V2,
        SHOT_VALIDITY_MODEL_SCHEMA,
    }:
        raise ValueError("unsupported shot validity model schema")
    expected_features = {
        SHOT_VALIDITY_MODEL_SCHEMA_V1: SHOT_VALIDITY_FEATURES_V1,
        SHOT_VALIDITY_MODEL_SCHEMA_V2: SHOT_VALIDITY_FEATURES_V2,
        SHOT_VALIDITY_MODEL_SCHEMA: SHOT_VALIDITY_FEATURES,
    }[schema_version]
    if artifact.get("feature_names") != list(expected_features):
        raise ValueError("shot validity feature contract mismatch")
    if not artifact.get("training_manifest_sha256"):
        raise ValueError("shot validity model requires training manifest provenance")
    threshold = _safe_float(artifact.get("threshold"), default=-1.0)
    if not 0 <= threshold <= 1:
        raise ValueError("shot validity model threshold must be in [0,1]")
    trees = artifact.get("trees")
    if not isinstance(trees, list) or not trees:
        raise ValueError("shot validity model requires trees")
    for tree in trees:
        if not isinstance(tree, Mapping):
            raise ValueError("shot validity tree must be an object")
        arrays = [
            tree.get(name)
            for name in (
                "children_left",
                "children_right",
                "feature",
                "threshold",
                "positive_probability",
            )
        ]
        if any(not isinstance(value, list) or not value for value in arrays):
            raise ValueError("shot validity tree arrays must be non-empty lists")
        if len({len(value) for value in arrays}) != 1:
            raise ValueError("shot validity tree arrays must have equal length")
        node_count = len(arrays[0])
        for node, (left, right, feature) in enumerate(
            zip(arrays[0], arrays[1], arrays[2], strict=True)
        ):
            leaf = int(left) < 0 or int(right) < 0 or int(feature) < 0
            if leaf:
                continue
            if not (0 <= int(left) < node_count and 0 <= int(right) < node_count):
                raise ValueError(f"shot validity tree child index is invalid at node {node}")
            if not 0 <= int(feature) < len(expected_features):
                raise ValueError(f"shot validity tree feature index is invalid at node {node}")


def _tree_probability(tree: Mapping[str, Any], vector: Sequence[float]) -> float:
    node = 0
    while True:
        left = int(tree["children_left"][node])
        right = int(tree["children_right"][node])
        feature = int(tree["feature"][node])
        if left < 0 or right < 0 or feature < 0:
            return max(0.0, min(1.0, float(tree["positive_probability"][node])))
        node = left if vector[feature] <= float(tree["threshold"][node]) else right


def _maximum_feature(
    rows: Sequence[Mapping[str, float]], key: str, *, default: float = 0.0
) -> float:
    return max((float(row.get(key, default)) for row in rows), default=default)


def _minimum_feature(
    rows: Sequence[Mapping[str, float]], key: str, *, default: float = 0.0
) -> float:
    return min((float(row.get(key, default)) for row in rows), default=default)


def _safe_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: object, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
