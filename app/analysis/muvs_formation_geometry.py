"""Training-only player/rim formation screening on reviewed MUVS frames."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MUVS_FORMATION_DETECTIONS_SCHEMA = "agu.muvs-formation-detections.v1"
MUVS_FORMATION_GEOMETRY_SCREEN_SCHEMA = "agu.muvs-formation-geometry-screen.v1"
MUVS_FORMATION_STATES = {
    "live_play",
    "free_throw_setup",
    "dead_ball_timeout",
    "uncertain",
}
MUVS_FORMATION_OBJECT_TYPES = {
    "basketball",
    "rim",
    "player",
    "referee",
}
_FEATURE_CANDIDATES = (
    "aggregate_pair",
    "player_slots_pair",
    "combined_pair",
)
_SLOT_COUNT = 6
_SHA256_LENGTH = 64


def seal_muvs_formation_detections(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal detector output that is isolated from the AGU runtime."""

    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["schema_version"] = MUVS_FORMATION_DETECTIONS_SCHEMA
    artifact["purpose"] = "offline_muvs_source_formation_pretraining"
    artifact["runtime_consumable"] = False
    artifact["codex_runtime_answer_used"] = False
    _validate_detections(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_muvs_formation_detections(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_detections(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("MUVS formation detection hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def screen_muvs_formation_geometry(
    detection_artifact: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate fixed player/rim geometry with leave-one-event-out folds."""

    payloads = (
        [detection_artifact]
        if isinstance(detection_artifact, Mapping)
        else list(detection_artifact)
    )
    if not payloads:
        raise ValueError("MUVS formation screen requires detections")
    artifacts = [
        verify_muvs_formation_detections(payload) for payload in payloads
    ]
    rows_by_frame_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for artifact in artifacts:
        for row in artifact["examples"]:
            if row["state"] == "uncertain":
                continue
            key = (
                str(row["frame_before_sha256"]),
                str(row["frame_after_sha256"]),
            )
            candidate = {
                **row,
                "_detection_artifact_sha256": artifact["artifact_sha256"],
            }
            previous = rows_by_frame_pair.get(key)
            if previous is not None:
                if (
                    previous["event_id"] != candidate["event_id"]
                    or previous["state"] != candidate["state"]
                    or previous["frames"] != candidate["frames"]
                ):
                    raise ValueError("duplicate MUVS source frames disagree")
                continue
            rows_by_frame_pair[key] = candidate

    rows = list(rows_by_frame_pair.values())
    labels = np.asarray(
        [row["state"] == "free_throw_setup" for row in rows],
        dtype=np.int64,
    )
    groups = np.asarray([str(row["event_id"]) for row in rows])
    if len(set(labels.tolist())) != 2 or len(set(groups.tolist())) < 3:
        raise ValueError("MUVS formation screen requires both classes and events")

    candidate_results = [
        _screen_candidate(
            rows,
            labels=labels,
            groups=groups,
            candidate=candidate,
        )
        for candidate in _FEATURE_CANDIDATES
    ]
    best = max(
        candidate_results,
        key=lambda row: (
            float(row["balanced_accuracy"]),
            float(row["worst_positive_event_recall"]),
            float(row["roc_auc"]),
            -_FEATURE_CANDIDATES.index(str(row["feature"])),
        ),
    )
    summary_keys = (
        "feature",
        "classifier",
        "feature_dimension",
        "balanced_accuracy",
        "roc_auc",
        "positive_recall",
        "negative_specificity",
        "worst_positive_event_recall",
        "worst_negative_event_specificity",
    )
    accepted = bool(
        float(best["balanced_accuracy"]) >= 0.85
        and float(best["worst_positive_event_recall"]) >= 0.5
        and float(best["negative_specificity"]) >= 0.85
    )
    artifact: dict[str, Any] = {
        "schema_version": MUVS_FORMATION_GEOMETRY_SCREEN_SCHEMA,
        "purpose": "offline_source_feature_acceptance_only",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "detection_artifact_sha256s": [
            artifact["artifact_sha256"] for artifact in artifacts
        ],
        "resolved_examples": len(rows),
        "event_count": len(set(groups.tolist())),
        "positive_examples": int(labels.sum()),
        "negative_examples": int((labels == 0).sum()),
        "protocol": {
            "grouping": "leave_one_event_out",
            "threshold": "fixed_zero_logistic_margin",
            "candidate_features": list(_FEATURE_CANDIDATES),
            "selection_data": "MUVS_source_only",
            "acceptance_metric": "source_oof_balanced_accuracy",
            "acceptance_threshold": 0.85,
            "minimum_positive_event_recall": 0.5,
            "minimum_negative_specificity": 0.85,
        },
        "candidates": candidate_results,
        "best_candidate": {key: best[key] for key in summary_keys},
        "accepted": accepted,
    }
    _validate_screen(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_muvs_formation_geometry_screen(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_screen(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("MUVS formation geometry screen hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _screen_candidate(
    rows: list[Mapping[str, Any]],
    *,
    labels: np.ndarray,
    groups: np.ndarray,
    candidate: str,
) -> dict[str, Any]:
    features = _feature_matrix(rows, candidate=candidate)
    scores = np.zeros(len(rows), dtype=np.float64)
    fold_rows = []
    splitter = LeaveOneGroupOut()
    for train, test in splitter.split(features, labels, groups):
        if len(set(labels[train].tolist())) != 2:
            raise ValueError("each MUVS formation fold must retain both classes")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=5000,
                random_state=0,
            ),
        )
        model.fit(features[train], labels[train])
        fold_scores = model.decision_function(features[test])
        scores[test] = fold_scores
        fold_rows.append(
            _fold_metrics(
                event_id=str(groups[test][0]),
                labels=labels[test],
                scores=fold_scores,
            )
        )

    predictions = scores >= 0.0
    positive_recall = _class_recall(labels, predictions, positive_class=1)
    negative_specificity = _class_recall(
        labels,
        predictions,
        positive_class=0,
    )
    positive_event_recalls = [
        float(row["positive_recall"])
        for row in fold_rows
        if row["positive_examples"] > 0
    ]
    negative_event_specificities = [
        float(row["negative_specificity"])
        for row in fold_rows
        if row["negative_examples"] > 0
    ]
    return {
        "feature": candidate,
        "classifier": "StandardScaler+balanced_LogisticRegression_C1",
        "feature_dimension": int(features.shape[1]),
        "balanced_accuracy": float(
            (positive_recall + negative_specificity) / 2.0
        ),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "positive_recall": positive_recall,
        "negative_specificity": negative_specificity,
        "worst_positive_event_recall": min(positive_event_recalls),
        "worst_negative_event_specificity": min(
            negative_event_specificities
        ),
        "folds": fold_rows,
        "oof_predictions": [
            {
                "sample_id": str(row["sample_id"]),
                "event_id": str(row["event_id"]),
                "detection_artifact_sha256": str(
                    row["_detection_artifact_sha256"]
                ),
                "label": bool(label),
                "score": float(score),
                "prediction": bool(score >= 0.0),
            }
            for row, label, score in zip(rows, labels, scores, strict=True)
        ],
    }


def _feature_matrix(
    rows: list[Mapping[str, Any]],
    *,
    candidate: str,
) -> np.ndarray:
    features = []
    for row in rows:
        frames = row["frames"]
        before_aggregate = _frame_aggregate(frames[0])
        after_aggregate = _frame_aggregate(frames[1])
        before_slots = _frame_player_slots(frames[0])
        after_slots = _frame_player_slots(frames[1])
        aggregate_pair = _pair_features(before_aggregate, after_aggregate)
        slots_pair = _pair_features(before_slots, after_slots)
        if candidate == "aggregate_pair":
            feature = aggregate_pair
        elif candidate == "player_slots_pair":
            feature = slots_pair
        elif candidate == "combined_pair":
            feature = np.concatenate((aggregate_pair, slots_pair))
        else:
            raise ValueError("unsupported MUVS formation feature candidate")
        features.append(feature)
    return np.stack(features)


def _pair_features(
    before: np.ndarray,
    after: np.ndarray,
) -> np.ndarray:
    return np.concatenate((before, after, np.abs(after - before)))


def _frame_aggregate(frame: Mapping[str, Any]) -> np.ndarray:
    geometry = _resolve_frame_geometry(frame)
    players = geometry["players"]
    if players:
        dx = np.asarray([row["dx"] for row in players], dtype=np.float32)
        dy = np.asarray([row["dy"] for row in players], dtype=np.float32)
        radius = np.asarray(
            [row["radius"] for row in players],
            dtype=np.float32,
        )
        height = np.asarray(
            [row["height"] for row in players],
            dtype=np.float32,
        )
        confidence = float(
            np.mean([row["confidence"] for row in players])
        )
        quantiles = np.concatenate(
            [
                np.quantile(values, (0.25, 0.5, 0.75))
                for values in (dx, dy, radius, height)
            ]
        )
        radial_counts = np.asarray(
            [(radius <= limit).sum() / 15.0 for limit in (0.12, 0.25, 0.4)]
        )
    else:
        confidence = 0.0
        quantiles = np.zeros(12, dtype=np.float32)
        radial_counts = np.zeros(3, dtype=np.float32)
    return np.asarray(
        [
            *geometry["rim_features"],
            min(len(players), 15) / 15.0,
            *quantiles.tolist(),
            *radial_counts.tolist(),
            confidence,
        ],
        dtype=np.float32,
    )


def _frame_player_slots(frame: Mapping[str, Any]) -> np.ndarray:
    geometry = _resolve_frame_geometry(frame)
    values = [*geometry["rim_features"], min(len(geometry["players"]), 15) / 15.0]
    for player in geometry["players"][:_SLOT_COUNT]:
        values.extend(
            (
                1.0,
                player["dx"],
                player["dy"],
                player["width"],
                player["height"],
                player["confidence"],
            )
        )
    values.extend([0.0] * (6 * (_SLOT_COUNT - len(geometry["players"][:_SLOT_COUNT]))))
    return np.asarray(values, dtype=np.float32)


def _resolve_frame_geometry(frame: Mapping[str, Any]) -> dict[str, Any]:
    detections = frame["detections"]
    rims = [row for row in detections if row["object_type"] == "rim"]
    selected_rim = max(rims, key=lambda row: row["confidence"]) if rims else None
    if selected_rim is None:
        anchor_x = 0.5
        anchor_y = 0.22
        mirror = False
        rim_features = [0.0, 0.0, 0.5, anchor_y, 0.0, 0.0]
    else:
        x1, y1, x2, y2 = selected_rim["bbox"]
        raw_anchor_x = (x1 + x2) / 2.0
        anchor_y = (y1 + y2) / 2.0
        mirror = raw_anchor_x > 0.5
        anchor_x = 1.0 - raw_anchor_x if mirror else raw_anchor_x
        rim_features = [
            1.0,
            float(selected_rim["confidence"]),
            anchor_x,
            anchor_y,
            x2 - x1,
            y2 - y1,
        ]

    players = []
    for row in detections:
        if row["object_type"] != "player":
            continue
        x1, y1, x2, y2 = row["bbox"]
        player_x = (x1 + x2) / 2.0
        if mirror:
            player_x = 1.0 - player_x
        foot_y = y2
        dx = player_x - anchor_x
        dy = foot_y - anchor_y
        players.append(
            {
                "dx": dx,
                "dy": dy,
                "radius": math.hypot(dx, dy),
                "width": x2 - x1,
                "height": y2 - y1,
                "confidence": float(row["confidence"]),
            }
        )
    players.sort(
        key=lambda row: (
            row["radius"],
            row["dx"],
            row["dy"],
        )
    )
    return {
        "rim_features": rim_features,
        "players": players,
    }


def _fold_metrics(
    *,
    event_id: str,
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    predictions = scores >= 0.0
    positives = int(labels.sum())
    negatives = int((labels == 0).sum())
    return {
        "event_id": event_id,
        "examples": len(labels),
        "positive_examples": positives,
        "negative_examples": negatives,
        "positive_recall": (
            _class_recall(labels, predictions, positive_class=1)
            if positives
            else None
        ),
        "negative_specificity": (
            _class_recall(labels, predictions, positive_class=0)
            if negatives
            else None
        ),
    }


def _class_recall(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    positive_class: int,
) -> float:
    mask = labels == positive_class
    if not mask.any():
        raise ValueError("class recall requires at least one example")
    return float((predictions[mask] == bool(positive_class)).mean())


def _validate_detections(artifact: Mapping[str, Any]) -> None:
    detector = artifact.get("detector")
    examples = artifact.get("examples")
    if (
        artifact.get("schema_version") != MUVS_FORMATION_DETECTIONS_SCHEMA
        or artifact.get("purpose")
        != "offline_muvs_source_formation_pretraining"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or not isinstance(detector, Mapping)
        or set(detector)
        != {
            "backend",
            "model_name",
            "model_sha256",
            "class_names",
            "image_size",
            "confidence",
        }
        or detector.get("backend") != "ultralytics_yolo"
        or not str(detector.get("model_name") or "")
        or detector.get("class_names")
        != ["basketball", "rim", "player", "referee"]
        or int(detector.get("image_size", 0)) < 32
        or not 0.0 <= float(detector.get("confidence", -1.0)) <= 1.0
        or not isinstance(examples, list)
        or not examples
    ):
        raise ValueError("invalid MUVS formation detection artifact")
    for key in ("plan_sha256", "frames_sha256", "review_sha256"):
        _require_sha256(artifact.get(key))
    _require_sha256(detector.get("model_sha256"))

    sample_ids = []
    for row in examples:
        if (
            set(row)
            != {
                "sample_id",
                "event_id",
                "split",
                "state",
                "frame_before_sha256",
                "frame_after_sha256",
                "frames",
            }
            or not str(row.get("sample_id") or "")
            or not str(row.get("event_id") or "")
            or row.get("split") not in {"train", "test"}
            or row.get("state") not in MUVS_FORMATION_STATES
            or not isinstance(row.get("frames"), list)
            or len(row["frames"]) != 2
        ):
            raise ValueError("invalid MUVS formation detection example")
        _require_sha256(row.get("frame_before_sha256"))
        _require_sha256(row.get("frame_after_sha256"))
        for frame in row["frames"]:
            _validate_frame(frame)
        sample_ids.append(str(row["sample_id"]))
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate MUVS formation detection sample")


def _validate_frame(frame: object) -> None:
    if (
        not isinstance(frame, Mapping)
        or set(frame) != {"width", "height", "detections"}
        or int(frame.get("width", 0)) < 1
        or int(frame.get("height", 0)) < 1
        or not isinstance(frame.get("detections"), list)
    ):
        raise ValueError("invalid MUVS formation detection frame")
    for detection in frame["detections"]:
        bbox = detection.get("bbox") if isinstance(detection, Mapping) else None
        if (
            not isinstance(detection, Mapping)
            or set(detection) != {"object_type", "confidence", "bbox"}
            or detection.get("object_type") not in MUVS_FORMATION_OBJECT_TYPES
            or not 0.0 <= float(detection.get("confidence", -1.0)) <= 1.0
            or not isinstance(bbox, list)
            or len(bbox) != 4
        ):
            raise ValueError("invalid MUVS formation detection")
        values = [float(value) for value in bbox]
        if (
            not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values)
            or values[0] >= values[2]
            or values[1] >= values[3]
        ):
            raise ValueError("invalid MUVS formation detection")


def _validate_screen(artifact: Mapping[str, Any]) -> None:
    candidates = artifact.get("candidates")
    best = artifact.get("best_candidate")
    protocol = artifact.get("protocol")
    if (
        artifact.get("schema_version")
        != MUVS_FORMATION_GEOMETRY_SCREEN_SCHEMA
        or artifact.get("purpose") != "offline_source_feature_acceptance_only"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or not isinstance(protocol, Mapping)
        or protocol.get("grouping") != "leave_one_event_out"
        or not isinstance(candidates, list)
        or [row.get("feature") for row in candidates]
        != list(_FEATURE_CANDIDATES)
        or not isinstance(best, Mapping)
        or best.get("feature") not in _FEATURE_CANDIDATES
        or not isinstance(artifact.get("accepted"), bool)
        or int(artifact.get("resolved_examples", 0)) < 1
        or int(artifact.get("event_count", 0)) < 3
    ):
        raise ValueError("invalid MUVS formation geometry screen")
    artifact_hashes = artifact.get("detection_artifact_sha256s")
    if not isinstance(artifact_hashes, list) or not artifact_hashes:
        raise ValueError("MUVS formation screen requires detection hashes")
    for value in artifact_hashes:
        _require_sha256(value)
    for row in candidates:
        if int(row.get("feature_dimension", 0)) < 1:
            raise ValueError("invalid MUVS formation feature dimension")
        for key in (
            "balanced_accuracy",
            "roc_auc",
            "positive_recall",
            "negative_specificity",
            "worst_positive_event_recall",
            "worst_negative_event_specificity",
        ):
            value = float(row.get(key, math.nan))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("invalid MUVS formation geometry metric")


def _require_sha256(value: object) -> str:
    text = str(value or "")
    if len(text) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError("invalid MUVS formation SHA-256")
    return text


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
