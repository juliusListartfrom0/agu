from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

REPLAY_TRANSITION_EVIDENCE_SCHEMA = "agu.replay-transition-evidence.v1"
REPLAY_LOGO_RECURRENCE_SCHEMA = "agu.replay-logo-recurrence-evidence.v1"


@dataclass(frozen=True)
class ReplayTransitionEvidence:
    event_id: str
    anchor_frame: int
    window_start_frame: int
    window_end_frame: int
    transition_frames: tuple[int, ...]
    transition_offsets_seconds: tuple[float, ...]
    transition_count: int
    near_transition_count: int
    maximum_transition_probability: float
    broadcast_state: str
    method: str = "transnetv2_scene_transition_v1"


def summarize_transition_probabilities(
    *,
    event_id: str,
    anchor_frame: int,
    window_start_frame: int,
    probabilities: Sequence[float],
    fps: float,
    transition_threshold: float = 0.5,
    near_seconds: float = 8.0,
    minimum_transition_count: int = 5,
    minimum_near_transition_count: int = 3,
) -> ReplayTransitionEvidence:
    """Summarize scene cuts as conservative replay evidence.

    Scene transitions are not replay semantics. The result rejects a window
    only when both the total and anchor-local cut floors pass; all other
    windows remain unknown rather than being promoted to live action.
    """

    if (
        fps <= 0
        or window_start_frame < 0
        or anchor_frame < window_start_frame
        or not 0.0 < transition_threshold < 1.0
        or near_seconds <= 0
        or minimum_transition_count <= 0
        or minimum_near_transition_count <= 0
    ):
        raise ValueError("replay transition configuration is invalid")
    values = tuple(float(value) for value in probabilities)
    if not values or any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("transition probabilities must be non-empty values in [0,1]")
    transition_indices = tuple(
        index for index, value in enumerate(values) if value >= transition_threshold
    )
    transition_frames = tuple(window_start_frame + index for index in transition_indices)
    offsets = tuple((frame - anchor_frame) / fps for frame in transition_frames)
    near_count = sum(abs(offset) <= near_seconds for offset in offsets)
    broadcast_state = (
        "replay"
        if len(transition_frames) >= minimum_transition_count
        and near_count >= minimum_near_transition_count
        else "unknown"
    )
    return ReplayTransitionEvidence(
        event_id=event_id,
        anchor_frame=anchor_frame,
        window_start_frame=window_start_frame,
        window_end_frame=window_start_frame + len(values) - 1,
        transition_frames=transition_frames,
        transition_offsets_seconds=offsets,
        transition_count=len(transition_frames),
        near_transition_count=near_count,
        maximum_transition_probability=max(values),
        broadcast_state=broadcast_state,
    )


def seal_replay_transition_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return verify_replay_transition_artifact(artifact)


def verify_replay_transition_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    if artifact.get("schema_version") != REPLAY_TRANSITION_EVIDENCE_SCHEMA:
        raise ValueError("unsupported replay transition evidence schema")
    if artifact.get("runtime_consumable") is not False:
        raise ValueError("replay transition screening evidence must remain offline")
    events = artifact.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("replay transition evidence requires events")
    for raw in events:
        if not isinstance(raw, Mapping):
            raise ValueError("replay transition event must be an object")
        evidence = ReplayTransitionEvidence(
            event_id=str(raw.get("event_id") or ""),
            anchor_frame=int(raw.get("anchor_frame", -1)),
            window_start_frame=int(raw.get("window_start_frame", -1)),
            window_end_frame=int(raw.get("window_end_frame", -1)),
            transition_frames=tuple(int(value) for value in raw.get("transition_frames", ())),
            transition_offsets_seconds=tuple(
                float(value) for value in raw.get("transition_offsets_seconds", ())
            ),
            transition_count=int(raw.get("transition_count", -1)),
            near_transition_count=int(raw.get("near_transition_count", -1)),
            maximum_transition_probability=float(
                raw.get("maximum_transition_probability", -1)
            ),
            broadcast_state=str(raw.get("broadcast_state") or ""),
            method=str(raw.get("method") or ""),
        )
        if (
            not evidence.event_id
            or evidence.window_start_frame < 0
            or evidence.anchor_frame < evidence.window_start_frame
            or evidence.window_end_frame < evidence.anchor_frame
            or evidence.transition_count != len(evidence.transition_frames)
            or len(evidence.transition_frames)
            != len(evidence.transition_offsets_seconds)
            or evidence.near_transition_count > evidence.transition_count
            or evidence.broadcast_state not in {"replay", "unknown"}
            or not 0.0 <= evidence.maximum_transition_probability <= 1.0
        ):
            raise ValueError("invalid replay transition event")
    claimed = str(artifact.pop("artifact_sha256", ""))
    if not claimed or claimed != _canonical_sha256(artifact):
        raise ValueError("replay transition artifact hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def replay_transition_payload(evidence: ReplayTransitionEvidence) -> dict[str, Any]:
    payload = asdict(evidence)
    payload["transition_frames"] = list(evidence.transition_frames)
    payload["transition_offsets_seconds"] = list(
        evidence.transition_offsets_seconds
    )
    return payload


def best_logo_recurrence_match(
    *,
    query_rgb: Any,
    timeline_rgb: Any,
    query_seconds: float,
    scan_fps: float,
    exclusion_seconds: float = 30.0,
) -> dict[str, float]:
    """Find a visually repeated full-frame transition outside the local window."""

    import numpy as np

    query = np.asarray(query_rgb, dtype=np.float32)
    timeline = np.asarray(timeline_rgb, dtype=np.float32)
    if (
        query.ndim != 3
        or timeline.ndim != 4
        or timeline.shape[1:] != query.shape
        or query.shape[-1] != 3
        or len(timeline) == 0
        or scan_fps <= 0
        or query_seconds < 0
        or exclusion_seconds <= 0
    ):
        raise ValueError("logo recurrence inputs are invalid")
    query_flat = query.reshape(-1)
    timeline_flat = timeline.reshape(len(timeline), -1)
    query_standard = (query_flat - query_flat.mean()) / (
        query_flat.std() + 1e-6
    )
    timeline_standard = (timeline_flat - timeline_flat.mean(axis=1, keepdims=True)) / (
        timeline_flat.std(axis=1, keepdims=True) + 1e-6
    )
    correlations = timeline_standard @ query_standard / len(query_standard)
    first_excluded = max(0, int((query_seconds - exclusion_seconds) * scan_fps))
    last_excluded = min(
        len(correlations),
        int((query_seconds + exclusion_seconds) * scan_fps) + 1,
    )
    correlations[first_excluded:last_excluded] = -2.0
    selected = int(np.argmax(correlations))
    mean_absolute_difference = float(
        np.mean(np.abs(timeline_flat[selected] - query_flat))
    )
    return {
        "matched_seconds": selected / scan_fps,
        "correlation": float(correlations[selected]),
        "mean_absolute_difference": mean_absolute_difference,
    }


def seal_replay_logo_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return verify_replay_logo_artifact(artifact)


def verify_replay_logo_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    if artifact.get("schema_version") != REPLAY_LOGO_RECURRENCE_SCHEMA:
        raise ValueError("unsupported replay logo recurrence schema")
    if artifact.get("runtime_consumable") is not False:
        raise ValueError("replay logo recurrence evidence must remain offline")
    events = artifact.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("replay logo recurrence evidence requires events")
    for event in events:
        if not isinstance(event, Mapping) or not str(event.get("event_id") or ""):
            raise ValueError("invalid replay logo event")
        candidates = event.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("replay logo event candidates must be a list")
        for candidate in candidates:
            if (
                not isinstance(candidate, Mapping)
                or int(candidate.get("frame", -1)) < 0
                or int(candidate.get("person_count", -1)) < 0
                or not -1.0
                <= float(candidate.get("recurrence_correlation", -2.0))
                <= 1.0
                or float(candidate.get("recurrence_mean_absolute_difference", -1.0))
                < 0.0
            ):
                raise ValueError("invalid replay logo candidate")
        if event.get("broadcast_state") not in {"replay", "unknown"}:
            raise ValueError("invalid replay logo broadcast state")
    claimed = str(artifact.pop("artifact_sha256", ""))
    if not claimed or claimed != _canonical_sha256(artifact):
        raise ValueError("replay logo artifact hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def evaluate_replay_logo_proxy(
    *,
    labels: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    minimum_precision: float = 0.95,
) -> dict[str, Any]:
    """Evaluate the replay-logo proxy on an explicit live/replay dev subset."""

    if (
        not labels
        or len(labels) != len(evidence)
        or not 0.0 < minimum_precision <= 1.0
    ):
        raise ValueError("replay logo evaluation inputs are invalid")
    true_positive = false_positive = false_negative = true_negative = 0
    false_positive_event_ids: list[str] = []
    evaluated_event_keys: set[tuple[str, str]] = set()
    for label_payload, evidence_payload in zip(labels, evidence, strict=True):
        if (
            label_payload.get("schema_version") != "agu.shot-validity-labels.v1"
            or label_payload.get("runtime_consumable") is not False
        ):
            raise ValueError("replay evaluation requires offline shot labels")
        verified = verify_replay_logo_artifact(evidence_payload)
        if (
            label_payload.get("source_video_sha256")
            != verified.get("raw_video_sha256")
            or label_payload.get("candidate_bundle_sha256")
            != verified.get("candidate_bundle_sha256")
        ):
            raise ValueError("replay evidence is not bound to its labels")
        label_by_id = {
            str(row.get("event_id") or ""): row
            for row in label_payload.get("examples", ())
            if isinstance(row, Mapping)
        }
        for event in verified["events"]:
            event_id = str(event["event_id"])
            event_key = (str(verified["raw_video_sha256"]), event_id)
            row = label_by_id.get(event_id)
            if (
                row is None
                or event_key in evaluated_event_keys
                or not isinstance(row.get("event_present"), bool)
            ):
                raise ValueError("replay evidence event is absent or duplicated")
            is_live = bool(row["event_present"])
            if not is_live and not any(
                marker in str(row.get("review_note") or "").lower()
                for marker in ("replay", "highlight")
            ):
                raise ValueError("negative replay subset contains a non-replay window")
            predicted_replay = event["broadcast_state"] == "replay"
            is_replay = not is_live
            true_positive += int(predicted_replay and is_replay)
            false_positive += int(predicted_replay and is_live)
            false_negative += int(not predicted_replay and is_replay)
            true_negative += int(not predicted_replay and is_live)
            if predicted_replay and is_live:
                false_positive_event_ids.append(
                    f"{str(verified['raw_video_sha256'])[:12]}/{event_id}"
                )
            evaluated_event_keys.add(event_key)
    predicted_count = true_positive + false_positive
    replay_count = true_positive + false_negative
    precision = true_positive / predicted_count if predicted_count else 0.0
    recall = true_positive / replay_count if replay_count else 0.0
    return {
        "evaluated_count": len(evaluated_event_keys),
        "live_count": false_positive + true_negative,
        "replay_count": replay_count,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "minimum_precision": minimum_precision,
        "promotion_eligible": predicted_count > 0 and precision >= minimum_precision,
        "false_positive_event_ids": false_positive_event_ids,
    }


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
