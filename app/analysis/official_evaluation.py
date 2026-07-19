"""Leakage-resistant raw-only prediction sealing and strict event evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from app.analysis.schemas import GameEventResponse, RawOnlyPredictionBundleResponse, RawVideoAssetResponse

AUTOMATIC_CONFIRMED_STATUSES = {"vision_confirmed", "edge_vlm_confirmed"}
REVIEWED_CONFIRMED_STATUSES = AUTOMATIC_CONFIRMED_STATUSES | {"codex_confirmed", "human_confirmed"}
AUTONOMOUS_PRODUCER = "agu"
AUTONOMOUS_INFERENCE_MODE = "traditional_cv+vlm"
FORBIDDEN_AUTONOMOUS_PROVENANCE_MARKERS = (
    "codex",
    "human",
    "ground_truth",
    "reference",
    "highlight",
    "manual",
)
TRAINING_ANNOTATION_PROVENANCE_KEYS = {
    "training_annotation_producer",
    "training_manifest_sha256",
    "training_benchmark_overlap",
}

VIDEO_SUFFIXES = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm"}
REFERENCE_NAME_MARKERS = (
    "highlight",
    "missed_",
    "集锦",
    "明细数据",
    "数据.csv",
    "ground_truth",
    "reference",
)
TRUTH_EVENT_ALIASES = {
    "2分投篮": "field_goal_attempt",
    "3分投篮": "field_goal_attempt",
    "投篮": "field_goal_attempt",
    "shoot": "field_goal_attempt",
    "shot": "field_goal_attempt",
    "篮板": "rebound",
    "rebound": "rebound",
    "抢断": "steal",
    "steal": "steal",
    "盖帽": "block",
    "block": "block",
    "助攻": "assist",
    "assist": "assist",
    "失误": "turnover",
    "turnover": "turnover",
    "犯规": "foul",
    "foul": "foul",
}


class RawOnlyEvaluationError(ValueError):
    pass


@dataclass(frozen=True)
class StrictEvent:
    event_id: str
    event_type: str
    source_video_id: str
    center_frame: int
    primary_player_id: str | None = None
    secondary_player_id: str | None = None
    team_id: str | None = None
    outcome: str | None = None
    shot_value: int | None = None


def seal_raw_only_predictions(
    *,
    game_id: str,
    raw_video_paths: Sequence[str | Path],
    events: Iterable[GameEventResponse],
    config: Mapping[str, Any],
    model_provenance: Mapping[str, str] | None = None,
    completed_at: datetime | None = None,
) -> RawOnlyPredictionBundleResponse:
    """Seal predictions without persisting source paths or truth metadata."""

    if not raw_video_paths:
        raise RawOnlyEvaluationError("at least one raw video is required")
    assets = [_raw_video_asset(index, Path(path)) for index, path in enumerate(raw_video_paths, start=1)]
    event_list = list(events)
    known_video_ids = {asset.video_id for asset in assets}
    unknown_video_ids = sorted({event.source_video_id for event in event_list} - known_video_ids)
    if unknown_video_ids:
        raise RawOnlyEvaluationError(f"events refer to undeclared raw videos: {unknown_video_ids}")

    event_payload = [event.model_dump(mode="json") for event in event_list]
    events_sha256 = _json_sha256(event_payload)
    payload: dict[str, Any] = {
        "schema_version": "agu.raw-only.v1",
        "game_id": game_id,
        "raw_videos": [asset.model_dump(mode="json") for asset in assets],
        "config_sha256": _json_sha256(dict(config)),
        "model_provenance": dict(model_provenance or {}),
        "events": event_payload,
        "inference_completed_at": (completed_at or datetime.now(timezone.utc)).isoformat(),
        "events_sha256": events_sha256,
    }
    payload["bundle_sha256"] = _json_sha256(payload)
    return RawOnlyPredictionBundleResponse.model_validate(payload)


def verify_raw_only_bundle(bundle: RawOnlyPredictionBundleResponse | Mapping[str, Any]) -> RawOnlyPredictionBundleResponse:
    validated = (
        bundle
        if isinstance(bundle, RawOnlyPredictionBundleResponse)
        else RawOnlyPredictionBundleResponse.model_validate(bundle)
    )
    payload = validated.model_dump(mode="json")
    claimed_bundle_hash = payload.pop("bundle_sha256")
    if _json_sha256(payload) != claimed_bundle_hash:
        raise RawOnlyEvaluationError("prediction bundle hash mismatch")
    if _json_sha256(payload["events"]) != validated.events_sha256:
        raise RawOnlyEvaluationError("sealed event hash mismatch")
    if not validated.raw_videos:
        raise RawOnlyEvaluationError("prediction bundle contains no raw videos")
    for asset in validated.raw_videos:
        _validate_raw_filename(asset.filename)
    known_video_ids = {asset.video_id for asset in validated.raw_videos}
    if any(event.source_video_id not in known_video_ids for event in validated.events):
        raise RawOnlyEvaluationError("prediction event references a non-raw input")
    return validated


def verify_agu_autonomous_bundle(
    bundle: RawOnlyPredictionBundleResponse | Mapping[str, Any],
) -> RawOnlyPredictionBundleResponse:
    """Require an AGU-owned prediction bundle with no reviewer/truth leakage.

    Raw-only describes the inputs, not who produced the labels.  This stricter
    gate is the only bundle class eligible for AGU's autonomous 85% acceptance
    metric.  Codex and human-reviewed bundles remain valid annotation artifacts
    but are deliberately rejected here.
    """

    validated = verify_raw_only_bundle(bundle)
    provenance = {str(key).lower(): str(value).lower() for key, value in validated.model_provenance.items()}
    if provenance.get("producer") != AUTONOMOUS_PRODUCER:
        raise RawOnlyEvaluationError("autonomous prediction bundle must declare producer=agu")
    if provenance.get("inference_mode") != AUTONOMOUS_INFERENCE_MODE:
        raise RawOnlyEvaluationError(
            "autonomous prediction bundle must declare inference_mode=traditional_cv+vlm"
        )
    _verify_training_annotation_provenance(provenance)
    joined_provenance = " ".join(
        f"{key}={value}"
        for key, value in provenance.items()
        if key not in TRAINING_ANNOTATION_PROVENANCE_KEYS
    )
    marker = next(
        (item for item in FORBIDDEN_AUTONOMOUS_PROVENANCE_MARKERS if item in joined_provenance),
        None,
    )
    if marker is not None:
        raise RawOnlyEvaluationError(
            f"autonomous prediction provenance contains forbidden reviewer/reference marker: {marker}"
        )
    for event in validated.events:
        if event.status in {"codex_confirmed", "human_confirmed"}:
            raise RawOnlyEvaluationError(
                f"autonomous prediction contains reviewer-confirmed event: {event.event_id}"
            )
        reviewer = (event.reviewer or "").lower()
        if reviewer and not reviewer.startswith("edge_vlm:"):
            raise RawOnlyEvaluationError(
                f"autonomous prediction contains a non-AGU reviewer: {event.event_id}"
            )
        for evidence in event.evidence:
            evidence_text = json.dumps(evidence.model_dump(mode="json"), ensure_ascii=False).lower()
            marker = next(
                (item for item in FORBIDDEN_AUTONOMOUS_PROVENANCE_MARKERS if item in evidence_text),
                None,
            )
            if marker is not None:
                raise RawOnlyEvaluationError(
                    f"autonomous event evidence contains forbidden reviewer/reference marker: {marker}"
                )
    return validated


def _verify_training_annotation_provenance(provenance: Mapping[str, str]) -> None:
    """Allow Codex training labels only behind a benchmark-disjoint manifest.

    Codex may replace manual labor when creating detector/tracker/action-model
    training annotations.  It must never become a runtime reviewer or provide
    answers for an acceptance video.  The latter remains enforced by event,
    evidence and non-training provenance checks in ``verify_agu_autonomous_bundle``.
    """

    producer = provenance.get("training_annotation_producer", "")
    if "codex" not in producer:
        return
    manifest_sha256 = provenance.get("training_manifest_sha256", "")
    if re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is None:
        raise RawOnlyEvaluationError(
            "Codex-assisted training requires a sha256-bound training annotation manifest"
        )
    if provenance.get("training_benchmark_overlap") != "false":
        raise RawOnlyEvaluationError(
            "Codex-assisted training must declare training_benchmark_overlap=false"
        )


def evaluate_strict_events(
    truth: Iterable[StrictEvent],
    predictions: Iterable[StrictEvent],
    *,
    tolerance_frames: int,
) -> dict[str, Any]:
    """One-to-one strict match on type, time, actor and applicable labels."""

    truth_events = list(truth)
    prediction_events = list(predictions)
    assigned = _maximum_cardinality_event_matching(
        truth_events,
        prediction_events,
        compatible=lambda expected, predicted: (
            _same_labels(expected, predicted)
            and predicted.source_video_id == expected.source_video_id
            and abs(predicted.center_frame - expected.center_frame) <= tolerance_frames
        ),
    )
    used = set(assigned.values())
    matches = _matching_report(truth_events, prediction_events, assigned)

    tp = len(used)
    fp = len(prediction_events) - tp
    fn = len(truth_events) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    by_type = _metrics_by_type(truth_events, prediction_events, used, matches)
    return {
        "metrics": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tolerance_frames": tolerance_frames,
            "strict_fields": [
                "event_type",
                "source_video_id",
                "center_frame",
                "primary_player_id",
                "secondary_player_id",
                "team_id",
                "outcome",
                "shot_value",
            ],
        },
        "by_event_type": by_type,
        "matches": matches,
        "unmatched_prediction_ids": [
            event.event_id for index, event in enumerate(prediction_events) if index not in used
        ],
    }


def evaluate_candidate_localization(
    truth: Iterable[StrictEvent],
    predictions: Iterable[StrictEvent],
    *,
    tolerance_frames: int,
) -> dict[str, Any]:
    """Measure event-type/time candidate coverage without claiming actor correctness."""

    truth_events = list(truth)
    prediction_events = list(predictions)
    assigned = _maximum_cardinality_event_matching(
        truth_events,
        prediction_events,
        compatible=lambda expected, predicted: (
            predicted.event_type == expected.event_type
            and predicted.source_video_id == expected.source_video_id
            and abs(predicted.center_frame - expected.center_frame) <= tolerance_frames
        ),
    )
    used = set(assigned.values())
    matches = _matching_report(truth_events, prediction_events, assigned)
    metrics = _summary_metrics(len(used), len(prediction_events) - len(used), len(truth_events) - len(used))
    metrics.update(
        {
            "tolerance_frames": tolerance_frames,
            "match_fields": ["event_type", "source_video_id", "center_frame"],
            "not_measured": [
                "primary_player_id",
                "secondary_player_id",
                "team_id",
                "outcome",
                "shot_value",
            ],
        }
    )
    return {
        "metrics": metrics,
        "matches": matches,
        "unmatched_prediction_ids": [
            event.event_id for index, event in enumerate(prediction_events) if index not in used
        ],
    }


def evaluate_official_tiers(
    truth: Iterable[StrictEvent],
    events: Iterable[GameEventResponse],
    *,
    tolerance_frames: int,
) -> dict[str, Any]:
    """Keep candidate recall, automatic precision and reviewed strict F1 separate."""

    truth_events = list(truth)
    game_events = [event for event in events if event.status != "rejected"]
    candidate_events = strict_events_from_game_events(game_events)
    automatic_events = strict_events_from_game_events(
        event for event in game_events if event.status in AUTOMATIC_CONFIRMED_STATUSES
    )
    reviewed_events = strict_events_from_game_events(
        event for event in game_events if event.status in REVIEWED_CONFIRMED_STATUSES
    )
    return {
        "candidate_localization": evaluate_candidate_localization(
            truth_events,
            candidate_events,
            tolerance_frames=tolerance_frames,
        ),
        "automatic_strict": evaluate_strict_events(
            truth_events,
            automatic_events,
            tolerance_frames=tolerance_frames,
        ),
        "reviewed_strict": evaluate_strict_events(
            truth_events,
            reviewed_events,
            tolerance_frames=tolerance_frames,
        ),
    }


def evaluate_identity_aligned_strict_events(
    truth: Iterable[StrictEvent],
    predictions: Iterable[StrictEvent],
    *,
    player_id_map: Mapping[str, str],
    team_id_map: Mapping[str, str],
    tolerance_frames: int,
) -> dict[str, Any]:
    """Evaluate anonymous raw-video identities with a frozen global one-to-one map.

    The alignment is evaluation-only. It may rename stable predicted clusters,
    but cannot change event type, time, outcome, value or relations. Metrics
    retain the mapping hash and unmapped IDs for audit.
    """

    _validate_one_to_one_map(player_id_map, "player")
    _validate_one_to_one_map(team_id_map, "team")
    prediction_events = list(predictions)
    unmapped_players = sorted(
        {
            player_id
            for event in prediction_events
            for player_id in (event.primary_player_id, event.secondary_player_id)
            if player_id is not None and player_id not in player_id_map
        }
    )
    unmapped_teams = sorted(
        {
            event.team_id
            for event in prediction_events
            if event.team_id is not None and event.team_id not in team_id_map
        }
    )
    aligned = [
        replace(
            event,
            primary_player_id=_aligned_id(event.primary_player_id, player_id_map, "player"),
            secondary_player_id=_aligned_id(event.secondary_player_id, player_id_map, "player"),
            team_id=_aligned_id(event.team_id, team_id_map, "team"),
        )
        for event in prediction_events
    ]
    result = evaluate_strict_events(truth, aligned, tolerance_frames=tolerance_frames)
    result["identity_alignment"] = {
        "schema_version": "agu.identity-alignment.v1",
        "sha256": _json_sha256(
            {
                "player_id_map": dict(sorted(player_id_map.items())),
                "team_id_map": dict(sorted(team_id_map.items())),
            }
        ),
        "player_mapping_count": len(player_id_map),
        "team_mapping_count": len(team_id_map),
        "unmapped_prediction_player_ids": unmapped_players,
        "unmapped_prediction_team_ids": unmapped_teams,
        "constraints": [
            "global one-to-one mapping",
            "evaluation-only renaming",
            "no event labels or timestamps changed",
        ],
    }
    return result


def strict_events_from_game_events(events: Iterable[GameEventResponse]) -> list[StrictEvent]:
    return [
        StrictEvent(
            event_id=event.event_id,
            event_type=event.event_type,
            source_video_id=event.source_video_id,
            center_frame=_canonical_event_frame(event),
            primary_player_id=event.primary_player_id,
            secondary_player_id=event.secondary_player_id,
            team_id=event.team_id,
            outcome=event.outcome,
            shot_value=event.shot_value,
        )
        for event in events
        if event.status != "rejected"
    ]


def _canonical_event_frame(event: GameEventResponse) -> int:
    if event.outcome_frame is not None:
        return event.outcome_frame
    for evidence in event.evidence:
        value = evidence.details.get("candidate_event_frame")
        try:
            frame = int(value)
        except (TypeError, ValueError):
            continue
        if event.start_frame <= frame <= event.end_frame:
            return frame
    return (event.start_frame + event.end_frame) // 2


def load_strict_truth_csv(
    path: str | Path,
    *,
    source_video_id: str,
    fps: float,
) -> list[StrictEvent]:
    rows: list[dict[str, str]] | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            with Path(path).open("r", encoding=encoding, newline="") as handle:
                rows = list(csv.DictReader(handle))
            break
        except UnicodeDecodeError:
            continue
    if rows is None:
        raise RawOnlyEvaluationError(f"unable to decode truth CSV: {path}")
    events: list[StrictEvent] = []
    for index, row in enumerate(rows, start=1):
        raw_type = str(row.get("event_type") or row.get("type") or row.get("action") or "").strip()
        event_type = TRUTH_EVENT_ALIASES.get(raw_type, raw_type)
        if not event_type:
            continue
        start_sec = float(row.get("start_sec") or row.get("time_sec") or 0)
        end_sec = float(row.get("end_sec") or start_sec)
        center_frame = int(round((start_sec + end_sec) / 2.0 * fps))
        raw_outcome = str(row.get("shot_result") or row.get("outcome") or "").strip()
        outcome = {"命中": "made", "未命中": "missed", "made": "made", "missed": "missed"}.get(raw_outcome)
        shot_value_raw = str(row.get("shot_value") or "").strip()
        shot_value = int(float(shot_value_raw)) if shot_value_raw else None
        events.append(
            StrictEvent(
                event_id=str(row.get("event_id") or f"event_{index:04d}"),
                event_type=event_type,
                source_video_id=source_video_id,
                center_frame=center_frame,
                primary_player_id=str(row.get("player_id") or "").strip() or None,
                secondary_player_id=str(row.get("related_player_id") or "").strip() or None,
                team_id=str(row.get("team_id") or "").strip() or None,
                outcome=outcome,
                shot_value=shot_value,
            )
        )
    return events


def strict_events_from_legacy_analysis(analysis: Mapping[str, Any]) -> list[StrictEvent]:
    """Expose the legacy candidate baseline without promoting it to official output."""

    events: list[StrictEvent] = []
    for index, record in enumerate(analysis.get("records") or [], start=1):
        final = record.get("final") or {}
        raw_type = str(final.get("action") or "")
        event_type = TRUTH_EVENT_ALIASES.get(raw_type)
        if event_type != "field_goal_attempt":
            continue
        events.append(
            StrictEvent(
                event_id=f"legacy_record_{index:06d}",
                event_type=event_type,
                source_video_id="video_001",
                center_frame=(int(record.get("start_frame") or 0) + int(record.get("end_frame") or 0)) // 2,
                primary_player_id=record.get("global_player_id") or record.get("local_player_id"),
            )
        )
    for index, event in enumerate((analysis.get("long_video") or {}).get("event_candidates") or [], start=1):
        raw_type = str(event.get("event_type") or "").replace("_candidate", "")
        event_type = TRUTH_EVENT_ALIASES.get(raw_type)
        if event_type not in {"rebound", "steal", "block"}:
            continue
        events.append(
            StrictEvent(
                event_id=f"legacy_candidate_{index:06d}",
                event_type=event_type,
                source_video_id="video_001",
                center_frame=(int(event.get("start_frame") or 0) + int(event.get("end_frame") or 0)) // 2,
                primary_player_id=event.get("player_id"),
            )
        )
    return events


def _raw_video_asset(index: int, path: Path) -> RawVideoAssetResponse:
    if not path.is_file():
        raise RawOnlyEvaluationError(f"raw video does not exist: {path}")
    _validate_raw_filename(path.name)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return RawVideoAssetResponse(
        video_id=f"video_{index:03d}",
        filename=path.name,
        sha256=digest.hexdigest(),
        size_bytes=path.stat().st_size,
    )


def _validate_raw_filename(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise RawOnlyEvaluationError(f"non-video input is forbidden in raw-only inference: {filename}")
    lowered = filename.lower()
    if any(marker.lower() in lowered for marker in REFERENCE_NAME_MARKERS):
        raise RawOnlyEvaluationError(f"reference/edited-video marker is forbidden in raw-only inference: {filename}")


def _same_labels(expected: StrictEvent, predicted: StrictEvent) -> bool:
    fields = (
        "event_type",
        "primary_player_id",
        "secondary_player_id",
        "team_id",
        "outcome",
        "shot_value",
    )
    return all(getattr(expected, field) == getattr(predicted, field) for field in fields)


def _maximum_cardinality_event_matching(
    truth: Sequence[StrictEvent],
    predictions: Sequence[StrictEvent],
    *,
    compatible: Callable[[StrictEvent, StrictEvent], bool],
) -> dict[int, int]:
    """Return deterministic maximum-cardinality one-to-one event matches.

    Candidate edges are visited by frame error, but an augmenting path may
    reassign an earlier truth event so a later event is not incorrectly lost
    by greedy nearest-neighbour matching.
    """

    adjacency = [
        sorted(
            (index for index, predicted in enumerate(predictions) if compatible(expected, predicted)),
            key=lambda index: (
                abs(predictions[index].center_frame - expected.center_frame),
                predictions[index].event_id,
                index,
            ),
        )
        for expected in truth
    ]
    truth_by_prediction: dict[int, int] = {}

    def augment(truth_index: int, visited_predictions: set[int]) -> bool:
        for prediction_index in adjacency[truth_index]:
            if prediction_index in visited_predictions:
                continue
            visited_predictions.add(prediction_index)
            previous_truth = truth_by_prediction.get(prediction_index)
            if previous_truth is None or augment(previous_truth, visited_predictions):
                truth_by_prediction[prediction_index] = truth_index
                return True
        return False

    for truth_index in range(len(truth)):
        augment(truth_index, set())
    return {truth_index: prediction_index for prediction_index, truth_index in truth_by_prediction.items()}


def _matching_report(
    truth: Sequence[StrictEvent],
    predictions: Sequence[StrictEvent],
    assigned: Mapping[int, int],
) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for truth_index, expected in enumerate(truth):
        prediction_index = assigned.get(truth_index)
        if prediction_index is None:
            report.append({"event_id": expected.event_id, "status": "fn"})
            continue
        predicted = predictions[prediction_index]
        report.append(
            {
                "event_id": expected.event_id,
                "prediction_event_id": predicted.event_id,
                "status": "tp",
                "frame_error": abs(predicted.center_frame - expected.center_frame),
            }
        )
    return report


def _metrics_by_type(
    truth: list[StrictEvent],
    predictions: list[StrictEvent],
    used_predictions: set[int],
    matches: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    matched_truth_ids = {match["event_id"] for match in matches if match["status"] == "tp"}
    event_types = sorted({event.event_type for event in truth + predictions})
    result: dict[str, dict[str, float | int]] = {}
    for event_type in event_types:
        tp = sum(1 for event in truth if event.event_type == event_type and event.event_id in matched_truth_ids)
        fn = sum(1 for event in truth if event.event_type == event_type) - tp
        fp = sum(
            1
            for index, event in enumerate(predictions)
            if event.event_type == event_type and index not in used_predictions
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[event_type] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
    return result


def _summary_metrics(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _validate_one_to_one_map(mapping: Mapping[str, str], kind: str) -> None:
    if any(not source or not target for source, target in mapping.items()):
        raise RawOnlyEvaluationError(f"{kind} identity alignment contains an empty id")
    targets = list(mapping.values())
    if len(set(targets)) != len(targets):
        raise RawOnlyEvaluationError(f"{kind} identity alignment must be one-to-one")


def _aligned_id(value: str | None, mapping: Mapping[str, str], kind: str) -> str | None:
    if value is None:
        return None
    return mapping.get(value, f"__unmapped_prediction_{kind}__:{value}")


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
