"""Label-free shot-clock reset and score-overlay evidence for shot screening."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from app.analysis.broadcast_clock import verify_broadcast_clock_artifact
from app.analysis.broadcast_scoreboard import (
    BroadcastScoreboardRead,
    candidate_component_score_delta,
)
from app.analysis.shot_validity_scene_state import verify_scene_embedding_artifact

OVERLAY_STATE_EVIDENCE_SCHEMA = "agu.shot-overlay-state-evidence.v1"
SCOREBOARD_TIMELINE_SCHEMA = "agu.broadcast-scoreboard-cache.v1"
FEATURE_NAMES = (
    "shot_clock_read_count",
    "shot_clock_read_fraction",
    "pre_anchor_shot_clock_read_count",
    "post_anchor_shot_clock_read_count",
    "distinct_shot_clock_count",
    "shot_clock_adjacent_comparable_count",
    "shot_clock_countdown_fraction",
    "shot_clock_frozen_fraction",
    "shot_clock_reset_count",
    "maximum_shot_clock_increase",
    "shot_clock_reset_flag",
    "post_anchor_24_fraction",
    "reset_after_anchor_flag",
    "first_reset_offset_seconds",
    "score_read_count",
    "before_score_read_count",
    "after_score_read_count",
    "score_change_flag",
    "score_change_points",
    "score_change_delay_seconds",
    "score_change_confidence",
    "score_unresolved_team_count",
)
_FORBIDDEN_LABEL_FIELDS = {
    "event_present",
    "ground_truth",
    "label",
    "review_note",
    "target",
}
_PERIOD = re.compile(r"(?:1ST|1S[T7]|2ND|3RD|4TH|OT)")
_NORMALIZE = re.compile(r"[^A-Z0-9:.]")


def parse_shot_clock_from_raw_text(
    raw_text: str,
    expected_game_clock_seconds: int,
) -> int | None:
    """Parse a plausible suffix only when it follows the verified game clock."""

    if not 0 <= expected_game_clock_seconds <= 12 * 60:
        raise ValueError("expected game clock is invalid")
    normalized = _NORMALIZE.sub("", str(raw_text or "").upper())
    period = _PERIOD.search(normalized)
    if period is None:
        return None
    minutes, seconds = divmod(expected_game_clock_seconds, 60)
    body = normalized[period.end() :]
    separated = re.search(
        rf"(?:^|[^0-9]){minutes}[:.]{seconds:02d}:(\d{{1,2}})$",
        body,
    )
    concatenated = re.search(
        rf"(?:^|[^0-9]){minutes}[:.]{seconds:02d}(\d{{2}})$",
        body,
    )
    match = separated or concatenated
    if match is None:
        return None
    value = int(match.group(1))
    return value if 0 <= value <= 24 else None


def extract_overlay_state_features(
    clock_event: Mapping[str, Any],
    scoreboard_timeline: Mapping[str, Any],
) -> dict[str, float]:
    """Summarize reset and score-change evidence without event labels."""

    samples = clock_event.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("overlay state requires clock samples")
    ordered = sorted(samples, key=lambda row: float(row["offset_seconds"]))
    anchor_frame = int(clock_event.get("anchor_frame", -1))
    fps = _infer_fps(ordered)
    if anchor_frame < 0:
        raise ValueError("overlay state anchor is invalid")

    shot_reads: list[tuple[float, int]] = []
    for sample in ordered:
        read = sample.get("read")
        if not isinstance(read, Mapping):
            continue
        value = parse_shot_clock_from_raw_text(
            str(read.get("raw_text") or ""),
            int(read.get("clock_seconds", -1)),
        )
        if value is not None:
            shot_reads.append((float(sample["offset_seconds"]), value))
    pairs = list(zip(shot_reads, shot_reads[1:], strict=False))
    changes = [second[1] - first[1] for first, second in pairs]
    resets = [
        (second[0], change)
        for (first, second), change in zip(pairs, changes, strict=True)
        if change >= 8 and second[1] >= 14 and second[0] >= first[0]
    ]
    post_values = [value for offset, value in shot_reads if offset >= 0.0]

    timeline = verify_scoreboard_timeline_artifact(scoreboard_timeline)
    before_start = anchor_frame - round(20.0 * fps)
    before_stop = anchor_frame - round(4.0 * fps)
    after_stop = anchor_frame + round(20.0 * fps)
    reads = [_score_read(row) for row in timeline["reads"]]
    before_reads = [
        read for read in reads if before_start <= read.frame <= before_stop
    ]
    after_reads = [
        read for read in reads if anchor_frame <= read.frame <= after_stop
    ]
    delta = candidate_component_score_delta(
        event_id=str(clock_event.get("event_id") or ""),
        before_reads=before_reads,
        after_reads=after_reads,
        minimum_support=2,
    )
    delay = 0.0
    if delta is not None:
        changed_score = int(delta.after_scores[delta.team_id])
        first_changed_frame = min(
            (
                read.frame
                for read in after_reads
                if int(read.scores.get(delta.team_id, -1)) == changed_score
            ),
            default=delta.after_frame,
        )
        delay = max(0.0, (first_changed_frame - anchor_frame) / fps)

    sample_count = len(ordered)
    values = (
        float(len(shot_reads)),
        len(shot_reads) / sample_count,
        float(sum(offset < 0.0 for offset, _ in shot_reads)),
        float(sum(offset >= 0.0 for offset, _ in shot_reads)),
        float(len({value for _, value in shot_reads})),
        float(len(pairs)),
        (
            sum(change < 0 for change in changes) / len(changes)
            if changes
            else 0.0
        ),
        (
            sum(change == 0 for change in changes) / len(changes)
            if changes
            else 0.0
        ),
        float(len(resets)),
        float(max((change for change in changes), default=0)),
        float(bool(resets)),
        (
            sum(value == 24 for value in post_values) / len(post_values)
            if post_values
            else 0.0
        ),
        float(any(offset >= 0.0 for offset, _ in resets)),
        float(resets[0][0] if resets else 0.0),
        float(len(before_reads) + len(after_reads)),
        float(len(before_reads)),
        float(len(after_reads)),
        float(delta is not None),
        float(delta.points if delta is not None else 0),
        delay,
        float(delta.confidence if delta is not None else 0.0),
        float(len(delta.unresolved_team_ids) if delta is not None else 0),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("overlay state features must be finite")
    return dict(zip(FEATURE_NAMES, values, strict=True))


def build_overlay_state_evidence_artifact(
    *,
    scene_artifact: Mapping[str, Any],
    clock_artifacts: Sequence[Mapping[str, Any]],
    scoreboard_timeline_artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Align clock events and one score timeline per video to scene examples."""

    scene = verify_scene_embedding_artifact(scene_artifact)
    scene_rows = {_example_key(row): row for row in scene["examples"]}
    if len(scene_rows) != len(scene["examples"]):
        raise ValueError("scene examples must be unique")

    clock_rows: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    clock_hashes: list[str] = []
    for raw in clock_artifacts:
        artifact = verify_broadcast_clock_artifact(raw)
        video = str(artifact.get("raw_video_sha256") or "")
        bundle = str(artifact.get("candidate_bundle_sha256") or "")
        clock_hashes.append(str(artifact["artifact_sha256"]))
        for event in artifact["events"]:
            key = (video, bundle, str(event["event_id"]))
            if key in clock_rows:
                raise ValueError("clock evidence example is duplicated")
            clock_rows[key] = event
    if set(clock_rows) != set(scene_rows):
        raise ValueError("clock evidence must exactly cover scene examples")

    timelines: dict[str, Mapping[str, Any]] = {}
    timeline_hashes: list[str] = []
    for raw in scoreboard_timeline_artifacts:
        artifact = verify_scoreboard_timeline_artifact(raw)
        video = str(artifact["raw_video_sha256"])
        if video in timelines:
            raise ValueError("scoreboard timeline video is duplicated")
        timelines[video] = artifact
        timeline_hashes.append(str(artifact["artifact_sha256"]))
    scene_videos = {key[0] for key in scene_rows}
    if set(timelines) != scene_videos:
        raise ValueError("scoreboard timelines must exactly cover scene videos")

    examples = []
    for row in scene["examples"]:
        key = _example_key(row)
        feature_map = extract_overlay_state_features(
            clock_rows[key],
            timelines[key[0]],
        )
        examples.append(
            {
                "source_video_sha256": key[0],
                "candidate_bundle_sha256": key[1],
                "event_id": key[2],
                "features": [feature_map[name] for name in FEATURE_NAMES],
            }
        )
    return seal_overlay_state_evidence_artifact(
        {
            "source_scene_artifact_sha256": scene["artifact_sha256"],
            "source_clock_artifact_sha256s": sorted(clock_hashes),
            "source_scoreboard_timeline_sha256s": sorted(timeline_hashes),
            "examples": examples,
        }
    )


def seal_scoreboard_timeline_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["schema_version"] = SCOREBOARD_TIMELINE_SCHEMA
    artifact["read_count"] = len(artifact.get("reads") or ())
    _validate_scoreboard_timeline(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_scoreboard_timeline_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_scoreboard_timeline(artifact)
    if not claimed or claimed != _canonical_sha256(artifact):
        raise ValueError("scoreboard timeline artifact hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def seal_overlay_state_evidence_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["schema_version"] = OVERLAY_STATE_EVIDENCE_SCHEMA
    artifact["purpose"] = "offline_label_free_overlay_state_screening"
    artifact["runtime_consumable"] = False
    artifact["codex_runtime_answer_used"] = False
    artifact["feature_names"] = list(FEATURE_NAMES)
    _validate_overlay_artifact(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_overlay_state_evidence_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_overlay_artifact(artifact)
    if not claimed or claimed != _canonical_sha256(artifact):
        raise ValueError("overlay state evidence artifact hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _validate_scoreboard_timeline(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != SCOREBOARD_TIMELINE_SCHEMA:
        raise ValueError("unsupported scoreboard timeline schema")
    if not artifact.get("raw_video_sha256"):
        raise ValueError("scoreboard timeline requires a video binding")
    team_ids = artifact.get("team_ids")
    reads = artifact.get("reads")
    if (
        not isinstance(team_ids, list)
        or len(team_ids) != 2
        or len(set(team_ids)) != 2
        or not isinstance(reads, list)
    ):
        raise ValueError("scoreboard timeline contract is invalid")
    for row in reads:
        if not isinstance(row, Mapping) or _FORBIDDEN_LABEL_FIELDS.intersection(row):
            raise ValueError("scoreboard timeline reads must be label-free objects")
        _score_read(row)


def _validate_overlay_artifact(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != OVERLAY_STATE_EVIDENCE_SCHEMA:
        raise ValueError("unsupported overlay state evidence schema")
    if (
        artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("overlay state evidence must remain offline and label-free")
    if tuple(artifact.get("feature_names") or ()) != FEATURE_NAMES:
        raise ValueError("overlay state feature contract mismatch")
    if not all(
        artifact.get(key)
        for key in (
            "source_scene_artifact_sha256",
            "source_clock_artifact_sha256s",
            "source_scoreboard_timeline_sha256s",
        )
    ):
        raise ValueError("overlay state evidence requires source provenance")
    examples = artifact.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("overlay state evidence requires examples")
    seen: set[tuple[str, str, str]] = set()
    for row in examples:
        if not isinstance(row, Mapping):
            raise ValueError("overlay state examples must be objects")
        if _FORBIDDEN_LABEL_FIELDS.intersection(row):
            raise ValueError("overlay state examples must not contain labels")
        key = _example_key(row)
        if key in seen:
            raise ValueError("overlay state example keys must be unique")
        seen.add(key)
        features = row.get("features")
        if (
            not isinstance(features, list)
            or len(features) != len(FEATURE_NAMES)
            or not all(math.isfinite(float(value)) for value in features)
        ):
            raise ValueError("overlay state feature vector is invalid")


def _infer_fps(samples: Sequence[Mapping[str, Any]]) -> float:
    estimates = []
    for first, second in zip(samples, samples[1:], strict=False):
        offset_delta = float(second["offset_seconds"]) - float(
            first["offset_seconds"]
        )
        frame_delta = int(second["frame"]) - int(first["frame"])
        if offset_delta > 0.0 and frame_delta > 0:
            estimates.append(frame_delta / offset_delta)
    if not estimates:
        raise ValueError("overlay state cannot infer video FPS")
    estimates.sort()
    fps = estimates[len(estimates) // 2]
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("overlay state inferred FPS is invalid")
    return fps


def _score_read(row: Mapping[str, Any]) -> BroadcastScoreboardRead:
    scores = row.get("scores")
    if (
        int(row.get("frame", -1)) < 0
        or not isinstance(scores, Mapping)
        or len(scores) != 2
        or not 0.0 <= float(row.get("confidence", -1.0)) <= 1.0
    ):
        raise ValueError("scoreboard timeline read is invalid")
    normalized = {str(team): int(score) for team, score in scores.items()}
    if any(score < 0 or score > 250 for score in normalized.values()):
        raise ValueError("scoreboard timeline score is invalid")
    return BroadcastScoreboardRead(
        frame=int(row["frame"]),
        scores=normalized,
        confidence=float(row["confidence"]),
        source=str(row.get("source") or "rapidocr_broadcast_scoreboard_v3"),
    )


def _example_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    key = (
        str(row.get("source_video_sha256") or ""),
        str(row.get("candidate_bundle_sha256") or ""),
        str(row.get("event_id") or ""),
    )
    if not all(key):
        raise ValueError("overlay state example key is incomplete")
    return key


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
