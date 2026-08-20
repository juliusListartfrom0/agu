"""Offline, label-hidden review queues derived from official PBP alignment.

The queue is deliberately a one-way view of an alignment artifact: it keeps only
video geometry and an opaque join key, while the official play-by-play fields are
written to a separate post-freeze truth artifact.  Nothing in this module is
runtime- or training-consumable.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from app.analysis.pbp_clock_event_alignment import (
    verify_pbp_clock_event_alignment_artifact,
)


QUEUE_SCHEMA = "agu.nba-games-pbp-label-hidden-review-queue.v1"
TRUTH_SCHEMA = "agu.nba-games-pbp-label-hidden-truth.v1"
REVIEW_BATCH_SCHEMA = "agu.nba-games-pbp-label-hidden-review-batch.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_QUEUE_KEYS = {
    "period",
    "clock",
    "action_type",
    "sub_type",
    "description",
    "shot_result",
    "shot_value",
    "is_field_goal",
    "player_name",
    "person_id",
    "team_id",
    "score_home",
    "score_away",
    "score_state",
    "score_transition",
}


def build_label_hidden_queue(
    *,
    alignment: Mapping[str, Any],
    game_slug: str,
    video_filename: str,
    video_sha256: str,
    fps: float,
    frame_count: int,
    width: int,
    height: int,
    window_before_seconds: float = 3.0,
    window_after_seconds: float = 4.0,
    maximum_events: int | None = None,
) -> dict[str, Any]:
    """Build a chronological queue from mapped field-goal rows only.

    Selection is intentionally based on the PBP alignment because this is a
    training/manual-review aid, not an independent benchmark.  Labels remain in
    ``build_post_freeze_truth`` and never appear in the queue payload.
    """

    verified = verify_pbp_clock_event_alignment_artifact(alignment)
    if not game_slug.strip() or not video_filename.strip():
        raise ValueError("game_slug and video_filename are required")
    if not _SHA256_RE.fullmatch(video_sha256):
        raise ValueError("video_sha256 must be a lowercase SHA-256")
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError("video geometry is invalid")
    if window_before_seconds < 0 or window_after_seconds <= 0:
        raise ValueError("review window durations are invalid")
    if maximum_events is not None and maximum_events <= 0:
        raise ValueError("maximum_events must be positive")
    if video_sha256 != str(verified["raw_video_sha256"]):
        raise ValueError("video_sha256 does not match the alignment")

    candidates = [
        event
        for event in verified["events"]
        if bool(event.get("is_field_goal")) and event.get("video_frame") is not None
    ]
    candidates.sort(key=lambda event: (int(event["video_frame"]), int(event["source_index"])))
    if maximum_events is not None and len(candidates) > maximum_events:
        candidates = _sample_evenly(candidates, maximum_events)
        candidates.sort(key=lambda event: (int(event["video_frame"]), int(event["source_index"])))
    before_frames = round(window_before_seconds * fps)
    after_frames = round(window_after_seconds * fps)
    queue: list[dict[str, Any]] = []
    for ordinal, event in enumerate(candidates, 1):
        anchor = int(event["video_frame"])
        start = max(0, anchor - before_frames)
        end = min(frame_count - 1, anchor + after_frames)
        if not start <= anchor <= end:
            raise ValueError("review window does not contain the anchor")
        queue.append(
            {
                "review_id": f"{game_slug}-review-{ordinal:03d}",
                "video_filename": video_filename,
                "video_sha256": video_sha256,
                "start_frame": start,
                "end_frame": end,
                "anchor_frame": anchor,
                "anchor_video_time_seconds": round(anchor / fps, 3),
                "alignment_status": "clock_read",
                "frame_uncertainty_frames": event.get("frame_uncertainty_frames"),
                "truth_join_key": f"{verified['game_id']}:{int(event['source_index'])}",
                "label_hidden": True,
                "review_fields": [
                    "event_present",
                    "release_frame",
                    "shot_type",
                    "outcome",
                    "player_identity",
                    "team_identity",
                    "notes",
                ],
                "review_note": "Official PBP truth withheld during visual review; offline-only.",
            }
        )

    if not queue:
        raise ValueError("alignment has no mapped field-goal rows")
    artifact = {
        "schema_version": QUEUE_SCHEMA,
        "generated_on": "2026-08-10",
        "purpose": "offline_manual_frame_causal_review_queue_from_official_pbp_clock_alignment",
        "runtime_consumable": False,
        "training_consumable": False,
        "independent_evaluation_eligible": False,
        "codex_runtime_answer_used": False,
        "labels_are_visual_causal_truth": False,
        "selection_policy": "all mapped field-goal PBP rows; PBP labels withheld from queue",
        "game_slug": game_slug,
        "game_id": str(verified["game_id"]),
        "raw_video": {
            "filename": video_filename,
            "sha256": video_sha256,
            "fps": float(fps),
            "frame_count": int(frame_count),
            "width": int(width),
            "height": int(height),
        },
        "source_alignment": {
            "path": "",
            "sha256": str(verified["artifact_sha256"]),
        },
        "review_window_seconds": {
            "before": float(window_before_seconds),
            "after": float(window_after_seconds),
        },
        "queue_count": len(queue),
        "queue": queue,
    }
    return seal_queue_artifact(artifact)


def build_post_freeze_truth(
    *,
    alignment: Mapping[str, Any],
    queue: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the official rows needed to join a reviewed queue after freeze."""

    verified_alignment = verify_pbp_clock_event_alignment_artifact(alignment)
    verified_queue = verify_queue_artifact(queue)
    if verified_queue["source_alignment"]["sha256"] != verified_alignment["artifact_sha256"]:
        raise ValueError("queue and alignment hashes do not match")
    by_key = {
        f"{verified_alignment['game_id']}:{int(event['source_index'])}": event
        for event in verified_alignment["events"]
    }
    truth: list[dict[str, Any]] = []
    for row in verified_queue["queue"]:
        key = str(row["truth_join_key"])
        event = by_key.get(key)
        if event is None:
            raise ValueError(f"queue truth join key is absent: {key}")
        truth.append(
            {
                "truth_join_key": key,
                "review_id": row["review_id"],
                "source_index": int(event["source_index"]),
                "period": int(event["period"]),
                "clock": str(event["clock"]),
                "team_id": str(event["team_id"]),
                "person_id": str(event["person_id"]),
                "player_name": str(event["player_name"]),
                "action_type": str(event["action_type"]),
                "sub_type": str(event["sub_type"]),
                "description": str(event["description"]),
                "shot_result": str(event["shot_result"]),
                "shot_value": event.get("shot_value"),
                "is_field_goal": bool(event["is_field_goal"]),
                "official_video_frame": int(event["video_frame"]),
                "official_clock_delta_seconds": event.get("clock_delta_seconds"),
            }
        )
    artifact = {
        "schema_version": TRUTH_SCHEMA,
        "generated_on": "2026-08-10",
        "purpose": "post_freeze_offline_truth_join_for_label_hidden_visual_review",
        "runtime_consumable": False,
        "training_consumable": False,
        "independent_evaluation_eligible": False,
        "codex_runtime_answer_used": False,
        "truth_access_policy": "post_freeze_evaluation_only",
        "source_alignment_sha256": verified_alignment["artifact_sha256"],
        "source_queue_sha256": verified_queue["artifact_sha256"],
        "truth_count": len(truth),
        "truth": truth,
    }
    return seal_truth_artifact(artifact)


def select_review_batch(
    queue: Mapping[str, Any],
    *,
    count: int,
    seed: int = 20260810,
) -> dict[str, Any]:
    """Select a label-free, chronological visual-review subset."""

    verified = verify_queue_artifact(queue)
    if count <= 0:
        raise ValueError("count must be positive")
    rows = list(verified["queue"])
    if count > len(rows):
        raise ValueError("count exceeds queue size")
    selected = _sample_evenly(rows, count, seed=seed)
    selected.sort(key=lambda row: (int(row["anchor_frame"]), str(row["review_id"])))
    artifact = {
        "schema_version": REVIEW_BATCH_SCHEMA,
        "generated_on": "2026-08-10",
        "purpose": "offline_manual_visual_review_subset_label_hidden",
        "runtime_consumable": False,
        "training_consumable": False,
        "independent_evaluation_eligible": False,
        "codex_runtime_answer_used": False,
        "source_queue_sha256": verified["artifact_sha256"],
        "selection_policy": "deterministic label-free chronological stratification",
        "selection_seed": int(seed),
        "batch_count": len(selected),
        "batch": selected,
    }
    return seal_review_batch_artifact(artifact)


def verify_queue_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    if (
        artifact.get("schema_version") != QUEUE_SCHEMA
        or artifact.get("runtime_consumable") is not False
        or artifact.get("training_consumable") is not False
        or artifact.get("independent_evaluation_eligible") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("label_hidden") is not None
    ):
        # The artifact-level label_hidden flag is intentionally absent; only rows
        # carry it, making accidental bulk truth insertion easier to detect.
        raise ValueError("label-hidden queue policy is invalid")
    raw = artifact.get("raw_video")
    source = artifact.get("source_alignment")
    rows = artifact.get("queue")
    if not isinstance(raw, Mapping) or not isinstance(source, Mapping) or not isinstance(rows, list) or not rows:
        raise ValueError("label-hidden queue payload is incomplete")
    if not _SHA256_RE.fullmatch(str(raw.get("sha256") or "")):
        raise ValueError("queue raw video hash is invalid")
    if not str(source.get("sha256") or "").strip():
        raise ValueError("queue source alignment hash is required")
    if int(artifact.get("queue_count", -1)) != len(rows):
        raise ValueError("queue count mismatch")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("queue rows must be objects")
        review_id = str(row.get("review_id") or "")
        if not review_id or review_id in seen:
            raise ValueError("queue review IDs must be unique")
        seen.add(review_id)
        if row.get("label_hidden") is not True:
            raise ValueError("queue row is not label-hidden")
        if any(key in row for key in _FORBIDDEN_QUEUE_KEYS):
            raise ValueError("queue row leaks official PBP labels")
        start = int(row.get("start_frame", -1))
        end = int(row.get("end_frame", -1))
        anchor = int(row.get("anchor_frame", -1))
        if start < 0 or start > anchor or anchor > end or end >= int(raw["frame_count"]):
            raise ValueError("queue frame geometry is invalid")
        if str(row.get("video_sha256")) != str(raw["sha256"]):
            raise ValueError("queue row video hash does not match raw video")
    claimed = str(artifact.pop("artifact_sha256", ""))
    if not claimed or claimed != _canonical_sha256(artifact):
        raise ValueError("label-hidden queue hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def verify_truth_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    if (
        artifact.get("schema_version") != TRUTH_SCHEMA
        or artifact.get("runtime_consumable") is not False
        or artifact.get("training_consumable") is not False
        or artifact.get("independent_evaluation_eligible") is not False
        or artifact.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("post-freeze truth policy is invalid")
    truth = artifact.get("truth")
    if not isinstance(truth, list) or int(artifact.get("truth_count", -1)) != len(truth):
        raise ValueError("truth payload is incomplete")
    claimed = str(artifact.pop("artifact_sha256", ""))
    if not claimed or claimed != _canonical_sha256(artifact):
        raise ValueError("post-freeze truth hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def verify_review_batch_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    if (
        artifact.get("schema_version") != REVIEW_BATCH_SCHEMA
        or artifact.get("runtime_consumable") is not False
        or artifact.get("training_consumable") is not False
        or artifact.get("independent_evaluation_eligible") is not False
        or artifact.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("review batch policy is invalid")
    rows = artifact.get("batch")
    if not isinstance(rows, list) or int(artifact.get("batch_count", -1)) != len(rows) or not rows:
        raise ValueError("review batch is incomplete")
    for row in rows:
        if not isinstance(row, Mapping) or any(key in row for key in _FORBIDDEN_QUEUE_KEYS):
            raise ValueError("review batch leaks official PBP labels")
    claimed = str(artifact.pop("artifact_sha256", ""))
    if not claimed or claimed != _canonical_sha256(artifact):
        raise ValueError("review batch hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def seal_queue_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return verify_queue_artifact(artifact)


def seal_truth_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return verify_truth_artifact(artifact)


def seal_review_batch_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return verify_review_batch_artifact(artifact)


def _sample_evenly(rows: Sequence[Mapping[str, Any]], count: int, *, seed: int | None = None) -> list[dict[str, Any]]:
    if count >= len(rows):
        return [dict(row) for row in rows]
    # A deterministic evenly spaced sample avoids exposing shot outcomes through
    # a seeded random choice while still covering all four periods.
    selected: list[dict[str, Any]] = []
    for index in range(count):
        source_index = min(len(rows) - 1, ((2 * index + 1) * len(rows)) // (2 * count))
        selected.append(dict(rows[source_index]))
    return selected


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
