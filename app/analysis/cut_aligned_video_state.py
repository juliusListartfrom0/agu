"""Label-hidden cut-aligned dense-video state research contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

from app.analysis.replay_transition import verify_replay_transition_artifact

CUT_ALIGNED_SEGMENT_ROLES = ("previous", "anchor", "next")
CUT_ALIGNED_VIDEO_EMBEDDING_SCHEMA = "agu.cut-aligned-video-embeddings.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def build_cut_aligned_clip_plan(
    *,
    review_examples: Sequence[Mapping[str, Any]],
    transition_artifact: Mapping[str, Any],
    sealed_blind_video_sha256s: Sequence[str],
    clip_frames: int = 16,
    context_radius_seconds: float = 8.0,
) -> list[dict[str, Any]]:
    """Create dense clip indexes from cut geometry without transition semantics."""

    if not review_examples:
        raise ValueError("cut-aligned planning requires review examples")
    if clip_frames < 16:
        raise ValueError("cut-aligned clips require at least 16 frames")
    if (
        not math.isfinite(float(context_radius_seconds))
        or float(context_radius_seconds) <= 0
    ):
        raise ValueError("context radius must be positive")

    artifact = verify_replay_transition_artifact(transition_artifact)
    if artifact.get("codex_runtime_answer_used") is not False:
        raise ValueError("cut-aligned planning rejects Codex runtime answers")
    source = _require_sha256(
        artifact.get("raw_video_sha256"),
        field="transition source video",
    )
    bundle = _require_sha256(
        artifact.get("candidate_bundle_sha256"),
        field="transition candidate bundle",
    )
    sealed = {
        _require_sha256(value, field="sealed blind video")
        for value in sealed_blind_video_sha256s
    }
    if source in sealed:
        raise ValueError("cut-aligned planning rejects sealed blind source")

    transitions = {}
    for event in artifact["events"]:
        event_id = str(event.get("event_id") or "")
        if not event_id or event_id in transitions:
            raise ValueError("duplicate or missing transition event")
        transitions[event_id] = [
            float(value) for value in event.get("transition_offsets_seconds") or ()
        ]

    indexed_review = {}
    for raw in review_examples:
        row = dict(raw)
        row_source = _require_sha256(
            row.get("source_video_sha256"),
            field="review source video",
        )
        row_bundle = _require_sha256(
            row.get("candidate_bundle_sha256"),
            field="review candidate bundle",
        )
        event_id = str(row.get("event_id") or "")
        if (
            row_source != source
            or row_bundle != bundle
            or not event_id
            or event_id in indexed_review
        ):
            raise ValueError("review examples do not match transition provenance")
        indexed_review[event_id] = row
    if set(indexed_review) != set(transitions):
        raise ValueError("cut-aligned plan requires exact transition coverage")

    planned = []
    for event_id in sorted(indexed_review):
        row = indexed_review[event_id]
        fps = float(row.get("source_fps", 0.0))
        anchor_frame = int(row.get("anchor_frame", -1))
        frame_count = int(row.get("frame_count", 0))
        filename = str(row.get("source_video_filename") or "")
        if (
            not filename
            or not math.isfinite(fps)
            or fps <= 0
            or anchor_frame < 0
            or frame_count <= anchor_frame
        ):
            raise ValueError("review example has invalid video geometry")
        intervals = _neighbor_intervals(
            transitions[event_id],
            radius=float(context_radius_seconds),
        )
        segments = []
        for role, interval in zip(
            CUT_ALIGNED_SEGMENT_ROLES,
            intervals,
            strict=True,
        ):
            available = interval is not None
            start, end = interval if interval is not None else (0.0, 0.0)
            segments.append(
                {
                    "role": role,
                    "available": available,
                    "start_offset_seconds": start,
                    "end_offset_seconds": end,
                    "frame_indexes": (
                        _interval_frame_indexes(
                            anchor_frame=anchor_frame,
                            fps=fps,
                            frame_count=frame_count,
                            start_offset_seconds=start,
                            end_offset_seconds=end,
                            clip_frames=clip_frames,
                        )
                        if available
                        else []
                    ),
                }
            )
        planned.append(
            {
                "source_video_sha256": source,
                "source_video_filename": filename,
                "candidate_bundle_sha256": bundle,
                "event_id": event_id,
                "anchor_frame": anchor_frame,
                "source_fps": fps,
                "frame_count": frame_count,
                "segments": segments,
            }
        )
    return planned


def seal_cut_aligned_video_embedding_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal a possibly resumable, training-only cut-aligned embedding artifact."""

    artifact = dict(payload)
    artifact["schema_version"] = CUT_ALIGNED_VIDEO_EMBEDDING_SCHEMA
    artifact.pop("artifact_sha256", None)
    _validate_embedding_artifact(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_cut_aligned_video_embedding_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_embedding_artifact(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("cut-aligned video embedding hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def extract_cut_aligned_video_embeddings(
    *,
    video_path: str | Path,
    planned_examples: Sequence[Mapping[str, Any]],
    backbone: nn.Module,
    transform: Any,
    device: torch.device,
    batch_size: int = 1,
) -> list[dict[str, Any]]:
    """Encode every available cut-bounded clip and retain missing-role masks."""

    if not planned_examples or batch_size < 1:
        raise ValueError("cut-aligned extraction requires examples and a positive batch")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open source video: {Path(video_path).name}")
    examples = copy.deepcopy(list(planned_examples))
    tensors: list[torch.Tensor] = []
    segment_refs: list[dict[str, Any]] = []

    def flush() -> None:
        if not tensors:
            return
        with torch.inference_mode():
            values = backbone(torch.stack(tensors).to(device))
        if values.ndim != 2 or values.shape[0] != len(segment_refs):
            raise ValueError("cut-aligned backbone returned an invalid embedding batch")
        for segment, embedding in zip(
            segment_refs,
            values.detach().cpu().to(torch.float32).tolist(),
            strict=True,
        ):
            segment["embedding"] = embedding
        tensors.clear()
        segment_refs.clear()

    try:
        for example in examples:
            for segment in example["segments"]:
                if not segment["available"]:
                    segment["embedding"] = None
                    continue
                frames = []
                for frame_index in segment["frame_indexes"]:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        raise ValueError(f"cannot decode source frame {frame_index}")
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                clip = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
                tensors.append(transform(clip))
                segment_refs.append(segment)
                if len(tensors) >= batch_size:
                    flush()
    finally:
        capture.release()
    flush()
    return examples


def _neighbor_intervals(
    offsets: Sequence[float],
    *,
    radius: float,
) -> tuple[
    tuple[float, float] | None,
    tuple[float, float],
    tuple[float, float] | None,
]:
    cuts = sorted(
        {
            max(-radius, min(radius, float(value)))
            for value in offsets
            if math.isfinite(float(value)) and -radius < float(value) < radius
        }
    )
    boundaries = [-radius, *cuts, radius]
    anchor_index = int(np.searchsorted(cuts, 0.0, side="right"))
    anchor = (boundaries[anchor_index], boundaries[anchor_index + 1])
    previous = (
        (boundaries[anchor_index - 1], boundaries[anchor_index])
        if anchor_index > 0
        else None
    )
    next_interval = (
        (boundaries[anchor_index + 1], boundaries[anchor_index + 2])
        if anchor_index + 2 < len(boundaries)
        else None
    )
    return previous, anchor, next_interval


def _interval_frame_indexes(
    *,
    anchor_frame: int,
    fps: float,
    frame_count: int,
    start_offset_seconds: float,
    end_offset_seconds: float,
    clip_frames: int,
) -> list[int]:
    start = max(0, min(frame_count - 1, round(anchor_frame + start_offset_seconds * fps)))
    end = max(0, min(frame_count - 1, round(anchor_frame + end_offset_seconds * fps)))
    if end < start:
        start, end = end, start
    return [
        int(value)
        for value in np.rint(np.linspace(start, end, clip_frames)).astype(np.int64)
    ]


def _validate_embedding_artifact(artifact: Mapping[str, Any]) -> None:
    if (
        artifact.get("schema_version") != CUT_ALIGNED_VIDEO_EMBEDDING_SCHEMA
        or artifact.get("purpose") != "cut_aligned_visual_state_training_only"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("transition_semantics_used") is not False
        or artifact.get("truth_used_for_training_only") is not False
    ):
        raise ValueError("invalid cut-aligned video embedding artifact")
    for field in (
        "review_plan_sha256",
        "transition_artifact_sha256",
        "raw_video_sha256",
        "candidate_bundle_sha256",
        "backbone_sha256",
    ):
        _require_sha256(artifact.get(field), field=field.replace("_", " "))
    if not str(artifact.get("backbone") or ""):
        raise ValueError("cut-aligned embedding backbone is required")
    dimension = int(artifact.get("embedding_dimension", 0))
    clip_frames = int(artifact.get("clip_frames", 0))
    if dimension < 1 or clip_frames < 16:
        raise ValueError("invalid cut-aligned embedding shape")

    expected = artifact.get("expected_event_ids")
    examples = artifact.get("examples")
    if (
        not isinstance(expected, list)
        or not expected
        or expected != sorted(set(str(value) for value in expected))
        or not isinstance(examples, list)
    ):
        raise ValueError("invalid cut-aligned event coverage")
    seen = set()
    for example in examples:
        if not isinstance(example, Mapping):
            raise ValueError("cut-aligned example must be an object")
        source = _require_sha256(
            example.get("source_video_sha256"),
            field="example source video",
        )
        bundle = _require_sha256(
            example.get("candidate_bundle_sha256"),
            field="example candidate bundle",
        )
        event_id = str(example.get("event_id") or "")
        if (
            source != artifact["raw_video_sha256"]
            or bundle != artifact["candidate_bundle_sha256"]
            or event_id not in expected
            or event_id in seen
        ):
            raise ValueError("cut-aligned example provenance is invalid")
        seen.add(event_id)
        segments = example.get("segments")
        if not isinstance(segments, list) or [
            row.get("role") for row in segments if isinstance(row, Mapping)
        ] != list(CUT_ALIGNED_SEGMENT_ROLES):
            raise ValueError("cut-aligned segment roles are invalid")
        for segment in segments:
            available = segment.get("available")
            indexes = segment.get("frame_indexes")
            embedding = segment.get("embedding")
            if not isinstance(available, bool) or not isinstance(indexes, list):
                raise ValueError("cut-aligned segment availability is invalid")
            if available:
                if (
                    len(indexes) != clip_frames
                    or not isinstance(embedding, list)
                    or len(embedding) != dimension
                    or not all(math.isfinite(float(value)) for value in embedding)
                ):
                    raise ValueError("cut-aligned segment embedding is invalid")
            elif indexes or embedding is not None:
                raise ValueError("unavailable cut-aligned segment must be empty")
    if bool(artifact.get("complete")) != (seen == set(expected)):
        raise ValueError("cut-aligned completion flag is inconsistent")


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value or "")
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"invalid {field} SHA-256")
    return text


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
