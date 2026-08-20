"""Hash-bound full-frame training data built from reviewed broadcast ball boxes.

This module is intentionally offline-only.  It turns already reviewed, source-bound
candidate boxes into a tiny detector fine-tuning set; it never changes the runtime
detector or consumes Codex answers as labels.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2

from app.analysis.ball_candidate_review import verify_artifact

DECISIONS = {"valid_ball", "false_positive", "uncertain"}


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_bbox(
    bbox: Mapping[str, Any], *, width: int, height: int
) -> tuple[float, float, float, float]:
    """Return a clamped YOLO ``xyxy`` box normalized to image dimensions."""
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")
    x1, y1, x2, y2 = (float(bbox[key]) for key in ("x1", "y1", "x2", "y2"))
    left, right = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
    top, bottom = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
    if right <= left or bottom <= top:
        raise ValueError("bounding box has no positive area")
    return left / width, top / height, right / width, bottom / height


def _read_frame(cap: cv2.VideoCapture, frame_index: int) -> tuple[Any, str]:
    if frame_index < 0:
        raise ValueError("frame index must be non-negative")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"could not decode frame {frame_index}")
    return frame, hashlib.sha256(frame.tobytes()).hexdigest()


def collect_review_records(
    *,
    source_id: str,
    plan: Mapping[str, Any],
    review: Mapping[str, Any],
    video_path: Path,
    split: str,
) -> list[dict[str, Any]]:
    """Validate one plan/review/video bundle and return determinate frame rows."""
    if not source_id or split not in {"train", "val"}:
        raise ValueError("source_id and split are required")
    verified_plan = verify_artifact(plan)
    verified_review = verify_artifact(review)
    if verified_review.get("plan_sha256") != verified_plan["artifact_sha256"]:
        raise ValueError("review is not bound to plan")
    expected_video_sha = verified_plan.get("raw_video_sha256")
    actual_video_sha = file_sha256(video_path)
    if expected_video_sha and expected_video_sha != actual_video_sha:
        raise ValueError("review video hash mismatch")
    candidates = {str(row["candidate_id"]): row for row in verified_plan["candidates"]}
    decisions = verified_review.get("decisions", [])
    if len(decisions) != len(candidates):
        raise ValueError("review does not cover every planned candidate")
    decision_map: dict[str, str] = {}
    for row in decisions:
        candidate_id = str(row.get("candidate_id", ""))
        decision = str(row.get("decision", ""))
        if candidate_id not in candidates or decision not in DECISIONS:
            raise ValueError("review contains an unknown candidate or decision")
        if candidate_id in decision_map:
            raise ValueError("review contains duplicate candidate")
        decision_map[candidate_id] = decision

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video {video_path}")
    records: list[dict[str, Any]] = []
    try:
        for candidate_id, candidate in candidates.items():
            decision = decision_map[candidate_id]
            if decision == "uncertain":
                continue
            frame_index = int(candidate["frame"])
            frame, pixels_sha = _read_frame(cap, frame_index)
            expected_pixels_sha = candidate.get("frame_pixels_sha256", candidate.get("frame_sha256"))
            if expected_pixels_sha and str(expected_pixels_sha) != pixels_sha:
                raise ValueError(f"frame pixel hash mismatch for {candidate_id}")
            height, width = frame.shape[:2]
            box = normalized_bbox(candidate["bbox"], width=width, height=height)
            records.append(
                {
                    "source_id": source_id,
                    "source_video_sha256": actual_video_sha,
                    "plan_sha256": verified_plan["artifact_sha256"],
                    "review_sha256": verified_review["artifact_sha256"],
                    "candidate_id": candidate_id,
                    "frame": frame_index,
                    "decision": decision,
                    "label": decision,
                    "bbox_xyxy_normalized": list(box),
                    "frame_pixels_sha256": pixels_sha,
                    "width": int(width),
                    "height": int(height),
                    "split": split,
                }
            )
    finally:
        cap.release()
    return records


def build_dataset_manifest(
    source_specs: Sequence[Mapping[str, Any]], *, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Build a deterministic, runtime-inert manifest and reject frame conflicts."""
    if not records:
        raise ValueError("dataset must contain at least one determinate record")
    seen: dict[tuple[str, int], str] = {}
    for row in records:
        key = (str(row["source_video_sha256"]), int(row["frame"]))
        decision = str(row["decision"])
        previous = seen.get(key)
        if previous is not None and previous != decision:
            raise ValueError("conflicting decision for duplicate frame")
        seen[key] = decision
    unique_records: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in records:
        key = (str(row["source_video_sha256"]), int(row["frame"]))
        unique_records.setdefault(key, row)
    sources = []
    for spec in sorted(source_specs, key=lambda item: str(item["source_id"])):
        sources.append(
            {
                "source_id": str(spec["source_id"]),
                "video_sha256": str(spec["video_sha256"]),
                "plan_sha256": str(spec["plan_sha256"]),
                "review_sha256": str(spec["review_sha256"]),
                "split": str(spec["split"]),
            }
        )
    payload = {
        "schema_version": "agu.broadcast-ball-detector-dataset.v1",
        "purpose": "offline_cross_broadcast_full_frame_detector_screen",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "label_policy": "determinate reviewed candidate boxes; uncertain rows excluded; false positives are empty labels",
        "sources": sources,
        "split_counts": {
            "train": sum(1 for row in unique_records.values() if row["split"] == "train"),
            "val": sum(1 for row in unique_records.values() if row["split"] == "val"),
        },
        "decision_counts": {
            decision: sum(1 for row in unique_records.values() if row["decision"] == decision)
            for decision in ("valid_ball", "false_positive")
        },
        "records": [dict(row) for row in sorted(unique_records.values(), key=lambda item: (str(item["source_id"]), int(item["frame"])))],
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    return payload
