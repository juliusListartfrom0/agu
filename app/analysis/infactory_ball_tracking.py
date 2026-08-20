"""Validate and seal the bounded Infactory tiny-ball tracking subset.

The upstream repository contains more image files than the public ``metadata.csv``
sample describes.  This module deliberately accepts only metadata-referenced rows,
keeps clip-level splits disjoint, and produces a runtime-inert provenance manifest.
The source is CC BY-NC 4.0, so artifacts made here are research-only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

_SCHEMA = "agu.infactory-ball-tracking.v1"
_SOURCE_URL = "https://huggingface.co/datasets/infactory-ai/ball-tracking"
_LICENSE = "CC-BY-NC-4.0"
_SPLITS = ("train", "val", "test")
_METADATA_FIELDS = (
    "file_path",
    "video_source",
    "frame_index",
    "visibility",
    "bboxes_count",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_nonnegative_int(value: object, *, field: str) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return parsed


def _validate_relative_data_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.startswith("data/"):
        raise ValueError(f"{field} must be a relative data path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".jpg":
        raise ValueError(f"{field} must be a safe JPG data path")
    if len(path.parts) != 3 or path.parts[1] not in {"visible", "not_visible"}:
        raise ValueError(f"{field} must use the visible/not_visible layout")
    return path.as_posix()


def _parse_yolo_labels(path: Path, *, expected_count: int) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != expected_count:
        raise ValueError(f"label count does not match metadata: {path}")
    for line in lines:
        fields = line.split()
        if len(fields) != 5 or fields[0] != "0":
            raise ValueError(f"invalid ball YOLO label: {path}")
        try:
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"invalid ball YOLO coordinate: {path}") from exc
        if (
            not all(math.isfinite(value) and 0 <= value <= 1 for value in coordinates)
            or coordinates[2] <= 0
            or coordinates[3] <= 0
        ):
            raise ValueError(f"ball YOLO coordinate is outside normalized bounds: {path}")
    return lines


def assign_infactory_clip_splits(
    clip_ids: Iterable[str],
    *,
    validation_clip_count: int = 2,
    test_clip_count: int = 2,
    seed: str = "infactory-ball-tracking-v1",
) -> dict[str, str]:
    """Assign complete source clips to deterministic train/val/test splits."""

    values = sorted(set(clip_ids))
    if (
        len(values) < 3
        or validation_clip_count < 1
        or test_clip_count < 1
        or validation_clip_count + test_clip_count >= len(values)
        or not seed
    ):
        raise ValueError("invalid Infactory clip split configuration")
    ordered = sorted(
        values,
        key=lambda clip_id: (
            hashlib.sha256(f"{seed}\0{clip_id}".encode("utf-8")).hexdigest(),
            clip_id,
        ),
    )
    test_ids = set(ordered[:test_clip_count])
    val_ids = set(ordered[test_clip_count : test_clip_count + validation_clip_count])
    return {
        clip_id: (
            "test"
            if clip_id in test_ids
            else "val"
            if clip_id in val_ids
            else "train"
        )
        for clip_id in values
    }


def match_ball_boxes(
    target_boxes: Sequence[Sequence[float]],
    predicted_boxes: Sequence[Sequence[float]],
    *,
    iou_threshold: float,
) -> dict[str, int]:
    """Return conservative one-to-one TP/FP/FN counts for one frame."""

    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    unmatched_targets = set(range(len(target_boxes)))
    true_positives = 0
    false_positives = 0
    for prediction in predicted_boxes:
        best_index: int | None = None
        best_iou = 0.0
        for target_index in unmatched_targets:
            iou = _box_iou(target_boxes[target_index], prediction)
            if iou >= iou_threshold and iou > best_iou:
                best_index = target_index
                best_iou = iou
        if best_index is None:
            false_positives += 1
        else:
            unmatched_targets.remove(best_index)
            true_positives += 1
    return {
        "tp": true_positives,
        "fp": false_positives,
        "fn": len(unmatched_targets),
    }


def _box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != 4 or len(right) != 4:
        raise ValueError("boxes must contain four coordinates")
    lx1, ly1, lx2, ly2 = (float(value) for value in left)
    rx1, ry1, rx2, ry2 = (float(value) for value in right)
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def build_infactory_ball_tracking_manifest(
    root: Path,
    *,
    source_revision: str,
    validation_clip_count: int = 2,
    test_clip_count: int = 2,
    split_seed: str = "infactory-ball-tracking-v1",
) -> dict[str, Any]:
    """Validate metadata-referenced files and return a sealed research manifest."""

    if not _REVISION.fullmatch(source_revision):
        raise ValueError("source_revision must be a 40-character lowercase commit")
    metadata_path = root / "metadata.csv"
    if not metadata_path.is_file():
        raise ValueError("Infactory metadata.csv is missing")
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _METADATA_FIELDS:
            raise ValueError("Infactory metadata fields are not canonical")
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError("Infactory metadata.csv is empty")

    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_frames: set[tuple[str, int]] = set()
    expected_images: set[str] = set()
    expected_labels: set[str] = set()
    for raw in raw_rows:
        relative_path = _validate_relative_data_path(raw.get("file_path"), field="file_path")
        if relative_path in seen_paths:
            raise ValueError("Infactory metadata contains duplicate file_path")
        seen_paths.add(relative_path)
        video_source = str(raw.get("video_source") or "")
        if not video_source:
            raise ValueError("Infactory video_source must be non-empty")
        frame_index = _parse_nonnegative_int(raw.get("frame_index"), field="frame_index")
        frame_key = (video_source, frame_index)
        if frame_key in seen_frames:
            raise ValueError("Infactory metadata contains duplicate source frame")
        seen_frames.add(frame_key)
        visibility = str(raw.get("visibility") or "")
        if visibility not in {"visible", "not_visible"}:
            raise ValueError("Infactory visibility is invalid")
        bboxes_count = _parse_nonnegative_int(raw.get("bboxes_count"), field="bboxes_count")
        if visibility == "visible" and bboxes_count < 1:
            raise ValueError("visible frame must have at least one box")
        if visibility == "not_visible" and bboxes_count != 0:
            raise ValueError("not_visible frame must have zero boxes")

        image_path = root / relative_path
        if not image_path.is_file():
            raise ValueError(f"Infactory image is missing: {relative_path}")
        expected_images.add(relative_path)
        label_path = image_path.with_suffix(".txt")
        label_relative = label_path.relative_to(root).as_posix()
        if visibility == "visible":
            if not label_path.is_file():
                raise ValueError(f"visible frame is missing a label: {relative_path}")
            _parse_yolo_labels(label_path, expected_count=bboxes_count)
            expected_labels.add(label_relative)
        elif label_path.exists():
            raise ValueError("not_visible frame must not have a label")
        rows.append(
            {
                "file_path": relative_path,
                "video_source": video_source,
                "frame_index": frame_index,
                "visibility": visibility,
                "bboxes_count": bboxes_count,
                "image_sha256": _file_sha256(image_path),
                "label_path": label_relative if visibility == "visible" else None,
                "label_sha256": _file_sha256(label_path) if visibility == "visible" else None,
            }
        )

    actual_images = {
        path.relative_to(root).as_posix()
        for path in (root / "data").rglob("*.jpg")
    }
    actual_labels = {
        path.relative_to(root).as_posix()
        for path in (root / "data").rglob("*.txt")
    }
    if actual_images != expected_images:
        raise ValueError("Infactory data contains images outside metadata.csv")
    if actual_labels != expected_labels:
        raise ValueError("Infactory data contains labels outside visible metadata rows")

    split_by_clip = assign_infactory_clip_splits(
        (row["video_source"] for row in rows),
        validation_clip_count=validation_clip_count,
        test_clip_count=test_clip_count,
        seed=split_seed,
    )
    for row in rows:
        row["split"] = split_by_clip[row["video_source"]]
    visibility_counts = Counter(row["visibility"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    split_clip_ids = {
        split: sorted(clip for clip, assigned in split_by_clip.items() if assigned == split)
        for split in _SPLITS
    }
    manifest: dict[str, Any] = {
        "schema_version": _SCHEMA,
        "purpose": "offline_noncommercial_tiny_ball_auxiliary_pretraining_only",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "source_url": _SOURCE_URL,
        "source_revision": source_revision,
        "license": _LICENSE,
        "subset_policy": "metadata_csv_rows_only_unreferenced_repository_payload_excluded",
        "metadata_sha256": _file_sha256(metadata_path),
        "split_seed": split_seed,
        "split_clip_ids": split_clip_ids,
        "row_counts": {visibility: visibility_counts.get(visibility, 0) for visibility in ("visible", "not_visible")},
        "split_counts": {split: split_counts.get(split, 0) for split in _SPLITS},
        "box_count": sum(int(row["bboxes_count"]) for row in rows),
        "examples": sorted(rows, key=lambda row: (row["video_source"], row["frame_index"])),
    }
    manifest["artifact_sha256"] = _canonical_sha256(manifest)
    return manifest


def verify_infactory_ball_tracking_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the self-contained hash of a sealed Infactory manifest."""

    verified = deepcopy(dict(manifest))
    claimed = verified.pop("artifact_sha256", None)
    if not isinstance(claimed, str) or _SHA256.fullmatch(claimed) is None:
        raise ValueError("Infactory artifact_sha256 is invalid")
    if verified.get("schema_version") != _SCHEMA:
        raise ValueError("Infactory manifest schema is invalid")
    if verified.get("runtime_consumable") is not False:
        raise ValueError("Infactory manifest must be runtime-inert")
    if verified.get("codex_runtime_answer_used") is not False:
        raise ValueError("Infactory manifest cannot use Codex runtime answers")
    if verified.get("license") != _LICENSE:
        raise ValueError("Infactory manifest license is invalid")
    if _canonical_sha256(verified) != claimed:
        raise ValueError("Infactory manifest artifact hash mismatch")
    verified["artifact_sha256"] = claimed
    return verified
