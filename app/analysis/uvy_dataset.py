"""Audit the bounded UVY basketball image/trajectory subset.

UVY is useful for detector transfer because the released basketball sequences
contain image frames and MOT-style boxes for players, the ball, goals and
officials under CC-BY-4.0.  The release does not contain shot outcome labels,
full-game coverage, or a causal ball--hand--rim contract, so this module keeps
it outside runtime and official-statistics truth.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

UVY_AUDIT_SCHEMA = "agu.uvy-basketball-subset-audit.v1"
UVY_MANIFEST_SCHEMA = "agu.uvy-basketball-subset-download-manifest.v1"
_FRAME_RE = re.compile(r"^(\d+)\.jpg$", re.IGNORECASE)


def build_uvy_basketball_audit(
    source_root: str | Path,
    manifest: Mapping[str, Any],
    *,
    source_url: str,
    source_record_url: str,
    source_revision: str,
    source_license: str,
) -> dict[str, Any]:
    """Build a deterministic audit over extracted UVY frames and MOT rows."""

    root = Path(source_root)
    if not root.is_dir():
        raise ValueError("UVY source root is missing")
    if manifest.get("schema_version") != UVY_MANIFEST_SCHEMA:
        raise ValueError("unsupported UVY download manifest")
    sequences = manifest.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("UVY manifest has no sequences")
    if not source_url or not source_record_url or not source_revision:
        raise ValueError("UVY source identity is required")

    audited: list[dict[str, Any]] = []
    for sequence in sequences:
        if not isinstance(sequence, Mapping):
            raise ValueError("UVY sequence manifest must be an object")
        name = str(sequence.get("sequence") or "").strip()
        if not name or not re.fullmatch(r"basketball_V0[1-4]", name):
            raise ValueError(f"invalid UVY sequence name: {name!r}")
        sequence_root = root / "UVY" / name
        audited.append(_audit_sequence(sequence_root, name, sequence))

    gt_hash_groups: dict[str, list[str]] = {}
    for item in audited:
        gt_hash_groups.setdefault(str(item["gt_sha256"]), []).append(str(item["sequence"]))
    duplicate_groups = {
        digest: sorted(names)
        for digest, names in gt_hash_groups.items()
        if len(names) > 1
    }
    for item in audited:
        item["duplicate_gt_sequences"] = duplicate_groups.get(str(item["gt_sha256"]), [])

    total_boxes = sum(int(item["box_count"]) for item in audited)
    total_images = sum(int(item["image_frame_count"]) for item in audited)
    class_counts: Counter[str] = Counter()
    for item in audited:
        class_counts.update(item["class_counts"])
    artifact: dict[str, Any] = {
        "schema_version": UVY_AUDIT_SCHEMA,
        "purpose": "offline_auxiliary_player_ball_goal_referee_detection_and_hard_negative_screen",
        "source_url": source_url,
        "source_record_url": source_record_url,
        "source_revision": source_revision,
        "source_license": source_license,
        "source_archive": dict(manifest.get("source_archive") or {}),
        "download_manifest_sha256": str(manifest.get("manifest_sha256") or ""),
        "sequence_count": len(audited),
        "sequences": audited,
        "duplicate_gt_groups": duplicate_groups,
        "image_frame_count": total_images,
        "box_count": total_boxes,
        "class_counts": dict(sorted(class_counts.items())),
        "media_included": True,
        "runtime_consumable": False,
        "training_media_eligible": True,
        "training_scope": "auxiliary_detector_and_hard_negative_only",
        "causal_truth_eligible": False,
        "shot_outcome_label_count": 0,
        "subsecond_ball_hand_rim_outcome_labels": False,
        "continuous_five_by_five_video": False,
        "broadcast_diverse": False,
        "exhaustive_non_shot_hard_negatives": False,
        "decision": (
            "retain_cc_by_auxiliary_detector_and_hard_negative_only; "
            "reject_runtime_and_shot_outcome_truth"
        ),
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_uvy_basketball_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a sealed UVY audit and its conservative promotion boundary."""

    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    if artifact.get("schema_version") != UVY_AUDIT_SCHEMA:
        raise ValueError("unsupported UVY audit schema")
    if artifact.get("source_license") != "CC-BY-4.0":
        raise ValueError("UVY audit must retain the CC-BY-4.0 source license")
    if artifact.get("runtime_consumable") is not False:
        raise ValueError("UVY subset cannot be runtime consumable")
    if artifact.get("causal_truth_eligible") is not False:
        raise ValueError("UVY subset cannot be causal truth")
    if artifact.get("training_media_eligible") is not True:
        raise ValueError("UVY detector auxiliary must remain training eligible")
    if artifact.get("training_scope") != "auxiliary_detector_and_hard_negative_only":
        raise ValueError("UVY training scope is too broad")
    if int(artifact.get("shot_outcome_label_count", -1)) != 0:
        raise ValueError("UVY audit cannot claim shot-outcome labels")
    if not claimed or claimed != _canonical_sha256(artifact):
        raise ValueError("UVY audit hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _audit_sequence(root: Path, name: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not root.is_dir():
        raise ValueError(f"UVY sequence directory is missing: {root}")
    info_path = root / "video_info.txt"
    labels_path = root / "gt" / "labels.txt"
    gt_path = root / "gt" / "gt.txt"
    for path in (info_path, labels_path, gt_path):
        if not path.is_file():
            raise ValueError(f"UVY required annotation file is missing: {path}")

    info = _read_video_info(info_path)
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"UVY labels are empty: {labels_path}")
    declared_image_dir_name = str(info.get("imgDir") or "").strip()
    if declared_image_dir_name not in {"img", "img1"}:
        raise ValueError(f"UVY image directory is invalid: {declared_image_dir_name!r}")
    image_root = root / declared_image_dir_name
    image_dir_name = declared_image_dir_name
    image_directory_mismatch = False
    if not image_root.is_dir():
        alternate = root / ("img1" if declared_image_dir_name == "img" else "img")
        if not alternate.is_dir():
            raise ValueError(f"UVY image directory is missing: {image_root}")
        image_root = alternate
        image_dir_name = alternate.name
        image_directory_mismatch = True
    image_frames = _image_frames(image_root)
    rows = _read_gt(gt_path, labels)
    class_counts: Counter[str] = Counter()
    class_track_ids: dict[str, set[int]] = {label: set() for label in labels}
    frame_rows: set[int] = set()
    ball_frames: set[int] = set()
    goal_frames: set[int] = set()
    for row in rows:
        label = row["label"]
        class_counts[label] += 1
        class_track_ids[label].add(row["track_id"])
        frame_rows.add(row["frame"])
        if label == "sports ball":
            ball_frames.add(row["frame"])
        if label == "goal":
            goal_frames.add(row["frame"])

    metadata_frame_count = _optional_int(info.get("numFrames"))
    image_frame_ids = set(image_frames)
    gt_only_frames = sorted(frame_rows - image_frame_ids)
    image_without_rows = sorted(image_frame_ids - frame_rows)
    expected_files = int(manifest.get("image_file_count") or 0)
    sequence_entry_count = int(manifest.get("entry_count") or 0)
    return {
        "sequence": name,
        "video_info": info,
        "labels": labels,
        "declared_image_directory": declared_image_dir_name,
        "image_directory": image_dir_name,
        "image_directory_mismatch": image_directory_mismatch,
        "image_frame_count": len(image_frames),
        "image_frame_min": min(image_frames) if image_frames else None,
        "image_frame_max": max(image_frames) if image_frames else None,
        "metadata_frame_count": metadata_frame_count,
        "metadata_frame_count_mismatch": (
            metadata_frame_count is not None and metadata_frame_count != len(image_frames)
        ),
        "gt_frame_count": len(frame_rows),
        "gt_only_frame_count": len(gt_only_frames),
        "gt_only_frames_sample": gt_only_frames[:20],
        "image_without_annotations_count": len(image_without_rows),
        "box_count": len(rows),
        "gt_sha256": _sha256_file(gt_path),
        "class_counts": dict(sorted(class_counts.items())),
        "track_counts": {
            label: len(class_track_ids[label]) for label in sorted(class_track_ids)
        },
        "ball_frame_count": len(ball_frames),
        "goal_frame_count": len(goal_frames),
        "annotation_frame_coverage": (
            len(frame_rows) / len(image_frames) if image_frames else 0.0
        ),
        "manifest_entry_count": sequence_entry_count,
        "manifest_image_file_count": expected_files,
    }


def _read_video_info(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"invalid UVY video_info line: {path}")
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _image_frames(path: Path) -> list[int]:
    if not path.is_dir():
        raise ValueError(f"UVY image directory is missing: {path}")
    frames: list[int] = []
    for image in sorted(path.glob("*.jpg")):
        match = _FRAME_RE.match(image.name)
        if match is None or image.stat().st_size <= 0:
            raise ValueError(f"invalid UVY image: {image}")
        frames.append(int(match.group(1)))
    if len(frames) != len(set(frames)):
        raise ValueError(f"duplicate UVY image frames: {path}")
    return sorted(frames)


def _read_gt(path: Path, labels: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for line_number, values in enumerate(csv.reader(handle), start=1):
            if not values or all(not value.strip() for value in values):
                continue
            if len(values) != 9:
                raise ValueError(f"UVY gt row must contain 9 columns: {path}:{line_number}")
            try:
                frame = int(values[0])
                track_id = int(values[1])
                x, y, width, height = (float(value) for value in values[2:6])
                confidence = float(values[6])
                class_id = int(values[7])
                visibility = float(values[8])
            except ValueError as exc:
                raise ValueError(f"invalid UVY gt row: {path}:{line_number}") from exc
            if frame < 1 or track_id < 1 or class_id < 1 or class_id > len(labels):
                raise ValueError(f"invalid UVY gt identity: {path}:{line_number}")
            if not all(math.isfinite(value) for value in (x, y, width, height, confidence, visibility)):
                raise ValueError(f"non-finite UVY gt row: {path}:{line_number}")
            if width <= 0 or height <= 0:
                raise ValueError(f"non-positive UVY gt box: {path}:{line_number}")
            rows.append(
                {
                    "frame": frame,
                    "track_id": track_id,
                    "label": labels[class_id - 1],
                }
            )
    if not rows:
        raise ValueError(f"UVY gt is empty: {path}")
    return rows


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except ValueError as exc:
        raise ValueError(f"invalid UVY integer: {value!r}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
