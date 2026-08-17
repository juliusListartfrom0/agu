"""Materialize UVY MOT boxes as a leakage-safe YOLO auxiliary dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

UVY_YOLO_SCHEMA = "agu.uvy-yolo-auxiliary.v1"
UVY_YOLO_CLASSES = ("sports_ball", "goal", "player", "referee")
_FRAME_RE = re.compile(r"^(\d+)\.jpg$", re.IGNORECASE)
_CLASS_MAP = {
    "sports ball": 0,
    "goal": 1,
    "player": 2,
    "referee": 3,
    "goalkeeper": 2,
    "person playing": 2,
}


def assign_uvy_sequence_splits(
    sequences: Sequence[str], *, validation_sequence: str, test_sequence: str
) -> dict[str, str]:
    """Assign complete UVY sequences to train/validation/test splits."""

    values = sorted({str(value).strip() for value in sequences if str(value).strip()})
    if not values or validation_sequence == test_sequence:
        raise ValueError("UVY split sequences must be non-empty and distinct")
    if validation_sequence not in values or test_sequence not in values:
        raise ValueError("UVY split sequence is absent")
    return {
        sequence: (
            "val"
            if sequence == validation_sequence
            else "test"
            if sequence == test_sequence
            else "train"
        )
        for sequence in values
    }


def materialize_uvy_yolo_subset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    source_manifest_sha256: str,
    sequence_splits: Mapping[str, str],
    ignored_sequences: Sequence[str] = (),
) -> dict[str, Any]:
    """Create hard-link/symlink images and normalized four-class labels."""

    source = Path(source_root)
    output = Path(output_root)
    if not source.is_dir():
        raise ValueError("UVY source root is missing")
    if not re.fullmatch(r"[0-9a-f]{64}", source_manifest_sha256):
        raise ValueError("UVY source manifest hash must be lowercase SHA-256")
    if set(sequence_splits.values()) != {"train", "val", "test"}:
        raise ValueError("UVY splits must contain train, val and test")
    ignored = sorted({str(value).strip() for value in ignored_sequences if str(value).strip()})
    if set(ignored) & set(sequence_splits):
        raise ValueError("UVY ignored sequence overlaps a materialized split")

    split_counts: Counter[str] = Counter()
    split_box_counts: Counter[str] = Counter()
    split_empty_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    link_modes: Counter[str] = Counter()
    for sequence, split in sorted(sequence_splits.items()):
        if not re.fullmatch(r"basketball_V0[1-4]", sequence):
            raise ValueError(f"invalid UVY sequence: {sequence}")
        sequence_root = source / "UVY" / sequence
        info = _read_info(sequence_root / "video_info.txt")
        declared_image_dir = str(info.get("imgDir") or "")
        image_root = sequence_root / declared_image_dir
        if not image_root.is_dir():
            alternate = sequence_root / ("img1" if declared_image_dir == "img" else "img")
            if not alternate.is_dir():
                raise ValueError(f"UVY image directory is missing: {sequence}")
            image_root = alternate
        width = _positive_int(info.get("imWidth"), "imWidth")
        height = _positive_int(info.get("imHeight"), "imHeight")
        rows_by_frame = _read_rows(sequence_root / "gt" / "gt.txt", sequence_root / "gt" / "labels.txt")
        frame_paths = sorted(image_root.glob("*.jpg"), key=_frame_number)
        if not frame_paths:
            raise ValueError(f"UVY sequence has no images: {sequence}")
        image_list: list[str] = []
        for image_path in frame_paths:
            frame = _frame_number(image_path)
            stem = f"{sequence}_{frame:06d}"
            image_target = output / "images" / split / f"{stem}.jpg"
            label_target = output / "labels" / split / f"{stem}.txt"
            mode = _link_or_replace(image_path, image_target)
            link_modes[mode] += 1
            labels: list[tuple[int, float, float, float, float, str]] = []
            for row in rows_by_frame.get(frame, ()):  # keep image-only negatives
                class_id = _CLASS_MAP.get(row["label"])
                if class_id is None:
                    continue
                x, y, box_width, box_height = row["x"], row["y"], row["width"], row["height"]
                center_x = (x + box_width / 2.0) / width
                center_y = (y + box_height / 2.0) / height
                normalized_width = box_width / width
                normalized_height = box_height / height
                if not all(
                    math.isfinite(value)
                    for value in (center_x, center_y, normalized_width, normalized_height)
                ) or not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 < normalized_width <= 1 and 0 < normalized_height <= 1):
                    raise ValueError(f"UVY box is outside image bounds: {sequence}:{frame}")
                labels.append(
                    (
                        class_id,
                        center_x,
                        center_y,
                        normalized_width,
                        normalized_height,
                        row["label"],
                    )
                )
            labels.sort(key=lambda value: (value[0], value[1], value[2]))
            label_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.write_text(
                "".join(
                    f"{class_id} {cx:.8f} {cy:.8f} {box_width:.8f} {box_height:.8f}\n"
                    for class_id, cx, cy, box_width, box_height, _ in labels
                ),
                encoding="utf-8",
            )
            relative_image = str(image_target.relative_to(output))
            image_list.append(relative_image)
            split_counts[split] += 1
            split_box_counts[split] += len(labels)
            split_empty_counts[split] += int(not labels)
            for class_id, *_rest, source_label in labels:
                class_counts[UVY_YOLO_CLASSES[class_id]] += 1
            records.append(
                {
                    "sequence": sequence,
                    "split": split,
                    "frame": frame,
                    "image": relative_image,
                    "label": str(label_target.relative_to(output)),
                    "box_count": len(labels),
                }
            )
        (output / f"{split}.txt").parent.mkdir(parents=True, exist_ok=True)
        list_path = output / f"{split}.txt"
        existing = set()
        if list_path.is_file():
            existing = {
                line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()
            }
        list_path.write_text("\n".join(sorted(existing | set(image_list))) + "\n", encoding="utf-8")

    data_yaml = output / "data.yaml"
    data_yaml.write_text(
        "path: .\ntrain: train.txt\nval: val.txt\ntest: test.txt\n"
        f"names: {list(UVY_YOLO_CLASSES)!r}\n",
        encoding="utf-8",
    )
    manifest: dict[str, Any] = {
        "schema_version": UVY_YOLO_SCHEMA,
        "purpose": "uvy_auxiliary_detector_and_hard_negative_training",
        "source_manifest_sha256": source_manifest_sha256,
        "sequence_splits": dict(sorted(sequence_splits.items())),
        "ignored_sequences": ignored,
        "class_names": list(UVY_YOLO_CLASSES),
        "split_counts": dict(sorted(split_counts.items())),
        "split_box_counts": dict(sorted(split_box_counts.items())),
        "split_empty_frame_counts": dict(sorted(split_empty_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "link_modes": dict(sorted(link_modes.items())),
        "example_count": len(records),
        "examples": records,
        "runtime_consumable": False,
        "training_media_eligible": True,
        "training_scope": "auxiliary_detector_and_hard_negative_only",
        "causal_truth_eligible": False,
        "codex_runtime_answer_used": False,
        "data_yaml": str(data_yaml.relative_to(output)),
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    return manifest


def verify_uvy_yolo_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(manifest)
    claimed = str(artifact.pop("manifest_sha256", ""))
    if artifact.get("schema_version") != UVY_YOLO_SCHEMA:
        raise ValueError("invalid UVY YOLO manifest schema")
    if artifact.get("runtime_consumable") is not False:
        raise ValueError("UVY YOLO manifest cannot be runtime consumable")
    if artifact.get("training_scope") != "auxiliary_detector_and_hard_negative_only":
        raise ValueError("UVY YOLO training scope is too broad")
    if artifact.get("causal_truth_eligible") is not False:
        raise ValueError("UVY YOLO manifest cannot be causal truth")
    if not claimed or claimed != _canonical_sha256(artifact):
        raise ValueError("UVY YOLO manifest hash mismatch")
    artifact["manifest_sha256"] = claimed
    return artifact


def _read_info(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    if not result:
        raise ValueError(f"UVY video info is empty: {path}")
    return result


def _read_rows(path: Path, labels_path: Path) -> dict[int, list[dict[str, Any]]]:
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for line_number, values in enumerate(csv.reader(handle), start=1):
            if len(values) != 9:
                raise ValueError(f"invalid UVY MOT row: {path}:{line_number}")
            try:
                frame, track_id = int(values[0]), int(values[1])
                x, y, width, height = (float(value) for value in values[2:6])
                class_id = int(values[7])
            except ValueError as exc:
                raise ValueError(f"invalid UVY MOT value: {path}:{line_number}") from exc
            if not 1 <= class_id <= len(labels) or frame < 1 or track_id < 1:
                raise ValueError(f"invalid UVY MOT identity: {path}:{line_number}")
            if not all(math.isfinite(value) for value in (x, y, width, height)) or width <= 0 or height <= 0:
                raise ValueError(f"invalid UVY MOT box: {path}:{line_number}")
            rows[frame].append(
                {
                    "track_id": track_id,
                    "label": labels[class_id - 1],
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                }
            )
    return rows


def _frame_number(path: Path) -> int:
    match = _FRAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"invalid UVY image name: {path}")
    return int(match.group(1))


def _positive_int(value: Any, key: str) -> int:
    try:
        result = int(str(value))
    except ValueError as exc:
        raise ValueError(f"invalid UVY {key}") from exc
    if result <= 0:
        raise ValueError(f"invalid UVY {key}")
    return result


def _link_or_replace(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return "symlink"
        destination.unlink()
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        os.symlink(source.resolve(), destination)
        return "symlink"


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
