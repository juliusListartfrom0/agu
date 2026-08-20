#!/usr/bin/env python3
"""Materialize a hash-bound, screen-only overlapping-tile YOLO dataset.

The input is an existing source-bound YOLO tree (for example the CC BY 4.0
E-BARD/MUVY ball data).  This utility crops pixels and projects labels; it does
not add labels, mix benchmark media, or alter an AGU runtime detector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2

from app.analysis.perception.tiled_ball import TileSpec, horizontal_tile_specs

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_CLASS_NAMES = ("basketball", "hoop", "player", "referee")


@dataclass(frozen=True, slots=True)
class YoloLabel:
    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float

    def as_line(self) -> str:
        return (
            f"{self.class_id} {self.center_x:.8f} {self.center_y:.8f} "
            f"{self.width:.8f} {self.height:.8f}\n"
        )


def parse_yolo_labels(lines: Iterable[str]) -> tuple[YoloLabel, ...]:
    """Parse normalized YOLO rows and reject malformed or unsafe labels."""

    labels: list[YoloLabel] = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) != 5:
            raise ValueError(f"label line {line_number} must contain five fields")
        try:
            class_id = int(fields[0])
            values = tuple(float(value) for value in fields[1:])
        except ValueError as exc:
            raise ValueError(f"label line {line_number} contains non-numeric values") from exc
        if class_id < 0 or not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"label line {line_number} must use normalized coordinates")
        center_x, center_y, width, height = values
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"label line {line_number} must have positive dimensions")
        labels.append(
            YoloLabel(
                class_id=class_id,
                center_x=center_x,
                center_y=center_y,
                width=width,
                height=height,
            )
        )
    return tuple(labels)


def project_labels_to_tile(
    labels: Iterable[YoloLabel],
    *,
    frame_width: int,
    frame_height: int,
    tile: TileSpec,
) -> tuple[YoloLabel, ...]:
    """Clip source-normalized labels to one horizontal tile and renormalize."""

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    if tile.frame_width != frame_width or not 0 <= tile.x1 < tile.x2 <= frame_width:
        raise ValueError("tile does not match frame width")

    projected: list[YoloLabel] = []
    for label in labels:
        source_x1 = (label.center_x - label.width / 2.0) * frame_width
        source_x2 = (label.center_x + label.width / 2.0) * frame_width
        source_y1 = (label.center_y - label.height / 2.0) * frame_height
        source_y2 = (label.center_y + label.height / 2.0) * frame_height
        x1 = max(float(tile.x1), min(float(tile.x2), source_x1))
        x2 = max(float(tile.x1), min(float(tile.x2), source_x2))
        y1 = max(0.0, min(float(frame_height), source_y1))
        y2 = max(0.0, min(float(frame_height), source_y2))
        if x2 <= x1 or y2 <= y1:
            continue
        projected.append(
            YoloLabel(
                class_id=label.class_id,
                center_x=((x1 + x2) / 2.0 - tile.x1) / tile.width,
                center_y=((y1 + y2) / 2.0) / frame_height,
                width=(x2 - x1) / tile.width,
                height=(y2 - y1) / frame_height,
            )
        )
    return tuple(projected)


def materialize_tiled_dataset(
    source_root: Path,
    output_root: Path,
    *,
    splits: tuple[str, ...] = ("train", "val"),
    overlap_ratio: float = 0.24,
    jpeg_quality: int = 92,
    max_images_per_split: int = 0,
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
) -> dict[str, object]:
    """Create two overlapping crops per source image and return a sealed manifest."""

    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist: {source_root}")
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("output root must not overwrite or live inside source root")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("output root must be new or empty")
    if not splits or any(not split for split in splits):
        raise ValueError("at least one non-empty split is required")
    if not 1 <= jpeg_quality <= 100 or max_images_per_split < 0:
        raise ValueError("jpeg quality and image limit are invalid")
    if not class_names:
        raise ValueError("class names are required")

    output_root.mkdir(parents=True, exist_ok=True)
    source_manifest = source_root / "manifest.json"
    records: list[dict[str, object]] = []
    split_counts = {split: 0 for split in splits}
    for split in splits:
        source_images = sorted(
            path
            for path in (source_root / "images" / split).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if max_images_per_split:
            source_images = source_images[:max_images_per_split]
        if not source_images:
            raise ValueError(f"split has no images: {split}")
        destination_images = output_root / "images" / split
        destination_labels = output_root / "labels" / split
        destination_images.mkdir(parents=True, exist_ok=True)
        destination_labels.mkdir(parents=True, exist_ok=True)
        for source_image in source_images:
            source_label = source_root / "labels" / split / f"{source_image.stem}.txt"
            if not source_label.is_file():
                raise FileNotFoundError(f"missing label for {source_image.name}")
            image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"could not decode {source_image}")
            height, width = image.shape[:2]
            labels = parse_yolo_labels(source_label.read_text(encoding="utf-8").splitlines())
            tiles = horizontal_tile_specs(width, overlap_ratio=overlap_ratio)
            source_image_sha = file_sha256(source_image)
            source_label_sha = file_sha256(source_label)
            for tile_index, tile in enumerate(tiles):
                stem = f"{source_image.stem}__tile{tile_index}"
                output_image = destination_images / f"{stem}.jpg"
                output_label = destination_labels / f"{stem}.txt"
                crop = image[:, tile.x1 : tile.x2]
                if not cv2.imwrite(str(output_image), crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                    raise ValueError(f"could not write {output_image}")
                projected = project_labels_to_tile(
                    labels,
                    frame_width=width,
                    frame_height=height,
                    tile=tile,
                )
                output_label.write_text(
                    "".join(label.as_line() for label in projected), encoding="utf-8"
                )
                records.append(
                    {
                        "split": split,
                        "source_image": str(source_image.relative_to(source_root)),
                        "source_label": str(source_label.relative_to(source_root)),
                        "source_image_sha256": source_image_sha,
                        "source_label_sha256": source_label_sha,
                        "source_width": int(width),
                        "source_height": int(height),
                        "tile": asdict(tile),
                        "tile_index": tile_index,
                        "image": str(output_image.relative_to(output_root)),
                        "label": str(output_label.relative_to(output_root)),
                        "image_sha256": file_sha256(output_image),
                        "label_sha256": file_sha256(output_label),
                        "label_count": len(projected),
                    }
                )
                split_counts[split] += 1

    manifest: dict[str, object] = {
        "schema_version": "agu.tiled-ball-yolo.v1",
        "purpose": "offline_tile_aware_ball_detector_training_screen",
        "runtime_consumable": False,
        "training_eligible": True,
        "codex_runtime_answer_used": False,
        "source_root_name": source_root.name,
        "source_manifest_sha256": file_sha256(source_manifest) if source_manifest.is_file() else None,
        "tile_protocol": {
            "layout": "two_overlapping_horizontal_tiles_v1",
            "overlap_ratio": overlap_ratio,
            "class_names": list(class_names),
        },
        "splits": list(splits),
        "source_image_count": sum(split_counts.values()) // 2,
        "tile_count": len(records),
        "split_counts": split_counts,
        "records": records,
    }
    manifest["artifact_sha256"] = canonical_sha256(manifest)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "data.yaml").write_text(
        "path: "
        + str(output_root)
        + "\ntrain: images/train\nval: images/val\nnames:\n"
        + "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
        + "\n",
        encoding="utf-8",
    )
    return manifest


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overlap-ratio", type=float, default=0.24)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--max-images-per-split", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = materialize_tiled_dataset(
        args.source_root,
        args.output_root,
        overlap_ratio=args.overlap_ratio,
        jpeg_quality=args.jpeg_quality,
        max_images_per_split=args.max_images_per_split,
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "tile_count": manifest["tile_count"],
                "split_counts": manifest["split_counts"],
                "artifact_sha256": manifest["artifact_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
