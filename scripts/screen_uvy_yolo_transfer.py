#!/usr/bin/env python3
"""Screen a generic YOLO detector on the held-out UVY basketball sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from app.analysis.uvy_detector_screen import (
    ball_detection_metrics,
    seal_uvy_detector_screen,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 10_000_000 else value * 1024


def _target_boxes(label_path: Path, *, width: int, height: int) -> list[list[float]]:
    boxes: list[list[float]] = []
    if not label_path.is_file():
        raise ValueError(f"UVY label file is missing: {label_path}")
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"invalid YOLO label row: {label_path}")
        class_id = int(fields[0])
        if class_id != 0:
            continue
        center_x, center_y, box_width, box_height = (float(value) for value in fields[1:])
        boxes.append(
            [
                (center_x - box_width / 2.0) * width,
                (center_y - box_height / 2.0) * height,
                (center_x + box_width / 2.0) * width,
                (center_y + box_height / 2.0) * height,
            ]
        )
    return boxes


def _ball_class_ids(names: Any) -> set[int]:
    if isinstance(names, dict):
        rows = names.items()
    elif isinstance(names, (list, tuple)):
        rows = enumerate(names)
    else:
        return set()
    return {
        int(index)
        for index, name in rows
        if str(name).strip().lower() in {"sports ball", "ball"}
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("yolov8n.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=120)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    if args.max_images < 1 or not 0 < args.confidence <= 1 or not 0 < args.iou <= 1:
        raise SystemExit("invalid screen bounds")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    image_list = [
        line.strip()
        for line in (args.data_root / "test.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not image_list:
        raise SystemExit("UVY test list is empty")
    if len(image_list) > args.max_images:
        stride = max(1, len(image_list) // args.max_images)
        image_list = image_list[::stride][: args.max_images]
    model = YOLO(str(args.model))
    target_rows: list[list[list[float]]] = []
    prediction_rows: list[list[list[float]]] = []
    rows: list[dict[str, Any]] = []
    ball_ids: set[int] = set()
    for relative in image_list:
        image_path = args.data_root / relative
        label_path = args.data_root / "labels" / "test" / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"could not decode UVY image: {image_path}")
        height, width = image.shape[:2]
        target = _target_boxes(label_path, width=width, height=height)
        result = model.predict(
            source=image,
            conf=args.confidence,
            iou=0.7,
            imgsz=640,
            device="cpu",
            max_det=50,
            verbose=False,
        )[0]
        ball_ids.update(_ball_class_ids(result.names))
        boxes = result.boxes
        predicted: list[list[float]] = []
        if boxes is not None and len(boxes) > 0:
            classes = boxes.cls.cpu().tolist()
            for box, class_id in zip(boxes.xyxy.cpu().tolist(), classes):
                if int(class_id) in _ball_class_ids(result.names):
                    predicted.append([float(value) for value in box])
        target_rows.append(target)
        prediction_rows.append(predicted)
        rows.append(
            {
                "image": relative,
                "target_box_count": len(target),
                "prediction_box_count": len(predicted),
                "target_boxes": target,
                "prediction_boxes": predicted,
            }
        )
    metrics = ball_detection_metrics(target_rows, prediction_rows, iou_threshold=args.iou)
    artifact = seal_uvy_detector_screen(
        {
            "purpose": "offline_generic_yolo_uvy_held_sequence_ball_screen",
            "runtime_consumable": False,
            "codex_runtime_answer_used": False,
            "source_manifest_sha256": str(manifest.get("source_manifest_sha256") or ""),
            "yolo_manifest_sha256": str(manifest.get("manifest_sha256") or ""),
            "model_path": str(args.model),
            "model_sha256": _sha256_file(args.model),
            "model_class_ids": sorted(ball_ids),
            "split": "test",
            "sample_count": len(rows),
            "confidence": args.confidence,
            "iou_threshold": args.iou,
            "metrics": metrics,
            "rows": rows,
            "promotion_eligible": False,
            "promotion_decision": "auxiliary_screen_only_no_runtime_or_cross_broadcast_promotion",
            "max_rss_bytes": _max_rss_bytes(),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sample_count": artifact["sample_count"],
                "metrics": metrics,
                "max_rss_bytes": artifact["max_rss_bytes"],
                "artifact_sha256": artifact["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
