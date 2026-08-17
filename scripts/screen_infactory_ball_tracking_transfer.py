#!/usr/bin/env python3
"""Screen a YOLO ball model on the held-out Infactory clip split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from app.analysis.infactory_ball_tracking import (
    match_ball_boxes,
    verify_infactory_ball_tracking_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _target_boxes(root: Path, example: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    label_path = example.get("label_path")
    if label_path is None:
        return []
    image = cv2.imread(str(root / str(example["file_path"])), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode {example['file_path']}")
    height, width = image.shape[:2]
    targets = []
    for line in (root / str(label_path)).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5 or fields[0] != "0":
            raise ValueError(f"invalid ball label {label_path}")
        _, x_center, y_center, box_width, box_height = (float(value) for value in fields)
        x1 = (x_center - box_width / 2) * width
        y1 = (y_center - box_height / 2) * height
        x2 = (x_center + box_width / 2) * width
        y2 = (y_center + box_height / 2) * height
        targets.append((x1, y1, x2, y2))
    return targets


def main() -> int:
    args = parse_args()
    if not 0 < args.confidence <= 1 or not 0 < args.iou <= 1 or args.image_size < 32:
        raise ValueError("confidence, IoU, and image-size are invalid")
    manifest = verify_infactory_ball_tracking_manifest(
        json.loads(args.manifest.read_text(encoding="utf-8"))
    )
    examples = [row for row in manifest["examples"] if row["split"] == args.split]
    if not examples:
        raise ValueError(f"manifest has no {args.split} examples")
    model = YOLO(str(args.model))
    total = {"tp": 0, "fp": 0, "fn": 0}
    rows: list[dict[str, Any]] = []
    for example in examples:
        image_path = args.root / str(example["file_path"])
        targets = _target_boxes(args.root, example)
        results = model.predict(
            source=str(image_path),
            conf=args.confidence,
            iou=0.7,
            imgsz=args.image_size,
            device=args.device,
            classes=[0],
            max_det=20,
            verbose=False,
        )
        boxes = results[0].boxes
        predictions = boxes.xyxy.cpu().tolist() if boxes is not None else []
        counts = match_ball_boxes(targets, predictions, iou_threshold=args.iou)
        for key in total:
            total[key] += counts[key]
        rows.append(
            {
                "file_path": example["file_path"],
                "video_source": example["video_source"],
                "frame_index": example["frame_index"],
                "visibility": example["visibility"],
                "target_count": len(targets),
                "prediction_count": len(predictions),
                **counts,
            }
        )
    denominator_precision = total["tp"] + total["fp"]
    denominator_recall = total["tp"] + total["fn"]
    precision = total["tp"] / denominator_precision if denominator_precision else 0.0
    recall = total["tp"] / denominator_recall if denominator_recall else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    artifact: dict[str, Any] = {
        "schema_version": "agu.infactory-ball-tracking-transfer-screen.v1",
        "purpose": "offline_noncommercial_tiny_ball_model_transfer_screen",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "model_sha256": _file_sha256(args.model),
        "source_manifest_sha256": manifest["artifact_sha256"],
        "split": args.split,
        "confidence": args.confidence,
        "iou_threshold": args.iou,
        "image_size": args.image_size,
        "device": args.device,
        "metrics": {
            **total,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "frame_count": len(examples),
        },
        "accepted": bool(precision >= 0.85 and recall >= 0.85),
        "rows": rows,
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"artifact_sha256": artifact["artifact_sha256"], "metrics": artifact["metrics"], "accepted": artifact["accepted"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
