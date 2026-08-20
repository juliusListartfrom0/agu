#!/usr/bin/env python3
"""Run a bounded UVY auxiliary YOLO training and held-sequence screen.

This command is intentionally offline-only.  It consumes the already
materialized UVY YOLO tree, never reads AGU blind/evaluation assets, and seals
the result as non-runtime, non-causal evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.uvy_detector_screen import ball_detection_metrics  # noqa: E402
from app.analysis.uvy_detector_training import (  # noqa: E402
    build_uvy_training_plan,
    seal_uvy_training_result,
)
from app.analysis.uvy_yolo import verify_uvy_yolo_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--screen-limit", type=int, default=240)
    parser.add_argument("--resource-log", type=Path, required=True)
    parser.add_argument("--max-memory-percent", type=float, default=85.0)
    parser.add_argument("--min-available-gib", type=float, default=4.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.confidence <= 1 or not 0 < args.iou <= 1:
        raise ValueError("confidence and IoU must be in (0, 1]")
    if args.screen_limit < 0:
        raise ValueError("screen-limit must be non-negative")
    dataset_root = args.dataset_root.resolve()
    manifest_path = args.manifest.resolve()
    model_path = args.model.resolve()
    args.run_dir = args.run_dir.resolve()
    args.output = args.output.resolve()
    args.resource_log = args.resource_log.resolve()
    data_yaml = dataset_root / "data.yaml"
    test_list = dataset_root / "test.txt"
    for required in (dataset_root, manifest_path, model_path, data_yaml, test_list):
        if not required.exists():
            raise FileNotFoundError(required)

    manifest = verify_uvy_yolo_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
    plan = build_uvy_training_plan(
        manifest,
        model_sha256=_sha256_file(model_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        output_dir=args.run_dir,
        max_memory_percent=args.max_memory_percent,
        min_available_memory_gib=args.min_available_gib,
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.run_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    absolute_data_yaml = _write_absolute_data_yaml(dataset_root, args.run_dir)

    model = YOLO(str(model_path))
    train_result = model.train(
        data=str(absolute_data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.run_dir.parent),
        name=args.run_dir.name,
        exist_ok=True,
        seed=42,
        deterministic=True,
        cache=False,
        amp=False,
        plots=False,
        verbose=False,
        patience=args.epochs,
    )
    del train_result

    checkpoint = args.run_dir / "weights" / "best.pt"
    if not checkpoint.is_file():
        checkpoint = args.run_dir / "weights" / "last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError("YOLO training did not emit best.pt or last.pt")

    trained = YOLO(str(checkpoint))
    test_images = [
        (dataset_root / line.strip()).resolve()
        for line in test_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected_indices = _sample_indices(len(test_images), args.screen_limit)
    selected_images = [test_images[index] for index in selected_indices]
    targets: list[list[list[float]]] = []
    predictions: list[list[list[float]]] = []
    rows: list[dict[str, Any]] = []
    for image_path in selected_images:
        label_path = _label_path(image_path)
        target_boxes = _read_ball_targets(image_path, label_path)
        result = trained.predict(
            source=str(image_path),
            conf=args.confidence,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            max_det=20,
            verbose=False,
        )[0]
        predicted_boxes: list[list[float]] = []
        if result.boxes is not None:
            classes = result.boxes.cls.cpu().tolist()
            boxes = result.boxes.xyxy.cpu().tolist()
            for class_id, box in zip(classes, boxes):
                if int(class_id) == 0:
                    predicted_boxes.append([float(value) for value in box])
        targets.append(target_boxes)
        predictions.append(predicted_boxes)
        rows.append(
            {
                "image": str(image_path.relative_to(dataset_root)),
                "target_count": len(target_boxes),
                "prediction_count": len(predicted_boxes),
            }
        )

    metrics = ball_detection_metrics(targets, predictions, iou_threshold=args.iou)
    resource_log = args.resource_log.resolve()
    artifact = seal_uvy_training_result(
        {
            "purpose": "offline_uvy_auxiliary_detector_training_screen",
            "plan_sha256": plan["plan_sha256"],
            "plan_path": str(plan_path),
            "source_yolo_manifest_sha256": manifest["manifest_sha256"],
            "source_manifest_sha256": manifest["source_manifest_sha256"],
            "base_model_sha256": plan["model_sha256"],
            "checkpoint_sha256": _sha256_file(checkpoint),
            "checkpoint_path": str(checkpoint),
            "run_dir": str(args.run_dir),
            "screen_split": "test",
            "screen_limit": args.screen_limit,
            "selected_frame_count": len(selected_images),
            "confidence": args.confidence,
            "iou_threshold": args.iou,
            "metrics": metrics,
            "rows": rows,
            "resource_guard_log": str(resource_log),
            "resource_guard_log_sha256_at_child_exit": (
                _sha256_file(resource_log) if resource_log.is_file() else None
            ),
            "checkpoint_promoted": False,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "metrics": metrics,
                "checkpoint": str(checkpoint),
                "artifact": str(args.output),
                "artifact_sha256": artifact["artifact_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _sample_indices(count: int, limit: int) -> list[int]:
    if count <= 0:
        return []
    if limit == 0 or limit >= count:
        return list(range(count))
    if limit == 1:
        return [count // 2]
    return sorted({round(index * (count - 1) / (limit - 1)) for index in range(limit)})


def _write_absolute_data_yaml(dataset_root: Path, run_dir: Path) -> Path:
    """Write a self-contained YAML/list bundle because Ultralytics resolves list entries from cwd."""

    path = run_dir / "data.absolute.yaml"
    split_paths: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        source_list = dataset_root / f"{split}.txt"
        target_list = run_dir / f"{split}.absolute.txt"
        entries = [
            str((dataset_root / line.strip()).resolve())
            for line in source_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        target_list.write_text("\n".join(entries) + "\n", encoding="utf-8")
        split_paths[split] = target_list
    escaped_root = str(dataset_root).replace("'", "''")
    path.write_text(
        f"path: '{escaped_root}'\n"
        f"train: '{split_paths['train']}'\n"
        f"val: '{split_paths['val']}'\n"
        f"test: '{split_paths['test']}'\n"
        "names: ['sports_ball', 'goal', 'player', 'referee']\n",
        encoding="utf-8",
    )
    return path


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        images_index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as exc:
        raise ValueError(f"UVY image is outside an images split: {image_path}") from exc
    parts[images_index] = "labels"
    return Path(*parts).with_suffix(".txt")


def _read_ball_targets(image_path: Path, label_path: Path) -> list[list[float]]:
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"could not decode {image_path}")
    height, width = image.shape[:2]
    targets: list[list[float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if not values:
            continue
        if len(values) != 5:
            raise ValueError(f"invalid YOLO label row: {label_path}")
        class_id, cx, cy, box_width, box_height = (float(value) for value in values)
        if int(class_id) != 0:
            continue
        targets.append(
            [
                (cx - box_width / 2) * width,
                (cy - box_height / 2) * height,
                (cx + box_width / 2) * width,
                (cy + box_height / 2) * height,
            ]
        )
    return targets


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
