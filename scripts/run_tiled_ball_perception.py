#!/usr/bin/env python3
"""Run an offline, horizontal-tile basketball-only perception screen.

This adapter is deliberately research-only.  It emits a specialist perception
artifact that can be passed to ``build_official_event_candidates.py`` as an
additional same-video source; it never changes AGU's runtime detector defaults.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.perception.tiled_ball import (  # noqa: E402
    deduplicate_ball_detections,
    horizontal_tile_specs,
    project_tile_box,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=0, help="Exclusive; zero means decode to EOF")
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--overlap-ratio", type=float, default=0.24)
    parser.add_argument("--dedup-center-distance", type=float, default=18.0)
    parser.add_argument("--tile-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tile_predictions(
    model: Any,
    frames: list[Any],
    frame_numbers: list[int],
    *,
    image_size: int,
    confidence: float,
    overlap_ratio: float,
    device: str,
    dedup_center_distance: float,
) -> list[dict[str, Any]]:
    if len(frames) != len(frame_numbers):
        raise ValueError("frames and frame_numbers must have equal length")
    tile_frames: list[Any] = []
    tile_context: list[tuple[int, Any]] = []
    for frame, frame_number in zip(frames, frame_numbers):
        height, width = frame.shape[:2]
        for tile in horizontal_tile_specs(width, overlap_ratio=overlap_ratio):
            tile_frames.append(frame[:, tile.x1 : tile.x2])
            tile_context.append((frame_number, tile))
    results = model.predict(
        source=tile_frames,
        conf=confidence,
        iou=0.7,
        imgsz=image_size,
        device=device,
        classes=[0],
        max_det=20,
        verbose=False,
    )
    candidates: list[dict[str, Any]] = []
    for tile_index, (result, (frame_number, tile)) in enumerate(zip(results, tile_context)):
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        coordinates = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        for object_index, (box, score) in enumerate(zip(coordinates, confidences)):
            x1, y1, x2, y2 = project_tile_box(box, tile)
            candidates.append(
                {
                    "detection_id": f"ultralytics_yolo_tiled_ebard_v1:{frame_number}:{tile_index % 2}:{object_index}",
                    "frame": int(frame_number),
                    "object_type": "basketball",
                    "bbox": {"x1": x1, "y1": float(y1), "x2": x2, "y2": float(y2)},
                    "confidence": float(score),
                    "track_id": None,
                    "player_id": None,
                    "team_id": None,
                    "keypoints": {},
                    "backend": "ultralytics_yolo_tiled_ebard_v1",
                }
            )
    return deduplicate_ball_detections(candidates, center_distance_px=dedup_center_distance)


def run_tiled_ball_scan(
    *,
    video_path: Path,
    model_path: Path,
    sample_fps: float = 2.0,
    start_frame: int = 0,
    end_frame: int = 0,
    image_size: int = 960,
    confidence: float = 0.05,
    overlap_ratio: float = 0.24,
    dedup_center_distance: float = 18.0,
    tile_batch_size: int = 8,
    device: str = "cpu",
) -> dict[str, Any]:
    if sample_fps <= 0 or start_frame < 0 or end_frame < 0:
        raise ValueError("sample-fps and frame bounds are invalid")
    if image_size < 32 or not 0 < confidence <= 1 or tile_batch_size <= 0:
        raise ValueError("image-size, confidence, or tile-batch-size is invalid")
    if not video_path.is_file() or not model_path.is_file():
        raise FileNotFoundError("video and model must exist")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(source_fps / sample_fps)))
    end = end_frame if end_frame > 0 else frame_count
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    model = YOLO(str(model_path))
    detections: list[dict[str, Any]] = []
    frames: list[Any] = []
    frame_numbers: list[int] = []
    decoded = start_frame
    sampled = 0
    started = time.monotonic()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number = decoded
            decoded += 1
            if frame_number >= end:
                break
            if (frame_number - start_frame) % stride:
                continue
            frames.append(frame)
            frame_numbers.append(frame_number)
            sampled += 1
            if len(frames) >= tile_batch_size:
                detections.extend(
                    _tile_predictions(
                        model,
                        frames,
                        frame_numbers,
                        image_size=image_size,
                        confidence=confidence,
                        overlap_ratio=overlap_ratio,
                        device=device,
                        dedup_center_distance=dedup_center_distance,
                    )
                )
                frames.clear()
                frame_numbers.clear()
                print(f"sampled={sampled} decoded={decoded}/{frame_count} ball_detections={len(detections)}", flush=True)
        if frames:
            detections.extend(
                _tile_predictions(
                    model,
                    frames,
                    frame_numbers,
                    image_size=image_size,
                    confidence=confidence,
                    overlap_ratio=overlap_ratio,
                    device=device,
                    dedup_center_distance=dedup_center_distance,
                )
            )
    finally:
        capture.release()
    return {
        "schema_version": "agu.official-perception.v1",
        "raw_video": {
            "filename": video_path.name,
            "sha256": _file_sha256(video_path),
            "size_bytes": video_path.stat().st_size,
            "source_fps": source_fps,
            "frame_count": frame_count,
        },
        "detector": {
            "backend": "ultralytics_yolo_tiled_ebard_v1",
            "model_filename": model_path.name,
            "model_sha256": _file_sha256(model_path),
            "device": device,
            "image_size": image_size,
            "confidence": confidence,
            "player_tracking": False,
            "tracker_config": "",
            "class_prompts": [],
            "object_type_allowlist": ["basketball"],
            "rim_max_center_y_ratio": 1.0,
            "rim_min_aspect_ratio": 0.0,
            "team_assignment": "",
            "tile_layout": "two_overlapping_horizontal_tiles_v1",
            "tile_overlap_ratio": overlap_ratio,
            "dedup_center_distance_px": dedup_center_distance,
        },
        "sampling": {
            "requested_fps": sample_fps,
            "stride_frames": stride,
            "start_frame": start_frame,
            "end_frame": decoded,
            "sample_count": sampled,
            "decoded_frame_count": decoded,
        },
        "runtime_seconds": time.monotonic() - started,
        "counts": {
            "player": 0,
            "basketball": sum(1 for item in detections if item["object_type"] == "basketball"),
            "rim": 0,
            "backboard": 0,
            "referee": 0,
        },
        "detections": detections,
        "ball_tracks": [],
    }


def main() -> int:
    args = parse_args()
    payload = run_tiled_ball_scan(
        video_path=args.video,
        model_path=args.model,
        sample_fps=args.sample_fps,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        image_size=args.image_size,
        confidence=args.confidence,
        overlap_ratio=args.overlap_ratio,
        dedup_center_distance=args.dedup_center_distance,
        tile_batch_size=args.tile_batch_size,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("runtime_seconds", "counts", "sampling")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
