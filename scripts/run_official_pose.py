#!/usr/bin/env python3
"""Extract raw-video-bound player pose observations for AGU action models."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.perception import PoseAdapter, UltralyticsPoseAdapter  # noqa: E402
from app.analysis.schemas import PerceptionDetectionResponse  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=704)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--keypoint-confidence", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=0.0)
    return parser.parse_args()


def run_pose_scan(
    *,
    video_path: Path,
    model_path: Path,
    sample_fps: float,
    batch_size: int,
    device: str,
    image_size: int,
    confidence: float,
    keypoint_confidence: float,
    max_samples: int = 0,
    start_sec: float = 0.0,
    duration_sec: float = 0.0,
    adapter: PoseAdapter | None = None,
) -> dict[str, Any]:
    if sample_fps <= 0 or batch_size <= 0:
        raise ValueError("sample_fps and batch_size must be positive")
    if not video_path.is_file() or (adapter is None and not model_path.is_file()):
        raise FileNotFoundError("video and pose model must exist")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(source_fps / sample_fps)))
    start_frame = max(0, int(round(start_sec * source_fps)))
    end_frame = (
        min(frame_count, start_frame + int(round(duration_sec * source_fps)))
        if duration_sec > 0 and frame_count > 0
        else frame_count
    )
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    pose_adapter = adapter or UltralyticsPoseAdapter(
        model_path=str(model_path),
        device=device,
        image_size=image_size,
        confidence=confidence,
        keypoint_confidence=keypoint_confidence,
    )
    frames: list[Any] = []
    frame_numbers: list[int] = []
    detections: list[PerceptionDetectionResponse] = []
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
            if end_frame and frame_number >= end_frame:
                break
            if (frame_number - start_frame) % stride:
                continue
            frames.append(frame)
            frame_numbers.append(frame_number)
            sampled += 1
            if len(frames) >= batch_size:
                detections.extend(pose_adapter.estimate(frames, frame_numbers))
                frames.clear()
                frame_numbers.clear()
            if max_samples and sampled >= max_samples:
                break
        if frames:
            detections.extend(pose_adapter.estimate(frames, frame_numbers))
    finally:
        capture.release()
    return {
        "schema_version": "agu.official-pose.v1",
        "raw_video": {
            "filename": video_path.name,
            "sha256": _file_sha256(video_path),
            "size_bytes": video_path.stat().st_size,
            "source_fps": source_fps,
            "frame_count": frame_count,
        },
        "pose": {
            "backend": pose_adapter.name,
            "model_filename": model_path.name,
            "model_sha256": _file_sha256(model_path) if model_path.is_file() else "injected-test-model",
            "device": device,
            "image_size": image_size,
            "confidence": confidence,
            "keypoint_confidence": keypoint_confidence,
        },
        "sampling": {
            "requested_fps": sample_fps,
            "stride_frames": stride,
            "start_frame": start_frame,
            "end_frame": decoded,
            "sample_count": sampled,
        },
        "runtime_seconds": time.monotonic() - started,
        "detections": [item.model_dump(mode="json") for item in detections],
    }


def main() -> int:
    args = parse_args()
    payload = run_pose_scan(
        video_path=args.video,
        model_path=args.model,
        sample_fps=args.sample_fps,
        batch_size=args.batch_size,
        device=args.device,
        image_size=args.image_size,
        confidence=args.confidence,
        keypoint_confidence=args.keypoint_confidence,
        max_samples=args.max_samples,
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sampling": payload["sampling"], "poses": len(payload["detections"])}, indent=2))
    return 0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
