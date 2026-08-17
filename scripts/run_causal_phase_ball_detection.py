#!/usr/bin/env python3
"""Run BODD only on hash-bound full causal review windows."""

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

from app.analysis.causal_phase_ball_windows import (  # noqa: E402
    causal_ball_windows,
)
from app.analysis.causal_shot_phase_review import (  # noqa: E402
    verify_causal_shot_phase_review_plan,
)
from app.analysis.perception import (  # noqa: E402
    BallTracker,
    TransformersRFDETRDetectorAdapter,
    UltralyticsDetectorAdapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("ultralytics_yolo", "transformers_rfdetr"),
        default="ultralytics_yolo",
    )
    parser.add_argument("--sample-fps", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--image-size", type=int, default=704)
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument("--merge-gap-sec", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def run_detection(
    *,
    plan_path: Path,
    video_path: Path,
    model_path: Path,
    sample_fps: float,
    batch_size: int,
    device: str,
    image_size: int,
    confidence: float,
    merge_gap_sec: float,
    max_samples: int,
    backend: str = "ultralytics_yolo",
) -> dict[str, Any]:
    if (
        min(sample_fps, float(batch_size), float(image_size)) <= 0
        or merge_gap_sec < 0
        or max_samples < 0
        or not 0 < confidence <= 1
    ):
        raise ValueError("invalid causal ball detection parameters")
    plan = verify_causal_shot_phase_review_plan(_read_json(plan_path))
    source_sha = _file_sha256(video_path)
    if (
        source_sha not in plan["source_video_sha256s"]
        or source_sha in plan["sealed_blind_video_sha256s"]
    ):
        raise ValueError("sealed blind or unplanned video entered ball detection")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open causal ball video: {video_path.name}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, round(source_fps / sample_fps))
    windows, review_ids = causal_ball_windows(
        plan["examples"],
        source_video_sha256=source_sha,
        merge_gap_frames=round(merge_gap_sec * source_fps),
    )
    if any(start < 0 or end > frame_count for start, end in windows):
        capture.release()
        raise ValueError("causal ball window exceeds source video")
    detector = _build_detector(
        backend=backend,
        model_path=model_path,
        device=device,
        image_size=image_size,
        confidence=confidence,
    )
    model_artifact_path = _resolve_model_artifact_path(model_path)
    frames = []
    frame_numbers = []
    detections = []
    sampled = 0
    started = time.monotonic()

    def flush() -> None:
        if not frames:
            return
        detections.extend(
            item
            for item in detector.detect(frames, frame_numbers)
            if item.object_type == "basketball"
        )
        frames.clear()
        frame_numbers.clear()

    try:
        stop = False
        for window_index, (start, end) in enumerate(windows, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            frame_number = start
            while frame_number < end:
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(
                        f"cannot decode causal ball frame {frame_number}"
                    )
                if (frame_number - start) % stride == 0:
                    frames.append(frame)
                    frame_numbers.append(frame_number)
                    sampled += 1
                    if len(frames) >= batch_size:
                        flush()
                    if max_samples and sampled >= max_samples:
                        stop = True
                        break
                frame_number += 1
            if window_index % 5 == 0 or window_index == len(windows):
                print(
                    json.dumps(
                        {
                            "stage": "causal_ball_detection",
                            "window": window_index,
                            "window_count": len(windows),
                            "sampled": sampled,
                            "detections": len(detections),
                        }
                    ),
                    flush=True,
                )
            if stop:
                break
        flush()
    finally:
        capture.release()
    tracker = BallTracker(
        max_distance_px=120.0,
        max_gap_frames=max(stride * 3, 4),
    )
    ball_tracks = [
        tracker.interpolate(track) for track in tracker.track(detections)
    ]
    artifact: dict[str, Any] = {
        "schema_version": "agu.official-perception.v1",
        "raw_video": {
            "filename": video_path.name,
            "sha256": source_sha,
            "size_bytes": video_path.stat().st_size,
            "source_fps": source_fps,
            "frame_count": frame_count,
        },
        "detector": {
            "backend": detector.name,
            "model_filename": model_artifact_path.name,
            "model_sha256": _file_sha256(model_artifact_path),
            "device": device,
            "image_size": image_size,
            "confidence": confidence,
            "player_tracking": False,
            "tracker_config": "",
            "class_prompts": [],
            "object_type_allowlist": ["basketball"],
            "rim_max_center_y_ratio": 1.0,
            "rim_min_aspect_ratio": 0.0,
            "team_assignment": "none",
        },
        "sampling": {
            "requested_fps": sample_fps,
            "stride_frames": stride,
            "start_frame": windows[0][0],
            "end_frame": windows[-1][1],
            "sample_count": sampled,
            "decoded_frame_count": sum(end - start for start, end in windows),
            "windows": [
                {"start_frame": start, "end_frame": end}
                for start, end in windows
            ],
        },
        "causal_source": {
            "review_plan_sha256": plan["artifact_sha256"],
            "phase_review_ids": review_ids,
            "event_count": len(review_ids),
            "merge_gap_sec": merge_gap_sec,
            "complete_run": max_samples == 0,
        },
        "runtime_seconds": time.monotonic() - started,
        "counts": {
            "player": 0,
            "basketball": len(detections),
            "rim": 0,
            "backboard": 0,
            "referee": 0,
        },
        "detections": [item.model_dump(mode="json") for item in detections],
        "ball_tracks": [
            track.model_dump(mode="json") for track in ball_tracks
        ],
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    args = parse_args()
    artifact = run_detection(
        plan_path=args.plan,
        video_path=args.video,
        model_path=args.model,
        sample_fps=args.sample_fps,
        batch_size=args.batch_size,
        device=args.device,
        image_size=args.image_size,
        confidence=args.confidence,
        merge_gap_sec=args.merge_gap_sec,
        max_samples=args.max_samples,
        backend=args.backend,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
    temporary.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "samples": artifact["sampling"]["sample_count"],
                "detections": artifact["counts"]["basketball"],
                "tracks": len(artifact["ball_tracks"]),
                "artifact_sha256": artifact["artifact_sha256"],
            }
        )
    )
    return 0


def _build_detector(
    *,
    backend: str,
    model_path: Path,
    device: str,
    image_size: int,
    confidence: float,
):
    if backend == "ultralytics_yolo":
        return UltralyticsDetectorAdapter(
            model_path=str(model_path),
            device=device,
            image_size=image_size,
            confidence=confidence,
            tracking=False,
        )
    if backend == "transformers_rfdetr":
        return TransformersRFDETRDetectorAdapter(
            model_path=str(model_path),
            device=device,
            confidence=confidence,
        )
    raise ValueError(f"unsupported causal ball detector backend: {backend}")


def _resolve_model_artifact_path(model_path: Path) -> Path:
    artifact_path = (
        model_path / "model.safetensors" if model_path.is_dir() else model_path
    )
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"detector artifact does not exist: {artifact_path}"
        )
    return artifact_path


if __name__ == "__main__":
    raise SystemExit(main())
