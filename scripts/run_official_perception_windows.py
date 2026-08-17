#!/usr/bin/env python3
"""Run a detector only over sealed official-candidate video windows."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.ball_candidate_review import seal_artifact  # noqa: E402
from app.analysis.official_evaluation import verify_raw_only_bundle  # noqa: E402
from app.analysis.perception import (  # noqa: E402
    BallTracker,
    DetectorAdapter,
    RFDETRDetectorAdapter,
    TransformersRFDETRDetectorAdapter,
    UltralyticsDetectorAdapter,
)
from app.analysis.schemas import (  # noqa: E402
    PerceptionDetectionResponse,
    RawOnlyPredictionBundleResponse,
)
from scripts.run_official_perception import _file_sha256  # noqa: E402
from scripts.run_official_pose_windows import _merge_windows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("ultralytics_yolo", "rfdetr", "transformers_rfdetr"),
        default="ultralytics_yolo",
    )
    parser.add_argument("--event-type", action="append", default=[])
    parser.add_argument("--object-type", action="append", default=[])
    parser.add_argument("--padding-sec", type=float, default=0.0)
    parser.add_argument("--merge-gap-sec", type=float, default=0.5)
    parser.add_argument("--sample-fps", type=float, default=15.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def run_perception_windows(
    *,
    candidate_bundle: RawOnlyPredictionBundleResponse,
    video_path: Path,
    model_path: Path,
    event_types: set[str] | None,
    object_types: set[str] | None,
    padding_sec: float,
    merge_gap_sec: float,
    sample_fps: float,
    batch_size: int,
    device: str,
    image_size: int,
    confidence: float,
    backend: str = "ultralytics_yolo",
    max_samples: int = 0,
    adapter: DetectorAdapter | None = None,
) -> dict[str, Any]:
    """Run one backend over deduplicated windows from a sealed bundle."""

    source = verify_raw_only_bundle(candidate_bundle)
    if min(sample_fps, float(batch_size), float(image_size)) <= 0:
        raise ValueError("sample_fps, batch_size and image_size must be positive")
    if padding_sec < 0 or merge_gap_sec < 0 or max_samples < 0:
        raise ValueError("window bounds and max_samples must be non-negative")
    if not 0 < confidence <= 1:
        raise ValueError("confidence must be in (0,1]")
    if not video_path.is_file() or (adapter is None and not model_path.exists()):
        raise FileNotFoundError("video and detector model must exist")
    video_sha256 = _file_sha256(video_path)
    matching_assets = [
        item
        for item in source.raw_videos
        if item.filename == video_path.name and item.sha256 == video_sha256
    ]
    if len(matching_assets) != 1:
        raise ValueError("candidate bundle must contain exactly one matching raw video")
    source_video_id = matching_assets[0].video_id

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(source_fps / sample_fps)))
    padding_frames = int(round(padding_sec * source_fps))
    merge_gap_frames = int(round(merge_gap_sec * source_fps))
    selected_events = [
        event
        for event in source.events
        if event.source_video_id == source_video_id
        and (not event_types or event.event_type in event_types)
        and event.status != "rejected"
    ]
    windows = _merge_windows(
        [
            (
                max(0, event.start_frame - padding_frames),
                (
                    min(frame_count, event.end_frame + padding_frames + 1)
                    if frame_count
                    else event.end_frame + padding_frames + 1
                ),
            )
            for event in selected_events
        ],
        maximum_gap=merge_gap_frames,
    )
    if not windows:
        capture.release()
        raise ValueError("candidate bundle contains no matching event windows")

    if backend not in {"ultralytics_yolo", "rfdetr", "transformers_rfdetr"}:
        raise ValueError(f"unsupported detector backend: {backend}")
    if adapter is not None:
        detector = adapter
    elif backend == "transformers_rfdetr":
        detector = TransformersRFDETRDetectorAdapter(
            model_path=str(model_path),
            device=device,
            confidence=confidence,
        )
    elif backend == "rfdetr":
        detector = RFDETRDetectorAdapter(
            model_path=str(model_path),
            device=device,
            image_size=image_size,
            confidence=confidence,
        )
    else:
        detector = UltralyticsDetectorAdapter(
            model_path=str(model_path),
            device=device,
            image_size=image_size,
            confidence=confidence,
            tracking=False,
        )
    model_artifact_path = (
        _resolve_model_artifact_path(model_path)
        if adapter is None
        else None
    )
    frames: list[Any] = []
    frame_numbers: list[int] = []
    detections: list[PerceptionDetectionResponse] = []
    sampled = 0
    started = time.monotonic()

    def flush() -> None:
        if not frames:
            return
        batch = list(detector.detect(frames, frame_numbers))
        if object_types:
            batch = [item for item in batch if item.object_type in object_types]
        detections.extend(batch)
        frames.clear()
        frame_numbers.clear()

    try:
        stop = False
        for start_frame, end_frame in windows:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_number = start_frame
            while frame_number < end_frame:
                ok, frame = capture.read()
                if not ok:
                    break
                if (frame_number - start_frame) % stride == 0:
                    frames.append(frame)
                    frame_numbers.append(frame_number)
                    sampled += 1
                    if len(frames) >= batch_size:
                        flush()
                    if max_samples and sampled >= max_samples:
                        stop = True
                        break
                frame_number += 1
            if stop:
                break
        flush()
    finally:
        capture.release()

    maximum_track_gap = max(stride * 3, 4)
    tracker = BallTracker(max_distance_px=120.0, max_gap_frames=maximum_track_gap)
    ball_tracks = [
        tracker.interpolate(track) for track in tracker.track(detections)
    ]
    counts = {
        object_type: sum(item.object_type == object_type for item in detections)
        for object_type in ("player", "basketball", "rim", "backboard", "referee")
    }
    payload = {
        "schema_version": "agu.official-perception.v1",
        "raw_video": {
            "filename": video_path.name,
            "sha256": video_sha256,
            "size_bytes": video_path.stat().st_size,
            "source_fps": source_fps,
            "frame_count": frame_count,
        },
        "detector": {
            "backend": detector.name,
            "model_filename": (
                model_artifact_path.name
                if model_artifact_path is not None
                else model_path.name
            ),
            "model_sha256": (
                _file_sha256(model_artifact_path)
                if model_artifact_path is not None
                else "injected-test-model"
            ),
            "device": device,
            "image_size": image_size,
            "confidence": confidence,
            "player_tracking": False,
            "tracker_config": "",
            "class_prompts": [],
            "object_type_allowlist": sorted(object_types or []),
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
        "candidate_source": {
            "bundle_sha256": source.bundle_sha256,
            "source_video_id": source_video_id,
            "event_types": sorted(event_types or []),
            "event_count": len(selected_events),
            "padding_sec": padding_sec,
            "merge_gap_sec": merge_gap_sec,
        },
        "runtime_seconds": time.monotonic() - started,
        "counts": counts,
        "detections": [item.model_dump(mode="json") for item in detections],
        "ball_tracks": [track.model_dump(mode="json") for track in ball_tracks],
    }
    return seal_artifact(payload)


def _resolve_model_artifact_path(model_path: Path) -> Path:
    if model_path.is_file():
        return model_path
    weights = sorted(model_path.glob("*.safetensors"))
    if len(weights) != 1:
        raise ValueError(
            "detector model directory must contain exactly one safetensors weight"
        )
    return weights[0]


def main() -> int:
    args = parse_args()
    source = RawOnlyPredictionBundleResponse.model_validate_json(
        args.candidate_bundle.read_text(encoding="utf-8")
    )
    payload = run_perception_windows(
        candidate_bundle=source,
        video_path=args.video,
        model_path=args.model,
        event_types=set(args.event_type) or None,
        object_types=set(args.object_type) or None,
        padding_sec=args.padding_sec,
        merge_gap_sec=args.merge_gap_sec,
        sample_fps=args.sample_fps,
        batch_size=args.batch_size,
        device=args.device,
        image_size=args.image_size,
        confidence=args.confidence,
        backend=args.backend,
        max_samples=args.max_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "windows": len(payload["sampling"]["windows"]),
                "samples": payload["sampling"]["sample_count"],
                "counts": payload["counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
