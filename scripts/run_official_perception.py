#!/usr/bin/env python3
"""Run the experimental official-stat perception adapter on raw video only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.perception import BallTracker, UltralyticsDetectorAdapter  # noqa: E402
from app.analysis.schemas import PerceptionDetectionResponse  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=704)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--track-players", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tracker-config", default="bytetrack.yaml")
    parser.add_argument(
        "--class-prompt",
        action="append",
        default=[],
        help="Optional repeatable open-vocabulary class prompt passed to a compatible model",
    )
    parser.add_argument(
        "--rim-max-center-y-ratio",
        type=float,
        default=1.0,
        help="Reject rim detections whose vertical center is below this normalized image height",
    )
    parser.add_argument(
        "--rim-min-aspect-ratio",
        type=float,
        default=0.0,
        help="Reject rim detections narrower than this width/height ratio",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=0.0)
    return parser.parse_args()


def run_perception_scan(
    *,
    video_path: Path,
    model_path: Path,
    sample_fps: float,
    batch_size: int,
    device: str,
    image_size: int,
    confidence: float,
    max_samples: int = 0,
    start_sec: float = 0.0,
    duration_sec: float = 0.0,
    track_players: bool = True,
    tracker_config: str = "bytetrack.yaml",
    class_prompts: tuple[str, ...] = (),
    rim_max_center_y_ratio: float = 1.0,
    rim_min_aspect_ratio: float = 0.0,
) -> dict[str, Any]:
    if sample_fps <= 0 or batch_size <= 0:
        raise ValueError("sample_fps and batch_size must be positive")
    if not 0 < rim_max_center_y_ratio <= 1:
        raise ValueError("rim_max_center_y_ratio must be in (0, 1]")
    if rim_min_aspect_ratio < 0:
        raise ValueError("rim_min_aspect_ratio must be non-negative")
    if not video_path.is_file() or not model_path.is_file():
        raise FileNotFoundError("video and detector model must exist")
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
    detector = UltralyticsDetectorAdapter(
        model_path=str(model_path),
        device=device,
        image_size=image_size,
        confidence=confidence,
        tracking=track_players,
        tracker_config=tracker_config,
        class_prompts=class_prompts,
    )
    detections: list[PerceptionDetectionResponse] = []
    player_color_features: dict[str, tuple[float, float, float]] = {}
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
            if end_frame and frame_number >= end_frame:
                break
            if (frame_number - start_frame) % stride:
                continue
            frames.append(frame)
            frame_numbers.append(frame_number)
            sampled += 1
            if len(frames) >= batch_size:
                batch_detections = _filter_rims_by_image_geometry(
                    list(detector.detect(frames, frame_numbers)),
                    frames,
                    frame_numbers,
                    max_center_y_ratio=rim_max_center_y_ratio,
                    min_aspect_ratio=rim_min_aspect_ratio,
                )
                player_color_features.update(
                    _extract_player_color_features(batch_detections, frames, frame_numbers)
                )
                detections.extend(batch_detections)
                frames.clear()
                frame_numbers.clear()
                print(
                    f"sampled={sampled} decoded={decoded}/{frame_count} detections={len(detections)}",
                    flush=True,
                )
            if max_samples and sampled >= max_samples:
                break
        if frames:
            batch_detections = _filter_rims_by_image_geometry(
                list(detector.detect(frames, frame_numbers)),
                frames,
                frame_numbers,
                max_center_y_ratio=rim_max_center_y_ratio,
                min_aspect_ratio=rim_min_aspect_ratio,
            )
            player_color_features.update(
                _extract_player_color_features(batch_detections, frames, frame_numbers)
            )
            detections.extend(batch_detections)
    finally:
        capture.release()

    detections = _assign_player_teams(detections, player_color_features)
    max_gap_frames = max(stride * 3, 4)
    ball_tracks = BallTracker(max_distance_px=120.0, max_gap_frames=max_gap_frames).track(detections)
    interpolated_tracks = [
        BallTracker(max_distance_px=120.0, max_gap_frames=max_gap_frames).interpolate(track)
        for track in ball_tracks
    ]
    counts = {
        object_type: sum(item.object_type == object_type for item in detections)
        for object_type in ("player", "basketball", "rim", "backboard", "referee")
    }
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
            "backend": detector.name,
            "model_filename": model_path.name,
            "model_sha256": _file_sha256(model_path),
            "device": device,
            "image_size": image_size,
            "confidence": confidence,
            "player_tracking": track_players,
            "tracker_config": tracker_config if track_players else "",
            "class_prompts": list(class_prompts),
            "rim_max_center_y_ratio": rim_max_center_y_ratio,
            "rim_min_aspect_ratio": rim_min_aspect_ratio,
            "team_assignment": "two_cluster_torso_lab_v1",
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
        "counts": counts,
        "detections": [item.model_dump(mode="json") for item in detections],
        "ball_tracks": [track.model_dump(mode="json") for track in interpolated_tracks],
    }


def main() -> int:
    args = parse_args()
    payload = run_perception_scan(
        video_path=args.video,
        model_path=args.model,
        sample_fps=args.sample_fps,
        batch_size=args.batch_size,
        device=args.device,
        image_size=args.image_size,
        confidence=args.confidence,
        max_samples=args.max_samples,
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
        track_players=args.track_players,
        tracker_config=args.tracker_config,
        class_prompts=tuple(args.class_prompt),
        rim_max_center_y_ratio=args.rim_max_center_y_ratio,
        rim_min_aspect_ratio=args.rim_min_aspect_ratio,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("runtime_seconds", "counts", "sampling")}, indent=2))
    return 0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _filter_rims_by_image_geometry(
    detections: list[PerceptionDetectionResponse],
    frames: list[Any],
    frame_numbers: list[int],
    *,
    max_center_y_ratio: float,
    min_aspect_ratio: float = 0.0,
) -> list[PerceptionDetectionResponse]:
    """Apply an optional camera-geometry prior without introducing event labels."""
    height_by_frame = {
        frame_number: int(frame.shape[0]) for frame, frame_number in zip(frames, frame_numbers)
    }
    output: list[PerceptionDetectionResponse] = []
    for detection in detections:
        if detection.object_type != "rim":
            output.append(detection)
            continue
        height = height_by_frame.get(detection.frame)
        if height is not None and ((detection.bbox.y1 + detection.bbox.y2) / 2.0) > (
            height * max_center_y_ratio
        ):
            continue
        box_height = max(1e-6, detection.bbox.y2 - detection.bbox.y1)
        aspect_ratio = (detection.bbox.x2 - detection.bbox.x1) / box_height
        if aspect_ratio < min_aspect_ratio:
            continue
        output.append(detection)
    return output


def _extract_player_color_features(
    detections: list[PerceptionDetectionResponse],
    frames: list[Any],
    frame_numbers: list[int],
) -> dict[str, tuple[float, float, float]]:
    frame_by_number = {number: frame for number, frame in zip(frame_numbers, frames)}
    features: dict[str, tuple[float, float, float]] = {}
    for detection in detections:
        if detection.object_type != "player":
            continue
        frame = frame_by_number.get(detection.frame)
        if frame is None:
            continue
        height, width = frame.shape[:2]
        box = detection.bbox
        x1 = max(0, min(width - 1, int(round(box.x1 + (box.x2 - box.x1) * 0.20))))
        x2 = max(x1 + 1, min(width, int(round(box.x2 - (box.x2 - box.x1) * 0.20))))
        y1 = max(0, min(height - 1, int(round(box.y1 + (box.y2 - box.y1) * 0.18))))
        y2 = max(y1 + 1, min(height, int(round(box.y1 + (box.y2 - box.y1) * 0.58))))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)
        median = np.median(lab, axis=0)
        features[detection.detection_id] = tuple(float(value) for value in median)
    return features


def _assign_player_teams(
    detections: list[PerceptionDetectionResponse],
    features: dict[str, tuple[float, float, float]],
) -> list[PerceptionDetectionResponse]:
    player_items = [item for item in detections if item.detection_id in features]
    if len(player_items) < 2:
        return detections
    matrix = np.asarray([features[item.detection_id] for item in player_items], dtype=np.float32)
    _compactness, labels, centers = cv2.kmeans(
        matrix,
        2,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2),
        5,
        cv2.KMEANS_PP_CENTERS,
    )
    dark_label = int(np.argmin(centers[:, 0]))
    detected_team = {
        item.detection_id: ("raw-dark" if int(label[0]) == dark_label else "raw-light")
        for item, label in zip(player_items, labels)
    }
    votes: dict[str, list[str]] = {}
    for item in player_items:
        if item.player_id:
            votes.setdefault(item.player_id, []).append(detected_team[item.detection_id])
    stable_team = {
        player_id: max(set(team_votes), key=lambda value: (team_votes.count(value), value))
        for player_id, team_votes in votes.items()
    }
    return [
        item.model_copy(
            update={
                "team_id": stable_team.get(item.player_id, detected_team.get(item.detection_id))
                if item.object_type == "player"
                else item.team_id
            }
        )
        for item in detections
    ]


if __name__ == "__main__":
    raise SystemExit(main())
