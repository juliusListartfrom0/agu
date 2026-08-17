#!/usr/bin/env python3
"""Find repeated full-screen replay logos among TransNetV2 cut frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.replay_transition import (  # noqa: E402
    REPLAY_LOGO_RECURRENCE_SCHEMA,
    best_logo_recurrence_match,
    seal_replay_logo_artifact,
    verify_replay_transition_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transition-evidence", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--detector-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan-fps", type=float, default=2.0)
    parser.add_argument("--exclusion-seconds", type=float, default=30.0)
    parser.add_argument("--person-confidence", type=float, default=0.25)
    parser.add_argument("--minimum-correlation", type=float, default=0.7)
    parser.add_argument("--maximum-mean-absolute-difference", type=float, default=0.12)
    return parser.parse_args()


def decode_timeline(
    video: Path,
    *,
    scan_fps: float,
    width: int = 48,
    height: int = 27,
) -> np.ndarray:
    if scan_fps <= 0:
        raise ValueError("replay logo scan FPS must be positive")
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"fps={scan_fps},scale={width}:{height}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    frame_bytes = width * height * 3
    if not completed.stdout or len(completed.stdout) % frame_bytes:
        raise RuntimeError("ffmpeg returned an incomplete replay-logo timeline")
    return (
        np.frombuffer(completed.stdout, dtype=np.uint8)
        .reshape(-1, height, width, 3)
        .astype(np.float32)
        / 255.0
    )


def main() -> int:
    args = parse_args()
    if (
        not 0.0 < args.person_confidence < 1.0
        or not 0.0 < args.minimum_correlation <= 1.0
        or args.maximum_mean_absolute_difference <= 0
    ):
        raise ValueError("replay logo recurrence configuration is invalid")
    transition = verify_replay_transition_artifact(
        json.loads(args.transition_evidence.read_text(encoding="utf-8"))
    )
    video_sha256 = _file_sha256(args.video)
    if transition.get("raw_video_sha256") != video_sha256:
        raise ValueError("transition evidence does not match the raw video")
    if _file_sha256(args.detector_checkpoint) != args.detector_checkpoint_sha256:
        raise ValueError("replay logo detector checkpoint hash mismatch")
    timeline = decode_timeline(args.video, scan_fps=args.scan_fps)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    if fps <= 0:
        capture.release()
        raise RuntimeError("video FPS is unavailable")
    frame_to_image: dict[int, np.ndarray] = {}
    try:
        for event in transition["events"]:
            for frame_number in event["transition_frames"]:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
                ok, image = capture.read()
                if ok and image is not None:
                    frame_to_image[int(frame_number)] = image
    finally:
        capture.release()
    ordered_frames = sorted(frame_to_image)
    detector = YOLO(str(args.detector_checkpoint))
    detections = detector.predict(
        [frame_to_image[frame] for frame in ordered_frames],
        conf=args.person_confidence,
        verbose=False,
    )
    person_counts = {
        frame: sum(int(class_id) == 0 for class_id in result.boxes.cls)
        for frame, result in zip(ordered_frames, detections, strict=True)
    }
    events = []
    for event in transition["events"]:
        candidates = []
        for frame_number in event["transition_frames"]:
            frame_number = int(frame_number)
            image = frame_to_image.get(frame_number)
            if image is None:
                continue
            query = (
                cv2.cvtColor(
                    cv2.resize(image, (48, 27), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2RGB,
                ).astype(np.float32)
                / 255.0
            )
            recurrence = best_logo_recurrence_match(
                query_rgb=query,
                timeline_rgb=timeline,
                query_seconds=frame_number / fps,
                scan_fps=args.scan_fps,
                exclusion_seconds=args.exclusion_seconds,
            )
            is_logo = (
                person_counts[frame_number] == 0
                and recurrence["correlation"] >= args.minimum_correlation
                and recurrence["mean_absolute_difference"]
                <= args.maximum_mean_absolute_difference
            )
            candidates.append(
                {
                    "frame": frame_number,
                    "person_count": person_counts[frame_number],
                    "matched_seconds": recurrence["matched_seconds"],
                    "recurrence_correlation": recurrence["correlation"],
                    "recurrence_mean_absolute_difference": recurrence[
                        "mean_absolute_difference"
                    ],
                    "repeated_full_screen_logo": is_logo,
                }
            )
        events.append(
            {
                "event_id": event["event_id"],
                "candidates": candidates,
                "repeated_logo_count": sum(
                    bool(candidate["repeated_full_screen_logo"])
                    for candidate in candidates
                ),
                "broadcast_state": (
                    "replay"
                    if any(
                        candidate["repeated_full_screen_logo"]
                        for candidate in candidates
                    )
                    else "unknown"
                ),
            }
        )
    artifact = seal_replay_logo_artifact(
        {
            "schema_version": REPLAY_LOGO_RECURRENCE_SCHEMA,
            "purpose": "offline_raw_only_repeated_replay_logo_screening",
            "runtime_consumable": False,
            "codex_runtime_answer_used": False,
            "raw_video_sha256": video_sha256,
            "candidate_bundle_sha256": transition["candidate_bundle_sha256"],
            "transition_evidence_artifact_sha256": transition["artifact_sha256"],
            "detector": {
                "name": "Ultralytics YOLOv8n COCO",
                "checkpoint_sha256": args.detector_checkpoint_sha256,
                "person_class_id": 0,
            },
            "configuration": {
                "scan_fps": args.scan_fps,
                "exclusion_seconds": args.exclusion_seconds,
                "person_confidence": args.person_confidence,
                "minimum_correlation": args.minimum_correlation,
                "maximum_mean_absolute_difference": (
                    args.maximum_mean_absolute_difference
                ),
            },
            "events": events,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact_sha256": artifact["artifact_sha256"],
                "events": len(events),
                "replay": sum(
                    event["broadcast_state"] == "replay" for event in events
                ),
            }
        )
    )
    return 0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
