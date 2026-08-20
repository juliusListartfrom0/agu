#!/usr/bin/env python3
"""Run a hash-bound, offline DeepBall-Large basketball frame probe."""

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

from app.analysis.deepball_large import (  # noqa: E402
    infer_rgb_frame,
    load_pinned_deepball_large_model,
)

DEFAULT_SOURCE = ROOT / "dataset/public_sources/open_models/wasb_sbdt_source/src/models/deepball.py"
DEFAULT_CONFIG = ROOT / "dataset/public_sources/open_models/wasb_sbdt_source/src/configs/model/deepball_large.yaml"
DEFAULT_MODEL = ROOT / "model_checkpoints/wasb_sbdt_official/deepball-large_basketball_best.pth.tar"
DEFAULT_SOURCE_SHA256 = "e8e46a457cb1ca7bb12ba356049d58a320cd3413c948ec534e32abecd2ee7f3a"
DEFAULT_MODEL_SHA256 = "387878a2f45e6d9e1e3c2867c87f454b7d9676e6e546bd979301b52ec2832dfd"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=0, help="Exclusive; zero means EOF")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=0, help="Zero means no cap")
    return parser.parse_args()


def run_probe(
    *,
    video_path: Path,
    output_path: Path | None = None,
    source_path: Path = DEFAULT_SOURCE,
    config_path: Path = DEFAULT_CONFIG,
    model_path: Path = DEFAULT_MODEL,
    sample_fps: float = 1.0,
    start_frame: int = 0,
    end_frame: int = 0,
    score_threshold: float = 0.5,
    device: str = "cpu",
    max_frames: int = 0,
) -> dict[str, Any]:
    if sample_fps <= 0 or start_frame < 0 or end_frame < 0 or max_frames < 0:
        raise ValueError("invalid sampling bounds")
    if not 0.0 < score_threshold < 1.0:
        raise ValueError("score_threshold must be between zero and one")
    for path in (video_path, source_path, config_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(source_fps / sample_fps)))
    end = end_frame if end_frame > 0 else frame_count
    loaded = load_pinned_deepball_large_model(
        source_path=source_path,
        config_path=config_path,
        checkpoint_path=model_path,
        expected_source_sha256=DEFAULT_SOURCE_SHA256,
        expected_checkpoint_sha256=DEFAULT_MODEL_SHA256,
        device=device,
    )

    rows: list[dict[str, Any]] = []
    decoded = start_frame
    sampled = 0
    started = time.monotonic()
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
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
            detections = infer_rgb_frame(
                loaded,
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                score_threshold=score_threshold,
            )
            rows.append({"frame": frame_number, "detections": detections})
            sampled += 1
            if sampled % 25 == 0:
                print(f"sampled={sampled} decoded={decoded}/{frame_count}", flush=True)
            if max_frames and sampled >= max_frames:
                break
    finally:
        capture.release()

    payload: dict[str, Any] = {
        "schema_version": "agu.deepball-large-probe.v1",
        "purpose": "offline_cross_broadcast_ball_detector_comparison",
        "runtime_consumable": False,
        "training_eligible": False,
        "codex_runtime_answer_used": False,
        "raw_video": {
            "filename": video_path.name,
            "sha256": _file_sha256(video_path),
            "size_bytes": video_path.stat().st_size,
            "source_fps": source_fps,
            "frame_count": frame_count,
        },
        "model": {
            "backend": "wasb_deepball_large_basketball_v1",
            "source_sha256": loaded.source_sha256,
            "checkpoint_sha256": loaded.checkpoint_sha256,
            "device": device,
            "score_threshold": score_threshold,
        },
        "sampling": {
            "requested_fps": sample_fps,
            "stride_frames": stride,
            "start_frame": start_frame,
            "end_frame": end,
            "decoded_frame_count": decoded,
            "sample_count": sampled,
        },
        "runtime_seconds": time.monotonic() - started,
        "detections": rows,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return payload


def main() -> int:
    args = parse_args()
    payload = run_probe(
        video_path=args.video,
        output_path=args.output,
        source_path=args.source,
        config_path=args.config,
        model_path=args.model,
        sample_fps=args.sample_fps,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        score_threshold=args.score_threshold,
        device=args.device,
        max_frames=args.max_frames,
    )
    print(
        json.dumps(
            {
                "runtime_seconds": payload["runtime_seconds"],
                "sampling": payload["sampling"],
                "detections": sum(len(row["detections"]) for row in payload["detections"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
