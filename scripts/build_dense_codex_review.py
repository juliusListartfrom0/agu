#!/usr/bin/env python3
"""Build continuous raw-video contact sheets for offline Codex event review."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=0.0)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--panel-width", type=int, default=480)
    return parser.parse_args()


def build_dense_review_package(
    *,
    video_path: Path,
    output_dir: Path,
    window_sec: float,
    sample_fps: float,
    start_sec: float = 0.0,
    duration_sec: float = 0.0,
    columns: int = 5,
    panel_width: int = 480,
) -> dict[str, object]:
    if window_sec <= 0 or sample_fps <= 0 or columns <= 0 or panel_width <= 0:
        raise ValueError("window, sampling, columns and panel width must be positive")
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_duration = total_frames / fps if total_frames else 0.0
    package_end = min(total_duration, start_sec + duration_sec) if duration_sec > 0 else total_duration
    if package_end <= start_sec:
        capture.release()
        raise ValueError("review interval is empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir = output_dir / "sheets"
    sheets_dir.mkdir(exist_ok=True)
    windows: list[dict[str, object]] = []
    sample_step = 1.0 / sample_fps
    panel_height = int(round(panel_width * 9 / 16))

    try:
        window_start = start_sec
        index = 1
        while window_start < package_end - 1e-6:
            window_end = min(package_end, window_start + window_sec)
            frames = []
            sample_times = []
            sample_time = window_start
            while sample_time < window_end - 1e-6:
                capture.set(cv2.CAP_PROP_POS_MSEC, sample_time * 1000.0)
                ok, frame = capture.read()
                if ok:
                    panel = cv2.resize(frame, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
                    cv2.rectangle(panel, (0, 0), (150, 30), (0, 0, 0), -1)
                    cv2.putText(
                        panel,
                        f"{sample_time:.2f}s",
                        (6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    frames.append(panel)
                    sample_times.append(round(sample_time, 3))
                sample_time += sample_step
            sheet_path = sheets_dir / f"window_{index:05d}.jpg"
            _write_sheet(frames, sheet_path, columns, panel_width, panel_height)
            windows.append(
                {
                    "window_id": f"window_{index:05d}",
                    "start_sec": round(window_start, 3),
                    "end_sec": round(window_end, 3),
                    "sample_times_sec": sample_times,
                    "contact_sheet": str(sheet_path.relative_to(output_dir)),
                    "status": "pending_codex_review",
                }
            )
            print(f"built {windows[-1]['window_id']} {window_start:.1f}-{window_end:.1f}s", flush=True)
            index += 1
            window_start = window_end
    finally:
        capture.release()

    manifest = {
        "schema_version": "agu.dense-codex-review.v1",
        "raw_video": {
            "filename": video_path.name,
            "sha256": _file_sha256(video_path),
            "size_bytes": video_path.stat().st_size,
            "fps": fps,
            "frame_count": total_frames,
        },
        "coverage": {
            "start_sec": start_sec,
            "end_sec": package_end,
            "window_sec": window_sec,
            "sample_fps": sample_fps,
            "window_count": len(windows),
            "gap_seconds": 0.0,
        },
        "windows": windows,
        "review_rules": [
            "Review every window; windows are continuous and non-overlapping.",
            "Use finer raw frames/clip when a contact sheet cannot prove actor, outcome or causal relation.",
            "Do not infer assist, block, steal or rebound from pose alone.",
            "Append decisions; never modify manifest or sheets.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "codex_events.jsonl").touch(exist_ok=True)
    return manifest


def _write_sheet(frames: list[object], path: Path, columns: int, width: int, height: int) -> None:
    if not frames:
        raise ValueError("contact sheet contains no readable frames")
    blank = frames[0] * 0
    rows = math.ceil(len(frames) / columns)
    padded = frames + [blank] * (rows * columns - len(frames))
    sheet_rows = [cv2.hconcat(padded[index : index + columns]) for index in range(0, len(padded), columns)]
    sheet = cv2.vconcat(sheet_rows)
    if not cv2.imwrite(str(path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 88]):
        raise RuntimeError(f"failed to write contact sheet: {path}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    manifest = build_dense_review_package(
        video_path=args.video,
        output_dir=args.output_dir,
        window_sec=args.window_sec,
        sample_fps=args.sample_fps,
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
        columns=args.columns,
        panel_width=args.panel_width,
    )
    print(json.dumps(manifest["coverage"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
