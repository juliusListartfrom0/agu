#!/usr/bin/env python3
"""Prepare a label-hidden, hash-bound cross-game shot-window review set."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.cross_game_shot_windows import (  # noqa: E402
    seal_shot_window_spec,
    select_uniform_shot_windows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-prefix", default="raw-shot")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--window-frames", type=int, default=120)
    parser.add_argument("--minimum-center-spacing-frames", type=int, default=1800)
    parser.add_argument("--panels-per-sheet", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if args.panels_per_sheet != 12:
        raise ValueError("the contact-sheet renderer currently requires 12 panels")
    capture = cv2.VideoCapture(str(args.video))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("could not read video geometry")
    source_sha = _file_sha256(args.video)
    candidates = select_uniform_shot_windows(
        frame_count=frame_count,
        fps=fps,
        count=args.count,
        window_frames=args.window_frames,
        minimum_center_spacing_frames=args.minimum_center_spacing_frames,
        seed=args.seed,
    )
    for row in candidates:
        ordinal = int(str(row["event_id"]).rsplit("-", 1)[-1])
        row["event_id"] = f"{args.event_prefix}-{ordinal:03d}"
    spec = seal_shot_window_spec(
        {
            "schema_version": "agu.cross-game-shot-window-spec.v1",
            "purpose": "offline_training_annotation_only",
            "runtime_consumable": False,
            "source_video_sha256": source_sha,
            "source_video_filename": args.video.name,
            "video_frame_count": frame_count,
            "video_fps": fps,
            "video_width": width,
            "video_height": height,
            "selection_seed": args.seed,
            "selection_protocol": "seeded_uniform_nonoverlap_windows_v1",
            "window_frames": args.window_frames,
            "minimum_center_spacing_frames": args.minimum_center_spacing_frames,
            "label_selection_not_used": True,
            "candidates": candidates,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidate_spec_v1.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    sheets_dir = args.output_dir / "raw_sheets"
    sheets_dir.mkdir(exist_ok=True)
    _render_sheets(args.video, candidates, sheets_dir, fps=fps)
    template = {
        "schema_version": "agu.cross-game-shot-window-labels.v1",
        "purpose": "offline_training_annotation_only",
        "runtime_consumable": False,
        "source_video_sha256": source_sha,
        "candidate_spec_sha256": spec["artifact_sha256"],
        "review_protocol": "raw-frame-contact-sheet-and-strip; labels hidden from selection",
        "examples": [
            {
                "event_id": row["event_id"],
                "event_present": None,
                "review_note": "pending",
            }
            for row in candidates
        ],
    }
    (args.output_dir / "labels_template_v1.json").write_text(
        json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "source_video_sha256": source_sha,
                "candidate_spec_sha256": spec["artifact_sha256"],
                "candidates": len(candidates),
                "sheets": (len(candidates) + 11) // 12,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _render_sheets(
    video_path: Path,
    candidates: list[dict[str, object]],
    output_dir: Path,
    *,
    fps: float,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    try:
        frames: dict[int, np.ndarray] = {}
        for row in candidates:
            start = int(row["start_frame"])
            end = int(row["end_frame"])
            for frame_index in np.linspace(start, end, 6).astype(int):
                if frame_index in frames:
                    continue
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"could not decode frame {frame_index}")
                frames[frame_index] = frame
    finally:
        capture.release()
    for sheet_number in range((len(candidates) + 11) // 12):
        canvas = np.full((900, 1920, 3), 18, dtype=np.uint8)
        for panel_index, row in enumerate(candidates[sheet_number * 12 : (sheet_number + 1) * 12]):
            panel = np.full((300, 480, 3), 18, dtype=np.uint8)
            strip_frames = np.linspace(
                int(row["start_frame"]), int(row["end_frame"]), 6
            ).astype(int)
            for strip_index, frame_index in enumerate(strip_frames):
                frame = cv2.resize(
                    frames[int(frame_index)], (160, 120), interpolation=cv2.INTER_AREA
                )
                y = (strip_index // 3) * 120
                x = (strip_index % 3) * 160
                panel[y : y + 120, x : x + 160] = frame
            caption = (
                f"{row['event_id']}  f={row['anchor_frame']}  "
                f"t={int(row['anchor_frame']) / fps:.1f}s"
            )
            cv2.putText(
                panel,
                caption,
                (8, 268),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                "chronological left-to-right, then top-to-bottom",
                (8, 291),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
            y = (panel_index // 4) * 300
            x = (panel_index % 4) * 480
            canvas[y : y + 300, x : x + 480] = panel
        path = output_dir / f"sheet-{sheet_number + 1:02d}.jpg"
        if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise ValueError(f"could not write {path}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
