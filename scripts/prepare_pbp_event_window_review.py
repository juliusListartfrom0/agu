#!/usr/bin/env python3
"""Prepare label-hidden contact sheets from training-only PBP alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.cross_game_shot_windows import seal_shot_window_spec  # noqa: E402
from app.analysis.pbp_clock_event_alignment import (  # noqa: E402
    verify_pbp_clock_event_alignment_artifact,
)
from app.analysis.pbp_event_alignment import verify_pbp_event_alignment_artifact  # noqa: E402
from app.analysis.pbp_event_windows import select_pbp_event_windows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--window-frames", type=int, default=120)
    parser.add_argument("--minimum-center-spacing-frames", type=int, default=120)
    parser.add_argument("--panels-per-sheet", type=int, default=12)
    return parser.parse_args()


def prepare_review(
    *,
    video_path: Path,
    alignment_path: Path,
    output_dir: Path,
    seed: int,
    count: int,
    window_frames: int,
    minimum_center_spacing_frames: int,
    panels_per_sheet: int = 12,
) -> dict[str, Any]:
    if panels_per_sheet != 12:
        raise ValueError("the contact-sheet renderer currently requires 12 panels")
    if not video_path.is_file() or not alignment_path.is_file():
        raise FileNotFoundError("video and alignment files are required")
    source_sha = _file_sha256(video_path)
    raw_alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    if raw_alignment.get("schema_version") == "agu.pbp-clock-event-alignment.v1":
        alignment = verify_pbp_clock_event_alignment_artifact(raw_alignment)
    else:
        alignment = verify_pbp_event_alignment_artifact(raw_alignment)
    if source_sha != str(alignment["raw_video_sha256"]):
        raise ValueError("video does not match the alignment raw_video_sha256")
    capture = cv2.VideoCapture(str(video_path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("could not read video geometry")
    candidates = select_pbp_event_windows(
        alignment=alignment,
        frame_count=frame_count,
        fps=fps,
        count=count,
        window_frames=window_frames,
        minimum_center_spacing_frames=minimum_center_spacing_frames,
        seed=seed,
    )
    spec = seal_shot_window_spec(
        {
            "schema_version": "agu.cross-game-shot-window-spec.v1",
            "purpose": "offline_training_annotation_only",
            "runtime_consumable": False,
            "source_video_sha256": source_sha,
            "source_video_filename": video_path.name,
            "video_frame_count": frame_count,
            "video_fps": fps,
            "video_width": width,
            "video_height": height,
            "selection_seed": seed,
            "selection_protocol": "official_pbp_event_aligned_balanced_training_only_v1",
            "window_frames": window_frames,
            "minimum_center_spacing_frames": minimum_center_spacing_frames,
            "source_alignment_sha256": alignment["artifact_sha256"],
            "official_pbp_used_for_event_selection": True,
            "label_selection_not_used": False,
            "independent_evaluation_eligible": False,
            "candidates": candidates,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "candidate_spec_v1.json", spec)
    _render_sheets(video_path, candidates, output_dir / "raw_sheets", fps=fps)
    labels = {
        "schema_version": "agu.cross-game-shot-window-labels.v1",
        "purpose": "offline_training_annotation_only",
        "runtime_consumable": False,
        "source_video_sha256": source_sha,
        "candidate_spec_sha256": spec["artifact_sha256"],
        "review_protocol": "raw-frame-contact-sheet-and-strip; official PBP withheld during visual review",
        "reviewer": "codex",
        "examples": [
            {
                "event_id": row["event_id"],
                "event_present": None,
                "release_frame": None,
                "review_note": "pending",
            }
            for row in candidates
        ],
    }
    _write_json(output_dir / "labels_template_v1.json", labels)
    return {
        "source_video_sha256": source_sha,
        "alignment_artifact_sha256": alignment["artifact_sha256"],
        "candidate_spec_sha256": spec["artifact_sha256"],
        "candidates": len(candidates),
        "sheets": (len(candidates) + 11) // 12,
        "selection_protocol": spec["selection_protocol"],
    }


def _render_sheets(
    video_path: Path,
    candidates: list[dict[str, Any]],
    output_dir: Path,
    *,
    fps: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    try:
        frames: dict[int, np.ndarray] = {}
        for row in candidates:
            for frame_index in np.linspace(
                int(row["start_frame"]), int(row["end_frame"]), 6
            ).astype(int):
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
                frame = cv2.resize(frames[int(frame_index)], (160, 120), interpolation=cv2.INTER_AREA)
                panel[(strip_index // 3) * 120 : (strip_index // 3 + 1) * 120,
                      (strip_index % 3) * 160 : (strip_index % 3 + 1) * 160] = frame
            caption = f"{row['event_id']}  f={row['anchor_frame']}  t={int(row['anchor_frame']) / fps:.1f}s"
            cv2.putText(panel, caption, (8, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(panel, "chronological left-to-right, then top-to-bottom", (8, 291), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 180, 180), 1, cv2.LINE_AA)
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            prepare_review(
                video_path=args.video,
                alignment_path=args.alignment,
                output_dir=args.output_dir,
                seed=args.seed,
                count=args.count,
                window_frames=args.window_frames,
                minimum_center_spacing_frames=args.minimum_center_spacing_frames,
                panels_per_sheet=args.panels_per_sheet,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
