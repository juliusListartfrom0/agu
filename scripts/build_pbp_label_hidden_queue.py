#!/usr/bin/env python3
"""Build an offline label-hidden PBP review queue and visual-review strips."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.pbp_clock_event_alignment import (  # noqa: E402
    verify_pbp_clock_event_alignment_artifact,
)
from app.analysis.pbp_label_hidden_review import (  # noqa: E402
    build_label_hidden_queue,
    build_post_freeze_truth,
    select_review_batch,
    seal_queue_artifact,
    verify_queue_artifact,
    verify_review_batch_artifact,
    verify_truth_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--game-slug", required=True)
    parser.add_argument("--review-count", type=int, default=24)
    parser.add_argument("--selection-seed", type=int, default=20260810)
    parser.add_argument("--window-before-seconds", type=float, default=3.0)
    parser.add_argument("--window-after-seconds", type=float, default=4.0)
    return parser.parse_args()


def build_artifacts(
    *,
    video_path: Path,
    alignment_path: Path,
    output_dir: Path,
    game_slug: str,
    review_count: int = 24,
    selection_seed: int = 20260810,
    window_before_seconds: float = 3.0,
    window_after_seconds: float = 4.0,
) -> dict[str, Any]:
    if not video_path.is_file() or not alignment_path.is_file():
        raise FileNotFoundError("video and alignment files are required")
    source_alignment = verify_pbp_clock_event_alignment_artifact(
        json.loads(alignment_path.read_text(encoding="utf-8"))
    )
    video_sha256 = _file_sha256(video_path)
    if video_sha256 != str(source_alignment["raw_video_sha256"]):
        raise ValueError("video does not match alignment raw_video_sha256")
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    queue = build_label_hidden_queue(
        alignment=source_alignment,
        game_slug=game_slug,
        video_filename=video_path.name,
        video_sha256=video_sha256,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        window_before_seconds=window_before_seconds,
        window_after_seconds=window_after_seconds,
    )
    queue["source_alignment"]["path"] = str(alignment_path)
    queue = seal_queue_artifact(queue)
    batch = select_review_batch(
        queue,
        count=min(review_count, int(queue["queue_count"])),
        seed=selection_seed,
    )
    truth = build_post_freeze_truth(alignment=source_alignment, queue=queue)
    output_dir.mkdir(parents=True, exist_ok=True)
    strips_dir = output_dir / "raw_strips"
    sheets_dir = output_dir / "sheets"
    _write_json(output_dir / "label_hidden_review_queue.json", queue)
    _write_json(output_dir / "review_batch_manifest.json", batch)
    _write_json(output_dir / "source_truth_post_freeze.json", truth)
    _render_strips_and_sheets(video_path, batch["batch"], strips_dir, sheets_dir, fps=fps)
    coverage = {
        "schema_version": "agu.nba-games-pbp-label-hidden-review-coverage.v1",
        "generated_on": "2026-08-10",
        "purpose": "offline_audit_of_label_hidden_pbp_visual_review_assets",
        "runtime_consumable": False,
        "training_consumable": False,
        "independent_evaluation_eligible": False,
        "codex_runtime_answer_used": False,
        "video_sha256": video_sha256,
        "alignment_sha256": source_alignment["artifact_sha256"],
        "queue_sha256": queue["artifact_sha256"],
        "batch_sha256": batch["artifact_sha256"],
        "truth_sha256": truth["artifact_sha256"],
        "queue_count": queue["queue_count"],
        "review_batch_count": batch["batch_count"],
        "raw_strip_count": len(batch["batch"]),
        "raw_strip_dir": str(strips_dir),
        "temporary_sheet_dir": str(sheets_dir),
        "manual_review_complete": False,
    }
    _write_json(output_dir / "coverage_summary.json", coverage)
    verify_queue_artifact(queue)
    verify_review_batch_artifact(batch)
    verify_truth_artifact(truth)
    return {
        "queue_sha256": queue["artifact_sha256"],
        "batch_sha256": batch["artifact_sha256"],
        "truth_sha256": truth["artifact_sha256"],
        "queue_count": queue["queue_count"],
        "review_batch_count": batch["batch_count"],
        "raw_strip_count": len(batch["batch"]),
        "video_sha256": video_sha256,
    }


def _render_strips_and_sheets(
    video_path: Path,
    rows: list[dict[str, Any]],
    strips_dir: Path,
    sheets_dir: Path,
    *,
    fps: float,
) -> None:
    strips_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    strips: list[tuple[str, np.ndarray]] = []
    try:
        for row in rows:
            frame_indexes = np.linspace(
                int(row["start_frame"]), int(row["end_frame"]), 6
            ).astype(int)
            frames: list[np.ndarray] = []
            for frame_index in frame_indexes:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ValueError(f"could not decode frame {frame_index}")
                frames.append(frame)
            strip = np.full((360, 960, 3), 18, dtype=np.uint8)
            for index, frame in enumerate(frames):
                resized = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
                y = (index // 3) * 180
                x = (index % 3) * 320
                strip[y : y + 180, x : x + 320] = resized
            cv2.putText(
                strip,
                str(row["review_id"]),
                (8, 352),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            filename = f"{row['review_id']}.jpg"
            if not cv2.imwrite(str(strips_dir / filename), strip, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise ValueError(f"could not write {strips_dir / filename}")
            strips.append((str(row["review_id"]), strip))
    finally:
        capture.release()
    for sheet_index in range((len(strips) + 11) // 12):
        canvas = np.full((900, 1920, 3), 18, dtype=np.uint8)
        for panel_index, (review_id, strip) in enumerate(strips[sheet_index * 12 : (sheet_index + 1) * 12]):
            panel = cv2.resize(strip, (480, 180), interpolation=cv2.INTER_AREA)
            panel_canvas = np.full((300, 480, 3), 18, dtype=np.uint8)
            panel_canvas[:180] = panel
            cv2.putText(
                panel_canvas,
                review_id,
                (8, 205),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel_canvas,
                "label hidden; inspect temporal evidence",
                (8, 232),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
            y = (panel_index // 4) * 300
            x = (panel_index % 4) * 480
            canvas[y : y + 300, x : x + 480] = panel_canvas
        path = sheets_dir / f"sheet-{sheet_index + 1:02d}.jpg"
        if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise ValueError(f"could not write {path}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            build_artifacts(
                video_path=args.video,
                alignment_path=args.alignment,
                output_dir=args.output_dir,
                game_slug=args.game_slug,
                review_count=args.review_count,
                selection_seed=args.selection_seed,
                window_before_seconds=args.window_before_seconds,
                window_after_seconds=args.window_after_seconds,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
