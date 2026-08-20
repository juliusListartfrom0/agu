#!/usr/bin/env python3
"""Build hash-bound offline review sheets for broadcast ball candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.analysis.ball_candidate_review import (
    seal_artifact,
    select_stratified_candidates,
    verify_artifact,
)

BANDS = ((0.1, 0.15), (0.15, 0.2), (0.2, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 1.01))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"could not decode frame {frame_index}")
    return frame


def _detail_crop(frame: np.ndarray, bbox: dict[str, float]) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(bbox[key]) for key in ("x1", "y1", "x2", "y2"))
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * 2.5
    left = max(0, int(center_x - side / 2))
    top = max(0, int(center_y - side / 2))
    right = min(width, max(left + 1, int(center_x + side / 2)))
    bottom = min(height, max(top + 1, int(center_y + side / 2)))
    return frame[top:bottom, left:right]


def _render_panel(frame: np.ndarray, row: dict[str, Any]) -> np.ndarray:
    panel = np.full((360, 480, 3), 24, dtype=np.uint8)
    view = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
    source_height, source_width = frame.shape[:2]
    bbox = row["bbox"]
    x1 = round(float(bbox["x1"]) * 480 / source_width)
    y1 = round(float(bbox["y1"]) * 270 / source_height)
    x2 = round(float(bbox["x2"]) * 480 / source_width)
    y2 = round(float(bbox["y2"]) * 270 / source_height)
    cv2.rectangle(view, (x1, y1), (x2, y2), (0, 0, 255), 3)
    detail = cv2.resize(
        _detail_crop(frame, bbox), (160, 120), interpolation=cv2.INTER_NEAREST
    )
    cv2.rectangle(detail, (0, 0), (159, 119), (0, 255, 255), 3)
    view[145:265, 315:475] = detail
    panel[:270] = view
    text = (
        f"{row['candidate_id']} f={row['frame']} "
        f"conf={float(row['confidence']):.3f}"
    )
    cv2.putText(
        panel,
        text,
        (8, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "judge only the red box; inset is enlarged context",
        (8, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perception", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-band", type=int, default=18)
    parser.add_argument("--panels-per-sheet", type=int, default=12)
    parser.add_argument("--candidate-prefix", default="atl-rfdetr-ball")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    perception = verify_artifact(json.loads(args.perception.read_text()))
    if _file_sha256(args.video) != perception["raw_video"]["sha256"]:
        raise ValueError("review video hash mismatch")
    if args.panels_per_sheet != 12:
        raise ValueError("current renderer requires 12 panels per sheet")
    selected = select_stratified_candidates(
        perception["detections"],
        bands=BANDS,
        samples_per_band=args.samples_per_band,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir = args.output_dir / "sheets-detail"
    sheets_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    frames: dict[int, np.ndarray] = {}
    candidates: list[dict[str, Any]] = []
    try:
        for index, source in enumerate(selected, 1):
            frame_index = int(source["frame"])
            if frame_index not in frames:
                frames[frame_index] = _read_frame(cap, frame_index)
            row = dict(source)
            row["candidate_id"] = f"{args.candidate_prefix}-{index:03d}"
            row["frame_pixels_sha256"] = hashlib.sha256(
                frames[frame_index].tobytes()
            ).hexdigest()
            row["sheet_index"] = (index - 1) // args.panels_per_sheet + 1
            row["panel_index"] = (index - 1) % args.panels_per_sheet + 1
            candidates.append(row)
    finally:
        cap.release()

    for sheet_index in range(1, (len(candidates) + 11) // 12 + 1):
        canvas = np.full((1080, 1920, 3), 16, dtype=np.uint8)
        rows = [
            row for row in candidates if int(row["sheet_index"]) == sheet_index
        ]
        for panel_index, row in enumerate(rows):
            frame = frames[int(row["frame"])]
            panel = _render_panel(frame, row)
            y = (panel_index // 4) * 360
            x = (panel_index % 4) * 480
            canvas[y : y + 360, x : x + 480] = panel
        path = sheets_dir / f"sheet-{sheet_index:02d}.jpg"
        if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise ValueError(f"could not write {path}")

    plan = seal_artifact(
        {
            "schema_version": "agu.broadcast-ball-offline-review-plan.v1",
            "purpose": "codex_offline_same_detector_hard_negative_annotation",
            "runtime_consumable": False,
            "codex_runtime_answer_used": False,
            "perception_artifact_sha256": perception["artifact_sha256"],
            "raw_video_sha256": perception["raw_video"]["sha256"],
            "sampling": {
                "strategy": "six_confidence_bands_even_frame_order",
                "bands": [list(band) for band in BANDS],
                "samples_per_band": args.samples_per_band,
            },
            "review_scope": {
                "allowed": ["visible_basketball_inside_red_box"],
                "forbidden": [
                    "shot_or_event_answer",
                    "made_or_missed",
                    "player_identity",
                    "technical_statistic",
                ],
            },
            "candidates": candidates,
        }
    )
    (args.output_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    template = {
        "schema_version": "agu.broadcast-ball-offline-review-decisions.v1",
        "plan_sha256": plan["artifact_sha256"],
        "allowed_decisions": ["valid_ball", "false_positive", "uncertain"],
        "decisions": [
            {"candidate_id": row["candidate_id"], "decision": "pending", "notes": ""}
            for row in candidates
        ],
    }
    (args.output_dir / "decisions.template.json").write_text(
        json.dumps(template, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "candidates": len(candidates),
                "sheets": (len(candidates) + 11) // 12,
                "artifact_sha256": plan["artifact_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
