#!/usr/bin/env python3
"""Render raw and selected-ball-path sheets for offline ball-release review."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.ball_release_annotation import verify_ball_release_plan  # noqa: E402
from app.analysis.perception.ball_tracking import GlobalBallPathSelector  # noqa: E402
from app.analysis.schemas import PerceptionDetectionResponse  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--detections", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--nearest-path-frame", type=int, default=4)
    return parser.parse_args()


def render_review_sheets(
    *,
    plan_path: Path,
    video_paths: list[Path],
    detection_paths: list[Path],
    output_dir: Path,
    panel_width: int = 640,
    columns: int = 3,
    nearest_path_frame: int = 4,
) -> dict[str, object]:
    if panel_width < 160 or columns < 1 or nearest_path_frame < 0:
        raise ValueError("invalid review-sheet rendering settings")
    plan = verify_ball_release_plan(_read_json(plan_path))
    videos = {_file_sha256(path): path for path in video_paths}
    if len(videos) != len(video_paths):
        raise ValueError("review videos must have unique content hashes")
    planned_sources = {
        str(row["source_video_sha256"]) for row in plan["examples"]
    }
    if set(videos) != planned_sources:
        raise ValueError("review videos must exactly match planned source hashes")

    detections_by_source = _load_detections(detection_paths)
    unknown_detection_sources = set(detections_by_source) - planned_sources
    if unknown_detection_sources:
        raise ValueError("detection input contains an unplanned source video")

    raw_dir = output_dir / "raw"
    overlay_dir = output_dir / "selected_path"
    raw_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    captures: dict[str, cv2.VideoCapture] = {}
    records = []
    try:
        for example in plan["examples"]:
            source_sha = str(example["source_video_sha256"])
            capture = captures.get(source_sha)
            if capture is None:
                capture = cv2.VideoCapture(str(videos[source_sha]))
                if not capture.isOpened():
                    raise ValueError(
                        f"cannot open planned review video: {videos[source_sha].name}"
                    )
                captures[source_sha] = capture
            source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if source_width <= 0 or source_height <= 0:
                raise ValueError("review video has invalid frame dimensions")
            panel_height = round(panel_width * source_height / source_width)
            frames = _read_frames(capture, list(example["frame_numbers"]))
            event_detections = [
                item
                for item in detections_by_source.get(source_sha, [])
                if int(example["start_frame"])
                <= item.frame
                <= int(example["end_frame"])
            ]
            selected = (
                GlobalBallPathSelector().select_detection_paths(event_detections)
                if event_detections
                else []
            )
            selected_path = selected[0] if selected else []
            selected_by_target = _nearest_path_detections(
                list(example["frame_numbers"]),
                selected_path,
                maximum_distance=nearest_path_frame,
            )
            raw_panels = []
            overlay_panels = []
            path_rows = []
            for frame_number, frame in zip(
                example["frame_numbers"], frames, strict=True
            ):
                raw_panel = cv2.resize(
                    frame, (panel_width, panel_height), interpolation=cv2.INTER_AREA
                )
                _draw_frame_label(raw_panel, int(frame_number))
                overlay = raw_panel.copy()
                selected_detection = selected_by_target.get(int(frame_number))
                bbox = None
                if selected_detection is not None:
                    scale_x = panel_width / source_width
                    scale_y = panel_height / source_height
                    bbox = selected_detection.bbox.model_dump(mode="json")
                    x1 = round(selected_detection.bbox.x1 * scale_x)
                    y1 = round(selected_detection.bbox.y1 * scale_y)
                    x2 = round(selected_detection.bbox.x2 * scale_x)
                    y2 = round(selected_detection.bbox.y2 * scale_y)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(
                        overlay,
                        f"PATH f={selected_detection.frame}",
                        (max(2, x1), max(28, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                raw_panels.append(raw_panel)
                overlay_panels.append(overlay)
                path_rows.append(
                    {
                        "planned_frame": int(frame_number),
                        "detection_frame": (
                            selected_detection.frame
                            if selected_detection is not None
                            else None
                        ),
                        "bbox": bbox,
                    }
                )
            review_id = str(example["review_id"])
            raw_path = raw_dir / f"{review_id}.jpg"
            overlay_path = overlay_dir / f"{review_id}.jpg"
            _write_sheet(raw_panels, raw_path, columns)
            _write_sheet(overlay_panels, overlay_path, columns)
            records.append(
                {
                    "review_id": review_id,
                    "source_video_sha256": source_sha,
                    "candidate_bundle_sha256": example[
                        "candidate_bundle_sha256"
                    ],
                    "event_id": example["event_id"],
                    "raw_sheet": str(raw_path.relative_to(output_dir)),
                    "selected_path_sheet": str(
                        overlay_path.relative_to(output_dir)
                    ),
                    "selected_path_observations": path_rows,
                }
            )
    finally:
        for capture in captures.values():
            capture.release()

    manifest: dict[str, object] = {
        "schema_version": "agu.ball-release-review-sheets.v1",
        "purpose": "codex_offline_ball_release_training_annotation",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "plan_sha256": plan["artifact_sha256"],
        "rendering": {
            "panel_width": panel_width,
            "columns": columns,
            "nearest_path_frame": nearest_path_frame,
            "raw_frames_preserved_without_detection_overlay": True,
        },
        "detection_files": [
            {"filename": path.name, "sha256": _file_sha256(path)}
            for path in detection_paths
        ],
        "records": records,
    }
    manifest["artifact_sha256"] = _json_sha256(manifest)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_detections(
    paths: list[Path],
) -> dict[str, list[PerceptionDetectionResponse]]:
    by_source: dict[str, list[PerceptionDetectionResponse]] = defaultdict(list)
    for path in paths:
        payload = _read_json(path)
        raw_video = payload.get("raw_video")
        if not isinstance(raw_video, dict) or not raw_video.get("sha256"):
            raise ValueError("detection artifact lacks raw-video provenance")
        source_sha = str(raw_video["sha256"])
        rows = payload.get("detections")
        if not isinstance(rows, list):
            raise ValueError("detection artifact requires a detections list")
        by_source[source_sha].extend(
            PerceptionDetectionResponse.model_validate(row)
            for row in rows
            if isinstance(row, dict) and row.get("object_type") == "basketball"
        )
    return dict(by_source)


def _nearest_path_detections(
    frame_numbers: list[int],
    path: list[PerceptionDetectionResponse],
    *,
    maximum_distance: int,
) -> dict[int, PerceptionDetectionResponse]:
    output = {}
    for target in frame_numbers:
        if not path:
            continue
        nearest = min(
            path,
            key=lambda row: (abs(row.frame - target), -row.confidence),
        )
        if abs(nearest.frame - target) <= maximum_distance:
            output[target] = nearest
    return output


def _read_frames(
    capture: cv2.VideoCapture, frame_numbers: list[int]
) -> list[Any]:
    frames = []
    for frame_number in frame_numbers:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"cannot decode planned review frame {frame_number}")
        frames.append(frame)
    return frames


def _draw_frame_label(panel: Any, frame_number: int) -> None:
    cv2.rectangle(panel, (0, 0), (175, 34), (0, 0, 0), -1)
    cv2.putText(
        panel,
        f"FRAME {frame_number}",
        (7, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _write_sheet(panels: list[Any], path: Path, columns: int) -> None:
    rows = math.ceil(len(panels) / columns)
    blank = panels[0] * 0
    padded = panels + [blank] * (rows * columns - len(panels))
    sheet_rows = [
        cv2.hconcat(padded[index : index + columns])
        for index in range(0, len(padded), columns)
    ]
    sheet = cv2.vconcat(sheet_rows)
    if not cv2.imwrite(str(path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
        raise RuntimeError(f"failed to write review sheet: {path.name}")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def main() -> int:
    args = parse_args()
    manifest = render_review_sheets(
        plan_path=args.plan,
        video_paths=args.video,
        detection_paths=args.detections,
        output_dir=args.output_dir,
        panel_width=args.panel_width,
        columns=args.columns,
        nearest_path_frame=args.nearest_path_frame,
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "records": len(manifest["records"]),
                "artifact_sha256": manifest["artifact_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
