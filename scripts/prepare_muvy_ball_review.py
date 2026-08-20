#!/usr/bin/env python3
"""Build hash-bound contact sheets for Codex review of MUVY ball boxes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.muvy_ball_review import (  # noqa: E402
    is_small_ball_candidate,
    parse_muvy_detection_line,
    seal_muvy_ball_review_plan,
)

SHEET_COLUMNS = 6
SHEET_ROWS = 6
PANEL_WIDTH = 320
PANEL_HEIGHT = 180


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def prepare_review(*, source: Path, output: Path) -> dict[str, Any]:
    source_manifest = _verify_source_manifest(source / "manifest.json")
    source_files = {
        str(row["path"]): row for row in source_manifest["files"]
    }
    candidates = []
    for annotation_path in sorted(source.rglob("detections_info.txt")):
        video_path = next(annotation_path.parent.glob("*.mp4"), None)
        info_path = annotation_path.parent / "video_info.txt"
        if video_path is None or not info_path.exists():
            raise ValueError("MUVY camera folder is incomplete")
        video_relative = video_path.relative_to(source).as_posix()
        annotation_relative = annotation_path.relative_to(source).as_posix()
        _verify_manifest_file(
            video_path,
            source_files.get(video_relative),
        )
        _verify_manifest_file(
            annotation_path,
            source_files.get(annotation_relative),
        )
        info = _parse_video_info(info_path)
        width = int(info["imWidth"])
        height = int(info["imHeight"])
        rows = []
        for line in annotation_path.read_text(encoding="utf-8").splitlines()[1:]:
            parsed = parse_muvy_detection_line(line)
            if parsed["object_class"] != "sports ball":
                continue
            if is_small_ball_candidate(
                parsed["bbox"],
                frame_width=width,
                frame_height=height,
            ):
                rows.append(parsed)
        for row in rows:
            candidates.append(
                {
                    "video_path": video_relative,
                    "video_sha256": source_files[video_relative]["sha256"],
                    "annotation_sha256": source_files[
                        annotation_relative
                    ]["sha256"],
                    "frame_id": row["frame_id"],
                    "frame_index": row["frame_id"] - 1,
                    "frame_width": width,
                    "frame_height": height,
                    "confidence": row["confidence"],
                    "bbox": row["bbox"],
                }
            )
    candidates.sort(
        key=lambda row: (
            row["video_path"],
            row["frame_id"],
            row["bbox"],
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    sheet_dir = output / "sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    panels_per_sheet = SHEET_COLUMNS * SHEET_ROWS
    for index, row in enumerate(candidates, start=1):
        row["candidate_id"] = f"muvy-ball-{index:04d}"
        row["sheet_index"] = (index - 1) // panels_per_sheet + 1
        row["panel_index"] = (index - 1) % panels_per_sheet + 1
    sheet_count = (len(candidates) + panels_per_sheet - 1) // panels_per_sheet
    canvases = [
        np.zeros(
            (SHEET_ROWS * PANEL_HEIGHT, SHEET_COLUMNS * PANEL_WIDTH, 3),
            dtype=np.uint8,
        )
        for _ in range(sheet_count)
    ]
    rows_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        rows_by_video[row["video_path"]].append(row)
    for video_relative, video_rows in rows_by_video.items():
        capture = cv2.VideoCapture(str(source / video_relative))
        if not capture.isOpened():
            raise ValueError(f"cannot open MUVY video: {video_relative}")
        try:
            for row in video_rows:
                capture.set(cv2.CAP_PROP_POS_FRAMES, row["frame_index"])
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(
                        f"cannot decode MUVY frame: "
                        f"{video_relative}#{row['frame_id']}"
                    )
                if (
                    frame.shape[1] != row["frame_width"]
                    or frame.shape[0] != row["frame_height"]
                ):
                    raise ValueError(
                        "MUVY video metadata dimensions mismatch"
                    )
                row["frame_sha256"] = _frame_sha256(frame)
                panel = _render_panel(row, frame=frame)
                panel_index = row["panel_index"] - 1
                top = panel_index // SHEET_COLUMNS * PANEL_HEIGHT
                left = panel_index % SHEET_COLUMNS * PANEL_WIDTH
                canvases[row["sheet_index"] - 1][
                    top : top + PANEL_HEIGHT,
                    left : left + PANEL_WIDTH,
                ] = panel
        finally:
            capture.release()
    sheet_files = []
    for sheet_index, canvas in enumerate(canvases, start=1):
        start = (sheet_index - 1) * panels_per_sheet
        sheet_rows = candidates[start : start + panels_per_sheet]
        sheet_path = sheet_dir / f"sheet-{sheet_index:03d}.jpg"
        if not cv2.imwrite(
            str(sheet_path),
            canvas,
            [cv2.IMWRITE_JPEG_QUALITY, 94],
        ):
            raise ValueError("cannot write MUVY review sheet")
        sheet_files.append(
            {
                "path": sheet_path.relative_to(output).as_posix(),
                "sha256": _file_sha256(sheet_path),
                "candidate_ids": [
                    row["candidate_id"] for row in sheet_rows
                ],
            }
        )
    plan_candidates = list(candidates)
    plan = seal_muvy_ball_review_plan(
        {
            "source_manifest_sha256": source_manifest["artifact_sha256"],
            "source_record_url": source_manifest["source"]["record_url"],
            "source_license": source_manifest["source"]["license"],
            "candidates": plan_candidates,
        }
    )
    _write_json(output / "plan.json", plan)
    _write_json(
        output / "decisions.template.json",
        {
            "plan_sha256": plan["artifact_sha256"],
            "reviewer": "codex_offline_source_annotation",
            "decisions": [
                {
                    "candidate_id": row["candidate_id"],
                    "decision": "uncertain",
                    "notes": "",
                }
                for row in plan_candidates
            ],
        },
    )
    package: dict[str, Any] = {
        "schema_version": "agu.muvy-ball-review-package.v1",
        "purpose": "codex_offline_muvy_ball_annotation_audit",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "source_manifest_sha256": source_manifest["artifact_sha256"],
        "plan_sha256": plan["artifact_sha256"],
        "candidate_count": len(plan_candidates),
        "sheet_count": len(sheet_files),
        "sheet_files": sheet_files,
    }
    package["artifact_sha256"] = _canonical_sha256(package)
    _write_json(output / "manifest.json", package)
    return package


def _render_panel(
    row: dict[str, Any],
    *,
    frame: np.ndarray,
) -> np.ndarray:
    source_height, source_width = frame.shape[:2]
    scale = min(PANEL_WIDTH / source_width, PANEL_HEIGHT / source_height)
    width = round(source_width * scale)
    height = round(source_height * scale)
    left = (PANEL_WIDTH - width) // 2
    top = (PANEL_HEIGHT - height) // 2
    panel = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    panel[top : top + height, left : left + width] = cv2.resize(
        frame,
        (width, height),
    )
    x1, y1, x2, y2 = row["bbox"]
    first = (round(left + x1 * scale), round(top + y1 * scale))
    second = (round(left + x2 * scale), round(top + y2 * scale))
    cv2.rectangle(panel, first, second, (0, 0, 255), 2)
    cv2.putText(
        panel,
        f"{row['candidate_id']} c={row['confidence']:.3f}",
        (4, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def _parse_video_info(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    required = {"FPS", "numFrames", "imWidth", "imHeight"}
    if not required.issubset(values):
        raise ValueError("MUVY video metadata is incomplete")
    return values


def _verify_source_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = str(payload.pop("artifact_sha256", ""))
    if (
        payload.get("schema_version") != "agu.muvy-basketball-subset.v1"
        or payload.get("runtime_consumable") is not False
        or payload.get("codex_runtime_answer_used") is not False
        or claimed != _canonical_sha256(payload)
    ):
        raise ValueError("MUVY source manifest is invalid")
    payload["artifact_sha256"] = claimed
    return payload


def _verify_manifest_file(path: Path, row: object) -> None:
    if (
        not isinstance(row, dict)
        or int(row.get("size_bytes", -1)) != path.stat().st_size
        or str(row.get("sha256") or "") != _file_sha256(path)
    ):
        raise ValueError(f"MUVY source file hash mismatch: {path.name}")


def _frame_sha256(frame: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(frame.shape)).encode())
    digest.update(frame.tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(f"{path.suffix}.part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    package = prepare_review(source=args.source, output=args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "candidate_count": package["candidate_count"],
                "sheet_count": package["sheet_count"],
                "artifact_sha256": package["artifact_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
