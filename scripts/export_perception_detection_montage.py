#!/usr/bin/env python3
"""Render an offline review montage for raw-bound perception detections."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.schemas import PerceptionDetectionResponse  # noqa: E402
from scripts.run_official_perception import _file_sha256  # noqa: E402


def export_detection_montage(
    *,
    perception_path: Path,
    video_path: Path,
    output_path: Path,
    object_types: set[str] | None,
    maximum_frames: int,
    columns: int,
    tile_width: int,
) -> dict[str, object]:
    if min(maximum_frames, columns, tile_width) <= 0:
        raise ValueError("montage dimensions must be positive")
    payload = json.loads(perception_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "agu.official-perception.v1":
        raise ValueError("unsupported perception schema")
    raw = payload.get("raw_video") or {}
    if (
        raw.get("filename") != video_path.name
        or raw.get("sha256") != _file_sha256(video_path)
    ):
        raise ValueError("perception artifact does not match raw video")
    detections = [
        PerceptionDetectionResponse.model_validate(item)
        for item in payload.get("detections") or []
        if not object_types or str(item.get("object_type") or "") in object_types
    ]
    by_frame: dict[int, list[PerceptionDetectionResponse]] = {}
    for detection in detections:
        by_frame.setdefault(detection.frame, []).append(detection)
    selected_frames = _evenly_sample(sorted(by_frame), maximum=maximum_frames)
    if not selected_frames:
        raise ValueError("perception artifact contains no requested detections")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    tiles: list[np.ndarray] = []
    try:
        for frame_number in selected_frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            if not ok:
                continue
            for detection in by_frame[frame_number]:
                box = detection.bbox
                color = _color(detection.object_type)
                cv2.rectangle(
                    frame,
                    (int(round(box.x1)), int(round(box.y1))),
                    (int(round(box.x2)), int(round(box.y2))),
                    color,
                    3,
                )
                label = f"{detection.object_type} {detection.confidence:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (int(round(box.x1)), max(22, int(round(box.y1)) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            height, width = frame.shape[:2]
            tile_height = max(1, int(round(height * tile_width / width)))
            tile = cv2.resize(frame, (tile_width, tile_height))
            cv2.putText(
                tile,
                f"frame {frame_number}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            tiles.append(tile)
    finally:
        capture.release()
    if not tiles:
        raise RuntimeError("unable to decode selected detection frames")

    tile_height = max(tile.shape[0] for tile in tiles)
    rows = math.ceil(len(tiles) / columns)
    montage = np.zeros(
        (rows * tile_height, columns * tile_width, 3),
        dtype=np.uint8,
    )
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        montage[
            row * tile_height : row * tile_height + tile.shape[0],
            column * tile_width : (column + 1) * tile_width,
        ] = tile
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), montage):
        raise RuntimeError(f"unable to write detection montage: {output_path}")
    return {
        "output": str(output_path),
        "selected_frame_count": len(tiles),
        "detection_count": sum(len(by_frame[frame]) for frame in selected_frames),
        "object_types": sorted(object_types or []),
    }


def _evenly_sample(values: list[int], *, maximum: int) -> list[int]:
    if len(values) <= maximum:
        return values
    return [
        values[round(index * (len(values) - 1) / (maximum - 1))]
        for index in range(maximum)
    ]


def _color(object_type: str) -> tuple[int, int, int]:
    return {
        "basketball": (0, 140, 255),
        "rim": (0, 255, 255),
        "player": (80, 220, 80),
    }.get(object_type, (255, 180, 60))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perception", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object-type", action="append", default=[])
    parser.add_argument("--maximum-frames", type=int, default=24)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--tile-width", type=int, default=480)
    args = parser.parse_args()
    result = export_detection_montage(
        perception_path=args.perception,
        video_path=args.video,
        output_path=args.output,
        object_types=set(args.object_type) or None,
        maximum_frames=args.maximum_frames,
        columns=args.columns,
        tile_width=args.tile_width,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
