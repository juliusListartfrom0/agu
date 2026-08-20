#!/usr/bin/env python3
"""Build a hash-bound APIDIS 16:47 UTC multi-view/event alignment artifact.

This command only reads the already-authorized APIDIS source slice.  It does
not add anything to AGU runtime configuration and does not retain decoded
frames; a later training materializer may consume this offline artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.apidis_alignment import (  # noqa: E402
    events_in_clip,
    frame_index_for_local_timestamp,
    parse_ball_position_rows,
    summarize_event_types,
)

_VIDEO_RE = re.compile(r"^camera(?P<camera>[1-7])-ps_(?P<stamp>\d{8}T\d{6})Z\.avi$")
_SCHEMA = "agu.apidis-alignment.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _video_specs(source_root: Path) -> tuple[list[dict[str, Any]], float, int, int, int, float]:
    video_paths = sorted((source_root / "raw" / "ps").glob("camera*-ps_*.avi"))
    if len(video_paths) != 7:
        raise ValueError(f"APIDIS alignment requires seven pseudo-sync videos, found {len(video_paths)}")
    specs: list[dict[str, Any]] = []
    common: tuple[float, int, int, int] | None = None
    clip_start_epoch: float | None = None
    for path in video_paths:
        match = _VIDEO_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected APIDIS video name: {path.name}")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"could not open APIDIS video: {path}")
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        finally:
            capture.release()
        if fps <= 0.0 or frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"invalid APIDIS video metadata: {path}")
        current = (round(fps, 6), frame_count, width, height)
        if common is None:
            common = current
        elif current != common:
            raise ValueError(f"APIDIS pseudo-sync video metadata mismatch: {path}")
        start = datetime.strptime(match.group("stamp"), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc).timestamp()
        if clip_start_epoch is None:
            clip_start_epoch = start
        elif start != clip_start_epoch:
            raise ValueError("APIDIS cameras do not share a pseudo-sync start time")
        specs.append(
            {
                "camera": int(match.group("camera")),
                "video": str(path.relative_to(source_root)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "fps": round(fps, 6),
                "frame_count": frame_count,
                "width": width,
                "height": height,
            }
        )
    if common is None or clip_start_epoch is None:
        raise ValueError("APIDIS videos are missing")
    return specs, clip_start_epoch, common[1], common[2], common[3], common[0]


def _ball_file(source_root: Path, camera: int) -> Path:
    candidates = sorted(
        (source_root / "raw" / "all" / "ball" / "GroundTruth_Ball_184700_3min").glob(
            f"camera{camera}_*.ballposition.txt"
        )
    )
    if len(candidates) != 1:
        raise ValueError(f"expected one APIDIS ball file for camera {camera}, found {len(candidates)}")
    return candidates[0]


def build_alignment_artifact(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    manifest_path = source_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing APIDIS manifest: {manifest_path}")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    specs, clip_start_epoch, frame_count, width, height, fps = _video_specs(source_root)
    local_clip_start = "184700"
    camera_rows: list[dict[str, Any]] = []
    for spec in specs:
        camera = int(spec["camera"])
        path = _ball_file(source_root, camera)
        rows = parse_ball_position_rows(path)
        mapped: list[dict[str, Any]] = []
        outside = 0
        for row in rows:
            frame = frame_index_for_local_timestamp(
                str(row["timestamp"]),
                local_clip_start,
                fps=fps,
                frame_count=frame_count,
            )
            if frame is None:
                outside += 1
                continue
            # Ball annotations are at the raw 1600x1200 camera resolution;
            # pseudo-sync AVI is 800x600, so preserve the exact linear scale.
            x = float(row["x"]) * width / 1600.0
            y = float(row["y"]) * height / 1200.0
            if not 0.0 <= x <= width or not 0.0 <= y <= height:
                raise ValueError(f"APIDIS ball centre is outside frame: camera {camera} frame {frame}")
            mapped.append(
                {
                    "timestamp": str(row["timestamp"]),
                    "frame": frame,
                    "x": round(x, 6),
                    "y": round(y, 6),
                }
            )
        by_frame: dict[int, list[dict[str, Any]]] = {}
        for row in mapped:
            by_frame.setdefault(int(row["frame"]), []).append(row)
        duplicate_frames = sum(max(0, len(values) - 1) for values in by_frame.values())
        camera_rows.append(
            {
                "camera": camera,
                "ball_annotation": str(path.relative_to(source_root)),
                "ball_annotation_sha256": _sha256(path),
                "source_row_count": len(rows),
                "mapped_row_count": len(mapped),
                "mapped_frame_count": len(by_frame),
                "duplicate_frame_rows": duplicate_frames,
                "outside_clip_row_count": outside,
                "rows": sorted(mapped, key=lambda row: (int(row["frame"]), str(row["timestamp"]))),
            }
        )
    event_paths = sorted((source_root / "raw" / "all" / "metadata").glob("*.events.xml"))
    if len(event_paths) != 4:
        raise ValueError(f"expected four APIDIS event XML files, found {len(event_paths)}")
    event_rows = events_in_clip(
        event_paths,
        clip_start_epoch=clip_start_epoch,
        clip_duration_seconds=frame_count / fps,
        fps=fps,
        frame_count=frame_count,
    )
    artifact: dict[str, Any] = {
        "schema_version": _SCHEMA,
        "purpose": "offline_multiview_ball_and_event_alignment_for_training_screen",
        "runtime_consumable": False,
        "training_eligible": True,
        "codex_runtime_answer_used": False,
        "source_page": "https://ispgroup.gitlab.io/code/apidis/",
        "source_download_page": "https://www.kaggle.com/datasets/gabrielvanzandycke/apidis-metadata",
        "license_terms": "non-commercial research in video signal processing only; mention APIDIS project",
        "source_manifest_sha256": _sha256(manifest_path),
        "source_archive_etag": source_manifest.get("source_archive", {}).get("archive_etag"),
        "clip": {
            "utc_start_epoch": clip_start_epoch,
            "utc_start_iso": datetime.fromtimestamp(clip_start_epoch, tz=timezone.utc).isoformat(),
            "local_start_timestamp": local_clip_start,
            "timezone_offset_hours": 2,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "annotation_source_width": 1600,
            "annotation_source_height": 1200,
            "coordinate_scale": [width / 1600.0, height / 1200.0],
        },
        "cameras": specs,
        "ball_labels": camera_rows,
        "event_labels": {
            "source_files": [str(path.relative_to(source_root)) for path in event_paths],
            "rows": event_rows,
            "event_count": len(event_rows),
            "action_type_counts": summarize_event_types(event_rows),
        },
        "quality_gate": {
            "camera_count": len(specs),
            "all_video_specs_match": True,
            "event_overlap": bool(event_rows),
            "ball_label_cameras_with_mapped_rows": sum(
                int(row["mapped_frame_count"]) > 0 for row in camera_rows
            ),
            "max_duplicate_frame_rows": max(int(row["duplicate_frame_rows"]) for row in camera_rows),
        },
        "rights_boundary": {
            "offline_training_and_evaluation_only": True,
            "runtime_answers_forbidden": True,
            "commercial_release_forbidden": True,
            "blind_game_truth_not_used": True,
        },
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = build_alignment_artifact(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "artifact_sha256": artifact["artifact_sha256"],
                "camera_count": artifact["quality_gate"]["camera_count"],
                "event_count": artifact["event_labels"]["event_count"],
                "event_types": artifact["event_labels"]["action_type_counts"],
                "mapped_ball_frames": sum(
                    int(row["mapped_frame_count"]) for row in artifact["ball_labels"]
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
