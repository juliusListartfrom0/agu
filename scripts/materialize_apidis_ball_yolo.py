#!/usr/bin/env python3
"""Materialize a camera-disjoint APIDIS ball detector dataset.

The output is an offline research artifact.  It keeps all sampled frames,
including empty-label hard negatives, and splits by camera to avoid leakage
from identical views.  It never changes AGU runtime defaults.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def _verify_alignment(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("APIDIS alignment must be a JSON object")
    claimed = str(payload.pop("artifact_sha256", ""))
    if payload.get("schema_version") != "agu.apidis-alignment.v1" or payload.get("runtime_consumable") is not False:
        raise ValueError("unexpected APIDIS alignment artifact")
    if not claimed or claimed != _canonical_sha256(payload):
        raise ValueError("APIDIS alignment artifact hash mismatch")
    payload["artifact_sha256"] = claimed
    return payload


def _label_rows_by_frame(rows: Iterable[dict[str, Any]]) -> dict[int, tuple[float, float]]:
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["frame"])].append((float(row["x"]), float(row["y"])))
    return {
        frame: (float(median(values[0] for values in points)), float(median(values[1] for values in points)))
        for frame, points in grouped.items()
    }


def _yolo_line(
    center: tuple[float, float],
    *,
    width: int,
    height: int,
    box_width_px: float,
    box_height_px: float,
) -> str:
    x, y = center
    if not 0.0 <= x <= width or not 0.0 <= y <= height:
        raise ValueError("APIDIS ball centre is outside the decoded frame")
    if box_width_px <= 0.0 or box_height_px <= 0.0:
        raise ValueError("ball box dimensions must be positive")
    return (
        f"0 {x / width:.10f} {y / height:.10f} "
        f"{min(1.0, box_width_px / width):.10f} {min(1.0, box_height_px / height):.10f}\n"
    )


def materialize_apidis_dataset(
    *,
    source_root: Path,
    alignment_path: Path,
    output_root: Path,
    frame_stride: int = 5,
    jpeg_quality: int = 90,
    train_cameras: tuple[int, ...] = (1, 2, 3, 4, 5),
    val_cameras: tuple[int, ...] = (6,),
    test_cameras: tuple[int, ...] = (7,),
    box_width_px: float = 13.5,
    box_height_px: float = 15.5,
) -> dict[str, Any]:
    if frame_stride <= 0 or not 1 <= jpeg_quality <= 100:
        raise ValueError("frame_stride and jpeg_quality are invalid")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output must be new or empty: {output_root}")
    source_root = source_root.resolve()
    alignment_path = alignment_path.resolve()
    output_root = output_root.resolve()
    alignment = _verify_alignment(alignment_path)
    camera_to_split = {
        **{camera: "train" for camera in train_cameras},
        **{camera: "val" for camera in val_cameras},
        **{camera: "test" for camera in test_cameras},
    }
    if set(camera_to_split) != set(range(1, 8)) or len(camera_to_split) != 7:
        raise ValueError("train/val/test cameras must partition cameras 1..7")
    by_camera = {int(row["camera"]): row for row in alignment["ball_labels"]}
    videos = {int(row["camera"]): source_root / str(row["video"]) for row in alignment["cameras"]}
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for camera in range(1, 8):
        split = camera_to_split[camera]
        video_path = videos[camera]
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"could not open APIDIS video: {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if frame_count <= 0 or fps <= 0.0 or width <= 0 or height <= 0:
            capture.release()
            raise ValueError(f"invalid APIDIS video metadata: {video_path}")
        labels = _label_rows_by_frame(by_camera[camera]["rows"])
        source_video_sha256 = _sha256(video_path)
        try:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % frame_stride == 0:
                    stem = f"camera{camera}_frame{frame_index:04d}"
                    image_path = output_root / "images" / split / f"{stem}.jpg"
                    label_path = output_root / "labels" / split / f"{stem}.txt"
                    encoded_ok, encoded = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                    )
                    if not encoded_ok:
                        raise ValueError(f"could not encode APIDIS frame: {video_path}:{frame_index}")
                    image_path.write_bytes(encoded.tobytes())
                    label_text = (
                        _yolo_line(
                            labels[frame_index],
                            width=width,
                            height=height,
                            box_width_px=box_width_px,
                            box_height_px=box_height_px,
                        )
                        if frame_index in labels
                        else ""
                    )
                    label_path.write_text(label_text, encoding="utf-8")
                    records.append(
                        {
                            "camera": camera,
                            "split": split,
                            "frame": frame_index,
                            "source_video": str(video_path.relative_to(source_root)),
                            "source_video_sha256": source_video_sha256,
                            "image": str(image_path.relative_to(output_root)),
                            "image_sha256": _sha256(image_path),
                            "label": str(label_path.relative_to(output_root)),
                            "label_sha256": _sha256(label_path),
                            "has_ball": frame_index in labels,
                            "width": width,
                            "height": height,
                            "fps": round(fps, 6),
                        }
                    )
                frame_index += 1
        finally:
            capture.release()
    data_yaml = (
        f"path: {output_root}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: basketball\n"
    )
    (output_root / "data.yaml").write_text(data_yaml, encoding="utf-8")
    manifest: dict[str, Any] = {
        "schema_version": "agu.apidis-ball-yolo.v1",
        "purpose": "offline_camera_disjoint_apidis_ball_detector_training_screen",
        "runtime_consumable": False,
        "training_eligible": True,
        "codex_runtime_answer_used": False,
        "source_alignment": str(alignment_path),
        "source_alignment_sha256": _sha256(alignment_path),
        "source_alignment_artifact_sha256": alignment["artifact_sha256"],
        "split_policy": {"train_cameras": list(train_cameras), "val_cameras": list(val_cameras), "test_cameras": list(test_cameras)},
        "sampling": {"frame_stride": frame_stride, "jpeg_quality": jpeg_quality},
        "label_policy": {
            "center_source": "APIDIS manual ball positions, median per nearest 25 FPS frame",
            "box_width_px": box_width_px,
            "box_height_px": box_height_px,
            "unlisted_sampled_frames": "empty label hard negatives",
        },
        "split_counts": {
            split: sum(row["split"] == split for row in records) for split in ("train", "val", "test")
        },
        "positive_counts": {
            split: sum(row["split"] == split and row["has_ball"] for row in records)
            for split in ("train", "val", "test")
        },
        "data_yaml_sha256": _sha256(output_root / "data.yaml"),
        "records": records,
    }
    manifest["artifact_sha256"] = _canonical_sha256(manifest)
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()
    manifest = materialize_apidis_dataset(
        source_root=args.source_root,
        alignment_path=args.alignment,
        output_root=args.output_root,
        frame_stride=args.frame_stride,
        jpeg_quality=args.jpeg_quality,
    )
    print(
        json.dumps(
            {
                "output": str(args.output_root),
                "artifact_sha256": manifest["artifact_sha256"],
                "split_counts": manifest["split_counts"],
                "positive_counts": manifest["positive_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
