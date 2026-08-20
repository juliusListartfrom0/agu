#!/usr/bin/env python3
"""Materialize a small cross-broadcast full-frame ball-detector screen set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2

from app.analysis.broadcast_ball_detector_dataset import (
    build_dataset_manifest,
    collect_review_records,
    file_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source",
        nargs=4,
        action="append",
        metavar=("SOURCE_ID", "PLAN", "REVIEW", "VIDEO"),
        required=True,
        help="source id, plan JSON, sealed review JSON and raw video",
    )
    parser.add_argument("--val-source-id", required=True)
    return parser.parse_args()


def _read_frame(cap: cv2.VideoCapture, frame_index: int) -> Any:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"could not decode frame {frame_index}")
    return frame


def _write_label(path: Path, record: dict[str, Any]) -> None:
    if record["decision"] == "false_positive":
        path.write_text("", encoding="utf-8")
        return
    x1, y1, x2, y2 = record["bbox_xyxy_normalized"]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    width, height = x2 - x1, y2 - y1
    path.write_text(f"0 {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not any(source[0] == args.val_source_id for source in args.source):
        raise ValueError("--val-source-id must identify one --source")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    source_specs: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    video_cache: dict[str, cv2.VideoCapture] = {}
    try:
        for source_id, plan_path, review_path, video_arg in args.source:
            plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
            review = json.loads(Path(review_path).read_text(encoding="utf-8"))
            video_path = Path(video_arg)
            split = "val" if source_id == args.val_source_id else "train"
            records = collect_review_records(
                source_id=source_id,
                plan=plan,
                review=review,
                video_path=video_path,
                split=split,
            )
            source_specs.append(
                {
                    "source_id": source_id,
                    "video_sha256": file_sha256(video_path),
                    "plan_sha256": plan["artifact_sha256"],
                    "review_sha256": review["artifact_sha256"],
                    "split": split,
                }
            )
            all_records.extend(records)

        # De-duplicate repeated plans that refer to the same source frame.  The
        # manifest builder has already rejected contradictory labels.
        unique: dict[tuple[str, int], dict[str, Any]] = {}
        for record in all_records:
            key = (record["source_video_sha256"], int(record["frame"]))
            unique.setdefault(key, record)
        for index, record in enumerate(
            sorted(unique.values(), key=lambda row: (row["split"], row["source_id"], int(row["frame"])))
        ):
            split = str(record["split"])
            stem = f"{record['source_id']}-{int(record['frame']):08d}-{index:04d}"
            image_path = output / "images" / split / f"{stem}.jpg"
            label_path = output / "labels" / split / f"{stem}.txt"
            cap = video_cache.get(record["source_video_sha256"])
            if cap is None:
                video_path = next(
                    Path(source[3])
                    for source in args.source
                    if source[0] == record["source_id"]
                )
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    raise ValueError(f"could not open video {video_path}")
                video_cache[record["source_video_sha256"]] = cap
            frame = _read_frame(cap, int(record["frame"]))
            pixels_sha = __import__("hashlib").sha256(frame.tobytes()).hexdigest()
            if pixels_sha != record["frame_pixels_sha256"]:
                raise ValueError(f"frame pixel hash changed for {record['candidate_id']}")
            if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise ValueError(f"could not write {image_path}")
            _write_label(label_path, record)
            record["image_path"] = str(image_path.relative_to(output))
            record["label_path"] = str(label_path.relative_to(output))
            record["image_sha256"] = file_sha256(image_path)
            record["label_sha256"] = file_sha256(label_path)
    finally:
        for cap in video_cache.values():
            cap.release()

    manifest = build_dataset_manifest(source_specs, records=list(unique.values()))
    manifest["records"] = [
        dict(record)
        for record in sorted(
            unique.values(), key=lambda row: (row["split"], row["source_id"], int(row["frame"]))
        )
    ]
    # Paths and file hashes are part of the sealed artifact, so recompute after
    # image materialization.
    manifest.pop("artifact_sha256", None)
    from app.analysis.broadcast_ball_detector_dataset import canonical_sha256

    manifest["artifact_sha256"] = canonical_sha256(manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "data.yaml").write_text(
        "path: " + str(output) + "\ntrain: images/train\nval: images/val\nnames:\n  0: basketball\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "train": manifest["split_counts"]["train"],
                "val": manifest["split_counts"]["val"],
                "decision_counts": manifest["decision_counts"],
                "artifact_sha256": manifest["artifact_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
