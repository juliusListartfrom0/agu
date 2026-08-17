#!/usr/bin/env python3
"""Materialize the sealed MUVY ball review as a YOLO detection dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.analysis.muvy_ball_review import (
    verify_muvy_ball_review,
    verify_muvy_ball_review_plan,
)
from app.analysis.muvy_ball_yolo import (
    build_muvy_ball_yolo_data_yaml,
    build_muvy_yolo_examples,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_sha256(frame: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(frame.shape)).encode())
    digest.update(frame.tobytes())
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(content)
    temporary.replace(path)


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--validation-event",
        action="append",
        required=True,
    )
    args = parser.parse_args()

    plan = verify_muvy_ball_review_plan(_read_json(args.plan))
    review = verify_muvy_ball_review(_read_json(args.review), plan=plan)
    examples = build_muvy_yolo_examples(
        plan,
        review,
        validation_events=set(args.validation_event),
    )
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"output directory is not empty: {args.output}")
    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        by_video[example["video_path"]].append(example)

    materialized = []
    for relative_video, video_examples in sorted(by_video.items()):
        video_path = args.source / relative_video
        if _file_sha256(video_path) != video_examples[0]["video_sha256"]:
            raise ValueError(f"MUVY source video hash mismatch: {relative_video}")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"cannot open MUVY source video: {relative_video}")
        try:
            for example in sorted(
                video_examples,
                key=lambda row: int(row["frame_index"]),
            ):
                capture.set(cv2.CAP_PROP_POS_FRAMES, example["frame_index"])
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ValueError(
                        f"cannot decode MUVY frame: {relative_video}"
                    )
                if (
                    frame.shape[1] != example["frame_width"]
                    or frame.shape[0] != example["frame_height"]
                    or _frame_sha256(frame) != example["frame_sha256"]
                ):
                    raise ValueError(
                        f"MUVY decoded frame hash mismatch: {relative_video}"
                    )
                stem = hashlib.sha256(
                    f"{relative_video}:{example['frame_id']}".encode()
                ).hexdigest()[:24]
                image_relative = Path("images") / example["split"] / f"{stem}.jpg"
                label_relative = Path("labels") / example["split"] / f"{stem}.txt"
                encoded_ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                if not encoded_ok:
                    raise ValueError("cannot encode MUVY training frame")
                _write_bytes(args.output / image_relative, encoded.tobytes())
                label_text = "".join(
                    " ".join(
                        [str(label[0]), *(f"{value:.10f}" for value in label[1:])]
                    )
                    + "\n"
                    for label in example["labels"]
                )
                _write_text(args.output / label_relative, label_text)
                materialized.append(
                    {
                        **example,
                        "image_path": image_relative.as_posix(),
                        "image_sha256": _file_sha256(
                            args.output / image_relative
                        ),
                        "label_path": label_relative.as_posix(),
                        "label_sha256": _file_sha256(
                            args.output / label_relative
                        ),
                    }
                )
        finally:
            capture.release()

    data_yaml = build_muvy_ball_yolo_data_yaml(args.output)
    _write_text(args.output / "data.yaml", data_yaml)
    manifest: dict[str, Any] = {
        "schema_version": "agu.muvy-ball-yolo.v1",
        "purpose": "licensed_public_small_ball_detector_training",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "source_record_url": "https://zenodo.org/records/13883315",
        "source_license": "CC-BY-4.0",
        "plan_sha256": plan["artifact_sha256"],
        "review_sha256": review["artifact_sha256"],
        "validation_events": sorted(set(args.validation_event)),
        "image_count": len(materialized),
        "box_count": sum(len(row["labels"]) for row in materialized),
        "split_counts": {
            split: sum(row["split"] == split for row in materialized)
            for split in ("train", "val")
        },
        "data_yaml_sha256": _file_sha256(args.output / "data.yaml"),
        "examples": materialized,
    }
    manifest["artifact_sha256"] = _canonical_sha256(manifest)
    _write_text(
        args.output / "manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "image_count": manifest["image_count"],
                "box_count": manifest["box_count"],
                "split_counts": manifest["split_counts"],
                "artifact_sha256": manifest["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
