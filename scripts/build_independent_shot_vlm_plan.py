#!/usr/bin/env python3
"""Freeze a balanced, label-free development plan for an independent shot VLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.independent_shot_vlm import (  # noqa: E402
    seal_independent_shot_vlm_plan,
    stable_example_rank,
)
from app.analysis.official_evaluation import verify_raw_only_bundle  # noqa: E402
from app.analysis.schemas import RawOnlyPredictionBundleResponse  # noqa: E402
from app.analysis.shot_validity import SHOT_VALIDITY_LABEL_SCHEMA  # noqa: E402
from app.analysis.training_annotation import verify_training_annotation_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--pair",
        nargs=3,
        metavar=("CANDIDATE_BUNDLE", "ANNOTATION", "VIDEO"),
        action="append",
        required=True,
    )
    parser.add_argument("--per-class-per-video", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_plan(
    *,
    manifest_path: Path,
    pairs: list[tuple[Path, Path, Path]],
    per_class_per_video: int,
) -> dict[str, object]:
    if per_class_per_video <= 0:
        raise ValueError("per-class-per-video must be positive")
    manifest = verify_training_annotation_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    allowed_sources = {str(row["sha256"]) for row in manifest["source_videos"]}
    allowed_annotations = {
        (str(row["filename"]), str(row["sha256"]))
        for row in manifest["annotation_files"]
    }
    labeled_rows: dict[tuple[str, str, str], dict[str, object]] = {}
    annotation_sha256: set[str] = set()
    video_details: dict[str, tuple[str, float]] = {}
    for bundle_path, annotation_path, video_path in pairs:
        bundle = verify_raw_only_bundle(
            RawOnlyPredictionBundleResponse.model_validate_json(
                bundle_path.read_text(encoding="utf-8")
            )
        )
        if len(bundle.raw_videos) != 1:
            raise ValueError("each candidate bundle must bind exactly one video")
        raw = bundle.raw_videos[0]
        video_sha = _file_sha256(video_path)
        if video_sha != raw.sha256 or video_sha not in allowed_sources:
            raise ValueError("video is not SHA-bound to the manifest and candidate bundle")
        label_sha = _file_sha256(annotation_path)
        if (annotation_path.name, label_sha) not in allowed_annotations:
            raise ValueError("annotation is not SHA-bound to the training manifest")
        labels = json.loads(annotation_path.read_text(encoding="utf-8"))
        if labels.get("schema_version") != SHOT_VALIDITY_LABEL_SCHEMA:
            raise ValueError("unsupported shot-validity label schema")
        if labels.get("runtime_consumable") is not False:
            raise ValueError("shot-validity annotations must be training-only")
        if labels.get("source_video_sha256") != video_sha:
            raise ValueError("shot-validity annotation source mismatch")
        if labels.get("candidate_bundle_sha256") != bundle.bundle_sha256:
            raise ValueError("shot-validity annotation candidate mismatch")
        fps = _video_fps(video_path)
        prior = video_details.setdefault(video_sha, (raw.filename, fps))
        if prior != (raw.filename, fps):
            raise ValueError("inconsistent video metadata across plan inputs")
        events = {event.event_id: event for event in bundle.events}
        for label in labels.get("examples", []):
            event_id = str(label.get("event_id") or "")
            event = events.get(event_id)
            if event is None or event.event_type != "field_goal_attempt":
                raise ValueError("label does not name a field-goal candidate")
            event_present = label.get("event_present")
            if not isinstance(event_present, bool):
                raise ValueError("shot-validity labels must be boolean")
            key = (video_sha, str(bundle.bundle_sha256), event_id)
            labeled_rows[key] = {
                "source_video_sha256": video_sha,
                "source_video_filename": raw.filename,
                "candidate_bundle_sha256": str(bundle.bundle_sha256),
                "event_id": event_id,
                "start_frame": event.start_frame,
                "end_frame": event.end_frame,
                "source_fps": fps,
                "_event_present": event_present,
            }
        annotation_sha256.add(label_sha)

    by_video_class: dict[tuple[str, bool], list[dict[str, object]]] = defaultdict(list)
    for row in labeled_rows.values():
        by_video_class[
            (str(row["source_video_sha256"]), bool(row["_event_present"]))
        ].append(row)
    selected: list[dict[str, object]] = []
    for video_sha in sorted(video_details):
        for label in (False, True):
            choices = sorted(
                by_video_class[(video_sha, label)],
                key=lambda row: stable_example_rank(
                    video_sha,
                    str(row["candidate_bundle_sha256"]),
                    str(row["event_id"]),
                ),
            )
            if len(choices) < per_class_per_video:
                raise ValueError(
                    f"video {video_sha} lacks {per_class_per_video} examples for class {label}"
                )
            for choice in choices[:per_class_per_video]:
                selected.append(
                    {key: value for key, value in choice.items() if not key.startswith("_")}
                )
    selected.sort(
        key=lambda row: (
            str(row["source_video_sha256"]),
            int(row["start_frame"]),
            str(row["event_id"]),
        )
    )
    return seal_independent_shot_vlm_plan(
        {
            "training_manifest_sha256": manifest["manifest_sha256"],
            "training_annotation_sha256": sorted(annotation_sha256),
            "selection": {
                "method": "sha256_rank_within_video_and_boolean_class",
                "positive_per_video": per_class_per_video,
                "negative_per_video": per_class_per_video,
                "video_count": len(video_details),
                "example_count": len(selected),
            },
            "input_contract": {
                "raw_frames_only": True,
                "labels_or_review_notes_exposed_to_model": False,
                "chronological_even_sampling": True,
                "max_frames": 6,
                "image_width": 704,
            },
            "examples": selected,
        }
    )


def _video_fps(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if fps <= 0:
        raise ValueError(f"could not read video FPS: {path}")
    return fps


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    plan = build_plan(
        manifest_path=args.manifest,
        pairs=[
            (Path(bundle), Path(annotation), Path(video))
            for bundle, annotation, video in args.pair
        ],
        per_class_per_video=args.per_class_per_video,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "plan_sha256": plan["plan_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
