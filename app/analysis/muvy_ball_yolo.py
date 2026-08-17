"""Build leakage-safe YOLO examples from the sealed MUVY ball review."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Set
from pathlib import Path
from typing import Any

from app.analysis.muvy_ball_review import (
    verify_muvy_ball_review,
    verify_muvy_ball_review_plan,
)


def build_muvy_ball_yolo_data_yaml(dataset_path: Path) -> str:
    """Build an Ultralytics YAML with an unambiguous dataset root."""

    return (
        f"path: {dataset_path.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: basketball\n"
        "  1: hoop\n"
        "  2: player\n"
        "  3: referee\n"
    )


def build_muvy_yolo_examples(
    plan: Mapping[str, Any],
    review: Mapping[str, Any],
    *,
    validation_events: Set[str],
) -> list[dict[str, Any]]:
    """Return positive examples split only at whole-event boundaries."""

    verified_plan = verify_muvy_ball_review_plan(plan)
    verified_review = verify_muvy_ball_review(review, plan=verified_plan)
    if not validation_events:
        raise ValueError("MUVY validation events must not be empty")

    decisions = {
        row["candidate_id"]: row["decision"]
        for row in verified_review["decisions"]
    }
    grouped: OrderedDict[tuple[str, int], dict[str, Any]] = OrderedDict()
    observed_events: set[str] = set()
    for candidate in verified_plan["candidates"]:
        if decisions[candidate["candidate_id"]] != "valid_ball":
            continue
        video_path = str(candidate["video_path"])
        event = video_path.split("/", 1)[0]
        observed_events.add(event)
        key = (video_path, int(candidate["frame_id"]))
        example = grouped.setdefault(
            key,
            {
                "event": event,
                "split": "val" if event in validation_events else "train",
                "video_path": video_path,
                "video_sha256": candidate["video_sha256"],
                "frame_id": candidate["frame_id"],
                "frame_index": candidate["frame_index"],
                "frame_width": candidate["frame_width"],
                "frame_height": candidate["frame_height"],
                "frame_sha256": candidate["frame_sha256"],
                "candidate_ids": [],
                "labels": [],
            },
        )
        example["candidate_ids"].append(candidate["candidate_id"])
        example["labels"].append(_normalised_yolo_box(candidate))

    missing_validation = validation_events - observed_events
    if missing_validation:
        raise ValueError("MUVY validation event is absent from valid labels")
    examples = list(grouped.values())
    splits = {example["split"] for example in examples}
    if splits != {"train", "val"}:
        raise ValueError("MUVY YOLO examples require train and validation data")
    return examples


def _normalised_yolo_box(candidate: Mapping[str, Any]) -> list[float | int]:
    x1, y1, x2, y2 = map(float, candidate["bbox"])
    frame_width = int(candidate["frame_width"])
    frame_height = int(candidate["frame_height"])
    return [
        0,
        (x1 + x2) / 2 / frame_width,
        (y1 + y2) / 2 / frame_height,
        (x2 - x1) / frame_width,
        (y2 - y1) / frame_height,
    ]
