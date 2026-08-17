from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.analysis.muvs_dense_sequence import (
    build_muvs_dense_sequence_plan,
    seal_muvs_dense_sequence_frames,
    seal_muvs_dense_sequence_review,
    verify_muvs_dense_sequence_frames,
    verify_muvs_dense_sequence_plan,
    verify_muvs_dense_sequence_review,
)
from app.analysis.muvs_event_state import build_muvs_event_state_plan


def _write_event(root: Path, *, event: str, split: str, sport: str = "basketball") -> None:
    annotations = root / "annotations"
    fragments = root / "fragments" / split
    annotations.mkdir(parents=True, exist_ok=True)
    fragments.mkdir(parents=True, exist_ok=True)
    with (annotations / f"{event}_ANNOTATIONS.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset_type",
                "event",
                "period",
                "fragment_number",
                "camera_selected",
                "offsets",
            ],
        )
        writer.writeheader()
        for fragment in range(1, 9):
            writer.writerow(
                {
                    "dataset_type": split,
                    "event": f"{event}_filtered",
                    "period": 3 if fragment <= 4 else 5,
                    "fragment_number": fragment if fragment <= 4 else fragment - 4,
                    "camera_selected": "cam_2",
                    "offsets": json.dumps({"cam_1": 0.0, "cam_2": 0.25}),
                }
            )
    with (fragments / f"{event}_FRAGMENTS.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sport_genre", "event_id", "camera_id", "period"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sport_genre": sport,
                "event_id": event,
                "camera_id": "cam_2",
                "period": 3,
            }
        )


def test_dense_plan_is_five_frame_label_hidden_and_excludes_old_coordinates(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    _write_event(data, event="event_a", split="train")
    _write_event(data, event="event_b", split="test")
    old_plan = build_muvs_event_state_plan(
        data,
        samples_per_event=2,
        expected_basketball_events=2,
    )

    plan = build_muvs_dense_sequence_plan(
        data,
        samples_per_event=2,
        expected_basketball_events=2,
        excluded_plans=(old_plan,),
    )

    assert plan["schema_version"] == "agu.muvs-dense-sequence-plan.v1"
    assert plan["sequence_length"] == 5
    assert plan["frame_offsets_sec"] == [0.25, 0.75, 1.25, 1.75, 2.25]
    assert plan["runtime_consumable"] is False
    assert plan["codex_runtime_answer_used"] is False
    assert plan["reviewer_visible_fields"] == ["sample_id", "frames"]
    assert all(len(row["frame_times_sec"]) == 5 for row in plan["samples"])
    assert all(
        all(a < b for a, b in zip(row["frame_times_sec"], row["frame_times_sec"][1:]))
        for row in plan["samples"]
    )
    old_coordinates = {
        (
            row["event_id"],
            row["period"],
            row["fragment_number"],
            row["camera_id"],
        )
        for row in old_plan["samples"]
    }
    new_coordinates = {
        (
            row["event_id"],
            row["period"],
            row["fragment_number"],
            row["camera_id"],
        )
        for row in plan["samples"]
    }
    assert old_coordinates.isdisjoint(new_coordinates)
    assert "state" not in json.dumps(plan)
    assert verify_muvs_dense_sequence_plan(plan) == plan


def test_dense_plan_requires_three_or_more_temporal_points(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_event(data, event="event_a", split="train")
    with pytest.raises(ValueError, match="three frame"):
        build_muvs_dense_sequence_plan(
            data,
            samples_per_event=1,
            expected_basketball_events=1,
            frame_offsets_sec=(0.0, 1.0),
        )


def test_dense_frames_and_review_are_exactly_bound_to_plan(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write_event(data, event="event_a", split="train")
    plan = build_muvs_dense_sequence_plan(
        data,
        samples_per_event=1,
        expected_basketball_events=1,
    )
    files = [
        {
            "sample_id": row["sample_id"],
            "frames": [
                {
                    "path": f"frames/{row['sample_id']}_{index:02d}.png",
                    "sha256": (chr(97 + index) * 64),
                    "size_bytes": 100 + index,
                    "width": 1280,
                    "height": 720,
                }
                for index in range(plan["sequence_length"])
            ],
        }
        for row in plan["samples"]
    ]
    frames = seal_muvs_dense_sequence_frames({"files": files}, plan=plan)
    assert frames["frame_count"] == 5
    assert verify_muvs_dense_sequence_frames(frames, plan=plan) == frames
    review = seal_muvs_dense_sequence_review(
        {
            "reviewer": "codex_offline_source_annotation",
            "decisions": [
                {
                    "sample_id": plan["samples"][0]["sample_id"],
                    "state": "live_play",
                    "notes": "Five-frame sequence shows active play.",
                }
            ],
        },
        plan=plan,
        frames=frames,
    )
    assert verify_muvs_dense_sequence_review(review, plan=plan, frames=frames) == review
    frames["files"][0]["frames"].pop()
    with pytest.raises(ValueError, match="frame row"):
        verify_muvs_dense_sequence_frames(frames, plan=plan)
