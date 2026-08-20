from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.analysis.muvs_event_state import (
    build_muvs_event_state_plan,
    seal_muvs_event_state_frames,
    seal_muvs_event_state_plan,
    seal_muvs_event_state_review,
    verify_muvs_event_state_frames,
    verify_muvs_event_state_plan,
    verify_muvs_event_state_review,
)


def _write_event(
    root: Path,
    *,
    event: str,
    split: str,
    sport: str = "basketball",
) -> None:
    annotations = root / "annotations"
    fragments = root / "fragments" / split
    annotations.mkdir(parents=True, exist_ok=True)
    fragments.mkdir(parents=True, exist_ok=True)
    with (annotations / f"{event}_ANNOTATIONS.csv").open(
        "w",
        newline="",
        encoding="utf-8",
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
        for fragment in range(1, 7):
            writer.writerow(
                {
                    "dataset_type": split,
                    "event": f"{event}_filtered",
                    "period": 3 if fragment <= 3 else 5,
                    "fragment_number": fragment if fragment <= 3 else fragment - 3,
                    "camera_selected": "cam_2",
                    "offsets": json.dumps({"cam_1": 0.0, "cam_2": 0.25}),
                }
            )
    with (fragments / f"{event}_FRAGMENTS.csv").open(
        "w",
        newline="",
        encoding="utf-8",
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


def test_muvs_plan_is_balanced_deterministic_and_label_hidden(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    _write_event(data, event="event_a", split="train")
    _write_event(data, event="event_b", split="test")
    _write_event(data, event="soccer_event", split="train", sport="soccer")

    plan = build_muvs_event_state_plan(
        data,
        samples_per_event=2,
        expected_basketball_events=2,
    )

    assert plan["source"]["record_id"] == 20_708_683
    assert plan["source"]["record_license"] == "CC-BY-4.0"
    assert plan["source"]["package_readme_license"] == "TO BE ADDED"
    assert plan["runtime_consumable"] is False
    assert plan["codex_runtime_answer_used"] is False
    assert plan["reviewer_visible_fields"] == [
        "sample_id",
        "frame_before",
        "frame_after",
    ]
    assert plan["event_count"] == 2
    assert plan["sample_count"] == 4
    assert {row["event_id"] for row in plan["samples"]} == {
        "event_a",
        "event_b",
    }
    assert all(
        set(row)
        == {
            "sample_id",
            "split",
            "event_id",
            "period",
            "fragment_number",
            "camera_id",
            "camera_offset_sec",
            "frame_times_sec",
            "video_url",
            "annotation_sha256",
            "fragments_sha256",
        }
        for row in plan["samples"]
    )
    assert plan["samples"][0]["frame_times_sec"] == [4.25, 5.25]
    assert "label" not in json.dumps(plan)
    assert verify_muvs_event_state_plan(plan) == plan


def test_muvs_plan_rejects_source_drift_and_non_basketball_shortfall(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    _write_event(data, event="event_a", split="train")
    with pytest.raises(ValueError, match="basketball event count"):
        build_muvs_event_state_plan(
            data,
            samples_per_event=2,
            expected_basketball_events=2,
        )

    plan = build_muvs_event_state_plan(
        data,
        samples_per_event=2,
        expected_basketball_events=1,
    )
    plan["samples"][0]["camera_id"] = "cam_9"
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_muvs_event_state_plan(plan)


def test_muvs_plan_can_be_resealed_after_label_hidden_source_repair(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    _write_event(data, event="event_a", split="train")
    original = build_muvs_event_state_plan(
        data,
        samples_per_event=2,
        expected_basketball_events=1,
    )
    repaired_payload = dict(original)
    repaired_payload["samples"] = [
        dict(row) for row in original["samples"]
    ]
    repaired_payload["samples"][0]["fragment_number"] = 1
    repaired_payload["samples"][0]["frame_times_sec"] = [1.25, 2.25]

    repaired = seal_muvs_event_state_plan(repaired_payload)

    assert repaired["artifact_sha256"] != original["artifact_sha256"]
    assert repaired["runtime_consumable"] is False
    assert repaired["codex_runtime_answer_used"] is False
    assert verify_muvs_event_state_plan(repaired) == repaired


def test_muvs_plan_can_limit_sampling_to_named_basketball_events(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    _write_event(data, event="event_a", split="train")
    _write_event(data, event="event_b", split="test")
    _write_event(data, event="event_c", split="train")

    plan = build_muvs_event_state_plan(
        data,
        samples_per_event=2,
        expected_basketball_events=2,
        included_event_ids=("event_a", "event_b"),
    )

    assert plan["event_count"] == 2
    assert {str(row["event_id"]) for row in plan["samples"]} == {"event_a", "event_b"}
    with pytest.raises(ValueError, match="requested event"):
        build_muvs_event_state_plan(
            data,
            samples_per_event=2,
            expected_basketball_events=2,
            included_event_ids=("event_a", "missing"),
        )


def test_muvs_plan_can_exclude_prior_source_coordinates(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    _write_event(data, event="event_a", split="train")
    _write_event(data, event="event_b", split="test")
    original = build_muvs_event_state_plan(
        data,
        samples_per_event=2,
        expected_basketball_events=2,
    )

    extension = build_muvs_event_state_plan(
        data,
        samples_per_event=4,
        expected_basketball_events=2,
        excluded_plans=(original,),
    )

    def source_coordinates(plan: dict[str, object]) -> set[tuple[object, ...]]:
        return {
            (
                row["event_id"],
                row["period"],
                row["fragment_number"],
                row["camera_id"],
            )
            for row in plan["samples"]  # type: ignore[index, union-attr]
        }

    assert source_coordinates(original).isdisjoint(
        source_coordinates(extension)
    )
    assert extension == build_muvs_event_state_plan(
        data,
        samples_per_event=4,
        expected_basketball_events=2,
        excluded_plans=(original,),
    )
    assert "label" not in json.dumps(extension)


def test_muvs_review_requires_exact_coverage_and_source_only_states(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    _write_event(data, event="event_a", split="train")
    plan = build_muvs_event_state_plan(
        data,
        samples_per_event=2,
        expected_basketball_events=1,
    )
    files = []
    for row in plan["samples"]:
        files.append(
            {
                "sample_id": row["sample_id"],
                "frame_before": {
                    "path": f"frames/{row['sample_id']}_before.png",
                    "sha256": "a" * 64,
                    "size_bytes": 100,
                    "width": 1280,
                    "height": 720,
                },
                "frame_after": {
                    "path": f"frames/{row['sample_id']}_after.png",
                    "sha256": "b" * 64,
                    "size_bytes": 101,
                    "width": 1280,
                    "height": 720,
                },
            }
        )
    frames = seal_muvs_event_state_frames(
        {"files": files},
        plan=plan,
    )
    assert frames["frame_count"] == 4
    assert verify_muvs_event_state_frames(frames, plan=plan) == frames
    decisions = [
        {
            "sample_id": plan["samples"][0]["sample_id"],
            "state": "live_play",
            "notes": "Players are spatially spread in active transition.",
        },
        {
            "sample_id": plan["samples"][1]["sample_id"],
            "state": "dead_ball_timeout",
            "notes": "Players and staff are gathered at the sideline.",
        },
    ]

    review = seal_muvs_event_state_review(
        {"reviewer": "codex_offline_source_annotation", "decisions": decisions},
        plan=plan,
        frames=frames,
    )

    assert review["runtime_consumable"] is False
    assert review["codex_runtime_answer_used"] is False
    assert review["plan_sha256"] == plan["artifact_sha256"]
    assert review["frames_sha256"] == frames["artifact_sha256"]
    assert verify_muvs_event_state_review(review, plan=plan, frames=frames) == review
    review["decisions"][0]["state"] = "made_shot"
    with pytest.raises(ValueError, match="state decision"):
        verify_muvs_event_state_review(review, plan=plan, frames=frames)
