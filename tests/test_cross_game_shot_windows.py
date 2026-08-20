from __future__ import annotations

import pytest

from app.analysis.cross_game_shot_windows import (
    seal_shot_window_spec,
    select_uniform_shot_windows,
    verify_shot_window_spec,
)


def test_uniform_shot_windows_are_deterministic_and_non_overlapping() -> None:
    first = select_uniform_shot_windows(
        frame_count=10_000,
        fps=25.0,
        count=12,
        window_frames=120,
        minimum_center_spacing_frames=300,
        seed=20260806,
    )
    second = select_uniform_shot_windows(
        frame_count=10_000,
        fps=25.0,
        count=12,
        window_frames=120,
        minimum_center_spacing_frames=300,
        seed=20260806,
    )
    assert first == second
    assert len(first) == 12
    centers = [int(row["anchor_frame"]) for row in first]
    assert centers == sorted(centers)
    assert all(
        0 <= int(row["start_frame"]) < int(row["end_frame"]) < 10_000
        for row in first
    )
    assert all(
        second_center - first_center >= 300
        for first_center, second_center in zip(centers, centers[1:], strict=False)
    )
    # The seeded slack composition must cover the source instead of silently
    # pushing nearly every candidate toward the final gap.
    assert centers[0] < 1_000
    assert centers[-1] > 9_000


def test_uniform_shot_windows_reject_impossible_spacing() -> None:
    with pytest.raises(ValueError, match="cannot place"):
        select_uniform_shot_windows(
            frame_count=500,
            fps=25.0,
            count=4,
            window_frames=120,
            minimum_center_spacing_frames=200,
            seed=1,
        )


def test_shot_window_spec_hash_round_trip_and_rejects_tampering() -> None:
    spec = seal_shot_window_spec(
        {
            "schema_version": "agu.cross-game-shot-window-spec.v1",
            "purpose": "offline_training_annotation_only",
            "runtime_consumable": False,
            "source_video_sha256": "a" * 64,
            "video_frame_count": 1_000,
            "video_fps": 25.0,
            "selection_seed": 7,
            "window_frames": 120,
            "minimum_center_spacing_frames": 200,
            "candidates": [
                {
                    "event_id": "raw-shot-001",
                    "start_frame": 10,
                    "end_frame": 129,
                    "anchor_frame": 70,
                }
            ],
        }
    )
    assert verify_shot_window_spec(spec)["artifact_sha256"] == spec["artifact_sha256"]
    tampered = dict(spec)
    tampered["selection_seed"] = 8
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_shot_window_spec(tampered)
