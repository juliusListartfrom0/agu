from __future__ import annotations

import pytest

from app.analysis.pbp_event_windows import select_pbp_event_windows


def _alignment() -> dict[str, object]:
    events: list[dict[str, object]] = []
    for index, frame in enumerate(range(120, 1320, 120)):
        events.append(
            {
                "source_index": index,
                "period": 1 + index // 6,
                "clock": f"PT{11 - (index % 6)}M00.00S",
                "game_elapsed_seconds": float(index * 30),
                "video_frame": frame,
                "is_field_goal": index % 2 == 0,
                "action_type": "Made Shot" if index % 2 == 0 else "Rebound",
                "shot_result": "Made" if index % 2 == 0 else "",
            }
        )
    return {"events": events}


def test_selector_is_deterministic_and_stratifies_field_goals_and_negatives() -> None:
    first = select_pbp_event_windows(
        alignment=_alignment(),
        frame_count=2_000,
        fps=30.0,
        count=8,
        window_frames=60,
        minimum_center_spacing_frames=60,
        seed=7,
    )
    second = select_pbp_event_windows(
        alignment=_alignment(),
        frame_count=2_000,
        fps=30.0,
        count=8,
        window_frames=60,
        minimum_center_spacing_frames=60,
        seed=7,
    )

    assert first == second
    assert len(first) == 8
    assert {row["selection_kind"] for row in first} == {"field_goal", "hard_negative"}
    assert [row["anchor_frame"] for row in first] == sorted(
        row["anchor_frame"] for row in first
    )
    assert all(
        second["anchor_frame"] - first["anchor_frame"] >= 60
        for first, second in zip(first, first[1:])
    )
    assert all(row["source_fps"] == 30.0 for row in first)


def test_selector_fails_closed_when_not_enough_mapped_events() -> None:
    alignment = {"events": [{"video_frame": None, "is_field_goal": True}]}
    with pytest.raises(ValueError, match="enough mapped"):
        select_pbp_event_windows(
            alignment=alignment,
            frame_count=100,
            fps=30.0,
            count=2,
            window_frames=30,
            minimum_center_spacing_frames=30,
            seed=1,
        )


def test_selector_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="window_frames"):
        select_pbp_event_windows(
            alignment=_alignment(),
            frame_count=100,
            fps=30.0,
            count=2,
            window_frames=1,
            minimum_center_spacing_frames=30,
            seed=1,
        )
