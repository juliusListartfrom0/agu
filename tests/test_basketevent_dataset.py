from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.analysis.basketevent_dataset import (
    build_basketevent_trajectory_audit,
    verify_basketevent_trajectory_audit,
)


def _write_clip(root: Path, split: str, game: str, name: str, payload: dict) -> None:
    path = root / split / game / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _payload(*, event: dict | None, ball: list[list[float] | None]) -> dict:
    return {
        "player_0": {
            "jersey_number": 7,
            "jersey_color": "blue",
            "player_name": "Test Player",
            "event": event,
            "trajectory": [[10, 20, 30, 40], None],
        },
        "ball": {"trajectory": ball},
    }


def test_basketevent_audit_counts_events_boxes_and_ball_visibility(tmp_path: Path) -> None:
    _write_clip(
        tmp_path,
        "valid",
        "game-a",
        "clip-a.json",
        _payload(
            event={
                "clock": "PT01M02.00S",
                "period": 2,
                "teamTricode": "AAA",
                "playerName": "Test Player",
                "description": "Test Player 3PT Shot",
                "actionType": "Made Shot",
                "subType": "Jump Shot",
            },
            ball=[[1, 2, 3, 4], None],
        ),
    )
    _write_clip(
        tmp_path,
        "valid",
        "game-b",
        "clip-b.json",
        _payload(event=None, ball=[None, None]),
    )

    audit = build_basketevent_trajectory_audit(
        tmp_path,
        source_url="https://example.invalid/basketevent",
        source_revision="revision",
        source_license=None,
    )

    assert audit["clip_count"] == 2
    assert audit["game_count"] == 2
    assert audit["event_clip_count"] == 1
    assert audit["event_action_type_counts"] == {"Made Shot": 1}
    assert audit["player_track_count"] == 2
    assert audit["ball_track_count"] == 2
    assert audit["ball_observation_count"] == 1
    assert audit["ball_observation_frame_count"] == 1
    assert audit["runtime_consumable"] is False
    assert audit["training_media_eligible"] is False
    assert verify_basketevent_trajectory_audit(audit)["artifact_sha256"] == audit[
        "artifact_sha256"
    ]


def test_basketevent_audit_rejects_non_box_trajectory(tmp_path: Path) -> None:
    _write_clip(
        tmp_path,
        "valid",
        "game-a",
        "clip-a.json",
        {
            "player_0": {"event": None, "trajectory": [[1, 2, 3]]},
            "ball": {"trajectory": [[1, 2, 3, 4]]},
        },
    )

    with pytest.raises(ValueError, match="trajectory box"):
        build_basketevent_trajectory_audit(
            tmp_path,
            source_url="https://example.invalid/basketevent",
            source_revision="revision",
            source_license=None,
        )


def test_basketevent_audit_records_missing_ball_track(tmp_path: Path) -> None:
    _write_clip(
        tmp_path,
        "valid",
        "game-a",
        "clip-a.json",
        {"player_0": {"event": None, "trajectory": [[1, 2, 3, 4], None]}},
    )

    audit = build_basketevent_trajectory_audit(
        tmp_path,
        source_url="https://example.invalid/basketevent",
        source_revision="revision",
        source_license=None,
    )

    assert audit["ball_track_count"] == 0
    assert audit["missing_ball_track_clip_count"] == 1
    assert audit["ball_observation_count"] == 0
