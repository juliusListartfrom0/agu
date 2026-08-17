from __future__ import annotations

import pytest

from app.analysis.pbp_event_alignment import (
    build_pbp_event_alignment_artifact,
    game_elapsed_seconds,
    interpolate_video_frame,
    verify_pbp_event_alignment_artifact,
)
from app.analysis.shot_overlay_state import seal_scoreboard_timeline_artifact


def _timeline() -> dict[str, object]:
    return seal_scoreboard_timeline_artifact(
        {
            "reader_method": "rapidocr_broadcast_scoreboard_v3",
            "raw_video_sha256": "video-sha",
            "team_ids": ["HOME", "AWAY"],
            "stride_frames": 60,
            "reads": [
                {"frame": 0, "scores": {"HOME": 0, "AWAY": 0}, "confidence": 0.9},
                {"frame": 60, "scores": {"HOME": 2, "AWAY": 0}, "confidence": 0.9},
                {"frame": 120, "scores": {"HOME": 249, "AWAY": 0}, "confidence": 0.9},
                {"frame": 180, "scores": {"HOME": 2, "AWAY": 3}, "confidence": 0.9},
                {"frame": 240, "scores": {"HOME": 2, "AWAY": 0}, "confidence": 0.9},
                {"frame": 300, "scores": {"HOME": 2, "AWAY": 5}, "confidence": 0.9},
            ],
        }
    )


def _rows() -> list[dict[str, object]]:
    return [
        {
            "actionId": 1,
            "period": 1,
            "clock": "PT12M00.00S",
            "teamTricode": "",
            "personId": 0,
            "playerName": "",
            "description": "Start",
            "actionType": "period",
            "subType": "start",
            "scoreHome": "0",
            "scoreAway": "0",
        },
        {
            "actionId": 2,
            "period": 1,
            "clock": "PT11M00.00S",
            "teamTricode": "HOME",
            "personId": 11,
            "playerName": "Home Player",
            "description": "Home 2PT",
            "actionType": "Made Shot",
            "subType": "Jump Shot",
            "scoreHome": "2",
            "scoreAway": "0",
        },
        {
            "actionId": 3,
            "period": 1,
            "clock": "PT10M30.00S",
            "teamTricode": "AWAY",
            "personId": 22,
            "playerName": "Away Player",
            "description": "Turnover",
            "actionType": "Turnover",
            "subType": "Bad Pass",
            "scoreHome": "",
            "scoreAway": "",
        },
        {
            "actionId": 4,
            "period": 1,
            "clock": "PT09M00.00S",
            "teamTricode": "AWAY",
            "personId": 22,
            "playerName": "Away Player",
            "description": "Away 3PT",
            "actionType": "Made Shot",
            "subType": "3PT Jump Shot",
            "scoreHome": "2",
            "scoreAway": "3",
        },
        {
            "actionId": 5,
            "period": 1,
            "clock": "PT08M30.00S",
            "teamTricode": "HOME",
            "personId": 11,
            "playerName": "Home Player",
            "description": "Home FT",
            "actionType": "Free Throw",
            "subType": "Free Throw 1 of 2",
            "scoreHome": "2",
            "scoreAway": "5",
        },
    ]


def test_game_clock_handles_regulation_and_overtime() -> None:
    assert game_elapsed_seconds(1, "PT12M00.00S") == 0.0
    assert game_elapsed_seconds(4, "PT00M00.00S") == 2880.0
    assert game_elapsed_seconds(5, "PT05M00.00S") == 2880.0
    assert game_elapsed_seconds(6, "PT00M00.00S") == 3480.0


def test_interpolation_is_bounded_and_monotonic() -> None:
    anchors = [
        {"game_elapsed_seconds": 10.0, "video_frame": 100},
        {"game_elapsed_seconds": 20.0, "video_frame": 200},
    ]
    assert interpolate_video_frame(5.0, anchors) is None
    assert interpolate_video_frame(10.0, anchors) == 100
    assert interpolate_video_frame(15.0, anchors) == 150
    assert interpolate_video_frame(25.0, anchors) is None


def test_alignment_reconciles_official_states_and_rejects_ocr_outliers() -> None:
    artifact = build_pbp_event_alignment_artifact(
        play_by_play_rows=_rows(),
        scoreboard_timeline=_timeline(),
        game_id="game-1",
        home_team_id="HOME",
        away_team_id="AWAY",
        source_play_by_play_sha256="pbp-sha",
    )

    assert artifact["runtime_consumable"] is False
    assert artifact["codex_runtime_answer_used"] is False
    assert artifact["coverage"]["score_anchor_count"] == 4
    assert artifact["coverage"]["mapped_event_count"] == 5
    assert all(row["video_frame"] is not None for row in artifact["events"])
    assert [row["video_frame"] for row in artifact["events"]] == [0, 60, 90, 180, 300]
    assert verify_pbp_event_alignment_artifact(artifact)["artifact_sha256"] == artifact[
        "artifact_sha256"
    ]


def test_alignment_forward_fills_zero_score_placeholders() -> None:
    rows = [
        {
            "actionId": 1,
            "period": 1,
            "clock": "PT12M00.00S",
            "actionType": "period",
            "subType": "start",
            "scoreHome": "0",
            "scoreAway": "0",
        },
        {
            "actionId": 2,
            "period": 1,
            "clock": "PT11M30.00S",
            "actionType": "Made Shot",
            "scoreHome": "2",
            "scoreAway": "0",
        },
        {
            "actionId": 3,
            "period": 1,
            "clock": "PT11M10.00S",
            "actionType": "Missed Shot",
            "scoreHome": "0",
            "scoreAway": "0",
        },
    ]
    artifact = build_pbp_event_alignment_artifact(
        play_by_play_rows=rows,
        scoreboard_timeline=seal_scoreboard_timeline_artifact(
            {
                "reader_method": "rapidocr_broadcast_scoreboard_v3",
                "raw_video_sha256": "video-sha-forward-fill",
                "team_ids": ["HOME", "AWAY"],
                "stride_frames": 60,
                "reads": [
                    {
                        "frame": 0,
                        "scores": {"HOME": 0, "AWAY": 0},
                        "confidence": 0.9,
                    },
                    {
                        "frame": 60,
                        "scores": {"HOME": 2, "AWAY": 0},
                        "confidence": 0.9,
                    },
                ],
            }
        ),
        game_id="game-forward-fill",
        home_team_id="HOME",
        away_team_id="AWAY",
        source_play_by_play_sha256="pbp-forward-fill",
    )
    assert [event["score_state"] for event in artifact["events"]] == [
        (0, 0),
        (2, 0),
        (2, 0),
    ]
    assert [event["video_frame"] for event in artifact["events"]] == [0, 60, None]


def test_alignment_fails_closed_for_a_sealed_blind_video() -> None:
    with pytest.raises(ValueError, match="sealed blind"):
        build_pbp_event_alignment_artifact(
            play_by_play_rows=_rows(),
            scoreboard_timeline=_timeline(),
            game_id="game-1",
            home_team_id="HOME",
            away_team_id="AWAY",
            source_play_by_play_sha256="pbp-sha",
            sealed_blind_video_sha256s=["video-sha"],
        )


def test_alignment_verifier_rejects_runtime_flags() -> None:
    artifact = build_pbp_event_alignment_artifact(
        play_by_play_rows=_rows(),
        scoreboard_timeline=_timeline(),
        game_id="game-1",
        home_team_id="HOME",
        away_team_id="AWAY",
        source_play_by_play_sha256="pbp-sha",
    )
    artifact["runtime_consumable"] = True
    with pytest.raises(ValueError, match="offline"):
        verify_pbp_event_alignment_artifact(artifact)
