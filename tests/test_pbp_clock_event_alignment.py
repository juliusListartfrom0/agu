from __future__ import annotations

import pytest

from app.analysis.broadcast_clock import seal_broadcast_clock_artifact
from app.analysis.pbp_clock_event_alignment import (
    build_pbp_clock_event_alignment_artifact,
    verify_pbp_clock_event_alignment_artifact,
)


def _clock() -> dict[str, object]:
    return seal_broadcast_clock_artifact(
        {
            "purpose": "offline_raw_only_broadcast_clock_replay_screening",
            "schema_version": "agu.broadcast-clock-replay-evidence.v1",
            "runtime_consumable": False,
            "codex_runtime_answer_used": False,
            "raw_video_sha256": "a" * 64,
            "candidate_bundle_sha256": "b" * 64,
            "source": {"ocr_engine": "test", "parser": "test"},
            "configuration": {"sample_offset_seconds": [0.0]},
            "events": [
                {
                    "event_id": "clock-1",
                    "anchor_frame": 300,
                    "samples": [
                        {
                            "frame": 300,
                            "offset_seconds": 0.0,
                            "read": {
                                "frame": 300,
                                "period": 1,
                                "clock_seconds": 690,
                                "confidence": 0.9,
                                "raw_text": "1ST11:30",
                                "source": "test",
                            },
                        }
                    ],
                    "pre_anchor_read_count": 1,
                    "post_anchor_missing_count": 0,
                    "frozen_period": None,
                    "frozen_clock_seconds": None,
                    "broadcast_state": "unknown",
                    "method": "test",
                }
            ],
        }
    )


def _rows() -> list[dict[str, object]]:
    return [
        {
            "actionId": 1,
            "period": 1,
            "clock": "PT11M30.00S",
            "actionType": "Made Shot",
            "scoreHome": "2",
            "scoreAway": "0",
            "teamTricode": "HOME",
        },
        {
            "actionId": 2,
            "period": 1,
            "clock": "PT11M00.00S",
            "actionType": "Turnover",
            "scoreHome": "2",
            "scoreAway": "0",
            "teamTricode": "AWAY",
        },
    ]


def test_clock_alignment_uses_same_period_and_clock_read() -> None:
    artifact = build_pbp_clock_event_alignment_artifact(
        play_by_play_rows=_rows(),
        clock_artifact=_clock(),
        game_id="game-1",
        home_team_id="HOME",
        away_team_id="AWAY",
        source_play_by_play_sha256="c" * 64,
    )
    assert artifact["coverage"] == {
        "event_count": 2,
        "mapped_event_count": 1,
        "mapped_event_fraction": 0.5,
        "unmapped_event_count": 1,
        "clock_read_count": 1,
    }
    assert artifact["events"][0]["video_frame"] == 300
    assert artifact["events"][0]["alignment_status"] == "clock_read"
    assert artifact["events"][1]["video_frame"] is None
    assert verify_pbp_clock_event_alignment_artifact(artifact)["artifact_sha256"] == artifact[
        "artifact_sha256"
    ]


def test_clock_alignment_rejects_sealed_blind_video() -> None:
    with pytest.raises(ValueError, match="sealed blind"):
        build_pbp_clock_event_alignment_artifact(
            play_by_play_rows=_rows(),
            clock_artifact=_clock(),
            game_id="game-1",
            home_team_id="HOME",
            away_team_id="AWAY",
            source_play_by_play_sha256="c" * 64,
            sealed_blind_video_sha256s=["a" * 64],
        )
