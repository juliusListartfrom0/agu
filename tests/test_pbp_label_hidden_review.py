from __future__ import annotations

from app.analysis.broadcast_clock import seal_broadcast_clock_artifact
from app.analysis.pbp_clock_event_alignment import build_pbp_clock_event_alignment_artifact
from app.analysis.pbp_label_hidden_review import (
    build_label_hidden_queue,
    build_post_freeze_truth,
    select_review_batch,
    verify_queue_artifact,
    verify_review_batch_artifact,
    verify_truth_artifact,
)


def _alignment() -> dict[str, object]:
    clock = seal_broadcast_clock_artifact(
        {
            "schema_version": "agu.broadcast-clock-replay-evidence.v1",
            "purpose": "offline_raw_only_broadcast_clock_replay_screening",
            "runtime_consumable": False,
            "codex_runtime_answer_used": False,
            "raw_video_sha256": "a" * 64,
            "candidate_bundle_sha256": "offline",
            "source": {"ocr_engine": "test", "parser": "test"},
            "configuration": {"sample_offset_seconds": [0.0]},
            "events": [
                {
                    "event_id": "clock-690",
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
                },
                {
                    "event_id": "clock-600",
                    "anchor_frame": 3000,
                    "samples": [
                        {
                            "frame": 3000,
                            "offset_seconds": 0.0,
                            "read": {
                                "frame": 3000,
                                "period": 1,
                                "clock_seconds": 600,
                                "confidence": 0.9,
                                "raw_text": "1ST10:00",
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
                },
            ],
        }
    )
    return build_pbp_clock_event_alignment_artifact(
        play_by_play_rows=[
            {
                "actionId": 1,
                "period": 1,
                "clock": "PT11M30.00S",
                "teamTricode": "HOME",
                "personId": 1,
                "playerName": "Player",
                "actionType": "Made Shot",
                "subType": "Jump Shot",
                "description": "Player 2PT Jump Shot",
                "shotResult": "Made",
                "shotValue": 2,
                "isFieldGoal": 1,
                "scoreHome": "2",
                "scoreAway": "0",
            },
            {
                "actionId": 2,
                "period": 1,
                "clock": "PT10M00.00S",
                "teamTricode": "AWAY",
                "personId": 2,
                "playerName": "Other",
                "actionType": "Missed Shot",
                "subType": "Jump Shot",
                "description": "MISS Other 2PT Jump Shot",
                "shotResult": "Missed",
                "shotValue": 2,
                "isFieldGoal": 1,
                "scoreHome": "2",
                "scoreAway": "0",
            },
            {
                "actionId": 3,
                "period": 1,
                "clock": "PT09M00.00S",
                "teamTricode": "HOME",
                "actionType": "Turnover",
                "scoreHome": "2",
                "scoreAway": "0",
            },
        ],
        clock_artifact=clock,
        game_id="game-1",
        home_team_id="HOME",
        away_team_id="AWAY",
        source_play_by_play_sha256="b" * 64,
    )


def test_label_hidden_queue_separates_official_truth() -> None:
    alignment = _alignment()
    queue = build_label_hidden_queue(
        alignment=alignment,
        game_slug="game-1",
        video_filename="video.mp4",
        video_sha256="a" * 64,
        fps=30.0,
        frame_count=5000,
        width=640,
        height=360,
    )
    assert queue["queue_count"] == 2
    assert verify_queue_artifact(queue)["artifact_sha256"] == queue["artifact_sha256"]
    assert all("shot_result" not in row for row in queue["queue"])
    truth = build_post_freeze_truth(alignment=alignment, queue=queue)
    assert truth["truth_count"] == 2
    assert truth["truth"][0]["shot_result"] == "Made"
    assert verify_truth_artifact(truth)["artifact_sha256"] == truth["artifact_sha256"]


def test_review_batch_is_label_free_and_deterministic() -> None:
    alignment = _alignment()
    queue = build_label_hidden_queue(
        alignment=alignment,
        game_slug="game-1",
        video_filename="video.mp4",
        video_sha256="a" * 64,
        fps=30.0,
        frame_count=5000,
        width=640,
        height=360,
    )
    batch = select_review_batch(queue, count=1)
    assert batch["batch_count"] == 1
    assert batch["batch"][0]["label_hidden"] is True
    assert verify_review_batch_artifact(batch)["artifact_sha256"] == batch["artifact_sha256"]
