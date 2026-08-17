from __future__ import annotations

from app.analysis.gcb_event_audit import (
    build_gcb_event_audit,
    parse_gcb_label,
    verify_gcb_event_audit,
)


def test_parse_gcb_label_normalizes_composite_event_labels() -> None:
    assert parse_gcb_label("Foul, Rebound, Shot") == [
        "foul",
        "rebound",
        "field_goal_attempt",
    ]


def test_gcb_audit_normalizes_period_timeout_and_technical_text() -> None:
    audit = build_gcb_event_audit(
        [
            {
                "file_name": "videos/000/000001.mp4",
                "game_state": {"label": "Timeout-Regular"},
                "background": "<match_name>A vs B</match_name> <date>2019-01-01</date>",
                "commentary": {
                    "source_commentary_text": (
                        "Start of 1st Period ; MAVERICKS Timeout: Regular ; "
                        "Double Technical - Ricky Rubio, Dennis Smith Jr."
                    )
                },
            }
        ],
        source_url="https://huggingface.co/datasets/A4Blind/GCB",
        source_revision="rev-1",
    )

    assert audit["source_event_counts"] == {
        "foul": 1,
        "period_boundary": 1,
        "timeout": 1,
    }
    assert audit["source_unknown_segment_count"] == 0


def test_gcb_audit_seals_source_text_outcomes_and_game_scope() -> None:
    audit = build_gcb_event_audit(
        [
            {
                "file_name": "videos/000/000001.mp4",
                "game_state": {"label": "Foul, Rebound, Shot"},
                "background": (
                    "<match_name>A vs B</match_name> <date>2019-01-01</date>"
                ),
                "commentary": {
                    "source_commentary_text": (
                        "MISS 3PT Jump Shot ; DEFENSIVE REBOUND ; S.FOUL"
                    )
                },
            }
        ],
        source_url="https://huggingface.co/datasets/A4Blind/GCB",
        source_revision="rev-1",
    )

    assert audit["video_count"] == 1
    assert audit["unique_game_count"] == 1
    assert audit["explicit_event_counts"] == {
        "field_goal_attempt": 1,
        "foul": 1,
        "rebound": 1,
    }
    assert audit["source_event_counts"] == {
        "field_goal_attempt": 1,
        "foul": 1,
        "rebound": 1,
    }
    assert audit["source_shot_result_counts"] == {"miss": 1}
    assert audit["source_unknown_segment_count"] == 0
    assert audit["runtime_consumable"] is False
    assert audit["training_media_eligible"] is False
    assert verify_gcb_event_audit(audit)["audit_sha256"] == audit["audit_sha256"]
