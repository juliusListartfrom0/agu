from __future__ import annotations

from app.analysis.nsva_event_text import (
    build_nsva_event_text_audit,
    parse_nsva_intent,
    verify_nsva_event_text_audit,
)


def test_parse_nsva_intent_preserves_shot_outcome_and_rebound_chain() -> None:
    parsed = parse_nsva_intent(
        "MISS 3' Layup , OFFENSIVE REBOUND , MISS 3' Turnaround Hook Shot , DEFENSIVE REBOUND"
    )

    assert [event["kind"] for event in parsed["events"]] == [
        "field_goal_attempt",
        "rebound",
        "field_goal_attempt",
        "rebound",
    ]
    assert [event.get("result") for event in parsed["events"]] == [
        "miss",
        None,
        "miss",
        None,
    ]
    assert parsed["events"][1]["rebound_type"] == "offensive"
    assert parsed["events"][3]["rebound_type"] == "defensive"
    assert parsed["contains_shot"] is True
    assert parsed["contains_rebound"] is True


def test_parse_nsva_intent_marks_assisted_made_shot() -> None:
    parsed = parse_nsva_intent("25' 3PT Jump Shot ( AST)")

    assert parsed["events"] == [
        {
            "kind": "field_goal_attempt",
            "result": "make",
            "shot_type": "3pt",
            "assisted": True,
            "text": "25' 3PT Jump Shot ( AST)",
        }
    ]
    assert parsed["contains_assist"] is True


def test_parse_nsva_intent_covers_non_shot_stat_events() -> None:
    parsed = parse_nsva_intent("Bad Pass Turnover , S.FOUL , STEAL")

    assert [event["kind"] for event in parsed["events"]] == [
        "turnover",
        "foul",
        "steal",
    ]
    assert parsed["contains_turnover"] is True
    assert parsed["contains_foul"] is True
    assert parsed["contains_steal"] is True


def test_parse_nsva_intent_covers_jump_ball_violation_and_ejection() -> None:
    parsed = parse_nsva_intent(
        "Jump vs. Tip to , Violation:Defensive Goaltending , Simmons Ejection:Other"
    )

    assert [event["kind"] for event in parsed["events"]] == [
        "jump_ball",
        "violation",
        "ejection",
    ]
    assert parsed["contains_unknown"] is False


def test_nsva_event_text_audit_is_sorted_and_self_verifying() -> None:
    audit = build_nsva_event_text_audit(
        [
            {"video_path": "b.mp4", "intent": "DEFENSIVE REBOUND"},
            {"video_path": "a.mp4", "intent": "MISS 2PT Jump Shot"},
        ],
        source_url="https://huggingface.co/datasets/sportsvision/nsva_subset",
        source_revision="rev-1",
    )

    assert audit["schema_version"] == "agu.nsva-event-text-audit.v1"
    assert audit["video_count"] == 2
    assert audit["event_counts"] == {"field_goal_attempt": 1, "rebound": 1}
    assert [row["video_path"] for row in audit["rows"]] == ["a.mp4", "b.mp4"]
    assert audit["runtime_consumable"] is False
    assert audit["training_media_eligible"] is False
    assert verify_nsva_event_text_audit(audit)["audit_sha256"] == audit["audit_sha256"]
