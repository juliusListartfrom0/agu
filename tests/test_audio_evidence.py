from __future__ import annotations

from app.analysis.audio_evidence import (
    AudioRosterPlayer,
    attach_audio_mentions,
    build_audio_evidence,
    derive_audio_evidence_subset,
    ranked_score_window_player_candidates,
    seal_audio_action_review,
    seal_audio_roster,
    seal_audio_roster_from_registration,
    seal_transcript_artifact,
    speech_candidate_events,
    verify_audio_action_review,
    verify_audio_evidence,
    verify_audio_roster,
)
from app.analysis.schemas import GameEventResponse


def _transcript() -> dict[str, object]:
    return seal_transcript_artifact({
        "schema_version": "agu.raw-audio-transcript.v1",
        "video_sha256": "video-sha",
        "roster_source_sha256": "test-roster-source",
        "model": "test-whisper",
        "transcript": {
            "segments": [
                {
                    "start": 1.0,
                    "end": 2.0,
                    "text": "Bryant, the rebound.",
                    "words": [{"probability": 0.9}, {"probability": 0.8}],
                },
                {
                    "start": 3.0,
                    "end": 4.0,
                    "text": "Bryant has 13 rebounds tonight.",
                    "words": [{"probability": 0.7}],
                },
                {"start": 5.0, "end": 6.0, "text": "Blocked by Garnett."},
                {"start": 7.0, "end": 8.0, "text": "They post him on the block."},
                {"start": 9.0, "end": 10.0, "text": "Bryant is whistled for the foul."},
                {"start": 11.0, "end": 12.0, "text": "Bryant has four fouls tonight."},
            ]
        },
    })


def test_audio_mentions_are_candidates_and_reject_statistical_context() -> None:
    roster = seal_audio_roster(
        [
            AudioRosterPlayer(person_id="977", team_id="LAL", display_name="Kobe Bryant"),
            AudioRosterPlayer(person_id="708", team_id="BOS", display_name="Kevin Garnett"),
        ],
        source_roster_sha256="test-roster-source",
    )

    artifact = build_audio_evidence(
        transcript_artifact=_transcript(),
        roster=roster,
        raw_video_sha256="video-sha",
        fps=30.0,
    )

    rebound_mentions = [mention for mention in artifact.mentions if mention.action == "rebound"]
    assert [mention.disposition for mention in rebound_mentions] == [
        "live_action_candidate",
        "rejected_context",
    ]
    assert rebound_mentions[0].speech_player_candidate_ids == ["977"]
    assert [mention.speech_player_candidate_ids for mention in artifact.player_mentions] == [
        ["977"],
        ["977"],
        ["708"],
        ["977"],
        ["977"],
    ]
    block_mentions = [mention for mention in artifact.mentions if mention.action == "block"]
    assert [mention.disposition for mention in block_mentions] == [
        "live_action_candidate",
        "rejected_context",
    ]
    foul_mentions = [mention for mention in artifact.mentions if mention.action == "foul"]
    assert [mention.disposition for mention in foul_mentions] == [
        "live_action_candidate",
        "rejected_context",
    ]

    events = speech_candidate_events(artifact, source_video_id="video_001")
    assert len(events) == 3
    assert [event.event_type for event in events] == ["rebound", "block", "foul"]
    missing_type_events = speech_candidate_events(
        artifact,
        source_video_id="video_001",
        include_action_types={"assist", "block", "foul", "steal", "turnover"},
    )
    assert [event.event_type for event in missing_type_events] == ["block", "foul"]
    assert all(event.status == "needs_review" for event in events)
    assert all(event.primary_player_id is None for event in events)
    assert events[0].evidence[0].details["requires_visual_confirmation"] is True
    try:
        verify_audio_evidence(artifact.model_copy(update={"artifact_sha256": "wrong"}))
    except ValueError as exc:
        assert "evidence artifact hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered audio evidence must fail")


def test_audio_roster_hash_and_video_binding_are_enforced() -> None:
    roster = seal_audio_roster(
        [AudioRosterPlayer(person_id="977", team_id="LAL", display_name="Kobe Bryant")],
        source_roster_sha256="test-roster-source",
    )
    broken = roster.model_copy(update={"artifact_sha256": "wrong"})
    try:
        build_audio_evidence(
            transcript_artifact=_transcript(),
            roster=broken,
            raw_video_sha256="video-sha",
            fps=30.0,
        )
    except ValueError as exc:
        assert "roster artifact hash mismatch" in str(exc)
    else:
        raise AssertionError("broken roster hash must fail")

    try:
        build_audio_evidence(
            transcript_artifact=_transcript(),
            roster=roster,
            raw_video_sha256="different-video",
            fps=30.0,
        )
    except ValueError as exc:
        assert "not bound" in str(exc)
    else:
        raise AssertionError("transcript/video mismatch must fail")

    tampered = _transcript()
    tampered["transcript"]["segments"][0]["text"] = "Bryant made every shot."
    try:
        build_audio_evidence(
            transcript_artifact=tampered,
            roster=roster,
            raw_video_sha256="video-sha",
            fps=30.0,
        )
    except ValueError as exc:
        assert "transcript artifact hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered transcript must fail")


def test_registration_roster_can_be_sealed_for_audio_without_event_answers() -> None:
    roster = seal_audio_roster_from_registration(
        {
            "schema_version": "agu.face-roster.v1",
            "benchmark_answers_included": False,
            "players": [
                {
                    "person_id": "977",
                    "team_id": "LAL",
                    "display_name": "Kobe Bryant",
                    "jersey_number": "24",
                }
            ],
        },
        source_roster_sha256="source-file-sha",
    )

    assert roster.source_roster_sha256 == "source-file-sha"
    assert roster.players[0].model_dump() == {
        "person_id": "977",
        "team_id": "LAL",
        "display_name": "Kobe Bryant",
        "aliases": [],
    }
    verify_audio_roster(roster)


def test_registration_roster_rejects_answer_assets() -> None:
    try:
        seal_audio_roster_from_registration(
            {
                "schema_version": "agu.face-roster.v1",
                "benchmark_answers_included": True,
                "players": [],
            },
            source_roster_sha256="source-file-sha",
        )
    except ValueError as exc:
        assert "answer" in str(exc)
    else:
        raise AssertionError("answer-bearing roster must fail")


def test_foul_parser_rejects_non_event_commentary_but_keeps_live_calls() -> None:
    roster = seal_audio_roster(
        [AudioRosterPlayer(person_id="977", team_id="LAL", display_name="Kobe Bryant")],
        source_roster_sha256="test-roster-source",
    )
    transcript = seal_transcript_artifact(
        {
            "schema_version": "agu.raw-audio-transcript.v1",
            "video_sha256": "video-sha",
            "roster_source_sha256": "test-roster-source",
            "model": "test-whisper",
            "transcript": {
                "segments": [
                    {"start": 1, "end": 2, "text": "He is in foul trouble."},
                    {"start": 3, "end": 4, "text": "There was no foul called."},
                    {"start": 5, "end": 6, "text": "He steps to the foul line."},
                    {"start": 7, "end": 8, "text": "The next foul puts them in the penalty."},
                    {"start": 9, "end": 10, "text": "The foul a moment ago was on Bryant."},
                    {"start": 11, "end": 12, "text": "The foul is on Bryant."},
                    {"start": 13, "end": 14, "text": "Draws contact and the foul."},
                ]
            },
        }
    )

    artifact = build_audio_evidence(
        transcript_artifact=transcript,
        roster=roster,
        raw_video_sha256="video-sha",
        fps=30,
    )

    foul_mentions = [mention for mention in artifact.mentions if mention.action == "foul"]
    assert [mention.disposition for mention in foul_mentions] == [
        "rejected_context",
        "rejected_context",
        "rejected_context",
        "rejected_context",
        "rejected_context",
        "live_action_candidate",
        "live_action_candidate",
    ]

    live_fouls = [
        mention for mention in foul_mentions if mention.disposition == "live_action_candidate"
    ]
    review = seal_audio_action_review(
        artifact,
        action="foul",
        decisions=[
            {
                "mention_id": mention.mention_id,
                "label": "live_current_event",
                "note": "fixture live call",
            }
            for mention in live_fouls
        ],
        producer="codex_offline_review",
    )
    assert review["runtime_consumable"] is False
    assert review["codex_runtime_answer_used"] is False
    assert verify_audio_action_review(review)["artifact_sha256"] == review["artifact_sha256"]


def test_field_goal_parser_keeps_live_result_inflections_and_rejects_non_events() -> None:
    roster = seal_audio_roster(
        [
            AudioRosterPlayer(person_id="893", team_id="CHI", display_name="Michael Jordan"),
            AudioRosterPlayer(person_id="937", team_id="CHI", display_name="Scottie Pippen"),
            AudioRosterPlayer(person_id="252", team_id="UTA", display_name="Karl Malone"),
        ],
        source_roster_sha256="test-roster-source",
    )
    transcript = seal_transcript_artifact(
        {
            "schema_version": "agu.raw-audio-transcript.v1",
            "video_sha256": "video-sha",
            "roster_source_sha256": "test-roster-source",
            "model": "test-whisper",
            "transcript": {
                "segments": [
                    {"start": 1, "end": 2, "text": "Jordan hit the open jumper."},
                    {"start": 3, "end": 4, "text": "Jordan missed it off the glass."},
                    {"start": 5, "end": 6, "text": "Pippen connects on the turnaround."},
                    {"start": 7, "end": 8, "text": "Malone buries the shot."},
                    {"start": 9, "end": 10, "text": "Jordan is off target."},
                    {"start": 11, "end": 12, "text": "Malone missed the first free throw."},
                    {"start": 13, "end": 14, "text": "They missed that call."},
                    {"start": 15, "end": 16, "text": "He had missed two jump shots earlier."},
                    {"start": 17, "end": 18, "text": "Hornacek hits the technical."},
                    {"start": 19, "end": 20, "text": "Russell missed the first."},
                    {"start": 21, "end": 22, "text": "But connects on the second."},
                    {"start": 23, "end": 24, "text": "He gives you that hits move."},
                    {"start": 25, "end": 26, "text": "He hasn't been able to hit them."},
                    {"start": 27, "end": 28, "text": "They cannot win unless Hornacek hits."},
                ]
            },
        }
    )

    artifact = build_audio_evidence(
        transcript_artifact=transcript,
        roster=roster,
        raw_video_sha256="video-sha",
        fps=30,
    )

    mentions = [
        mention for mention in artifact.mentions if mention.action == "field_goal_attempt"
    ]
    assert [mention.disposition for mention in mentions] == [
        "live_action_candidate",
        "live_action_candidate",
        "live_action_candidate",
        "live_action_candidate",
        "live_action_candidate",
        "rejected_context",
        "rejected_context",
        "rejected_context",
        "rejected_context",
        "rejected_context",
        "rejected_context",
        "rejected_context",
        "rejected_context",
        "rejected_context",
    ]
    assert [mention.shot_outcome_candidate for mention in mentions] == [
        "made",
        "missed",
        "made",
        "made",
        "missed",
        "missed",
        "missed",
        "missed",
        "made",
        "missed",
        "made",
        "made",
        "made",
        "made",
    ]
    events = speech_candidate_events(
        artifact,
        source_video_id="video_001",
        include_action_types={"field_goal_attempt"},
    )
    assert [
        event.evidence[0].details["speech_shot_outcome_candidate"] for event in events
    ] == ["made", "missed", "made", "made", "missed"]
    visual = GameEventResponse(
        event_id="visual-shot",
        revision=1,
        event_type="field_goal_attempt",
        source_video_id="video_001",
        start_frame=30,
        end_frame=60,
        outcome="unknown",
        status="needs_review",
    )
    attached_events, _ = attach_audio_mentions([visual], artifact, padding_sec=0)
    assert (
        attached_events[0].evidence[0].details["speech_shot_outcome_candidate"]
        == "made"
    )


def test_audio_action_review_requires_every_live_candidate_once() -> None:
    roster = seal_audio_roster(
        [AudioRosterPlayer(person_id="977", team_id="LAL", display_name="Kobe Bryant")],
        source_roster_sha256="test-roster-source",
    )
    artifact = build_audio_evidence(
        transcript_artifact=_transcript(),
        roster=roster,
        raw_video_sha256="video-sha",
        fps=30,
    )

    try:
        seal_audio_action_review(
            artifact,
            action="foul",
            decisions=[],
            producer="codex_offline_review",
        )
    except ValueError as exc:
        assert "exactly cover" in str(exc)
    else:
        raise AssertionError("partial review must fail")


def test_audio_mentions_attach_only_by_action_and_time_without_confirming_actor() -> None:
    roster = seal_audio_roster(
        [AudioRosterPlayer(person_id="977", team_id="LAL", display_name="Kobe Bryant")],
        source_roster_sha256="test-roster-source",
    )
    artifact = build_audio_evidence(
        transcript_artifact=_transcript(),
        roster=roster,
        raw_video_sha256="video-sha",
        fps=30.0,
    )
    event = GameEventResponse(
        event_id="visual-rebound",
        revision=1,
        event_type="rebound",
        source_video_id="video_001",
        start_frame=30,
        end_frame=60,
        status="needs_review",
    )

    updated, attached = attach_audio_mentions([event], artifact, padding_sec=0.0)

    assert len(attached) == 1
    assert updated[0].primary_player_id is None
    assert updated[0].status == "needs_review"
    assert updated[0].evidence[0].details["speech_player_candidate_ids"] == ["977"]

    subset = derive_audio_evidence_subset(artifact, excluded_mention_ids=attached)
    verify_audio_evidence(subset)
    assert not attached.intersection(mention.mention_id for mention in subset.mentions)

    ranked = ranked_score_window_player_candidates(
        artifact,
        roster,
        team_id="LAL",
        before_frame=0,
        after_frame=120,
        expected_scoreboard_lag_sec=2.0,
    )
    assert ranked == ["977"]


def test_audio_mention_attaches_to_only_the_best_overlapping_candidate() -> None:
    roster = seal_audio_roster(
        [AudioRosterPlayer(person_id="977", team_id="LAL", display_name="Kobe Bryant")],
        source_roster_sha256="test-roster-source",
    )
    artifact = build_audio_evidence(
        transcript_artifact=_transcript(),
        roster=roster,
        raw_video_sha256="video-sha",
        fps=30.0,
    )
    broad = GameEventResponse(
        event_id="broad-rebound",
        revision=1,
        event_type="rebound",
        source_video_id="video_001",
        start_frame=0,
        end_frame=90,
        status="needs_review",
        confidence=0.9,
    )
    exact = GameEventResponse(
        event_id="exact-rebound",
        revision=1,
        event_type="rebound",
        source_video_id="video_001",
        start_frame=30,
        end_frame=60,
        status="needs_review",
        confidence=0.5,
    )

    updated, attached = attach_audio_mentions([broad, exact], artifact, padding_sec=0.0)

    assert len(attached) == 1
    assert not updated[0].evidence
    assert [item.kind for item in updated[1].evidence] == ["raw_audio_action_candidate"]
