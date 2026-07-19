from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis.box_score import EventLedger, EventLedgerError
from app.analysis.official_evaluation import (
    RawOnlyEvaluationError,
    StrictEvent,
    evaluate_candidate_localization,
    evaluate_identity_aligned_strict_events,
    evaluate_official_tiers,
    evaluate_strict_events,
    load_strict_truth_csv,
    seal_raw_only_predictions,
    strict_events_from_game_events,
    verify_raw_only_bundle,
)
from app.analysis.review import apply_review_decisions, build_review_package, finalize_reviewed_bundle
from app.analysis.schemas import EventEvidenceResponse, GameEventResponse, ReviewDecisionResponse


def _event(event_id: str, event_type: str, **updates: object) -> GameEventResponse:
    payload: dict[str, object] = {
        "event_id": event_id,
        "revision": 1,
        "event_type": event_type,
        "source_video_id": "video_001",
        "start_frame": 100,
        "end_frame": 130,
        "team_id": "dark",
        "primary_player_id": "p1",
        "status": "human_confirmed",
        "confidence": 1.0,
    }
    payload.update(updates)
    return GameEventResponse.model_validate(payload)


def test_ledger_is_append_only_and_idempotent() -> None:
    event = _event("shot-1", "field_goal_attempt", shot_value=2, outcome="made")
    ledger = EventLedger([event])

    assert ledger.append(event) is False
    with pytest.raises(EventLedgerError, match="conflicting or stale"):
        ledger.append(event.model_copy(update={"outcome": "missed"}))

    revised = event.model_copy(
        update={"revision": 2, "supersedes_revision": 1, "status": "rejected", "reason": "duplicate"}
    )
    assert ledger.append(revised) is True
    assert ledger.latest("shot-1").revision == 2


def test_review_decision_adds_revision_without_mutating_prior_event() -> None:
    ledger = EventLedger([_event("shot-1", "field_goal_attempt", shot_value=3, outcome="unknown", status="candidate")])
    revised = ledger.apply_decision(
        ReviewDecisionResponse(
            decision_id="review-1",
            event_id="shot-1",
            expected_revision=1,
            decision="revise",
            reviewer_type="codex",
            reviewer="codex-session",
            input_sha256="abc",
            labels={"outcome": "made"},
            confidence=0.94,
            reason="visible downward rim crossing",
        )
    )

    assert revised.revision == 2
    assert revised.status == "codex_confirmed"
    assert revised.outcome == "made"
    assert ledger._revisions["shot-1"][0].outcome == "unknown"


def test_review_decision_can_add_event_found_during_fine_review() -> None:
    ledger = EventLedger([_event("shot-1", "field_goal_attempt", shot_value=2, outcome="missed")])
    added = ledger.apply_decision(
        ReviewDecisionResponse(
            decision_id="review-add-putback",
            event_id="shot-2",
            expected_revision=1,
            decision="add",
            reviewer_type="codex",
            reviewer="raw-fine",
            input_sha256="bundle-hash",
            labels={
                "event_type": "field_goal_attempt",
                "source_video_id": "video_001",
                "start_frame": 130,
                "end_frame": 145,
                "release_frame": 136,
                "outcome_frame": 142,
                "team_id": "dark",
                "primary_player_id": "p1",
                "shot_value": 2,
                "outcome": "made",
            },
            confidence=0.96,
            reason="Fine review found a putback after the coarse candidate ended.",
        )
    )

    assert added.revision == 1
    assert added.status == "codex_confirmed"
    assert added.outcome == "made"
    assert ledger.aggregate().players[0].two_pt_attempted == 2


def test_review_decision_rejects_add_for_existing_event_id() -> None:
    ledger = EventLedger([_event("shot-1", "field_goal_attempt", shot_value=2, outcome="missed")])
    decision = ReviewDecisionResponse(
        decision_id="review-duplicate-add",
        event_id="shot-1",
        expected_revision=1,
        decision="add",
        reviewer_type="codex",
        reviewer="raw-fine",
        input_sha256="bundle-hash",
        labels={
            "event_type": "field_goal_attempt",
            "source_video_id": "video_001",
            "start_frame": 130,
            "end_frame": 145,
        },
    )

    with pytest.raises(EventLedgerError, match="cannot add existing event"):
        ledger.apply_decision(decision)


def test_official_box_score_aggregates_only_accepted_related_events() -> None:
    shot = _event("shot-1", "field_goal_attempt", shot_value=3, outcome="made")
    assist = _event(
        "assist-1",
        "assist",
        primary_player_id="p2",
        related_event_ids=["shot-1"],
    )
    missed = _event(
        "shot-2",
        "field_goal_attempt",
        start_frame=200,
        end_frame=230,
        primary_player_id="p3",
        team_id="light",
        shot_value=2,
        outcome="missed",
    )
    rebound = _event(
        "rebound-1",
        "rebound",
        start_frame=231,
        end_frame=250,
        primary_player_id="p2",
        rebound_type="defensive",
        related_event_ids=["shot-2"],
    )
    candidate = _event(
        "steal-unknown",
        "steal",
        status="needs_review",
        primary_player_id="p2",
    )
    result = EventLedger([shot, assist, missed, rebound, candidate]).aggregate(expected_team_points={"dark": 3})

    p1 = next(player for player in result.players if player.player_id == "p1")
    p2 = next(player for player in result.players if player.player_id == "p2")
    assert (p1.points, p1.three_pt_made, p1.three_pt_attempted) == (3, 1, 1)
    assert (p2.assists, p2.defensive_rebounds, p2.steals) == (1, 1, 0)
    assert p2.unresolved_event_count == 1
    assert result.reconciliation.valid is True
    assert result.status == "needs_review"


def test_invalid_relationship_prevents_official_status() -> None:
    orphan = _event("assist-1", "assist", primary_player_id="p2")
    result = EventLedger([orphan]).aggregate()

    assert result.reconciliation.valid is False
    assert result.reconciliation.issues[0].code == "invalid_assist_relation"


def test_rebound_can_link_to_missed_free_throw() -> None:
    free_throw = _event(
        "free-throw-1",
        "free_throw_attempt",
        shot_value=1,
        outcome="missed",
    )
    rebound = _event(
        "rebound-1",
        "rebound",
        start_frame=131,
        end_frame=150,
        primary_player_id="p2",
        rebound_type="offensive",
        related_event_ids=["free-throw-1"],
    )

    result = EventLedger([free_throw, rebound]).aggregate()

    assert result.reconciliation.valid is True
    assert result.reconciliation.issues == []
    p2 = next(player for player in result.players if player.player_id == "p2")
    assert p2.offensive_rebounds == 1


def test_raw_only_bundle_rejects_reference_and_detects_tampering(tmp_path: Path) -> None:
    reference = tmp_path / "highlight_player.mov"
    reference.write_bytes(b"edited")
    with pytest.raises(RawOnlyEvaluationError, match="reference/edited-video"):
        seal_raw_only_predictions(game_id="g1", raw_video_paths=[reference], events=[], config={})

    raw = tmp_path / "period1.mov"
    raw.write_bytes(b"raw-video")
    bundle = seal_raw_only_predictions(game_id="g1", raw_video_paths=[raw], events=[], config={"preset": "official"})
    assert verify_raw_only_bundle(bundle).raw_videos[0].filename == "period1.mov"

    tampered = bundle.model_dump(mode="json")
    tampered["events_sha256"] = "0" * 64
    with pytest.raises(RawOnlyEvaluationError, match="bundle hash mismatch"):
        verify_raw_only_bundle(tampered)


def test_strict_evaluation_requires_actor_outcome_and_shot_value() -> None:
    truth = [
        StrictEvent("e1", "field_goal_attempt", "video_001", 100, "p1", team_id="dark", outcome="made", shot_value=3),
        StrictEvent("e2", "rebound", "video_001", 150, "p2", team_id="light"),
    ]
    predictions = [
        StrictEvent("p1", "field_goal_attempt", "video_001", 102, "wrong", team_id="dark", outcome="made", shot_value=3),
        StrictEvent("p2", "rebound", "video_001", 153, "p2", team_id="light"),
    ]
    result = evaluate_strict_events(truth, predictions, tolerance_frames=5)

    assert result["metrics"]["true_positive"] == 1
    assert result["metrics"]["false_positive"] == 1
    assert result["metrics"]["false_negative"] == 1
    assert result["metrics"]["f1"] == 0.5


def test_tiered_evaluation_never_promotes_candidate_localization_to_strict() -> None:
    truth = [
        StrictEvent(
            "e1",
            "field_goal_attempt",
            "video_001",
            100,
            "p1",
            team_id="dark",
            outcome="made",
            shot_value=3,
        )
    ]
    candidate = _event(
        "candidate-1",
        "field_goal_attempt",
        start_frame=98,
        end_frame=102,
        status="needs_review",
        primary_player_id=None,
        team_id=None,
        outcome="unknown",
        shot_value=None,
    )

    localization = evaluate_candidate_localization(
        truth,
        [StrictEvent("p1", "field_goal_attempt", "video_001", 100)],
        tolerance_frames=5,
    )
    tiers = evaluate_official_tiers(truth, [candidate], tolerance_frames=5)

    assert localization["metrics"]["f1"] == 1.0
    assert tiers["candidate_localization"]["metrics"]["f1"] == 1.0
    assert tiers["automatic_strict"]["metrics"]["f1"] == 0.0
    assert tiers["reviewed_strict"]["metrics"]["f1"] == 0.0


def test_one_to_one_matching_maximizes_events_instead_of_greedy_nearest() -> None:
    truth = [
        StrictEvent("early", "field_goal_attempt", "video_001", 100),
        StrictEvent("late", "field_goal_attempt", "video_001", 108),
    ]
    predictions = [
        StrictEvent("shared", "field_goal_attempt", "video_001", 105),
        StrictEvent("early-only", "field_goal_attempt", "video_001", 94),
    ]

    result = evaluate_candidate_localization(truth, predictions, tolerance_frames=6)

    assert result["metrics"]["true_positive"] == 2
    assert result["metrics"]["f1"] == 1.0
    assert {item["prediction_event_id"] for item in result["matches"]} == {"shared", "early-only"}


def test_strict_event_uses_outcome_frame_as_canonical_event_time() -> None:
    event = _event("steal", "steal", start_frame=100, end_frame=140, status="needs_review")
    event = event.model_copy(update={"outcome_frame": 135})

    strict = strict_events_from_game_events([event])

    assert strict[0].center_frame == 135


def test_candidate_event_uses_raw_evidence_anchor_instead_of_review_window_midpoint() -> None:
    event = _event(
        "shot",
        "field_goal_attempt",
        start_frame=80,
        end_frame=160,
        status="needs_review",
        evidence=[
            EventEvidenceResponse(
                evidence_id="anchor",
                kind="ball_rim_proximity_cluster",
                source_video_id="video_001",
                start_frame=118,
                end_frame=125,
                details={"candidate_event_frame": 120},
            )
        ],
    )

    strict = strict_events_from_game_events([event])

    assert strict[0].center_frame == 120


def test_identity_aligned_strict_evaluation_is_global_hashed_and_one_to_one() -> None:
    truth = [
        StrictEvent("e1", "rebound", "video_001", 100, "truth-p1", team_id="truth-team")
    ]
    predictions = [
        StrictEvent("p1", "rebound", "video_001", 102, "raw-light-34", team_id="raw-light")
    ]

    result = evaluate_identity_aligned_strict_events(
        truth,
        predictions,
        player_id_map={"raw-light-34": "truth-p1"},
        team_id_map={"raw-light": "truth-team"},
        tolerance_frames=5,
    )

    assert result["metrics"]["f1"] == 1.0
    assert len(result["identity_alignment"]["sha256"]) == 64
    assert result["identity_alignment"]["unmapped_prediction_player_ids"] == []
    with pytest.raises(RawOnlyEvaluationError, match="one-to-one"):
        evaluate_identity_aligned_strict_events(
            truth,
            predictions,
            player_id_map={"raw-a": "truth-p1", "raw-b": "truth-p1"},
            team_id_map={"raw-light": "truth-team"},
            tolerance_frames=5,
        )


def test_strict_truth_loader_normalizes_basketball_labels(tmp_path: Path) -> None:
    path = tmp_path / "truth.csv"
    path.write_text(
        "event_id,start_sec,end_sec,event_type,team_id,player_id,shot_result,shot_value\n"
        "e1,2,4,3分投篮,dark,p1,命中,3\n",
        encoding="utf-8",
    )

    events = load_strict_truth_csv(path, source_video_id="video_001", fps=30)

    assert events == [StrictEvent("e1", "field_goal_attempt", "video_001", 90, "p1", team_id="dark", outcome="made", shot_value=3)]


def test_review_package_uses_sealed_raw_video_and_decisions_append_revisions(tmp_path: Path) -> None:
    raw = tmp_path / "period1.mov"
    raw.write_bytes(b"raw-video")
    candidate = _event(
        "shot-1",
        "field_goal_attempt",
        shot_value=2,
        outcome="unknown",
        status="needs_review",
    )
    bundle = seal_raw_only_predictions(game_id="g1", raw_video_paths=[raw], events=[candidate], config={})
    package_dir = tmp_path / "review"

    manifest = build_review_package(
        bundle,
        raw_video_paths=[raw],
        output_dir=package_dir,
        materialize_media=False,
    )
    decision = ReviewDecisionResponse(
        decision_id="d1",
        event_id="shot-1",
        expected_revision=1,
        decision="revise",
        reviewer_type="human",
        reviewer="reviewer-1",
        input_sha256=bundle.bundle_sha256,
        labels={"outcome": "made"},
        confidence=1.0,
        reason="visible make",
    )
    ledger = apply_review_decisions([candidate], [decision, decision])

    assert manifest["candidate_count"] == 1
    assert (package_dir / "candidates.jsonl").is_file()
    assert ledger.latest("shot-1").revision == 2
    assert ledger.latest("shot-1").outcome == "made"

    with pytest.raises(ValueError, match="input hash mismatch"):
        apply_review_decisions(
            [candidate],
            [decision],
            expected_input_sha256="different-bundle",
        )


def test_ledger_exports_deterministic_revision_history() -> None:
    original = _event("shot-1", "field_goal_attempt", shot_value=2, outcome="unknown", status="candidate")
    ledger = EventLedger([original])
    ledger.apply_decision(
        ReviewDecisionResponse(
            decision_id="d1",
            event_id="shot-1",
            expected_revision=1,
            decision="revise",
            reviewer_type="codex",
            reviewer="reviewer",
            input_sha256="bundle",
            labels={"outcome": "missed"},
            confidence=0.9,
        )
    )

    assert [event.revision for event in ledger.all_revisions()] == [1, 2]


def test_finalize_reviewed_bundle_verifies_hash_and_preserves_revision_history(tmp_path: Path) -> None:
    raw = tmp_path / "period1.mov"
    raw.write_bytes(b"raw-video")
    candidate = _event(
        "shot-1",
        "field_goal_attempt",
        shot_value=2,
        outcome="unknown",
        status="needs_review",
    )
    bundle = seal_raw_only_predictions(game_id="g1", raw_video_paths=[raw], events=[candidate], config={})
    decision = ReviewDecisionResponse(
        decision_id="d1",
        event_id="shot-1",
        expected_revision=1,
        decision="revise",
        reviewer_type="codex",
        reviewer="codex-test",
        input_sha256=bundle.bundle_sha256,
        labels={"outcome": "made"},
        confidence=0.95,
    )

    reviewed, ledger, score = finalize_reviewed_bundle(
        bundle,
        decisions=[decision],
        raw_video_paths=[raw],
        config={"pipeline": "review"},
        expected_team_points={"dark": 2},
    )

    assert reviewed.model_provenance["reviewed_from_bundle_sha256"] == bundle.bundle_sha256
    assert reviewed.events[0].revision == 2
    assert len(ledger.all_revisions()) == 2
    assert score.players[0].points == 2


def test_finalize_reviewed_bundle_preserves_incomplete_dense_coverage_gate(tmp_path: Path) -> None:
    raw = tmp_path / "period1.mov"
    raw.write_bytes(b"raw-video")
    candidate = _event(
        "shot-1",
        "field_goal_attempt",
        shot_value=2,
        outcome="unknown",
        status="needs_review",
    )
    bundle = seal_raw_only_predictions(
        game_id="g1",
        raw_video_paths=[raw],
        events=[candidate],
        config={},
        model_provenance={
            "review_complete_video_coverage": "false",
            "reviewed_window_count": "1",
            "total_window_count": "2",
        },
    )
    decision = ReviewDecisionResponse(
        decision_id="d1",
        event_id="shot-1",
        expected_revision=1,
        decision="revise",
        reviewer_type="codex",
        reviewer="codex-test",
        input_sha256=bundle.bundle_sha256,
        labels={"outcome": "made"},
        confidence=0.95,
    )

    reviewed, _, score = finalize_reviewed_bundle(
        bundle,
        decisions=[decision],
        raw_video_paths=[raw],
        config={"pipeline": "review"},
        expected_team_points={"dark": 2},
    )

    assert reviewed.model_provenance["review_complete_video_coverage"] == "false"
    assert score.status == "needs_review"
    assert score.reconciliation.valid is False
    assert score.reconciliation.issues[-1].code == "incomplete_video_coverage"


def test_review_package_rejects_invalid_sheet_sampling(tmp_path: Path) -> None:
    raw = tmp_path / "period1.mov"
    raw.write_bytes(b"raw-video")
    bundle = seal_raw_only_predictions(game_id="g1", raw_video_paths=[raw], events=[], config={})

    with pytest.raises(ValueError, match="sampling settings"):
        build_review_package(
            bundle,
            raw_video_paths=[raw],
            output_dir=tmp_path / "review",
            materialize_media=False,
            review_sample_fps=0,
        )
