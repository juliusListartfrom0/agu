from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.analysis.action_ownership import ActionOwnerModel, seal_extra_trees_action_owner_model
from app.analysis.box_score import EventLedger
from app.analysis.official_evaluation import (
    RawOnlyEvaluationError,
    seal_raw_only_predictions,
    verify_agu_autonomous_bundle,
)
from app.analysis.official_inference import (
    OfficialVLMResult,
    OllamaOfficialEventReviewer,
    TwoPassOfficialEventReviewer,
    _constrain_rebound_actor_candidates,
    _decision_from_vlm_result,
    _event_with_related_context,
    _frame_index_to_sampled_frame,
    _recover_single_player_alias_from_reason,
    adjudicate_official_candidates,
    apply_action_owner_prior,
    seal_agu_autonomous_predictions,
)
from app.analysis.schemas import EventEvidenceResponse, GameEventResponse


class FakeReviewer:
    name = "fake_official_vlm"
    model = "fixture-model"

    def __init__(self, result: OfficialVLMResult) -> None:
        self.result = result

    def review(self, event: GameEventResponse, frames: object) -> OfficialVLMResult:
        return self.result


def test_displayed_frame_index_maps_to_exact_sampled_raw_frame() -> None:
    sampled = [468, 484, 500, 516, 532, 548, 564, 580, 596, 612, 628, 644]

    assert _frame_index_to_sampled_frame(6, sampled) == 548
    assert _frame_index_to_sampled_frame(12, sampled) == 644
    assert _frame_index_to_sampled_frame(0, sampled) is None


def test_official_vlm_cache_is_bound_to_rendered_images(tmp_path: Path) -> None:
    reviewer = OllamaOfficialEventReviewer(
        model="fixture-model",
        host="http://127.0.0.1:11434",
        timeout=1.0,
        image_width=64,
        max_frames=1,
        cache_path=tmp_path / "review-cache.json",
    )
    response = MagicMock()
    response.read.return_value = json.dumps(
        {"response": ('{"event_present":false,"confidence":0.95,"reason":"not present"}')}
    ).encode("utf-8")

    with (
        patch(
            "app.analysis.official_inference.encode_frames_jpeg",
            side_effect=lambda frames, max_width: [str(int(frames[0][0, 0, 0]))],
        ),
        patch("urllib.request.urlopen") as urlopen,
    ):
        urlopen.return_value.__enter__.return_value = response
        reviewer.review(_candidate(), [np.zeros((4, 4, 3), dtype=np.uint8)])
        reviewer.review(_candidate(), [np.ones((4, 4, 3), dtype=np.uint8)])
        reviewer.review(_candidate(), [np.zeros((4, 4, 3), dtype=np.uint8)])

    assert urlopen.call_count == 2


def test_rebound_review_inherits_only_post_release_parent_candidates() -> None:
    parent = _candidate(
        event_id="shot-parent",
        release_frame=110,
        outcome_frame=124,
        outcome="missed",
        evidence=[
            EventEvidenceResponse(
                evidence_id="shot-evidence",
                kind="traditional_cv",
                source_video_id="video_001",
                start_frame=100,
                end_frame=130,
                details={
                    "candidate_player_observations": [
                        {"player_id": "before", "team_id": "raw-dark", "frame": 108},
                        {"player_id": "rebounder", "team_id": "raw-light", "frame": 120},
                    ]
                },
            )
        ],
    )
    rebound = GameEventResponse(
        event_id="rebound-1",
        revision=1,
        event_type="rebound",
        source_video_id="video_001",
        start_frame=124,
        end_frame=160,
        status="needs_review",
        related_event_ids=["shot-parent"],
        evidence=[
            EventEvidenceResponse(
                evidence_id="rebound-evidence",
                kind="traditional_cv",
                source_video_id="video_001",
                start_frame=124,
                end_frame=160,
                details={"candidate_player_observations": []},
            )
        ],
    )

    enriched = _event_with_related_context(rebound, EventLedger([parent, rebound]))
    details = enriched.evidence[0].details

    assert details["related_shot_release_frame"] == 110
    assert details["related_shot_outcome"] == "missed"
    assert details["related_shot_team_id"] == "raw-dark"
    assert details["candidate_player_ids"] == ["rebounder"]
    assert [item["player_id"] for item in details["candidate_player_observations"]] == ["rebounder"]


def test_defensive_rebound_actor_candidates_are_constrained_to_opponent_team() -> None:
    event = GameEventResponse(
        event_id="rebound-1",
        revision=1,
        event_type="rebound",
        source_video_id="video_001",
        start_frame=120,
        end_frame=150,
        rebound_type="defensive",
        status="needs_review",
        evidence=[
            EventEvidenceResponse(
                evidence_id="rebound-evidence",
                kind="traditional_cv",
                source_video_id="video_001",
                start_frame=120,
                end_frame=150,
                details={
                    "related_shot_team_id": "raw-dark",
                    "candidate_player_ids": ["dark-player", "light-player"],
                    "candidate_team_ids": ["raw-dark", "raw-light"],
                    "candidate_player_observations": [
                        {"player_id": "dark-player", "team_id": "raw-dark", "frame": 126},
                        {"player_id": "light-player", "team_id": "raw-light", "frame": 126},
                    ],
                },
            )
        ],
    )

    constrained = _constrain_rebound_actor_candidates(event)
    details = constrained.evidence[0].details

    assert constrained.team_id == "raw-light"
    assert details["candidate_player_ids"] == ["light-player"]
    assert details["candidate_team_ids"] == ["raw-light"]
    assert [item["player_id"] for item in details["candidate_player_observations"]] == ["light-player"]


def test_rebound_decision_accepts_raw_parent_candidate_added_to_review_context() -> None:
    parent = _candidate(
        event_id="shot-parent",
        status="edge_vlm_confirmed",
        primary_player_id="dark-shooter",
        team_id="raw-dark",
        outcome="missed",
        shot_value=2,
        release_frame=110,
        evidence=[
            EventEvidenceResponse(
                evidence_id="shot-evidence",
                kind="traditional_cv",
                source_video_id="video_001",
                start_frame=100,
                end_frame=130,
                details={
                    "candidate_player_observations": [
                        {"player_id": "light-rebounder", "team_id": "raw-light", "frame": 120}
                    ]
                },
            )
        ],
    )
    rebound = GameEventResponse(
        event_id="rebound-1",
        revision=1,
        event_type="rebound",
        source_video_id="video_001",
        start_frame=124,
        end_frame=160,
        status="needs_review",
        related_event_ids=["shot-parent"],
        evidence=[
            EventEvidenceResponse(
                evidence_id="rebound-evidence",
                kind="traditional_cv",
                source_video_id="video_001",
                start_frame=124,
                end_frame=160,
                details={"candidate_player_observations": []},
            )
        ],
    )
    review_event = _event_with_related_context(rebound, EventLedger([parent, rebound]))

    decision = _decision_from_vlm_result(
        rebound,
        result=OfficialVLMResult(
            True,
            0.95,
            "visible stable control",
            {
                "primary_player_id": "light-rebounder",
                "team_id": "raw-light",
                "rebound_type": "defensive",
            },
        ),
        reviewer_name="fixture",
        minimum_confidence=0.8,
        validation_event=review_event,
        related_events={parent.event_id: parent},
    )

    assert decision.decision == "revise"
    assert decision.labels["primary_player_id"] == "light-rebounder"
    assert decision.labels["rebound_type"] == "defensive"


def test_two_pass_reviewer_merges_clean_semantics_with_overlaid_actor() -> None:
    semantic = FakeReviewer(OfficialVLMResult(True, 0.95, "made two", {"outcome": "made", "shot_value": 2}))
    actor = FakeReviewer(
        OfficialVLMResult(
            True,
            0.9,
            "P01 shoots",
            {"primary_player_id": "track-7", "team_id": "raw-dark"},
        )
    )
    reviewer = TwoPassOfficialEventReviewer(
        semantic_reviewer=semantic,
        actor_reviewer=actor,
        actor_frame_provider=lambda event: [np.ones((8, 8, 3), dtype=np.uint8)],
    )

    result = reviewer.review(_candidate(), [np.zeros((8, 8, 3), dtype=np.uint8)])

    assert result.event_present is True
    assert result.labels == {
        "outcome": "made",
        "shot_value": 2,
        "primary_player_id": "track-7",
        "team_id": "raw-dark",
    }


def test_two_pass_reviewer_rejects_referee_as_action_owner() -> None:
    semantic = FakeReviewer(OfficialVLMResult(True, 0.95, "missed two", {"outcome": "missed", "shot_value": 2}))
    actor = FakeReviewer(
        OfficialVLMResult(
            True,
            0.95,
            "P01 is an official, not a team-uniform player",
            {
                "primary_player_id": "track-referee",
                "team_id": "raw-dark",
                "primary_actor_role": "referee",
            },
        )
    )
    reviewer = TwoPassOfficialEventReviewer(
        semantic_reviewer=semantic,
        actor_reviewer=actor,
        actor_frame_provider=lambda event: [np.ones((8, 8, 3), dtype=np.uint8)],
    )

    result = reviewer.review(_candidate(), [np.zeros((8, 8, 3), dtype=np.uint8)])

    assert result.event_present is True
    assert result.labels == {"outcome": "missed", "shot_value": 2}
    assert "rejected by AGU role gate as referee" in result.reason


def test_action_owner_prior_is_hash_bound_and_does_not_prefill_actor() -> None:
    artifact = seal_extra_trees_action_owner_model(
        {
            "training_manifest_sha256": "training-sha",
            "trees": [
                {
                    "children_left": [-1],
                    "children_right": [-1],
                    "feature": [-2],
                    "threshold": [-2.0],
                    "positive_probability": [0.7],
                }
            ],
        }
    )
    event = _candidate(primary_player_id=None)
    event.evidence[0].details["candidate_player_observations"] = [
        {
            "player_id": "track-7",
            "frame": 110,
            "ball_player_distance": 0.2,
            "bbox": {"x1": 0, "y1": 0, "x2": 20, "y2": 40},
        },
        {
            "player_id": "track-9",
            "frame": 110,
            "ball_player_distance": 0.4,
            "bbox": {"x1": 20, "y1": 0, "x2": 40, "y2": 40},
        },
    ]

    enriched = apply_action_owner_prior(event, model=ActionOwnerModel(artifact), anchor_frame=110)

    details = enriched.evidence[0].details
    assert details["action_owner_model_sha256"] == artifact["model_sha256"]
    assert all("action_owner_probability" in item for item in details["candidate_player_observations"])
    assert enriched.primary_player_id is None


def _candidate(**updates: object) -> GameEventResponse:
    payload: dict[str, object] = {
        "event_id": "shot-1",
        "revision": 1,
        "event_type": "field_goal_attempt",
        "source_video_id": "video_001",
        "start_frame": 100,
        "end_frame": 130,
        "status": "needs_review",
        "team_id": "raw-dark",
        "primary_player_id": "track-7",
        "outcome": "unknown",
        "evidence": [
            EventEvidenceResponse(
                evidence_id="e1",
                kind="traditional_cv",
                source_video_id="video_001",
                start_frame=100,
                end_frame=130,
                details={
                    "candidate_team_ids": ["raw-dark", "raw-light"],
                    "candidate_player_ids": ["track-7", "track-9"],
                },
            )
        ],
    }
    payload.update(updates)
    return GameEventResponse.model_validate(payload)


def test_agu_vlm_confirms_only_complete_schema_bounded_labels() -> None:
    reviewer = FakeReviewer(
        OfficialVLMResult(
            event_present=True,
            confidence=0.94,
            reason="release and made basket visible",
            labels={
                "team_id": "raw-dark",
                "primary_player_id": "track-7",
                "outcome": "made",
                "shot_value": 3,
                "three_point_line_visible": True,
                "shooter_feet_visible": True,
                "release_beyond_arc": True,
            },
        )
    )

    events = adjudicate_official_candidates(
        [_candidate()],
        reviewer=reviewer,
        frame_provider=lambda event: [np.zeros((8, 8, 3), dtype=np.uint8)],
        minimum_confidence=0.8,
    )

    assert events[0].status == "edge_vlm_confirmed"
    assert events[0].reviewer == "edge_vlm:fake_official_vlm/fixture-model"
    assert (events[0].outcome, events[0].shot_value) == ("made", 3)


def test_agu_vlm_leaves_incomplete_event_unresolved() -> None:
    reviewer = FakeReviewer(OfficialVLMResult(True, 0.99, "actor unavailable", {"outcome": "made", "shot_value": 2}))
    event = _candidate(team_id=None, primary_player_id=None, evidence=[])

    result = adjudicate_official_candidates(
        [event], reviewer=reviewer, frame_provider=lambda current: [], minimum_confidence=0.8
    )[0]

    assert result.status == "needs_review"
    assert result.outcome == "unknown"


def test_single_overlaid_alias_in_vlm_reason_recovers_missing_schema_field() -> None:
    event = _candidate(primary_player_id=None)
    parsed = {"event_present": True, "reason": "P02 releases the visible shot"}

    _recover_single_player_alias_from_reason(event, parsed)

    assert parsed["primary_player_alias"] == "P02"


def test_multiple_aliases_in_vlm_reason_remain_unresolved() -> None:
    event = _candidate(primary_player_id=None)
    parsed = {"event_present": True, "reason": "P01 passes near P02"}

    _recover_single_player_alias_from_reason(event, parsed)

    assert "primary_player_alias" not in parsed


def test_invalid_alias_field_is_recovered_from_one_grounded_reason_alias() -> None:
    event = _candidate(primary_player_id=None)
    parsed = {
        "event_present": True,
        "primary_player_alias": "P02 raw-dark",
        "reason": "P02 releases the shot",
    }

    _recover_single_player_alias_from_reason(event, parsed)

    assert parsed["primary_player_alias"] == "P02"


def test_traditional_trajectory_outcome_cannot_be_overwritten_by_vlm() -> None:
    event = _candidate(outcome="missed", outcome_frame=124)
    reviewer = FakeReviewer(
        OfficialVLMResult(
            True,
            0.99,
            "semantic model mistakes rim contact for a make",
            {
                "primary_player_id": "track-7",
                "outcome": "made",
                "outcome_frame": 102,
                "shot_value": 2,
            },
        )
    )

    result = adjudicate_official_candidates(
        [event], reviewer=reviewer, frame_provider=lambda current: [], minimum_confidence=0.8
    )[0]

    assert result.status == "edge_vlm_confirmed"
    assert result.outcome == "missed"
    assert result.outcome_frame == 124


def test_three_point_label_requires_grounded_release_geometry() -> None:
    reviewer = FakeReviewer(
        OfficialVLMResult(
            True,
            0.99,
            "claims three without showing the arc",
            {"primary_player_id": "track-7", "outcome": "made", "shot_value": 3},
        )
    )

    result = adjudicate_official_candidates(
        [_candidate()], reviewer=reviewer, frame_provider=lambda current: [], minimum_confidence=0.8
    )[0]

    assert result.status == "needs_review"
    assert result.shot_value is None


def test_agu_vlm_cannot_invent_identity_outside_traditional_candidates() -> None:
    reviewer = FakeReviewer(
        OfficialVLMResult(
            True,
            0.99,
            "hallucinated identity",
            {"team_id": "truth-team", "primary_player_id": "truth-player", "outcome": "made", "shot_value": 2},
        )
    )

    result = adjudicate_official_candidates(
        [_candidate(team_id=None, primary_player_id=None)],
        reviewer=reviewer,
        frame_provider=lambda current: [],
        minimum_confidence=0.8,
    )[0]

    assert result.status == "needs_review"
    assert result.team_id is None
    assert result.primary_player_id is None


def test_vlm_team_is_bound_to_selected_traditional_player_observation() -> None:
    event = _candidate(
        evidence=[
            EventEvidenceResponse(
                evidence_id="e1",
                kind="traditional_cv",
                source_video_id="video_001",
                start_frame=100,
                end_frame=130,
                details={
                    "candidate_team_ids": ["raw-dark", "raw-light"],
                    "candidate_player_ids": ["track-7"],
                    "candidate_player_observations": [
                        {"player_id": "track-7", "team_id": "raw-dark", "frame": 110, "bbox": {}}
                    ],
                },
            )
        ]
    )
    reviewer = FakeReviewer(
        OfficialVLMResult(
            True,
            0.99,
            "mismatched team response",
            {"primary_player_id": "track-7", "team_id": "raw-light", "outcome": "made", "shot_value": 2},
        )
    )

    result = adjudicate_official_candidates(
        [event], reviewer=reviewer, frame_provider=lambda current: [], minimum_confidence=0.8
    )[0]

    assert result.status == "edge_vlm_confirmed"
    assert result.team_id == "raw-dark"


def test_vlm_short_player_alias_maps_to_traditional_identity() -> None:
    reviewer = FakeReviewer(
        OfficialVLMResult(
            True,
            0.99,
            "P01 releases the shot",
            {"primary_player_alias": "p01", "outcome": "made", "shot_value": 2},
        )
    )

    result = adjudicate_official_candidates(
        [_candidate()], reviewer=reviewer, frame_provider=lambda current: [], minimum_confidence=0.8
    )[0]

    assert result.status == "edge_vlm_confirmed"
    assert result.primary_player_id == "track-7"


def test_non_shot_event_drops_vlm_outcome_and_shot_value() -> None:
    event = _candidate(event_type="foul", outcome=None, team_id="raw-dark", primary_player_id="track-7")
    reviewer = FakeReviewer(OfficialVLMResult(True, 0.99, "foul", {"outcome": "missed", "shot_value": 3}))

    result = adjudicate_official_candidates(
        [event], reviewer=reviewer, frame_provider=lambda current: [], minimum_confidence=0.8
    )[0]

    assert result.outcome is None
    assert result.shot_value is None


def test_rebound_is_rejected_after_agu_confirmed_make() -> None:
    shot = _candidate(outcome="made", shot_value=2, status="vision_confirmed")
    rebound = _candidate(
        event_id="rebound-1",
        event_type="rebound",
        start_frame=131,
        end_frame=150,
        outcome=None,
        shot_value=None,
        rebound_type="defensive",
        related_event_ids=[shot.event_id],
    )
    reviewer = FakeReviewer(OfficialVLMResult(True, 0.99, "player secures ball", {}))

    result = adjudicate_official_candidates(
        [shot, rebound], reviewer=reviewer, frame_provider=lambda current: [], minimum_confidence=0.8
    )

    assert result[1].status == "rejected"
    assert "no rebound may follow" in result[1].reason


def test_rebound_fails_closed_until_linked_shot_is_confirmed_missed() -> None:
    shot = _candidate(outcome="unknown", status="needs_review")
    rebound = _candidate(
        event_id="rebound-1",
        event_type="rebound",
        start_frame=131,
        end_frame=150,
        outcome=None,
        shot_value=None,
        rebound_type="defensive",
        related_event_ids=[shot.event_id],
    )
    reviewer = FakeReviewer(OfficialVLMResult(True, 0.99, "player secures ball", {}))

    result = adjudicate_official_candidates(
        [shot, rebound], reviewer=reviewer, frame_provider=lambda current: [], minimum_confidence=0.8
    )

    assert result[1].status == "needs_review"
    assert "confirmed miss" in result[1].reason


def test_assist_requires_automatically_confirmed_made_shot() -> None:
    unresolved_shot = _candidate(outcome="unknown", status="needs_review")
    assist = _candidate(
        event_id="assist-1",
        event_type="assist",
        start_frame=80,
        end_frame=110,
        outcome=None,
        shot_value=None,
        related_event_ids=[unresolved_shot.event_id],
    )
    reviewer = FakeReviewer(OfficialVLMResult(True, 0.99, "pass leads to basket", {}))

    unresolved = adjudicate_official_candidates(
        [unresolved_shot, assist],
        reviewer=reviewer,
        frame_provider=lambda current: [],
        minimum_confidence=0.8,
    )
    unresolved_assist = next(item for item in unresolved if item.event_id == assist.event_id)
    assert unresolved_assist.status == "needs_review"
    assert "confirmed make" in unresolved_assist.reason

    made_shot = unresolved_shot.model_copy(update={"status": "needs_review", "outcome": "made", "shot_value": 2})
    confirmed = adjudicate_official_candidates(
        [made_shot, assist],
        reviewer=reviewer,
        frame_provider=lambda current: [],
        minimum_confidence=0.8,
    )
    confirmed_assist = next(item for item in confirmed if item.event_id == assist.event_id)
    assert confirmed_assist.status == "edge_vlm_confirmed"
    assert confirmed_assist.secondary_player_id == made_shot.primary_player_id

    opponent_assist = assist.model_copy(update={"team_id": "raw-light"})
    rejected = adjudicate_official_candidates(
        [made_shot, opponent_assist],
        reviewer=reviewer,
        frame_provider=lambda current: [],
        minimum_confidence=0.8,
    )
    rejected_assist = next(item for item in rejected if item.event_id == assist.event_id)
    assert rejected_assist.status == "rejected"
    assert "same team" in rejected_assist.reason


def test_rebound_type_must_match_confirmed_miss_team() -> None:
    shot = _candidate(outcome="missed", shot_value=2, status="vision_confirmed")
    rebound = _candidate(
        event_id="rebound-1",
        event_type="rebound",
        start_frame=131,
        end_frame=150,
        outcome=None,
        shot_value=None,
        team_id="raw-light",
        rebound_type="offensive",
        related_event_ids=[shot.event_id],
        evidence=[
            EventEvidenceResponse(
                evidence_id="rebound-evidence",
                kind="traditional_cv",
                source_video_id="video_001",
                start_frame=131,
                end_frame=150,
                details={
                    "candidate_team_ids": ["raw-light"],
                    "candidate_player_ids": ["track-7"],
                },
            )
        ],
    )
    reviewer = FakeReviewer(OfficialVLMResult(True, 0.99, "rebound visible", {}))

    result = adjudicate_official_candidates(
        [shot, rebound],
        reviewer=reviewer,
        frame_provider=lambda current: [],
        minimum_confidence=0.8,
    )
    resolved = next(item for item in result if item.event_id == rebound.event_id)
    assert resolved.status == "rejected"
    assert "defensive rebound" in resolved.reason


def test_autonomous_adjudication_refuses_codex_prefilled_event() -> None:
    event = _candidate(status="codex_confirmed", reviewer="codex:annotation")
    reviewer = FakeReviewer(OfficialVLMResult(True, 1.0, "", {}))

    with pytest.raises(ValueError, match="refuses pre-reviewed"):
        adjudicate_official_candidates(
            [event], reviewer=reviewer, frame_provider=lambda current: [], minimum_confidence=0.8
        )


def test_autonomous_bundle_gate_rejects_codex_and_accepts_agu(tmp_path: Path) -> None:
    raw = tmp_path / "period.mov"
    raw.write_bytes(b"raw")
    codex_event = _candidate(status="codex_confirmed", reviewer="codex:annotation")
    reviewed = seal_raw_only_predictions(
        game_id="g1",
        raw_video_paths=[raw],
        events=[codex_event],
        config={},
        model_provenance={"producer": "agu", "inference_mode": "traditional_cv+vlm"},
    )
    with pytest.raises(RawOnlyEvaluationError, match="reviewer-confirmed"):
        verify_agu_autonomous_bundle(reviewed)

    automatic = _candidate(
        revision=2,
        status="edge_vlm_confirmed",
        reviewer="edge_vlm:ollama/model",
        outcome="made",
        shot_value=2,
    )
    bundle = seal_agu_autonomous_predictions(
        game_id="g1",
        raw_video_paths=[raw],
        events=[automatic],
        config={},
        candidate_backend="traditional_cv",
        semantic_backend="ollama",
        semantic_model="qwen-vl",
    )
    assert verify_agu_autonomous_bundle(bundle).model_provenance["producer"] == "agu"


def test_autonomous_bundle_allows_manifest_bound_codex_training_annotations(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "period.mov"
    raw.write_bytes(b"raw")
    automatic = _candidate(
        revision=2,
        status="edge_vlm_confirmed",
        reviewer="edge_vlm:ollama/model",
        outcome="made",
        shot_value=2,
    )

    bundle = seal_agu_autonomous_predictions(
        game_id="g1",
        raw_video_paths=[raw],
        events=[automatic],
        config={},
        candidate_backend="traditional_cv",
        semantic_backend="ollama",
        semantic_model="qwen-vl",
        model_provenance={
            "training_annotation_producer": "codex",
            "training_manifest_sha256": "a" * 64,
            "training_benchmark_overlap": "false",
        },
    )

    assert bundle.model_provenance["training_annotation_producer"] == "codex"


@pytest.mark.parametrize(
    ("manifest_sha256", "overlap", "message"),
    [
        ("missing", "false", "sha256-bound"),
        ("a" * 64, "true", "training_benchmark_overlap=false"),
    ],
)
def test_autonomous_bundle_rejects_unsafe_codex_training_provenance(
    tmp_path: Path,
    manifest_sha256: str,
    overlap: str,
    message: str,
) -> None:
    raw = tmp_path / "period.mov"
    raw.write_bytes(b"raw")
    automatic = _candidate(
        revision=2,
        status="edge_vlm_confirmed",
        reviewer="edge_vlm:ollama/model",
        outcome="made",
        shot_value=2,
    )

    with pytest.raises(RawOnlyEvaluationError, match=message):
        seal_agu_autonomous_predictions(
            game_id="g1",
            raw_video_paths=[raw],
            events=[automatic],
            config={},
            candidate_backend="traditional_cv",
            semantic_backend="ollama",
            semantic_model="qwen-vl",
            model_provenance={
                "training_annotation_producer": "codex",
                "training_manifest_sha256": manifest_sha256,
                "training_benchmark_overlap": overlap,
            },
        )
