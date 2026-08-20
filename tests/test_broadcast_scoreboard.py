from __future__ import annotations

import numpy as np
import pytest

from app.analysis.broadcast_scoreboard import (
    BroadcastScoreboardRead,
    BroadcastScoreboardReader,
    CandidateScoreDelta,
    ScoreboardDelta,
    attach_score_delta,
    candidate_component_score_delta,
    candidate_ids_for_delta,
    candidate_score_delta,
    extract_scoreboard_deltas,
    score_delta_event,
    stable_score_read,
)
from app.analysis.schemas import GameEventResponse


class FakeOCR:
    def __init__(self, rows: list[tuple[str, float, float]]) -> None:
        self.rows = rows

    def __call__(self, image: np.ndarray) -> tuple[list[list[object]], None]:
        return [
            [
                [[x, 20], [x + 20, 20], [x + 20, 35], [x, 35]],
                text,
                confidence,
            ]
            for text, confidence, x in self.rows
        ], None


def test_reader_parses_separate_team_and_score_tokens() -> None:
    reader = BroadcastScoreboardReader(
        ["ATL", "CHI"],
        ocr=FakeOCR(
            [
                ("ATL", 0.99, 100),
                ("35", 0.98, 140),
                ("CHI", 0.97, 180),
                ("43", 0.99, 220),
                ("1ST10:00", 0.98, 260),
            ]
        ),
    )

    result = reader.read(np.zeros((100, 320, 3), dtype=np.uint8), frame=120)

    assert result is not None
    assert result.scores == {"ATL": 35, "CHI": 43}
    assert result.confidence == 0.97


def test_reader_parses_ocr_tokens_merged_across_score_and_team() -> None:
    reader = BroadcastScoreboardReader(
        ["ATL", "CHI"],
        ocr=FakeOCR(
            [
                ("ATL", 0.99, 100),
                ("35CHI", 0.93, 150),
                ("42", 0.99, 220),
                ("2ND3:26", 0.98, 260),
            ]
        ),
    )

    result = reader.read(np.zeros((100, 320, 3), dtype=np.uint8), frame=120)

    assert result is not None
    assert result.scores == {"ATL": 35, "CHI": 42}


def test_reader_parses_network_prefix_and_score_suffix() -> None:
    reader = BroadcastScoreboardReader(
        ["LAL", "BOS"],
        ocr=FakeOCR(
            [
                ("ESPTLAL19", 0.90, 100),
                ("BOS", 0.99, 180),
                ("21", 0.99, 220),
                ("1ST1:22", 0.98, 260),
            ]
        ),
    )

    result = reader.read(np.zeros((100, 320, 3), dtype=np.uint8), frame=120)

    assert result is not None
    assert result.scores == {"LAL": 19, "BOS": 21}


def test_reader_rejects_team_numbers_without_live_period_marker() -> None:
    reader = BroadcastScoreboardReader(
        ["LAL", "BOS"],
        ocr=FakeOCR(
            [
                ("2008 NBA FINALS", 0.99, 40),
                ("LAL", 0.99, 100),
                ("0", 0.99, 140),
                ("BOS", 0.99, 180),
                ("19", 0.99, 220),
            ]
        ),
    )

    assert reader.read(np.zeros((100, 320, 3), dtype=np.uint8), frame=120) is None


def test_reader_rejects_distant_shot_clock_when_team_score_is_missing() -> None:
    reader = BroadcastScoreboardReader(
        ["LAL", "BOS"],
        ocr=FakeOCR(
            [
                ("LAL", 0.99, 100),
                ("0", 0.99, 140),
                ("BOS", 0.99, 180),
                ("1ST11:53", 0.98, 300),
                ("19", 0.99, 380),
            ]
        ),
    )

    assert reader.read(np.zeros((100, 440, 3), dtype=np.uint8), frame=120) is None


def test_stable_score_read_rejects_tied_states() -> None:
    reads = [
        _read(1, ATL=10, CHI=12),
        _read(2, ATL=10, CHI=12),
        _read(3, ATL=12, CHI=12),
        _read(4, ATL=12, CHI=12),
    ]

    assert stable_score_read(reads) is None


def test_candidate_score_delta_requires_one_stable_plausible_increase() -> None:
    delta = candidate_score_delta(
        event_id="shot-1",
        before_reads=[_read(10, ATL=35, CHI=42), _read(11, ATL=35, CHI=42)],
        after_reads=[_read(20, ATL=35, CHI=43), _read(21, ATL=35, CHI=43)],
    )

    assert delta is not None
    assert (delta.team_id, delta.points) == ("CHI", 1)


def test_candidate_score_delta_rejects_no_change_and_multi_team_change() -> None:
    before = [_read(10, ATL=35, CHI=42), _read(11, ATL=35, CHI=42)]

    assert candidate_score_delta(event_id="shot-1", before_reads=before, after_reads=before) is None
    assert (
        candidate_score_delta(
            event_id="shot-1",
            before_reads=before,
            after_reads=[_read(20, ATL=37, CHI=43), _read(21, ATL=37, CHI=43)],
        )
        is None
    )


def test_component_score_delta_tolerates_unresolved_opponent_digits() -> None:
    delta = candidate_component_score_delta(
        event_id="shot-1",
        before_reads=[
            _read(10, LAL=83, BOS=97),
            _read(11, LAL=83, BOS=66),
            _read(12, LAL=83, BOS=97),
        ],
        after_reads=[
            _read(20, LAL=86, BOS=99),
            _read(21, LAL=86, BOS=66),
            _read(22, LAL=86, BOS=66),
            _read(23, LAL=86, BOS=99),
        ],
    )

    assert delta is not None
    assert (delta.team_id, delta.points) == ("LAL", 3)
    assert delta.unresolved_team_ids == ("BOS",)
    assert delta.score_resolution_method == "independent_team_consensus_v2"


def test_component_score_delta_fails_closed_on_resolved_opponent_change() -> None:
    delta = candidate_component_score_delta(
        event_id="shot-1",
        before_reads=[
            _read(10, LAL=83, BOS=97),
            _read(11, LAL=83, BOS=97),
        ],
        after_reads=[
            _read(20, LAL=86, BOS=99),
            _read(21, LAL=86, BOS=99),
        ],
    )

    assert delta is None


def test_component_score_delta_ignores_implausible_opponent_jump() -> None:
    delta = candidate_component_score_delta(
        event_id="shot-1",
        before_reads=[
            _read(10, LAL=83, BOS=66),
            _read(11, LAL=83, BOS=66),
        ],
        after_reads=[
            _read(20, LAL=86, BOS=99),
            _read(21, LAL=86, BOS=99),
        ],
    )

    assert delta is not None
    assert (delta.team_id, delta.points) == ("LAL", 3)
    assert delta.unresolved_team_ids == ("BOS",)


def test_component_score_delta_fails_closed_without_a_positive_change() -> None:
    reads = [
        _read(10, LAL=83, BOS=97),
        _read(11, LAL=83, BOS=97),
    ]

    assert (
        candidate_component_score_delta(
            event_id="shot-1",
            before_reads=reads,
            after_reads=reads,
        )
        is None
    )


def test_attach_score_delta_retains_unresolved_actor_and_classifies_free_throw() -> None:
    event = _event().model_copy(update={"team_id": "raw-dark"})
    delta = CandidateScoreDelta(
        event_id=event.event_id,
        team_id="CHI",
        points=1,
        before_frame=100,
        after_frame=200,
        before_scores={"ATL": 35, "CHI": 42},
        after_scores={"ATL": 35, "CHI": 43},
        confidence=0.96,
    )

    result = attach_score_delta(event, delta)

    assert result.status == "needs_review"
    assert result.event_type == "free_throw_attempt"
    assert (result.outcome, result.shot_value, result.team_id) == ("made", 1, "CHI")
    assert result.primary_player_id is None
    assert result.evidence[-1].kind == "scoreboard_delta"


def test_attach_score_delta_downgrades_conflicting_traditional_miss() -> None:
    event = _event().model_copy(update={"outcome": "missed", "shot_value": 2})
    delta = CandidateScoreDelta(
        event_id=event.event_id,
        team_id="ATL",
        points=2,
        before_frame=100,
        after_frame=200,
        before_scores={"ATL": 35, "CHI": 42},
        after_scores={"ATL": 37, "CHI": 42},
        confidence=0.96,
    )

    result = attach_score_delta(event, delta)

    assert result.status == "needs_review"
    assert result.outcome == "unknown"
    assert result.shot_value is None
    assert "conflicts" in result.reason


def test_extract_scoreboard_deltas_ignores_one_off_and_replay_decreases() -> None:
    reads = [
        _read(1, ATL=10, CHI=12),
        _read(2, ATL=10, CHI=12),
        _read(3, ATL=88, CHI=12),
        _read(4, ATL=12, CHI=12),
        _read(5, ATL=12, CHI=12),
        _read(6, ATL=10, CHI=12),
        _read(7, ATL=10, CHI=12),
    ]

    deltas = extract_scoreboard_deltas(reads)

    assert len(deltas) == 1
    assert (deltas[0].team_id, deltas[0].points) == ("ATL", 2)
    assert (deltas[0].before_frame, deltas[0].after_frame) == (2, 4)


def test_extract_scoreboard_deltas_preserves_each_team_in_sparse_transition() -> None:
    reads = [
        _read(1, ATL=10, CHI=12),
        _read(2, ATL=10, CHI=12),
        _read(5, ATL=12, CHI=15),
        _read(6, ATL=12, CHI=15),
    ]

    deltas = extract_scoreboard_deltas(reads)

    assert [(delta.team_id, delta.points) for delta in deltas] == [("ATL", 2), ("CHI", 3)]


def test_candidate_ids_for_delta_exposes_ambiguous_replay_candidates() -> None:
    first = _event().model_copy(update={"event_id": "shot-1", "start_frame": 100, "end_frame": 120})
    second = _event().model_copy(update={"event_id": "shot-2", "start_frame": 140, "end_frame": 160})
    delta = ScoreboardDelta(
        team_id="ATL",
        points=2,
        before_frame=90,
        after_frame=170,
        before_scores={"ATL": 10, "CHI": 12},
        after_scores={"ATL": 12, "CHI": 12},
        confidence=0.95,
    )

    assert candidate_ids_for_delta([first, second], delta) == ["shot-1", "shot-2"]


def test_score_delta_event_preserves_points_but_requires_actor_review() -> None:
    delta = ScoreboardDelta(
        team_id="CHI",
        points=3,
        before_frame=100,
        after_frame=140,
        before_scores={"ATL": 10, "CHI": 12},
        after_scores={"ATL": 10, "CHI": 15},
        confidence=0.96,
    )

    event = score_delta_event(
        delta,
        source_video_id="video_001",
        candidate_event_ids=["shot-a", "shot-b"],
    )

    assert event.status == "needs_review"
    assert event.primary_player_id is None
    assert (event.event_type, event.outcome, event.shot_value, event.team_id) == (
        "field_goal_attempt",
        "made",
        3,
        "CHI",
    )
    assert event.evidence[0].details["candidate_event_ids"] == ["shot-a", "shot-b"]


def test_score_delta_event_bounds_localization_but_preserves_delta_evidence() -> None:
    delta = ScoreboardDelta(
        team_id="ATL",
        points=2,
        before_frame=100,
        after_frame=1000,
        before_scores={"ATL": 10, "CHI": 12},
        after_scores={"ATL": 12, "CHI": 12},
        confidence=0.96,
    )

    event = score_delta_event(
        delta,
        source_video_id="video_001",
        localization_lookback_frames=240,
    )

    assert (event.start_frame, event.end_frame) == (760, 1000)
    assert (event.evidence[0].start_frame, event.evidence[0].end_frame) == (100, 1000)
    assert event.evidence[0].details["localization_start_frame"] == 760
    assert event.evidence[0].details["localization_lookback_frames"] == 240


def test_score_delta_event_rejects_non_positive_localization_lookback() -> None:
    delta = ScoreboardDelta(
        team_id="ATL",
        points=2,
        before_frame=100,
        after_frame=140,
        before_scores={"ATL": 10, "CHI": 12},
        after_scores={"ATL": 12, "CHI": 12},
        confidence=0.96,
    )

    with pytest.raises(ValueError, match="localization_lookback_frames"):
        score_delta_event(
            delta,
            source_video_id="video_001",
            localization_lookback_frames=0,
        )


def _read(frame: int, **scores: int) -> BroadcastScoreboardRead:
    return BroadcastScoreboardRead(frame=frame, scores=scores, confidence=0.95)


def _event() -> GameEventResponse:
    return GameEventResponse(
        event_id="shot-1",
        revision=1,
        event_type="field_goal_attempt",
        source_video_id="video_001",
        start_frame=90,
        end_frame=220,
        status="needs_review",
        outcome="unknown",
    )
