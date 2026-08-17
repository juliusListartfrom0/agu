from __future__ import annotations

import hashlib
from pathlib import Path

from app.analysis.official_evaluation import seal_raw_only_predictions, verify_raw_only_bundle
from app.analysis.schemas import EventEvidenceResponse, GameEventResponse
from scripts.select_official_candidate_probe import build_probe_bundle, select_probe_events


def _event(index: int, *, confidence: float) -> GameEventResponse:
    return GameEventResponse(
        event_id=f"shot-{index}",
        revision=1,
        event_type="field_goal_attempt",
        source_video_id="video_001",
        start_frame=index * 100,
        end_frame=index * 100 + 60,
        status="needs_review",
        confidence=confidence,
        reason="fixture",
    )


def test_probe_selection_is_temporally_distributed_and_strength_ranked() -> None:
    events = [_event(index, confidence=0.5) for index in range(8)]
    events[1] = _event(1, confidence=0.9)
    events[5] = _event(5, confidence=0.8)

    selected = select_probe_events(
        events,
        event_types=["field_goal_attempt"],
        maximum_per_type=2,
    )

    assert [event.event_id for event in selected] == ["shot-1", "shot-5"]


def test_probe_can_exclude_already_resolved_trajectory_outcomes() -> None:
    events = [_event(index, confidence=0.5) for index in range(3)]
    events[1] = events[1].model_copy(update={"outcome": "made"})

    selected = select_probe_events(
        events,
        event_types=["field_goal_attempt"],
        maximum_per_type=3,
        unresolved_only=True,
    )

    assert [event.event_id for event in selected] == ["shot-0", "shot-2"]


def test_probe_can_filter_registered_primary_and_include_causal_parent(tmp_path: Path) -> None:
    video = tmp_path / "game.mp4"
    video.write_bytes(b"raw-video")
    parent = _event(1, confidence=0.8)
    rebound = GameEventResponse(
        event_id="rebound-1",
        revision=1,
        event_type="rebound",
        source_video_id="video_001",
        start_frame=170,
        end_frame=190,
        primary_player_id="registered-1",
        status="needs_review",
        confidence=0.8,
        related_event_ids=[parent.event_id],
    )
    other = rebound.model_copy(
        update={"event_id": "rebound-2", "primary_player_id": "anonymous-2", "related_event_ids": []}
    )
    source = seal_raw_only_predictions(
        game_id="game",
        raw_video_paths=[video],
        events=[parent, rebound, other],
        config={"pipeline": "fixture"},
        model_provenance={"producer": "agu", "candidate_backend": "fixture"},
    )

    probe = build_probe_bundle(
        source,
        video_paths=[video],
        event_types=["rebound"],
        maximum_per_type=9,
        allowed_player_ids=frozenset({"registered-1"}),
        include_related_events=True,
    )

    assert [event.event_id for event in probe.events] == [parent.event_id, rebound.event_id]
    assert probe.model_provenance["registered_candidate_only"] == "true"
    assert probe.model_provenance["include_related_events"] == "true"


def test_probe_registered_filter_accepts_evidence_candidate_without_primary() -> None:
    event = _event(1, confidence=0.8).model_copy(
        update={
            "primary_player_id": None,
            "evidence": [
                EventEvidenceResponse(
                    evidence_id="identity-candidates",
                    kind="traditional_cv",
                    source_video_id="video_001",
                    start_frame=100,
                    end_frame=160,
                    details={"candidate_player_ids": ["registered-2", "anonymous-1"]},
                )
            ],
        }
    )

    selected = select_probe_events(
        [event],
        event_types=["field_goal_attempt"],
        maximum_per_type=1,
        allowed_player_ids=frozenset({"registered-2"}),
    )

    assert [item.event_id for item in selected] == [event.event_id]


def test_probe_bundle_remains_hash_bound_to_raw_video_and_source(tmp_path: Path) -> None:
    video = tmp_path / "game.mp4"
    video.write_bytes(b"raw-video")
    source = seal_raw_only_predictions(
        game_id="game",
        raw_video_paths=[video],
        events=[_event(index, confidence=0.5) for index in range(4)],
        config={"pipeline": "fixture"},
        model_provenance={"producer": "agu", "candidate_backend": "fixture"},
    )

    probe = build_probe_bundle(
        source,
        video_paths=[video],
        event_types=["field_goal_attempt"],
        maximum_per_type=2,
    )

    assert verify_raw_only_bundle(probe) is probe
    assert len(probe.events) == 2
    assert probe.raw_videos[0].sha256 == hashlib.sha256(video.read_bytes()).hexdigest()
    assert probe.model_provenance["source_candidate_bundle_sha256"] == source.bundle_sha256
