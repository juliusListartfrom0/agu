from __future__ import annotations

import pytest

from app.analysis.schemas import GameEventResponse
from app.analysis.shot_validity_sampling_window import (
    SHOT_SAMPLING_PROTOCOL,
    ShotSamplingWindow,
    resolve_shot_sampling_window,
    validate_sampling_window_payload,
    video_frame_indexes,
)


def _event(
    *,
    start_frame: int,
    end_frame: int,
    release_frame: int | None = None,
    evidence_kind: str = "ball_rim_proximity_cluster",
    candidate_event_frame: int | None = None,
) -> GameEventResponse:
    details = {}
    if candidate_event_frame is not None:
        details["candidate_event_frame"] = candidate_event_frame
    return GameEventResponse.model_validate(
        {
            "event_id": "shot-1",
            "revision": 1,
            "event_type": "field_goal_attempt",
            "source_video_id": "video-1",
            "start_frame": start_frame,
            "end_frame": end_frame,
            "release_frame": release_frame,
            "outcome": "unknown",
            "confidence": 0.5,
            "status": "needs_review",
            "evidence": [
                {
                    "evidence_id": "evidence-1",
                    "kind": evidence_kind,
                    "source_video_id": "video-1",
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "confidence": 0.5,
                    "details": details,
                }
            ],
        }
    )


def test_release_or_raw_candidate_anchor_produces_atomic_four_second_window() -> None:
    released = resolve_shot_sampling_window(
        _event(start_frame=100, end_frame=1000, release_frame=500),
        fps=30.0,
    )
    proposed = resolve_shot_sampling_window(
        _event(start_frame=100, end_frame=1000, candidate_event_frame=400),
        fps=30.0,
    )

    assert released == ShotSamplingWindow(
        start_frame=455,
        end_frame=575,
        anchor_frame=500,
        anchor_source="release_frame",
        protocol=SHOT_SAMPLING_PROTOCOL,
    )
    assert (proposed.start_frame, proposed.end_frame) == (355, 475)
    assert proposed.anchor_source == "candidate_event_frame"


def test_scoreboard_interval_is_bounded_to_last_twelve_seconds() -> None:
    window = resolve_shot_sampling_window(
        _event(
            start_frame=100,
            end_frame=1000,
            evidence_kind="scoreboard_delta",
        ),
        fps=30.0,
    )

    assert (window.start_frame, window.end_frame) == (640, 1000)
    assert window.anchor_frame == 1000
    assert window.anchor_source == "scoreboard_after_frame"


def test_short_unanchored_window_is_preserved_but_long_unknown_window_is_rejected() -> None:
    short = resolve_shot_sampling_window(
        _event(start_frame=100, end_frame=220, evidence_kind="other"),
        fps=30.0,
    )
    assert (short.start_frame, short.end_frame) == (100, 220)
    assert short.anchor_source == "bounded_event"

    with pytest.raises(ValueError, match="reliable sampling anchor"):
        resolve_shot_sampling_window(
            _event(start_frame=100, end_frame=1000, evidence_kind="other"),
            fps=30.0,
        )


@pytest.mark.parametrize("fps", [0.0, -1.0])
def test_sampling_window_rejects_invalid_fps(fps: float) -> None:
    with pytest.raises(ValueError, match="fps"):
        resolve_shot_sampling_window(
            _event(start_frame=100, end_frame=220),
            fps=fps,
        )


def test_video_frame_indexes_only_cover_the_resolved_window() -> None:
    window = ShotSamplingWindow(
        start_frame=355,
        end_frame=475,
        anchor_frame=400,
        anchor_source="candidate_event_frame",
    )

    indexes = video_frame_indexes(window, clip_frames=16)

    assert len(indexes) == 16
    assert indexes[0] == 355
    assert indexes[-1] == 475
    assert all(355 <= value <= 475 for value in indexes)


def test_sampling_window_payload_requires_bound_anchor_and_protocol() -> None:
    payload = {
        "start_frame": 355,
        "end_frame": 475,
        "anchor_frame": 400,
        "anchor_source": "candidate_event_frame",
        "protocol": SHOT_SAMPLING_PROTOCOL,
    }
    validate_sampling_window_payload(payload)

    payload["anchor_frame"] = 500
    with pytest.raises(ValueError, match="anchor"):
        validate_sampling_window_payload(payload)
