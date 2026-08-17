from __future__ import annotations

import pytest

from app.analysis.replay_transition import (
    REPLAY_LOGO_RECURRENCE_SCHEMA,
    REPLAY_TRANSITION_EVIDENCE_SCHEMA,
    best_logo_recurrence_match,
    evaluate_replay_logo_proxy,
    replay_transition_payload,
    seal_replay_logo_artifact,
    seal_replay_transition_artifact,
    summarize_transition_probabilities,
    verify_replay_transition_artifact,
)


def test_replay_transition_requires_dense_anchor_local_cuts() -> None:
    values = [0.1] * 600
    for index in (100, 220, 260, 310, 440):
        values[index] = 0.9

    evidence = summarize_transition_probabilities(
        event_id="shot-1",
        anchor_frame=1_000,
        window_start_frame=700,
        probabilities=values,
        fps=30.0,
    )

    assert evidence.transition_count == 5
    assert evidence.near_transition_count == 5
    assert evidence.broadcast_state == "replay"
    assert evidence.transition_offsets_seconds == pytest.approx(
        (-6.6666667, -2.6666667, -1.3333333, 0.3333333, 4.6666667)
    )


def test_replay_transition_never_promotes_low_cut_window_to_live() -> None:
    values = [0.1] * 600
    values[250] = 0.99

    evidence = summarize_transition_probabilities(
        event_id="shot-2",
        anchor_frame=1_000,
        window_start_frame=700,
        probabilities=values,
        fps=30.0,
    )

    assert evidence.transition_count == 1
    assert evidence.broadcast_state == "unknown"


def test_replay_transition_artifact_is_hash_bound_and_offline() -> None:
    evidence = summarize_transition_probabilities(
        event_id="shot-3",
        anchor_frame=100,
        window_start_frame=50,
        probabilities=[0.1] * 100,
        fps=10.0,
    )
    artifact = seal_replay_transition_artifact(
        {
            "schema_version": REPLAY_TRANSITION_EVIDENCE_SCHEMA,
            "runtime_consumable": False,
            "events": [replay_transition_payload(evidence)],
        }
    )

    assert verify_replay_transition_artifact(artifact) == artifact
    tampered = {**artifact, "runtime_consumable": True}
    with pytest.raises(ValueError, match="offline"):
        verify_replay_transition_artifact(tampered)


def test_logo_recurrence_excludes_local_window_and_finds_remote_match() -> None:
    import numpy as np

    query = np.arange(18, dtype=np.float32).reshape(2, 3, 3) / 18.0
    timeline = np.zeros((20, 2, 3, 3), dtype=np.float32)
    timeline[10] = query
    timeline[2] = query

    match = best_logo_recurrence_match(
        query_rgb=query,
        timeline_rgb=timeline,
        query_seconds=10.0,
        scan_fps=1.0,
        exclusion_seconds=2.0,
    )

    assert match["matched_seconds"] == 2.0
    assert match["correlation"] == pytest.approx(1.0, abs=1e-5)
    assert match["mean_absolute_difference"] == pytest.approx(0.0)


def test_replay_logo_artifact_is_offline_and_hash_bound() -> None:
    artifact = seal_replay_logo_artifact(
        {
            "schema_version": REPLAY_LOGO_RECURRENCE_SCHEMA,
            "runtime_consumable": False,
            "events": [
                {
                    "event_id": "shot-5",
                    "candidates": [
                        {
                            "frame": 100,
                            "person_count": 0,
                            "recurrence_correlation": 0.8,
                            "recurrence_mean_absolute_difference": 0.1,
                        }
                    ],
                    "broadcast_state": "replay",
                }
            ],
        }
    )

    assert artifact["artifact_sha256"]


def test_replay_logo_proxy_fails_closed_on_live_transition_false_positive() -> None:
    evidence = seal_replay_logo_artifact(
        {
            "schema_version": REPLAY_LOGO_RECURRENCE_SCHEMA,
            "runtime_consumable": False,
            "raw_video_sha256": "video-sha",
            "candidate_bundle_sha256": "bundle-sha",
            "events": [
                {
                    "event_id": "live-shot",
                    "candidates": [],
                    "broadcast_state": "replay",
                },
                {
                    "event_id": "replay-shot",
                    "candidates": [],
                    "broadcast_state": "replay",
                },
            ],
        }
    )
    labels = {
        "schema_version": "agu.shot-validity-labels.v1",
        "runtime_consumable": False,
        "source_video_sha256": "video-sha",
        "candidate_bundle_sha256": "bundle-sha",
        "examples": [
            {
                "event_id": "live-shot",
                "event_present": True,
                "review_note": "continuous live release",
            },
            {
                "event_id": "replay-shot",
                "event_present": False,
                "review_note": "broadcast replay",
            },
        ],
    }

    metrics = evaluate_replay_logo_proxy(labels=[labels], evidence=[evidence])

    assert metrics["precision"] == 0.5
    assert metrics["promotion_eligible"] is False
    assert metrics["false_positive_event_ids"] == ["video-sha/live-shot"]


@pytest.mark.parametrize(
    "update",
    [
        {"fps": 0.0},
        {"transition_threshold": 0.0},
        {"minimum_transition_count": 0},
        {"window_start_frame": 101},
    ],
)
def test_replay_transition_rejects_invalid_configuration(
    update: dict[str, float | int],
) -> None:
    arguments = {
        "event_id": "shot-4",
        "anchor_frame": 100,
        "window_start_frame": 50,
        "probabilities": [0.1] * 100,
        "fps": 10.0,
    }
    arguments.update(update)
    with pytest.raises(ValueError, match="configuration"):
        summarize_transition_probabilities(**arguments)
