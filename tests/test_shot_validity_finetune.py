from pathlib import Path

import torch

from app.analysis.schemas import EventEvidenceResponse, GameEventResponse
from app.analysis.shot_validity_finetune import (
    ShotWindowRecord,
    anchor_window_bounds,
    binary_metrics,
    evaluate_game_gate,
    normalize_probability_by_threshold,
    prepare_window,
    resolve_training_anchor,
    select_precision_threshold,
)


def _record(game: str, label: bool, index: int) -> ShotWindowRecord:
    return ShotWindowRecord(game, "bundle", f"event-{index}", label, 0, 15, Path("game.mp4"))


def test_resolve_training_anchor_prefers_release_then_candidate_frame() -> None:
    event = GameEventResponse(
        event_id="shot",
        revision=1,
        event_type="field_goal_attempt",
        source_video_id="video_001",
        start_frame=100,
        end_frame=220,
        outcome="unknown",
        status="needs_review",
        confidence=0.5,
        evidence=[
            EventEvidenceResponse(
                evidence_id="shot:vision",
                kind="ball_rim_proximity_cluster",
                source_video_id="video_001",
                start_frame=140,
                end_frame=150,
                confidence=0.5,
                details={"candidate_event_frame": 145, "review_anchor_frame": 150},
            )
        ],
    )

    assert resolve_training_anchor(event) == (145, "candidate_event_frame")
    assert resolve_training_anchor(event.model_copy(update={"release_frame": 180})) == (
        180,
        "release_frame",
    )


def test_anchor_window_bounds_are_bounded_and_keep_anchor() -> None:
    record = ShotWindowRecord(
        "game",
        "bundle",
        "event",
        True,
        100,
        220,
        Path("game.mp4"),
        145,
        "candidate_event_frame",
    )

    assert anchor_window_bounds(record, context_frames=30) == (115, 175)
    assert anchor_window_bounds(record, context_frames=200) == (100, 220)


def test_prepare_window_is_deterministic_for_evaluation() -> None:
    window = (
        torch.arange(32 * 3 * 128 * 171, dtype=torch.int64)
        .remainder(256)
        .to(torch.uint8)
        .reshape(32, 3, 128, 171)
    )
    first = prepare_window(window, training=False)
    second = prepare_window(window, training=False)
    assert first.shape == (3, 16, 112, 112)
    assert torch.equal(first, second)


def test_prepare_window_training_uses_seeded_augmentation() -> None:
    window = torch.full((32, 3, 128, 171), 127, dtype=torch.uint8)
    first = prepare_window(window, training=True, generator=torch.Generator().manual_seed(7))
    second = prepare_window(window, training=True, generator=torch.Generator().manual_seed(7))
    assert torch.equal(first, second)


def test_prepare_window_preserves_agu_v3_bgr_range() -> None:
    window = torch.zeros((32, 3, 128, 171), dtype=torch.uint8)
    window[:, 0] = 10
    window[:, 1] = 20
    window[:, 2] = 30
    clip = prepare_window(window, training=False, preprocessing="agu_v3")
    assert clip[:, 0, 0, 0].tolist() == [30.0, 20.0, 10.0]


def test_prepare_window_anchor_sampling_keeps_local_temporal_context() -> None:
    window = torch.stack(
        [torch.full((3, 128, 171), frame, dtype=torch.uint8) for frame in range(32)]
    )
    clip = prepare_window(
        window,
        training=False,
        preprocessing="agu_v3",
        temporal_sampling="anchor",
        anchor_fraction=0.75,
    )

    assert clip.shape == (3, 16, 112, 112)
    assert clip[0, :, 0, 0].tolist() == list(range(15, 31))


def test_threshold_selection_prioritizes_recall_under_precision_floor() -> None:
    threshold, metrics = select_precision_threshold(
        [True, True, False, False], [0.9, 0.8, 0.7, 0.1], minimum_precision=0.95
    )
    assert threshold == 0.8
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_probability_normalization_maps_calibration_threshold_to_half() -> None:
    assert normalize_probability_by_threshold(0.8, 0.8) == 0.5
    assert normalize_probability_by_threshold(0.9, 0.8) > 0.5
    assert normalize_probability_by_threshold(0.7, 0.8) < 0.5


def test_game_gate_requires_every_game_to_pass() -> None:
    records = [
        _record("a", True, 1),
        _record("a", False, 2),
        _record("b", True, 3),
        _record("b", False, 4),
    ]
    result = evaluate_game_gate(records, [0.9, 0.1, 0.4, 0.2], threshold=0.5)
    assert result["pooled"] == binary_metrics([True, False, True, False], [0.9, 0.1, 0.4, 0.2], 0.5)
    assert result["per_game"]["a"]["recall"] == 1.0
    assert result["per_game"]["b"]["recall"] == 0.0
    assert result["promoted"] is False
