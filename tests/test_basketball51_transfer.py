from __future__ import annotations

from app.analysis.basketball51_transfer import evaluate_free_throw_transfer


def test_transfer_evaluation_requires_every_game_to_pass() -> None:
    rows = [
        {"source_video_sha256": "game-a", "event_id": "a1", "free_throw": True},
        {"source_video_sha256": "game-a", "event_id": "a2", "free_throw": False},
        {"source_video_sha256": "game-b", "event_id": "b1", "free_throw": True},
        {"source_video_sha256": "game-b", "event_id": "b2", "free_throw": False},
    ]
    probabilities = [0.9, 0.1, 0.4, 0.2]

    result = evaluate_free_throw_transfer(rows, probabilities, threshold=0.5)

    assert result["overall"] == {
        "rows": 4,
        "positive_count": 2,
        "precision": 1.0,
        "recall": 0.5,
        "f1": 2 / 3,
    }
    assert result["per_game"][0]["source_video_sha256"] == "game-a"
    assert result["per_game"][0]["f1"] == 1.0
    assert result["per_game"][1]["source_video_sha256"] == "game-b"
    assert result["per_game"][1]["recall"] == 0.0
    assert result["promotion_eligible"] is False


def test_transfer_evaluation_rejects_prediction_length_mismatch() -> None:
    rows = [{"source_video_sha256": "game-a", "event_id": "a1", "free_throw": True}]

    try:
        evaluate_free_throw_transfer(rows, [], threshold=0.5)
    except ValueError as exc:
        assert "align" in str(exc)
    else:
        raise AssertionError("prediction mismatch must fail closed")
