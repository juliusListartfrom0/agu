from __future__ import annotations

import hashlib

import pytest

from app.analysis.continuous_game_annotation_queue import (
    build_continuous_game_annotation_queue,
    select_annotation_batch,
    verify_continuous_game_annotation_queue,
)


def _sha(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def _sources() -> list[dict[str, object]]:
    return [
        {
            "source_id": "hazen",
            "source_video_filename": "hazen.webm",
            "source_video_sha256": _sha("hazen"),
            "source_fps": 30.0,
            "frame_count": 300,
            "duration_seconds": 10.0,
        },
        {
            "source_id": "vtv",
            "source_video_filename": "vtv.webm",
            "source_video_sha256": _sha("vtv"),
            "source_fps": 25.0,
            "frame_count": 250,
            "duration_seconds": 10.0,
        },
    ]


def test_queue_is_hash_bound_label_hidden_and_frame_safe() -> None:
    queue = build_continuous_game_annotation_queue(
        _sources(), stride_seconds=2.0, window_seconds=1.0
    )

    assert queue["runtime_consumable"] is False
    assert queue["codex_runtime_answer_used"] is False
    assert queue["labels_hidden_from_reviewer"] is True
    assert len(queue["windows"]) == 10
    for row in queue["windows"]:
        source = next(item for item in queue["sources"] if item["source_id"] == row["source_id"])
        assert 0 <= row["start_frame"] < row["end_frame"] <= source["frame_count"]
        assert row["source_video_sha256"] == source["source_video_sha256"]
        assert "event_present" not in row
        assert "label" not in row

    verified = verify_continuous_game_annotation_queue(queue)
    assert verified["artifact_sha256"] == queue["artifact_sha256"]


def test_annotation_batch_is_deterministic_and_balanced_by_source() -> None:
    queue = build_continuous_game_annotation_queue(
        _sources(), stride_seconds=1.0, window_seconds=0.5
    )
    first = select_annotation_batch(queue, per_source=2, seed=17)
    second = select_annotation_batch(queue, per_source=2, seed=17)

    assert first == second
    assert len(first) == 4
    assert {row["source_id"] for row in first} == {"hazen", "vtv"}
    assert len({row["window_id"] for row in first}) == len(first)


def test_queue_rejects_label_bearing_source_metadata() -> None:
    sources = _sources()
    sources[0]["event_present"] = False
    with pytest.raises(ValueError, match="label-bearing"):
        build_continuous_game_annotation_queue(sources)


def test_queue_hash_tampering_fails_closed() -> None:
    queue = build_continuous_game_annotation_queue(
        _sources(), stride_seconds=2.0, window_seconds=1.0
    )
    queue["windows"][0]["start_seconds"] = 0.25
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_continuous_game_annotation_queue(queue)
