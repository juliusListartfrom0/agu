from __future__ import annotations

import pytest

from app.analysis.continuous_game_annotation_queue import build_continuous_game_annotation_queue
from app.analysis.heldout_annotation_batch import (
    build_heldout_annotation_batch,
    temporal_bucket,
    verify_heldout_annotation_batch,
)


def _queue() -> dict[str, object]:
    return build_continuous_game_annotation_queue(
        [
            {
                "source_id": "a",
                "source_video_filename": "a.webm",
                "source_video_sha256": "a" * 64,
                "source_fps": 10.0,
                "frame_count": 1000,
                "duration_seconds": 100.0,
            },
            {
                "source_id": "b",
                "source_video_filename": "b.webm",
                "source_video_sha256": "b" * 64,
                "source_fps": 10.0,
                "frame_count": 1000,
                "duration_seconds": 100.0,
            },
        ],
        stride_seconds=10.0,
        window_seconds=4.0,
    )


def test_heldout_batch_is_balanced_deterministic_and_excludes_anchor() -> None:
    queue = _queue()
    kwargs = {
        "per_source": 3,
        "seed": 17,
        "excluded_anchors": [{"source_id": "a", "center_seconds": 50.0}],
        "exclusion_radius_seconds": 20.0,
    }
    first = build_heldout_annotation_batch(queue, **kwargs)
    second = build_heldout_annotation_batch(queue, **kwargs)
    assert first == second
    assert len(first["windows"]) == 6
    assert {row["source_id"] for row in first["windows"]} == {"a", "b"}
    assert all(
        abs((float(row["start_seconds"]) + float(row["end_seconds"])) / 2.0 - 50.0) > 20.0
        for row in first["windows"]
        if row["source_id"] == "a"
    )
    assert verify_heldout_annotation_batch(first, queue=queue) == first


def test_heldout_batch_rejects_label_bearing_row() -> None:
    queue = _queue()
    batch = build_heldout_annotation_batch(queue, per_source=1, seed=1)
    batch["windows"][0]["outcome"] = "made"
    with pytest.raises(ValueError, match="label-bearing"):
        verify_heldout_annotation_batch(batch, queue=queue)


def test_heldout_batch_supports_deterministic_temporal_quotas() -> None:
    queue = _queue()
    quotas = {
        "a": {"early": 1, "middle": 1, "late": 1},
        "b": {"early": 1, "middle": 1, "late": 1},
    }
    first = build_heldout_annotation_batch(
        queue,
        per_source=3,
        seed=23,
        temporal_bucket_quotas=quotas,
    )
    second = build_heldout_annotation_batch(
        queue,
        per_source=3,
        seed=23,
        temporal_bucket_quotas=quotas,
    )
    assert first == second
    assert first["selection"]["temporal_bucket_field"] == "temporal_3way_v1"
    durations = {str(row["source_id"]): float(row["duration_seconds"]) for row in queue["sources"]}
    observed = {source: {bucket: 0 for bucket in ("early", "middle", "late")} for source in durations}
    for row in first["windows"]:
        observed[str(row["source_id"])][temporal_bucket(row, durations[str(row["source_id"])])] += 1
    assert observed == quotas
    assert verify_heldout_annotation_batch(first, queue=queue) == first


def test_heldout_batch_supports_a_source_disjoint_subset() -> None:
    queue = _queue()
    first = build_heldout_annotation_batch(
        queue,
        per_source=3,
        seed=31,
        source_ids=["b"],
        temporal_bucket_quotas={
            "b": {"early": 1, "middle": 1, "late": 1},
        },
    )
    assert first["selection"]["source_ids"] == ["b"]
    assert {row["source_id"] for row in first["windows"]} == {"b"}
    assert len(first["windows"]) == 3
    assert verify_heldout_annotation_batch(first, queue=queue) == first


def test_heldout_batch_rejects_unknown_or_duplicate_source_subset() -> None:
    queue = _queue()
    with pytest.raises(ValueError, match="source_ids"):
        build_heldout_annotation_batch(queue, per_source=1, seed=1, source_ids=["missing"])
    with pytest.raises(ValueError, match="source_ids"):
        build_heldout_annotation_batch(queue, per_source=1, seed=1, source_ids=["a", "a"])


def test_heldout_batch_rejects_temporal_quotas_with_wrong_total() -> None:
    with pytest.raises(ValueError, match="sum to per_source"):
        build_heldout_annotation_batch(
            _queue(),
            per_source=3,
            seed=23,
            temporal_bucket_quotas={
                "a": {"early": 1, "middle": 1, "late": 0},
                "b": {"early": 1, "middle": 1, "late": 1},
            },
        )
