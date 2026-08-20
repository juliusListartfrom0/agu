"""Tests for the TASK-0258 v2 verification extraction wiring."""

from __future__ import annotations

import pytest

from app.analysis.task0258_v2_verification_extract import build_verification_examples


class _FakePlan:
    def __init__(self, rows):
        self._payload = {"ordered_examples": rows}


def _plan_row(ordinal, event_id):
    return {
        "ordinal": ordinal,
        "source_video_sha256": "0" * 64,
        "candidate_bundle_sha256": "0" * 64,
        "event_id": event_id,
        "tile_frame_indexes": [[0, 1, 2, 3] * 4] * 4,
    }


def test_build_verification_examples_structure():
    plan_rows = [_plan_row(1, "e1"), _plan_row(2, "e2")]
    plan = _FakePlan(plan_rows)
    tiles = [
        [[0.1, 0.2] * 384] * 4,
        [[0.3, 0.4] * 384] * 4,
    ]
    examples = build_verification_examples(plan, tiles)
    assert len(examples) == 2
    for ex, plan_row in zip(examples, plan_rows):
        assert set(ex) == {"ordinal", "key", "tile_frame_indexes", "derivation_only_tile_embeddings", "model_input"}
        assert ex["ordinal"] == plan_row["ordinal"]
        assert ex["key"] == {
            "source_video_sha256": plan_row["source_video_sha256"],
            "candidate_bundle_sha256": plan_row["candidate_bundle_sha256"],
            "event_id": plan_row["event_id"],
        }
        assert len(ex["derivation_only_tile_embeddings"]) == 4
        assert len(ex["derivation_only_tile_embeddings"][0]) == 768
        assert len(ex["model_input"]) == 1536


def test_build_verification_examples_rejects_mismatch():
    plan = _FakePlan([_plan_row(1, "e1"), _plan_row(2, "e2")])
    with pytest.raises(ValueError):
        build_verification_examples(plan, [[[0.0] * 768] * 4])
