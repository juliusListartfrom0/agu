from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.analysis.broadcast_clock import (
    BROADCAST_CLOCK_EVIDENCE_SCHEMA,
    seal_broadcast_clock_artifact,
)
from app.analysis.shot_validity_scene_state import (
    SCENE_BACKBONE,
    SCENE_EMBEDDING_DIMENSION,
    seal_scene_embedding_artifact,
)
from app.analysis.schemas import GameEventResponse
from scripts.run_broadcast_clock_replay_evidence import (
    candidate_anchor,
    load_resume_events,
    select_scene_event_ids,
)


def _scene() -> dict[str, object]:
    examples = []
    for bundle, event_id, present in (
        ("c" * 64, "event-1", True),
        ("f" * 64, "event-2", False),
    ):
        examples.append(
            {
                "source_video_sha256": "d" * 64,
                "candidate_bundle_sha256": bundle,
                "event_id": event_id,
                "event_present": present,
                "phase_embeddings": [
                    [0.0] * SCENE_EMBEDDING_DIMENSION,
                    [0.0] * SCENE_EMBEDDING_DIMENSION,
                    [0.0] * SCENE_EMBEDDING_DIMENSION,
                ],
            }
        )
    return seal_scene_embedding_artifact(
        {
            "purpose": "test",
            "producer": "test",
            "training_manifest_sha256": "a" * 64,
            "training_annotation_sha256": ["b" * 64],
            "source_video_sha256": ["d" * 64],
            "backbone": SCENE_BACKBONE,
            "backbone_sha256": "e" * 64,
            "backbone_license": "BSD-3-Clause",
            "backbone_weights_url": "https://example.invalid/model",
            "embedding_dimension": SCENE_EMBEDDING_DIMENSION,
            "phase_fractions": [0.15, 0.5, 0.85],
            "examples": examples,
        }
    )


def _clock_event(event_id: str) -> dict[str, object]:
    return {
        "event_id": event_id,
        "anchor_frame": 100,
        "samples": [
            {"frame": 100, "offset_seconds": 0.0, "read": None},
        ],
        "pre_anchor_read_count": 0,
        "post_anchor_missing_count": 1,
        "frozen_period": None,
        "frozen_clock_seconds": None,
        "broadcast_state": "unknown",
        "method": "pre_anchor_frozen_clock_disappearance_v1",
    }


def test_scene_filter_uses_only_source_bundle_keys_not_labels() -> None:
    selected = select_scene_event_ids(
        _scene(),
        raw_video_sha256="d" * 64,
        candidate_bundle_sha256="c" * 64,
    )

    assert selected == {"event-1"}


def test_candidate_anchor_uses_generic_candidate_event_frame() -> None:
    event = GameEventResponse.model_validate(
        {
            "event_id": "extra-event",
            "revision": 1,
            "event_type": "field_goal_attempt",
            "source_video_id": "video-001",
            "start_frame": 100,
            "end_frame": 220,
            "outcome": "unknown",
            "status": "candidate",
            "confidence": 0.5,
            "reason": "test",
            "evidence": [
                {
                    "evidence_id": "extra-event-anchor",
                    "kind": "offline_uniform_extra_window",
                    "source_video_id": "video-001",
                    "start_frame": 100,
                    "end_frame": 220,
                    "confidence": 0.5,
                    "details": {"candidate_event_frame": 160},
                }
            ],
        }
    )

    assert candidate_anchor(event) == 160


def test_resume_requires_matching_binding_configuration_and_event_subset(
    tmp_path: Path,
) -> None:
    configuration = {
        "sample_offset_seconds": [-6.0, -3.0, 0.0, 3.0, 6.0],
        "crop_top_ratio": 0.58,
        "ocr_confidence": 0.55,
        "maximum_pre_anchor_offset_seconds": -3.0,
        "minimum_pre_anchor_read_count": 2,
        "minimum_post_anchor_missing_count": 3,
    }
    artifact = seal_broadcast_clock_artifact(
        {
            "schema_version": BROADCAST_CLOCK_EVIDENCE_SCHEMA,
            "purpose": "offline_raw_only_broadcast_clock_replay_screening",
            "runtime_consumable": False,
            "codex_runtime_answer_used": False,
            "raw_video_sha256": "d" * 64,
            "candidate_bundle_sha256": "c" * 64,
            "source": {
                "ocr_engine": "RapidOCR",
                "parser": "period_clock_same_row_v1",
            },
            "configuration": configuration,
            "events": [_clock_event("event-1")],
        }
    )
    output = tmp_path / "checkpoint.json"
    output.write_text(json.dumps(artifact), encoding="utf-8")

    resumed = load_resume_events(
        output,
        raw_video_sha256="d" * 64,
        candidate_bundle_sha256="c" * 64,
        configuration=configuration,
        allowed_event_ids={"event-1", "event-2"},
    )

    assert set(resumed) == {"event-1"}

    with pytest.raises(ValueError, match="configuration"):
        load_resume_events(
            output,
            raw_video_sha256="d" * 64,
            candidate_bundle_sha256="c" * 64,
            configuration={**configuration, "crop_top_ratio": 0.6},
            allowed_event_ids={"event-1", "event-2"},
        )

    with pytest.raises(ValueError, match="outside"):
        load_resume_events(
            output,
            raw_video_sha256="d" * 64,
            candidate_bundle_sha256="c" * 64,
            configuration=configuration,
            allowed_event_ids={"event-2"},
        )
