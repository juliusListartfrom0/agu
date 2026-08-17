from __future__ import annotations

import copy
import json

import pytest

from app.analysis.causal_shot_phase_review import (
    CAUSAL_PHASE_OFFSETS_SECONDS,
    build_causal_shot_phase_review_plan,
    seal_causal_shot_phase_review,
    verify_causal_shot_phase_review,
    verify_causal_shot_phase_review_plan,
)
from app.analysis.pbp_visual_state_frames import (
    DINO_V2_SMALL_BACKBONE,
    PRE_ANCHOR_OFFSETS_SECONDS,
    seal_anchor_state_embedding_artifact,
)
from app.analysis.pbp_visual_state_review import (
    build_visual_state_review_plan,
)


def _base_plan() -> dict:
    embeddings = seal_anchor_state_embedding_artifact(
        {
            "purpose": "test",
            "truth_used_for_training_only": True,
            "codex_runtime_answer_used": False,
            "training_manifest_sha256s": ["1" * 64],
            "clock_artifact_sha256s": ["2" * 64],
            "source_video_sha256s": ["3" * 64],
            "sealed_blind_video_sha256s": ["4" * 64],
            "backbone": DINO_V2_SMALL_BACKBONE,
            "backbone_sha256": "5" * 64,
            "anchor_offsets_seconds": list(PRE_ANCHOR_OFFSETS_SECONDS),
            "examples": [
                {
                    "source_video_sha256": "3" * 64,
                    "source_video_filename": "development.mp4",
                    "candidate_bundle_sha256": "6" * 64,
                    "event_id": "event-1",
                    "state": "free_throw",
                    "anchor_frame": 300,
                    "source_fps": 30.0,
                    "frame_count": 1000,
                    "frame_indexes": [180, 210, 240, 270, 300],
                    "embeddings": [
                        [float(index)] * 384 for index in range(5)
                    ],
                }
            ],
        }
    )
    return build_visual_state_review_plan(embeddings)


def test_causal_phase_plan_is_dense_label_hidden_and_offline_only() -> None:
    plan = build_causal_shot_phase_review_plan(_base_plan())

    assert plan["schema_version"] == "agu.causal-shot-phase-review-plan.v1"
    assert plan["runtime_consumable"] is False
    assert plan["codex_runtime_answer_used"] is False
    assert plan["labels_hidden_from_reviewer"] is True
    assert plan["anchor_offsets_seconds"] == list(
        CAUSAL_PHASE_OFFSETS_SECONDS
    )
    assert plan["reviewer_visible_fields"] == [
        "phase_review_id",
        "frame_indexes",
        "anchor_offsets_seconds",
        "raw_frames",
    ]
    assert plan["annotation_rules"]["release_position"] == (
        "earliest sampled position with the ball visibly separated from "
        "the shooter's hand"
    )
    encoded = json.dumps(plan["examples"], sort_keys=True)
    assert "free_throw" not in encoded
    assert "field_goal" not in encoded
    row = plan["examples"][0]
    assert row["phase_review_id"] == "causal-phase-0001"
    assert row["frame_indexes"][CAUSAL_PHASE_OFFSETS_SECONDS.index(0.0)] == 300
    verify_causal_shot_phase_review_plan(plan)


def test_causal_phase_plan_rejects_blind_source_and_tampering() -> None:
    with pytest.raises(ValueError, match="sealed blind"):
        build_causal_shot_phase_review_plan(
            _base_plan(),
            additional_sealed_blind_video_sha256s=["3" * 64],
        )

    plan = build_causal_shot_phase_review_plan(_base_plan())
    tampered = copy.deepcopy(plan)
    tampered["examples"][0]["frame_indexes"][0] += 1
    with pytest.raises(ValueError, match="hash mismatch|frame window"):
        verify_causal_shot_phase_review_plan(tampered)


def test_causal_phase_review_requires_exact_coverage_and_causal_order() -> None:
    plan = build_causal_shot_phase_review_plan(_base_plan())
    review = seal_causal_shot_phase_review(
        {
            "plan_sha256": plan["artifact_sha256"],
            "reviewer": "codex_offline_training_annotation",
            "reviews": [
                {
                    "phase_review_id": "causal-phase-0001",
                    "formation_state": "free_throw_setup",
                    "broadcast_context": "live_action",
                    "shot_sequence": "complete",
                    "release_position": 7,
                    "rim_arrival_position": 13,
                    "outcome": "made",
                    "confidence": "high",
                    "notes": "Release and rim crossing are both visible.",
                }
            ],
        },
        plan=plan,
    )

    verified = verify_causal_shot_phase_review(review, plan=plan)
    assert verified["runtime_consumable"] is False
    assert verified["reviews"][0]["release_frame"] == (
        plan["examples"][0]["frame_indexes"][7]
    )
    assert verified["reviews"][0]["rim_arrival_frame"] == (
        plan["examples"][0]["frame_indexes"][13]
    )

    invalid = copy.deepcopy(review)
    invalid.pop("artifact_sha256")
    invalid["reviews"][0]["release_position"] = 14
    invalid["reviews"][0]["rim_arrival_position"] = 13
    with pytest.raises(ValueError, match="precede"):
        seal_causal_shot_phase_review(invalid, plan=plan)

    incomplete = copy.deepcopy(review)
    incomplete.pop("artifact_sha256")
    incomplete["reviews"] = []
    with pytest.raises(ValueError, match="exactly cover"):
        seal_causal_shot_phase_review(incomplete, plan=plan)


def test_causal_phase_review_enforces_sequence_specific_fields() -> None:
    plan = build_causal_shot_phase_review_plan(_base_plan())
    payload = {
        "plan_sha256": plan["artifact_sha256"],
        "reviewer": "codex_offline_training_annotation",
        "reviews": [
            {
                "phase_review_id": "causal-phase-0001",
                "formation_state": "stoppage_other",
                "broadcast_context": "non_action",
                "shot_sequence": "not_a_shot",
                "release_position": 7,
                "rim_arrival_position": None,
                "outcome": "not_applicable",
                "confidence": "medium",
                "notes": "",
            }
        ],
    }
    with pytest.raises(ValueError, match="not-a-shot"):
        seal_causal_shot_phase_review(payload, plan=plan)
