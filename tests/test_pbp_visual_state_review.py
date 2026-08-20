from __future__ import annotations

import copy
import json

import pytest

from app.analysis.pbp_visual_state_frames import (
    DINO_V2_SMALL_BACKBONE,
    PRE_ANCHOR_OFFSETS_SECONDS,
    seal_anchor_state_embedding_artifact,
)
from app.analysis.pbp_visual_state_review import (
    FOLLOWUP_OFFSETS_SECONDS,
    build_visual_state_followup_review_plan,
    build_visual_state_review_plan,
    derive_visual_state_label_corrections,
    seal_visual_state_review,
    verify_visual_state_label_corrections,
    verify_visual_state_review_plan,
)


def _embedding_artifact() -> dict:
    return seal_anchor_state_embedding_artifact(
        {
            "purpose": "test",
            "truth_used_for_training_only": True,
            "codex_runtime_answer_used": False,
            "training_manifest_sha256s": ["1" * 64],
            "clock_artifact_sha256s": ["2" * 64],
            "source_video_sha256s": ["3" * 64],
            "sealed_blind_video_sha256s": ["4" * 64, "5" * 64],
            "backbone": DINO_V2_SMALL_BACKBONE,
            "backbone_sha256": "6" * 64,
            "anchor_offsets_seconds": list(PRE_ANCHOR_OFFSETS_SECONDS),
            "examples": [
                {
                    "source_video_sha256": "3" * 64,
                    "source_video_filename": "development.mp4",
                    "candidate_bundle_sha256": "7" * 64,
                    "event_id": "event-1",
                    "state": "free_throw",
                    "anchor_frame": 300,
                    "source_fps": 30.0,
                    "frame_count": 1000,
                    "frame_indexes": [180, 210, 240, 270, 300],
                    "embeddings": [[float(index)] * 384 for index in range(5)],
                }
            ],
        }
    )


def test_review_plan_hides_training_labels_and_remains_offline_only() -> None:
    plan = build_visual_state_review_plan(_embedding_artifact())

    assert plan["schema_version"] == "agu.pbp-visual-state-review-plan.v1"
    assert plan["runtime_consumable"] is False
    assert plan["codex_runtime_answer_used"] is False
    assert plan["reviewer_visible_fields"] == [
        "review_id",
        "frame_indexes",
        "raw_frames",
    ]
    encoded_examples = json.dumps(plan["examples"], sort_keys=True)
    assert "free_throw" not in encoded_examples
    assert "field_goal" not in encoded_examples
    assert plan["examples"][0]["frame_indexes"] == [180, 210, 240, 270, 300]
    verify_visual_state_review_plan(plan)


def test_review_plan_rejects_any_sealed_blind_source() -> None:
    with pytest.raises(ValueError, match="sealed blind"):
        build_visual_state_review_plan(
            _embedding_artifact(),
            additional_sealed_blind_video_sha256s=["3" * 64],
        )


def test_review_requires_exact_plan_coverage_and_derives_corrections() -> None:
    embeddings = _embedding_artifact()
    plan = build_visual_state_review_plan(embeddings)
    review = seal_visual_state_review(
        {
            "plan_sha256": plan["artifact_sha256"],
            "reviewer": "codex_offline_visual_review",
            "reviews": [
                {
                    "review_id": "visual-state-0001",
                    "visual_state": "live_play",
                    "confidence": "high",
                    "notes": "Half-court possession; no free-throw formation.",
                }
            ],
        },
        plan=plan,
    )

    corrections = derive_visual_state_label_corrections(
        embedding_artifact=embeddings,
        plan=plan,
        review=review,
    )

    assert corrections["runtime_consumable"] is False
    assert corrections["codex_runtime_answer_used"] is False
    assert corrections["summary"] == {
        "reviewed": 1,
        "resolved": 1,
        "changed": 1,
        "unresolved": 0,
    }
    assert corrections["decisions"][0]["original_state"] == "free_throw"
    assert corrections["decisions"][0]["corrected_state"] == "field_goal"

    incomplete = copy.deepcopy(review)
    incomplete.pop("artifact_sha256")
    incomplete["reviews"] = []
    with pytest.raises(ValueError, match="exactly cover"):
        seal_visual_state_review(incomplete, plan=plan)


def test_review_plan_hash_detects_tampering() -> None:
    plan = build_visual_state_review_plan(_embedding_artifact())
    tampered = copy.deepcopy(plan)
    tampered["examples"][0]["anchor_frame"] += 1

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_visual_state_review_plan(tampered)


def test_followup_plan_contains_only_unresolved_rows_with_wider_frames() -> None:
    embeddings = _embedding_artifact()
    second = copy.deepcopy(embeddings["examples"][0])
    second["event_id"] = "event-2"
    second["anchor_frame"] = 600
    second["frame_indexes"] = [480, 510, 540, 570, 600]
    embeddings.pop("artifact_sha256")
    embeddings["examples"].append(second)
    embeddings = seal_anchor_state_embedding_artifact(embeddings)
    plan = build_visual_state_review_plan(embeddings)
    review = seal_visual_state_review(
        {
            "plan_sha256": plan["artifact_sha256"],
            "reviewer": "codex_offline_visual_review",
            "reviews": [
                {
                    "review_id": "visual-state-0001",
                    "visual_state": "live_play",
                    "confidence": "high",
                    "notes": "Resolved.",
                },
                {
                    "review_id": "visual-state-0002",
                    "visual_state": "stoppage_other",
                    "confidence": "medium",
                    "notes": "Needs wider temporal evidence.",
                },
            ],
        },
        plan=plan,
    )
    corrections = derive_visual_state_label_corrections(
        embedding_artifact=embeddings,
        plan=plan,
        review=review,
    )

    followup = build_visual_state_followup_review_plan(
        plan,
        corrections=corrections,
    )

    assert followup["parent_plan_sha256"] == plan["artifact_sha256"]
    assert (
        followup["parent_corrections_sha256"]
        == corrections["artifact_sha256"]
    )
    assert followup["anchor_offsets_seconds"] == list(
        FOLLOWUP_OFFSETS_SECONDS
    )
    assert [row["review_id"] for row in followup["examples"]] == [
        "visual-state-0002"
    ]
    assert followup["examples"][0]["frame_indexes"] == [
        360,
        420,
        480,
        540,
        600,
        660,
        720,
        780,
        840,
    ]
    assert "state" not in followup["examples"][0]
    verify_visual_state_review_plan(followup)
    verify_visual_state_label_corrections(corrections, plan=plan)


def test_followup_plan_rejects_tampered_or_fully_resolved_corrections() -> None:
    embeddings = _embedding_artifact()
    plan = build_visual_state_review_plan(embeddings)
    review = seal_visual_state_review(
        {
            "plan_sha256": plan["artifact_sha256"],
            "reviewer": "codex_offline_visual_review",
            "reviews": [
                {
                    "review_id": "visual-state-0001",
                    "visual_state": "live_play",
                    "confidence": "high",
                    "notes": "Resolved.",
                }
            ],
        },
        plan=plan,
    )
    corrections = derive_visual_state_label_corrections(
        embedding_artifact=embeddings,
        plan=plan,
        review=review,
    )

    with pytest.raises(ValueError, match="unresolved"):
        build_visual_state_followup_review_plan(
            plan,
            corrections=corrections,
        )

    corrections["decisions"][0]["corrected_state"] = None
    with pytest.raises(ValueError, match="hash mismatch"):
        build_visual_state_followup_review_plan(
            plan,
            corrections=corrections,
        )
