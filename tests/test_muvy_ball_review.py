from __future__ import annotations

import copy

import pytest

from app.analysis.muvy_ball_review import (
    build_muvy_ball_review_decisions,
    is_small_ball_candidate,
    parse_muvy_detection_line,
    seal_muvy_ball_review,
    seal_muvy_ball_review_plan,
    verify_muvy_ball_review,
    verify_muvy_ball_review_plan,
)


def _plan() -> dict:
    return seal_muvy_ball_review_plan(
        {
            "source_manifest_sha256": "1" * 64,
            "source_record_url": "https://zenodo.org/records/13883315",
            "source_license": "CC-BY-4.0",
            "candidates": [
                {
                    "candidate_id": "muvy-ball-0001",
                    "video_path": "event/cam/video.mp4",
                    "video_sha256": "2" * 64,
                    "annotation_sha256": "3" * 64,
                    "frame_id": 7,
                    "frame_index": 6,
                    "frame_width": 1280,
                    "frame_height": 720,
                    "frame_sha256": "4" * 64,
                    "confidence": 0.01,
                    "bbox": [100.0, 200.0, 120.0, 220.0],
                }
            ],
        }
    )


def test_muvy_detection_parser_and_small_ball_filter() -> None:
    parsed = parse_muvy_detection_line(
        "1 sports ball 0.004 521 246 543 261"
    )

    assert parsed == {
        "frame_id": 1,
        "object_class": "sports ball",
        "confidence": 0.004,
        "bbox": [521.0, 246.0, 543.0, 261.0],
    }
    assert is_small_ball_candidate(
        parsed["bbox"],
        frame_width=1280,
        frame_height=720,
    )
    assert not is_small_ball_candidate(
        [0.0, 0.0, 500.0, 500.0],
        frame_width=1280,
        frame_height=720,
    )


def test_muvy_review_requires_exact_hash_bound_coverage() -> None:
    plan = _plan()
    verified_plan = verify_muvy_ball_review_plan(plan)
    review = seal_muvy_ball_review(
        {
            "plan_sha256": plan["artifact_sha256"],
            "reviewer": "codex_offline_source_annotation",
            "decisions": [
                {
                    "candidate_id": "muvy-ball-0001",
                    "decision": "valid_ball",
                    "notes": "Visible basketball inside the proposed box.",
                }
            ],
        },
        plan=plan,
    )

    assert verified_plan == plan
    assert verify_muvy_ball_review(review, plan=plan) == review
    assert review["runtime_consumable"] is False
    assert review["codex_runtime_answer_used"] is False

    tampered = copy.deepcopy(review)
    tampered["decisions"][0]["decision"] = "false_positive"
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_muvy_ball_review(tampered, plan=plan)

    with pytest.raises(ValueError, match="exactly cover"):
        seal_muvy_ball_review(
            {
                "plan_sha256": plan["artifact_sha256"],
                "reviewer": "codex_offline_source_annotation",
                "decisions": [],
            },
            plan=plan,
        )


def test_build_muvy_ball_review_decisions_expands_ranges_and_defaults_false() -> None:
    candidate_ids = [f"muvy-ball-{index:04d}" for index in range(1, 7)]

    decisions = build_muvy_ball_review_decisions(
        candidate_ids,
        valid_ranges=["1-2", "5"],
        uncertain_ranges=["4"],
    )

    assert [row["decision"] for row in decisions] == [
        "valid_ball",
        "valid_ball",
        "false_positive",
        "uncertain",
        "valid_ball",
        "false_positive",
    ]
    with pytest.raises(ValueError, match="overlap"):
        build_muvy_ball_review_decisions(
            candidate_ids,
            valid_ranges=["1-3"],
            uncertain_ranges=["3-4"],
        )
    with pytest.raises(ValueError, match="outside"):
        build_muvy_ball_review_decisions(
            candidate_ids,
            valid_ranges=["7"],
            uncertain_ranges=[],
        )
