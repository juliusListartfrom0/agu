from __future__ import annotations

import pytest

from app.analysis.independent_shot_auxiliary_transfer import (
    seal_transfer_artifact,
    summarize_event_scores,
    verify_label_hidden_plan,
    verify_transfer_artifact,
)


def test_summarize_event_scores_is_label_free_and_bounded() -> None:
    summary = summarize_event_scores(
        detections_by_frame=[
            {"frame": 10, "detections": [{"score": 0.81}]},
            {"frame": 20, "detections": []},
            {"frame": 30, "detections": [{"score": 0.49}, {"score": 0.73}]},
        ],
        start_frame=10,
        end_frame=31,
    )

    assert summary["sampled_frame_count"] == 3
    assert summary["positive_frame_count"] == 2
    assert summary["presence_fraction"] == pytest.approx(2 / 3)
    assert summary["max_detector_score"] == pytest.approx(0.81)
    assert summary["transfer_score"] == pytest.approx((0.81 * (2 / 3)) ** 0.5)
    assert "event_present" not in summary


def test_summarize_event_scores_rejects_out_of_bounds_and_invalid_scores() -> None:
    with pytest.raises(ValueError, match="outside"):
        summarize_event_scores(
            detections_by_frame=[{"frame": 31, "detections": []}],
            start_frame=10,
            end_frame=31,
        )
    with pytest.raises(ValueError, match="within"):
        summarize_event_scores(
            detections_by_frame=[{"frame": 10, "detections": [{"score": 1.1}]}],
            start_frame=10,
            end_frame=31,
        )


def test_transfer_artifact_is_fail_closed_and_hash_bound() -> None:
    artifact = seal_transfer_artifact(
        {
            "plan": {"plan_sha256": "a" * 64},
            "model": {"checkpoint_sha256": "b" * 64},
            "transfer_predictions": [
                {
                    "source_video_sha256": "c" * 64,
                    "candidate_bundle_sha256": "d" * 64,
                    "event_id": "event-1",
                    "transfer_score": 0.25,
                }
            ],
        }
    )
    assert artifact["auxiliary_oof_evidence_available"] is False
    assert artifact["oof_predictions"] == []
    assert verify_transfer_artifact(artifact)["artifact_sha256"] == artifact["artifact_sha256"]

    tampered = dict(artifact)
    tampered["transfer_predictions"] = [dict(artifact["transfer_predictions"][0], transfer_score=0.9)]
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_transfer_artifact(tampered)


def test_label_hidden_plan_rejects_target_field() -> None:
    plan = {
        "schema_version": "agu.independent-shot-vlm-plan.v1",
        "purpose": "offline_development_model_screening",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "examples": [
            {
                "source_video_sha256": "a" * 64,
                "candidate_bundle_sha256": "b" * 64,
                "event_id": "event-1",
                "start_frame": 0,
                "end_frame": 10,
                "source_fps": 30.0,
            }
        ],
    }
    from app.analysis.independent_shot_vlm import canonical_sha256

    plan["plan_sha256"] = canonical_sha256(plan, hash_field="plan_sha256")
    assert verify_label_hidden_plan(plan)["plan_sha256"] == plan["plan_sha256"]
    leaked = dict(plan, examples=[dict(plan["examples"][0], event_present=True)])
    leaked["plan_sha256"] = canonical_sha256(leaked, hash_field="plan_sha256")
    with pytest.raises(ValueError, match="leaks labels"):
        verify_label_hidden_plan(leaked)

