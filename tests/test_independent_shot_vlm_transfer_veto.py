from __future__ import annotations

import pytest

from app.analysis.independent_shot_auxiliary_transfer import seal_transfer_artifact
from app.analysis.independent_shot_vlm import seal_independent_shot_vlm_predictions
from app.analysis.independent_shot_vlm_transfer_veto import (
    build_label_free_transfer_veto,
    evaluate_transfer_veto,
    verify_transfer_veto_artifact,
)


def _vlm_row(event_id: str, *, state: str) -> dict[str, object]:
    return {
        "source_video_sha256": "a" * 64,
        "candidate_bundle_sha256": "b" * 64,
        "event_id": event_id,
        "field_goal_state": state,
        "confidence": 0.9,
        "observables": {},
        "reason": "offline test",
    }


def _transfer(*, plan_sha256: str, rows: list[dict[str, object]]) -> dict[str, object]:
    return seal_transfer_artifact(
        {
            "plan": {
                "path": "plan.json",
                "plan_sha256": plan_sha256,
                "example_count": len(rows),
                "labels_or_review_notes_exposed_to_model": False,
            },
            "model": {"checkpoint_sha256": "c" * 64},
            "transfer_predictions": rows,
        }
    )


def test_label_free_transfer_veto_is_conservative_and_sealed() -> None:
    plan_sha256 = "d" * 64
    vlm = seal_independent_shot_vlm_predictions(
        {
            "plan_sha256": plan_sha256,
            "predictions": [
                _vlm_row("e1", state="live_field_goal"),
                _vlm_row("e2", state="live_field_goal"),
                _vlm_row("e3", state="not_field_goal"),
            ],
        }
    )
    transfer = _transfer(
        plan_sha256=plan_sha256,
        rows=[
            {
                "source_video_sha256": "a" * 64,
                "candidate_bundle_sha256": "b" * 64,
                "event_id": "e1",
                "transfer_score": 0.8,
            },
            {
                "source_video_sha256": "a" * 64,
                "candidate_bundle_sha256": "b" * 64,
                "event_id": "e2",
                "transfer_score": 0.2,
            },
            {
                "source_video_sha256": "a" * 64,
                "candidate_bundle_sha256": "b" * 64,
                "event_id": "e3",
                "transfer_score": 0.9,
            },
        ],
    )

    artifact = build_label_free_transfer_veto(
        vlm_predictions=vlm,
        auxiliary_transfer=transfer,
        transfer_threshold=0.5,
    )

    assert artifact["runtime_consumable"] is False
    assert artifact["training_eligible"] is False
    assert artifact["decision"]["promotion_eligible"] is False
    assert artifact["decision"]["fusion_output_emitted"] is False
    assert artifact["summary"] == {
        "event_count": 3,
        "vlm_live_count": 2,
        "transfer_above_threshold_count": 2,
        "accepted_live_count": 1,
        "vetoed_live_count": 1,
        "unknown_count": 2,
        "transfer_score_mean": pytest.approx(0.6333333333333333),
        "labels_read": False,
        "oof_evidence": False,
    }
    rows = {row["event_id"]: row for row in artifact["rows"]}
    assert rows["e1"]["field_goal_state"] == "live_field_goal"
    assert rows["e2"]["field_goal_state"] == "unknown"
    assert rows["e2"]["decision_reason"] == "transfer_below_fixed_threshold"
    assert rows["e3"]["field_goal_state"] == "unknown"
    assert all("event_present" not in row for row in artifact["rows"])
    assert verify_transfer_veto_artifact(artifact)["artifact_sha256"] == artifact[
        "artifact_sha256"
    ]


def test_transfer_veto_requires_exact_plan_contract() -> None:
    vlm = seal_independent_shot_vlm_predictions(
        {
            "plan_sha256": "d" * 64,
            "predictions": [_vlm_row("e1", state="live_field_goal")],
        }
    )
    transfer = _transfer(
        plan_sha256="e" * 64,
        rows=[
            {
                "source_video_sha256": "a" * 64,
                "candidate_bundle_sha256": "b" * 64,
                "event_id": "e1",
                "transfer_score": 0.8,
            }
        ],
    )

    with pytest.raises(ValueError, match="frozen plan"):
        build_label_free_transfer_veto(
            vlm_predictions=vlm,
            auxiliary_transfer=transfer,
        )


def test_transfer_veto_evaluation_reads_truth_only_after_inference() -> None:
    artifact = {
        "schema_version": "agu.independent-shot-vlm-transfer-veto.v1",
        "purpose": "offline_label_free_transfer_veto_screen",
        "runtime_consumable": False,
        "training_eligible": False,
        "codex_runtime_answer_used": False,
        "contracts": {
            "vlm_plan_sha256": "d" * 64,
            "auxiliary_plan_sha256": "d" * 64,
            "plan_contract_hashes_equal": True,
            "transfer_threshold": 0.5,
            "threshold_selected": False,
            "oof_evidence": False,
        },
        "summary": {
            "event_count": 2,
            "vlm_live_count": 2,
            "transfer_above_threshold_count": 1,
            "accepted_live_count": 1,
            "vetoed_live_count": 1,
            "unknown_count": 1,
            "transfer_score_mean": 0.5,
            "labels_read": False,
            "oof_evidence": False,
        },
        "decision": {
            "promotion_eligible": False,
            "fusion_output_emitted": False,
            "reason": "diagnostic",
        },
        "rows": [
            {
                "source_video_sha256": "a" * 64,
                "candidate_bundle_sha256": "b" * 64,
                    "event_id": "e1",
                    "field_goal_state": "live_field_goal",
                    "transfer_score": 0.8,
                    "threshold": 0.5,
                    "runtime_consumable": False,
                },
            {
                "source_video_sha256": "a" * 64,
                "candidate_bundle_sha256": "b" * 64,
                    "event_id": "e2",
                    "field_goal_state": "unknown",
                    "transfer_score": 0.2,
                    "threshold": 0.5,
                    "runtime_consumable": False,
                },
        ],
    }
    from app.analysis.independent_shot_vlm_transfer_veto import canonical_sha256

    artifact["artifact_sha256"] = canonical_sha256(artifact)
    truth = {
        ("a" * 64, "b" * 64, "e1"): True,
        ("a" * 64, "b" * 64, "e2"): False,
    }

    result = evaluate_transfer_veto(artifact, truth=truth)

    assert result["labels_used_after_inference"] is True
    assert result["promotion_eligible"] is False
    assert result["true_positive"] == 1
    assert result["true_negative"] == 1
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
