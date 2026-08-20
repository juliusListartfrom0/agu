from __future__ import annotations

import pytest

from app.analysis.independent_shot_auxiliary_transfer import seal_transfer_artifact
from app.analysis.independent_shot_vlm import seal_independent_shot_vlm_predictions
from app.analysis.independent_shot_vlm_consistency import (
    screen_cross_model_consistency,
    verify_consistency_artifact,
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


def test_same_plan_consistency_is_label_free_and_sealed() -> None:
    plan_sha256 = "d" * 64
    vlm = seal_independent_shot_vlm_predictions(
        {
            "plan_sha256": plan_sha256,
            "predictions": [
                _vlm_row("e1", state="live_field_goal"),
                _vlm_row("e2", state="not_field_goal"),
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
        ],
    )

    result = screen_cross_model_consistency(
        vlm_predictions=vlm,
        auxiliary_transfer=transfer,
    )

    assert result["contracts"]["plan_contract_hashes_equal"] is True
    assert result["summary"] == {
        "event_count": 2,
        "vlm_live_count": 1,
        "transfer_above_default_threshold_count": 1,
        "cross_model_agreement_count": 2,
        "cross_model_agreement_rate": 1.0,
        "transfer_score_mean": 0.5,
        "labels_read": False,
        "oof_evidence": False,
    }
    assert result["decision"]["promotion_eligible"] is False
    assert result["decision"]["fusion_output_emitted"] is False
    assert result["runtime_consumable"] is False
    assert result["training_eligible"] is False
    assert all("event_present" not in row for row in result["rows"])
    assert verify_consistency_artifact(result)["artifact_sha256"] == result["artifact_sha256"]


def test_consistency_rejects_missing_or_duplicate_cross_model_keys() -> None:
    plan_sha256 = "e" * 64
    vlm = seal_independent_shot_vlm_predictions(
        {
            "plan_sha256": plan_sha256,
            "predictions": [_vlm_row("e1", state="live_field_goal")],
        }
    )
    transfer = _transfer(
        plan_sha256=plan_sha256,
        rows=[
            {
                "source_video_sha256": "a" * 64,
                "candidate_bundle_sha256": "b" * 64,
                "event_id": "different",
                "transfer_score": 0.8,
            }
        ],
    )

    with pytest.raises(ValueError, match="exactly cover"):
        screen_cross_model_consistency(
            vlm_predictions=vlm,
            auxiliary_transfer=transfer,
        )


def test_plan_mismatch_is_retained_as_non_promotable_evidence() -> None:
    vlm = seal_independent_shot_vlm_predictions(
        {
            "plan_sha256": "f" * 64,
            "predictions": [_vlm_row("e1", state="live_field_goal")],
        }
    )
    transfer = _transfer(
        plan_sha256="0" * 64,
        rows=[
            {
                "source_video_sha256": "a" * 64,
                "candidate_bundle_sha256": "b" * 64,
                "event_id": "e1",
                "transfer_score": 0.8,
            }
        ],
    )

    result = screen_cross_model_consistency(
        vlm_predictions=vlm,
        auxiliary_transfer=transfer,
    )

    assert result["contracts"]["plan_contract_hashes_equal"] is False
    assert result["decision"]["promotion_eligible"] is False
    assert "different frozen plan" in result["decision"]["reason"]
