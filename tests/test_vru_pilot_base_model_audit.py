from __future__ import annotations

from scripts.audit_vru_pilot_base_model import (
    binary_metrics,
    rank_auc,
    summarize_diagnostic,
)


def test_binary_metrics_uses_fixed_threshold_without_promotion() -> None:
    result = binary_metrics([True, False, True, False], [0.8, 0.7, 0.2, 0.1], 0.5)
    assert result == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "tn": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }


def test_rank_auc_handles_ties_deterministically() -> None:
    assert rank_auc([True, False], [0.5, 0.5]) == 0.5
    assert rank_auc([True, False], [0.9, 0.1]) == 1.0


def test_summary_reports_argmax_and_fixed_threshold_metrics() -> None:
    rows = [
        {"event_present": True, "shoot_probability": 0.9, "action": "shoot"},
        {"event_present": False, "shoot_probability": 0.8, "action": "pass"},
        {"event_present": False, "shoot_probability": 0.1, "action": "dribble"},
    ]
    result = summarize_diagnostic(rows, threshold=0.5)
    assert result["examples"] == 3
    assert result["positive_examples"] == 1
    assert result["negative_examples"] == 2
    assert result["fixed_threshold"]["tp"] == 1
    assert result["fixed_threshold"]["fp"] == 1
    assert result["argmax_shoot"]["tp"] == 1
    assert result["argmax_shoot"]["fp"] == 0
    assert result["rank_auc"] == 1.0
