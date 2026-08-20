from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.analysis.independent_auxiliary_external_calibration import (
    build_screen_score_rows,
    evaluate_labeled_scores,
    select_external_score_threshold,
)


def test_build_screen_score_rows_reuses_uncalibrated_transfer_score_formula() -> None:
    screen = {
        "video_sha256": "a" * 64,
        "rows": [
            {
                "event_id": "event-1",
                "start_frame": 10,
                "end_frame": 40,
                "sampled_frames": [10, 20, 30],
                "detection_frames": [10, 30],
                "detections": [
                    {"frame": 10, "score": 0.81},
                    {"frame": 30, "score": 0.49},
                ],
                "event_present": True,
            }
        ],
    }

    rows = build_screen_score_rows(screen)

    assert rows[0]["source_video_sha256"] == "a" * 64
    assert rows[0]["score"] == pytest.approx((0.81 * (2 / 3)) ** 0.5)
    assert rows[0]["event_present"] is True


def test_external_threshold_is_selected_from_calibration_groups_only() -> None:
    rows = [
        {"source_video_sha256": "a" * 64, "event_id": "a-pos", "score": 0.9, "event_present": True},
        {"source_video_sha256": "a" * 64, "event_id": "a-neg", "score": 0.1, "event_present": False},
        {"source_video_sha256": "b" * 64, "event_id": "b-pos", "score": 0.8, "event_present": True},
        {"source_video_sha256": "b" * 64, "event_id": "b-neg", "score": 0.2, "event_present": False},
    ]

    selection = select_external_score_threshold(rows, minimum_recall=1.0)

    assert selection["training_groups"] == ["a" * 64, "b" * 64]
    assert selection["threshold"] == pytest.approx(0.8)
    assert selection["precision"] == pytest.approx(1.0)
    assert selection["recall"] == pytest.approx(1.0)


def test_evaluate_labeled_scores_rejects_missing_boolean_labels() -> None:
    with pytest.raises(ValueError, match="boolean"):
        evaluate_labeled_scores(
            [{"source_video_sha256": "a" * 64, "event_id": "e", "score": 0.5, "event_present": "yes"}],
            threshold=0.5,
        )


def test_sealed_external_calibration_keeps_target_labels_out_of_predictions() -> None:
    path = Path(
        "analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v6/"
        "independent_auxiliary_external_calibration_v1.json"
    )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    claimed = artifact.pop("artifact_sha256")
    computed = hashlib.sha256(
        json.dumps(artifact, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert claimed == computed
    assert artifact["runtime_consumable"] is False
    assert artifact["oof_predictions"] == []
    assert all(
        "event_present" not in row
        for row in artifact["target"]["label_free_predictions"]
    )
