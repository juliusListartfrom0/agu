from __future__ import annotations

from app.analysis.ball_candidate_vlm import (
    evaluate_ball_candidate_vlm,
    parse_ball_candidate_vlm_response,
    select_vlm_probe_candidates,
)
from scripts.run_ball_candidate_vlm import configure_image_processor


def test_select_vlm_probe_candidates_is_label_free_and_band_balanced() -> None:
    plan = {
        "sampling": {"bands": [[0.1, 0.2], [0.2, 0.3]]},
        "candidates": [
                {
                    "candidate_id": f"c:{index}",
                    "detection_id": f"d:{index}",
                    "frame": index,
                    "confidence": confidence,
                }
            for index, confidence in enumerate(
                [0.11, 0.12, 0.13, 0.14, 0.21, 0.22, 0.23, 0.24]
            )
        ],
    }

    selected = select_vlm_probe_candidates(plan, samples_per_band=2)

    assert [row["candidate_id"] for row in selected] == [
        "c:0",
        "c:3",
        "c:4",
        "c:7",
    ]
    assert all("decision" not in row and "label" not in row for row in selected)


def test_parse_ball_candidate_vlm_response_fails_closed() -> None:
    parsed = parse_ball_candidate_vlm_response(
        'prefix {"ball_state":"ball","confidence":1.7,"reason":"orange sphere"}'
    )
    invalid = parse_ball_candidate_vlm_response("I cannot return JSON")

    assert parsed == {
        "ball_state": "ball",
        "confidence": 1.0,
        "reason": "orange sphere",
        "available": True,
    }
    assert invalid["ball_state"] == "uncertain"
    assert invalid["confidence"] == 0.0
    assert invalid["available"] is False


def test_evaluate_ball_candidate_vlm_uses_uncertain_bounds_and_band_population() -> None:
    perception = {
        "artifact_sha256": "perception",
        "detections": [
            *[
                {"detection_id": f"low:{index}", "confidence": 0.15}
                for index in range(90)
            ],
            *[
                {"detection_id": f"high:{index}", "confidence": 0.25}
                for index in range(10)
            ],
        ],
    }
    plan = {
        "artifact_sha256": "plan",
        "perception_artifact_sha256": "perception",
        "sampling": {"bands": [[0.1, 0.2], [0.2, 0.3]]},
        "candidates": [
            {
                "candidate_id": "low-valid",
                "confidence_band": [0.1, 0.2],
            },
            {
                "candidate_id": "low-false",
                "confidence_band": [0.1, 0.2],
            },
            {
                "candidate_id": "high-valid",
                "confidence_band": [0.2, 0.3],
            },
            {
                "candidate_id": "high-uncertain",
                "confidence_band": [0.2, 0.3],
            },
        ],
    }
    review = {
        "plan_sha256": "plan",
        "runtime_consumable": False,
        "decisions": [
            {"candidate_id": "low-valid", "decision": "valid_ball"},
            {"candidate_id": "low-false", "decision": "false_positive"},
            {"candidate_id": "high-valid", "decision": "valid_ball"},
            {"candidate_id": "high-uncertain", "decision": "uncertain"},
        ],
    }
    predictions = {
        "review_plan_sha256": "plan",
        "predictions": [
            {"candidate_id": "low-valid", "ball_state": "ball"},
            {"candidate_id": "low-false", "ball_state": "ball"},
            {"candidate_id": "high-valid", "ball_state": "ball"},
            {"candidate_id": "high-uncertain", "ball_state": "not_ball"},
        ],
    }

    result = evaluate_ball_candidate_vlm(
        perception=perception,
        plan=plan,
        review=review,
        predictions=predictions,
    )

    assert result["population_by_band"] == {"0.10-0.20": 90, "0.20-0.30": 10}
    assert result["metrics"]["lower"]["precision"] == 50.0 / 95.0
    assert result["metrics"]["lower"]["recall"] == 1.0
    assert result["metrics"]["upper"]["recall"] == 50.0 / 55.0


def test_configure_image_processor_applies_bounded_pixel_budget() -> None:
    class _ImageProcessor:
        min_pixels = 12_544
        max_pixels = 940_800
        size = {"shortest_edge": 12_544, "longest_edge": 940_800}

    class _Processor:
        image_processor = _ImageProcessor()

    configure_image_processor(_Processor(), max_pixels=301_056)

    assert _Processor.image_processor.max_pixels == 301_056
    assert _Processor.image_processor.size["longest_edge"] == 301_056
