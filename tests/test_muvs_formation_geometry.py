from __future__ import annotations

import copy
import hashlib

import pytest

from app.analysis.muvs_formation_geometry import (
    screen_muvs_formation_geometry,
    seal_muvs_formation_detections,
    verify_muvs_formation_detections,
    verify_muvs_formation_geometry_screen,
)


def _frame(*, free_throw: bool) -> dict:
    detections = [
        {
            "object_type": "rim",
            "confidence": 0.9,
            "bbox": [0.47, 0.18, 0.53, 0.24],
        }
    ]
    if free_throw:
        player_centers = (
            (0.42, 0.48),
            (0.47, 0.51),
            (0.53, 0.50),
            (0.58, 0.48),
            (0.49, 0.64),
            (0.51, 0.65),
        )
    else:
        player_centers = (
            (0.10, 0.70),
            (0.25, 0.52),
            (0.40, 0.78),
            (0.62, 0.68),
            (0.78, 0.52),
            (0.90, 0.76),
        )
    for center_x, foot_y in player_centers:
        detections.append(
            {
                "object_type": "player",
                "confidence": 0.8,
                "bbox": [
                    center_x - 0.025,
                    foot_y - 0.16,
                    center_x + 0.025,
                    foot_y,
                ],
            }
        )
    return {"width": 1280, "height": 720, "detections": detections}


def _artifact(*, hash_seed: int = 0) -> dict:
    examples = []
    for event_index in range(4):
        for label_index, state in enumerate(("live_play", "free_throw_setup")):
            sample_index = event_index * 2 + label_index
            frame_before_sha256 = hashlib.sha256(
                f"{hash_seed}:{sample_index}:before".encode()
            ).hexdigest()
            frame_after_sha256 = hashlib.sha256(
                f"{hash_seed}:{sample_index}:after".encode()
            ).hexdigest()
            examples.append(
                {
                    "sample_id": f"muvs-state-{sample_index + 1:04d}",
                    "event_id": f"event-{event_index}",
                    "split": "train" if event_index < 3 else "test",
                    "state": state,
                    "frame_before_sha256": frame_before_sha256,
                    "frame_after_sha256": frame_after_sha256,
                    "frames": [
                        _frame(free_throw=bool(label_index)),
                        _frame(free_throw=bool(label_index)),
                    ],
                }
            )
    return seal_muvs_formation_detections(
        {
            "plan_sha256": "1" * 64,
            "frames_sha256": "2" * 64,
            "review_sha256": "3" * 64,
            "detector": {
                "backend": "ultralytics_yolo",
                "model_name": "E-BARD/BODD_yolov8n_0001",
                "model_sha256": "4" * 64,
                "class_names": [
                    "basketball",
                    "rim",
                    "player",
                    "referee",
                ],
                "image_size": 1280,
                "confidence": 0.1,
            },
            "examples": examples,
        }
    )


def test_muvs_formation_detections_are_training_only_and_hash_bound() -> None:
    artifact = _artifact()

    assert artifact["runtime_consumable"] is False
    assert artifact["codex_runtime_answer_used"] is False
    assert verify_muvs_formation_detections(artifact) == artifact

    tampered = copy.deepcopy(artifact)
    tampered["examples"][0]["frames"][0]["detections"][0]["bbox"][0] = 0.46
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_muvs_formation_detections(tampered)


def test_muvs_geometry_screen_uses_event_grouped_oof_predictions() -> None:
    detections = _artifact()

    screen = screen_muvs_formation_geometry(detections)

    assert screen["runtime_consumable"] is False
    assert screen["protocol"]["grouping"] == "leave_one_event_out"
    assert screen["resolved_examples"] == 8
    assert screen["event_count"] == 4
    assert screen["best_candidate"]["balanced_accuracy"] == pytest.approx(1.0)
    assert screen["best_candidate"]["worst_positive_event_recall"] == pytest.approx(1.0)
    assert screen["accepted"] is True
    assert verify_muvs_formation_geometry_screen(screen) == screen

    tampered = copy.deepcopy(screen)
    tampered["accepted"] = False
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_muvs_formation_geometry_screen(tampered)


def test_muvs_geometry_screen_combines_and_deduplicates_sources() -> None:
    first = _artifact()
    second = _artifact(hash_seed=1)

    screen = screen_muvs_formation_geometry([first, second])
    assert screen["resolved_examples"] == 16
    assert screen["detection_artifact_sha256s"] == [
        first["artifact_sha256"],
        second["artifact_sha256"],
    ]

    deduplicated = screen_muvs_formation_geometry([first, first])
    assert deduplicated["resolved_examples"] == 8

    conflict = copy.deepcopy(first)
    conflict["examples"][0]["state"] = "free_throw_setup"
    conflict = seal_muvs_formation_detections(conflict)
    with pytest.raises(ValueError, match="duplicate MUVS source frames disagree"):
        screen_muvs_formation_geometry([first, conflict])


def test_muvs_formation_detection_validation_rejects_invalid_geometry() -> None:
    payload = copy.deepcopy(_artifact())
    payload["examples"][0]["frames"][0]["detections"][0]["bbox"] = [
        -0.1,
        0.2,
        0.3,
        0.4,
    ]

    with pytest.raises(ValueError, match="invalid MUVS formation detection"):
        seal_muvs_formation_detections(payload)
