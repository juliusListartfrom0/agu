from __future__ import annotations

import copy
import hashlib
import json

import pytest

from app.analysis.muvs_local_visual import (
    choose_muvs_local_view,
    screen_muvs_local_visual,
    seal_muvs_local_visual_embeddings,
    verify_muvs_local_visual_embeddings,
    verify_muvs_local_visual_screen,
)


def _artifact(*, hash_seed: int = 0) -> dict:
    examples = []
    for event_index in range(4):
        for label_index, state in enumerate(("live_play", "free_throw_setup")):
            sample_index = event_index * 2 + label_index
            value = -2.0 if label_index == 0 else 2.0
            views = [
                [
                    [value + view_index * 0.01] * 384
                    for view_index in range(3)
                ]
                for _ in range(2)
            ]
            examples.append(
                {
                    "sample_id": f"muvs-state-{sample_index + 1:04d}",
                    "event_id": f"event-{event_index}",
                    "split": "train" if event_index < 3 else "test",
                    "state": state,
                    "frame_before_sha256": hashlib.sha256(
                        f"{hash_seed}:{sample_index}:before".encode()
                    ).hexdigest(),
                    "frame_after_sha256": hashlib.sha256(
                        f"{hash_seed}:{sample_index}:after".encode()
                    ).hexdigest(),
                    "selected_view_indices": [label_index, label_index],
                    "embeddings": views,
                }
            )
    return seal_muvs_local_visual_embeddings(
        {
            "detection_artifact_sha256": "1" * 64,
            "backbone": "facebook/dinov2-small",
            "backbone_revision": (
                "ed25f3a31f01632728cabb09d1542f84ab7b0056"
            ),
            "backbone_sha256": "2" * 64,
            "embedding_dimension": 384,
            "views": [
                {"name": "left_65", "x_range": [0.0, 0.65]},
                {"name": "center_65", "x_range": [0.175, 0.825]},
                {"name": "right_65", "x_range": [0.35, 1.0]},
            ],
            "examples": examples,
        }
    )


def test_muvs_local_visual_embeddings_are_hash_bound_and_offline() -> None:
    artifact = _artifact()

    assert artifact["runtime_consumable"] is False
    assert artifact["codex_runtime_answer_used"] is False
    assert verify_muvs_local_visual_embeddings(artifact) == artifact

    tampered = copy.deepcopy(artifact)
    tampered["examples"][0]["embeddings"][0][0][0] = 9.0
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_muvs_local_visual_embeddings(tampered)


def test_muvs_local_visual_screen_uses_grouped_oof_pca() -> None:
    artifact = _artifact()

    screen = screen_muvs_local_visual(artifact)

    assert screen["protocol"]["grouping"] == "leave_one_event_out"
    assert screen["protocol"]["pca_components"] == 8
    assert screen["protocol"]["candidate_features"] == [
        "selected_context",
        "selected_directed_temporal",
        "symmetric_context",
    ]
    directed = next(
        candidate
        for candidate in screen["candidates"]
        if candidate["feature"] == "selected_directed_temporal"
    )
    assert directed["feature_dimension"] == 3 * 384
    assert screen["resolved_examples"] == 8
    assert screen["best_candidate"]["balanced_accuracy"] == pytest.approx(1.0)
    assert screen["accepted"] is True
    assert verify_muvs_local_visual_screen(screen) == screen


def test_muvs_local_visual_verifier_accepts_legacy_two_candidate_screen() -> None:
    screen = screen_muvs_local_visual(_artifact())
    legacy = copy.deepcopy(screen)
    legacy.pop("artifact_sha256")
    legacy["protocol"]["candidate_features"] = [
        "selected_context",
        "symmetric_context",
    ]
    legacy["candidates"] = [
        row
        for row in legacy["candidates"]
        if row["feature"] != "selected_directed_temporal"
    ]
    legacy["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            legacy,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    assert verify_muvs_local_visual_screen(legacy) == legacy


def test_muvs_local_visual_screen_combines_and_rejects_conflicts() -> None:
    first = _artifact()
    second = _artifact(hash_seed=1)

    combined = screen_muvs_local_visual([first, second])
    assert combined["resolved_examples"] == 16

    deduplicated = screen_muvs_local_visual([first, first])
    assert deduplicated["resolved_examples"] == 8

    conflict = copy.deepcopy(first)
    conflict["examples"][0]["state"] = "free_throw_setup"
    conflict = seal_muvs_local_visual_embeddings(conflict)
    with pytest.raises(ValueError, match="duplicate MUVS local views disagree"):
        screen_muvs_local_visual([first, conflict])


def test_muvs_local_view_selection_prefers_rim_then_player_density() -> None:
    rim_frame = {
        "detections": [
            {
                "object_type": "rim",
                "confidence": 0.9,
                "bbox": [0.8, 0.1, 0.9, 0.2],
            },
            {
                "object_type": "player",
                "confidence": 0.8,
                "bbox": [0.1, 0.4, 0.2, 0.8],
            },
        ]
    }
    dense_center_frame = {
        "detections": [
            {
                "object_type": "player",
                "confidence": 0.8,
                "bbox": [0.45, 0.4, 0.55, 0.8],
            }
        ]
    }

    assert choose_muvs_local_view(rim_frame) == 2
    assert choose_muvs_local_view(dense_center_frame) == 1
