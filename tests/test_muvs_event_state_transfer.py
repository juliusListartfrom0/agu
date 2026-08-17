from __future__ import annotations

import copy

import pytest

from app.analysis.muvs_event_state_transfer import (
    screen_muvs_free_throw_source,
    seal_muvs_event_state_embeddings,
    verify_muvs_event_state_embeddings,
    verify_muvs_free_throw_source_screen,
)


def _embedding_artifact() -> dict:
    examples = []
    for group_index in range(4):
        for label_index, state in enumerate(("live_play", "free_throw_setup")):
            sample_index = group_index * 2 + label_index
            value = -2.0 if label_index == 0 else 2.0
            examples.append(
                {
                    "sample_id": f"muvs-state-{sample_index + 1:04d}",
                    "event_id": f"event-{group_index}",
                    "split": "train" if group_index < 3 else "test",
                    "state": state,
                    "frame_before_sha256": ("12345678"[sample_index] * 64),
                    "frame_after_sha256": ("89abcdef"[sample_index] * 64),
                    "embeddings": [
                        [value] * 576,
                        [value + 0.1] * 576,
                    ],
                }
            )
    return seal_muvs_event_state_embeddings(
        {
            "plan_sha256": "1" * 64,
            "frames_sha256": "2" * 64,
            "review_sha256": "3" * 64,
            "backbone": "torchvision/mobilenet_v3_small/imagenet1k_v1",
            "backbone_sha256": "4" * 64,
            "embedding_dimension": 576,
            "examples": examples,
        }
    )


def test_muvs_embeddings_are_training_only_and_hash_bound() -> None:
    artifact = _embedding_artifact()

    assert artifact["runtime_consumable"] is False
    assert artifact["codex_runtime_answer_used"] is False
    assert verify_muvs_event_state_embeddings(artifact) == artifact

    tampered = copy.deepcopy(artifact)
    tampered["examples"][0]["embeddings"][0][0] = 99.0
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_muvs_event_state_embeddings(tampered)


def test_muvs_source_screen_uses_event_grouped_oof_predictions() -> None:
    embeddings = _embedding_artifact()

    screen = screen_muvs_free_throw_source(embeddings)

    assert screen["runtime_consumable"] is False
    assert screen["protocol"]["grouping"] == "leave_one_event_out"
    assert screen["resolved_examples"] == 8
    assert screen["event_count"] == 4
    assert screen["best_candidate"]["balanced_accuracy"] == pytest.approx(1.0)
    assert screen["best_candidate"]["worst_positive_event_recall"] == (pytest.approx(1.0))
    assert screen["accepted"] is True
    assert verify_muvs_free_throw_source_screen(screen) == screen

    tampered = copy.deepcopy(screen)
    tampered["accepted"] = False
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_muvs_free_throw_source_screen(tampered)


def test_muvs_source_screen_can_combine_separately_sealed_sources() -> None:
    first = _embedding_artifact()
    second = copy.deepcopy(_embedding_artifact())
    second["plan_sha256"] = "5" * 64
    unique_hex = "9abcdef0"
    for index, row in enumerate(second["examples"]):
        row["frame_before_sha256"] = unique_hex[index] * 64
        row["frame_after_sha256"] = unique_hex[index] * 64
    second = seal_muvs_event_state_embeddings(second)

    screen = screen_muvs_free_throw_source([first, second])

    assert screen["resolved_examples"] == 16
    assert screen["event_count"] == 4
    assert screen["embedding_artifact_sha256s"] == [
        first["artifact_sha256"],
        second["artifact_sha256"],
    ]
    assert screen["accepted"] is True

    deduplicated = screen_muvs_free_throw_source([first, first])
    assert deduplicated["resolved_examples"] == 8
