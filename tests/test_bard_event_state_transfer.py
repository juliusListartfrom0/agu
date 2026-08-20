from __future__ import annotations

import json

import numpy as np
import pytest

from app.analysis.bard_event_state_transfer import (
    evaluate_bard_transfer_predictions,
    normalized_frame_indexes,
    seal_bard_frame_embeddings,
    sequence_features,
    train_bard_transfer_predictor,
    verify_bard_frame_embeddings,
)
from app.analysis.causal_phase_dense_temporal import (
    seal_dense_causal_frame_embeddings,
)
from app.analysis.causal_shot_phase_review import (
    CAUSAL_PHASE_OFFSETS_SECONDS,
)


def test_normalized_frame_indexes_are_bounded_and_deterministic() -> None:
    assert normalized_frame_indexes(101, positions=5) == [5, 28, 50, 72, 95]
    assert normalized_frame_indexes(101, positions=5) == normalized_frame_indexes(
        101,
        positions=5,
    )
    with pytest.raises(ValueError, match="frames"):
        normalized_frame_indexes(4, positions=5)


def test_sequence_features_preserve_fixed_dimension() -> None:
    values = np.arange(24, dtype=np.float32).reshape(6, 4)

    assert sequence_features(values, representation="mean").shape == (4,)
    assert sequence_features(values, representation="mean_std").shape == (8,)
    assert sequence_features(values, representation="quarters").shape == (16,)
    with pytest.raises(ValueError, match="representation"):
        sequence_features(values, representation="unknown")


def _bard_embeddings() -> dict[str, object]:
    examples = []
    for group_index in range(4):
        for label, state in ((0, "live_field_goal"), (1, "free_throw")):
            base = float(label * 4 + group_index * 0.01)
            examples.append(
                {
                    "clip_id": f"clip-{group_index}-{label}",
                    "game_id": f"00223000{group_index:02d}",
                    "event_state": state,
                    "shot_outcome": "made",
                    "frame_indexes": [0, 1, 2, 3],
                    "embeddings": [
                        [base + position * 0.01] * 4
                        for position in range(4)
                    ],
                }
            )
    return seal_bard_frame_embeddings(
        {
            "purpose": "event_state_pretraining_only",
            "codex_runtime_answer_used": False,
            "subset_sha256": "a" * 64,
            "backbone": "fixture/mobilenet",
            "backbone_sha256": "b" * 64,
            "embedding_dimension": 4,
            "positions": 4,
            "sampling_protocol": "normalized_full_clip_5_to_95_percent_v1",
            "examples": examples,
        }
    )


def _target_embeddings() -> dict[str, object]:
    sources = ["c" * 64, "d" * 64]
    examples = []
    for index, label in enumerate((0, 1, 0, 1)):
        source = sources[index // 2]
        base = float(label * 4 + index * 0.01)
        examples.append(
            {
                "phase_review_id": f"phase-{index}",
                "source_video_sha256": source,
                "candidate_bundle_sha256": f"{index + 1:064x}",
                "event_id": f"event-{index}",
                "frame_indexes": list(range(len(CAUSAL_PHASE_OFFSETS_SECONDS))),
                "embeddings": [
                    [base + position * 0.001] * 4
                    for position in range(len(CAUSAL_PHASE_OFFSETS_SECONDS))
                ],
            }
        )
    return seal_dense_causal_frame_embeddings(
        {
            "purpose": "training_only_dense_causal_frame_embeddings",
            "codex_runtime_answer_used": False,
            "review_plan_sha256": "e" * 64,
            "sheet_manifest_sha256": "f" * 64,
            "source_video_sha256s": sources,
            "sealed_blind_video_sha256s": ["9" * 64],
            "backbone": "fixture/mobilenet",
            "backbone_sha256": "b" * 64,
            "embedding_dimension": 4,
            "anchor_offsets_seconds": list(CAUSAL_PHASE_OFFSETS_SECONDS),
            "examples": examples,
        }
    )


def test_bard_embedding_artifact_is_hash_bound() -> None:
    artifact = _bard_embeddings()

    assert verify_bard_frame_embeddings(json.loads(json.dumps(artifact))) == artifact
    artifact["examples"][0]["embeddings"][0][0] = 99
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_bard_frame_embeddings(artifact)


def test_transfer_predictions_are_label_free_and_target_exact() -> None:
    artifact = train_bard_transfer_predictor(
        bard_embedding_artifact=_bard_embeddings(),
        target_embedding_artifact=_target_embeddings(),
        configurations=[
            {"representation": "mean", "c": 0.1},
            {"representation": "mean_std", "c": 0.1},
        ],
        seed=13,
    )

    assert artifact["runtime_consumable"] is False
    assert len(artifact["predictions"]) == 4
    assert all("target" not in row for row in artifact["predictions"])
    assert {row["phase_review_id"] for row in artifact["predictions"]} == {
        "phase-0",
        "phase-1",
        "phase-2",
        "phase-3",
    }
    assert artifact["source_oof_metrics"]["balanced_accuracy"] >= 0.99


def test_transfer_evaluation_requires_exact_delayed_label_join() -> None:
    predictions = train_bard_transfer_predictor(
        bard_embedding_artifact=_bard_embeddings(),
        target_embedding_artifact=_target_embeddings(),
        configurations=[{"representation": "mean", "c": 0.1}],
        seed=17,
    )
    references = [
        {
            "phase_review_id": f"phase-{index}",
            "source_video_sha256": (
                "c" * 64 if index < 2 else "d" * 64
            ),
            "event_id": f"event-{index}",
            "target": label,
        }
        for index, label in enumerate((0, 1, 0, 1))
    ]

    evaluation = evaluate_bard_transfer_predictions(
        prediction_artifact=predictions,
        reference_predictions=references,
        reference_artifact_sha256="8" * 64,
        acceptance_threshold=0.85,
    )

    assert evaluation["metrics"]["balanced_accuracy"] >= 0.99
    assert evaluation["accepted"] is True

    with pytest.raises(ValueError, match="exactly"):
        evaluate_bard_transfer_predictions(
            prediction_artifact=predictions,
            reference_predictions=references[:-1],
            reference_artifact_sha256="8" * 64,
            acceptance_threshold=0.85,
        )
