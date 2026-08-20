from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from app.analysis.causal_phase_auxiliary_screen import PHASE_TARGET_NAMES
from app.analysis.causal_phase_dense_temporal import (
    DenseCausalTemporalClassifier,
    attach_dense_reason_features,
    build_dense_causal_temporal_examples,
    crop_dense_causal_sheet_panels,
    screen_dense_causal_temporal_examples,
    seal_dense_causal_frame_embeddings,
    verify_dense_causal_frame_embeddings,
    verify_dense_causal_temporal_screen,
)
from app.analysis.causal_shot_phase_review import (
    CAUSAL_PHASE_OFFSETS_SECONDS,
)
from app.analysis.shot_reason_evidence import (
    FEATURE_NAMES as REASON_FEATURE_NAMES,
)
from app.analysis.shot_reason_evidence import (
    seal_reason_evidence_artifact,
)

SOURCES = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)


def _embedding_artifact() -> dict:
    examples = []
    for game_index, source in enumerate(SOURCES):
        for event_index in range(6):
            is_free_throw = event_index < 3
            sign = 1.0 if is_free_throw else -1.0
            examples.append(
                {
                    "phase_review_id": (
                        f"phase-{game_index:02d}-{event_index:02d}"
                    ),
                    "source_video_sha256": source,
                    "candidate_bundle_sha256": str(game_index + 1) * 64,
                    "event_id": f"event-{event_index:02d}",
                    "frame_indexes": list(range(24)),
                    "embeddings": [
                        [
                            sign * (1.0 + position / 24.0)
                            + feature * 0.001
                            for feature in range(8)
                        ]
                        for position in range(24)
                    ],
                }
            )
    return seal_dense_causal_frame_embeddings(
        {
            "purpose": "test_dense_causal_frames",
            "codex_runtime_answer_used": False,
            "review_plan_sha256": "1" * 64,
            "sheet_manifest_sha256": "2" * 64,
            "source_video_sha256s": list(SOURCES),
            "sealed_blind_video_sha256s": ["e" * 64, "f" * 64],
            "backbone": "test/backbone",
            "backbone_sha256": "9" * 64,
            "embedding_dimension": 8,
            "anchor_offsets_seconds": list(CAUSAL_PHASE_OFFSETS_SECONDS),
            "examples": examples,
        }
    )


def _decisions() -> tuple[list[dict], list[dict], list[dict]]:
    base = []
    followup = []
    reviews = []
    for game_index, source in enumerate(SOURCES):
        for event_index in range(6):
            label = "free_throw" if event_index < 3 else "field_goal"
            row = {
                "source_video_sha256": source,
                "candidate_bundle_sha256": str(game_index + 1) * 64,
                "event_id": f"event-{event_index:02d}",
                "corrected_state": label,
            }
            if game_index == 0 and event_index == 0:
                base.append({**row, "corrected_state": None})
                followup.append(row)
            else:
                base.append(row)
            reviews.append(
                {
                    "phase_review_id": (
                        f"phase-{game_index:02d}-{event_index:02d}"
                    ),
                    "source_video_sha256": source,
                    "candidate_bundle_sha256": str(game_index + 1) * 64,
                    "event_id": f"event-{event_index:02d}",
                    "formation_state": (
                        "free_throw_setup"
                        if label == "free_throw"
                        else "live_play"
                    ),
                    "broadcast_context": "live_action",
                    "shot_sequence": "complete",
                    "release_position": 10,
                    "rim_arrival_position": 15,
                }
            )
    return base, followup, reviews


def test_dense_embedding_artifact_is_hash_sealed_and_rejects_blind_source() -> None:
    artifact = _embedding_artifact()
    assert verify_dense_causal_frame_embeddings(artifact) == artifact

    tampered = copy.deepcopy(artifact)
    tampered["examples"][0]["embeddings"][0][0] += 1.0
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_dense_causal_frame_embeddings(tampered)

    blind = copy.deepcopy(artifact)
    blind.pop("artifact_sha256")
    blind["sealed_blind_video_sha256s"] = [SOURCES[0]]
    with pytest.raises(ValueError, match="sealed blind"):
        seal_dense_causal_frame_embeddings(blind)


def test_dense_examples_exact_join_corrections_and_phase_targets() -> None:
    base, followup, reviews = _decisions()

    rows, provenance = build_dense_causal_temporal_examples(
        embedding_artifact=_embedding_artifact(),
        base_decisions=base,
        followup_decisions=followup,
        phase_reviews=reviews,
    )

    assert len(rows) == 24
    assert rows[0]["label"] in {"field_goal", "free_throw"}
    assert set(rows[0]["phase_targets"]) == set(PHASE_TARGET_NAMES)
    assert len(rows[0]["embeddings"]) == 24
    assert provenance["embedding_dimension"] == 8

    with pytest.raises(ValueError, match="exactly cover"):
        build_dense_causal_temporal_examples(
            embedding_artifact=_embedding_artifact(),
            base_decisions=base[:-1],
            followup_decisions=followup,
            phase_reviews=reviews,
        )


def test_dense_reason_fusion_exact_join_and_feature_dimension() -> None:
    base, followup, reviews = _decisions()
    rows, _ = build_dense_causal_temporal_examples(
        embedding_artifact=_embedding_artifact(),
        base_decisions=base,
        followup_decisions=followup,
        phase_reviews=reviews,
    )
    reason_rows = []
    for game_index, source in enumerate(SOURCES):
        for event_index in range(6):
            reason_rows.append(
                {
                    "source_video_sha256": source,
                    "candidate_bundle_sha256": str(game_index + 1) * 64,
                    "event_id": f"event-{event_index:02d}",
                    "features": [
                        float(event_index)
                        for _ in REASON_FEATURE_NAMES
                    ],
                }
            )
    reason = seal_reason_evidence_artifact(
        {
            "source_scene_artifact_sha256": "7" * 64,
            "pose_source_sha256s": ["8" * 64],
            "candidate_bundle_sha256s": [
                str(index + 1) * 64 for index in range(4)
            ],
            "examples": reason_rows,
        }
    )

    fused, provenance = attach_dense_reason_features(
        rows,
        reason_artifact=reason,
    )

    assert len(fused[0]["embeddings"][0]) == 8 + len(
        REASON_FEATURE_NAMES
    )
    assert provenance["reason_artifact_sha256"] == reason["artifact_sha256"]

    with pytest.raises(ValueError, match="exactly cover"):
        attach_dense_reason_features(
            rows,
            reason_artifact=seal_reason_evidence_artifact(
                {
                    "source_scene_artifact_sha256": "7" * 64,
                    "pose_source_sha256s": ["8" * 64],
                    "candidate_bundle_sha256s": [
                        str(index + 1) * 64 for index in range(4)
                    ],
                    "examples": reason_rows[:-1],
                }
            ),
        )


def test_dense_temporal_classifier_shapes_main_and_phase_outputs() -> None:
    model = DenseCausalTemporalClassifier(
        embedding_dimension=8,
        hidden_dimension=12,
        kernel_size=3,
    )

    main_logits, phase_logits = model(torch.ones(3, 24, 8))

    assert main_logits.shape == (3, 2)
    assert phase_logits.shape == (3, len(PHASE_TARGET_NAMES))

    recurrent = DenseCausalTemporalClassifier(
        embedding_dimension=8,
        hidden_dimension=12,
        kernel_size=3,
        temporal_operator="gru",
    )
    recurrent_main, recurrent_phase = recurrent(torch.ones(3, 24, 8))
    assert recurrent_main.shape == (3, 2)
    assert recurrent_phase.shape == (3, len(PHASE_TARGET_NAMES))

    gated = DenseCausalTemporalClassifier(
        embedding_dimension=8,
        hidden_dimension=12,
        kernel_size=3,
        temporal_operator="gated",
    )
    gated_main, gated_phase = gated(torch.ones(3, 24, 8))
    attention = gated.temporal_attention_weights(torch.ones(3, 24, 8))
    assert gated_main.shape == (3, 2)
    assert gated_phase.shape == (3, len(PHASE_TARGET_NAMES))
    assert attention.shape == (3, 24)
    assert torch.allclose(attention.sum(dim=1), torch.ones(3))


def test_dense_sheet_crop_recovers_24_ordered_panels_without_headers() -> None:
    image = np.zeros((1440, 1920, 3), dtype=np.uint8)
    for example_index in range(2):
        for position in range(24):
            row = example_index * 4 + position // 6
            column = position % 6
            image[
                row * 180 + 28 : (row + 1) * 180,
                column * 320 : (column + 1) * 320,
            ] = example_index * 100 + position

    panels = crop_dense_causal_sheet_panels(
        image,
        example_index=1,
        examples_in_sheet=2,
    )

    assert len(panels) == 24
    assert panels[0].shape == (152, 320, 3)
    assert int(panels[0][0, 0, 0]) == 100
    assert int(panels[-1][-1, -1, -1]) == 123


def test_dense_screen_keeps_every_held_game_out_and_is_hash_sealed() -> None:
    base, followup, reviews = _decisions()
    rows, provenance = build_dense_causal_temporal_examples(
        embedding_artifact=_embedding_artifact(),
        base_decisions=base,
        followup_decisions=followup,
        phase_reviews=reviews,
    )

    result = screen_dense_causal_temporal_examples(
        examples=rows,
        provenance={
            **provenance,
            "label_correction_artifact_sha256s": ["3" * 64, "4" * 64],
            "phase_review_artifact_sha256": "5" * 64,
        },
        configurations=[
            {
                "hidden_dimension": 8,
                "kernel_size": 3,
                "auxiliary_weight": 0.25,
                "epochs": 4,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
            }
        ],
        acceptance_threshold=0.85,
    )

    assert result["runtime_consumable"] is False
    assert result["codex_runtime_answer_used"] is False
    assert len(result["outer_folds"]) == 4
    for fold in result["outer_folds"]:
        assert fold["held_target_group"] not in fold["training_target_groups"]
        assert fold["held_target_group"] not in fold["selection_target_groups"]
        for inner in fold["selection"]["folds"]:
            assert inner["held_selection_group"] not in inner[
                "training_target_groups"
            ]
    assert verify_dense_causal_temporal_screen(result) == result

    tampered = copy.deepcopy(result)
    tampered["metrics"]["balanced_accuracy"] = 0.0
    with pytest.raises(ValueError, match="inconsistent|hash mismatch"):
        verify_dense_causal_temporal_screen(tampered)

    leaked = copy.deepcopy(result)
    leaked["outer_folds"][0]["training_target_groups"].append(
        leaked["outer_folds"][0]["held_target_group"]
    )
    with pytest.raises(ValueError, match="leaks held game"):
        verify_dense_causal_temporal_screen(leaked)
