from __future__ import annotations

import copy

import pytest

from app.analysis.pbp_visual_state_frames import (
    ANCHOR_OFFSETS_SECONDS,
    DINO_V2_SMALL_BACKBONE,
    POST_ANCHOR_OFFSETS_SECONDS,
    PRE_ANCHOR_OFFSETS_SECONDS,
    WIDE_ANCHOR_OFFSETS_SECONDS,
    seal_anchor_state_embedding_artifact,
)
from app.analysis.visual_state_temporal_screen import (
    build_visual_state_temporal_examples,
    screen_visual_state_temporal_examples,
    verify_visual_state_temporal_screen,
)


def _artifact(offsets: tuple[float, ...]) -> dict:
    examples = []
    for game_index, source in enumerate(("a" * 64, "b" * 64, "c" * 64, "d" * 64)):
        for event_index, state in enumerate(
            ("free_throw", "free_throw", "field_goal", "field_goal")
        ):
            sign = 1.0 if state == "free_throw" else -1.0
            anchor = 300 + event_index * 30
            examples.append(
                {
                    "source_video_sha256": source,
                    "source_video_filename": f"game-{game_index}.mp4",
                    "candidate_bundle_sha256": str(game_index + 1) * 64,
                    "event_id": f"event-{event_index}",
                    "state": state,
                    "anchor_frame": anchor,
                    "source_fps": 30.0,
                    "frame_count": 2000,
                    "frame_indexes": [
                        round(anchor + offset * 30.0) for offset in offsets
                    ],
                    "embeddings": [
                        [
                            sign + game_index * 0.001 + offset * 0.0001
                            for _ in range(384)
                        ]
                        for offset in offsets
                    ],
                }
            )
    return seal_anchor_state_embedding_artifact(
        {
            "purpose": "test",
            "truth_used_for_training_only": True,
            "codex_runtime_answer_used": False,
            "training_manifest_sha256s": ["1" * 64],
            "clock_artifact_sha256s": ["2" * 64],
            "source_video_sha256s": ["a" * 64, "b" * 64, "c" * 64, "d" * 64],
            "sealed_blind_video_sha256s": ["e" * 64, "f" * 64],
            "backbone": DINO_V2_SMALL_BACKBONE,
            "backbone_sha256": "9" * 64,
            "anchor_offsets_seconds": list(offsets),
            "examples": examples,
        }
    )


def _decisions() -> tuple[list[dict], list[dict]]:
    base = []
    followup = []
    for game_index, source in enumerate(("a" * 64, "b" * 64, "c" * 64, "d" * 64)):
        for event_index, state in enumerate(
            ("free_throw", "free_throw", "field_goal", "field_goal")
        ):
            decision = {
                "review_id": f"review-{game_index}-{event_index}",
                "source_video_sha256": source,
                "candidate_bundle_sha256": str(game_index + 1) * 64,
                "event_id": f"event-{event_index}",
                "corrected_state": state,
            }
            if game_index == 0 and event_index == 0:
                base.append({**decision, "corrected_state": None})
                followup.append(decision)
            else:
                base.append(decision)
    return base, followup


def test_temporal_examples_exact_join_and_deduplicate_shared_offsets() -> None:
    base, followup = _decisions()

    rows, provenance = build_visual_state_temporal_examples(
        pre_anchor_artifact=_artifact(PRE_ANCHOR_OFFSETS_SECONDS),
        anchor_artifact=_artifact(ANCHOR_OFFSETS_SECONDS),
        post_anchor_artifact=_artifact(POST_ANCHOR_OFFSETS_SECONDS),
        base_decisions=base,
        followup_decisions=followup,
    )

    assert len(rows) == 16
    assert rows[0]["offsets_seconds"] == [
        -4.0,
        -3.0,
        -2.0,
        -1.0,
        0.0,
        2.0,
        5.0,
        10.0,
        15.0,
        20.0,
    ]
    assert len(rows[0]["embeddings"]) == 10
    assert provenance["backbone"] == DINO_V2_SMALL_BACKBONE
    assert len(provenance["embedding_artifact_sha256s"]) == 3


def test_temporal_examples_reject_cross_artifact_mismatch_and_blind_source() -> None:
    base, followup = _decisions()
    post = _artifact(POST_ANCHOR_OFFSETS_SECONDS)
    post["examples"][0]["embeddings"][0][0] += 1.0
    with pytest.raises(ValueError, match="hash mismatch"):
        build_visual_state_temporal_examples(
            pre_anchor_artifact=_artifact(PRE_ANCHOR_OFFSETS_SECONDS),
            anchor_artifact=_artifact(ANCHOR_OFFSETS_SECONDS),
            post_anchor_artifact=post,
            base_decisions=base,
            followup_decisions=followup,
        )

    pre = _artifact(PRE_ANCHOR_OFFSETS_SECONDS)
    pre.pop("artifact_sha256")
    pre["sealed_blind_video_sha256s"] = ["a" * 64]
    with pytest.raises(ValueError, match="sealed blind|provenance"):
        build_visual_state_temporal_examples(
            pre_anchor_artifact=pre,
            anchor_artifact=_artifact(ANCHOR_OFFSETS_SECONDS),
            post_anchor_artifact=_artifact(POST_ANCHOR_OFFSETS_SECONDS),
            base_decisions=base,
            followup_decisions=followup,
        )


def test_temporal_screen_is_nested_game_disjoint_and_hash_sealed() -> None:
    base, followup = _decisions()
    rows, provenance = build_visual_state_temporal_examples(
        pre_anchor_artifact=_artifact(PRE_ANCHOR_OFFSETS_SECONDS),
        anchor_artifact=_artifact(ANCHOR_OFFSETS_SECONDS),
        post_anchor_artifact=_artifact(POST_ANCHOR_OFFSETS_SECONDS),
        base_decisions=base,
        followup_decisions=followup,
    )

    result = screen_visual_state_temporal_examples(
        examples=rows,
        provenance=provenance,
        label_correction_artifact_sha256s=["5" * 64, "6" * 64],
        configurations=[
            {
                "representation": "full_summary",
                "c": 0.1,
                "pca_components": 2,
            }
        ],
    )

    assert result["runtime_consumable"] is False
    assert result["accepted"] is True
    assert result["metrics"]["worst_game_balanced_accuracy"] == 1.0
    for fold in result["outer_folds"]:
        assert fold["held_target_group"] not in fold["training_target_groups"]
        assert fold["held_target_group"] not in fold["selection_target_groups"]
    assert verify_visual_state_temporal_screen(result) == result

    tampered = copy.deepcopy(result)
    tampered["metrics"]["worst_game_balanced_accuracy"] = 0.5
    with pytest.raises(ValueError, match="inconsistent|hash mismatch"):
        verify_visual_state_temporal_screen(tampered)


def test_temporal_screen_accepts_wide_centered_embeddings() -> None:
    base, followup = _decisions()
    rows, provenance = build_visual_state_temporal_examples(
        pre_anchor_artifact=_artifact(PRE_ANCHOR_OFFSETS_SECONDS),
        anchor_artifact=_artifact(ANCHOR_OFFSETS_SECONDS),
        post_anchor_artifact=_artifact(POST_ANCHOR_OFFSETS_SECONDS),
        wide_anchor_artifact=_artifact(WIDE_ANCHOR_OFFSETS_SECONDS),
        base_decisions=base,
        followup_decisions=followup,
    )

    assert rows[0]["offsets_seconds"] == [
        -8.0,
        -6.0,
        -4.0,
        -3.0,
        -2.0,
        -1.0,
        0.0,
        2.0,
        4.0,
        5.0,
        6.0,
        8.0,
        10.0,
        15.0,
        20.0,
    ]
    result = screen_visual_state_temporal_examples(
        examples=rows,
        provenance=provenance,
        label_correction_artifact_sha256s=["5" * 64, "6" * 64],
        configurations=[
            {
                "representation": "wide_summary",
                "c": 0.1,
                "pca_components": 2,
            }
        ],
    )

    assert result["accepted"] is True
    assert result["configurations"][0]["representation"] == "wide_summary"
