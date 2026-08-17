from __future__ import annotations

import copy

import pytest

from app.analysis.causal_phase_auxiliary_screen import (
    join_causal_phase_targets,
    screen_causal_phase_auxiliary_examples,
    verify_causal_phase_auxiliary_screen,
)
from app.analysis.pbp_visual_state_frames import (
    POST_ANCHOR_OFFSETS_SECONDS,
    PRE_ANCHOR_OFFSETS_SECONDS,
    WIDE_ANCHOR_OFFSETS_SECONDS,
)


def _examples() -> list[dict[str, object]]:
    offsets = sorted(
        set(
            PRE_ANCHOR_OFFSETS_SECONDS
            + (2.0,)
            + POST_ANCHOR_OFFSETS_SECONDS
            + WIDE_ANCHOR_OFFSETS_SECONDS
        )
    )
    rows = []
    for group_index, group in enumerate(("a", "b", "c", "d")):
        source = group * 64
        for event_index in range(8):
            is_free_throw = event_index % 2
            signal = 4.0 if is_free_throw else -4.0
            rows.append(
                {
                    "source_video_sha256": source,
                    "candidate_bundle_sha256": "e" * 64,
                    "event_id": f"{group}-{event_index}",
                    "label": (
                        "free_throw" if is_free_throw else "field_goal"
                    ),
                    "offsets_seconds": offsets,
                    "embeddings": [
                        [
                            signal,
                            signal + offset / 100.0,
                            float(group_index),
                            float(event_index) / 10.0,
                        ]
                        for offset in offsets
                    ],
                    "phase_targets": {
                        "formation_free_throw": is_free_throw,
                        "formation_live_play": 1 - is_free_throw,
                        "context_replay": event_index % 3 == 0,
                        "context_live_action": event_index % 3 != 0,
                        "sequence_complete": event_index % 4 != 0,
                        "sequence_not_a_shot": event_index % 4 == 0,
                        "release_visible": event_index % 4 != 0,
                        "rim_visible": event_index % 2,
                    },
                }
            )
    return rows


def test_auxiliary_screen_is_doubly_game_disjoint_and_hash_sealed() -> None:
    artifact = screen_causal_phase_auxiliary_examples(
        examples=_examples(),
        provenance={
            "backbone": "synthetic",
            "backbone_sha256": "1" * 64,
            "embedding_dimension": 4,
            "embedding_artifact_sha256s": ["2" * 64],
            "label_correction_artifact_sha256s": ["3" * 64],
            "phase_review_artifact_sha256": "4" * 64,
        },
        configurations=[
            {
                "representation": "wide_summary",
                "c": 1.0,
                "pca_components": 4,
            }
        ],
    )

    assert artifact["target_example_count"] == 32
    assert set(artifact["variants"]) == {"baseline", "phase_auxiliary"}
    for variant in artifact["variants"].values():
        for fold in variant["outer_folds"]:
            held = fold["held_target_group"]
            assert held not in fold["training_target_groups"]
            for row in fold.get("meta_oof_provenance", []):
                assert held not in row["phase_training_groups"]
                assert row["scored_group"] not in row[
                    "phase_training_groups"
                ]
    assert verify_causal_phase_auxiliary_screen(artifact) == artifact

    tampered = copy.deepcopy(artifact)
    fold = tampered["variants"]["phase_auxiliary"]["outer_folds"][0]
    fold["meta_oof_provenance"][0]["phase_training_groups"].append(
        fold["held_target_group"]
    )
    with pytest.raises(ValueError, match="leaked"):
        verify_causal_phase_auxiliary_screen(tampered)


def test_join_causal_phase_targets_requires_exact_event_coverage() -> None:
    temporal = [
        {
            "source_video_sha256": "a" * 64,
            "candidate_bundle_sha256": "b" * 64,
            "event_id": "event-1",
        }
    ]
    reviews = [
        {
            "source_video_sha256": "a" * 64,
            "candidate_bundle_sha256": "b" * 64,
            "event_id": "event-1",
            "formation_state": "free_throw_setup",
            "broadcast_context": "live_action",
            "shot_sequence": "partial",
            "release_position": 4,
            "rim_arrival_position": None,
        }
    ]

    joined = join_causal_phase_targets(temporal, reviews)

    assert joined[0]["phase_targets"] == {
        "formation_free_throw": 1,
        "formation_live_play": 0,
        "context_replay": 0,
        "context_live_action": 1,
        "sequence_complete": 0,
        "sequence_not_a_shot": 0,
        "release_visible": 1,
        "rim_visible": 0,
    }
    with pytest.raises(ValueError, match="exactly cover"):
        join_causal_phase_targets(temporal, [])
