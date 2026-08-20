from __future__ import annotations

import copy

import pytest

from app.analysis.visual_state_domain_screen import (
    join_reviewed_target_embeddings,
    screen_visual_state_domain_embeddings,
    verify_visual_state_domain_screen,
)


def _embedding(label: str, *, offset: float = 0.0) -> list[float]:
    sign = 1.0 if label == "free_throw" else -1.0
    return [sign + offset, sign * 0.5 - offset]


def _source_examples() -> list[dict]:
    rows = []
    for group_index, group in enumerate(("v001", "v002", "v003")):
        for index, label in enumerate(("ft0", "ft1", "2p0", "3p1")):
            state = "free_throw" if label.startswith("ft") else "field_goal"
            rows.append(
                {
                    "relative_path": f"{label}/{group}-{index}.mp4",
                    "source_group": group,
                    "label": label,
                    "embedding": _embedding(
                        state,
                        offset=(group_index - 1) * 0.01,
                    ),
                }
            )
    return rows


def _target_examples() -> list[dict]:
    rows = []
    for game_index, source in enumerate(("a" * 64, "b" * 64, "c" * 64, "d" * 64)):
        for event_index, state in enumerate(
            ("free_throw", "free_throw", "field_goal", "field_goal")
        ):
            rows.append(
                {
                    "source_video_sha256": source,
                    "candidate_bundle_sha256": str(game_index + 1) * 64,
                    "event_id": f"event-{event_index}",
                    "label": state,
                    "embedding": _embedding(
                        state,
                        offset=(game_index - 1.5) * 0.01,
                    ),
                }
            )
    return rows


def test_join_reviewed_rows_applies_only_valid_followup_overlays() -> None:
    target = _target_examples()[:3]
    base = [
        {
            "review_id": f"review-{index}",
            "source_video_sha256": row["source_video_sha256"],
            "candidate_bundle_sha256": row["candidate_bundle_sha256"],
            "event_id": row["event_id"],
            "corrected_state": label,
        }
        for index, (row, label) in enumerate(
            zip(target, ("free_throw", None, "field_goal"), strict=True)
        )
    ]
    followup = [
        {
            **base[1],
            "corrected_state": "free_throw",
        }
    ]

    joined = join_reviewed_target_embeddings(
        target,
        base_decisions=base,
        followup_decisions=followup,
        sealed_blind_video_sha256s=["f" * 64],
    )

    assert [row["label"] for row in joined] == [
        "free_throw",
        "free_throw",
        "field_goal",
    ]

    invalid = copy.deepcopy(followup)
    invalid[0]["event_id"] = "already-resolved"
    with pytest.raises(ValueError, match="unresolved"):
        join_reviewed_target_embeddings(
            target,
            base_decisions=base,
            followup_decisions=invalid,
            sealed_blind_video_sha256s=["f" * 64],
        )


def test_join_reviewed_rows_rejects_sealed_blind_source() -> None:
    target = _target_examples()[:1]
    decision = {
        "review_id": "review-1",
        "source_video_sha256": target[0]["source_video_sha256"],
        "candidate_bundle_sha256": target[0]["candidate_bundle_sha256"],
        "event_id": target[0]["event_id"],
        "corrected_state": "free_throw",
    }

    with pytest.raises(ValueError, match="sealed blind"):
        join_reviewed_target_embeddings(
            target,
            base_decisions=[decision],
            followup_decisions=[],
            sealed_blind_video_sha256s=[target[0]["source_video_sha256"]],
        )


def test_domain_screen_is_nested_game_disjoint_and_hash_sealed() -> None:
    result = screen_visual_state_domain_embeddings(
        source_examples=_source_examples(),
        target_examples=_target_examples(),
        backbone="torchvision/mvit_v2_s/kinetics400_v1",
        backbone_sha256="9" * 64,
        source_artifact_sha256="7" * 64,
        target_artifact_sha256="8" * 64,
        label_correction_sha256s=["5" * 64, "6" * 64],
        configurations=[
            {
                "c": 1.0,
                "source_weight": 0.25,
                "pca_components": None,
            }
        ],
    )

    assert result["runtime_consumable"] is False
    assert result["accepted"] is True
    assert result["acceptance_gate"]["minimum_game_balanced_accuracy"] == 0.85
    assert len(result["outer_folds"]) == 4
    for fold in result["outer_folds"]:
        assert fold["held_target_group"] not in fold["training_target_groups"]
        assert fold["held_target_group"] not in fold["selection_target_groups"]
    assert result["metrics"]["worst_game_balanced_accuracy"] == 1.0
    assert verify_visual_state_domain_screen(result) == result

    tampered = copy.deepcopy(result)
    tampered["metrics"]["worst_game_balanced_accuracy"] = 0.5
    with pytest.raises(ValueError, match="inconsistent|hash mismatch"):
        verify_visual_state_domain_screen(tampered)


def test_domain_screen_requires_multiple_target_games_and_matching_dimensions() -> None:
    with pytest.raises(ValueError, match="four target groups"):
        screen_visual_state_domain_embeddings(
            source_examples=_source_examples(),
            target_examples=_target_examples()[:4],
            backbone="torchvision/mvit_v2_s/kinetics400_v1",
            backbone_sha256="9" * 64,
            source_artifact_sha256="7" * 64,
            target_artifact_sha256="8" * 64,
            label_correction_sha256s=["5" * 64],
        )

    broken_source = _source_examples()
    broken_source[0]["embedding"] = [1.0]
    with pytest.raises(ValueError, match="dimension"):
        screen_visual_state_domain_embeddings(
            source_examples=broken_source,
            target_examples=_target_examples(),
            backbone="torchvision/mvit_v2_s/kinetics400_v1",
            backbone_sha256="9" * 64,
            source_artifact_sha256="7" * 64,
            target_artifact_sha256="8" * 64,
            label_correction_sha256s=["5" * 64],
        )
