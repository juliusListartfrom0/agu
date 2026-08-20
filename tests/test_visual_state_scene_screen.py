from __future__ import annotations

import copy

import pytest

from app.analysis.replay_transition import (
    REPLAY_TRANSITION_EVIDENCE_SCHEMA,
    replay_transition_payload,
    seal_replay_transition_artifact,
    summarize_transition_probabilities,
)
from app.analysis.visual_state_scene_screen import (
    build_visual_state_scene_examples,
    screen_visual_state_scene_examples,
    verify_visual_state_scene_screen,
)


def _examples() -> list[dict]:
    rows = []
    offsets = [-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0]
    for game_index, source in enumerate(("a" * 64, "b" * 64, "c" * 64, "d" * 64)):
        for event_index, label in enumerate(
            ("free_throw", "free_throw", "field_goal", "field_goal")
        ):
            sign = 1.0 if label == "free_throw" else -1.0
            rows.append(
                {
                    "source_video_sha256": source,
                    "candidate_bundle_sha256": str(game_index + 1) * 64,
                    "event_id": f"event-{event_index}",
                    "label": label,
                    "offsets_seconds": offsets,
                    "embeddings": [
                        [sign + game_index * 0.001, sign, sign, sign]
                        for _ in offsets
                    ],
                }
            )
    return rows


def _transition_artifacts() -> list[dict]:
    artifacts = []
    for game_index, source in enumerate(("a" * 64, "b" * 64, "c" * 64, "d" * 64)):
        events = []
        for event_index in range(4):
            probabilities = [0.1] * 200
            probabilities[80 + event_index] = 0.9
            evidence = summarize_transition_probabilities(
                event_id=f"event-{event_index}",
                anchor_frame=100,
                window_start_frame=0,
                probabilities=probabilities,
                fps=10.0,
            )
            events.append(replay_transition_payload(evidence))
        artifacts.append(
            seal_replay_transition_artifact(
                {
                    "schema_version": REPLAY_TRANSITION_EVIDENCE_SCHEMA,
                    "runtime_consumable": False,
                    "codex_runtime_answer_used": False,
                    "raw_video_sha256": source,
                    "candidate_bundle_sha256": str(game_index + 1) * 64,
                    "events": events,
                }
            )
        )
    return artifacts


def test_scene_examples_exact_join_ignores_replay_semantic_field() -> None:
    examples = _examples()
    artifacts = _transition_artifacts()
    expected_keys = {
        (
            row["source_video_sha256"],
            row["candidate_bundle_sha256"],
            row["event_id"],
        )
        for row in examples
    }

    rows, hashes = build_visual_state_scene_examples(
        temporal_examples=examples,
        transition_artifacts=artifacts,
        expected_event_keys=expected_keys,
        sealed_blind_video_sha256s=["e" * 64],
    )

    assert len(rows) == 16
    assert len(hashes) == 4
    assert "broadcast_state" not in rows[0]
    assert rows[0]["transition_offsets_seconds"]

    changed = copy.deepcopy(artifacts)
    for artifact in changed:
        artifact.pop("artifact_sha256")
        for event in artifact["events"]:
            event["broadcast_state"] = "replay"
        changed_artifact = seal_replay_transition_artifact(artifact)
        artifact.clear()
        artifact.update(changed_artifact)
    changed_rows, _ = build_visual_state_scene_examples(
        temporal_examples=examples,
        transition_artifacts=changed,
        expected_event_keys=expected_keys,
        sealed_blind_video_sha256s=["e" * 64],
    )
    assert changed_rows == rows


def test_scene_examples_reject_sealed_blind_source() -> None:
    examples = _examples()
    with pytest.raises(ValueError, match="sealed blind"):
        build_visual_state_scene_examples(
            temporal_examples=examples,
            transition_artifacts=_transition_artifacts(),
            expected_event_keys={
                (
                    row["source_video_sha256"],
                    row["candidate_bundle_sha256"],
                    row["event_id"],
                )
                for row in examples
            },
            sealed_blind_video_sha256s=["a" * 64],
        )


def test_scene_examples_reject_codex_runtime_transition_artifact() -> None:
    examples = _examples()
    artifacts = _transition_artifacts()
    untrusted = copy.deepcopy(artifacts[0])
    untrusted.pop("artifact_sha256")
    untrusted["codex_runtime_answer_used"] = True
    artifacts[0] = seal_replay_transition_artifact(untrusted)

    with pytest.raises(ValueError, match="runtime answer"):
        build_visual_state_scene_examples(
            temporal_examples=examples,
            transition_artifacts=artifacts,
            expected_event_keys={
                (
                    row["source_video_sha256"],
                    row["candidate_bundle_sha256"],
                    row["event_id"],
                )
                for row in examples
            },
            sealed_blind_video_sha256s=["e" * 64],
        )


def test_scene_screen_is_nested_game_disjoint_and_hash_sealed() -> None:
    examples = _examples()
    rows, transition_hashes = build_visual_state_scene_examples(
        temporal_examples=examples,
        transition_artifacts=_transition_artifacts(),
        expected_event_keys={
            (
                row["source_video_sha256"],
                row["candidate_bundle_sha256"],
                row["event_id"],
            )
            for row in examples
        },
        sealed_blind_video_sha256s=["e" * 64],
    )

    result = screen_visual_state_scene_examples(
        examples=rows,
        provenance={
            "backbone": "test-backbone",
            "backbone_sha256": "9" * 64,
            "embedding_dimension": 4,
            "embedding_artifact_sha256s": ["1" * 64],
            "transition_artifact_sha256s": transition_hashes,
        },
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
    for fold in result["outer_folds"]:
        assert fold["held_target_group"] not in fold["training_target_groups"]
        assert fold["held_target_group"] not in fold["selection_target_groups"]
    assert verify_visual_state_scene_screen(result) == result
