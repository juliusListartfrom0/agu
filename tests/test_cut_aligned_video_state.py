from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch import nn

from app.analysis.cut_aligned_video_state import (
    build_cut_aligned_clip_plan,
    extract_cut_aligned_video_embeddings,
    seal_cut_aligned_video_embedding_artifact,
    verify_cut_aligned_video_embedding_artifact,
)
from app.analysis.cut_aligned_video_state_screen import (
    build_cut_aligned_video_state_examples,
    screen_cut_aligned_video_state_examples,
    verify_cut_aligned_video_state_screen,
)
from app.analysis.replay_transition import (
    REPLAY_TRANSITION_EVIDENCE_SCHEMA,
    replay_transition_payload,
    seal_replay_transition_artifact,
    summarize_transition_probabilities,
)

SOURCE_SHA = "a" * 64
BUNDLE_SHA = "b" * 64


def _review_examples() -> list[dict[str, object]]:
    return [
        {
            "source_video_sha256": SOURCE_SHA,
            "source_video_filename": "development.mp4",
            "candidate_bundle_sha256": BUNDLE_SHA,
            "event_id": "event-with-neighbors",
            "anchor_frame": 1000,
            "source_fps": 10.0,
            "frame_count": 2000,
        },
        {
            "source_video_sha256": SOURCE_SHA,
            "source_video_filename": "development.mp4",
            "candidate_bundle_sha256": BUNDLE_SHA,
            "event_id": "event-without-cuts",
            "anchor_frame": 1500,
            "source_fps": 10.0,
            "frame_count": 2000,
        },
    ]


def _transition_artifact() -> dict[str, object]:
    events = []
    for event_id, anchor, spike_indexes in (
        ("event-with-neighbors", 1000, (40, 70, 110, 140)),
        ("event-without-cuts", 1500, ()),
    ):
        probabilities = [0.0] * 200
        for index in spike_indexes:
            probabilities[index] = 0.9
        evidence = summarize_transition_probabilities(
            event_id=event_id,
            anchor_frame=anchor,
            window_start_frame=anchor - 100,
            probabilities=probabilities,
            fps=10.0,
        )
        events.append(replay_transition_payload(evidence))
    return seal_replay_transition_artifact(
        {
            "schema_version": REPLAY_TRANSITION_EVIDENCE_SCHEMA,
            "runtime_consumable": False,
            "codex_runtime_answer_used": False,
            "raw_video_sha256": SOURCE_SHA,
            "candidate_bundle_sha256": BUNDLE_SHA,
            "events": events,
        }
    )


def test_cut_aligned_plan_samples_dense_clips_inside_neighboring_shots() -> None:
    rows = build_cut_aligned_clip_plan(
        review_examples=_review_examples(),
        transition_artifact=_transition_artifact(),
        sealed_blind_video_sha256s=["e" * 64],
        clip_frames=16,
        context_radius_seconds=8.0,
    )

    assert len(rows) == 2
    with_neighbors = rows[0]
    segments = {row["role"]: row for row in with_neighbors["segments"]}
    assert set(segments) == {"previous", "anchor", "next"}
    assert all(segments[role]["available"] for role in segments)
    assert segments["previous"]["start_offset_seconds"] == pytest.approx(-6.0)
    assert segments["previous"]["end_offset_seconds"] == pytest.approx(-3.0)
    assert segments["anchor"]["start_offset_seconds"] == pytest.approx(-3.0)
    assert segments["anchor"]["end_offset_seconds"] == pytest.approx(1.0)
    assert segments["next"]["start_offset_seconds"] == pytest.approx(1.0)
    assert segments["next"]["end_offset_seconds"] == pytest.approx(4.0)
    for segment in segments.values():
        assert len(segment["frame_indexes"]) == 16
        assert min(segment["frame_indexes"]) >= 0
        assert max(segment["frame_indexes"]) < 2000

    without_cuts = {row["role"]: row for row in rows[1]["segments"]}
    assert without_cuts["anchor"]["available"] is True
    assert without_cuts["previous"]["available"] is False
    assert without_cuts["next"]["available"] is False


def test_cut_aligned_plan_ignores_transition_semantics_and_rejects_blind() -> None:
    artifact = _transition_artifact()
    baseline = build_cut_aligned_clip_plan(
        review_examples=_review_examples(),
        transition_artifact=artifact,
        sealed_blind_video_sha256s=["e" * 64],
    )
    changed = copy.deepcopy(artifact)
    changed.pop("artifact_sha256")
    for event in changed["events"]:
        event["broadcast_state"] = "replay"
    changed = seal_replay_transition_artifact(changed)

    assert build_cut_aligned_clip_plan(
        review_examples=_review_examples(),
        transition_artifact=changed,
        sealed_blind_video_sha256s=["e" * 64],
    ) == baseline

    with pytest.raises(ValueError, match="sealed blind"):
        build_cut_aligned_clip_plan(
            review_examples=_review_examples(),
            transition_artifact=artifact,
            sealed_blind_video_sha256s=[SOURCE_SHA],
        )


def test_cut_aligned_embedding_artifact_is_hash_bound_and_training_only() -> None:
    planned = build_cut_aligned_clip_plan(
        review_examples=_review_examples(),
        transition_artifact=_transition_artifact(),
        sealed_blind_video_sha256s=["e" * 64],
    )
    for example in planned:
        for index, segment in enumerate(example["segments"]):
            segment["embedding"] = (
                [float(index)] * 4 if segment["available"] else None
            )
    artifact = seal_cut_aligned_video_embedding_artifact(
        {
            "purpose": "cut_aligned_visual_state_training_only",
            "runtime_consumable": False,
            "codex_runtime_answer_used": False,
            "transition_semantics_used": False,
            "truth_used_for_training_only": False,
            "review_plan_sha256": "1" * 64,
            "transition_artifact_sha256": _transition_artifact()["artifact_sha256"],
            "raw_video_sha256": SOURCE_SHA,
            "candidate_bundle_sha256": BUNDLE_SHA,
            "backbone": "test-backbone",
            "backbone_sha256": "2" * 64,
            "embedding_dimension": 4,
            "clip_frames": 16,
            "expected_event_ids": [
                "event-with-neighbors",
                "event-without-cuts",
            ],
            "complete": True,
            "examples": planned,
        }
    )

    assert verify_cut_aligned_video_embedding_artifact(artifact)["complete"] is True
    assert artifact["runtime_consumable"] is False
    assert artifact["transition_semantics_used"] is False

    tampered = copy.deepcopy(artifact)
    tampered["examples"][0]["segments"][1]["embedding"][0] += 1.0
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_cut_aligned_video_embedding_artifact(tampered)


def test_cut_aligned_extractor_encodes_only_available_segments(monkeypatch) -> None:
    class FakeCapture:
        def __init__(self, _path: str) -> None:
            self.position = 0

        def isOpened(self) -> bool:
            return True

        def set(self, _property: int, value: int) -> None:
            self.position = value

        def read(self):
            return True, np.full((4, 4, 3), self.position % 255, dtype=np.uint8)

        def release(self) -> None:
            return None

    class FakeBackbone(nn.Module):
        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            mean = batch.to(torch.float32).mean(dim=(1, 2, 3, 4))
            return mean[:, None].repeat(1, 4)

    monkeypatch.setattr(
        "app.analysis.cut_aligned_video_state.cv2.VideoCapture",
        FakeCapture,
    )
    planned = build_cut_aligned_clip_plan(
        review_examples=_review_examples(),
        transition_artifact=_transition_artifact(),
        sealed_blind_video_sha256s=["e" * 64],
    )

    extracted = extract_cut_aligned_video_embeddings(
        video_path="development.mp4",
        planned_examples=planned,
        backbone=FakeBackbone(),
        transform=lambda clip: clip.permute(1, 0, 2, 3).to(torch.float32),
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert len(extracted) == 2
    assert all(
        len(segment["embedding"]) == 4
        for segment in extracted[0]["segments"]
        if segment["available"]
    )
    unavailable = [
        segment
        for segment in extracted[1]["segments"]
        if not segment["available"]
    ]
    assert unavailable
    assert all(segment["embedding"] is None for segment in unavailable)


def _screen_inputs() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    artifacts = []
    decisions = []
    for game_index, source in enumerate(("a" * 64, "b" * 64, "c" * 64, "d" * 64)):
        bundle = str(game_index + 1) * 64
        examples = []
        expected = []
        for event_index, label in enumerate(
            ("free_throw", "free_throw", "field_goal", "field_goal")
        ):
            event_id = f"event-{event_index}"
            expected.append(event_id)
            sign = 2.0 if label == "free_throw" else -2.0
            examples.append(
                {
                    "source_video_sha256": source,
                    "source_video_filename": f"game-{game_index}.mp4",
                    "candidate_bundle_sha256": bundle,
                    "event_id": event_id,
                    "anchor_frame": 100,
                    "source_fps": 10.0,
                    "frame_count": 1000,
                    "segments": [
                        {
                            "role": role,
                            "available": True,
                            "start_offset_seconds": float(role_index - 2),
                            "end_offset_seconds": float(role_index - 1),
                            "frame_indexes": list(range(16)),
                            "embedding": [
                                sign + game_index * 0.001 + role_index * 0.01
                            ]
                            * 4,
                        }
                        for role_index, role in enumerate(
                            ("previous", "anchor", "next")
                        )
                    ],
                }
            )
            decisions.append(
                {
                    "source_video_sha256": source,
                    "candidate_bundle_sha256": bundle,
                    "event_id": event_id,
                    "corrected_state": label,
                }
            )
        artifacts.append(
            seal_cut_aligned_video_embedding_artifact(
                {
                    "purpose": "cut_aligned_visual_state_training_only",
                    "runtime_consumable": False,
                    "codex_runtime_answer_used": False,
                    "transition_semantics_used": False,
                    "truth_used_for_training_only": False,
                    "review_plan_sha256": "1" * 64,
                    "transition_artifact_sha256": str(game_index + 5) * 64,
                    "raw_video_sha256": source,
                    "candidate_bundle_sha256": bundle,
                    "backbone": "test-backbone",
                    "backbone_sha256": "2" * 64,
                    "embedding_dimension": 4,
                    "clip_frames": 16,
                    "expected_event_ids": expected,
                    "complete": True,
                    "examples": examples,
                }
            )
        )
    return artifacts, decisions


def test_cut_aligned_screen_is_nested_game_disjoint_and_hash_sealed() -> None:
    artifacts, decisions = _screen_inputs()
    examples, provenance = build_cut_aligned_video_state_examples(
        embedding_artifacts=artifacts,
        base_decisions=decisions,
        followup_decisions=[],
        sealed_blind_video_sha256s=["e" * 64],
    )
    result = screen_cut_aligned_video_state_examples(
        examples=examples,
        provenance=provenance,
        label_correction_artifact_sha256s=["3" * 64, "4" * 64],
        configurations=[
            {
                "representation": "neighbor_summary",
                "c": 0.1,
                "pca_components": 2,
            }
        ],
    )

    assert result["accepted"] is True
    assert result["target_example_count"] == 16
    assert all(
        fold["held_target_group"] not in fold["training_target_groups"]
        and fold["held_target_group"] not in fold["selection_target_groups"]
        for fold in result["outer_folds"]
    )
    verify_cut_aligned_video_state_screen(result)
    tampered = copy.deepcopy(result)
    tampered["accepted"] = False
    with pytest.raises(ValueError):
        verify_cut_aligned_video_state_screen(tampered)


def test_cut_aligned_screen_rejects_sealed_blind_embedding_source() -> None:
    artifacts, decisions = _screen_inputs()
    with pytest.raises(ValueError, match="sealed blind"):
        build_cut_aligned_video_state_examples(
            embedding_artifacts=artifacts,
            base_decisions=decisions,
            followup_decisions=[],
            sealed_blind_video_sha256s=["a" * 64],
        )


def test_cut_aligned_screen_rejects_empty_backbone_provenance() -> None:
    artifacts, decisions = _screen_inputs()
    examples, provenance = build_cut_aligned_video_state_examples(
        embedding_artifacts=artifacts,
        base_decisions=decisions,
        followup_decisions=[],
        sealed_blind_video_sha256s=["e" * 64],
    )
    provenance["backbone"] = ""

    with pytest.raises(ValueError, match="backbone provenance"):
        screen_cut_aligned_video_state_examples(
            examples=examples,
            provenance=provenance,
            label_correction_artifact_sha256s=["3" * 64, "4" * 64],
            configurations=[
                {
                    "representation": "neighbor_summary",
                    "c": 0.1,
                    "pca_components": 2,
                }
            ],
        )
