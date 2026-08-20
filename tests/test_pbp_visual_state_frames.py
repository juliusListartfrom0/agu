from __future__ import annotations

import copy

import pytest

from app.analysis.pbp_visual_state_frames import (
    ANCHOR_OFFSETS_SECONDS,
    DINO_V2_SMALL_BACKBONE,
    POST_ANCHOR_OFFSETS_SECONDS,
    PRE_ANCHOR_OFFSETS_SECONDS,
    WIDE_ANCHOR_OFFSETS_SECONDS,
    anchor_frame_indexes,
    seal_anchor_state_embedding_artifact,
    verify_anchor_state_embedding_artifact,
)
from app.analysis.shot_validity_scene_state import SCENE_EMBEDDING_DIMENSION


def _artifact() -> dict:
    return seal_anchor_state_embedding_artifact(
        {
            "purpose": "test",
            "truth_used_for_training_only": True,
            "codex_runtime_answer_used": False,
            "training_manifest_sha256s": ["1" * 64],
            "clock_artifact_sha256s": ["2" * 64],
            "source_video_sha256s": ["3" * 64],
            "sealed_blind_video_sha256s": ["4" * 64, "5" * 64],
            "backbone_sha256": "6" * 64,
            "anchor_offsets_seconds": list(ANCHOR_OFFSETS_SECONDS),
            "examples": [
                {
                    "source_video_sha256": "3" * 64,
                    "candidate_bundle_sha256": "7" * 64,
                    "event_id": "event-1",
                    "state": "free_throw",
                    "anchor_frame": 300,
                    "source_fps": 30.0,
                    "frame_count": 1000,
                    "frame_indexes": [240, 300, 360],
                    "embeddings": [
                        [float(index)] * SCENE_EMBEDDING_DIMENSION
                        for index in range(3)
                    ],
                }
            ],
        }
    )


def test_anchor_frame_indexes_use_fixed_seconds_around_anchor() -> None:
    assert anchor_frame_indexes(
        anchor_frame=300,
        source_fps=30.0,
        frame_count=1000,
    ) == (240, 300, 360)


def test_anchor_frame_indexes_support_pre_anchor_state_protocol() -> None:
    assert anchor_frame_indexes(
        anchor_frame=300,
        source_fps=30.0,
        frame_count=1000,
        offsets_seconds=PRE_ANCHOR_OFFSETS_SECONDS,
    ) == (180, 210, 240, 270, 300)


def test_anchor_frame_indexes_support_post_anchor_state_protocol() -> None:
    assert anchor_frame_indexes(
        anchor_frame=300,
        source_fps=30.0,
        frame_count=1000,
        offsets_seconds=POST_ANCHOR_OFFSETS_SECONDS,
    ) == (300, 450, 600, 750, 900)


def test_anchor_frame_indexes_support_wide_centered_state_protocol() -> None:
    assert anchor_frame_indexes(
        anchor_frame=300,
        source_fps=10.0,
        frame_count=1000,
        offsets_seconds=WIDE_ANCHOR_OFFSETS_SECONDS,
    ) == (220, 240, 260, 280, 300, 320, 340, 360, 380)


@pytest.mark.parametrize(
    ("anchor", "fps", "count"),
    [
        (10, 30.0, 1000),
        (990, 30.0, 1000),
        (300, 0.0, 1000),
        (300, 30.0, 0),
    ],
)
def test_anchor_frame_indexes_reject_incomplete_temporal_windows(
    anchor: int,
    fps: float,
    count: int,
) -> None:
    with pytest.raises(ValueError):
        anchor_frame_indexes(
            anchor_frame=anchor,
            source_fps=fps,
            frame_count=count,
        )


def test_anchor_state_artifact_is_training_only_and_sealed() -> None:
    artifact = _artifact()

    assert artifact["schema_version"] == "agu.pbp-anchor-state-embeddings.v1"
    assert artifact["runtime_consumable"] is False
    assert artifact["backbone"] == "torchvision/mobilenet_v3_small/imagenet1k_v1"
    verify_anchor_state_embedding_artifact(artifact)


def test_anchor_state_artifact_accepts_supported_dinov2_backbone() -> None:
    payload = _artifact()
    payload.pop("artifact_sha256")
    payload["backbone"] = DINO_V2_SMALL_BACKBONE
    payload["examples"][0]["embeddings"] = [
        [float(index)] * 384 for index in range(3)
    ]

    artifact = seal_anchor_state_embedding_artifact(payload)

    assert artifact["backbone"] == DINO_V2_SMALL_BACKBONE
    assert artifact["embedding_dimension"] == 384
    verify_anchor_state_embedding_artifact(artifact)


def test_anchor_state_artifact_rejects_blind_or_foul_examples() -> None:
    payload = _artifact()
    payload.pop("artifact_sha256")
    payload["examples"][0]["source_video_sha256"] = "4" * 64
    with pytest.raises(ValueError, match="sealed blind"):
        seal_anchor_state_embedding_artifact(payload)

    payload = _artifact()
    payload.pop("artifact_sha256")
    payload["examples"][0]["state"] = "foul_only"
    with pytest.raises(ValueError, match="state"):
        seal_anchor_state_embedding_artifact(payload)


def test_anchor_state_artifact_rejects_hash_tampering() -> None:
    artifact = _artifact()
    tampered = copy.deepcopy(artifact)
    tampered["examples"][0]["embeddings"][0][0] = -1.0

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_anchor_state_embedding_artifact(tampered)
