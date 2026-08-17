from __future__ import annotations

import copy

import pytest

from scripts.extract_bard_nonshot_mvit_probe import (
    normalized_full_clip_frame_indexes,
    seal_probe_artifact,
    verify_probe_artifact,
)


def test_normalized_sampling_is_deterministic_and_codec_safe() -> None:
    indexes = normalized_full_clip_frame_indexes(100, clip_frames=16)

    assert indexes == tuple(round(value) for value in __import__("numpy").linspace(4.95, 94.05, 16))
    assert len(indexes) == 16
    assert len(set(indexes)) == 16
    assert indexes[0] >= 0
    assert indexes[-1] < 100


def test_probe_artifact_hash_and_runtime_boundary_are_verified() -> None:
    artifact = seal_probe_artifact(
        {
            "source_manifest_sha256": "a" * 64,
            "backbone": "torchvision/mvit_v2_s/kinetics400_v1",
            "backbone_sha256": "b" * 64,
            "embedding_dimension": 768,
            "clip_frames": 16,
            "clip_count_expected": 1,
            "clip_count_completed": 1,
            "complete": True,
            "examples": [
                {
                    "clip_id": "example",
                    "source_video_sha256": "c" * 64,
                    "frame_indexes": list(range(16)),
                    "embedding": [0.0] * 768,
                }
            ],
        }
    )

    assert verify_probe_artifact(artifact) == artifact
    assert artifact["runtime_consumable"] is False
    assert artifact["codex_runtime_answer_used"] is False

    tampered = copy.deepcopy(artifact)
    tampered["examples"][0]["embedding"][0] = 1.0
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_probe_artifact(tampered)
