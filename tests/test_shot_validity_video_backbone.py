from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from app.analysis.schemas import GameEventResponse
from app.analysis.shot_validity_video_backbone import (
    VIDEO_BACKBONES,
    _release_accelerator_cache,
    extract_video_embeddings,
    get_video_backbone_spec,
    seal_video_embedding_artifact,
    verify_video_embedding_artifact,
)
from scripts.screen_shot_validity_video_embeddings import screen_video_embeddings


def _artifact() -> dict[str, object]:
    backbone = "torchvision/swin3d_t/kinetics400_v1"
    dimension = get_video_backbone_spec(backbone).embedding_dimension
    examples = []
    for game, offset in (("game-a", 0.0), ("game-b", 0.1), ("game-c", -0.1)):
        for index in range(3):
            examples.append(
                {
                    "source_video_sha256": game,
                    "event_id": f"{game}-positive-{index}",
                    "event_present": True,
                    "embedding": [2.0 + offset + index * 0.1] * dimension,
                }
            )
            examples.append(
                {
                    "source_video_sha256": game,
                    "event_id": f"{game}-negative-{index}",
                    "event_present": False,
                    "embedding": [-2.0 + offset - index * 0.1] * dimension,
                }
            )
    return seal_video_embedding_artifact(
        {
            "purpose": "backbone_screening_training_only",
            "training_manifest_sha256": "manifest",
            "backbone": backbone,
            "backbone_sha256": "checkpoint",
            "embedding_dimension": dimension,
            "clip_frames": 16,
            "examples": examples,
        }
    )


def test_video_backbone_registry_has_open_source_candidates() -> None:
    assert set(VIDEO_BACKBONES) == {
        "torchvision/mvit_v2_s/kinetics400_v1",
        "torchvision/swin3d_t/kinetics400_v1",
    }
    assert all(spec.license.startswith("BSD-3-Clause") for spec in VIDEO_BACKBONES.values())


def test_accelerator_cache_release_is_mps_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append("mps"))

    _release_accelerator_cache(torch.device("cpu"))
    assert calls == []

    _release_accelerator_cache(torch.device("mps"))
    assert calls == ["mps"]


def test_embedding_flush_releases_accelerator_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.analysis import shot_validity_video_backbone as module

    class FakeCapture:
        def isOpened(self) -> bool:
            return True

        def set(self, *_args: object) -> bool:
            return True

        def read(self) -> tuple[bool, np.ndarray]:
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        def release(self) -> None:
            return None

    class FakeBackbone(nn.Module):
        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            return torch.ones((batch.shape[0], 2), dtype=torch.float32)

    released: list[torch.device] = []
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda _path: FakeCapture())
    monkeypatch.setattr(module.cv2, "cvtColor", lambda frame, _code: frame)
    monkeypatch.setattr(
        module,
        "_release_accelerator_cache",
        lambda device: released.append(device),
    )
    event = GameEventResponse(
        event_id="event-1",
        revision=1,
        event_type="field_goal_attempt",
        source_video_id="video_001",
        start_frame=0,
        end_frame=15,
        status="candidate",
        confidence=0.5,
    )

    result = extract_video_embeddings(
        video_path=Path("unused.mp4"),
        events=[event],
        backbone=FakeBackbone(),
        transform=lambda _clip: torch.zeros((3, 16, 2, 2)),
        device=torch.device("cpu"),
        clip_frames=16,
        batch_size=1,
    )

    assert result == [[1.0, 1.0]]
    assert released == [torch.device("cpu")]


def test_video_embedding_artifact_is_hash_bound() -> None:
    artifact = _artifact()
    assert verify_video_embedding_artifact(artifact)["runtime_consumable"] is False
    artifact["clip_frames"] = 8
    with pytest.raises(ValueError, match="shorter"):
        verify_video_embedding_artifact(artifact)


def test_screening_uses_game_level_oof_and_strict_gate(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")

    result = screen_video_embeddings(path)

    assert result["gate"]["promoted"] is True
    assert len(result["folds"]) == 3
    assert all(row["roc_auc"] == 1.0 for row in result["folds"])
    assert result["runtime_consumable"] is False


def test_screening_can_fuse_aligned_backbones(tmp_path: Path) -> None:
    first = _artifact()
    second = _artifact()
    second["backbone"] = "torchvision/mvit_v2_s/kinetics400_v1"
    second["backbone_sha256"] = "mvit-checkpoint"
    second = seal_video_embedding_artifact(
        {key: value for key, value in second.items() if key != "artifact_sha256"}
    )
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")

    result = screen_video_embeddings([first_path, second_path])

    assert result["gate"]["promoted"] is True
    assert result["purpose"] == "backbone_fusion_screening_training_only"
    assert result["backbones"] == [
        "torchvision/swin3d_t/kinetics400_v1",
        "torchvision/mvit_v2_s/kinetics400_v1",
    ]


def test_screening_rejects_misaligned_fusion_examples(tmp_path: Path) -> None:
    first = _artifact()
    second = _artifact()
    second["examples"] = second["examples"][:-1]
    second = seal_video_embedding_artifact(
        {key: value for key, value in second.items() if key != "artifact_sha256"}
    )
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")

    with pytest.raises(ValueError, match="identical examples"):
        screen_video_embeddings([first_path, second_path])
