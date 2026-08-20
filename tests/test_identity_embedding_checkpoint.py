from __future__ import annotations

from pathlib import Path

import pytest
import torch

from app.analysis.identity_embedding import TorchvisionMobileNetV3SmallEmbedder
from app.analysis.reid_training import REID_CHECKPOINT_SCHEMA


def test_mobilenet_embedder_loads_hash_identified_agu_checkpoint(tmp_path: Path) -> None:
    from torchvision.models import mobilenet_v3_small

    model = mobilenet_v3_small(weights=None)
    features = torch.nn.Sequential(model.features, model.avgpool, torch.nn.Flatten())
    path = tmp_path / "reid.pt"
    torch.save(
        {
            "schema_version": REID_CHECKPOINT_SCHEMA,
            "architecture": "torchvision_mobilenet_v3_small",
            "feature_state_dict": features.state_dict(),
            "training_manifest_sha256": "a" * 64,
        },
        path,
    )
    embedder = TorchvisionMobileNetV3SmallEmbedder(str(path), "cpu", batch_size=1)
    assert "manifest_aaaaaaaaaaaa" in embedder.model_id


def test_mobilenet_embedder_rejects_unrecognized_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"schema_version": "other"}, path)
    with pytest.raises(ValueError, match="unsupported AGU ReID checkpoint"):
        TorchvisionMobileNetV3SmallEmbedder(str(path), "cpu", batch_size=1)
