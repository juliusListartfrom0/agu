from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from app.analysis.reid_training import (
    batch_hard_triplet_loss,
    load_training_manifest,
    split_samples_by_sequence,
    validate_reid_promotion,
)


def _write_manifest(path: Path, **overrides: object) -> None:
    payload = {
        "training_only": True,
        "acceptance_media_allowed": False,
        "benchmark_disjoint": True,
        "source_catalog_sha256": "catalog",
        "samples": [
            {"identity_id": "p0", "path": "train/a.jpg"},
            {"identity_id": "p0", "path": "validation/b.jpg"},
        ],
        **overrides,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["manifest_sha256"] = hashlib.sha256(serialized.encode()).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_manifest_is_hash_bound_and_sequence_split_is_disjoint(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    _write_manifest(path)
    payload = load_training_manifest(path)
    train, validation = split_samples_by_sequence(payload, ["validation"])
    assert [sample["path"] for sample in train] == ["train/a.jpg"]
    assert [sample["path"] for sample in validation] == ["validation/b.jpg"]


def test_manifest_rejects_acceptance_media(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    _write_manifest(path, acceptance_media_allowed=True)
    with pytest.raises(ValueError, match="acceptance-forbidden"):
        load_training_manifest(path)


def test_batch_hard_triplet_loss_rewards_separated_identities() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    separated = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    collapsed = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]])
    assert batch_hard_triplet_loss(separated, labels) < batch_hard_triplet_loss(collapsed, labels)


def test_reid_promotion_requires_absolute_gate_and_baseline_improvement() -> None:
    validate_reid_promotion(0.44, 0.86, 0.85)
    with pytest.raises(RuntimeError, match="rejected by the held-out gate"):
        validate_reid_promotion(0.44, 0.84, 0.85)
    with pytest.raises(RuntimeError, match="rejected by the held-out gate"):
        validate_reid_promotion(0.90, 0.90, 0.85)
