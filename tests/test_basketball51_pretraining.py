from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch import nn


def _module() -> ModuleType:
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "train_basketball51_field_goal_backbone.py"
    )
    spec = importlib.util.spec_from_file_location(
        "train_basketball51_field_goal_backbone",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _TinyVideoModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])
        self.norm = nn.LayerNorm(2)
        self.head = nn.Sequential(nn.Dropout(), nn.Linear(2, 400))


def test_group_split_is_disjoint_deterministic_and_label_complete() -> None:
    rows = [
        {"source_group": f"g{group}", "label": label}
        for group in range(10)
        for label in ("2p0", "2p1", "ft0", "ft1")
    ]

    first = _module().split_source_groups(rows, fold_count=5, held_fold=2)
    second = _module().split_source_groups(rows, fold_count=5, held_fold=2)

    assert first == second
    train, held = first
    assert set(train).isdisjoint(held)
    assert {rows[index]["label"] for index in train} == {
        "2p0",
        "2p1",
        "ft0",
        "ft1",
    }
    assert {rows[index]["label"] for index in held} == {
        "2p0",
        "2p1",
        "ft0",
        "ft1",
    }


def test_configure_trainable_tail_only_unfreezes_requested_blocks() -> None:
    model = _TinyVideoModel()

    count = _module().configure_trainable_tail(model, trainable_blocks=1)

    assert count > 0
    assert not any(parameter.requires_grad for parameter in model.blocks[0].parameters())
    assert not any(parameter.requires_grad for parameter in model.blocks[1].parameters())
    assert all(parameter.requires_grad for parameter in model.blocks[2].parameters())
    assert all(parameter.requires_grad for parameter in model.norm.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())
    assert model.head[-1].out_features == 2


def test_balanced_sample_weights_equalize_free_throw_and_field_goal_mass() -> None:
    rows = [
        {"label": "ft0"},
        {"label": "ft1"},
        {"label": "2p0"},
        {"label": "2p1"},
        {"label": "3p0"},
        {"label": "3p1"},
    ]

    weights = _module().build_balanced_sample_weights(rows, list(range(len(rows))))

    assert isinstance(weights, torch.Tensor)
    assert weights.dtype == torch.double
    assert weights.tolist() == pytest.approx([0.5, 0.5, 0.25, 0.25, 0.25, 0.25])
    assert sum(weights[:2]).item() == pytest.approx(sum(weights[2:]).item())


def test_checkpoint_payload_is_training_only_and_provenance_bound() -> None:
    model = _TinyVideoModel()
    _module().configure_trainable_tail(model, trainable_blocks=1)

    payload = _module().build_checkpoint_payload(
        model,
        subset_manifest_sha256="subset",
        official_checkpoint_sha256="official",
        backbone="torchvision/mvit_v2_s/kinetics400_v1",
        trainable_blocks=1,
        held_source_groups=["g1", "g2"],
        metrics={"precision": 0.9, "recall": 0.9, "f1": 0.9},
    )

    assert payload["schema_version"] == "agu.basketball51-pretrained-backbone.v1"
    assert payload["runtime_consumable"] is False
    assert payload["promotion_eligible"] is False
    assert payload["subset_manifest_sha256"] == "subset"
    assert payload["official_checkpoint_sha256"] == "official"
    assert payload["classifier_classes"] == ["free_throw", "field_goal"]
    assert isinstance(payload["model_state_dict"], dict)


def test_checkpoint_output_must_not_preexist(tmp_path: Path) -> None:
    output = tmp_path / "candidate.pth"
    output.write_bytes(b"stale checkpoint")

    with pytest.raises(FileExistsError, match="already exists"):
        _module().require_new_checkpoint_output(output)

    assert output.read_bytes() == b"stale checkpoint"


def test_new_checkpoint_output_is_accepted(tmp_path: Path) -> None:
    output = tmp_path / "candidate.pth"

    _module().require_new_checkpoint_output(output)

    assert not output.exists()
