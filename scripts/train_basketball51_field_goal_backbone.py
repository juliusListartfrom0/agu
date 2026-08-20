#!/usr/bin/env python3
"""Fine-tune an MViT tail on Basketball-51 field-goal versus free-throw clips."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.basketball51 import verify_basketball51_subset_manifest  # noqa: E402
from app.analysis.shot_validity_video_backbone import file_sha256  # noqa: E402

SCHEMA = "agu.basketball51-pretrained-backbone.v1"
BACKBONE = "torchvision/mvit_v2_s/kinetics400_v1"


def require_new_checkpoint_output(path: Path) -> None:
    """Refuse to overwrite or silently retain a checkpoint from an earlier run."""
    if path.exists():
        raise FileExistsError(
            f"checkpoint output already exists; choose a new path or archive it first: {path}"
        )


def split_source_groups(
    rows: Sequence[Mapping[str, object]],
    *,
    fold_count: int,
    held_fold: int,
) -> tuple[list[int], list[int]]:
    if fold_count < 3 or not 0 <= held_fold < fold_count:
        raise ValueError("invalid source-group fold")
    labels = np.asarray([str(row["label"]) for row in rows])
    groups = np.asarray([str(row["source_group"]) for row in rows])
    splitter = StratifiedGroupKFold(
        n_splits=fold_count,
        shuffle=True,
        random_state=0,
    )
    folds = list(splitter.split(np.zeros(len(rows)), labels, groups))
    train, held = folds[held_fold]
    if set(groups[train]) & set(groups[held]):
        raise ValueError("Basketball-51 source groups overlap")
    if set(labels[train]) != set(labels) or set(labels[held]) != set(labels):
        raise ValueError("every split must contain every Basketball-51 label")
    return train.tolist(), held.tolist()


def configure_trainable_tail(model: nn.Module, *, trainable_blocks: int) -> int:
    blocks = getattr(model, "blocks")
    if not 1 <= trainable_blocks <= len(blocks):
        raise ValueError("invalid trainable block count")
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.head[-1] = nn.Linear(model.head[-1].in_features, 2)
    for block in blocks[-trainable_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for module in (model.norm, model.head):
        for parameter in module.parameters():
            parameter.requires_grad = True
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_balanced_sample_weights(
    rows: Sequence[Mapping[str, object]],
    indexes: Sequence[int],
) -> torch.Tensor:
    """Give free throws and field goals equal expected sampling mass."""

    binary_labels = [
        "free_throw" if str(rows[index]["label"]).startswith("ft") else "field_goal"
        for index in indexes
    ]
    counts = Counter(binary_labels)
    if set(counts) != {"free_throw", "field_goal"}:
        raise ValueError("balanced Basketball-51 training requires both classes")
    return torch.tensor(
        [1.0 / counts[label] for label in binary_labels],
        dtype=torch.double,
    )


def build_checkpoint_payload(
    model: nn.Module,
    *,
    subset_manifest_sha256: str,
    official_checkpoint_sha256: str,
    backbone: str,
    trainable_blocks: int,
    held_source_groups: Sequence[str],
    metrics: Mapping[str, float],
) -> dict[str, Any]:
    if not subset_manifest_sha256 or not official_checkpoint_sha256:
        raise ValueError("pretrained backbone requires source provenance")
    return {
        "schema_version": SCHEMA,
        "purpose": "source_domain_pretraining_only",
        "runtime_consumable": False,
        "promotion_eligible": False,
        "promotion_exclusion_reason": "requires target-domain game-disjoint validation",
        "subset_manifest_sha256": subset_manifest_sha256,
        "official_checkpoint_sha256": official_checkpoint_sha256,
        "backbone": backbone,
        "classifier_classes": ["free_throw", "field_goal"],
        "trainable_blocks": trainable_blocks,
        "held_source_groups": sorted(held_source_groups),
        "source_validation_metrics": dict(metrics),
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }


class _ClipDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        root: Path,
        rows: Sequence[Mapping[str, object]],
        indexes: Sequence[int],
        *,
        clip_frames: int,
    ) -> None:
        self.root = root
        self.rows = rows
        self.indexes = list(indexes)
        self.clip_frames = clip_frames
        self.transform = MViT_V2_S_Weights.KINETICS400_V1.transforms()

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[self.indexes[item]]
        path = self.root / str(row["relative_path"])
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"cannot open Basketball-51 clip: {path.name}")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        try:
            for index in np.rint(np.linspace(0, count - 1, self.clip_frames)).astype(int):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"cannot decode Basketball-51 clip: {path.name}")
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        clip = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
        target = 0 if str(row["label"]).startswith("ft") else 1
        return self.transform(clip), torch.tensor(target, dtype=torch.long)


def _predict(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> tuple[list[int], list[int]]:
    truth: list[int] = []
    prediction: list[int] = []
    model.eval()
    with torch.inference_mode():
        for clips, targets in loader:
            values = model(clips.to(device)).argmax(dim=1).cpu().tolist()
            truth.extend(targets.tolist())
            prediction.extend(values)
    return truth, prediction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--trainable-blocks", type=int, default=1)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--held-fold", type=int, default=0)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.clip_frames) < 1:
        raise ValueError("training dimensions must be positive")
    require_new_checkpoint_output(args.checkpoint_output)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest = verify_basketball51_subset_manifest(
        json.loads(args.subset_manifest.read_text(encoding="utf-8"))
    )
    rows = manifest["clips"]
    for row in rows:
        path = args.subset_manifest.parent / row["relative_path"]
        if not path.is_file() or file_sha256(path) != row["sha256"]:
            raise ValueError(f"Basketball-51 clip hash mismatch: {row['relative_path']}")
    train, held = split_source_groups(
        rows,
        fold_count=args.fold_count,
        held_fold=args.held_fold,
    )
    device = torch.device(args.device)
    model = mvit_v2_s(weights=None)
    model.load_state_dict(
        torch.load(args.official_checkpoint, map_location="cpu", weights_only=True)
    )
    trainable = configure_trainable_tail(
        model,
        trainable_blocks=args.trainable_blocks,
    )
    model.to(device)
    train_loader = DataLoader(
        _ClipDataset(
            args.subset_manifest.parent,
            rows,
            train,
            clip_frames=args.clip_frames,
        ),
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            build_balanced_sample_weights(rows, train),
            num_samples=len(train),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        ),
        num_workers=0,
    )
    held_loader = DataLoader(
        _ClipDataset(
            args.subset_manifest.parent,
            rows,
            held,
            clip_frames=args.clip_frames,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for clips, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(clips.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(targets)
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "loss": total_loss / len(train),
                    "trainable_parameters": trainable,
                }
            ),
            flush=True,
        )
    truth, prediction = _predict(model, held_loader, device)
    metrics = {
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
    }
    payload = build_checkpoint_payload(
        model,
        subset_manifest_sha256=manifest["artifact_sha256"],
        official_checkpoint_sha256=file_sha256(args.official_checkpoint),
        backbone=BACKBONE,
        trainable_blocks=args.trainable_blocks,
        held_source_groups=sorted({str(rows[index]["source_group"]) for index in held}),
        metrics=metrics,
    )
    metadata = {key: value for key, value in payload.items() if key != "model_state_dict"}
    metadata["training_sampling"] = "inverse_binary_class_frequency_with_replacement"
    gate_passed = metrics["precision"] >= 0.85 and metrics["recall"] >= 0.85
    metadata["source_validation_gate_passed"] = gate_passed
    metadata["checkpoint_emitted"] = gate_passed
    if gate_passed:
        args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.checkpoint_output)
        metadata["checkpoint_sha256"] = file_sha256(args.checkpoint_output)
    else:
        metadata["checkpoint_sha256"] = None
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics), flush=True)
    return 0 if gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
