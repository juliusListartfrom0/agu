#!/usr/bin/env python3
"""Train a game-held lightweight temporal state head on reviewed contact sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v3_small

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.ebqwen_visual_state_lora import (  # noqa: E402
    verify_lora_dataset_manifest,
)
from app.analysis.shot_validity_scene_state import file_sha256  # noqa: E402
from app.analysis.visual_state_temporal_head import (  # noqa: E402
    VisualStateRecord,
    VisualStateTemporalClassifier,
    load_visual_state_records,
    prepare_contact_sheet_tensor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--trainable-stage",
        choices=("head", "last_block"),
        default="last_block",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


class ContactSheetDataset(Dataset):
    def __init__(
        self,
        records: list[VisualStateRecord],
        *,
        training: bool,
        seed: int,
    ) -> None:
        self.records = records
        self.training = training
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode contact sheet: {record.image_path.name}")
        generator = torch.Generator().manual_seed(self.seed + index)
        return (
            prepare_contact_sheet_tensor(
                image,
                training=self.training,
                generator=generator,
            ),
            record.target,
        )


def main() -> int:
    args = parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.head_learning_rate <= 0
        or args.backbone_learning_rate <= 0
        or args.torch_threads < 1
    ):
        raise ValueError("training parameters must be positive")
    torch.set_num_threads(args.torch_threads)
    _set_seed(0)
    manifest = verify_lora_dataset_manifest(
        json.loads(args.manifest.read_text(encoding="utf-8"))
    )
    records = load_visual_state_records(
        manifest,
        root=args.manifest.parent,
    )
    groups = sorted({record.source_video_sha256 for record in records})
    if len(groups) < 3:
        raise ValueError("visual-state training requires at least three games")
    if any(
        {record.target for record in records if record.source_video_sha256 == group}
        != {0, 1}
        for group in groups
    ):
        raise ValueError("every held game must contain both visual states")
    device = _resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    all_labels = []
    all_probabilities = []
    for fold_index, held_group in enumerate(groups):
        seed = 1000 + fold_index
        _set_seed(seed)
        train_records = [
            record
            for record in records
            if record.source_video_sha256 != held_group
        ]
        test_records = [
            record
            for record in records
            if record.source_video_sha256 == held_group
        ]
        model = _load_model(
            args.backbone_checkpoint,
            device=device,
            trainable_stage=args.trainable_stage,
        )
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": model.head.parameters(),
                    "lr": args.head_learning_rate,
                },
                {
                    "params": [
                        parameter
                        for parameter in model.encoder.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": args.backbone_learning_rate,
                },
            ],
            weight_decay=1e-4,
        )
        counts = np.bincount(
            [record.target for record in train_records],
            minlength=2,
        )
        class_weights = torch.tensor(
            [len(train_records) / (2 * count) for count in counts],
            dtype=torch.float32,
            device=device,
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        losses = []
        for epoch in range(args.epochs):
            model.train()
            _freeze_batch_norm(model)
            loader = DataLoader(
                ContactSheetDataset(
                    train_records,
                    training=True,
                    seed=seed + epoch * 10000,
                ),
                batch_size=args.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed + epoch),
                num_workers=0,
            )
            total_loss = 0.0
            total_examples = 0
            for frames, labels in loader:
                frames = frames.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(frames), labels)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(labels)
                total_examples += len(labels)
            losses.append(total_loss / total_examples)
            print(
                json.dumps(
                    {
                        "stage": "visual_state_training",
                        "fold": fold_index + 1,
                        "fold_count": len(groups),
                        "epoch": epoch + 1,
                        "loss": losses[-1],
                    }
                ),
                flush=True,
            )
        labels, probabilities = _predict(
            model,
            test_records,
            device=device,
            batch_size=args.batch_size,
            seed=seed,
        )
        metrics = _metrics(labels, probabilities)
        checkpoint_path = args.output_dir / f"fold-{fold_index + 1:02d}.pt"
        torch.save(
            {
                "architecture": "mobilenet_v3_small_temporal_formation_v1",
                "trainable_stage": args.trainable_stage,
                "held_out_source_video_sha256": held_group,
                "state_dict": model.state_dict(),
            },
            checkpoint_path,
        )
        folds.append(
            {
                "held_out_source_video_sha256": held_group,
                "train_examples": len(train_records),
                "test_examples": len(test_records),
                "losses": losses,
                "metrics": metrics,
                "checkpoint": checkpoint_path.name,
                "checkpoint_sha256": file_sha256(checkpoint_path),
            }
        )
        all_labels.extend(labels)
        all_probabilities.extend(probabilities)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    weakest = min(float(fold["metrics"]["balanced_accuracy"]) for fold in folds)
    artifact = {
        "schema_version": "agu.visual-state-temporal-head-screen.v1",
        "purpose": "offline_candidate_feature_acceptance_only",
        "runtime_consumable": False,
        "truth_used_for_training_only": True,
        "codex_runtime_answer_used": False,
        "dataset_artifact_sha256": manifest["artifact_sha256"],
        "backbone": "torchvision/mobilenet_v3_small/imagenet1k_v1",
        "backbone_sha256": file_sha256(args.backbone_checkpoint),
        "protocol": {
            "outer_validation": "leave_one_source_video_out",
            "frame_count": 5,
            "temporal_pooling": "mean_std_last_minus_first",
            "trainable_stage": args.trainable_stage,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "head_learning_rate": args.head_learning_rate,
            "backbone_learning_rate": args.backbone_learning_rate,
            "threshold": 0.5,
            "acceptance_metric": "worst_game_balanced_accuracy",
            "acceptance_threshold": 0.85,
        },
        "folds": folds,
        "pooled_metrics": _metrics(all_labels, all_probabilities),
        "worst_game_balanced_accuracy": weakest,
        "accepted": weakest >= 0.85,
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    _write_json(args.output_dir / "screen.json", artifact)
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "worst_game_balanced_accuracy": weakest,
                "accepted": artifact["accepted"],
                "artifact_sha256": artifact["artifact_sha256"],
            }
        )
    )
    return 0


def _load_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
    trainable_stage: str,
) -> VisualStateTemporalClassifier:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    base = mobilenet_v3_small(weights=None, progress=False)
    base.load_state_dict(state)
    encoder = nn.Sequential(base.features, base.avgpool, nn.Flatten())
    model = VisualStateTemporalClassifier(
        encoder,
        embedding_dimension=576,
    )
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    if trainable_stage == "last_block":
        for parameter in model.encoder[0][-1].parameters():
            parameter.requires_grad = True
    return model.to(device)


def _freeze_batch_norm(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def _predict(
    model: nn.Module,
    records: list[VisualStateRecord],
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> tuple[list[int], list[float]]:
    model.eval()
    labels = []
    probabilities = []
    loader = DataLoader(
        ContactSheetDataset(records, training=False, seed=seed),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    with torch.inference_mode():
        for frames, batch_labels in loader:
            logits = model(frames.to(device))
            values = torch.softmax(logits, dim=1)[:, 1]
            labels.extend(int(value) for value in batch_labels)
            probabilities.extend(float(value) for value in values.cpu())
    return labels, probabilities


def _metrics(labels: list[int], probabilities: list[float]) -> dict[str, float]:
    guesses = [value >= 0.5 for value in probabilities]
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, guesses)),
        "precision": float(precision_score(labels, guesses, zero_division=0)),
        "recall": float(recall_score(labels, guesses, zero_division=0)),
        "f1": float(f1_score(labels, guesses, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
