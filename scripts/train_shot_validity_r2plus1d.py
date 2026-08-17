#!/usr/bin/env python3
"""Fine-tune R(2+1)D with game-level OOF calibration on sealed raw windows."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.shot_validity_finetune import (  # noqa: E402
    ShotWindowRecord,
    decode_dense_windows,
    evaluate_game_gate,
    freeze_batch_norm_stats,
    load_finetune_model,
    load_training_records,
    normalize_probability_by_threshold,
    prepare_window,
    seal_finetuned_model_metadata,
    select_precision_threshold,
)
from app.analysis.shot_validity_temporal import file_sha256  # noqa: E402


class _WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        windows: list[torch.Tensor],
        records: list[ShotWindowRecord],
        indexes: list[int],
        *,
        training: bool,
        seed: int,
        preprocessing: str,
        temporal_sampling: str,
    ) -> None:
        self.windows = windows
        self.records = records
        self.indexes = indexes
        self.training = training
        self.generator = torch.Generator().manual_seed(seed)
        self.preprocessing = preprocessing
        self.temporal_sampling = temporal_sampling

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = self.indexes[item]
        clip = prepare_window(
            self.windows[index],
            training=self.training,
            generator=self.generator,
            preprocessing=self.preprocessing,
            temporal_sampling=self.temporal_sampling,
            anchor_fraction=(
                (self.records[index].anchor_frame - self.records[index].start_frame)
                / max(self.records[index].end_frame - self.records[index].start_frame, 1)
                if self.records[index].anchor_frame is not None
                else None
            ),
        )
        label = torch.tensor(int(self.records[index].event_present), dtype=torch.long)
        return clip, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-bundle", type=Path, action="append", required=True)
    parser.add_argument("--annotation", type=Path, action="append", required=True)
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--trainable-stage", choices=("fc", "layer4", "layer3"), default="layer4")
    parser.add_argument("--checkpoint-format", choices=("kinetics", "agu"), default="kinetics")
    parser.add_argument("--preprocessing", choices=("kinetics", "agu_v3"), default="kinetics")
    parser.add_argument(
        "--temporal-sampling",
        choices=("uniform", "anchor"),
        default="uniform",
        help="Training-only temporal sampler; anchor uses raw candidate evidence frame when available.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _fit(
    *,
    windows: list[torch.Tensor],
    records: list[ShotWindowRecord],
    train_indexes: list[int],
    checkpoint_path: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    trainable_stage: str,
    seed: int,
    checkpoint_format: str,
    preprocessing: str,
    temporal_sampling: str,
) -> nn.Module:
    _seed_everything(seed)
    model = load_finetune_model(
        checkpoint_path,
        device=device,
        trainable_stage=trainable_stage,
        checkpoint_format=checkpoint_format,
    )
    labels = [records[index].event_present for index in train_indexes]
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("every training fold requires positive and negative windows")
    class_weights = torch.tensor(
        [1.0, negatives / positives], dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    backbone_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (head_parameters if name.startswith("fc.") else backbone_parameters).append(parameter)
    groups = [{"params": head_parameters, "lr": learning_rate * 5.0}]
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    dataset = _WindowDataset(
        windows,
        records,
        train_indexes,
        training=True,
        seed=seed + 1,
        preprocessing=preprocessing,
        temporal_sampling=temporal_sampling,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 2),
        num_workers=0,
    )
    for epoch in range(epochs):
        model.train()
        freeze_batch_norm_stats(model)
        running_loss = 0.0
        for clips, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(clips.to(device))
            loss = criterion(logits, targets.to(device))
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * len(targets)
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "training_examples": len(dataset),
                    "loss": running_loss / len(dataset),
                }
            ),
            flush=True,
        )
    return model.eval()


def _predict(
    model: nn.Module,
    *,
    windows: list[torch.Tensor],
    records: list[ShotWindowRecord],
    indexes: list[int],
    device: torch.device,
    batch_size: int,
    seed: int,
    preprocessing: str,
    temporal_sampling: str,
) -> list[float]:
    dataset = _WindowDataset(
        windows,
        records,
        indexes,
        training=False,
        seed=seed,
        preprocessing=preprocessing,
        temporal_sampling=temporal_sampling,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities: list[float] = []
    with torch.inference_mode():
        for clips, _targets in loader:
            logits = model(clips.to(device))
            probabilities.extend(
                torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
            )
    return probabilities


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, batch size, and learning rate must be positive")
    manifest, records = load_training_records(
        manifest_path=args.manifest,
        candidate_bundle_paths=args.candidate_bundle,
        annotation_paths=args.annotation,
        video_paths=args.video,
    )
    games = sorted({record.source_video_sha256 for record in records})
    if len(games) < 3:
        raise ValueError("game-level OOF calibration requires at least three games")
    print(json.dumps({"stage": "decode", "examples": len(records), "games": len(games)}), flush=True)
    windows = decode_dense_windows(records)
    device = _resolve_device(args.device)
    oof_probabilities = [float("nan")] * len(records)
    oof_raw_probabilities = [float("nan")] * len(records)
    fold_metrics = []
    for fold_index, held_game in enumerate(games):
        train_indexes = [i for i, record in enumerate(records) if record.source_video_sha256 != held_game]
        held_indexes = [i for i, record in enumerate(records) if record.source_video_sha256 == held_game]
        print(json.dumps({"stage": "fold", "held_game": held_game, "train": len(train_indexes), "held": len(held_indexes)}), flush=True)
        model = _fit(
            windows=windows,
            records=records,
            train_indexes=train_indexes,
            checkpoint_path=args.backbone_checkpoint,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            trainable_stage=args.trainable_stage,
            seed=args.seed + fold_index * 100,
            checkpoint_format=args.checkpoint_format,
            preprocessing=args.preprocessing,
            temporal_sampling=args.temporal_sampling,
        )
        calibration_probabilities = _predict(
            model,
            windows=windows,
            records=records,
            indexes=train_indexes,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed,
            preprocessing=args.preprocessing,
            temporal_sampling=args.temporal_sampling,
        )
        fold_threshold, calibration_metrics = select_precision_threshold(
            [records[index].event_present for index in train_indexes],
            calibration_probabilities,
        )
        fold_probabilities = _predict(
            model,
            windows=windows,
            records=records,
            indexes=held_indexes,
            device=device,
            batch_size=args.batch_size,
            seed=args.seed,
            preprocessing=args.preprocessing,
            temporal_sampling=args.temporal_sampling,
        )
        for index, probability in zip(held_indexes, fold_probabilities, strict=True):
            oof_raw_probabilities[index] = probability
            oof_probabilities[index] = normalize_probability_by_threshold(
                probability, fold_threshold
            )
        fold_metrics.append(
            {
                "held_game_sha256": held_game,
                "training_calibration_threshold": fold_threshold,
                "training_calibration_metrics": calibration_metrics,
            }
        )
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    oof_threshold = 0.5
    gate = evaluate_game_gate(records, oof_probabilities, threshold=oof_threshold)
    for fold in fold_metrics:
        fold.update(gate["per_game"][fold["held_game_sha256"]])
    print(
        json.dumps(
            {"stage": "oof_gate", "normalized_threshold": oof_threshold, **gate},
            sort_keys=True,
        ),
        flush=True,
    )

    final_model = _fit(
        windows=windows,
        records=records,
        train_indexes=list(range(len(records))),
        checkpoint_path=args.backbone_checkpoint,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        trainable_stage=args.trainable_stage,
        seed=args.seed + 10_000,
        checkpoint_format=args.checkpoint_format,
        preprocessing=args.preprocessing,
        temporal_sampling=args.temporal_sampling,
    )
    final_probabilities = _predict(
        final_model,
        windows=windows,
        records=records,
        indexes=list(range(len(records))),
        device=device,
        batch_size=args.batch_size,
        seed=args.seed,
        preprocessing=args.preprocessing,
        temporal_sampling=args.temporal_sampling,
    )
    final_threshold, final_calibration_metrics = select_precision_threshold(
        [record.event_present for record in records], final_probabilities
    )
    args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": final_model.cpu().state_dict(),
            "schema_version": "agu.shot-validity-r2plus1d-checkpoint.v1",
            "threshold": final_threshold,
        },
        args.checkpoint_output,
    )
    metadata = seal_finetuned_model_metadata(
        {
            "purpose": "agu_runtime_model",
            "runtime_consumable": bool(gate["promoted"]),
            "promoted": bool(gate["promoted"]),
            "promotion_requirements": {
                "minimum_pooled_precision": 0.95,
                "minimum_per_game_precision": 0.95,
                "minimum_per_game_recall": 0.85,
            },
            "training_manifest_sha256": manifest["manifest_sha256"],
            "backbone_sha256": file_sha256(args.backbone_checkpoint),
            "checkpoint_filename": args.checkpoint_output.name,
            "checkpoint_sha256": file_sha256(args.checkpoint_output),
            "threshold": final_threshold,
            "final_training_calibration_metrics": final_calibration_metrics,
            "clip_frames": 16,
            "dense_frames": 32,
            "trainable_stage": args.trainable_stage,
            "checkpoint_format": args.checkpoint_format,
            "preprocessing": args.preprocessing,
            "temporal_sampling": args.temporal_sampling,
            "anchored_examples": sum(record.anchor_frame is not None for record in records),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "examples": len(records),
            "positive_examples": sum(record.event_present for record in records),
            "negative_examples": sum(not record.event_present for record in records),
            "oof_gate": gate,
            "oof_folds": fold_metrics,
            "oof_predictions": [
                {
                    "source_video_sha256": record.source_video_sha256,
                    "event_id": record.event_id,
                    "event_present": record.event_present,
                    "raw_probability": raw_probability,
                    "normalized_probability": probability,
                }
                for record, raw_probability, probability in zip(
                    records, oof_raw_probabilities, oof_probabilities, strict=True
                )
            ],
        }
    )
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"metadata": str(args.metadata_output), "promoted": gate["promoted"]}), flush=True)
    return 0 if gate["promoted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
