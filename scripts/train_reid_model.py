#!/usr/bin/env python3
"""Train AGU's benchmark-disjoint basketball player ReID checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.reid_training import train_mobilenet_reid  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--crop-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validation-sequence", action="append", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--metric-loss-weight", type=float, default=1.0)
    parser.add_argument("--triplet-margin", type=float, default=0.20)
    parser.add_argument("--freeze-feature-blocks", type=int, default=6)
    parser.add_argument("--minimum-validation-top1", type=float, default=0.85)
    parser.add_argument("--device", default="mps_if_available")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    result = train_mobilenet_reid(
        manifest_path=args.manifest,
        crop_root=args.crop_root,
        output_path=args.output,
        validation_sequences=args.validation_sequence,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        metric_loss_weight=args.metric_loss_weight,
        triplet_margin=args.triplet_margin,
        freeze_feature_blocks=args.freeze_feature_blocks,
        minimum_validation_top1=args.minimum_validation_top1,
        device_name=args.device,
        seed=args.seed,
    )
    print(json.dumps({
        "checkpoint_path": str(result.checkpoint_path),
        "checkpoint_sha256": result.checkpoint_sha256,
        "baseline_top1": result.baseline_top1,
        "trained_top1": result.trained_top1,
        "train_samples": result.train_samples,
        "validation_samples": result.validation_samples,
    }, indent=2))


if __name__ == "__main__":
    main()
