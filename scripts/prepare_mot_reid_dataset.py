#!/usr/bin/env python3
"""Create hash-bound ReID crops from a sealed training-only MOT catalog."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.reid_training_data import build_reid_training_crops  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--maximum-crops-per-track", type=int, default=12)
    parser.add_argument("--minimum-crop-width", type=int, default=24)
    parser.add_argument("--minimum-crop-height", type=int, default=48)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--identity-scope",
        choices=("sequence_track", "dataset_track"),
        default="sequence_track",
        help="Use dataset_track only when upstream track IDs are stable across every selected sequence",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_reid_training_crops(
        catalog_payload=json.loads(args.catalog.read_text(encoding="utf-8")),
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        maximum_crops_per_track=args.maximum_crops_per_track,
        minimum_crop_width=args.minimum_crop_width,
        minimum_crop_height=args.minimum_crop_height,
        jpeg_quality=args.jpeg_quality,
        identity_scope=args.identity_scope,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest_sha256": payload["manifest_sha256"],
                "identity_count": payload["identity_count"],
                "sample_count": payload["sample_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
