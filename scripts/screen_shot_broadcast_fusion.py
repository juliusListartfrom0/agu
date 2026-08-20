#!/usr/bin/env python3
"""Screen label-free broadcast-state features under game-held isolation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.shot_broadcast_fusion import (  # noqa: E402
    screen_shot_broadcast_fusion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-embeddings", type=Path, required=True)
    parser.add_argument("--video-embeddings", type=Path, action="append", default=[])
    parser.add_argument("--broadcast-state", type=Path, required=True)
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--regularization-c", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = screen_shot_broadcast_fusion(
        scene_artifact=_read_json(args.scene_embeddings),
        video_artifacts=[_read_json(path) for path in args.video_embeddings],
        broadcast_state_artifact=_read_json(args.broadcast_state),
        pca_components=args.pca_components,
        regularization_c=args.regularization_c,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "best_variant": result["best_variant"]["name"],
                "best_gate": result["best_variant"]["gate"],
                "artifact_sha256": result["artifact_sha256"],
            }
        )
    )
    return 0


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
