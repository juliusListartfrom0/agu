#!/usr/bin/env python3
"""Run the canonical AGU v3 player-action model on a frozen shot plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from app.analysis.independent_base_vlm_fusion import run_agu_v3_base_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, action="append", required=True)
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--maximum-player-clips", type=int, default=3)
    parser.add_argument("--bbox-expansion", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    artifact = run_agu_v3_base_plan(
        plan=json.loads(args.plan.read_text(encoding="utf-8")),
        bundle_paths=args.bundle,
        video_paths=args.video,
        checkpoint_path=args.checkpoint,
        device=device,
        clip_frames=args.clip_frames,
        maximum_player_clips=args.maximum_player_clips,
        bbox_expansion=args.bbox_expansion,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    states: dict[str, int] = {}
    for row in artifact["predictions"]:
        state = str(row["field_goal_state"])
        states[state] = states.get(state, 0) + 1
    print(
        json.dumps(
            {
                "output": str(args.output),
                "artifact_sha256": artifact["artifact_sha256"],
                "states": states,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
