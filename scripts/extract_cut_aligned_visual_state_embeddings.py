#!/usr/bin/env python3
"""Extract training-only MViT clips bounded by label-hidden broadcast cuts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.cut_aligned_video_state import (  # noqa: E402
    build_cut_aligned_clip_plan,
    extract_cut_aligned_video_embeddings,
    seal_cut_aligned_video_embedding_artifact,
)
from app.analysis.pbp_visual_state_review import (  # noqa: E402
    verify_visual_state_review_plan,
)
from app.analysis.replay_transition import (  # noqa: E402
    verify_replay_transition_artifact,
)
from app.analysis.shot_validity_video_backbone import (  # noqa: E402
    VIDEO_BACKBONES,
    file_sha256,
    get_video_backbone_spec,
    load_video_backbone,
)


def run_extraction(
    *,
    review_plan_path: Path,
    transition_path: Path,
    video_path: Path,
    backbone_name: str,
    backbone_checkpoint: Path,
    output_path: Path,
    clip_frames: int,
    batch_size: int,
    device_name: str,
) -> dict[str, Any]:
    review_plan = verify_visual_state_review_plan(_read_json(review_plan_path))
    transition = verify_replay_transition_artifact(_read_json(transition_path))
    source = str(transition["raw_video_sha256"])
    if source in set(review_plan["sealed_blind_video_sha256s"]):
        raise ValueError("cut-aligned extraction rejects sealed blind video")
    if file_sha256(video_path) != source:
        raise ValueError("cut-aligned source video hash mismatch")

    source_examples = [
        row
        for row in review_plan["examples"]
        if row["source_video_sha256"] == source
    ]
    if not source_examples or {
        str(row["source_video_filename"]) for row in source_examples
    } != {video_path.name}:
        raise ValueError("cut-aligned video filename is not review-plan bound")
    planned = build_cut_aligned_clip_plan(
        review_examples=source_examples,
        transition_artifact=transition,
        sealed_blind_video_sha256s=review_plan["sealed_blind_video_sha256s"],
        clip_frames=clip_frames,
    )

    device = _resolve_device(device_name)
    spec = get_video_backbone_spec(backbone_name)
    checkpoint_sha256 = file_sha256(backbone_checkpoint)
    backbone, transform = load_video_backbone(
        backbone_name,
        backbone_checkpoint,
        device=device,
    )
    examples = extract_cut_aligned_video_embeddings(
        video_path=video_path,
        planned_examples=planned,
        backbone=backbone,
        transform=transform,
        device=device,
        batch_size=batch_size,
    )
    artifact = seal_cut_aligned_video_embedding_artifact(
        {
            "purpose": "cut_aligned_visual_state_training_only",
            "runtime_consumable": False,
            "codex_runtime_answer_used": False,
            "transition_semantics_used": False,
            "truth_used_for_training_only": False,
            "review_plan_sha256": review_plan["artifact_sha256"],
            "transition_artifact_sha256": transition["artifact_sha256"],
            "raw_video_sha256": source,
            "candidate_bundle_sha256": transition["candidate_bundle_sha256"],
            "backbone": backbone_name,
            "backbone_sha256": checkpoint_sha256,
            "backbone_license": spec.license,
            "backbone_weights_url": spec.weights_url,
            "embedding_dimension": spec.embedding_dimension,
            "clip_frames": clip_frames,
            "expected_event_ids": sorted(row["event_id"] for row in planned),
            "complete": True,
            "examples": examples,
        }
    )
    _write_json(output_path, artifact)
    return artifact


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-plan", type=Path, required=True)
    parser.add_argument("--transition-artifact", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--backbone", choices=sorted(VIDEO_BACKBONES), required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    artifact = run_extraction(
        review_plan_path=args.review_plan,
        transition_path=args.transition_artifact,
        video_path=args.video,
        backbone_name=args.backbone,
        backbone_checkpoint=args.backbone_checkpoint,
        output_path=args.output,
        clip_frames=args.clip_frames,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "events": len(artifact["examples"]),
                "artifact_sha256": artifact["artifact_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
