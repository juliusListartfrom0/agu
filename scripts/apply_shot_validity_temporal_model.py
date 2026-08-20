#!/usr/bin/env python3
"""Apply a sealed temporal shot-validity model to raw-only candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.official_evaluation import verify_raw_only_bundle  # noqa: E402
from app.analysis.official_inference import seal_agu_autonomous_predictions  # noqa: E402
from app.analysis.schemas import RawOnlyPredictionBundleResponse  # noqa: E402
from app.analysis.shot_validity_temporal import (  # noqa: E402
    TEMPORAL_SHOT_BACKBONE,
    TemporalShotValidityModel,
    apply_temporal_shot_gate,
    extract_event_embeddings,
    file_sha256,
    load_temporal_backbone,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser.parse_args()


def apply_model(
    *,
    candidate_bundle: RawOnlyPredictionBundleResponse,
    video_path: Path,
    model_path: Path,
    backbone_checkpoint: Path,
    batch_size: int,
    device_name: str,
) -> RawOnlyPredictionBundleResponse:
    source = verify_raw_only_bundle(candidate_bundle)
    if len(source.raw_videos) != 1:
        raise ValueError("temporal shot gate requires exactly one raw video")
    asset = source.raw_videos[0]
    if asset.filename != video_path.name or asset.sha256 != file_sha256(video_path):
        raise ValueError("candidate bundle does not match the raw video")
    model = TemporalShotValidityModel(json.loads(model_path.read_text(encoding="utf-8")))
    backbone_hash = file_sha256(backbone_checkpoint)
    if backbone_hash != model.artifact["backbone_sha256"]:
        raise ValueError("temporal backbone checkpoint hash mismatch")
    device = _resolve_device(device_name)
    backbone = load_temporal_backbone(backbone_checkpoint, device=device)
    shots = [event for event in source.events if event.event_type == "field_goal_attempt"]
    embeddings = extract_event_embeddings(
        video_path=video_path,
        events=shots,
        backbone=backbone,
        device=device,
        clip_frames=int(model.artifact["clip_frames"]),
        batch_size=batch_size,
    )
    events = apply_temporal_shot_gate(
        source.events,
        shot_events=shots,
        embeddings=embeddings,
        model=model,
    )
    return seal_agu_autonomous_predictions(
        game_id=source.game_id,
        raw_video_paths=[video_path],
        events=events,
        config={
            "pipeline": "agu_temporal_shot_validity_v1",
            "source_candidate_config_sha256": source.config_sha256,
            "clip_frames": model.artifact["clip_frames"],
            "threshold": model.artifact["threshold"],
        },
        candidate_backend=source.model_provenance.get(
            "candidate_backend", "agu_traditional_perception"
        ),
        semantic_backend="agu_temporal_shot_validity",
        semantic_model=TEMPORAL_SHOT_BACKBONE,
        model_provenance={
            "source_candidate_bundle_sha256": source.bundle_sha256,
            "temporal_shot_model_sha256": model.artifact["model_sha256"],
            "temporal_backbone_sha256": backbone_hash,
        },
    )


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    args = parse_args()
    source = RawOnlyPredictionBundleResponse.model_validate_json(
        args.candidate_bundle.read_text(encoding="utf-8")
    )
    result = apply_model(
        candidate_bundle=source,
        video_path=args.video,
        model_path=args.model,
        backbone_checkpoint=args.backbone_checkpoint,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event_count": len(result.events), "bundle_sha256": result.bundle_sha256}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
