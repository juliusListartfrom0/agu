#!/usr/bin/env python3
"""Extract sealed training-only R(2+1)D embeddings from labeled raw windows."""

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
from app.analysis.schemas import RawOnlyPredictionBundleResponse  # noqa: E402
from app.analysis.shot_validity import SHOT_VALIDITY_LABEL_SCHEMA  # noqa: E402
from app.analysis.shot_validity_temporal import (  # noqa: E402
    TEMPORAL_SHOT_BACKBONE,
    TEMPORAL_SHOT_EMBEDDING_DIMENSION,
    extract_event_embeddings,
    file_sha256,
    load_temporal_backbone,
    seal_temporal_embedding_artifact,
)
from app.analysis.training_annotation import verify_training_annotation_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-bundle", type=Path, action="append", required=True)
    parser.add_argument("--annotation", type=Path, action="append", required=True)
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser.parse_args()


def extract_labeled_embeddings(
    *,
    manifest_path: Path,
    candidate_bundle_paths: list[Path],
    annotation_paths: list[Path],
    video_paths: list[Path],
    backbone_checkpoint: Path,
    clip_frames: int,
    batch_size: int,
    device_name: str,
) -> dict[str, object]:
    if not (len(candidate_bundle_paths) == len(annotation_paths) == len(video_paths)):
        raise ValueError("candidate bundles, annotations, and videos must be paired")
    manifest = verify_training_annotation_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    if "shot_validity" not in manifest.get("task_types", []):
        raise ValueError("training manifest does not authorize shot_validity")
    allowed_sources = {str(item["sha256"]) for item in manifest["source_videos"]}
    allowed_annotations = {
        (str(item["filename"]), str(item["sha256"]))
        for item in manifest["annotation_files"]
    }
    device = _resolve_device(device_name)
    backbone = load_temporal_backbone(backbone_checkpoint, device=device)
    examples: list[dict[str, object]] = []
    annotation_hashes = []
    source_hashes = []
    for bundle_path, annotation_path, video_path in zip(
        candidate_bundle_paths, annotation_paths, video_paths, strict=True
    ):
        bundle = verify_raw_only_bundle(
            RawOnlyPredictionBundleResponse.model_validate_json(
                bundle_path.read_text(encoding="utf-8")
            )
        )
        if len(bundle.raw_videos) != 1:
            raise ValueError("temporal embedding extraction requires one video per bundle")
        source_hash = file_sha256(video_path)
        if source_hash not in allowed_sources or bundle.raw_videos[0].sha256 != source_hash:
            raise ValueError("video is not SHA-bound to bundle and training manifest")
        annotation_hash = file_sha256(annotation_path)
        if (annotation_path.name, annotation_hash) not in allowed_annotations:
            raise ValueError("annotation is not SHA-bound by training manifest")
        labels = json.loads(annotation_path.read_text(encoding="utf-8"))
        if labels.get("schema_version") != SHOT_VALIDITY_LABEL_SCHEMA:
            raise ValueError("unsupported shot-validity labels")
        if labels.get("candidate_bundle_sha256") != bundle.bundle_sha256:
            raise ValueError("annotation candidate bundle hash mismatch")
        events_by_id = {event.event_id: event for event in bundle.events}
        labeled_events = []
        label_rows = []
        for row in labels.get("examples", []):
            event = events_by_id.get(str(row.get("event_id") or ""))
            if event is None or event.event_type != "field_goal_attempt":
                raise ValueError("temporal label does not name a field-goal candidate")
            if not isinstance(row.get("event_present"), bool):
                raise ValueError("temporal shot labels must be boolean")
            labeled_events.append(event)
            label_rows.append(row)
        embeddings = extract_event_embeddings(
            video_path=video_path,
            events=labeled_events,
            backbone=backbone,
            device=device,
            clip_frames=clip_frames,
            batch_size=batch_size,
        )
        for event, row, embedding in zip(labeled_events, label_rows, embeddings, strict=True):
            examples.append(
                {
                    "source_video_sha256": source_hash,
                    "candidate_bundle_sha256": bundle.bundle_sha256,
                    "event_id": event.event_id,
                    "event_present": bool(row["event_present"]),
                    "embedding": embedding,
                }
            )
        annotation_hashes.append(annotation_hash)
        source_hashes.append(source_hash)
    return seal_temporal_embedding_artifact(
        {
            "purpose": "model_training_only",
            "producer": "agu",
            "training_manifest_sha256": manifest["manifest_sha256"],
            "training_annotation_sha256": sorted(annotation_hashes),
            "source_video_sha256": sorted(source_hashes),
            "backbone": TEMPORAL_SHOT_BACKBONE,
            "backbone_sha256": file_sha256(backbone_checkpoint),
            "embedding_dimension": TEMPORAL_SHOT_EMBEDDING_DIMENSION,
            "clip_frames": clip_frames,
            "examples": examples,
        }
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
    artifact = extract_labeled_embeddings(
        manifest_path=args.manifest,
        candidate_bundle_paths=args.candidate_bundle,
        annotation_paths=args.annotation,
        video_paths=args.video,
        backbone_checkpoint=args.backbone_checkpoint,
        clip_frames=args.clip_frames,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"examples": len(artifact["examples"]), "artifact_sha256": artifact["artifact_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
