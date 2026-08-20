#!/usr/bin/env python3
"""Extract sealed three-phase MobileNet scene embeddings for training only."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import torch
from torchvision.models import MobileNet_V3_Small_Weights

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.official_evaluation import verify_raw_only_bundle  # noqa: E402
from app.analysis.schemas import RawOnlyPredictionBundleResponse  # noqa: E402
from app.analysis.shot_validity import SHOT_VALIDITY_LABEL_SCHEMA  # noqa: E402
from app.analysis.shot_validity_sampling_window import (  # noqa: E402
    SHOT_SAMPLING_PROTOCOL,
    resolve_shot_sampling_window,
)
from app.analysis.shot_validity_scene_state import (  # noqa: E402
    PHASE_FRACTIONS,
    SCENE_BACKBONE,
    SCENE_EMBEDDING_DIMENSION,
    extract_scene_phase_embeddings,
    file_sha256,
    load_scene_backbone,
    seal_scene_embedding_artifact,
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser.parse_args()


def extract_labeled_scene_embeddings(
    *,
    manifest_path: Path,
    candidate_bundle_paths: list[Path],
    annotation_paths: list[Path],
    video_paths: list[Path],
    backbone_checkpoint: Path,
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
    backbone, transform = load_scene_backbone(
        backbone_checkpoint, device=device
    )
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
            raise ValueError("scene embedding extraction requires one video per bundle")
        source_hash = file_sha256(video_path)
        if source_hash not in allowed_sources or bundle.raw_videos[0].sha256 != source_hash:
            raise ValueError("video is not SHA-bound to bundle and training manifest")
        annotation_hash = file_sha256(annotation_path)
        if (annotation_path.name, annotation_hash) not in allowed_annotations:
            raise ValueError("annotation is not SHA-bound by training manifest")
        labels = json.loads(annotation_path.read_text(encoding="utf-8"))
        if labels.get("schema_version") != SHOT_VALIDITY_LABEL_SCHEMA:
            raise ValueError("unsupported shot-validity labels")
        if labels.get("runtime_consumable") is not False:
            raise ValueError("training labels must not be runtime-consumable")
        if labels.get("source_video_sha256") != source_hash:
            raise ValueError("annotation source video hash mismatch")
        if labels.get("candidate_bundle_sha256") != bundle.bundle_sha256:
            raise ValueError("annotation candidate bundle hash mismatch")
        events_by_id = {event.event_id: event for event in bundle.events}
        rows = []
        events = []
        for row in labels.get("examples", []):
            event = events_by_id.get(str(row.get("event_id") or ""))
            if event is None or event.event_type != "field_goal_attempt":
                raise ValueError("scene label does not name a field-goal candidate")
            if not isinstance(row.get("event_present"), bool):
                raise ValueError("scene shot labels must be boolean")
            rows.append(row)
            events.append(event)
        fps = _read_video_fps(video_path)
        sampling_windows = [
            resolve_shot_sampling_window(event, fps=fps) for event in events
        ]
        phase_embeddings = extract_scene_phase_embeddings(
            video_path=video_path,
            events=events,
            backbone=backbone,
            transform=transform,
            device=device,
            batch_size=batch_size,
            sampling_windows=sampling_windows,
        )
        for event, row, window, embeddings in zip(
            events, rows, sampling_windows, phase_embeddings, strict=True
        ):
            examples.append(
                {
                    "source_video_sha256": source_hash,
                    "candidate_bundle_sha256": bundle.bundle_sha256,
                    "event_id": event.event_id,
                    "event_present": bool(row["event_present"]),
                    "sampling_window": asdict(window),
                    "phase_embeddings": embeddings,
                }
            )
        annotation_hashes.append(annotation_hash)
        source_hashes.append(source_hash)
        print(
            json.dumps(
                {
                    "stage": "scene_embedding_batch",
                    "annotation": annotation_path.name,
                    "batch_examples": len(rows),
                    "total_examples": len(examples),
                }
            ),
            flush=True,
        )
    return seal_scene_embedding_artifact(
        {
            "purpose": "scene_state_screening_training_only",
            "producer": "agu",
            "training_manifest_sha256": manifest["manifest_sha256"],
            "training_annotation_sha256": sorted(annotation_hashes),
            "source_video_sha256": sorted(set(source_hashes)),
            "backbone": SCENE_BACKBONE,
            "backbone_sha256": file_sha256(backbone_checkpoint),
            "backbone_license": "BSD-3-Clause (torchvision); upstream weight terms apply",
            "backbone_weights_url": MobileNet_V3_Small_Weights.IMAGENET1K_V1.url,
            "embedding_dimension": SCENE_EMBEDDING_DIMENSION,
            "phase_fractions": list(PHASE_FRACTIONS),
            "sampling_protocol": SHOT_SAMPLING_PROTOCOL,
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


def _read_video_fps(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open source video: {video_path.name}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if fps <= 0:
        raise ValueError(f"source video has invalid fps: {video_path.name}")
    return fps


def main() -> int:
    args = parse_args()
    artifact = extract_labeled_scene_embeddings(
        manifest_path=args.manifest,
        candidate_bundle_paths=args.candidate_bundle,
        annotation_paths=args.annotation,
        video_paths=args.video,
        backbone_checkpoint=args.backbone_checkpoint,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "examples": len(artifact["examples"]),
                "backbone": artifact["backbone"],
                "artifact_sha256": artifact["artifact_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
