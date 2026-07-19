#!/usr/bin/env python3
"""Extract benchmark-disjoint R(2+1)D action-owner training embeddings.

This tool is deliberately training-only.  It consumes SHA-bound action-owner
labels, crops only AGU-generated player candidates from the corresponding raw
video, and uses torchvision's Kinetics-400 R(2+1)D backbone as an open
pretrained feature extractor.  The output keeps the Codex-selected positive ID
for supervised training but is marked non-runtime so it cannot become an
inference answer side channel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.action_ownership import (  # noqa: E402
    ACTION_OWNER_FEATURES,
    extract_action_owner_features,
)
from app.analysis.training_annotation import (  # noqa: E402
    verify_training_annotation_manifest,
)

EMBEDDING_SCHEMA = "agu.action-owner-video-embeddings.v1"
PRETRAINED_BACKBONE = "torchvision/r2plus1d_18/kinetics400_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--source-video", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--crop-padding", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def extract_embeddings(
    *,
    manifest_path: Path,
    annotation_paths: Sequence[Path],
    source_video_paths: Sequence[Path],
    device_name: str = "auto",
    clip_frames: int = 16,
    crop_padding: float = 0.25,
    batch_size: int = 8,
) -> dict[str, Any]:
    if clip_frames < 2:
        raise ValueError("clip_frames must be at least 2")
    if crop_padding < 0:
        raise ValueError("crop_padding must be non-negative")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    manifest = verify_training_annotation_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    allowed_annotations = {
        (str(item["filename"]), str(item["sha256"]))
        for item in manifest.get("annotation_files", [])
    }
    allowed_sources = {
        str(item["sha256"]) for item in manifest.get("source_videos", [])
    }
    videos_by_hash: dict[str, Path] = {}
    for path in source_video_paths:
        digest = _file_sha256(path)
        if digest not in allowed_sources:
            raise ValueError(f"source video is not SHA-bound by training manifest: {path.name}")
        if digest in videos_by_hash:
            raise ValueError(f"duplicate source video hash: {path.name}")
        videos_by_hash[digest] = path

    weights = R2Plus1D_18_Weights.KINETICS400_V1
    model = r2plus1d_18(weights=weights, progress=True)
    model.fc = nn.Identity()
    device = _resolve_device(device_name)
    model = model.to(device).eval()
    transform = weights.transforms()

    output_examples: list[dict[str, Any]] = []
    annotation_hashes: list[str] = []
    for annotation_path in annotation_paths:
        annotation_hash = _file_sha256(annotation_path)
        if (annotation_path.name, annotation_hash) not in allowed_annotations:
            raise ValueError(
                f"annotation is not SHA-bound by training manifest: {annotation_path.name}"
            )
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "agu.action-ownership-labels.v1":
            raise ValueError(f"unsupported action-owner labels: {annotation_path.name}")
        source_hash = str(payload.get("source_video_sha256") or "")
        video_path = videos_by_hash.get(source_hash)
        if video_path is None:
            raise ValueError(
                f"missing source video for annotation: {annotation_path.name}"
            )
        annotation_hashes.append(annotation_hash)
        output_examples.extend(
            _extract_annotation_examples(
                payload=payload,
                annotation_name=annotation_path.name,
                annotation_sha256=annotation_hash,
                video_path=video_path,
                model=model,
                transform=transform,
                device=device,
                clip_frames=clip_frames,
                crop_padding=crop_padding,
                batch_size=batch_size,
            )
        )

    artifact: dict[str, Any] = {
        "schema_version": EMBEDDING_SCHEMA,
        "producer": "agu",
        "purpose": "model_training_only",
        "runtime_consumable": False,
        "training_manifest_sha256": manifest["manifest_sha256"],
        "training_annotation_sha256": sorted(annotation_hashes),
        "source_video_sha256": sorted(videos_by_hash),
        "backbone": PRETRAINED_BACKBONE,
        "embedding_dimension": 512,
        "clip_frames": clip_frames,
        "crop_padding": crop_padding,
        "traditional_feature_names": list(ACTION_OWNER_FEATURES),
        "examples": output_examples,
    }
    artifact["artifact_sha256"] = _json_sha256(artifact)
    return artifact


def verify_embedding_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    claimed_hash = str(artifact.pop("artifact_sha256", ""))
    if artifact.get("schema_version") != EMBEDDING_SCHEMA:
        raise ValueError("unsupported action-owner embedding schema")
    if artifact.get("runtime_consumable") is not False:
        raise ValueError("action-owner embeddings must remain non-runtime")
    if artifact.get("backbone") != PRETRAINED_BACKBONE:
        raise ValueError("action-owner embedding backbone mismatch")
    if artifact.get("traditional_feature_names") != list(ACTION_OWNER_FEATURES):
        raise ValueError("action-owner traditional feature contract mismatch")
    if _json_sha256(artifact) != claimed_hash:
        raise ValueError("action-owner embedding artifact hash mismatch")
    artifact["artifact_sha256"] = claimed_hash
    return artifact


def _extract_annotation_examples(
    *,
    payload: Mapping[str, Any],
    annotation_name: str,
    annotation_sha256: str,
    video_path: Path,
    model: nn.Module,
    transform: Any,
    device: torch.device,
    clip_frames: int,
    crop_padding: float,
    batch_size: int,
) -> list[dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open source video: {video_path.name}")
    try:
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        records = []
        for example_index, example in enumerate(payload.get("examples", [])):
            anchor_frame = int(example["anchor_frame"])
            positive = str(example.get("positive_player_id") or "")
            grouped: dict[str, list[Mapping[str, Any]]] = {}
            for observation in example.get("candidate_player_observations", []):
                player_id = str(observation.get("player_id") or "")
                if player_id:
                    grouped.setdefault(player_id, []).append(observation)
            if positive not in grouped or len(grouped) < 2:
                raise ValueError("each embedding example needs a positive and a negative")

            prepared: list[dict[str, Any]] = []
            required_frames: set[int] = set()
            for player_id, observations in sorted(grouped.items()):
                ordered = sorted(observations, key=lambda item: int(item.get("frame", 0)))
                sampled_frames = _sample_frame_indexes(
                    [int(item.get("frame", 0)) for item in ordered], clip_frames
                )
                required_frames.update(sampled_frames)
                traditional = extract_action_owner_features(
                    ordered, anchor_frame=anchor_frame
                )
                prepared.append(
                    {
                        "player_id": player_id,
                        "observations": ordered,
                        "sampled_frames": sampled_frames,
                        "traditional_features": [
                            traditional[name] for name in ACTION_OWNER_FEATURES
                        ],
                    }
                )

            frames = _read_frames(capture, sorted(required_frames))
            tensors: list[torch.Tensor] = []
            valid_prepared: list[dict[str, Any]] = []
            for candidate in prepared:
                clip = _candidate_clip(
                    frames=frames,
                    observations=candidate["observations"],
                    sampled_frames=candidate["sampled_frames"],
                    frame_width=frame_width,
                    frame_height=frame_height,
                    crop_padding=crop_padding,
                )
                tensors.append(transform(torch.from_numpy(clip).permute(0, 3, 1, 2)))
                valid_prepared.append(candidate)

            embeddings: list[list[float]] = []
            with torch.inference_mode():
                for start in range(0, len(tensors), batch_size):
                    batch = torch.stack(tensors[start : start + batch_size]).to(device)
                    values = model(batch).detach().cpu().to(torch.float32).numpy()
                    embeddings.extend(values.tolist())
            candidates = []
            for candidate, embedding in zip(valid_prepared, embeddings):
                candidates.append(
                    {
                        "player_id": candidate["player_id"],
                        "sampled_frames": candidate["sampled_frames"],
                        "traditional_features": candidate["traditional_features"],
                        "embedding": embedding,
                    }
                )
            records.append(
                {
                    "source_video_sha256": payload["source_video_sha256"],
                    "annotation_filename": annotation_name,
                    "annotation_sha256": annotation_sha256,
                    "example_index": example_index,
                    "event_id": example.get("event_id"),
                    "anchor_frame": anchor_frame,
                    "positive_player_id": positive,
                    "candidates": candidates,
                }
            )
        return records
    finally:
        capture.release()


def _sample_frame_indexes(frames: Sequence[int], count: int) -> list[int]:
    unique = sorted(set(frames))
    if not unique:
        raise ValueError("candidate has no observation frames")
    if len(unique) == 1:
        return [unique[0]] * count
    positions = np.linspace(0, len(unique) - 1, count)
    return [unique[int(round(position))] for position in positions]


def _read_frames(
    capture: cv2.VideoCapture, frame_indexes: Sequence[int]
) -> dict[int, np.ndarray]:
    frames: dict[int, np.ndarray] = {}
    for frame_index in frame_indexes:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"cannot decode source frame {frame_index}")
        frames[frame_index] = frame
    return frames


def _candidate_clip(
    *,
    frames: Mapping[int, np.ndarray],
    observations: Sequence[Mapping[str, Any]],
    sampled_frames: Sequence[int],
    frame_width: int,
    frame_height: int,
    crop_padding: float,
) -> np.ndarray:
    crops = []
    for frame_index in sampled_frames:
        observation = min(
            observations,
            key=lambda item: abs(int(item.get("frame", 0)) - frame_index),
        )
        bounds = _square_crop_bounds(
            observation.get("bbox") or {},
            frame_width=frame_width,
            frame_height=frame_height,
            padding=crop_padding,
        )
        x1, y1, x2, y2 = bounds
        crop = frames[frame_index][y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError(f"empty candidate crop at frame {frame_index}")
        crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)
        crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return np.stack(crops).astype(np.uint8, copy=False)


def _square_crop_bounds(
    bbox: Mapping[str, Any],
    *,
    frame_width: int,
    frame_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    values = [float(bbox.get(name, math.nan)) for name in ("x1", "y1", "x2", "y2")]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("candidate bbox must contain finite coordinates")
    x1, y1, x2, y2 = values
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    side = max(width, height) * (1.0 + 2.0 * padding)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    left = max(0, min(frame_width - 1, int(math.floor(center_x - side / 2.0))))
    top = max(0, min(frame_height - 1, int(math.floor(center_y - side / 2.0))))
    right = max(left + 1, min(frame_width, int(math.ceil(center_x + side / 2.0))))
    bottom = max(top + 1, min(frame_height, int(math.ceil(center_y + side / 2.0))))
    return left, top, right, bottom


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(name)
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    return device


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    args = parse_args()
    artifact = extract_embeddings(
        manifest_path=args.manifest,
        annotation_paths=args.annotations,
        source_video_paths=args.source_video,
        device_name=args.device,
        clip_frames=args.clip_frames,
        crop_padding=args.crop_padding,
        batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "artifact_sha256": artifact["artifact_sha256"],
                "examples": len(artifact["examples"]),
                "candidates": sum(
                    len(example["candidates"]) for example in artifact["examples"]
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
