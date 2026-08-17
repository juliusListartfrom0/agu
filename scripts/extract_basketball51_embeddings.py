#!/usr/bin/env python3
"""Extract resumable training-only embeddings from a Basketball-51 subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.basketball51 import (  # noqa: E402
    seal_basketball51_embedding_artifact,
    verify_basketball51_subset_manifest,
)
from app.analysis.schemas import GameEventResponse  # noqa: E402
from app.analysis.shot_validity_video_backbone import (  # noqa: E402
    VIDEO_BACKBONES,
    extract_video_embeddings,
    file_sha256,
    get_video_backbone_spec,
    load_video_backbone,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--backbone", choices=sorted(VIDEO_BACKBONES), required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser.parse_args()


def extract_subset_embeddings(
    *,
    subset_manifest_path: Path,
    backbone_name: str,
    backbone_checkpoint: Path,
    output_path: Path,
    clip_frames: int,
    device_name: str,
) -> dict[str, object]:
    manifest = verify_basketball51_subset_manifest(
        json.loads(subset_manifest_path.read_text(encoding="utf-8"))
    )
    if clip_frames < 16:
        raise ValueError("Basketball-51 embedding clips require at least 16 frames")
    subset_root = subset_manifest_path.parent
    for row in manifest["clips"]:
        clip_path = subset_root / row["relative_path"]
        if (
            not clip_path.is_file()
            or clip_path.stat().st_size != int(row["size_bytes"])
            or file_sha256(clip_path) != row["sha256"]
        ):
            raise ValueError(f"Basketball-51 clip hash mismatch: {row['relative_path']}")

    device = _resolve_device(device_name)
    checkpoint_sha256 = file_sha256(backbone_checkpoint)
    spec = get_video_backbone_spec(backbone_name)
    partial_path = output_path.with_suffix(f"{output_path.suffix}.partial")
    cached = _load_partial(
        partial_path,
        subset_manifest_sha256=manifest["artifact_sha256"],
        backbone=backbone_name,
        backbone_sha256=checkpoint_sha256,
        clip_frames=clip_frames,
    )
    examples_by_path = {
        str(row["relative_path"]): row for row in cached.get("examples", [])
    }
    backbone, transform = load_video_backbone(
        backbone_name,
        backbone_checkpoint,
        device=device,
    )
    for index, row in enumerate(manifest["clips"], start=1):
        relative_path = str(row["relative_path"])
        if relative_path in examples_by_path:
            continue
        clip_path = subset_root / relative_path
        frame_count = _frame_count(clip_path)
        event = GameEventResponse(
            event_id=Path(relative_path).stem,
            revision=1,
            event_type="field_goal_attempt",
            source_video_id=str(row["source_group"]),
            start_frame=0,
            end_frame=frame_count - 1,
            outcome="unknown",
            status="needs_review",
        )
        embedding = extract_video_embeddings(
            video_path=clip_path,
            events=[event],
            backbone=backbone,
            transform=transform,
            device=device,
            clip_frames=clip_frames,
            batch_size=1,
        )[0]
        examples_by_path[relative_path] = {
            "relative_path": relative_path,
            "source_group": row["source_group"],
            "label": row["label"],
            "shot_type": row["shot_type"],
            "outcome": row["outcome"],
            "embedding": embedding,
        }
        if index % 8 == 0 or index == len(manifest["clips"]):
            _write_partial(
                partial_path,
                {
                    "subset_manifest_sha256": manifest["artifact_sha256"],
                    "backbone": backbone_name,
                    "backbone_sha256": checkpoint_sha256,
                    "clip_frames": clip_frames,
                    "examples": [
                        examples_by_path[path]
                        for path in sorted(examples_by_path)
                    ],
                },
            )
            print(
                json.dumps(
                    {
                        "stage": "basketball51_embedding",
                        "completed": len(examples_by_path),
                        "total": len(manifest["clips"]),
                        "device": str(device),
                    }
                ),
                flush=True,
            )
    artifact = seal_basketball51_embedding_artifact(
        {
            "purpose": "backbone_screening_training_only",
            "producer": "agu",
            "subset_manifest_sha256": manifest["artifact_sha256"],
            "backbone": backbone_name,
            "backbone_sha256": checkpoint_sha256,
            "backbone_license": spec.license,
            "backbone_weights_url": spec.weights_url,
            "embedding_dimension": spec.embedding_dimension,
            "clip_frames": clip_frames,
            "examples": [
                examples_by_path[path] for path in sorted(examples_by_path)
            ],
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    partial_path.unlink(missing_ok=True)
    return artifact


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open Basketball-51 clip: {path.name}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if frame_count < 16:
        raise ValueError(f"Basketball-51 clip is too short: {path.name}")
    return frame_count


def _load_partial(
    path: Path,
    *,
    subset_manifest_sha256: str,
    backbone: str,
    backbone_sha256: str,
    clip_frames: int,
) -> dict[str, object]:
    if not path.is_file():
        return {"examples": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "subset_manifest_sha256": subset_manifest_sha256,
        "backbone": backbone,
        "backbone_sha256": backbone_sha256,
        "clip_frames": clip_frames,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Basketball-51 embedding partial cache provenance mismatch")
    if not isinstance(payload.get("examples"), list):
        raise ValueError("Basketball-51 embedding partial cache is malformed")
    return payload


def _write_partial(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    artifact = extract_subset_embeddings(
        subset_manifest_path=args.subset_manifest,
        backbone_name=args.backbone,
        backbone_checkpoint=args.backbone_checkpoint,
        output_path=args.output,
        clip_frames=args.clip_frames,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "artifact_sha256": artifact["artifact_sha256"],
                "examples": len(artifact["examples"]),
                "backbone": artifact["backbone"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
