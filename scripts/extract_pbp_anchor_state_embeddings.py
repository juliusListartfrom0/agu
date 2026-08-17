#!/usr/bin/env python3
"""Extract training-only MobileNet embeddings around PBP-bound anchors."""

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

from app.analysis.broadcast_clock import verify_broadcast_clock_artifact  # noqa: E402
from app.analysis.pbp_visual_state import (  # noqa: E402
    verify_pbp_visual_state_manifest,
)
from app.analysis.pbp_visual_state_frames import (  # noqa: E402
    ANCHOR_OFFSETS_SECONDS,
    BACKBONE_DIMENSIONS,
    DINO_V2_SMALL_BACKBONE,
    POST_ANCHOR_OFFSETS_SECONDS,
    PRE_ANCHOR_OFFSETS_SECONDS,
    WIDE_ANCHOR_OFFSETS_SECONDS,
    anchor_frame_indexes,
    seal_anchor_state_embedding_artifact,
)
from app.analysis.shot_validity_scene_state import (  # noqa: E402
    SCENE_BACKBONE,
    file_sha256,
    load_scene_backbone,
)
from scripts.build_pbp_visual_state_manifest import (  # noqa: E402
    load_sealed_blind_hashes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument(
        "--clock-artifact",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument(
        "--sealed-blind-acquisition",
        type=Path,
        required=True,
    )
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--backbone",
        choices=(SCENE_BACKBONE, DINO_V2_SMALL_BACKBONE),
        default=SCENE_BACKBONE,
    )
    parser.add_argument(
        "--sampling-protocol",
        choices=("anchor3", "preanchor5", "postanchor5", "wide9"),
        default="anchor3",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (
        len(args.manifest) == len(args.clock_artifact) == len(args.video)
        and args.batch_size > 0
    ):
        raise ValueError("manifests, clocks, and videos must be paired")
    blind_hashes = load_sealed_blind_hashes(args.sealed_blind_acquisition)
    offsets_seconds = {
        "anchor3": ANCHOR_OFFSETS_SECONDS,
        "preanchor5": PRE_ANCHOR_OFFSETS_SECONDS,
        "postanchor5": POST_ANCHOR_OFFSETS_SECONDS,
        "wide9": WIDE_ANCHOR_OFFSETS_SECONDS,
    }[args.sampling_protocol]
    device = _resolve_device(args.device)
    backbone, transform = _load_backbone(
        backbone_name=args.backbone,
        checkpoint_path=args.backbone_checkpoint,
        device=device,
    )
    examples: list[dict[str, object]] = []
    manifest_hashes: list[str] = []
    clock_hashes: list[str] = []
    source_hashes: list[str] = []
    for manifest_path, clock_path, video_path in zip(
        args.manifest,
        args.clock_artifact,
        args.video,
        strict=True,
    ):
        manifest = verify_pbp_visual_state_manifest(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        clock = verify_broadcast_clock_artifact(
            json.loads(clock_path.read_text(encoding="utf-8"))
        )
        source_sha = file_sha256(video_path)
        if (
            source_sha != manifest["raw_video_sha256"]
            or source_sha != clock["raw_video_sha256"]
            or manifest["clock_artifact_sha256"] != clock["artifact_sha256"]
            or manifest["candidate_bundle_sha256"]
            != clock["candidate_bundle_sha256"]
            or source_sha in blind_hashes
        ):
            raise ValueError("anchor-state inputs are not bound or include sealed blind data")
        frame_count, fps = _video_metadata(video_path)
        clock_by_id = {str(row["event_id"]): row for row in clock["events"]}
        rows = [
            row
            for row in manifest["examples"]
            if row["state"] in {"field_goal", "free_throw"}
        ]
        frame_indexes = [
            anchor_frame_indexes(
                anchor_frame=int(clock_by_id[str(row["event_id"])]["anchor_frame"]),
                source_fps=fps,
                frame_count=frame_count,
                offsets_seconds=offsets_seconds,
            )
            for row in rows
        ]
        embeddings = _extract_embeddings(
            video_path=video_path,
            frame_indexes=frame_indexes,
            backbone=backbone,
            transform=transform,
            device=device,
            batch_size=args.batch_size,
            offsets_seconds=offsets_seconds,
        )
        for row, indexes, values in zip(
            rows,
            frame_indexes,
            embeddings,
            strict=True,
        ):
            anchor = int(clock_by_id[str(row["event_id"])]["anchor_frame"])
            examples.append(
                {
                    "source_video_sha256": source_sha,
                    "source_video_filename": video_path.name,
                    "candidate_bundle_sha256": manifest["candidate_bundle_sha256"],
                    "event_id": row["event_id"],
                    "state": row["state"],
                    "anchor_frame": anchor,
                    "source_fps": fps,
                    "frame_count": frame_count,
                    "frame_indexes": list(indexes),
                    "embeddings": values,
                }
            )
        manifest_hashes.append(str(manifest["artifact_sha256"]))
        clock_hashes.append(str(clock["artifact_sha256"]))
        source_hashes.append(source_sha)
        print(
            json.dumps(
                {
                    "stage": "pbp_anchor_embeddings",
                    "video": video_path.name,
                    "examples": len(rows),
                    "total_examples": len(examples),
                }
            ),
            flush=True,
        )
    artifact = seal_anchor_state_embedding_artifact(
        {
            "purpose": "offline_pbp_visual_state_training",
            "truth_used_for_training_only": True,
            "codex_runtime_answer_used": False,
            "training_manifest_sha256s": manifest_hashes,
            "clock_artifact_sha256s": clock_hashes,
            "source_video_sha256s": source_hashes,
            "sealed_blind_video_sha256s": list(blind_hashes),
            "backbone": args.backbone,
            "backbone_sha256": _backbone_sha256(args.backbone_checkpoint),
            "anchor_offsets_seconds": list(offsets_seconds),
            "examples": examples,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
    temporary.write_text(
        json.dumps(artifact, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "examples": len(examples),
                "embedding_dimension": BACKBONE_DIMENSIONS[args.backbone],
                "artifact_sha256": artifact["artifact_sha256"],
            }
        )
    )
    return 0


def _extract_embeddings(
    *,
    video_path: Path,
    frame_indexes: list[tuple[int, ...]],
    backbone: torch.nn.Module,
    transform: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    offsets_seconds: tuple[float, ...],
) -> list[list[list[float]]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open source video: {video_path.name}")
    pending: list[torch.Tensor] = []
    flattened: list[list[float]] = []

    def flush() -> None:
        if not pending:
            return
        with torch.inference_mode():
            values = backbone(torch.stack(pending).to(device))
        flattened.extend(values.detach().cpu().to(torch.float32).tolist())
        pending.clear()

    try:
        for indexes in frame_indexes:
            for frame_index in indexes:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ValueError(f"cannot decode source frame {frame_index}")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2, 0, 1)
                pending.append(transform(tensor))
                if len(pending) >= batch_size:
                    flush()
    finally:
        capture.release()
    flush()
    expected = len(frame_indexes) * len(offsets_seconds)
    if len(flattened) != expected:
        raise ValueError("anchor-state backbone returned an unexpected count")
    return [
        flattened[index : index + len(offsets_seconds)]
        for index in range(0, expected, len(offsets_seconds))
    ]


def _video_metadata(path: Path) -> tuple[int, float]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open source video: {path.name}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if frame_count <= 0 or fps <= 0:
        raise ValueError(f"source video metadata is invalid: {path.name}")
    return frame_count, fps


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_backbone(
    *,
    backbone_name: str,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, object]:
    if backbone_name == SCENE_BACKBONE:
        return load_scene_backbone(checkpoint_path, device=device)
    if backbone_name != DINO_V2_SMALL_BACKBONE or not checkpoint_path.is_dir():
        raise ValueError("unsupported anchor-state backbone checkpoint")
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
        checkpoint_path,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        checkpoint_path,
        local_files_only=True,
    ).to(device).eval()

    class DinoPooler(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, pixels: torch.Tensor) -> torch.Tensor:
            return self.inner(pixel_values=pixels).pooler_output

    def transform(image: torch.Tensor) -> torch.Tensor:
        return processor(images=image, return_tensors="pt")["pixel_values"][0]

    return DinoPooler(model).to(device).eval(), transform


def _backbone_sha256(path: Path) -> str:
    target = path / "model.safetensors" if path.is_dir() else path
    if not target.is_file():
        raise ValueError("anchor-state backbone weights are missing")
    return file_sha256(target)


if __name__ == "__main__":
    raise SystemExit(main())
