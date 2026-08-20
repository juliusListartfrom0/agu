#!/usr/bin/env python3
"""Extract training-only DINOv2 embeddings from the paired BARD subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.bard_visual_state import (  # noqa: E402
    bard_clip_frame_indexes,
    seal_bard_visual_state_embedding_artifact,
)
from app.analysis.shot_validity_scene_state import file_sha256  # noqa: E402
from scripts.build_pbp_visual_state_manifest import (  # noqa: E402
    load_sealed_blind_hashes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--sealed-blind-acquisition", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
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
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    acquisition = verify_acquisition_artifact(
        json.loads(args.acquisition.read_text(encoding="utf-8"))
    )
    repository = args.repository.resolve()
    blind_hashes = set(
        load_sealed_blind_hashes(args.sealed_blind_acquisition)
    )
    blind_hashes.update(acquisition["sealed_blind_video_sha256s"])
    device = _resolve_device(args.device)
    backbone, transform = _load_dinov2(
        checkpoint_path=args.backbone_checkpoint,
        device=device,
    )
    examples = []
    for position, row in enumerate(acquisition["examples"], start=1):
        source_sha = str(row["sha256"])
        if source_sha in blind_hashes:
            raise ValueError("sealed blind video cannot enter BARD embeddings")
        relative = Path(str(row["path"]))
        path = (repository / relative).resolve()
        if (
            relative.is_absolute()
            or not path.is_relative_to(repository)
            or not path.is_file()
            or file_sha256(path) != source_sha
        ):
            raise ValueError("BARD source media is absent or hash-mismatched")
        frame_count, fps = _video_metadata(path)
        if (
            frame_count != int(row["frame_count"])
            or abs(fps - float(row["fps"])) > 1e-6
        ):
            raise ValueError("BARD source metadata mismatch")
        indexes = bard_clip_frame_indexes(frame_count=frame_count)
        embeddings = _extract_embeddings(
            video_path=path,
            frame_indexes=indexes,
            backbone=backbone,
            transform=transform,
            device=device,
            batch_size=args.batch_size,
        )
        examples.append(
            {
                "source_video_sha256": source_sha,
                "source_path": str(row["path"]),
                "pair_id": str(row["pair_id"]),
                "state": str(row["state"]),
                "frame_count": frame_count,
                "source_fps": fps,
                "frame_indexes": list(indexes),
                "embeddings": embeddings,
            }
        )
        print(
            json.dumps(
                {
                    "stage": "bard_visual_state_embeddings",
                    "example": position,
                    "total": len(acquisition["examples"]),
                }
            ),
            flush=True,
        )
    artifact = seal_bard_visual_state_embedding_artifact(
        {
            "purpose": "offline_broadcast_visual_state_pretraining",
            "acquisition_artifact_sha256": acquisition["artifact_sha256"],
            "backbone": "facebook/dinov2-small",
            "backbone_sha256": _backbone_sha256(args.backbone_checkpoint),
            "sampling_protocol": "clip_quantiles_005_0275_050_0725_095",
            "sealed_blind_video_sha256s": sorted(blind_hashes),
            "examples": examples,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
    temporary.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "examples": len(examples),
                "artifact_sha256": artifact["artifact_sha256"],
            }
        )
    )
    return 0


def verify_acquisition_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    if (
        artifact.get("schema_version")
        != "agu.bard-visual-state-acquisition.v1"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("truth_used_for_training_only") is not True
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("license") != "CC-BY-4.0"
        or not isinstance(artifact.get("examples"), list)
        or not artifact["examples"]
        or not isinstance(artifact.get("sealed_blind_video_sha256s"), list)
    ):
        raise ValueError("invalid BARD acquisition artifact")
    if claimed != _canonical_sha256(artifact):
        raise ValueError("BARD acquisition artifact hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _extract_embeddings(
    *,
    video_path: Path,
    frame_indexes: tuple[int, ...],
    backbone: torch.nn.Module,
    transform: object,
    device: torch.device,
    batch_size: int,
) -> list[list[float]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open BARD clip: {video_path.name}")
    pending: list[torch.Tensor] = []
    output: list[list[float]] = []

    def flush() -> None:
        if not pending:
            return
        with torch.inference_mode():
            values = backbone(torch.stack(pending).to(device))
        output.extend(values.detach().cpu().to(torch.float32).tolist())
        pending.clear()

    try:
        for frame_index in frame_indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"cannot decode BARD frame {frame_index}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1)
            pending.append(transform(tensor))  # type: ignore[operator]
            if len(pending) >= batch_size:
                flush()
    finally:
        capture.release()
    flush()
    if len(output) != len(frame_indexes):
        raise ValueError("DINOv2 returned an unexpected BARD embedding count")
    return output


def _video_metadata(path: Path) -> tuple[int, float]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open BARD clip: {path.name}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if frame_count <= 0 or fps <= 0:
        raise ValueError(f"invalid BARD clip metadata: {path.name}")
    return frame_count, fps


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_dinov2(
    *,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, object]:
    if not checkpoint_path.is_dir():
        raise ValueError("DINOv2 checkpoint directory is missing")
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
    target = path / "model.safetensors"
    if not target.is_file():
        raise ValueError("DINOv2 model weights are missing")
    return file_sha256(target)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
