#!/usr/bin/env python3
"""Extract 24-position MobileNet embeddings from BARD embedded clips."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import torch

from app.analysis.bard_event_state import verify_bard_embedded_subset
from app.analysis.bard_event_state_transfer import (
    BARD_SAMPLING_PROTOCOL,
    normalized_frame_indexes,
    seal_bard_frame_embeddings,
)
from app.analysis.shot_validity_scene_state import (
    SCENE_BACKBONE,
    SCENE_EMBEDDING_DIMENSION,
    load_scene_backbone,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clip-root", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    args = parser.parse_args()
    if args.positions < 2 or args.batch_size < 1:
        raise ValueError("BARD embedding extraction limits are invalid")

    manifest = verify_bard_embedded_subset(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        root=args.clip_root,
    )
    device = _resolve_device(args.device)
    backbone, transform = load_scene_backbone(
        args.backbone_checkpoint,
        device=device,
    )
    values_by_clip: dict[str, list[list[float] | None]] = {
        row["clip_id"]: [None] * args.positions
        for row in manifest["clips"]
    }
    indexes_by_clip: dict[str, list[int]] = {}
    tensors: list[torch.Tensor] = []
    references: list[tuple[str, int]] = []

    def flush() -> None:
        if not tensors:
            return
        with torch.inference_mode():
            values = backbone(torch.stack(tensors).to(device))
        embeddings = values.detach().cpu().to(torch.float32).tolist()
        for (clip_id, position), embedding in zip(
            references,
            embeddings,
            strict=True,
        ):
            values_by_clip[clip_id][position] = embedding
        tensors.clear()
        references.clear()

    for clip_index, row in enumerate(manifest["clips"], start=1):
        clip_id = str(row["clip_id"])
        path = args.clip_root / str(row["relative_path"])
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f"cannot open BARD clip: {path.name}")
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            indexes = normalized_frame_indexes(
                frame_count,
                positions=args.positions,
            )
            indexes_by_clip[clip_id] = indexes
            for position, frame_index in enumerate(indexes):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ValueError(
                        f"cannot decode BARD frame {frame_index}: {path.name}"
                    )
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2, 0, 1)
                tensors.append(transform(tensor))
                references.append((clip_id, position))
                if len(tensors) >= args.batch_size:
                    flush()
        finally:
            capture.release()
        print(
            json.dumps(
                {
                    "stage": "bard_frame_embeddings",
                    "clip": clip_index,
                    "clip_count": len(manifest["clips"]),
                    "clip_id": clip_id,
                }
            ),
            flush=True,
        )
    flush()

    examples = []
    for row in manifest["clips"]:
        clip_id = str(row["clip_id"])
        embeddings = values_by_clip[clip_id]
        if any(value is None for value in embeddings):
            raise ValueError("BARD frame embedding extraction is incomplete")
        examples.append(
            {
                "clip_id": clip_id,
                "game_id": row["game_id"],
                "event_state": row["event_state"],
                "shot_outcome": row["shot_outcome"],
                "frame_indexes": indexes_by_clip[clip_id],
                "embeddings": embeddings,
            }
        )
    artifact = seal_bard_frame_embeddings(
        {
            "purpose": "event_state_pretraining_only",
            "codex_runtime_answer_used": False,
            "subset_sha256": manifest["subset_sha256"],
            "backbone": SCENE_BACKBONE,
            "backbone_sha256": _file_sha256(args.backbone_checkpoint),
            "embedding_dimension": SCENE_EMBEDDING_DIMENSION,
            "positions": args.positions,
            "sampling_protocol": BARD_SAMPLING_PROTOCOL,
            "examples": examples,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "examples": len(examples),
                "artifact_sha256": artifact["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(value)
    if value == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    return device


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
