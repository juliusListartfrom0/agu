#!/usr/bin/env python3
"""Extract hash-bound MobileNet embeddings from reviewed MUVS source frames."""

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

from app.analysis.muvs_event_state import (  # noqa: E402
    verify_muvs_event_state_frames,
    verify_muvs_event_state_plan,
    verify_muvs_event_state_review,
)
from app.analysis.muvs_event_state_transfer import (  # noqa: E402
    seal_muvs_event_state_embeddings,
)
from app.analysis.shot_validity_scene_state import (  # noqa: E402
    SCENE_BACKBONE,
    SCENE_EMBEDDING_DIMENSION,
    file_sha256,
    load_scene_backbone,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        default="cpu",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    plan = verify_muvs_event_state_plan(_read_json(args.plan))
    frames = verify_muvs_event_state_frames(
        _read_json(args.frames),
        plan=plan,
    )
    review = verify_muvs_event_state_review(
        _read_json(args.review),
        plan=plan,
        frames=frames,
    )
    device = torch.device(args.device)
    backbone, transform = load_scene_backbone(
        args.backbone_checkpoint,
        device=device,
    )

    frame_root = args.frames.resolve().parent
    frame_rows = {str(row["sample_id"]): row for row in frames["files"]}
    state_rows = {str(row["sample_id"]): row for row in review["decisions"]}
    pending: list[torch.Tensor] = []
    flattened: list[list[float]] = []

    def flush() -> None:
        if not pending:
            return
        with torch.inference_mode():
            values = backbone(torch.stack(pending).to(device))
        flattened.extend(values.detach().cpu().to(torch.float32).tolist())
        pending.clear()

    for sample in plan["samples"]:
        sample_id = str(sample["sample_id"])
        frame_row = frame_rows[sample_id]
        for key in ("frame_before", "frame_after"):
            metadata = frame_row[key]
            relative = Path(str(metadata["path"]))
            path = (frame_root / relative).resolve()
            if (
                relative.is_absolute()
                or not path.is_relative_to(frame_root)
                or not path.is_file()
                or file_sha256(path) != metadata["sha256"]
            ):
                raise ValueError(f"MUVS source frame is absent or hash-mismatched: {sample_id}")
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape[1] != int(metadata["width"]) or bgr.shape[0] != int(metadata["height"]):
                raise ValueError(f"MUVS source frame shape mismatch: {sample_id}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1)
            pending.append(transform(tensor))
            if len(pending) >= args.batch_size:
                flush()
    flush()
    if len(flattened) != 2 * len(plan["samples"]):
        raise ValueError("MobileNet returned an unexpected MUVS embedding count")

    examples = []
    for index, sample in enumerate(plan["samples"]):
        sample_id = str(sample["sample_id"])
        frame_row = frame_rows[sample_id]
        examples.append(
            {
                "sample_id": sample_id,
                "event_id": str(sample["event_id"]),
                "split": str(sample["split"]),
                "state": str(state_rows[sample_id]["state"]),
                "frame_before_sha256": str(frame_row["frame_before"]["sha256"]),
                "frame_after_sha256": str(frame_row["frame_after"]["sha256"]),
                "embeddings": flattened[2 * index : 2 * index + 2],
            }
        )
    artifact = seal_muvs_event_state_embeddings(
        {
            "plan_sha256": plan["artifact_sha256"],
            "frames_sha256": frames["artifact_sha256"],
            "review_sha256": review["artifact_sha256"],
            "backbone": SCENE_BACKBONE,
            "backbone_sha256": file_sha256(args.backbone_checkpoint),
            "embedding_dimension": SCENE_EMBEDDING_DIMENSION,
            "examples": examples,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "examples": len(examples),
                "device": str(device),
                "artifact_sha256": artifact["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
