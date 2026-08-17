#!/usr/bin/env python3
"""Extract fixed local-view DINOv2 embeddings from reviewed MUVS frames."""

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

from app.analysis.muvs_event_state import (  # noqa: E402
    verify_muvs_event_state_frames,
    verify_muvs_event_state_plan,
    verify_muvs_event_state_review,
)
from app.analysis.muvs_formation_geometry import (  # noqa: E402
    verify_muvs_formation_detections,
)
from app.analysis.muvs_local_visual import (  # noqa: E402
    MUVS_LOCAL_VISUAL_BACKBONE,
    MUVS_LOCAL_VISUAL_DIMENSION,
    MUVS_LOCAL_VISUAL_REVISION,
    MUVS_LOCAL_VISUAL_VIEWS,
    choose_muvs_local_view,
    seal_muvs_local_visual_embeddings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        default="cpu",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rgb_frame(
    *,
    frame_root: Path,
    metadata: Mapping[str, Any],
    sample_id: str,
) -> torch.Tensor:
    relative = Path(str(metadata["path"]))
    path = (frame_root / relative).resolve()
    if (
        relative.is_absolute()
        or not path.is_relative_to(frame_root)
        or not path.is_file()
        or _file_sha256(path) != metadata["sha256"]
    ):
        raise ValueError(
            f"MUVS source frame is absent or hash-mismatched: {sample_id}"
        )
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if (
        bgr is None
        or bgr.shape[1] != int(metadata["width"])
        or bgr.shape[0] != int(metadata["height"])
    ):
        raise ValueError(f"MUVS source frame shape mismatch: {sample_id}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1)


def main() -> int:
    args = parse_args()
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
    detections = verify_muvs_formation_detections(
        _read_json(args.detections)
    )
    expected_hashes = (
        plan["artifact_sha256"],
        frames["artifact_sha256"],
        review["artifact_sha256"],
    )
    actual_hashes = (
        detections["plan_sha256"],
        detections["frames_sha256"],
        detections["review_sha256"],
    )
    if actual_hashes != expected_hashes:
        raise ValueError("MUVS detection source hashes do not match frame inputs")

    checkpoint = args.backbone_checkpoint.resolve()
    weights = checkpoint / "model.safetensors"
    if not weights.is_file():
        raise ValueError("DINOv2 model weights are missing")
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
        checkpoint,
        local_files_only=True,
    )
    device = torch.device(args.device)
    model = AutoModel.from_pretrained(
        checkpoint,
        local_files_only=True,
    ).to(device).eval()

    frame_root = args.frames.resolve().parent
    frame_rows = {str(row["sample_id"]): row for row in frames["files"]}
    detection_rows = {
        str(row["sample_id"]): row for row in detections["examples"]
    }
    if set(frame_rows) != set(detection_rows):
        raise ValueError("MUVS detection coverage does not match frame coverage")

    examples = []
    for sample in plan["samples"]:
        sample_id = str(sample["sample_id"])
        frame_row = frame_rows[sample_id]
        detection_row = detection_rows[sample_id]
        selected_indices = []
        frame_embeddings = []
        for frame_index, frame_key in enumerate(
            ("frame_before", "frame_after")
        ):
            metadata = frame_row[frame_key]
            image = _load_rgb_frame(
                frame_root=frame_root,
                metadata=metadata,
                sample_id=sample_id,
            )
            width = int(image.shape[2])
            crops = []
            for view in MUVS_LOCAL_VISUAL_VIEWS:
                start, end = view["x_range"]
                left = round(width * float(start))
                right = round(width * float(end))
                crops.append(image[:, :, left:right])
            batch = processor(
                images=crops,
                return_tensors="pt",
            )["pixel_values"].to(device)
            with torch.inference_mode():
                values = model(pixel_values=batch).pooler_output
            values = values.detach().cpu().to(torch.float32)
            if values.shape != (3, MUVS_LOCAL_VISUAL_DIMENSION):
                raise ValueError("DINOv2 returned an unexpected local-view shape")
            frame_embeddings.append(values.tolist())
            selected_indices.append(
                choose_muvs_local_view(
                    detection_row["frames"][frame_index]
                )
            )
        examples.append(
            {
                "sample_id": sample_id,
                "event_id": str(sample["event_id"]),
                "split": str(sample["split"]),
                "state": str(detection_row["state"]),
                "frame_before_sha256": str(
                    frame_row["frame_before"]["sha256"]
                ),
                "frame_after_sha256": str(
                    frame_row["frame_after"]["sha256"]
                ),
                "selected_view_indices": selected_indices,
                "embeddings": frame_embeddings,
            }
        )

    artifact = seal_muvs_local_visual_embeddings(
        {
            "detection_artifact_sha256": detections["artifact_sha256"],
            "backbone": MUVS_LOCAL_VISUAL_BACKBONE,
            "backbone_revision": MUVS_LOCAL_VISUAL_REVISION,
            "backbone_sha256": _file_sha256(weights),
            "embedding_dimension": MUVS_LOCAL_VISUAL_DIMENSION,
            "views": MUVS_LOCAL_VISUAL_VIEWS,
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
    selected_counts = {
        view["name"]: sum(
            index == view_index
            for example in examples
            for index in example["selected_view_indices"]
        )
        for view_index, view in enumerate(MUVS_LOCAL_VISUAL_VIEWS)
    }
    print(
        json.dumps(
            {
                "output": str(args.output),
                "examples": len(examples),
                "device": str(device),
                "selected_counts": selected_counts,
                "artifact_sha256": artifact["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
