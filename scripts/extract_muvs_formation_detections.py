#!/usr/bin/env python3
"""Extract hash-bound BODD player/rim detections from reviewed MUVS frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.muvs_event_state import (  # noqa: E402
    verify_muvs_event_state_frames,
    verify_muvs_event_state_plan,
    verify_muvs_event_state_review,
)
from app.analysis.muvs_formation_geometry import (  # noqa: E402
    seal_muvs_formation_detections,
)
from app.analysis.perception import UltralyticsDetectorAdapter  # noqa: E402

_EXPECTED_RAW_NAMES = {
    0: "basketball",
    1: "hoop",
    2: "player",
    3: "referee",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Must remain 1 so duplicate frames have batch-composition-independent "
            "detections."
        ),
    )
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.1)
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


def _load_frame(
    *,
    frame_root: Path,
    metadata: dict[str, Any],
    sample_id: str,
) -> Any:
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
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _normalized_detection(
    detection: Any,
    *,
    width: int,
    height: int,
) -> dict[str, Any] | None:
    x1 = max(0.0, min(1.0, float(detection.bbox.x1) / width))
    y1 = max(0.0, min(1.0, float(detection.bbox.y1) / height))
    x2 = max(0.0, min(1.0, float(detection.bbox.x2) / width))
    y2 = max(0.0, min(1.0, float(detection.bbox.y2) / height))
    if x1 >= x2 or y1 >= y2:
        return None
    return {
        "object_type": str(detection.object_type),
        "confidence": float(detection.confidence),
        "bbox": [x1, y1, x2, y2],
    }


def main() -> int:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError(
            "MUVS formation extraction requires batch size 1 for reproducibility"
        )
    if args.image_size < 32:
        raise ValueError("image size must be at least 32")
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("confidence must be in [0,1]")

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
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"BODD model is absent: {model_path}")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    raw_names = {
        int(index): str(name).strip().lower()
        for index, name in dict(model.names).items()
    }
    if raw_names != _EXPECTED_RAW_NAMES:
        raise ValueError(f"unexpected BODD class map: {raw_names}")
    adapter = UltralyticsDetectorAdapter(
        model_path=str(model_path),
        model=model,
        device=args.device,
        image_size=args.image_size,
        confidence=args.confidence,
    )

    frame_root = args.frames.resolve().parent
    frame_rows = {str(row["sample_id"]): row for row in frames["files"]}
    state_rows = {str(row["sample_id"]): row for row in review["decisions"]}
    examples = []
    for batch_start in range(0, len(plan["samples"]), args.batch_size):
        batch_samples = plan["samples"][
            batch_start : batch_start + args.batch_size
        ]
        images = []
        image_metadata = []
        for sample in batch_samples:
            sample_id = str(sample["sample_id"])
            frame_row = frame_rows[sample_id]
            for frame_key in ("frame_before", "frame_after"):
                metadata = frame_row[frame_key]
                images.append(
                    _load_frame(
                        frame_root=frame_root,
                        metadata=metadata,
                        sample_id=sample_id,
                    )
                )
                image_metadata.append(metadata)
        frame_numbers = list(range(len(images)))
        grouped: dict[int, list[dict[str, Any]]] = {
            index: [] for index in frame_numbers
        }
        for detection in adapter.detect(images, frame_numbers):
            metadata = image_metadata[int(detection.frame)]
            normalized = _normalized_detection(
                detection,
                width=int(metadata["width"]),
                height=int(metadata["height"]),
            )
            if normalized is not None:
                grouped[int(detection.frame)].append(normalized)
        for detections in grouped.values():
            detections.sort(
                key=lambda row: (
                    row["object_type"],
                    -float(row["confidence"]),
                    row["bbox"],
                )
            )

        for batch_index, sample in enumerate(batch_samples):
            sample_id = str(sample["sample_id"])
            frame_row = frame_rows[sample_id]
            rendered_frames = []
            for offset, frame_key in enumerate(
                ("frame_before", "frame_after")
            ):
                metadata = frame_row[frame_key]
                rendered_frames.append(
                    {
                        "width": int(metadata["width"]),
                        "height": int(metadata["height"]),
                        "detections": grouped[2 * batch_index + offset],
                    }
                )
            examples.append(
                {
                    "sample_id": sample_id,
                    "event_id": str(sample["event_id"]),
                    "split": str(sample["split"]),
                    "state": str(state_rows[sample_id]["state"]),
                    "frame_before_sha256": str(
                        frame_row["frame_before"]["sha256"]
                    ),
                    "frame_after_sha256": str(
                        frame_row["frame_after"]["sha256"]
                    ),
                    "frames": rendered_frames,
                }
            )

    artifact = seal_muvs_formation_detections(
        {
            "plan_sha256": plan["artifact_sha256"],
            "frames_sha256": frames["artifact_sha256"],
            "review_sha256": review["artifact_sha256"],
            "detector": {
                "backend": "ultralytics_yolo",
                "model_name": "E-BARD/BODD_yolov8n_0001",
                "model_sha256": _file_sha256(model_path),
                "class_names": [
                    "basketball",
                    "rim",
                    "player",
                    "referee",
                ],
                "image_size": args.image_size,
                "confidence": args.confidence,
            },
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
    counts = {
        object_type: sum(
            detection["object_type"] == object_type
            for example in examples
            for frame in example["frames"]
            for detection in frame["detections"]
        )
        for object_type in ("basketball", "rim", "player", "referee")
    }
    print(
        json.dumps(
            {
                "output": str(args.output),
                "examples": len(examples),
                "device": args.device,
                "counts": counts,
                "artifact_sha256": artifact["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
