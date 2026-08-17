#!/usr/bin/env python3
"""Screen a tiled ball detector on fixed, independently labelled video windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_tiled_ball_perception import run_tiled_ball_scan  # noqa: E402


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--tile-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def screen_windows(
    *,
    video_path: Path,
    model_path: Path,
    labels_path: Path,
    sample_fps: float,
    image_size: int,
    confidence: float,
    tile_batch_size: int,
    device: str,
) -> dict[str, Any]:
    labels_payload = json.loads(labels_path.read_text(encoding="utf-8"))
    examples = labels_payload.get("examples") or []
    if not examples:
        raise ValueError("labels contain no examples")
    rows: list[dict[str, Any]] = []
    for example in examples:
        start_frame = int(example["start_frame"])
        end_frame = int(example["end_frame"])
        scan = run_tiled_ball_scan(
            video_path=video_path,
            model_path=model_path,
            sample_fps=sample_fps,
            start_frame=start_frame,
            end_frame=end_frame,
            image_size=image_size,
            confidence=confidence,
            tile_batch_size=tile_batch_size,
            device=device,
        )
        detections = scan["detections"]
        rows.append(
            {
                "event_id": str(example["event_id"]),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "event_present": bool(example["event_present"]),
                "prediction_present": bool(detections),
                "detection_count": len(detections),
                "detection_frames": sorted({int(item["frame"]) for item in detections}),
            }
        )
    tp = sum(row["event_present"] and row["prediction_present"] for row in rows)
    fp = sum((not row["event_present"]) and row["prediction_present"] for row in rows)
    fn = sum(row["event_present"] and (not row["prediction_present"]) for row in rows)
    tn = sum((not row["event_present"]) and (not row["prediction_present"]) for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    artifact: dict[str, Any] = {
        "schema_version": "agu.tiled-ball-window-screen.v1",
        "purpose": "offline_independent_cross_broadcast_window_screen",
        "runtime_consumable": False,
        "training_eligible": False,
        "codex_runtime_answer_used": False,
        "video_sha256": _file_sha256(video_path),
        "model_sha256": _file_sha256(model_path),
        "labels_sha256": _file_sha256(labels_path),
        "label_policy": "fixed event_present labels from sealed raw-video review; prediction is any basketball detection in the fixed window",
        "sampling": {
            "requested_fps": sample_fps,
            "image_size": image_size,
            "confidence": confidence,
            "tile_batch_size": tile_batch_size,
            "device": device,
        },
        "metrics": {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "window_count": len(rows),
        },
        "accepted": bool(precision >= 0.85 and recall >= 0.85),
        "rows": rows,
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def main() -> int:
    args = parse_args()
    artifact = screen_windows(
        video_path=args.video,
        model_path=args.model,
        labels_path=args.labels,
        sample_fps=args.sample_fps,
        image_size=args.image_size,
        confidence=args.confidence,
        tile_batch_size=args.tile_batch_size,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": artifact["metrics"], "accepted": artifact["accepted"], "artifact_sha256": artifact["artifact_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
