#!/usr/bin/env python3
"""Screen DeepBall-Large on fixed, independently reviewed raw-video windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.deepball_large import (  # noqa: E402
    infer_rgb_frame,
    load_pinned_deepball_large_model,
)
from scripts.run_deepball_large_probe import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_MODEL,
    DEFAULT_MODEL_SHA256,
    DEFAULT_SOURCE,
    DEFAULT_SOURCE_SHA256,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def summarize_window_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    tp = sum(bool(row["event_present"]) and bool(row["prediction_present"]) for row in rows)
    fp = sum((not bool(row["event_present"])) and bool(row["prediction_present"]) for row in rows)
    fn = sum(bool(row["event_present"]) and (not bool(row["prediction_present"])) for row in rows)
    tn = sum((not bool(row["event_present"])) and (not bool(row["prediction_present"])) for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "window_count": len(rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def screen_windows(
    *,
    video_path: Path,
    labels_path: Path,
    output_path: Path | None = None,
    source_path: Path = DEFAULT_SOURCE,
    config_path: Path = DEFAULT_CONFIG,
    model_path: Path = DEFAULT_MODEL,
    sample_fps: float = 1.0,
    score_threshold: float = 0.5,
    device: str = "cpu",
) -> dict[str, Any]:
    if sample_fps <= 0 or not 0.0 < score_threshold < 1.0:
        raise ValueError("invalid sampling or score threshold")
    for path in (video_path, labels_path, source_path, config_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    labels_payload = json.loads(labels_path.read_text(encoding="utf-8"))
    examples = labels_payload.get("examples") or []
    if not examples:
        raise ValueError("labels contain no examples")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(source_fps / sample_fps)))
    loaded = load_pinned_deepball_large_model(
        source_path=source_path,
        config_path=config_path,
        checkpoint_path=model_path,
        expected_source_sha256=DEFAULT_SOURCE_SHA256,
        expected_checkpoint_sha256=DEFAULT_MODEL_SHA256,
        device=device,
    )

    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    total_samples = 0
    try:
        for example_index, example in enumerate(examples, start=1):
            start_frame = int(example["start_frame"])
            end_frame = int(example["end_frame"])
            if start_frame < 0 or end_frame <= start_frame:
                raise ValueError("label window bounds are invalid")
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_number = start_frame
            detections: list[dict[str, Any]] = []
            sampled_frames: list[int] = []
            while frame_number < end_frame:
                ok, frame = capture.read()
                if not ok:
                    break
                current_frame = frame_number
                frame_number += 1
                if (current_frame - start_frame) % stride:
                    continue
                sampled_frames.append(current_frame)
                current = infer_rgb_frame(
                    loaded,
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    score_threshold=score_threshold,
                )
                for detection in current:
                    detections.append({"frame": current_frame, **detection})
                total_samples += 1
            prediction_present = bool(detections)
            rows.append(
                {
                    "event_id": str(example["event_id"]),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "event_present": bool(example["event_present"]),
                    "prediction_present": prediction_present,
                    "detection_count": len(detections),
                    "detection_frames": sorted({int(item["frame"]) for item in detections}),
                    "sampled_frames": sampled_frames,
                    "detections": detections,
                }
            )
            print(
                f"window={example_index}/{len(examples)} samples={total_samples} "
                f"detections={sum(len(row['detections']) for row in rows)}",
                flush=True,
            )
    finally:
        capture.release()

    payload: dict[str, Any] = {
        "schema_version": "agu.deepball-large-window-screen.v1",
        "purpose": "offline_independent_cross_broadcast_window_screen",
        "runtime_consumable": False,
        "training_eligible": False,
        "codex_runtime_answer_used": False,
        "video_sha256": _file_sha256(video_path),
        "labels_sha256": _file_sha256(labels_path),
        "model_sha256": _file_sha256(model_path),
        "model": {
            "backend": "wasb_deepball_large_basketball_v1",
            "source_sha256": loaded.source_sha256,
            "checkpoint_sha256": loaded.checkpoint_sha256,
            "device": device,
            "score_threshold": score_threshold,
        },
        "video": {
            "source_fps": source_fps,
            "frame_count": frame_count,
        },
        "sampling": {
            "requested_fps": sample_fps,
            "stride_frames": stride,
            "window_count": len(rows),
            "sample_count": total_samples,
        },
        "runtime_seconds": time.monotonic() - started,
        "metrics": summarize_window_rows(rows),
        "accepted": False,
        "rows": rows,
    }
    payload["accepted"] = bool(
        payload["metrics"]["precision"] >= 0.85
        and payload["metrics"]["recall"] >= 0.85
    )
    payload["artifact_sha256"] = _canonical_sha256(payload)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return payload


def main() -> int:
    args = parse_args()
    artifact = screen_windows(
        video_path=args.video,
        labels_path=args.labels,
        output_path=args.output,
        source_path=args.source,
        config_path=args.config,
        model_path=args.model,
        sample_fps=args.sample_fps,
        score_threshold=args.score_threshold,
        device=args.device,
    )
    print(json.dumps({"metrics": artifact["metrics"], "accepted": artifact["accepted"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
