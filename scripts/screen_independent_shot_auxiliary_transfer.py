#!/usr/bin/env python3
"""Run a label-hidden DeepBall transfer screen over a frozen VLM plan.

The command never loads an annotation/review file.  It verifies raw-video
hashes against the plan, emits transfer-only event scores, and deliberately
leaves ``oof_predictions`` empty so the evidence gate cannot treat target-game
scores as game-held OOF evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.deepball_large import (  # noqa: E402
    infer_rgb_frame,
    load_pinned_deepball_large_model,
)
from app.analysis.independent_shot_auxiliary_transfer import (  # noqa: E402
    canonical_sha256,
    file_sha256,
    seal_transfer_artifact,
    summarize_event_scores,
    verify_label_hidden_plan,
)
from scripts.run_deepball_large_probe import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_MODEL,
    DEFAULT_MODEL_SHA256,
    DEFAULT_SOURCE,
    DEFAULT_SOURCE_SHA256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--video",
        action="append",
        required=True,
        metavar="VIDEO_SHA256=PATH",
        help="one mapping for every source hash in the plan",
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    return parser.parse_args()


def _parse_video_mappings(values: list[str]) -> dict[str, Path]:
    mappings: dict[str, Path] = {}
    for value in values:
        source_sha, separator, raw_path = value.partition("=")
        if separator != "=" or len(source_sha) != 64 or not raw_path:
            raise ValueError("--video must use SOURCE_SHA256=PATH")
        if source_sha in mappings:
            raise ValueError("duplicate --video source hash")
        mappings[source_sha] = Path(raw_path)
    return mappings


def _read_event(
    capture: cv2.VideoCapture,
    *,
    start_frame: int,
    end_frame: int,
    stride: int,
    score_threshold: float,
    loaded_model: Any,
) -> list[dict[str, Any]]:
    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError("event frame bounds are invalid")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    rows: list[dict[str, Any]] = []
    frame_number = start_frame
    while frame_number < end_frame:
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"could not decode frame {frame_number}")
        current = frame_number
        frame_number += 1
        if (current - start_frame) % stride:
            continue
        detections = infer_rgb_frame(
            loaded_model,
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            score_threshold=score_threshold,
        )
        rows.append({"frame": current, "detections": detections})
    if not rows:
        raise ValueError("event produced no sampled frames")
    return rows


def screen_transfer(
    *,
    plan_path: Path,
    video_map: dict[str, Path],
    output_path: Path | None = None,
    source_path: Path = DEFAULT_SOURCE,
    config_path: Path = DEFAULT_CONFIG,
    model_path: Path = DEFAULT_MODEL,
    sample_fps: float = 2.0,
    score_threshold: float = 0.5,
) -> dict[str, Any]:
    if sample_fps <= 0 or not 0.0 < score_threshold < 1.0:
        raise ValueError("sampling and score threshold must be positive")
    plan = verify_label_hidden_plan(json.loads(plan_path.read_text(encoding="utf-8")))
    expected_hashes = {str(row["source_video_sha256"]) for row in plan["examples"]}
    if set(video_map) != expected_hashes:
        raise ValueError("--video mappings must exactly cover plan source hashes")
    source_rows: dict[str, dict[str, Any]] = {}
    for source_sha, video_path in sorted(video_map.items()):
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        actual_sha = file_sha256(video_path)
        if actual_sha != source_sha:
            raise ValueError(f"raw video hash mismatch for {video_path}")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"could not open video {video_path}")
        source_rows[source_sha] = {
            "filename": video_path.name,
            "path": video_path.as_posix(),
            "sha256": actual_sha,
            "fps": float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        }
        capture.release()

    loaded_model = load_pinned_deepball_large_model(
        source_path=source_path,
        config_path=config_path,
        checkpoint_path=model_path,
        expected_source_sha256=DEFAULT_SOURCE_SHA256,
        expected_checkpoint_sha256=DEFAULT_MODEL_SHA256,
        device="cpu",
    )
    captures = {
        source_sha: cv2.VideoCapture(str(video_map[source_sha]))
        for source_sha in expected_hashes
    }
    if any(not capture.isOpened() for capture in captures.values()):
        for capture in captures.values():
            capture.release()
        raise ValueError("could not open every mapped video")

    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        for index, example in enumerate(plan["examples"], start=1):
            source_sha = str(example["source_video_sha256"])
            source_fps = source_rows[source_sha]["fps"]
            stride = max(1, int(round(source_fps / sample_fps)))
            detections = _read_event(
                captures[source_sha],
                start_frame=int(example["start_frame"]),
                end_frame=int(example["end_frame"]),
                stride=stride,
                score_threshold=score_threshold,
                loaded_model=loaded_model,
            )
            summary = summarize_event_scores(
                detections_by_frame=detections,
                start_frame=int(example["start_frame"]),
                end_frame=int(example["end_frame"]),
            )
            rows.append(
                {
                    "source_video_sha256": source_sha,
                    "candidate_bundle_sha256": str(example["candidate_bundle_sha256"]),
                    "event_id": str(example["event_id"]),
                    "start_frame": int(example["start_frame"]),
                    "end_frame": int(example["end_frame"]),
                    "source_fps": source_fps,
                    "sampling_stride_frames": stride,
                    **summary,
                }
            )
            print(
                f"event={index}/{len(plan['examples'])} "
                f"samples={summary['sampled_frame_count']} "
                f"score={summary['transfer_score']:.4f}",
                flush=True,
            )
    finally:
        for capture in captures.values():
            capture.release()

    artifact = seal_transfer_artifact(
        {
            "plan": {
                "path": plan_path.as_posix(),
                "plan_sha256": plan["plan_sha256"],
                "example_count": len(plan["examples"]),
                "labels_or_review_notes_exposed_to_model": False,
            },
            "model": {
                "backend": "wasb_deepball_large_basketball_v1",
                "source_path": source_path.as_posix(),
                "source_sha256": loaded_model.source_sha256,
                "config_path": config_path.as_posix(),
                "checkpoint_path": model_path.as_posix(),
                "checkpoint_sha256": loaded_model.checkpoint_sha256,
                "score_threshold": score_threshold,
                "device": "cpu",
            },
            "sampling": {
                "requested_fps": sample_fps,
                "event_count": len(rows),
            },
            "sources": [source_rows[key] for key in sorted(source_rows)],
            "transfer_predictions": rows,
            "runtime_seconds": time.monotonic() - started,
            "decision": {
                "threshold_selected": False,
                "oof_thresholds": [],
                "promotion_eligible": False,
                "reason": "transfer scores are uncalibrated and target-game OOF coverage is absent",
            },
        }
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return artifact


def main() -> int:
    args = parse_args()
    artifact = screen_transfer(
        plan_path=args.plan,
        video_map=_parse_video_mappings(args.video),
        output_path=args.output,
        source_path=args.source,
        config_path=args.config,
        model_path=args.model,
        sample_fps=args.sample_fps,
        score_threshold=args.score_threshold,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "artifact_sha256": artifact["artifact_sha256"],
                "event_count": len(artifact["transfer_predictions"]),
                "auxiliary_oof_evidence_available": artifact[
                    "auxiliary_oof_evidence_available"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
