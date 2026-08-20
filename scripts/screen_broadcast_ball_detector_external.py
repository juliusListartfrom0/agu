#!/usr/bin/env python3
"""Run an offline detector on a sealed, disjoint broadcast review set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from app.analysis.ball_candidate_review import canonical_sha256, verify_artifact
from app.analysis.broadcast_ball_detector_dataset import file_sha256
from app.analysis.broadcast_ball_detector_screen import detector_frame_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.25)
    return parser.parse_args()


def _read_frame(cap: cv2.VideoCapture, frame_index: int) -> tuple[Any, str]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"could not decode frame {frame_index}")
    return frame, hashlib.sha256(frame.tobytes()).hexdigest()


def _verify_frame_pixels_sha(candidate: dict[str, Any], pixels_sha: str) -> None:
    """Verify pixels only when the sealed candidate carries a pixel hash.

    Older detector review plans use ``frame_sha256`` for the detector's source
    record, not for decoded OpenCV pixels.  Treating that field as a pixel hash
    makes otherwise valid sealed plans fail before model inference.
    """
    expected_pixels_sha = candidate.get("frame_pixels_sha256")
    if expected_pixels_sha and str(expected_pixels_sha) != pixels_sha:
        raise ValueError(f"frame pixel hash mismatch for {candidate['candidate_id']}")


def main() -> int:
    args = parse_args()
    if not 0 < args.confidence <= 1 or not 0 < args.iou <= 1:
        raise ValueError("confidence and IoU must be in (0, 1]")
    plan = verify_artifact(json.loads(args.plan.read_text(encoding="utf-8")))
    review = verify_artifact(json.loads(args.review.read_text(encoding="utf-8")))
    if review.get("plan_sha256") != plan["artifact_sha256"]:
        raise ValueError("review is not bound to plan")
    raw_sha = file_sha256(args.video)
    if raw_sha != plan.get("raw_video_sha256"):
        raise ValueError("external video hash mismatch")
    candidate_map = {str(row["candidate_id"]): row for row in plan["candidates"]}
    decisions: dict[str, str] = {}
    for row in review["decisions"]:
        candidate_id = str(row["candidate_id"])
        decision = str(row["decision"])
        if candidate_id not in candidate_map or decision not in {"valid_ball", "false_positive", "uncertain"}:
            raise ValueError("review contains an unknown candidate or decision")
        decisions[candidate_id] = decision
    if len(decisions) != len(candidate_map):
        raise ValueError("review does not cover every candidate")

    rows: list[dict[str, Any]] = []
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise ValueError(f"could not open video {args.video}")
    model = YOLO(str(args.model))
    try:
        for candidate_id, candidate in candidate_map.items():
            decision = decisions[candidate_id]
            if decision == "uncertain":
                continue
            frame, pixels_sha = _read_frame(cap, int(candidate["frame"]))
            _verify_frame_pixels_sha(candidate, pixels_sha)
            result = model.predict(
                source=frame,
                conf=args.confidence,
                iou=0.7,
                imgsz=640,
                device="cpu",
                max_det=20,
                verbose=False,
            )[0]
            boxes = result.boxes
            predicted_boxes = boxes.xyxy.cpu().tolist() if boxes is not None else []
            predicted_conf = boxes.conf.cpu().tolist() if boxes is not None else []
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "frame": int(candidate["frame"]),
                    "decision": decision,
                    "bbox": dict(candidate["bbox"]),
                    "frame_pixels_sha256": pixels_sha,
                    "predicted_boxes": predicted_boxes,
                    "predicted_confidence": predicted_conf,
                }
            )
    finally:
        cap.release()

    metrics = detector_frame_metrics(
        rows,
        [row["predicted_boxes"] for row in rows],
        iou_threshold=args.iou,
    )
    artifact: dict[str, Any] = {
        "schema_version": "agu.broadcast-ball-detector-external-screen.v1",
        "purpose": "offline_disjoint_broadcast_detector_screen",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "model_sha256": file_sha256(args.model),
        "raw_video_sha256": raw_sha,
        "plan_sha256": plan["artifact_sha256"],
        "review_sha256": review["artifact_sha256"],
        "confidence": args.confidence,
        "iou_threshold": args.iou,
        "label_policy": "valid rows require an IoU-matched model box; false-positive rows count every model box as FP; uncertain rows excluded",
        "metrics": metrics,
        "accepted": bool(metrics["precision"] >= 0.85 and metrics["recall"] >= 0.85),
        "predictions": rows,
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics, "accepted": artifact["accepted"], "artifact_sha256": artifact["artifact_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
