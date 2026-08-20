from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.analysis.ball_candidate_review import seal_artifact
from app.analysis.broadcast_ball_detector_dataset import (
    build_dataset_manifest,
    collect_review_records,
    normalized_bbox,
)


def _fixture_bundle(tmp_path: Path) -> tuple[dict, dict, Path]:
    video = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 24)
    )
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    frame[4:10, 8:14] = (0, 120, 240)
    writer.write(frame)
    writer.release()
    cap = cv2.VideoCapture(str(video))
    ok, decoded = cap.read()
    cap.release()
    assert ok
    pixels_sha = hashlib.sha256(decoded.tobytes()).hexdigest()
    plan = seal_artifact(
        {
            "schema_version": "agu.broadcast-ball-offline-review-plan.v1",
            "purpose": "codex_offline_same_detector_hard_negative_annotation",
            "runtime_consumable": False,
            "codex_runtime_answer_used": False,
            "perception_artifact_sha256": "p" * 64,
            "raw_video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "sampling": {"strategy": "fixture"},
            "review_scope": {"allowed": ["visible_basketball_inside_red_box"]},
            "candidates": [
                {
                    "candidate_id": "fixture-001",
                    "frame": 0,
                    "bbox": {"x1": 8.0, "y1": 4.0, "x2": 14.0, "y2": 10.0},
                    "confidence": 0.5,
                    "frame_pixels_sha256": pixels_sha,
                }
            ],
        }
    )
    review = seal_artifact(
        {
            "schema_version": "agu.broadcast-ball-offline-review.v1",
            "purpose": plan["purpose"],
            "runtime_consumable": False,
            "codex_runtime_answer_used": False,
            "plan_sha256": plan["artifact_sha256"],
            "reviewer": "fixture",
            "review_method": "fixture",
            "counts": {"valid_ball": 1},
            "decisions": [{"candidate_id": "fixture-001", "decision": "valid_ball"}],
        }
    )
    return plan, review, video


def test_normalized_bbox_clamps_to_frame_and_preserves_order() -> None:
    assert normalized_bbox(
        {"x1": -2, "y1": 4, "x2": 40, "y2": 20}, width=32, height=24
    ) == pytest.approx((0.0, 4 / 24, 1.0, 20 / 24))


def test_collect_review_records_binds_video_pixels_and_decision(tmp_path: Path) -> None:
    plan, review, video = _fixture_bundle(tmp_path)
    records = collect_review_records(
        source_id="fixture",
        plan=plan,
        review=review,
        video_path=video,
        split="train",
    )
    assert len(records) == 1
    assert records[0]["label"] == "valid_ball"
    assert records[0]["frame_pixels_sha256"] == plan["candidates"][0]["frame_pixels_sha256"]
    assert records[0]["width"] == 32
    assert records[0]["height"] == 24


def test_manifest_rejects_duplicate_frame_with_conflicting_decisions(tmp_path: Path) -> None:
    plan, review, video = _fixture_bundle(tmp_path)
    conflicting = json.loads(json.dumps(review))
    conflicting["decisions"][0]["decision"] = "false_positive"
    conflicting = seal_artifact({k: v for k, v in conflicting.items() if k != "artifact_sha256"})
    records = collect_review_records(
        source_id="a",
        plan=plan,
        review=review,
        video_path=video,
        split="train",
    ) + collect_review_records(
        source_id="b",
        plan=plan,
        review=conflicting,
        video_path=video,
        split="train",
    )
    with pytest.raises(ValueError, match="conflicting decision"):
        build_dataset_manifest(
            [
                {
                    "source_id": "a",
                    "video_sha256": records[0]["source_video_sha256"],
                    "plan_sha256": plan["artifact_sha256"],
                    "review_sha256": review["artifact_sha256"],
                    "split": "train",
                },
                {
                    "source_id": "b",
                    "video_sha256": records[1]["source_video_sha256"],
                    "plan_sha256": plan["artifact_sha256"],
                    "review_sha256": conflicting["artifact_sha256"],
                    "split": "train",
                },
            ],
            records=records,
        )
