from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from app.analysis.official_evaluation import seal_raw_only_predictions
from app.analysis.schemas import (
    BoundingBoxResponse,
    GameEventResponse,
    PerceptionDetectionResponse,
    Point2DResponse,
)
from scripts.run_official_pose_windows import _merge_windows, run_pose_windows


class _PoseAdapter:
    name = "fixture_pose"

    def estimate(
        self, frames: Sequence[Any], frame_numbers: Sequence[int]
    ) -> Iterable[PerceptionDetectionResponse]:
        return [
            PerceptionDetectionResponse(
                detection_id=f"pose:{frame}:0",
                frame=frame,
                object_type="player",
                bbox=BoundingBoxResponse(x1=0, y1=0, x2=10, y2=20),
                confidence=0.9,
                keypoints={"right_wrist": Point2DResponse(x=4, y=5)},
                backend=self.name,
            )
            for frame in frame_numbers
        ]


def _event(event_id: str, start: int, end: int, *, event_type: str = "field_goal_attempt") -> GameEventResponse:
    return GameEventResponse(
        event_id=event_id,
        revision=1,
        event_type=event_type,
        source_video_id="video_001",
        start_frame=start,
        end_frame=end,
        status="needs_review",
    )


def _video(path: Path, frames: int = 80) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    for value in range(frames):
        writer.write(np.full((24, 32, 3), value % 255, dtype=np.uint8))
    writer.release()


def test_pose_windows_merge_overlap_and_filter_event_types(tmp_path: Path) -> None:
    video = tmp_path / "raw.mov"
    _video(video)
    source = seal_raw_only_predictions(
        game_id="fixture",
        raw_video_paths=[video],
        events=[
            _event("shot-1", 10, 20),
            _event("shot-2", 22, 30),
            _event("rebound-1", 60, 65, event_type="rebound"),
        ],
        config={"fixture": True},
    )

    payload = run_pose_windows(
        candidate_bundle=source,
        video_path=video,
        model_path=tmp_path / "unused.pt",
        event_types={"field_goal_attempt"},
        padding_sec=0.2,
        merge_gap_sec=0.1,
        sample_fps=5,
        batch_size=3,
        device="cpu",
        image_size=64,
        confidence=0.1,
        keypoint_confidence=0.2,
        adapter=_PoseAdapter(),
    )

    assert payload["candidate_source"]["bundle_sha256"] == source.bundle_sha256
    assert payload["candidate_source"]["event_count"] == 2
    assert payload["sampling"]["windows"] == [{"start_frame": 8, "end_frame": 33}]
    assert payload["sampling"]["sample_count"] == 13
    assert len({item["frame"] for item in payload["detections"]}) == 13


def test_pose_windows_reject_modified_raw_video(tmp_path: Path) -> None:
    video = tmp_path / "raw.mov"
    _video(video, frames=20)
    source = seal_raw_only_predictions(
        game_id="fixture",
        raw_video_paths=[video],
        events=[_event("shot-1", 2, 8)],
        config={"fixture": True},
    )
    video.write_bytes(video.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="matching raw video"):
        run_pose_windows(
            candidate_bundle=source,
            video_path=video,
            model_path=tmp_path / "unused.pt",
            event_types=None,
            padding_sec=0,
            merge_gap_sec=0,
            sample_fps=5,
            batch_size=2,
            device="cpu",
            image_size=64,
            confidence=0.1,
            keypoint_confidence=0.2,
            adapter=_PoseAdapter(),
        )


def test_merge_windows_respects_maximum_gap() -> None:
    assert _merge_windows([(21, 30), (10, 15), (16, 18), (40, 45)], maximum_gap=2) == [
        (10, 18),
        (21, 30),
        (40, 45),
    ]
