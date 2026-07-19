from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.analysis.schemas import BoundingBoxResponse, PerceptionDetectionResponse, Point2DResponse
from scripts.run_official_pose import run_pose_scan


class _PoseAdapter:
    name = "test_pose"

    def available(self) -> bool:
        return True

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


def test_pose_scan_is_bound_to_raw_video(tmp_path: Path) -> None:
    video = tmp_path / "raw.mov"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24)
    )
    for value in range(10):
        writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    writer.release()

    payload = run_pose_scan(
        video_path=video,
        model_path=tmp_path / "unused.pt",
        sample_fps=5,
        batch_size=2,
        device="cpu",
        image_size=64,
        confidence=0.1,
        keypoint_confidence=0.2,
        adapter=_PoseAdapter(),
    )

    assert payload["schema_version"] == "agu.official-pose.v1"
    assert payload["raw_video"]["filename"] == "raw.mov"
    assert payload["sampling"]["sample_count"] == 5
    assert len(payload["detections"]) == 5
    assert payload["detections"][0]["keypoints"]["right_wrist"] == {"x": 4.0, "y": 5.0}
