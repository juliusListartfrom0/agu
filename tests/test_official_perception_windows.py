from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from app.analysis.ball_candidate_review import verify_artifact
from app.analysis.official_evaluation import seal_raw_only_predictions
from app.analysis.schemas import (
    BoundingBoxResponse,
    GameEventResponse,
    PerceptionDetectionResponse,
)
from scripts.run_official_perception_windows import run_perception_windows


class _Detector:
    name = "fixture_detector"

    def available(self) -> bool:
        return True

    def detect(
        self, frames: Sequence[Any], frame_numbers: Sequence[int]
    ) -> Iterable[PerceptionDetectionResponse]:
        detections = []
        for frame in frame_numbers:
            detections.extend(
                [
                    PerceptionDetectionResponse(
                        detection_id=f"ball:{frame}",
                        frame=frame,
                        object_type="basketball",
                        bbox=BoundingBoxResponse(x1=frame, y1=1, x2=frame + 2, y2=3),
                        confidence=0.9,
                        backend=self.name,
                    ),
                    PerceptionDetectionResponse(
                        detection_id=f"player:{frame}",
                        frame=frame,
                        object_type="player",
                        bbox=BoundingBoxResponse(x1=0, y1=0, x2=10, y2=20),
                        confidence=0.8,
                        backend=self.name,
                    ),
                ]
            )
        return detections


def _event(event_id: str, start: int, end: int) -> GameEventResponse:
    return GameEventResponse(
        event_id=event_id,
        revision=1,
        event_type="field_goal_attempt",
        source_video_id="video_001",
        start_frame=start,
        end_frame=end,
        status="needs_review",
    )


def _video(path: Path, frames: int = 80) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24)
    )
    for value in range(frames):
        writer.write(np.full((24, 32, 3), value % 255, dtype=np.uint8))
    writer.release()


def test_perception_windows_merge_filter_and_bind_source(tmp_path: Path) -> None:
    video = tmp_path / "raw.mov"
    _video(video)
    source = seal_raw_only_predictions(
        game_id="fixture",
        raw_video_paths=[video],
        events=[_event("shot-1", 10, 20), _event("shot-2", 22, 30)],
        config={"fixture": True},
    )

    payload = run_perception_windows(
        candidate_bundle=source,
        video_path=video,
        model_path=tmp_path / "unused.pt",
        event_types={"field_goal_attempt"},
        object_types={"basketball"},
        padding_sec=0.2,
        merge_gap_sec=0.1,
        sample_fps=5,
        batch_size=3,
        device="cpu",
        image_size=64,
        confidence=0.1,
        adapter=_Detector(),
    )

    assert payload["candidate_source"]["bundle_sha256"] == source.bundle_sha256
    assert payload["candidate_source"]["event_count"] == 2
    assert payload["sampling"]["windows"] == [{"start_frame": 8, "end_frame": 33}]
    assert payload["sampling"]["sample_count"] == 13
    assert payload["counts"]["basketball"] == 13
    assert payload["counts"]["player"] == 0
    assert len(payload["ball_tracks"]) == 1
    assert verify_artifact(payload)["artifact_sha256"] == payload["artifact_sha256"]


def test_perception_windows_reject_unknown_backend(tmp_path: Path) -> None:
    video = tmp_path / "raw.mov"
    _video(video, frames=20)
    source = seal_raw_only_predictions(
        game_id="fixture",
        raw_video_paths=[video],
        events=[_event("shot-1", 2, 8)],
        config={"fixture": True},
    )

    with pytest.raises(ValueError, match="unsupported detector backend"):
        run_perception_windows(
            candidate_bundle=source,
            video_path=video,
            model_path=tmp_path / "unused.pt",
            event_types=None,
            object_types=None,
            padding_sec=0,
            merge_gap_sec=0,
            sample_fps=5,
            batch_size=2,
            device="cpu",
            image_size=64,
            confidence=0.1,
            backend="unknown",
            adapter=_Detector(),
        )


def test_perception_windows_accept_transformers_rfdetr_backend(
    tmp_path: Path,
) -> None:
    video = tmp_path / "raw.mov"
    _video(video, frames=20)
    source = seal_raw_only_predictions(
        game_id="fixture",
        raw_video_paths=[video],
        events=[_event("shot-1", 2, 8)],
        config={"fixture": True},
    )

    payload = run_perception_windows(
        candidate_bundle=source,
        video_path=video,
        model_path=tmp_path / "unused-model",
        event_types=None,
        object_types={"basketball"},
        padding_sec=0,
        merge_gap_sec=0,
        sample_fps=5,
        batch_size=2,
        device="cpu",
        image_size=64,
        confidence=0.1,
        backend="transformers_rfdetr",
        adapter=_Detector(),
    )

    assert payload["counts"]["basketball"] == 4


def test_perception_windows_reject_modified_raw_video(tmp_path: Path) -> None:
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
        run_perception_windows(
            candidate_bundle=source,
            video_path=video,
            model_path=tmp_path / "unused.pt",
            event_types=None,
            object_types=None,
            padding_sec=0,
            merge_gap_sec=0,
            sample_fps=5,
            batch_size=2,
            device="cpu",
            image_size=64,
            confidence=0.1,
            adapter=_Detector(),
        )
