from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.export_perception_detection_montage import (
    _evenly_sample,
    export_detection_montage,
)
from scripts.run_official_perception import _file_sha256


def _video(path: Path, frames: int = 20) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
    )
    for value in range(frames):
        writer.write(np.full((48, 64, 3), value % 255, dtype=np.uint8))
    writer.release()


def _artifact(video: Path) -> dict[str, object]:
    return {
        "schema_version": "agu.official-perception.v1",
        "raw_video": {
            "filename": video.name,
            "sha256": _file_sha256(video),
        },
        "detections": [
            {
                "detection_id": f"ball:{frame}",
                "frame": frame,
                "object_type": "basketball",
                "bbox": {"x1": 10, "y1": 10, "x2": 16, "y2": 16},
                "confidence": 0.8,
                "backend": "fixture",
            }
            for frame in (2, 5, 8, 11)
        ],
    }


def test_export_detection_montage_verifies_and_renders(tmp_path: Path) -> None:
    video = tmp_path / "raw.mov"
    artifact = tmp_path / "perception.json"
    output = tmp_path / "montage.jpg"
    _video(video)
    artifact.write_text(json.dumps(_artifact(video)), encoding="utf-8")

    result = export_detection_montage(
        perception_path=artifact,
        video_path=video,
        output_path=output,
        object_types={"basketball"},
        maximum_frames=3,
        columns=2,
        tile_width=128,
    )

    assert result["selected_frame_count"] == 3
    assert output.is_file()
    assert cv2.imread(str(output)).shape[1] == 256


def test_export_detection_montage_rejects_modified_video(tmp_path: Path) -> None:
    video = tmp_path / "raw.mov"
    artifact = tmp_path / "perception.json"
    _video(video)
    artifact.write_text(json.dumps(_artifact(video)), encoding="utf-8")
    video.write_bytes(video.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="does not match raw video"):
        export_detection_montage(
            perception_path=artifact,
            video_path=video,
            output_path=tmp_path / "montage.jpg",
            object_types=None,
            maximum_frames=3,
            columns=2,
            tile_width=128,
        )


def test_evenly_sample_keeps_endpoints() -> None:
    assert _evenly_sample([1, 2, 3, 4, 5], maximum=3) == [1, 3, 5]
