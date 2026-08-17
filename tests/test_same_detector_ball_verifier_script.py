from __future__ import annotations

import numpy as np

from scripts import screen_ball_candidate_verifier
from scripts.screen_same_detector_ball_verifier import _determinate_source_rows


class _FakeCapture:
    def __init__(self, _path: str) -> None:
        self.frame = 0

    def set(self, _property: int, frame: int) -> None:
        self.frame = int(frame)

    def read(self) -> tuple[bool, np.ndarray]:
        return True, np.full((20, 20, 3), self.frame, dtype=np.uint8)

    def release(self) -> None:
        return None


class _FakeEmbedder:
    def encode(self, images: list[object]) -> np.ndarray:
        return np.arange(len(images), dtype=np.float32).reshape(-1, 1)


def test_broadcast_features_can_decode_only_selected_detections(
    monkeypatch,
    tmp_path,
) -> None:
    video = tmp_path / "game.mp4"
    video.touch()
    perception = {
        "raw_video": {"sha256": "video-sha"},
        "detections": [
            {
                "detection_id": "keep",
                "frame": 3,
                "bbox": {"x1": 1, "y1": 1, "x2": 5, "y2": 5},
            },
            {
                "detection_id": "skip",
                "frame": 7,
                "bbox": {"x1": 1, "y1": 1, "x2": 5, "y2": 5},
            },
        ],
    }
    monkeypatch.setattr(screen_ball_candidate_verifier, "_sha256", lambda _path: "video-sha")
    monkeypatch.setattr(
        screen_ball_candidate_verifier.cv2, "VideoCapture", _FakeCapture
    )

    features, rows = screen_ball_candidate_verifier._broadcast_features(
        perception,
        video=video,
        embedder=_FakeEmbedder(),
        detection_ids={"keep"},
    )

    assert features.shape == (1, 1)
    assert [row["detection_id"] for row in rows] == ["keep"]


def test_determinate_source_rows_excludes_uncertain_and_namespaces_groups() -> None:
    perception = {
        "artifact_sha256": "perception-sha",
        "raw_video": {"sha256": "game-sha"},
        "sampling": {"windows": [{"start_frame": 0, "end_frame": 20}]},
    }
    plan = {
        "artifact_sha256": "plan-sha",
        "perception_artifact_sha256": "perception-sha",
        "candidates": [
            {
                "candidate_id": "one",
                "detection_id": "d:1",
                "frame": 1,
                "confidence": 0.2,
            },
            {
                "candidate_id": "two",
                "detection_id": "d:2",
                "frame": 2,
                "confidence": 0.3,
            },
            {
                "candidate_id": "three",
                "detection_id": "d:3",
                "frame": 3,
                "confidence": 0.4,
            },
        ],
    }
    review = {
        "artifact_sha256": "review-sha",
        "plan_sha256": "plan-sha",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "decisions": [
            {"candidate_id": "one", "decision": "valid_ball"},
            {"candidate_id": "two", "decision": "uncertain"},
            {"candidate_id": "three", "decision": "false_positive"},
        ],
    }

    rows = _determinate_source_rows(perception, plan, review)

    assert [row["detection_id"] for row in rows] == ["d:1", "d:3"]
    assert [row["label"] for row in rows] == [1, 0]
    assert {row["group"] for row in rows} == {"game-sha:window-001"}
