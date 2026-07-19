from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from app.analysis.schemas import (
    BoundingBoxResponse,
    PerceptionDetectionResponse,
    Point2DResponse,
)

from .ultralytics_adapter import _tolist

COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


class UltralyticsPoseAdapter:
    """Optional pose adapter for experiments; deployers must review its license."""

    name = "ultralytics_pose"

    def __init__(
        self,
        *,
        model_path: str,
        device: str = "cpu",
        image_size: int = 640,
        confidence: float = 0.2,
        keypoint_confidence: float = 0.2,
        model: Any | None = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.image_size = image_size
        self.confidence = confidence
        self.keypoint_confidence = keypoint_confidence
        self._model = model
        self._load_error: str | None = None

    def available(self) -> bool:
        try:
            self._ensure_model()
        except Exception as exc:  # optional dependency/model availability boundary
            self._load_error = f"{type(exc).__name__}: {exc}"
            return False
        return True

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def estimate(
        self,
        frames: Sequence[Any],
        frame_numbers: Sequence[int],
    ) -> Iterable[PerceptionDetectionResponse]:
        if len(frames) != len(frame_numbers):
            raise ValueError("frames and frame_numbers must have equal length")
        model = self._ensure_model()
        results = model.predict(
            source=list(frames),
            device=self.device,
            imgsz=self.image_size,
            conf=self.confidence,
            verbose=False,
        )
        detections: list[PerceptionDetectionResponse] = []
        for frame_number, result in zip(frame_numbers, results):
            boxes = getattr(result, "boxes", None)
            poses = getattr(result, "keypoints", None)
            if boxes is None or poses is None:
                continue
            coordinates = _tolist(boxes.xyxy)
            confidences = _tolist(boxes.conf)
            pose_xy = _tolist(poses.xy)
            raw_pose_conf = getattr(poses, "conf", None)
            pose_conf = _tolist(raw_pose_conf) if raw_pose_conf is not None else []
            for index, (xyxy, score, points) in enumerate(
                zip(coordinates, confidences, pose_xy)
            ):
                point_confidences = pose_conf[index] if index < len(pose_conf) else []
                keypoints = {
                    name: Point2DResponse(x=float(point[0]), y=float(point[1]))
                    for point_index, (name, point) in enumerate(zip(COCO_KEYPOINT_NAMES, points))
                    if (
                        len(point) >= 2
                        and float(point[0]) > 0
                        and float(point[1]) > 0
                        and (
                            point_index >= len(point_confidences)
                            or float(point_confidences[point_index]) >= self.keypoint_confidence
                        )
                    )
                }
                detections.append(
                    PerceptionDetectionResponse(
                        detection_id=f"{self.name}:{int(frame_number)}:{index}",
                        frame=int(frame_number),
                        object_type="player",
                        bbox=BoundingBoxResponse(
                            x1=float(xyxy[0]),
                            y1=float(xyxy[1]),
                            x2=float(xyxy[2]),
                            y2=float(xyxy[3]),
                        ),
                        confidence=float(score),
                        keypoints=keypoints,
                        backend=self.name,
                    )
                )
        return detections

    def _ensure_model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
        return self._model
