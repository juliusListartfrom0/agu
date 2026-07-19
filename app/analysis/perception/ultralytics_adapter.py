from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from app.analysis.schemas import BoundingBoxResponse, PerceptionDetectionResponse

CLASS_MAP = {
    "person": "player",
    "player": "player",
    "sports ball": "basketball",
    "basketball": "basketball",
    "ball": "basketball",
    "rim": "rim",
    "hoop": "rim",
    "basketball hoop": "rim",
    "basketball rim": "rim",
    "backboard": "backboard",
    "referee": "referee",
}


class UltralyticsDetectorAdapter:
    """Optional Ultralytics adapter; license choice belongs to the deployer."""

    name = "ultralytics_yolo"

    def __init__(
        self,
        *,
        model_path: str,
        device: str = "cpu",
        image_size: int = 640,
        confidence: float = 0.2,
        tracking: bool = False,
        tracker_config: str = "bytetrack.yaml",
        class_prompts: Sequence[str] | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.image_size = image_size
        self.confidence = confidence
        self.tracking = tracking
        self.tracker_config = tracker_config
        self.class_prompts = tuple(
            prompt.strip() for prompt in (class_prompts or ()) if prompt.strip()
        )
        self._model = model
        self._prompts_configured = False
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

    def detect(
        self,
        frames: Sequence[Any],
        frame_numbers: Sequence[int],
    ) -> Iterable[PerceptionDetectionResponse]:
        if len(frames) != len(frame_numbers):
            raise ValueError("frames and frame_numbers must have equal length")
        model = self._ensure_model()
        inference = model.track if self.tracking else model.predict
        kwargs = {
            "source": list(frames),
            "device": self.device,
            "imgsz": self.image_size,
            "conf": self.confidence,
            "verbose": False,
        }
        if self.tracking:
            kwargs.update({"persist": True, "tracker": self.tracker_config})
        results = inference(**kwargs)
        detections: list[PerceptionDetectionResponse] = []
        for batch_index, (frame_number, result) in enumerate(zip(frame_numbers, results)):
            names = getattr(result, "names", None) or getattr(model, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            coordinates = _tolist(boxes.xyxy)
            confidences = _tolist(boxes.conf)
            classes = _tolist(boxes.cls)
            track_ids = _tolist(boxes.id) if getattr(boxes, "id", None) is not None else [None] * len(coordinates)
            for object_index, (xyxy, score, class_index, track_id) in enumerate(
                zip(coordinates, confidences, classes, track_ids)
            ):
                class_name = str(names.get(int(class_index), class_index)).strip().lower()
                object_type = CLASS_MAP.get(class_name)
                if object_type is None:
                    continue
                detections.append(
                    PerceptionDetectionResponse(
                        detection_id=f"{self.name}:{int(frame_number)}:{object_index}",
                        frame=int(frame_number),
                        object_type=object_type,
                        bbox=BoundingBoxResponse(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3]),
                        confidence=float(score),
                        track_id=f"track-{int(track_id)}" if track_id is not None else None,
                        player_id=(
                            f"raw-player-track-{int(track_id)}"
                            if object_type == "player" and track_id is not None
                            else None
                        ),
                        backend=self.name,
                    )
                )
        return detections

    def _ensure_model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
        if self.class_prompts and not self._prompts_configured:
            set_classes = getattr(self._model, "set_classes", None)
            if not callable(set_classes):
                raise ValueError("configured class prompts require a set_classes-capable model")
            set_classes(list(self.class_prompts))
            self._prompts_configured = True
        return self._model


def _tolist(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)
