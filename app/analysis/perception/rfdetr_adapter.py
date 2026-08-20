from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from app.analysis.schemas import BoundingBoxResponse, PerceptionDetectionResponse

DEFAULT_E_BARD_CLASSES = ("basketball", "rim", "player", "referee")


class RFDETRDetectorAdapter:
    """Optional RF-DETR adapter for independently trained detector evidence."""

    name = "rfdetr"

    def __init__(
        self,
        *,
        model_path: str,
        device: str = "cpu",
        image_size: int = 704,
        confidence: float = 0.25,
        class_names: Sequence[str] = DEFAULT_E_BARD_CLASSES,
        model: Any | None = None,
    ) -> None:
        normalized_names = tuple(name.strip().lower() for name in class_names)
        if not normalized_names or any(not name for name in normalized_names):
            raise ValueError("RF-DETR class_names must be non-empty")
        self.model_path = model_path
        self.device = device
        self.image_size = image_size
        self.confidence = confidence
        self.class_names = normalized_names
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

    def detect(
        self,
        frames: Sequence[Any],
        frame_numbers: Sequence[int],
    ) -> Iterable[PerceptionDetectionResponse]:
        if len(frames) != len(frame_numbers):
            raise ValueError("frames and frame_numbers must have equal length")
        if not frames:
            return []
        model = self._ensure_model()
        rgb_frames = [_as_rgb_array(frame) for frame in frames]
        results = model.predict(
            rgb_frames,
            threshold=self.confidence,
            shape=(self.image_size, self.image_size),
            include_source_image=False,
        )
        if not isinstance(results, list):
            results = [results]
        detections: list[PerceptionDetectionResponse] = []
        for frame_number, result in zip(frame_numbers, results):
            coordinates = _tolist(getattr(result, "xyxy", []))
            confidences = _tolist(getattr(result, "confidence", []))
            classes = _tolist(getattr(result, "class_id", []))
            for object_index, (xyxy, score, class_index) in enumerate(
                zip(coordinates, confidences, classes)
            ):
                index = int(class_index)
                # Current RF-DETR may expose the legacy checkpoint's background
                # logit as an extra class. It is never an AGU object.
                if index < 0 or index >= len(self.class_names):
                    continue
                detections.append(
                    PerceptionDetectionResponse(
                        detection_id=f"{self.name}:{int(frame_number)}:{object_index}",
                        frame=int(frame_number),
                        object_type=self.class_names[index],
                        bbox=BoundingBoxResponse(
                            x1=xyxy[0],
                            y1=xyxy[1],
                            x2=xyxy[2],
                            y2=xyxy[3],
                        ),
                        confidence=float(score),
                        backend=self.name,
                    )
                )
        return detections

    def _ensure_model(self) -> Any:
        if self._model is None:
            from rfdetr import RFDETRNano

            self._model = RFDETRNano(
                pretrain_weights=self.model_path,
                num_classes=len(self.class_names),
                resolution=self.image_size,
                device=self.device,
            )
        context = getattr(self._model, "model", None)
        if context is not None:
            context.class_names = list(self.class_names)
        return self._model


def _as_rgb_array(frame: Any) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("RF-DETR frames must be HxWx3 BGR arrays")
    return np.ascontiguousarray(array[:, :, ::-1])


def _tolist(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)
