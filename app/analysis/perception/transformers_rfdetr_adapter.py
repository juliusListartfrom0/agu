from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from app.analysis.schemas import BoundingBoxResponse, PerceptionDetectionResponse

SUPPORTED_CLASS_NAMES = {
    "ball": "basketball",
    "basketball": "basketball",
    "sports ball": "basketball",
    "player": "player",
    "referee": "referee",
    "rim": "rim",
}


class TransformersRFDETRDetectorAdapter:
    """Offline Transformers RF-DETR adapter with AGU-normalized detections."""

    name = "transformers_rfdetr"

    def __init__(
        self,
        *,
        model_path: str,
        device: str = "cpu",
        confidence: float = 0.25,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        if not model_path:
            raise ValueError("Transformers RF-DETR model_path must be non-empty")
        if not 0 < confidence <= 1:
            raise ValueError("Transformers RF-DETR confidence must be in (0, 1]")
        self.model_path = model_path
        self.device = device
        self.confidence = confidence
        self._model = model
        self._processor = processor
        self._load_error: str | None = None

    def available(self) -> bool:
        try:
            self._ensure_components()
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
        import torch

        model, processor = self._ensure_components()
        rgb_frames = [_as_rgb_array(frame) for frame in frames]
        inputs = processor(images=rgb_frames, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = model(**inputs)
        target_sizes = torch.tensor(
            [[frame.shape[0], frame.shape[1]] for frame in rgb_frames],
            dtype=torch.int64,
        )
        results = processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.confidence,
        )
        if len(results) != len(frame_numbers):
            raise RuntimeError(
                "Transformers RF-DETR result count does not match input frame count"
            )
        detections: list[PerceptionDetectionResponse] = []
        id2label = getattr(getattr(model, "config", None), "id2label", {})
        for frame_number, result in zip(frame_numbers, results):
            boxes = _tolist(result.get("boxes", []))
            scores = _tolist(result.get("scores", []))
            labels = _tolist(result.get("labels", []))
            for object_index, (box, score, label_id) in enumerate(
                zip(boxes, scores, labels)
            ):
                raw_name = id2label.get(int(label_id), id2label.get(str(int(label_id)), ""))
                object_type = SUPPORTED_CLASS_NAMES.get(
                    str(raw_name).strip().lower().replace("_", " ")
                )
                if object_type is None:
                    continue
                detections.append(
                    PerceptionDetectionResponse(
                        detection_id=f"{self.name}:{int(frame_number)}:{object_index}",
                        frame=int(frame_number),
                        object_type=object_type,
                        bbox=BoundingBoxResponse(
                            x1=box[0],
                            y1=box[1],
                            x2=box[2],
                            y2=box[3],
                        ),
                        confidence=float(score),
                        backend=self.name,
                    )
                )
        return detections

    def _ensure_components(self) -> tuple[Any, Any]:
        if self._processor is None or self._model is None:
            from transformers import AutoImageProcessor, AutoModelForObjectDetection

            self._processor = AutoImageProcessor.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
            self._model = AutoModelForObjectDetection.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
        self._model = self._model.to(self.device).eval()
        return self._model, self._processor


def _as_rgb_array(frame: Any) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Transformers RF-DETR frames must be HxWx3 BGR arrays")
    return np.ascontiguousarray(array[:, :, ::-1])


def _tolist(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)
