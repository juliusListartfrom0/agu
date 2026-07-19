from __future__ import annotations

from typing import Any

from .adapters import DetectorAdapter
from .ultralytics_adapter import UltralyticsDetectorAdapter


def build_official_detector(settings: Any) -> DetectorAdapter | None:
    backend = str(settings.official_detector_backend).strip().lower()
    if backend in {"", "off", "none"}:
        return None
    if backend == "ultralytics_yolo":
        if not settings.official_detector_model_path:
            raise ValueError("official_detector_model_path is required for ultralytics_yolo")
        return UltralyticsDetectorAdapter(
            model_path=settings.official_detector_model_path,
            device=settings.official_detector_device,
            image_size=settings.official_detector_imgsz,
            confidence=settings.official_detector_confidence,
            tracking=settings.official_player_tracking_enabled,
            tracker_config=settings.official_player_tracker_config,
        )
    raise ValueError(f"unsupported official detector backend: {backend}")
