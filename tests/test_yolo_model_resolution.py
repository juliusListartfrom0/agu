from __future__ import annotations

import inspect
from pathlib import Path

from app.analysis.tracking import extract_tracked_frames
from app.config import Settings

ROOT = Path(__file__).resolve().parents[1]
LOCAL_YOLO_MODEL = "model_checkpoints/yolov8n.pt"


def test_default_yolo_model_uses_retained_local_checkpoint() -> None:
    settings = Settings(_env_file=None)

    assert settings.yolo_model_name == LOCAL_YOLO_MODEL
    assert (ROOT / settings.yolo_model_name).is_file()
    assert inspect.signature(extract_tracked_frames).parameters["yolo_model_name"].default == LOCAL_YOLO_MODEL


def test_yolo_model_environment_override_is_preserved(monkeypatch) -> None:
    monkeypatch.setenv("BASKETBALL_YOLO_MODEL_NAME", "custom/model.pt")

    assert Settings(_env_file=None).yolo_model_name == "custom/model.pt"
