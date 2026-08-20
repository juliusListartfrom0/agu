from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis.perception import (
    TransformersRFDETRDetectorAdapter,
    UltralyticsDetectorAdapter,
)
from scripts.run_causal_phase_ball_detection import (
    _build_detector,
    _resolve_model_artifact_path,
)


def test_build_detector_selects_local_transformers_rfdetr_backend() -> None:
    detector = _build_detector(
        backend="transformers_rfdetr",
        model_path=Path("/models/basketball-rfdetr"),
        device="mps",
        image_size=384,
        confidence=0.15,
    )

    assert isinstance(detector, TransformersRFDETRDetectorAdapter)
    assert detector.model_path == "/models/basketball-rfdetr"
    assert detector.device == "mps"
    assert detector.confidence == 0.15


def test_build_detector_preserves_ultralytics_backend() -> None:
    detector = _build_detector(
        backend="ultralytics_yolo",
        model_path=Path("/models/ball.pt"),
        device="cpu",
        image_size=704,
        confidence=0.1,
    )

    assert isinstance(detector, UltralyticsDetectorAdapter)
    assert detector.model_path == "/models/ball.pt"


def test_resolve_model_artifact_path_hashes_transformers_safetensors(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weight = model_dir / "model.safetensors"
    weight.write_bytes(b"safe")

    assert _resolve_model_artifact_path(model_dir) == weight
    assert _resolve_model_artifact_path(weight) == weight

    missing_dir = tmp_path / "missing"
    missing_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        _resolve_model_artifact_path(missing_dir)
