from __future__ import annotations

import numpy as np
import pytest
import torch

from app.analysis.deepball_large import (
    DEFAULT_INPUT_WH,
    DEFAULT_OUTPUT_WH,
    decode_deepball_logits,
    preprocess_rgb_frame,
)


def test_preprocess_rgb_frame_matches_pinned_model_contract() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    tensor, inverse_affine = preprocess_rgb_frame(frame)

    assert tensor.shape == (1, 3, DEFAULT_INPUT_WH[1], DEFAULT_INPUT_WH[0])
    assert tensor.dtype == torch.float32
    assert inverse_affine.shape == (2, 3)


def test_decode_deepball_logits_returns_foreground_peak_in_source_coordinates() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    _, inverse_affine = preprocess_rgb_frame(frame)
    logits = torch.full((1, 2, DEFAULT_OUTPUT_WH[1], DEFAULT_OUTPUT_WH[0]), -10.0)
    logits[0, 1, 90, 160] = 10.0

    detections = decode_deepball_logits(logits, inverse_affine, score_threshold=0.5)

    assert len(detections) == 1
    assert detections[0]["score"] > 0.99
    assert detections[0]["x"] == pytest.approx(320.0, abs=2.0)
    assert detections[0]["y"] == pytest.approx(180.0, abs=2.0)


def test_decode_deepball_logits_abstains_below_threshold() -> None:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    _, inverse_affine = preprocess_rgb_frame(frame)
    logits = torch.zeros((1, 2, DEFAULT_OUTPUT_WH[1], DEFAULT_OUTPUT_WH[0]))

    assert decode_deepball_logits(logits, inverse_affine, score_threshold=0.9) == []


def test_preprocess_rgb_frame_rejects_non_rgb_input() -> None:
    with pytest.raises(ValueError, match="uint8 RGB"):
        preprocess_rgb_frame(np.zeros((360, 640), dtype=np.uint8))
