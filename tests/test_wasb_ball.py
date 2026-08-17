from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from app.analysis.wasb_ball import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    affine_point,
    decode_heatmap_components,
    frame_affine,
    load_pinned_wasb_model,
    preprocess_rgb_triplet,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frame_affine_round_trips_non_widescreen_frame() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    forward = frame_affine(frame.shape, (512, 288))
    inverse = frame_affine(frame.shape, (512, 288), inverse=True)
    source_point = np.asarray([173.25, 302.5], dtype=np.float32)

    output_point = affine_point(source_point, forward)
    restored_point = affine_point(output_point, inverse)

    assert output_point.tolist() == pytest.approx([138.6, 194.0], abs=1e-4)
    assert restored_point.tolist() == pytest.approx(source_point.tolist(), abs=1e-4)


def test_preprocess_rgb_triplet_matches_official_channel_contract() -> None:
    frames = [
        np.full((288, 512, 3), fill_value=value, dtype=np.uint8)
        for value in (0, 127, 255)
    ]

    batch, inverse = preprocess_rgb_triplet(frames)

    assert batch.shape == (1, 9, 288, 512)
    assert batch.dtype == torch.float32
    expected_zero = (np.zeros(3, dtype=np.float32) - IMAGENET_MEAN) / IMAGENET_STD
    assert batch[0, :3, 0, 0].numpy().tolist() == pytest.approx(
        expected_zero.tolist()
    )
    assert affine_point(
        np.asarray([256.0, 144.0], dtype=np.float32), inverse
    ).tolist() == pytest.approx([256.0, 144.0])


def test_preprocess_rgb_triplet_rejects_mixed_shapes() -> None:
    frames = [
        np.zeros((288, 512, 3), dtype=np.uint8),
        np.zeros((300, 512, 3), dtype=np.uint8),
        np.zeros((288, 512, 3), dtype=np.uint8),
    ]

    with pytest.raises(ValueError, match="same shape"):
        preprocess_rgb_triplet(frames)


def test_decode_heatmap_components_uses_weighted_center_and_inverse_affine() -> None:
    logits = torch.full((1, 3, 288, 512), -10.0)
    logits[0, 1, 100:102, 200:202] = torch.tensor(
        [[7.0, 8.0], [9.0, 10.0]]
    )
    inverse = frame_affine((288, 512, 3), (512, 288), inverse=True)

    detections = decode_heatmap_components(logits, inverse, threshold=0.5)

    assert len(detections) == 3
    assert detections[0] == []
    assert detections[2] == []
    assert len(detections[1]) == 1
    detection = detections[1][0]
    weights = torch.sigmoid(torch.tensor([7.0, 8.0, 9.0, 10.0])).numpy()
    expected_x = float(np.dot(np.asarray([200, 201, 200, 201]), weights) / weights.sum())
    expected_y = float(np.dot(np.asarray([100, 100, 101, 101]), weights) / weights.sum())
    assert detection["x"] == pytest.approx(expected_x)
    assert detection["y"] == pytest.approx(expected_y)
    assert detection["peak_probability"] == pytest.approx(
        float(torch.sigmoid(torch.tensor(10.0)))
    )
    assert detection["component_score"] == pytest.approx(float(weights.sum()))
    assert detection["pixel_count"] == 4


def test_load_pinned_wasb_model_loads_strict_safe_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "tiny_hrnet.py"
    source.write_text(
        "\n".join(
            [
                "import torch",
                "class HRNet(torch.nn.Module):",
                "    def __init__(self, cfg):",
                "        super().__init__()",
                "        self.proj = torch.nn.Conv2d(",
                "            3 * cfg.frames_in, cfg.frames_out, kernel_size=1",
                "        )",
                "    def forward(self, value):",
                "        return {0: self.proj(value)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "model.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "frames_in": 3,
                "frames_out": 3,
                "out_scales": [0],
                "MODEL": {"EXTRA": {"DECONV": {"NUM_DECONVS": 0}}},
            }
        ),
        encoding="utf-8",
    )
    module_namespace: dict[str, object] = {}
    exec(compile(source.read_text(), str(source), "exec"), module_namespace)
    model = module_namespace["HRNet"](
        type("Config", (), {"frames_in": 3, "frames_out": 3})()
    )
    checkpoint = tmp_path / "weights.pth.tar"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)

    loaded = load_pinned_wasb_model(
        source_path=source,
        config_path=config,
        checkpoint_path=checkpoint,
        expected_source_sha256=_sha256(source),
        expected_checkpoint_sha256=_sha256(checkpoint),
        device="cpu",
    )

    output = loaded.model(torch.zeros((1, 9, 4, 4)))
    assert loaded.device.type == "cpu"
    assert output[0].shape == (1, 3, 4, 4)
    assert not loaded.model.training


def test_load_pinned_wasb_model_rejects_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "model.py"
    source.write_text("class HRNet: pass\n", encoding="utf-8")
    config = tmp_path / "model.yaml"
    config.write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "weights.pth.tar"
    torch.save({"model_state_dict": {}}, checkpoint)

    with pytest.raises(ValueError, match="source SHA-256"):
        load_pinned_wasb_model(
            source_path=source,
            config_path=config,
            checkpoint_path=checkpoint,
            expected_source_sha256="0" * 64,
            expected_checkpoint_sha256=_sha256(checkpoint),
            device="cpu",
        )
