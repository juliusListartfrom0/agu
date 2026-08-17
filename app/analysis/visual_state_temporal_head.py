"""Lightweight temporal formation model for reviewed broadcast contact sheets."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from app.analysis.ebqwen_visual_state_lora import (
    verify_lora_dataset_manifest,
)

VISUAL_STATE_FRAME_COUNT = 5
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class VisualStateRecord:
    review_id: str
    source_video_sha256: str
    event_id: str
    image_path: Path
    target: int


def load_visual_state_records(
    manifest: Mapping[str, object],
    *,
    root: Path,
) -> list[VisualStateRecord]:
    verified = verify_lora_dataset_manifest(manifest)
    root = root.resolve()
    records = []
    for row in verified["examples"]:
        image_path = (root / str(row["image"])).resolve()
        if (
            not image_path.is_relative_to(root)
            or not image_path.is_file()
            or _file_sha256(image_path) != row["image_sha256"]
        ):
            raise ValueError("visual-state training image hash mismatch")
        records.append(
            VisualStateRecord(
                review_id=str(row["review_id"]),
                source_video_sha256=str(row["source_video_sha256"]),
                event_id=str(row["event_id"]),
                image_path=image_path,
                target=1 if row["label"] == "FREE_THROW" else 0,
            )
        )
    return records


class VisualStateTemporalClassifier(nn.Module):
    """Encode five frames with shared weights and classify temporal formation."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        embedding_dimension: int,
    ) -> None:
        super().__init__()
        if embedding_dimension < 1:
            raise ValueError("embedding dimension must be positive")
        self.encoder = encoder
        self.embedding_dimension = embedding_dimension
        self.head = nn.Linear(embedding_dimension * 3, 2)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if (
            frames.ndim != 5
            or frames.shape[1] != VISUAL_STATE_FRAME_COUNT
            or frames.shape[2] != 3
        ):
            raise ValueError("visual-state input requires five RGB frames")
        batch_size = frames.shape[0]
        embeddings = self.encoder(
            frames.reshape(-1, *frames.shape[2:])
        ).reshape(batch_size, VISUAL_STATE_FRAME_COUNT, self.embedding_dimension)
        pooled = torch.cat(
            (
                embeddings.mean(dim=1),
                embeddings.std(dim=1, unbiased=False),
                embeddings[:, -1] - embeddings[:, 0],
            ),
            dim=1,
        )
        return self.head(pooled)


def split_contact_sheet_panels(
    image: np.ndarray,
    *,
    header_height: int = 32,
) -> np.ndarray:
    """Split five chronological panels and remove their rendered headers."""

    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.shape[1] % VISUAL_STATE_FRAME_COUNT
        or header_height < 0
        or header_height >= image.shape[0]
    ):
        raise ValueError("invalid visual-state contact sheet")
    panels = np.stack(
        np.split(image, VISUAL_STATE_FRAME_COUNT, axis=1),
        axis=0,
    )
    return panels[:, header_height:].copy()


def prepare_contact_sheet_tensor(
    image: np.ndarray,
    *,
    header_height: int = 32,
    output_size: int = 224,
    training: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Create five normalized, aspect-preserving tensors with shared jitter."""

    if output_size < 32:
        raise ValueError("visual-state output size is too small")
    panels = split_contact_sheet_panels(
        image,
        header_height=header_height,
    )
    rgb = panels[..., ::-1].copy()
    tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(torch.float32) / 255.0
    generator = generator or torch.default_generator
    if training:
        if bool(torch.rand((), generator=generator) < 0.5):
            tensor = torch.flip(tensor, dims=(3,))
        brightness = float(
            torch.empty((), dtype=torch.float32).uniform_(
                0.9,
                1.1,
                generator=generator,
            )
        )
        contrast = float(
            torch.empty((), dtype=torch.float32).uniform_(
                0.9,
                1.1,
                generator=generator,
            )
        )
        channel_mean = tensor.mean(dim=(0, 2, 3), keepdim=True)
        tensor = ((tensor - channel_mean) * contrast + channel_mean) * brightness
        tensor = tensor.clamp(0.0, 1.0)
    source_height, source_width = tensor.shape[-2:]
    scale = min(output_size / source_height, output_size / source_width)
    resized_height = max(1, round(source_height * scale))
    resized_width = max(1, round(source_width * scale))
    resized = F.interpolate(
        tensor,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    canvas = mean.expand(
        VISUAL_STATE_FRAME_COUNT,
        3,
        output_size,
        output_size,
    ).clone()
    top = (output_size - resized_height) // 2
    left = (output_size - resized_width) // 2
    canvas[
        :,
        :,
        top : top + resized_height,
        left : left + resized_width,
    ] = resized
    std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
    return (canvas - mean) / std


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
