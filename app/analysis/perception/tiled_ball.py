"""Small, deterministic helpers for screen-only tiled ball perception.

The helpers deliberately contain no model or video I/O.  They make the
overlapping-tile geometry used by the HOU--SAC research screen reproducible
without changing AGU's runtime detector defaults.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import hypot
from typing import Any


@dataclass(frozen=True, slots=True)
class TileSpec:
    """Horizontal crop bounds in full-frame pixel coordinates."""

    x1: int
    x2: int
    frame_width: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1


def horizontal_tile_specs(
    frame_width: int,
    *,
    overlap_ratio: float = 0.24,
) -> tuple[TileSpec, TileSpec]:
    """Return the two overlapping horizontal crops used by the screen route.

    For two tiles, an overlap ratio ``r`` means each crop covers
    ``(1 + r) / 2`` of the source width.  The second crop is right aligned,
    so the union is exactly the full frame even after integer rounding.
    """

    if int(frame_width) != frame_width or frame_width <= 0:
        raise ValueError("frame width must be a positive integer")
    if not 0.0 <= float(overlap_ratio) < 1.0:
        raise ValueError("overlap ratio must be in [0, 1)")

    width = int(round(frame_width * (1.0 + float(overlap_ratio)) / 2.0))
    width = min(frame_width, max(1, width))
    return (
        TileSpec(x1=0, x2=width, frame_width=frame_width),
        TileSpec(x1=frame_width - width, x2=frame_width, frame_width=frame_width),
    )


def project_tile_box(box: Sequence[float], tile: TileSpec) -> tuple[float, float, float, float]:
    """Project a tile-local ``xyxy`` box into full-frame coordinates.

    Horizontal coordinates are clipped to the tile before applying its
    offset.  Vertical coordinates are left untouched because a ``TileSpec``
    is horizontal-only and therefore has no frame-height contract.
    """

    if len(box) != 4:
        raise ValueError("box must contain four xyxy coordinates")
    x1, y1, x2, y2 = (float(value) for value in box)
    local_x1 = min(tile.width, max(0.0, x1))
    local_x2 = min(tile.width, max(0.0, x2))
    if local_x2 < local_x1:
        local_x1, local_x2 = local_x2, local_x1
    return (tile.x1 + local_x1, y1, tile.x1 + local_x2, y2)


def deduplicate_ball_detections(
    detections: Sequence[Mapping[str, Any]],
    *,
    center_distance_px: float = 18.0,
) -> list[dict[str, Any]]:
    """Collapse same-frame tile duplicates using center distance.

    The highest-confidence member of each greedy same-frame cluster wins.
    Inputs are copied, never mutated, and output order is deterministic.
    """

    if float(center_distance_px) < 0.0:
        raise ValueError("center distance must be non-negative")

    ranked = sorted(
        (dict(item) for item in detections),
        key=lambda item: (
            int(item["frame"]),
            -float(item["confidence"]),
            str(item.get("detection_id", "")),
        ),
    )
    kept: list[dict[str, Any]] = []
    for candidate in ranked:
        center = _box_center(candidate["bbox"])
        duplicate = any(
            int(previous["frame"]) == int(candidate["frame"])
            and hypot(center[0] - _box_center(previous["bbox"])[0], center[1] - _box_center(previous["bbox"])[1])
            <= float(center_distance_px)
            for previous in kept
        )
        if not duplicate:
            kept.append(candidate)
    return sorted(
        kept,
        key=lambda item: (int(item["frame"]), str(item.get("detection_id", ""))),
    )


def _box_center(box: Mapping[str, Any]) -> tuple[float, float]:
    return (
        (float(box["x1"]) + float(box["x2"])) / 2.0,
        (float(box["y1"]) + float(box["y2"])) / 2.0,
    )
