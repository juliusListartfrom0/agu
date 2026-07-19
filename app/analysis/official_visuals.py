"""Visual evidence layouts for AGU official-event VLM stages."""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np


def build_temporal_contact_sheet(
    frames: Sequence[np.ndarray],
    *,
    duration_sec: float | None = None,
    columns: int = 4,
    cell_width: int = 320,
) -> np.ndarray | None:
    """Lay chronological frames out left-to-right with visible time labels."""

    if not frames or columns <= 0 or cell_width <= 0:
        return None
    valid = [frame for frame in frames if isinstance(frame, np.ndarray) and frame.size]
    if not valid:
        return None
    first_height, first_width = valid[0].shape[:2]
    cell_height = max(1, int(round(first_height * cell_width / max(1, first_width))))
    rows = int(math.ceil(len(valid) / columns))
    sheet = np.zeros((rows * cell_height, columns * cell_width, 3), dtype=np.uint8)
    for index, frame in enumerate(valid):
        resized = cv2.resize(frame, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        x1, y1 = column * cell_width, row * cell_height
        sheet[y1 : y1 + cell_height, x1 : x1 + cell_width] = resized
        if duration_sec is not None and len(valid) > 1:
            time_sec = duration_sec * index / (len(valid) - 1)
            label = f"F{index + 1:02d} t+{time_sec:.1f}s"
        else:
            label = f"F{index + 1:02d}"
        cv2.rectangle(sheet, (x1, y1), (x1 + min(cell_width, 155), y1 + 25), (0, 0, 0), -1)
        cv2.putText(
            sheet,
            label,
            (x1 + 5, y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return sheet
