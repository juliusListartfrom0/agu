from __future__ import annotations

import numpy as np
import pytest

from scripts.run_replay_logo_recurrence import decode_timeline


def test_decode_timeline_rejects_invalid_fps() -> None:
    with pytest.raises(ValueError, match="positive"):
        decode_timeline("unused.mp4", scan_fps=0.0)  # type: ignore[arg-type]


def test_logo_timeline_contract_is_rgb_float() -> None:
    sample = np.zeros((2, 27, 48, 3), dtype=np.float32)

    assert sample.shape[1:] == (27, 48, 3)
    assert sample.dtype == np.float32
