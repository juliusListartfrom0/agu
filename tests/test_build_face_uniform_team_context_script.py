from __future__ import annotations

import numpy as np

from scripts.build_face_uniform_team_context import _extract_sample_features


def test_uniform_context_skips_face_when_no_torso_pixels_are_available() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    assert _extract_sample_features(frame, [40, 95, 20, 20]) is None
    assert _extract_sample_features(frame, [40, 10, 20, 20]) is not None
