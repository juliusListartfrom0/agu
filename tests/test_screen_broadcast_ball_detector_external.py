from __future__ import annotations

import pytest

from scripts.screen_broadcast_ball_detector_external import _verify_frame_pixels_sha


def test_frame_sha256_without_pixel_hash_is_not_treated_as_pixel_hash() -> None:
    candidate = {"candidate_id": "c1", "frame_sha256": "detector-record-hash"}

    _verify_frame_pixels_sha(candidate, "decoded-pixel-hash")


def test_explicit_pixel_hash_is_verified() -> None:
    candidate = {"candidate_id": "c1", "frame_pixels_sha256": "expected-pixel-hash"}

    with pytest.raises(ValueError, match="frame pixel hash mismatch for c1"):
        _verify_frame_pixels_sha(candidate, "different-pixel-hash")
