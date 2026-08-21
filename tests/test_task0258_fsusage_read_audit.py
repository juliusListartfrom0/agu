"""Tests for bounded TASK-0258 fs_usage diagnostic capture."""

from __future__ import annotations

import pytest

from scripts.run_fsusage_read_audit import _BoundedTextCapture


def test_bounded_text_capture_returns_text_within_byte_cap():
    capture = _BoundedTextCapture(maximum_bytes=8)
    capture.append("1234\n")
    capture.append("56\n")
    assert capture.finish() == "1234\n56\n"


def test_bounded_text_capture_drains_after_cap_without_retaining_more_data():
    capture = _BoundedTextCapture(maximum_bytes=4)
    capture.append("1234")
    capture.append("this line is over the cap")
    capture.append("and this later line is discarded")
    with pytest.raises(ValueError, match="byte cap"):
        capture.finish()


@pytest.mark.parametrize("maximum_bytes", [0, -1, True, 1.0])
def test_bounded_text_capture_rejects_invalid_caps(maximum_bytes):
    with pytest.raises(ValueError):
        _BoundedTextCapture(maximum_bytes=maximum_bytes)
