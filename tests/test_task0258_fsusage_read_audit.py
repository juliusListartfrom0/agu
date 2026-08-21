"""Tests for bounded TASK-0258 fs_usage diagnostic capture."""

from __future__ import annotations

import pytest

from scripts.run_fsusage_read_audit import (
    _BoundedTextCapture,
    _drain_text_stream,
    _join_diagnostic_drains,
)


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


def test_bounded_text_capture_defers_malformed_input_failure_to_finish():
    capture = _BoundedTextCapture(maximum_bytes=8)
    capture.append(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="text"):
        capture.finish()


def test_drain_text_stream_records_iterator_failures_for_main_thread():
    class FailingStream:
        def __iter__(self):
            yield "first line\n"
            raise RuntimeError("decoder failed")

    capture = _BoundedTextCapture(maximum_bytes=32)
    errors = []
    _drain_text_stream(FailingStream(), capture, errors)

    assert capture.finish() == "first line\n"
    assert len(errors) == 1
    assert str(errors[0]) == "decoder failed"


def test_join_diagnostic_drains_rejects_a_stream_that_did_not_finish():
    class HangingThread:
        def join(self, timeout):
            assert timeout == 0.25

        def is_alive(self):
            return True

    with pytest.raises(RuntimeError, match="did not finish"):
        _join_diagnostic_drains([HangingThread()], timeout_seconds=0.25)


@pytest.mark.parametrize("maximum_bytes", [0, -1, True, 1.0])
def test_bounded_text_capture_rejects_invalid_caps(maximum_bytes):
    with pytest.raises(ValueError):
        _BoundedTextCapture(maximum_bytes=maximum_bytes)
