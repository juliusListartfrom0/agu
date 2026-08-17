from __future__ import annotations

import io
import zipfile

import pytest

from app.analysis.remote_zip_subset import (
    locate_remote_zip_directory,
    parse_remote_zip_entries,
    read_remote_zip_entry,
    subset_relative_path,
)


def _archive() -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(
            "root/basketball/event/cam/detections_info.txt",
            "sports ball,1,2,3,4\n",
            compress_type=zipfile.ZIP_DEFLATED,
        )
        archive.writestr(
            "root/basketball/event/cam/video.mp4",
            b"\x00\x01video",
            compress_type=zipfile.ZIP_STORED,
        )
        archive.writestr("../escape.txt", "unsafe")
    return stream.getvalue()


def test_remote_zip_directory_and_entries_round_trip() -> None:
    payload = _archive()
    tail_start = max(0, len(payload) - 65_536)
    directory = locate_remote_zip_directory(
        payload[tail_start:],
        tail_start=tail_start,
        archive_size=len(payload),
    )
    entries = parse_remote_zip_entries(
        payload[directory.offset : directory.offset + directory.size],
        expected_count=directory.entry_count,
    )

    assert len(entries) == 3
    annotation = next(
        row for row in entries if row.name.endswith("detections_info.txt")
    )
    extracted = read_remote_zip_entry(
        annotation,
        fetch_range=lambda start, end: payload[start : end + 1],
    )
    assert extracted == b"sports ball,1,2,3,4\n"


def test_remote_zip_subset_path_is_prefix_bound_and_traversal_safe() -> None:
    assert subset_relative_path(
        "root/basketball/event/cam/video.mp4",
        prefix="root/basketball/",
        allowed_names={"detections_info.txt", "video_info.txt"},
        allowed_suffixes={".mp4"},
    ).as_posix() == "event/cam/video.mp4"

    assert (
        subset_relative_path(
            "root/soccer/event/video.mp4",
            prefix="root/basketball/",
            allowed_names=set(),
            allowed_suffixes={".mp4"},
        )
        is None
    )
    assert (
        subset_relative_path(
            "root/basketball/",
            prefix="root/basketball/",
            allowed_names=set(),
            allowed_suffixes={".mp4"},
        )
        is None
    )
    with pytest.raises(ValueError, match="unsafe"):
        subset_relative_path(
            "root/basketball/../escape.mp4",
            prefix="root/basketball/",
            allowed_names=set(),
            allowed_suffixes={".mp4"},
        )
