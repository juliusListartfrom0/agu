from __future__ import annotations

import struct

import pytest

import scripts.extract_uvy_basketball_subset as extractor


def _central_entry(name: str, *, offset: int = 100, compressed: int = 4, uncompressed: int = 8) -> bytes:
    encoded = name.encode("utf-8")
    header = struct.pack(
        "<4s6H3L5H2L",
        b"PK\x01\x02",
        20,
        20,
        0,
        8,
        0,
        0,
        0x12345678,
        compressed,
        uncompressed,
        len(encoded),
        0,
        0,
        0,
        0,
        0,
        offset,
    )
    return header + encoded


def test_uvy_central_parser_and_selection_are_bounded() -> None:
    entries = extractor.parse_central_directory(
        _central_entry("UVY/basketball_V01/video_info.txt")
        + _central_entry("UVY/basketball_V01/img1/000001.jpg", offset=200)
        + _central_entry("UVY/football_V01/img1/000001.jpg", offset=300)
    )
    assert len(entries) == 3
    assert extractor._selected(entries[0]) == ("basketball_V01", "annotation")
    assert extractor._selected(entries[1]) == ("basketball_V01", "image")
    assert extractor._selected(entries[2]) is None


def test_uvy_range_groups_split_large_excluded_payload_gap() -> None:
    first = extractor.ZipEntry("a", 10, 10, 8, 0, 100, 1, 0)
    second = extractor.ZipEntry("b", 10, 10, 8, 0, 2_000_000, 1, 0)
    groups = extractor._range_groups([first, second], max_gap=100)
    assert len(groups) == 2


def test_uvy_large_range_splits_between_entries() -> None:
    first = extractor.ZipEntry("a", 10, 10, 8, 0, 100, 1, 0)
    second = extractor.ZipEntry("b", 10, 10, 8, 0, 200, 1, 0)
    groups = extractor._split_range_groups([(100, 1_000, [first, second])], max_bytes=50)
    assert len(groups) == 2
    assert groups[0][2] == [first]
    assert groups[1][2] == [second]


def test_uvy_safe_relative_rejects_traversal() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        extractor._safe_relative("UVY/../secret.jpg")
