from __future__ import annotations

import io
import json
import struct
import zipfile
from pathlib import Path

import pytest

from app.analysis.basketball51 import (
    Basketball51ZipEntry,
    locate_zip_central_directory,
    materialize_basketball51_subset,
    parse_zip_central_directory,
    read_zip_member,
    seal_basketball51_embedding_artifact,
    select_balanced_entries,
    verify_basketball51_embedding_artifact,
    verify_basketball51_subset_manifest,
)


def _zip64_central_entry(
    name: str,
    *,
    compressed_size: int,
    uncompressed_size: int,
    local_header_offset: int,
) -> bytes:
    encoded = name.encode()
    extra = struct.pack(
        "<HHQQQ",
        0x0001,
        24,
        uncompressed_size,
        compressed_size,
        local_header_offset,
    )
    return (
        struct.pack(
            "<4s6H3L5H2L",
            b"PK\x01\x02",
            45,
            45,
            0,
            zipfile.ZIP_DEFLATED,
            0,
            0,
            0x12345678,
            0xFFFFFFFF,
            0xFFFFFFFF,
            len(encoded),
            len(extra),
            0,
            0,
            0,
            0,
            0xFFFFFFFF,
        )
        + encoded
        + extra
    )


def test_parse_zip64_central_directory_recovers_sizes_groups_and_labels() -> None:
    name = "Basketball_51 dataset/ft1/ft1_v108_000437_x264.mp4"

    entries = parse_zip_central_directory(
        _zip64_central_entry(
            name,
            compressed_size=1234,
            uncompressed_size=1250,
            local_header_offset=9876,
        )
    )

    assert entries == [
        Basketball51ZipEntry(
            name=name,
            label="ft1",
            source_group="v108",
            compression_method=zipfile.ZIP_DEFLATED,
            crc32=0x12345678,
            compressed_size=1234,
            uncompressed_size=1250,
            local_header_offset=9876,
        )
    ]


def test_balanced_selection_is_deterministic_and_spreads_source_games() -> None:
    entries = [
        Basketball51ZipEntry(
            name=f"Basketball_51 dataset/{label}/{label}_{group}_{index:06d}_x264.mp4",
            label=label,
            source_group=group,
            compression_method=zipfile.ZIP_STORED,
            crc32=index,
            compressed_size=10,
            uncompressed_size=10,
            local_header_offset=index * 10,
        )
        for label in ("2p0", "2p1", "ft0", "ft1")
        for group in ("v001", "v002", "v003")
        for index in range(4)
    ]

    first = select_balanced_entries(entries, per_class=5, seed=7)
    second = select_balanced_entries(reversed(entries), per_class=5, seed=7)

    assert first == second
    assert {label: sum(item.label == label for item in first) for label in ("2p0", "2p1", "ft0", "ft1")} == {
        "2p0": 5,
        "2p1": 5,
        "ft0": 5,
        "ft1": 5,
    }
    assert all(
        len({item.source_group for item in first if item.label == label}) == 3
        for label in ("2p0", "2p1", "ft0", "ft1")
    )


def test_read_zip_member_fetches_only_local_header_and_payload() -> None:
    archive = io.BytesIO()
    expected = b"basketball-clip" * 50
    name = "Basketball_51 dataset/ft0/ft0_v003_000001_x264.mp4"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr(name, expected)
    payload = archive.getvalue()
    eocd = payload.rfind(b"PK\x05\x06")
    central_size, central_offset = struct.unpack_from("<LL", payload, eocd + 12)
    entry = parse_zip_central_directory(
        payload[central_offset : central_offset + central_size]
    )[0]
    ranges: list[tuple[int, int]] = []

    def read_range(start: int, end: int) -> bytes:
        ranges.append((start, end))
        return payload[start : end + 1]

    actual = read_zip_member(entry, read_range=read_range)

    assert actual == expected
    assert len(ranges) == 2
    assert sum(end - start + 1 for start, end in ranges) < len(payload)


def test_read_zip_member_rejects_crc_mismatch() -> None:
    entry = Basketball51ZipEntry(
        name="Basketball_51 dataset/ft0/ft0_v003_000001_x264.mp4",
        label="ft0",
        source_group="v003",
        compression_method=zipfile.ZIP_STORED,
        crc32=0,
        compressed_size=4,
        uncompressed_size=4,
        local_header_offset=0,
    )
    local_header = struct.pack(
        "<4s5H3L2H",
        b"PK\x03\x04",
        20,
        0,
        zipfile.ZIP_STORED,
        0,
        0,
        0,
        4,
        4,
        0,
        0,
    )

    with pytest.raises(ValueError, match="CRC"):
        read_zip_member(
            entry,
            read_range=lambda start, end: (local_header + b"data")[start : end + 1],
        )


def test_locate_standard_zip_central_directory_from_tail() -> None:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(
            "Basketball_51 dataset/2p1/2p1_v001_000001_x264.mp4",
            b"made",
        )
    payload = archive.getvalue()
    tail_start = max(0, len(payload) - 128)
    central_offset, central_size, entry_count = locate_zip_central_directory(
        payload[tail_start:],
        tail_start=tail_start,
    )

    assert payload[central_offset : central_offset + 4] == b"PK\x01\x02"
    assert central_size > 0
    assert entry_count == 1


def test_materialize_subset_seals_training_only_files(
    tmp_path: Path,
) -> None:
    archive = io.BytesIO()
    contents = {
        "Basketball_51 dataset/2p1/2p1_v001_000001_x264.mp4": b"made-two",
        "Basketball_51 dataset/ft0/ft0_v002_000001_x264.mp4": b"missed-free-throw",
    }
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as handle:
        for name, value in contents.items():
            handle.writestr(name, value)
    payload = archive.getvalue()
    eocd = payload.rfind(b"PK\x05\x06")
    central_size, central_offset = struct.unpack_from("<LL", payload, eocd + 12)
    entries = parse_zip_central_directory(
        payload[central_offset : central_offset + central_size]
    )

    manifest = materialize_basketball51_subset(
        entries,
        output_dir=tmp_path,
        read_range=lambda start, end: payload[start : end + 1],
        archive_size=len(payload),
        archive_etag="fixture-etag",
        per_class=1,
        seed=11,
    )
    verified = verify_basketball51_subset_manifest(
        json.loads(json.dumps(manifest))
    )

    assert verified["runtime_consumable"] is False
    assert verified["purpose"] == "model_training_only"
    assert verified["source_group_count"] == 2
    assert {row["label"] for row in verified["clips"]} == {"2p1", "ft0"}
    assert {
        row["outcome"] for row in verified["clips"]
    } == {"made", "missed"}
    assert all((tmp_path / row["relative_path"]).is_file() for row in verified["clips"])

    verified["clips"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_basketball51_subset_manifest(verified)


def test_basketball51_embedding_artifact_is_training_only_and_hash_bound() -> None:
    artifact = seal_basketball51_embedding_artifact(
        {
            "purpose": "backbone_screening_training_only",
            "subset_manifest_sha256": "subset",
            "backbone": "fixture/backbone",
            "backbone_sha256": "backbone",
            "embedding_dimension": 4,
            "clip_frames": 16,
            "examples": [
                {
                    "relative_path": "2p1/a.mp4",
                    "source_group": "v001",
                    "label": "2p1",
                    "shot_type": "two_point",
                    "outcome": "made",
                    "embedding": [1.0, 2.0, 3.0, 4.0],
                },
                {
                    "relative_path": "ft0/b.mp4",
                    "source_group": "v002",
                    "label": "ft0",
                    "shot_type": "free_throw",
                    "outcome": "missed",
                    "embedding": [-1.0, -2.0, -3.0, -4.0],
                },
            ],
        }
    )

    assert verify_basketball51_embedding_artifact(artifact) == artifact
    assert artifact["runtime_consumable"] is False

    artifact["examples"][0]["embedding"][0] = 99
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_basketball51_embedding_artifact(artifact)
