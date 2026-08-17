"""Bounded HTTP-range extraction for selected files in a remote ZIP archive.

ZIP record layouts follow the official PKWARE APPNOTE:
https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT
"""

from __future__ import annotations

import binascii
import struct
import zlib
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import PurePosixPath

_EOCD_SIGNATURE = b"PK\x05\x06"
_ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
_ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
_CENTRAL_SIGNATURE = b"PK\x01\x02"
_LOCAL_SIGNATURE = b"PK\x03\x04"


@dataclass(frozen=True)
class RemoteZipDirectory:
    offset: int
    size: int
    entry_count: int


@dataclass(frozen=True)
class RemoteZipEntry:
    name: str
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int
    compression_method: int
    crc32: int
    flags: int


def locate_remote_zip_directory(
    tail: bytes,
    *,
    tail_start: int,
    archive_size: int,
) -> RemoteZipDirectory:
    """Locate the central directory from a bounded archive tail."""

    eocd_index = tail.rfind(_EOCD_SIGNATURE)
    if eocd_index < 0 or eocd_index + 22 > len(tail):
        raise ValueError("remote ZIP end record is missing")
    (
        _,
        disk_number,
        directory_disk,
        disk_entries,
        total_entries,
        directory_size,
        directory_offset,
        comment_length,
    ) = struct.unpack_from("<4s4H2LH", tail, eocd_index)
    if eocd_index + 22 + comment_length != len(tail):
        raise ValueError("remote ZIP end record is not at archive end")
    if disk_number or directory_disk or disk_entries != total_entries:
        raise ValueError("multi-disk remote ZIP archives are unsupported")
    if (
        total_entries != 0xFFFF
        and directory_size != 0xFFFFFFFF
        and directory_offset != 0xFFFFFFFF
    ):
        directory = RemoteZipDirectory(
            offset=directory_offset,
            size=directory_size,
            entry_count=total_entries,
        )
        _validate_directory(directory, archive_size=archive_size)
        return directory

    locator_index = tail.rfind(
        _ZIP64_LOCATOR_SIGNATURE,
        0,
        eocd_index,
    )
    if locator_index < 0 or locator_index + 20 > eocd_index:
        raise ValueError("remote ZIP64 locator is missing")
    _, locator_disk, zip64_offset, disk_count = struct.unpack_from(
        "<4sLQL",
        tail,
        locator_index,
    )
    if locator_disk != 0 or disk_count != 1:
        raise ValueError("multi-disk remote ZIP64 archives are unsupported")
    relative_zip64_offset = zip64_offset - tail_start
    if (
        relative_zip64_offset < 0
        or relative_zip64_offset + 56 > len(tail)
        or tail[relative_zip64_offset : relative_zip64_offset + 4]
        != _ZIP64_EOCD_SIGNATURE
    ):
        raise ValueError("remote ZIP64 end record is outside fetched tail")
    (
        _,
        record_size,
        _,
        _,
        zip64_disk,
        zip64_directory_disk,
        zip64_disk_entries,
        zip64_total_entries,
        zip64_directory_size,
        zip64_directory_offset,
    ) = struct.unpack_from(
        "<4sQ2H2L4Q",
        tail,
        relative_zip64_offset,
    )
    if (
        record_size < 44
        or zip64_disk
        or zip64_directory_disk
        or zip64_disk_entries != zip64_total_entries
    ):
        raise ValueError("remote ZIP64 directory metadata is invalid")
    directory = RemoteZipDirectory(
        offset=zip64_directory_offset,
        size=zip64_directory_size,
        entry_count=zip64_total_entries,
    )
    _validate_directory(directory, archive_size=archive_size)
    return directory


def parse_remote_zip_entries(
    payload: bytes,
    *,
    expected_count: int,
) -> list[RemoteZipEntry]:
    """Parse one complete central-directory payload."""

    position = 0
    entries = []
    while position < len(payload):
        if payload[position : position + 4] != _CENTRAL_SIGNATURE:
            raise ValueError("remote ZIP central directory is malformed")
        if position + 46 > len(payload):
            raise ValueError("remote ZIP central entry is truncated")
        (
            _,
            _,
            _,
            flags,
            method,
            _,
            _,
            crc32,
            compressed_size,
            uncompressed_size,
            name_length,
            extra_length,
            comment_length,
            disk_start,
            _,
            _,
            local_offset,
        ) = struct.unpack_from("<4s6H3L5H2L", payload, position)
        end = position + 46 + name_length + extra_length + comment_length
        if end > len(payload):
            raise ValueError("remote ZIP central entry data is truncated")
        name_bytes = payload[position + 46 : position + 46 + name_length]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        name = name_bytes.decode(encoding)
        extra_start = position + 46 + name_length
        extra = payload[extra_start : extra_start + extra_length]
        (
            uncompressed_size,
            compressed_size,
            local_offset,
            disk_start,
        ) = _apply_zip64_extra(
            extra,
            uncompressed_size=uncompressed_size,
            compressed_size=compressed_size,
            local_offset=local_offset,
            disk_start=disk_start,
        )
        if flags & 0x1 or disk_start != 0:
            raise ValueError("encrypted or multi-disk ZIP entries are unsupported")
        entries.append(
            RemoteZipEntry(
                name=name,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                local_header_offset=local_offset,
                compression_method=method,
                crc32=crc32,
                flags=flags,
            )
        )
        position = end
    if len(entries) != expected_count:
        raise ValueError("remote ZIP central entry count mismatch")
    return entries


def read_remote_zip_entry(
    entry: RemoteZipEntry,
    *,
    fetch_range: Callable[[int, int], bytes],
) -> bytes:
    """Fetch, decompress and CRC-check one central-directory entry."""

    header = fetch_range(
        entry.local_header_offset,
        entry.local_header_offset + 29,
    )
    if len(header) != 30 or header[:4] != _LOCAL_SIGNATURE:
        raise ValueError("remote ZIP local header is invalid")
    (
        _,
        _,
        local_flags,
        local_method,
        _,
        _,
        _,
        _,
        _,
        name_length,
        extra_length,
    ) = struct.unpack("<4s5H3L2H", header)
    if local_flags & 0x1 or local_method != entry.compression_method:
        raise ValueError("remote ZIP local header conflicts with directory")
    data_start = (
        entry.local_header_offset + 30 + name_length + extra_length
    )
    compressed = (
        b""
        if entry.compressed_size == 0
        else fetch_range(
            data_start,
            data_start + entry.compressed_size - 1,
        )
    )
    if len(compressed) != entry.compressed_size:
        raise ValueError("remote ZIP entry download is truncated")
    if entry.compression_method == 0:
        output = compressed
    elif entry.compression_method == 8:
        output = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise ValueError(
            f"unsupported remote ZIP compression method: "
            f"{entry.compression_method}"
        )
    if (
        len(output) != entry.uncompressed_size
        or binascii.crc32(output) & 0xFFFFFFFF != entry.crc32
    ):
        raise ValueError("remote ZIP entry size or CRC mismatch")
    return output


def subset_relative_path(
    name: str,
    *,
    prefix: str,
    allowed_names: Collection[str],
    allowed_suffixes: Collection[str],
) -> PurePosixPath | None:
    """Return a traversal-safe selected path relative to an exact prefix."""

    if not name.startswith(prefix):
        return None
    remainder = name.removeprefix(prefix)
    if not remainder:
        return None
    relative = PurePosixPath(remainder)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("unsafe remote ZIP subset path")
    if (
        relative.name not in allowed_names
        and relative.suffix.lower() not in allowed_suffixes
    ):
        return None
    return relative


def _apply_zip64_extra(
    extra: bytes,
    *,
    uncompressed_size: int,
    compressed_size: int,
    local_offset: int,
    disk_start: int,
) -> tuple[int, int, int, int]:
    fields = {
        "uncompressed_size": uncompressed_size,
        "compressed_size": compressed_size,
        "local_offset": local_offset,
        "disk_start": disk_start,
    }
    position = 0
    zip64 = None
    while position + 4 <= len(extra):
        header_id, size = struct.unpack_from("<HH", extra, position)
        end = position + 4 + size
        if end > len(extra):
            raise ValueError("remote ZIP extra field is truncated")
        if header_id == 0x0001:
            zip64 = extra[position + 4 : end]
            break
        position = end
    zip64_position = 0
    for name, sentinel, size in (
        ("uncompressed_size", 0xFFFFFFFF, 8),
        ("compressed_size", 0xFFFFFFFF, 8),
        ("local_offset", 0xFFFFFFFF, 8),
        ("disk_start", 0xFFFF, 4),
    ):
        if fields[name] != sentinel:
            continue
        if zip64 is None or zip64_position + size > len(zip64):
            raise ValueError("remote ZIP64 entry metadata is missing")
        fields[name] = int.from_bytes(
            zip64[zip64_position : zip64_position + size],
            "little",
        )
        zip64_position += size
    return (
        fields["uncompressed_size"],
        fields["compressed_size"],
        fields["local_offset"],
        fields["disk_start"],
    )


def _validate_directory(
    directory: RemoteZipDirectory,
    *,
    archive_size: int,
) -> None:
    if (
        directory.offset < 0
        or directory.size < 0
        or directory.entry_count < 0
        or directory.offset + directory.size > archive_size
    ):
        raise ValueError("remote ZIP central directory bounds are invalid")
