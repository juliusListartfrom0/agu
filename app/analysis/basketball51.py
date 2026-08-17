"""Bounded, game-grouped access to the public Basketball-51 ZIP archive."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import zlib
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASKETBALL51_LABELS = (
    "2p0",
    "2p1",
    "3p0",
    "3p1",
    "ft0",
    "ft1",
    "mp0",
    "mp1",
)

_CENTRAL_HEADER = struct.Struct("<4s6H3L5H2L")
_LOCAL_HEADER = struct.Struct("<4s5H3L2H")
_END_OF_CENTRAL_DIRECTORY = struct.Struct("<4s4H2LH")
_ZIP64_END_OF_CENTRAL_DIRECTORY = struct.Struct("<4sQ2H2L4Q")
_ZIP64_END_LOCATOR = struct.Struct("<4sLQL")
_SOURCE_GROUP = re.compile(r"_(v\d+)_", re.IGNORECASE)
_ZIP64_EXTRA_ID = 0x0001
_UINT32_MAX = 0xFFFFFFFF


@dataclass(frozen=True, order=True)
class Basketball51ZipEntry:
    name: str
    label: str
    source_group: str
    compression_method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int


def locate_zip_central_directory(
    tail: bytes,
    *,
    tail_start: int,
) -> tuple[int, int, int]:
    """Locate a standard or ZIP64 central directory from the archive tail."""

    if tail_start < 0:
        raise ValueError("tail_start must be non-negative")
    eocd_offset = tail.rfind(b"PK\x05\x06")
    if eocd_offset < 0 or len(tail) - eocd_offset < _END_OF_CENTRAL_DIRECTORY.size:
        raise ValueError("ZIP end-of-central-directory record is missing")
    values = _END_OF_CENTRAL_DIRECTORY.unpack_from(tail, eocd_offset)
    entry_count = int(values[4])
    central_size = int(values[5])
    central_offset = int(values[6])
    if (
        entry_count != 0xFFFF
        and central_size != _UINT32_MAX
        and central_offset != _UINT32_MAX
    ):
        return central_offset, central_size, entry_count

    locator_offset = tail.rfind(b"PK\x06\x07", 0, eocd_offset)
    if (
        locator_offset < 0
        or len(tail) - locator_offset < _ZIP64_END_LOCATOR.size
    ):
        raise ValueError("ZIP64 end locator is missing")
    locator = _ZIP64_END_LOCATOR.unpack_from(tail, locator_offset)
    zip64_record_offset = int(locator[2])
    local_record_offset = zip64_record_offset - tail_start
    if (
        local_record_offset < 0
        or len(tail) - local_record_offset
        < _ZIP64_END_OF_CENTRAL_DIRECTORY.size
    ):
        raise ValueError("ZIP64 end record is outside the provided archive tail")
    zip64 = _ZIP64_END_OF_CENTRAL_DIRECTORY.unpack_from(
        tail,
        local_record_offset,
    )
    if zip64[0] != b"PK\x06\x06":
        raise ValueError("invalid ZIP64 end-of-central-directory signature")
    return int(zip64[9]), int(zip64[8]), int(zip64[7])


def parse_zip_central_directory(payload: bytes) -> list[Basketball51ZipEntry]:
    """Parse standard or ZIP64 central entries without downloading the archive."""

    entries: list[Basketball51ZipEntry] = []
    offset = 0
    while offset < len(payload):
        if len(payload) - offset < _CENTRAL_HEADER.size:
            raise ValueError("truncated ZIP central directory")
        values = _CENTRAL_HEADER.unpack_from(payload, offset)
        if values[0] != b"PK\x01\x02":
            raise ValueError("invalid ZIP central-directory signature")
        compression_method = int(values[4])
        crc32 = int(values[7])
        compressed_size = int(values[8])
        uncompressed_size = int(values[9])
        name_length = int(values[10])
        extra_length = int(values[11])
        comment_length = int(values[12])
        local_header_offset = int(values[16])
        variable_start = offset + _CENTRAL_HEADER.size
        variable_end = (
            variable_start + name_length + extra_length + comment_length
        )
        if variable_end > len(payload):
            raise ValueError("truncated ZIP central-directory entry")
        name_bytes = payload[variable_start : variable_start + name_length]
        extra = payload[
            variable_start + name_length : variable_start + name_length + extra_length
        ]
        name = name_bytes.decode("utf-8")
        compressed_size, uncompressed_size, local_header_offset = _read_zip64_values(
            extra,
            compressed_size=compressed_size,
            uncompressed_size=uncompressed_size,
            local_header_offset=local_header_offset,
        )
        label, source_group = _basketball51_identity(name)
        entries.append(
            Basketball51ZipEntry(
                name=name,
                label=label,
                source_group=source_group,
                compression_method=compression_method,
                crc32=crc32,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                local_header_offset=local_header_offset,
            )
        )
        offset = variable_end
    return entries


def select_balanced_entries(
    entries: Iterable[Basketball51ZipEntry],
    *,
    per_class: int,
    seed: int = 0,
) -> list[Basketball51ZipEntry]:
    """Select each available label round-robin across original source games."""

    if per_class <= 0:
        raise ValueError("per_class must be positive")
    by_label: dict[str, dict[str, list[Basketball51ZipEntry]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for entry in entries:
        by_label[entry.label][entry.source_group].append(entry)
    if not by_label:
        raise ValueError("Basketball-51 selection requires entries")

    selected: list[Basketball51ZipEntry] = []
    for label in sorted(by_label):
        groups = by_label[label]
        ordered_groups = sorted(
            groups,
            key=lambda group: _stable_key(seed, label, group),
        )
        for group in ordered_groups:
            groups[group].sort(key=lambda entry: _stable_key(seed, entry.name))
        label_selection: list[Basketball51ZipEntry] = []
        cursor = 0
        while len(label_selection) < per_class:
            added = False
            for group in ordered_groups:
                if cursor < len(groups[group]):
                    label_selection.append(groups[group][cursor])
                    added = True
                    if len(label_selection) == per_class:
                        break
            if not added:
                raise ValueError(
                    f"Basketball-51 label {label} has fewer than {per_class} entries"
                )
            cursor += 1
        selected.extend(label_selection)
    return sorted(selected)


def read_zip_member(
    entry: Basketball51ZipEntry,
    *,
    read_range: Callable[[int, int], bytes],
) -> bytes:
    """Read and verify one member using two bounded byte-range requests."""

    header = read_range(
        entry.local_header_offset,
        entry.local_header_offset + _LOCAL_HEADER.size - 1,
    )
    if len(header) != _LOCAL_HEADER.size:
        raise ValueError("truncated ZIP local header")
    values = _LOCAL_HEADER.unpack(header)
    if values[0] != b"PK\x03\x04":
        raise ValueError("invalid ZIP local-header signature")
    compression_method = int(values[3])
    if compression_method != entry.compression_method:
        raise ValueError("ZIP compression method changed after index read")
    name_length = int(values[9])
    extra_length = int(values[10])
    payload_offset = (
        entry.local_header_offset + _LOCAL_HEADER.size + name_length + extra_length
    )
    compressed = read_range(
        payload_offset,
        payload_offset + entry.compressed_size - 1,
    )
    if len(compressed) != entry.compressed_size:
        raise ValueError("truncated ZIP member payload")
    if compression_method == 0:
        decoded = compressed
    elif compression_method == 8:
        decoded = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise ValueError(f"unsupported ZIP compression method: {compression_method}")
    if len(decoded) != entry.uncompressed_size:
        raise ValueError("ZIP member size mismatch")
    if zlib.crc32(decoded) & _UINT32_MAX != entry.crc32:
        raise ValueError("ZIP member CRC mismatch")
    return decoded


def materialize_basketball51_subset(
    entries: Iterable[Basketball51ZipEntry],
    *,
    output_dir: Path,
    read_range: Callable[[int, int], bytes],
    archive_size: int,
    archive_etag: str,
    per_class: int,
    seed: int,
) -> dict[str, Any]:
    """Download a deterministic balanced subset and seal its provenance."""

    if archive_size <= 0 or not archive_etag.strip():
        raise ValueError("Basketball-51 archive identity is incomplete")
    selected = select_balanced_entries(
        entries,
        per_class=per_class,
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    clips: list[dict[str, Any]] = []
    for entry in selected:
        relative_path = Path(entry.label) / Path(entry.name).name
        target = output_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if not _matches_entry(target, entry):
            decoded = read_zip_member(entry, read_range=read_range)
            temporary = target.with_suffix(f"{target.suffix}.part")
            temporary.write_bytes(decoded)
            temporary.replace(target)
        clips.append(
            {
                "archive_member": entry.name,
                "relative_path": relative_path.as_posix(),
                "label": entry.label,
                "shot_type": _shot_type(entry.label),
                "outcome": "made" if entry.label.endswith("1") else "missed",
                "source_group": entry.source_group,
                "size_bytes": target.stat().st_size,
                "crc32": f"{entry.crc32:08x}",
                "sha256": _file_sha256(target),
            }
        )

    payload: dict[str, Any] = {
        "schema_version": "agu.basketball51-subset.v1",
        "purpose": "model_training_only",
        "runtime_consumable": False,
        "dataset_id": "basketball-51",
        "source_url": (
            "https://www.kaggle.com/datasets/sarbagyashakya/"
            "basketball-51-dataset"
        ),
        "declared_license": "Apache-2.0-as-declared-by-Kaggle-uploader",
        "underlying_broadcast_rights": "not_granted_for_redistribution",
        "archive_size_bytes": archive_size,
        "archive_etag": archive_etag,
        "selection": {
            "strategy": "deterministic_label_balanced_source_group_round_robin_v1",
            "per_class": per_class,
            "seed": seed,
        },
        "source_group_count": len({entry.source_group for entry in selected}),
        "clip_count": len(clips),
        "clips": clips,
    }
    payload["artifact_sha256"] = _json_sha256(payload)
    return verify_basketball51_subset_manifest(payload)


def verify_basketball51_subset_manifest(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the training-only boundary and canonical manifest hash."""

    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    if claimed != _json_sha256(artifact):
        raise ValueError("Basketball-51 subset manifest hash mismatch")
    if (
        artifact.get("schema_version") != "agu.basketball51-subset.v1"
        or artifact.get("purpose") != "model_training_only"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("dataset_id") != "basketball-51"
    ):
        raise ValueError("invalid Basketball-51 training-only manifest")
    clips = artifact.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("Basketball-51 subset manifest requires clips")
    identities: set[tuple[str, str]] = set()
    for clip in clips:
        if not isinstance(clip, Mapping):
            raise ValueError("Basketball-51 clip metadata must be an object")
        label = str(clip.get("label") or "")
        group = str(clip.get("source_group") or "")
        relative_path = str(clip.get("relative_path") or "")
        if (
            label not in BASKETBALL51_LABELS
            or not group
            or not relative_path
            or str(clip.get("outcome") or "") not in {"made", "missed"}
            or len(str(clip.get("sha256") or "")) != 64
        ):
            raise ValueError("invalid Basketball-51 clip metadata")
        identity = (str(clip.get("archive_member") or ""), relative_path)
        if identity in identities:
            raise ValueError("duplicate Basketball-51 clip metadata")
        identities.add(identity)
    artifact["artifact_sha256"] = claimed
    return artifact


def seal_basketball51_embedding_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal Basketball-51 embeddings without making labels runtime evidence."""

    artifact = dict(payload)
    artifact["schema_version"] = "agu.basketball51-video-embeddings.v1"
    artifact["runtime_consumable"] = False
    artifact.pop("artifact_sha256", None)
    _validate_basketball51_embedding_artifact(artifact)
    artifact["artifact_sha256"] = _json_sha256(artifact)
    return artifact


def verify_basketball51_embedding_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_basketball51_embedding_artifact(artifact)
    if claimed != _json_sha256(artifact):
        raise ValueError("Basketball-51 embedding artifact hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _read_zip64_values(
    extra: bytes,
    *,
    compressed_size: int,
    uncompressed_size: int,
    local_header_offset: int,
) -> tuple[int, int, int]:
    needs_zip64 = (
        compressed_size == _UINT32_MAX
        or uncompressed_size == _UINT32_MAX
        or local_header_offset == _UINT32_MAX
    )
    cursor = 0
    zip64_payload: bytes | None = None
    while cursor < len(extra):
        if len(extra) - cursor < 4:
            raise ValueError("truncated ZIP extra field")
        field_id, field_size = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        field_end = cursor + field_size
        if field_end > len(extra):
            raise ValueError("truncated ZIP extra-field payload")
        if field_id == _ZIP64_EXTRA_ID:
            zip64_payload = extra[cursor:field_end]
        cursor = field_end
    if not needs_zip64:
        return compressed_size, uncompressed_size, local_header_offset
    if zip64_payload is None:
        raise ValueError("ZIP64 entry is missing its ZIP64 extra field")

    cursor = 0

    def read_value() -> int:
        nonlocal cursor
        if len(zip64_payload) - cursor < 8:
            raise ValueError("truncated ZIP64 extra field")
        value = struct.unpack_from("<Q", zip64_payload, cursor)[0]
        cursor += 8
        return int(value)

    if uncompressed_size == _UINT32_MAX:
        uncompressed_size = read_value()
    if compressed_size == _UINT32_MAX:
        compressed_size = read_value()
    if local_header_offset == _UINT32_MAX:
        local_header_offset = read_value()
    return compressed_size, uncompressed_size, local_header_offset


def _basketball51_identity(name: str) -> tuple[str, str]:
    parts = name.split("/")
    if len(parts) < 3 or parts[-1].endswith("/"):
        raise ValueError(f"unexpected Basketball-51 archive member: {name}")
    label = parts[-2]
    if label not in BASKETBALL51_LABELS:
        raise ValueError(f"unknown Basketball-51 label: {label}")
    match = _SOURCE_GROUP.search(parts[-1])
    if match is None:
        raise ValueError(f"Basketball-51 filename lacks source group: {name}")
    return label, match.group(1).lower()


def _stable_key(seed: int, *values: str) -> str:
    encoded = ":".join((str(seed), *values)).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_basketball51_embedding_artifact(
    artifact: Mapping[str, Any],
) -> None:
    if (
        artifact.get("schema_version")
        != "agu.basketball51-video-embeddings.v1"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("purpose") != "backbone_screening_training_only"
    ):
        raise ValueError("invalid Basketball-51 embedding artifact")
    dimension = int(artifact.get("embedding_dimension", 0))
    if (
        dimension <= 0
        or int(artifact.get("clip_frames", 0)) < 16
        or not artifact.get("subset_manifest_sha256")
        or not artifact.get("backbone")
        or not artifact.get("backbone_sha256")
    ):
        raise ValueError("Basketball-51 embedding provenance is incomplete")
    examples = artifact.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("Basketball-51 embedding artifact requires examples")
    paths: set[str] = set()
    for example in examples:
        if not isinstance(example, Mapping):
            raise ValueError("Basketball-51 embedding example must be an object")
        path = str(example.get("relative_path") or "")
        label = str(example.get("label") or "")
        embedding = example.get("embedding")
        if (
            not path
            or path in paths
            or label not in BASKETBALL51_LABELS
            or example.get("shot_type") != _shot_type(label)
            or example.get("outcome")
            != ("made" if label.endswith("1") else "missed")
            or not example.get("source_group")
            or not isinstance(embedding, list)
            or len(embedding) != dimension
            or not all(math.isfinite(float(value)) for value in embedding)
        ):
            raise ValueError("invalid Basketball-51 embedding example")
        paths.add(path)


def _matches_entry(path: Path, entry: Basketball51ZipEntry) -> bool:
    if not path.is_file() or path.stat().st_size != entry.uncompressed_size:
        return False
    crc = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            crc = zlib.crc32(chunk, crc)
    return crc & _UINT32_MAX == entry.crc32


def _shot_type(label: str) -> str:
    return {
        "2p": "two_point",
        "3p": "three_point",
        "ft": "free_throw",
        "mp": "mid_range",
    }[label[:2]]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Mapping[str, Any] | Sequence[Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
