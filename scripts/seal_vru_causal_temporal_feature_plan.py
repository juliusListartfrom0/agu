#!/usr/bin/env python3
"""Seal the approved TASK-0258 label-hidden temporal feature plan."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.analysis.vru_causal_temporal_retrospective import (
    FileReceipt,
    StoredArtifactReceipt,
    Task0257ExpectedReceipts,
    Task0257InputPaths,
    observe_temporal_environment_contract,
    seal_vru_causal_temporal_feature_plan,
    verify_disk_write_budget,
    verify_module_a_spec_approval,
    verify_task0257_temporal_inputs,
)

_MAX_CONTRACT_BYTES = 8_388_608
_CONTRACT_FIELDS = {"paths", "expected_receipts"}
_FILE_RECEIPT_FIELDS = {"file_sha256", "filename", "size_bytes"}
_STORED_RECEIPT_FIELDS = {
    "schema_version",
    "internal_sha256_field",
    "internal_sha256",
    "file_sha256",
    "filename",
    "size_bytes",
}


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _read_contract(path: Path, *, expected_file_sha256: str) -> dict[str, Any]:
    if not isinstance(expected_file_sha256, str) or len(expected_file_sha256) != 64:
        raise ValueError("input-contract expected file SHA-256 is invalid")
    raw_path = Path(path)
    if raw_path.is_symlink():
        raise ValueError("input-contract path must not be a symlink")
    resolved = raw_path.resolve(strict=True)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(resolved, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_CONTRACT_BYTES:
            raise ValueError("input contract must be a bounded regular file")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1_048_576))
            if not chunk:
                raise ValueError("input contract changed during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        final = os.fstat(descriptor)
        if len(encoded) != metadata.st_size or (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        ) != (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns):
            raise ValueError("input contract changed during read")
    finally:
        os.close(descriptor)
    if hashlib.sha256(encoded).hexdigest() != expected_file_sha256:
        raise ValueError("input-contract file SHA-256 does not match")
    try:
        payload = json.loads(
            encoded,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("input contract JSON is invalid") from error
    if not isinstance(payload, dict) or set(payload) != _CONTRACT_FIELDS:
        raise ValueError("input contract fields are invalid")
    return payload


def _path_identity(path: Path) -> tuple[int, int] | None:
    if not os.path.lexists(path):
        return None
    metadata = path.stat()
    return metadata.st_dev, metadata.st_ino


def _reject_output_alias(*, output_path: Path, input_paths: Sequence[Path]) -> None:
    output = Path(output_path).absolute()
    output_resolved = output.resolve(strict=False)
    output_identity = _path_identity(output)

    def normalized_parts(path: Path) -> tuple[str, ...]:
        return tuple(os.path.normcase(part).casefold() for part in path.parts)

    def overlaps(left: Path, right: Path) -> bool:
        left_parts = normalized_parts(left)
        right_parts = normalized_parts(right)
        shorter = min(len(left_parts), len(right_parts))
        return left_parts[:shorter] == right_parts[:shorter]

    for raw_input in input_paths:
        input_path = Path(raw_input).absolute()
        input_resolved = input_path.resolve(strict=False)
        input_identity = _path_identity(input_path)
        if (
            overlaps(output, input_path)
            or overlaps(output_resolved, input_resolved)
            or (output_identity is not None and input_identity == output_identity)
        ):
            raise ValueError("output aliases an input")
    if os.path.lexists(output):
        raise ValueError("output already exists")


def _preflight_explicit_paths(
    *,
    approval_path: Path,
    contract_path: Path,
    spec_paths: Sequence[Path],
    output_path: Path,
) -> None:
    _reject_output_alias(
        output_path=output_path,
        input_paths=(approval_path, contract_path, *spec_paths),
    )


def _preflight_contract_paths(payload: Mapping[str, object], *, output_path: Path) -> None:
    raw_paths = payload.get("paths")
    if not isinstance(raw_paths, Mapping) or set(raw_paths) != set(Task0257InputPaths.__dataclass_fields__):
        raise ValueError("input-contract path fields are invalid")
    leaf_paths: list[Path] = []
    for value in raw_paths.values():
        if isinstance(value, str):
            leaf_paths.append(Path(value))
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            leaf_paths.extend(Path(item) for item in value)
        else:
            raise ValueError("input-contract path value is invalid")
    _reject_output_alias(output_path=output_path, input_paths=leaf_paths)


@contextmanager
def _exclusive_lock(lock_path: Path):
    lock = Path(lock_path)
    if not lock.is_dir() or lock.is_symlink():
        raise ValueError("lock path must be an existing non-symlink directory")
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock, flags)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("module-exclusive lock already exists") from error
        yield descriptor
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _receipt_from_mapping(value: object, *, stored: bool) -> StoredArtifactReceipt | FileReceipt:
    expected_fields = _STORED_RECEIPT_FIELDS if stored else _FILE_RECEIPT_FIELDS
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("input receipt fields are invalid")
    return StoredArtifactReceipt(**value) if stored else FileReceipt(**value)


def _parse_input_contract(payload: Mapping[str, object]) -> tuple[Task0257InputPaths, Task0257ExpectedReceipts]:
    raw_paths = payload.get("paths")
    raw_receipts = payload.get("expected_receipts")
    expected_fields = set(Task0257InputPaths.__dataclass_fields__)
    if not isinstance(raw_paths, Mapping) or set(raw_paths) != expected_fields:
        raise ValueError("input-contract path fields are invalid")
    if not isinstance(raw_receipts, Mapping) or set(raw_receipts) != set(Task0257ExpectedReceipts.__dataclass_fields__):
        raise ValueError("input-contract receipt fields are invalid")
    sequence_fields = {
        "parent_review_jpegs",
        "parent_candidate_children",
        "parent_label_children",
        "old_embedding_files",
        "harwood_review_jpegs",
        "source_videos",
        "checkpoints",
    }
    path_values: dict[str, object] = {}
    for field in Task0257InputPaths.__dataclass_fields__:
        value = raw_paths[field]
        if field in sequence_fields:
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                raise ValueError("input-contract path sequence is invalid")
            path_values[field] = tuple(Path(item) for item in value)
        else:
            if not isinstance(value, str):
                raise ValueError("input-contract path is invalid")
            path_values[field] = Path(value)
    stored_sequences = {"parent_candidate_children", "old_embedding_files"}
    file_sequences = {
        "parent_review_jpegs",
        "parent_label_children",
        "harwood_review_jpegs",
        "source_videos",
        "checkpoints",
    }
    file_scalars = {"parent_source_manifest", "v2_label_child", "harwood_source_manifest"}
    receipt_values: dict[str, object] = {}
    for field in Task0257ExpectedReceipts.__dataclass_fields__:
        value = raw_receipts[field]
        if field in stored_sequences | file_sequences:
            if not isinstance(value, list):
                raise ValueError("input-contract receipt sequence is invalid")
            receipt_values[field] = tuple(_receipt_from_mapping(row, stored=field in stored_sequences) for row in value)
        else:
            receipt_values[field] = _receipt_from_mapping(value, stored=field not in file_scalars)
    return Task0257InputPaths(**path_values), Task0257ExpectedReceipts(**receipt_values)


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    if os.path.lexists(path):
        raise ValueError("output already exists")
    path.parent.mkdir(parents=False, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--expected-approval-artifact-sha256", required=True)
    parser.add_argument("--expected-approval-file-sha256", required=True)
    parser.add_argument("--requirement", type=Path, required=True)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--gate-review", type=Path, required=True)
    parser.add_argument("--expected-fresh-review-internal-sha256", required=True)
    parser.add_argument("--expected-fresh-review-file-sha256", required=True)
    parser.add_argument("--expected-approval-statement-sha256", required=True)
    parser.add_argument("--input-contract", type=Path, required=True)
    parser.add_argument("--expected-input-contract-file-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _verify_and_build_plan(args: argparse.Namespace) -> dict[str, object]:
    spec_paths = (args.requirement, args.solution, args.gate_review)
    verify_module_a_spec_approval(
        approval_path=args.approval,
        expected_artifact_sha256=args.expected_approval_artifact_sha256,
        expected_file_sha256=args.expected_approval_file_sha256,
        approved_spec_paths=spec_paths,
        expected_fresh_review_internal_sha256=args.expected_fresh_review_internal_sha256,
        expected_fresh_review_file_sha256=args.expected_fresh_review_file_sha256,
        expected_approval_statement_sha256=args.expected_approval_statement_sha256,
    )
    contract = _read_contract(
        args.input_contract,
        expected_file_sha256=args.expected_input_contract_file_sha256,
    )
    _preflight_contract_paths(contract, output_path=args.output)
    paths, receipts = _parse_input_contract(contract)
    environment = observe_temporal_environment_contract()
    verified_inputs = verify_task0257_temporal_inputs(paths=paths, expected_receipts=receipts)
    plan = seal_vru_causal_temporal_feature_plan(
        inputs=verified_inputs,
        environment_contract=environment,
    )
    verify_disk_write_budget(
        output_paths=(args.output,),
        worst_case_new_bytes=67_108_864,
    )
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec_paths = (args.requirement, args.solution, args.gate_review)
    _preflight_explicit_paths(
        approval_path=args.approval,
        contract_path=args.input_contract,
        spec_paths=spec_paths,
        output_path=args.output,
    )
    if args.output.name != "temporal_feature_plan.json":
        raise ValueError("plan output filename is invalid")
    first_plan = _verify_and_build_plan(args)
    lock_path = args.output.parent
    with _exclusive_lock(lock_path):
        _preflight_explicit_paths(
            approval_path=args.approval,
            contract_path=args.input_contract,
            spec_paths=spec_paths,
            output_path=args.output,
        )
        plan = _verify_and_build_plan(args)
        if plan != first_plan:
            raise ValueError("plan changed across the under-lock revalidation")
        _write_json_atomic(args.output, plan)
    print(json.dumps({"artifact_sha256": plan["artifact_sha256"], "row_count": 45}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
