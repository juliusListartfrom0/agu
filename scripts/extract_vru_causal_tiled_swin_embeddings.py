#!/usr/bin/env python3
"""Run the guarded TASK-0258 tiled-Swin extraction and terminal transaction."""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import fcntl
import json
import math
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from app.analysis.vru_causal_temporal_retrospective import (
    TEMPORAL_ENVIRONMENT_CONTRACT,
    TEMPORAL_RESOURCE_POLICY,
    DiskWriteBudgetError,
    _build_fresh_tiled_swin_embeddings,
    _build_vru_causal_temporal_final_generation,
    _canonical_sha256,
    _extract_tiled_swin_rows,
    _read_bounded_bytes,
    _read_bounded_json,
    _sha256_bytes,
    build_resource_sample,
    encode_resource_sample_line,
    load_verified_temporal_feature_plan,
    load_verified_tiled_swin_embeddings,
    observe_temporal_environment_contract,
    seal_tiled_swin_attempt,
    seal_tiled_swin_resume,
    verify_disk_write_budget,
    verify_module_a_final_receipt_registry,
    verify_module_a_spec_approval,
    verify_resource_log_bytes,
    verify_task0257_temporal_inputs,
    verify_tiled_swin_attempt,
    verify_tiled_swin_resume,
    verify_vru_causal_temporal_final_generation,
)
from scripts import seal_vru_causal_temporal_feature_plan as plan_cli

RESOURCE_LIMIT_EXIT_CODE = 75
_WORKER_REQUEST_FIELDS = {
    "plan_path",
    "plan_artifact_sha256",
    "plan_file_sha256",
    "source_video_paths",
    "checkpoint_path",
    "progress_path",
    "started_prefix_count",
}


class _GuardedWorkerFailure(RuntimeError):
    def __init__(self, stop_reason: str, message: str) -> None:
        super().__init__(message)
        self.stop_reason = stop_reason
        self.summary: dict[str, object] = {}


class _GuardedWorkerInterrupted(RuntimeError):
    def __init__(self, signal_name: str) -> None:
        super().__init__(f"worker interrupted by {signal_name}")
        self.signal_name = signal_name
        self.summary: dict[str, object] = {}


def _classify_finalization_failure(error: Exception) -> str:
    if isinstance(error, DiskWriteBudgetError):
        return "disk_failure"
    if isinstance(error, OSError):
        return "publication_failure"
    return "determinism_failure"


def _batch_one(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("batch size must be exactly 1") from error
    if parsed != 1:
        raise argparse.ArgumentTypeError("batch size must be exactly 1")
    return parsed


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
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-artifact-sha256", required=True)
    parser.add_argument("--expected-plan-file-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt-registry", type=Path)
    parser.add_argument("--expected-receipt-registry-artifact-sha256")
    parser.add_argument("--expected-receipt-registry-file-sha256")
    parser.add_argument("--expected-prior-attempt-receipt", action="append", default=[])
    parser.add_argument("--expected-resume-cas")
    parser.add_argument("--batch-size", type=_batch_one, default=1)
    return parser


@dataclass(frozen=True)
class _PriorAttemptState:
    attempt_chain: tuple[dict[str, object], ...]
    attempt_records: tuple[dict[str, object], ...]
    latest_resume: dict[str, object] | None
    latest_resume_cas: dict[str, object] | None
    prior_consecutive_breach_count: int


def _parse_json_argument(value: str, *, label: str) -> dict[str, object]:
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=lambda pairs: _pairs_without_duplicates(pairs, label=label),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-standard JSON constant {constant}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} JSON is invalid") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _pairs_without_duplicates(
    pairs: list[tuple[str, object]],
    *,
    label: str,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"{label} contains duplicate key {key}")
        result[key] = value
    return result


@contextmanager
def _exclusive_directory_lock(directory_path: Path):
    directory = Path(directory_path)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("module lock directory must be an existing non-symlink directory")
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(directory, flags)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Module-A extractor is already running") from error
        yield descriptor
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _atomic_replace_private(path: Path, payload: Mapping[str, object]) -> None:
    encoded = _canonical_json_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _worker_main(request_path: Path, *, expected_file_sha256: str) -> int:
    request, encoded = _read_bounded_json(request_path, max_bytes=1_048_576)
    if _sha256_bytes(encoded) != expected_file_sha256:
        raise ValueError("worker request file SHA-256 does not match")
    if set(request) != _WORKER_REQUEST_FIELDS:
        raise ValueError("worker request fields are invalid")
    plan = load_verified_temporal_feature_plan(
        plan_path=Path(request["plan_path"]),
        expected_artifact_sha256=request["plan_artifact_sha256"],
        expected_file_sha256=request["plan_file_sha256"],
    )
    progress_path = Path(request["progress_path"])
    request_parent = Path(request_path).resolve(strict=True).parent
    if progress_path.name != "private-progress.json" or progress_path.parent.resolve(strict=True) != request_parent:
        raise ValueError("worker progress must remain in the request private directory")
    started_prefix_count = request.get("started_prefix_count")
    if type(started_prefix_count) is not int or not 0 <= started_prefix_count < 45:
        raise ValueError("worker started prefix count is invalid")
    if started_prefix_count:
        completed = copy.deepcopy(
            _read_private_progress(
                progress_path,
                expected_completed_count=started_prefix_count,
            )["tile_embeddings"]
        )
    else:
        if progress_path.exists() or progress_path.is_symlink():
            raise ValueError("attempt-one progress path must initially be absent")
        completed: list[list[list[float]]] = []

    def checkpoint(ordinal: int, row: list[list[float]]) -> None:
        if ordinal != len(completed):
            raise ValueError("worker progress must advance in exact plan order")
        completed.append(row)
        _atomic_replace_private(
            progress_path,
            {
                "schema_version": "agu.vru-causal-tiled-swin-private-progress.v1",
                "completed_count": len(completed),
                "tile_embeddings": completed,
            },
        )

    _extract_tiled_swin_rows(
        plan=plan,
        source_video_paths=tuple(Path(path) for path in request["source_video_paths"]),
        checkpoint_path=Path(request["checkpoint_path"]),
        row_callback=checkpoint,
        initial_rows=completed,
    )
    return 0


def _read_private_progress(path: Path, *, expected_completed_count: int) -> dict[str, object]:
    payload, _encoded = _read_bounded_json(path, max_bytes=67_108_864)
    if set(payload) != {"schema_version", "completed_count", "tile_embeddings"}:
        raise ValueError("private progress fields are invalid")
    if (
        payload.get("schema_version") != "agu.vru-causal-tiled-swin-private-progress.v1"
        or payload.get("completed_count") != expected_completed_count
        or type(payload.get("completed_count")) is not int
    ):
        raise ValueError("private progress count is invalid")
    rows = payload.get("tile_embeddings")
    if not isinstance(rows, list) or len(rows) != expected_completed_count:
        raise ValueError("private progress rows are invalid")
    for row in rows:
        if (
            not isinstance(row, list)
            or len(row) != 4
            or any(
                not isinstance(tile, list)
                or len(tile) != 768
                or any(type(value) is not float or not math.isfinite(value) for value in tile)
                for tile in row
            )
        ):
            raise ValueError("private progress embedding values are invalid")
    return payload


def _write_file_fsync(path: Path, encoded: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        remaining = memoryview(encoded)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("exclusive file write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_receipt(path: Path) -> dict[str, object]:
    raw_path = Path(path)
    if raw_path.is_symlink():
        raise ValueError("receipt path must not be a symlink")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(raw_path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 67_108_864:
            raise ValueError("receipt file must be a bounded regular file")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1_048_576))
            if not chunk:
                raise ValueError("receipt file changed during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        final = os.fstat(descriptor)
        if len(encoded) != metadata.st_size or (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        ) != (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
        ):
            raise ValueError("receipt file changed during read")
    finally:
        os.close(descriptor)
    return {"file_sha256": _sha256_bytes(encoded), "filename": path.name, "size_bytes": len(encoded)}


def _stored_receipt(path: Path, *, internal_field: str = "artifact_sha256") -> dict[str, object]:
    payload, encoded = _read_bounded_json(path, max_bytes=67_108_864)
    return {
        "schema_version": payload["schema_version"],
        "internal_sha256_field": internal_field,
        "internal_sha256": payload[internal_field],
        "file_sha256": _sha256_bytes(encoded),
        "filename": path.name,
        "size_bytes": len(encoded),
    }


def _resume_cas(path: Path, payload: Mapping[str, object], encoded: bytes) -> dict[str, object]:
    metadata = path.stat(follow_symlinks=False)
    internal_sha256 = payload.get("artifact_sha256")
    if not isinstance(internal_sha256, str):
        raise ValueError("resume internal SHA-256 is invalid")
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "size_bytes": metadata.st_size,
        "internal_sha256": internal_sha256,
        "file_sha256": _sha256_bytes(encoded),
    }


def _read_resource_log(
    path: Path,
    *,
    expected_attempt_ordinal: int,
    prior_cumulative_samples: int,
    prior_consecutive_breach_count: int,
) -> tuple[dict[str, object], int, int]:
    encoded = _read_bounded_bytes(
        path,
        max_bytes=TEMPORAL_RESOURCE_POLICY["max_resource_log_bytes"],
    )
    if encoded and not encoded.endswith(b"\n"):
        raise ValueError("resource log has a truncated final row")
    row_count, consecutive = verify_resource_log_bytes(
        encoded,
        expected_attempt_ordinal=expected_attempt_ordinal,
        prior_cumulative_samples=prior_cumulative_samples,
        prior_consecutive_breach_count=prior_consecutive_breach_count,
    )
    return _file_receipt(path), row_count, consecutive


def _verify_expected_stored_receipt(value: object, *, ordinal: int) -> dict[str, object]:
    expected_fields = {
        "schema_version",
        "internal_sha256_field",
        "internal_sha256",
        "file_sha256",
        "filename",
        "size_bytes",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("external prior-attempt receipt fields are invalid")
    normalized = dict(value)
    if (
        normalized["schema_version"] != "agu.vru-causal-tiled-swin-attempt.v1"
        or normalized["internal_sha256_field"] != "artifact_sha256"
        or normalized["filename"] != "attempt_record.json"
        or any(
            not isinstance(normalized[field], str)
            or len(normalized[field]) != 64
            or any(character not in "0123456789abcdef" for character in normalized[field])
            for field in ("internal_sha256", "file_sha256")
        )
        or type(normalized["size_bytes"]) is not int
        or normalized["size_bytes"] <= 0
        or ordinal not in (1, 2)
    ):
        raise ValueError("external prior-attempt receipt is invalid")
    return normalized


def _load_prior_attempt_state(
    *,
    output_root: Path,
    plan: Any,
    raw_expected_attempt_receipts: Sequence[str],
    raw_expected_resume_cas: str | None,
) -> _PriorAttemptState:
    expected_attempt_receipts = tuple(
        _verify_expected_stored_receipt(
            _parse_json_argument(value, label="prior-attempt receipt"),
            ordinal=ordinal,
        )
        for ordinal, value in enumerate(raw_expected_attempt_receipts, start=1)
    )
    if len(expected_attempt_receipts) > 2:
        raise ValueError("at most two recoverable prior attempts are allowed")
    expected_resume_cas = (
        _parse_json_argument(raw_expected_resume_cas, label="resume CAS")
        if raw_expected_resume_cas is not None
        else None
    )
    attempts_root = output_root / "attempts"
    if not expected_attempt_receipts:
        if expected_resume_cas is not None:
            raise ValueError("attempt one cannot receive a resume CAS")
        if attempts_root.exists() or attempts_root.is_symlink():
            raise ValueError("unexpected prior-attempt directory without external receipts")
        return _PriorAttemptState((), (), None, None, 0)
    if expected_resume_cas is None:
        raise ValueError("resumed extraction requires an external resume CAS")
    if attempts_root.is_symlink() or not attempts_root.is_dir():
        raise ValueError("prior-attempt root is invalid")
    expected_directory_names = {f"attempt-{ordinal:04d}" for ordinal in range(1, len(expected_attempt_receipts) + 1)}
    if {path.name for path in attempts_root.iterdir()} != expected_directory_names:
        raise ValueError("prior-attempt directory coverage is invalid")
    attempt_chain: list[dict[str, object]] = []
    attempt_records: list[dict[str, object]] = []
    latest_resume: dict[str, object] | None = None
    latest_cas: dict[str, object] | None = None
    prior_cumulative_samples = 0
    prior_cumulative_log_bytes = 0
    prior_cumulative_runtime = 0
    consecutive_breach_count = 0
    for ordinal, expected_receipt in enumerate(expected_attempt_receipts, start=1):
        attempt_directory = attempts_root / f"attempt-{ordinal:04d}"
        if attempt_directory.is_symlink() or not attempt_directory.is_dir():
            raise ValueError("prior-attempt directory is invalid")
        if {path.name for path in attempt_directory.iterdir()} != {
            "attempt_record.json",
            "resource_guard.jsonl",
            "resume.json",
        }:
            raise ValueError("prior-attempt generation fields are invalid")
        attempt_path = attempt_directory / "attempt_record.json"
        attempt_payload, attempt_bytes = _read_bounded_json(attempt_path, max_bytes=67_108_864)
        actual_attempt_receipt = _stored_receipt(attempt_path)
        if (
            actual_attempt_receipt != expected_receipt
            or _sha256_bytes(attempt_bytes) != expected_receipt["file_sha256"]
        ):
            raise ValueError("prior-attempt external receipt does not match")
        verify_tiled_swin_attempt(attempt_payload)
        if (
            attempt_payload["attempt_ordinal"] != ordinal
            or attempt_payload["disposition"] != "interrupted_recoverable"
            or attempt_payload["plan_receipt"]
            != {"artifact_sha256": plan._artifact_sha256, "file_sha256": plan._file_sha256}
            or attempt_payload["task0257_input_receipts"] != plan._payload["task0257_receipts"]
            or attempt_payload["prior_attempt_receipt"]
            != (expected_attempt_receipts[ordinal - 2] if ordinal > 1 else None)
            or attempt_payload["resume_input_cas"] != (latest_cas if ordinal > 1 else None)
        ):
            raise ValueError("prior-attempt chain binding is invalid")
        resource_path = attempt_directory / "resource_guard.jsonl"
        actual_resource_receipt, attempt_samples, consecutive_breach_count = _read_resource_log(
            resource_path,
            expected_attempt_ordinal=ordinal,
            prior_cumulative_samples=prior_cumulative_samples,
            prior_consecutive_breach_count=consecutive_breach_count,
        )
        if actual_resource_receipt != attempt_payload["resource_log_receipt"]:
            raise ValueError("prior-attempt resource-log receipt does not match")
        if (
            attempt_payload["cumulative_resource_samples"] != prior_cumulative_samples + attempt_samples
            or attempt_payload["cumulative_resource_log_bytes"]
            != prior_cumulative_log_bytes + actual_resource_receipt["size_bytes"]
            or attempt_payload["cumulative_active_runtime_nanoseconds"] < prior_cumulative_runtime
        ):
            raise ValueError("prior-attempt cumulative counters are invalid")
        resume_path = attempt_directory / "resume.json"
        resume_metadata = resume_path.stat(follow_symlinks=False)
        resume_payload, resume_bytes = _read_bounded_json(resume_path, max_bytes=67_108_864)
        if resume_path.is_symlink() or resume_path.stat(follow_symlinks=False) != resume_metadata:
            raise ValueError("prior resume identity changed during read")
        verify_tiled_swin_resume(resume_payload, plan=plan)
        actual_cas = _resume_cas(resume_path, resume_payload, resume_bytes)
        if (
            actual_cas != attempt_payload["resume_output_cas"]
            or resume_payload["attempt_chain_receipts"] != attempt_chain
        ):
            raise ValueError("prior resume CAS or attempt chain does not match")
        chain_entry = {
            "attempt_ordinal": ordinal,
            "attempt_record": actual_attempt_receipt,
            "resource_log": actual_resource_receipt,
            "resume_output_cas": actual_cas,
        }
        attempt_chain.append(chain_entry)
        attempt_records.append(attempt_payload)
        latest_resume = resume_payload
        latest_cas = actual_cas
        prior_cumulative_samples = attempt_payload["cumulative_resource_samples"]
        prior_cumulative_log_bytes = attempt_payload["cumulative_resource_log_bytes"]
        prior_cumulative_runtime = attempt_payload["cumulative_active_runtime_nanoseconds"]
    if latest_cas != expected_resume_cas:
        raise ValueError("latest resume CAS does not match the external value")
    return _PriorAttemptState(
        tuple(attempt_chain),
        tuple(attempt_records),
        latest_resume,
        latest_cas,
        consecutive_breach_count,
    )


def _publish_final_directory(stage: Path, final: Path) -> None:
    if final.exists() or final.is_symlink():
        raise ValueError("terminal final generation already exists")
    directory = os.open(stage, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    _rename_exclusive(stage, final)
    parent = os.open(final.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_file_fsync(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError("private publication source must be a regular file")
    _write_file_fsync(destination, source.read_bytes())


def _publish_recoverable_attempt(
    *,
    work_stage: Path,
    output_root: Path,
    plan: Any,
    prior_state: _PriorAttemptState,
    worker_summary: Mapping[str, object],
    completed_rows: Sequence[Sequence[Sequence[float]]],
    signal_name: str,
) -> tuple[dict[str, object], dict[str, object]]:
    attempt_ordinal = len(prior_state.attempt_chain) + 1
    if attempt_ordinal not in (1, 2) or signal_name not in ("SIGINT", "SIGTERM"):
        raise ValueError("recoverable attempt signal or ordinal is invalid")
    started_count = prior_state.latest_resume["completed_count"] if prior_state.latest_resume is not None else 0
    completed_count = len(completed_rows)
    if not started_count < completed_count < 45:
        raise ValueError("recoverable attempt did not produce a strict prefix extension")
    attempts_root = output_root / "attempts"
    if attempts_root.exists():
        if attempts_root.is_symlink() or not attempts_root.is_dir():
            raise ValueError("attempt publication root is invalid")
    else:
        attempts_root.mkdir(mode=0o700)
        _fsync_directory(output_root)
    target = attempts_root / f"attempt-{attempt_ordinal:04d}"
    if target.exists() or target.is_symlink():
        raise ValueError("recoverable attempt target already exists")
    stage = Path(tempfile.mkdtemp(prefix=f".attempt-{attempt_ordinal:04d}.", dir=attempts_root))
    try:
        resource_path = stage / "resource_guard.jsonl"
        _copy_file_fsync(work_stage / "resource_guard.jsonl", resource_path)
        resource_receipt = _file_receipt(resource_path)
        resume = seal_tiled_swin_resume(
            {
                "plan_receipt": {
                    "artifact_sha256": plan._artifact_sha256,
                    "file_sha256": plan._file_sha256,
                },
                "task0257_input_receipts": copy.deepcopy(plan._payload["task0257_receipts"]),
                "representation": copy.deepcopy(plan._payload["representation"]),
                "producer_environment": copy.deepcopy(plan._payload["environment_contract"]),
                "checkpoint_receipt": copy.deepcopy(plan._payload["task0257_receipts"]["checkpoints"][1]),
                "source_video_receipts": copy.deepcopy(plan._payload["task0257_receipts"]["source_videos"]),
                "attempt_chain_receipts": copy.deepcopy(list(prior_state.attempt_chain)),
                "row_count": 45,
                "completed_count": completed_count,
                "completed_examples": copy.deepcopy(list(completed_rows)),
            },
            plan=plan,
        )
        resume_path = stage / "resume.json"
        resume_bytes = _canonical_json_bytes(resume)
        _write_file_fsync(resume_path, resume_bytes)
        resume_output_cas = _resume_cas(resume_path, resume, resume_bytes)
        attempt = seal_tiled_swin_attempt(
            {
                "attempt_ordinal": attempt_ordinal,
                "prior_attempt_receipt": (
                    copy.deepcopy(prior_state.attempt_chain[-1]["attempt_record"])
                    if prior_state.attempt_chain
                    else None
                ),
                "plan_receipt": {
                    "artifact_sha256": plan._artifact_sha256,
                    "file_sha256": plan._file_sha256,
                },
                "task0257_input_receipts": copy.deepcopy(plan._payload["task0257_receipts"]),
                "started_prefix_count": started_count,
                "completed_prefix_count": completed_count,
                "new_rows_verified": completed_count - started_count,
                "cumulative_active_runtime_nanoseconds": worker_summary["active_runtime_nanoseconds"],
                "cumulative_resource_samples": worker_summary["sample_count"],
                "cumulative_resource_log_bytes": worker_summary["log_bytes"],
                "resource_log_receipt": resource_receipt,
                "resume_input_cas": copy.deepcopy(prior_state.latest_resume_cas),
                "resume_output_cas": resume_output_cas,
                "received_signal": signal_name,
                "disposition": "interrupted_recoverable",
                "stop_reason": {
                    "SIGINT": "external_sigint",
                    "SIGTERM": "external_sigterm",
                }[signal_name],
            }
        )
        attempt_path = stage / "attempt_record.json"
        _write_file_fsync(attempt_path, _canonical_json_bytes(attempt))
        if {path.name for path in stage.iterdir()} != {
            "resource_guard.jsonl",
            "resume.json",
            "attempt_record.json",
        }:
            raise ValueError("recoverable attempt staging coverage is invalid")
        _fsync_directory(stage)
        _publish_final_directory(stage, target)
        published_resume, published_bytes = _read_bounded_json(
            target / "resume.json",
            max_bytes=67_108_864,
        )
        if _resume_cas(target / "resume.json", published_resume, published_bytes) != resume_output_cas:
            raise ValueError("published resume CAS changed")
        return _stored_receipt(target / "attempt_record.json"), resume_output_cas
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _rename_exclusive(source: Path, destination: Path) -> None:
    """Atomically publish without any overwrite-capable rename fallback."""
    if sys.platform != "darwin":
        raise RuntimeError("exclusive directory publication requires macOS renamex_np")
    libc = ctypes.CDLL(None, use_errno=True)
    renamex_np = libc.renamex_np
    renamex_np.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
    renamex_np.restype = ctypes.c_int
    rename_excl = 0x00000004
    result = renamex_np(
        os.fsencode(source),
        os.fsencode(destination),
        rename_excl,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, "exclusive publication target exists", destination)
        raise OSError(error_number, os.strerror(error_number), destination)


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    else:
        process.wait()


def _run_guarded_worker(
    *,
    command: Sequence[str],
    log_path: Path,
    attempt_ordinal: int = 1,
    prior_active_runtime_nanoseconds: int = 0,
    prior_resource_samples: int = 0,
    prior_resource_log_bytes: int = 0,
    prior_consecutive_breach_count: int = 0,
) -> dict[str, object]:
    if (
        type(attempt_ordinal) is not int
        or attempt_ordinal not in (1, 2, 3)
        or any(
            type(value) is not int or value < 0
            for value in (
                prior_active_runtime_nanoseconds,
                prior_resource_samples,
                prior_resource_log_bytes,
                prior_consecutive_breach_count,
            )
        )
    ):
        raise ValueError("guarded worker prior counters are invalid")
    if log_path.is_symlink() or not log_path.is_file():
        failure = _GuardedWorkerFailure(
            "publication_failure",
            "resource log must be an existing non-symlink regular file",
        )
        failure.summary = {
            "started_nanoseconds": time.monotonic_ns(),
            "active_runtime_nanoseconds": prior_active_runtime_nanoseconds,
            "sample_count": prior_resource_samples,
            "log_bytes": prior_resource_log_bytes,
            "attempt_active_runtime_nanoseconds": 0,
            "attempt_sample_count": 0,
            "attempt_log_bytes": 0,
            "process_tree_reaped": True,
        }
        raise failure
    started = time.monotonic_ns()
    sample_ordinal = 0
    attempt_log_bytes = 0
    consecutive = prior_consecutive_breach_count
    psutil.cpu_percent(interval=None)
    process: subprocess.Popen[bytes] | None = None
    received_signal: str | None = None
    pending_error: _GuardedWorkerFailure | _GuardedWorkerInterrupted | None = None
    handled_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)
    original_handlers: dict[signal.Signals, Any] = {}

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal received_signal
        received_signal = signal.Signals(signum).name
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    for handled_signal in handled_signals:
        original_handlers[handled_signal] = signal.getsignal(handled_signal)
        signal.signal(handled_signal, handle_signal)
    try:
        log_flags = os.O_WRONLY | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            log_flags |= os.O_NOFOLLOW
        log_descriptor = os.open(log_path, log_flags)
        with os.fdopen(log_descriptor, "wb", buffering=0) as log:
            process = subprocess.Popen(list(command), start_new_session=True)
            while True:
                cumulative_sample_ordinal = prior_resource_samples + sample_ordinal + 1
                next_scheduled_nanoseconds = (
                    TEMPORAL_RESOURCE_POLICY["sample_interval_seconds"] * cumulative_sample_ordinal * 1_000_000_000
                )
                remaining_nanoseconds = next_scheduled_nanoseconds - (
                    prior_active_runtime_nanoseconds + (time.monotonic_ns() - started)
                )
                if remaining_nanoseconds <= 0:
                    wait_seconds = 0.001
                else:
                    wait_seconds = remaining_nanoseconds / 1_000_000_000
                try:
                    return_code = process.wait(timeout=wait_seconds)
                    break
                except subprocess.TimeoutExpired:
                    pass
                sample_ordinal += 1
                cumulative_sample_ordinal = prior_resource_samples + sample_ordinal
                if cumulative_sample_ordinal > TEMPORAL_RESOURCE_POLICY["max_resource_samples"]:
                    raise _GuardedWorkerFailure("sample_cap", "resource sample cap reached")
                elapsed = time.monotonic_ns() - started
                cumulative_elapsed = prior_active_runtime_nanoseconds + elapsed
                if cumulative_elapsed > TEMPORAL_RESOURCE_POLICY["max_cumulative_active_seconds"] * 1_000_000_000:
                    raise _GuardedWorkerFailure("runtime_cap", "active runtime cap reached")
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()
                sample = build_resource_sample(
                    attempt_ordinal=attempt_ordinal,
                    attempt_sample_ordinal=sample_ordinal,
                    cumulative_sample_ordinal=cumulative_sample_ordinal,
                    observed_attempt_active_nanoseconds=elapsed,
                    system_memory_percent=float(memory.percent),
                    available_memory_bytes=int(memory.available),
                    free_swap_bytes=int(swap.free),
                    system_cpu_percent=float(psutil.cpu_percent(interval=None)),
                    prior_consecutive_breach_count=consecutive,
                )
                line = encode_resource_sample_line(sample)
                if (
                    prior_resource_log_bytes + attempt_log_bytes + len(line)
                    > TEMPORAL_RESOURCE_POLICY["max_resource_log_bytes"]
                ):
                    raise _GuardedWorkerFailure("log_byte_cap", "resource log byte cap reached")
                remaining_line = memoryview(line)
                while remaining_line:
                    written = log.write(remaining_line)
                    if written is None or written <= 0:
                        raise OSError("resource log write made no progress")
                    remaining_line = remaining_line[written:]
                os.fsync(log.fileno())
                attempt_log_bytes += len(line)
                consecutive = sample["consecutive_breach_count"]
                if consecutive >= TEMPORAL_RESOURCE_POLICY["consecutive_breach_limit"]:
                    raise _GuardedWorkerFailure("resource_breach", "sustained resource threshold breach")
            if received_signal is not None:
                if received_signal in ("SIGINT", "SIGTERM"):
                    raise _GuardedWorkerInterrupted(received_signal)
                failure = _GuardedWorkerFailure("unsupported_signal", f"unsupported signal {received_signal}")
                failure.signal_name = received_signal
                raise failure
            if return_code != 0:
                raise _GuardedWorkerFailure("model_failure", f"worker failed with exit code {return_code}")
            if (
                prior_active_runtime_nanoseconds + (time.monotonic_ns() - started)
                > TEMPORAL_RESOURCE_POLICY["max_cumulative_active_seconds"] * 1_000_000_000
            ):
                raise _GuardedWorkerFailure("runtime_cap", "active runtime cap reached")
    except (_GuardedWorkerFailure, _GuardedWorkerInterrupted) as error:
        pending_error = error
    except Exception as error:
        pending_error = _GuardedWorkerFailure(
            "publication_failure",
            f"guarded worker supervision failed: {type(error).__name__}",
        )
    finally:
        if process is not None:
            _terminate_process_group(process)
        for handled_signal, original_handler in original_handlers.items():
            signal.signal(handled_signal, original_handler)
    summary = {
        "started_nanoseconds": started,
        "active_runtime_nanoseconds": prior_active_runtime_nanoseconds + (time.monotonic_ns() - started),
        "sample_count": prior_resource_samples + sample_ordinal,
        "log_bytes": prior_resource_log_bytes + attempt_log_bytes,
        "attempt_active_runtime_nanoseconds": time.monotonic_ns() - started,
        "attempt_sample_count": sample_ordinal,
        "attempt_log_bytes": attempt_log_bytes,
        "process_tree_reaped": process is not None and process.poll() is not None,
    }
    if pending_error is not None:
        pending_error.summary = summary
        raise pending_error
    return summary


def _verify_preflight(args: argparse.Namespace):
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
    contract = plan_cli._read_contract(
        args.input_contract,
        expected_file_sha256=args.expected_input_contract_file_sha256,
    )
    paths, receipts = plan_cli._parse_input_contract(contract)
    environment = observe_temporal_environment_contract()
    if environment != TEMPORAL_ENVIRONMENT_CONTRACT:
        raise ValueError("extractor environment is invalid")
    verified_inputs = verify_task0257_temporal_inputs(paths=paths, expected_receipts=receipts)
    plan = load_verified_temporal_feature_plan(
        plan_path=args.plan,
        expected_artifact_sha256=args.expected_plan_artifact_sha256,
        expected_file_sha256=args.expected_plan_file_sha256,
    )
    replayed_plan = __import__(
        "app.analysis.vru_causal_temporal_retrospective",
        fromlist=["seal_vru_causal_temporal_feature_plan"],
    ).seal_vru_causal_temporal_feature_plan(
        inputs=verified_inputs,
        environment_contract=environment,
    )
    if replayed_plan != plan._payload:
        raise ValueError("stored temporal plan does not replay from TASK-0257")
    if args.output_root != args.plan.parent or args.output_root.name != "vru_causal_temporal_retrospective_v1":
        raise ValueError("extractor output root is invalid")
    final_path = args.output_root / "final_v1"
    if not final_path.exists() and not final_path.is_symlink():
        verify_disk_write_budget(output_paths=(final_path,), worst_case_new_bytes=1_073_741_824)
    return verified_inputs, plan, paths


def _preflight_output_state(output_root: Path) -> None:
    root = Path(output_root)
    if root.name != "vru_causal_temporal_retrospective_v1" or root.is_symlink() or not root.is_dir():
        raise ValueError("extractor output state is invalid")
    final_path = root / "final_v1"
    terminal_path = root / "terminal_failure_v1"
    if (final_path.exists() or final_path.is_symlink()) and (terminal_path.exists() or terminal_path.is_symlink()):
        raise ValueError("extractor output state contains dual terminal generations")
    if terminal_path.exists() or terminal_path.is_symlink():
        raise ValueError("extractor output state contains a terminal failure")
    allowed = {"temporal_feature_plan.json", "attempts", "final_v1"}
    entries = tuple(root.iterdir())
    if any(path.name not in allowed or path.name.startswith(".task0258-") for path in entries):
        raise ValueError("extractor output state is not empty")
    plan_path = root / "temporal_feature_plan.json"
    if plan_path.is_symlink() or not plan_path.is_file():
        raise ValueError("extractor output state is missing the frozen plan")
    attempts = root / "attempts"
    if attempts.exists() and (attempts.is_symlink() or not attempts.is_dir()):
        raise ValueError("extractor output state has an invalid attempts directory")


def _verify_existing_final_noop(
    *,
    args: argparse.Namespace,
    verified_inputs: Any,
    plan: Any,
) -> Any:
    required_values = (
        getattr(args, "receipt_registry", None),
        getattr(args, "expected_receipt_registry_artifact_sha256", None),
        getattr(args, "expected_receipt_registry_file_sha256", None),
    )
    if any(value is None for value in required_values):
        raise ValueError("existing final generation requires external receipt registry values")
    receipt_registry = Path(args.receipt_registry)
    registry, registry_bytes = _read_bounded_json(
        receipt_registry,
        max_bytes=1_048_576,
    )
    if _sha256_bytes(registry_bytes) != args.expected_receipt_registry_file_sha256:
        raise ValueError("receipt registry file SHA-256 does not match")
    verified_registry = verify_module_a_final_receipt_registry(
        registry,
        expected_artifact_sha256=args.expected_receipt_registry_artifact_sha256,
    )
    final_path = args.output_root / "final_v1"
    tiled_embeddings = load_verified_tiled_swin_embeddings(
        embeddings_path=final_path / "tiled_swin_embeddings.json",
        expected_artifact_sha256=verified_registry["artifact_receipts"]["tiled_swin_embeddings.json"],
        expected_file_sha256=verified_registry["file_receipts"]["tiled_swin_embeddings.json"],
        plan=plan,
    )
    return verify_vru_causal_temporal_final_generation(
        generation_dir=final_path,
        expected_file_receipts=verified_registry["file_receipts"],
        expected_artifact_receipts=verified_registry["artifact_receipts"],
        verified_inputs=verified_inputs,
        plan=plan,
        tiled_embeddings=tiled_embeddings,
    )


def _populate_and_publish_terminal_failure(
    *,
    work_stage: Path,
    terminal_path: Path,
    plan: Any,
    prior_state: _PriorAttemptState,
    failure: _GuardedWorkerFailure,
    publication_revalidator: Any = None,
) -> None:
    progress_path = work_stage / "private-progress.json"
    completed_count = 0
    if progress_path.exists():
        raw_progress, _encoded = _read_bounded_json(progress_path, max_bytes=67_108_864)
        raw_count = raw_progress.get("completed_count")
        if type(raw_count) is int and 0 < raw_count <= 45:
            _read_private_progress(progress_path, expected_completed_count=raw_count)
            completed_count = raw_count
    started_count = prior_state.latest_resume["completed_count"] if prior_state.latest_resume is not None else 0
    completed_count = max(completed_count, started_count)
    attempt_ordinal = len(prior_state.attempt_chain) + 1
    stage = Path(tempfile.mkdtemp(prefix=".task0258-terminal.", dir=terminal_path.parent))
    terminal_attempt = stage / "terminal_attempt"
    terminal_attempt.mkdir(mode=0o700)
    log_path = terminal_attempt / "resource_guard.jsonl"
    _copy_file_fsync(work_stage / "resource_guard.jsonl", log_path)
    latest_record = prior_state.attempt_records[-1] if prior_state.attempt_records else None
    prior_samples = latest_record["cumulative_resource_samples"] if latest_record is not None else 0
    prior_log_bytes = latest_record["cumulative_resource_log_bytes"] if latest_record is not None else 0
    resource_receipt, attempt_samples, _consecutive = _read_resource_log(
        log_path,
        expected_attempt_ordinal=attempt_ordinal,
        prior_cumulative_samples=prior_samples,
        prior_consecutive_breach_count=prior_state.prior_consecutive_breach_count,
    )
    active_runtime = int(failure.summary.get("active_runtime_nanoseconds", 0))
    active_runtime = min(
        active_runtime,
        TEMPORAL_RESOURCE_POLICY["max_cumulative_active_seconds"] * 1_000_000_000,
    )
    attempt = seal_tiled_swin_attempt(
        {
            "attempt_ordinal": attempt_ordinal,
            "prior_attempt_receipt": (
                copy.deepcopy(prior_state.attempt_chain[-1]["attempt_record"]) if prior_state.attempt_chain else None
            ),
            "plan_receipt": {
                "artifact_sha256": plan._artifact_sha256,
                "file_sha256": plan._file_sha256,
            },
            "task0257_input_receipts": copy.deepcopy(plan._payload["task0257_receipts"]),
            "started_prefix_count": started_count,
            "completed_prefix_count": completed_count,
            "new_rows_verified": completed_count - started_count,
            "cumulative_active_runtime_nanoseconds": active_runtime,
            "cumulative_resource_samples": prior_samples + attempt_samples,
            "cumulative_resource_log_bytes": prior_log_bytes + resource_receipt["size_bytes"],
            "resource_log_receipt": resource_receipt,
            "resume_input_cas": copy.deepcopy(prior_state.latest_resume_cas),
            "resume_output_cas": None,
            "received_signal": getattr(failure, "signal_name", None),
            "disposition": "terminal_failure",
            "stop_reason": failure.stop_reason,
        }
    )
    attempt_path = terminal_attempt / "attempt_record.json"
    _write_file_fsync(attempt_path, _canonical_json_bytes(attempt))
    attempt_chain = [
        *copy.deepcopy(list(prior_state.attempt_chain)),
        {
            "attempt_ordinal": attempt_ordinal,
            "attempt_record": _stored_receipt(attempt_path),
            "resource_log": resource_receipt,
            "resume_output_cas": None,
        },
    ]
    failure_record: dict[str, object] = {
        "schema_version": "agu.vru-causal-temporal-mechanical-gate.v1",
        "module_id": "existing-45-temporal-retrospective",
        "purpose": "development_diagnostic_only",
        "runtime_consumable": False,
        "training_consumable": False,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
        "input_receipts": {
            "plan": {
                "provider": "temporal_feature_plan",
                "verification_state": "verified",
                "receipt": {
                    "artifact_sha256": plan._artifact_sha256,
                    "file_sha256": plan._file_sha256,
                },
            },
            "task0257": {
                "provider": "task0257_temporal_inputs",
                "verification_state": "verified",
                "receipt": copy.deepcopy(plan._payload["task0257_receipts"]),
            },
            "tiled_embeddings": {
                "provider": "tiled_swin_embeddings",
                "verification_state": "not_reached",
                "receipt": None,
            },
            "baseline_final_evaluator": {
                "provider": "baseline_final_evaluator",
                "verification_state": "not_reached",
                "receipt": None,
            },
            "candidate_final_evaluator": {
                "provider": "candidate_final_evaluator",
                "verification_state": "not_reached",
                "receipt": None,
            },
        },
        "attempt_chain": attempt_chain,
        "ordered_check_results": [
            {"check": "preflight", "passed": True},
            {"check": "guarded_extraction", "passed": False},
        ],
        "error_bound_result": None,
        "resource_summary": {
            **copy.deepcopy(failure.summary),
            "all_sustained_limits_passed": False,
            "process_tree_reaped": True,
        },
        "publication_summary": {
            "final_generation_published": False,
            "atomic_directory_publication_required": True,
        },
        "decision": "mechanical_failure",
        "stop_reason": failure.stop_reason,
        "conditional_downstream": {
            "module_b_permitted_only_for_mechanical_pass": False,
            "module_b_spec_and_user_approval_still_required": True,
            "formal_or_runtime_authority_granted": False,
        },
    }
    failure_record["artifact_sha256"] = _canonical_sha256(failure_record)
    _write_file_fsync(stage / "mechanical_failure.json", _canonical_json_bytes(failure_record))
    _fsync_directory(terminal_attempt)
    if {path.name for path in terminal_attempt.iterdir()} != {
        "resource_guard.jsonl",
        "attempt_record.json",
    } or {path.name for path in stage.iterdir()} != {
        "terminal_attempt",
        "mechanical_failure.json",
    }:
        raise ValueError("terminal failure generation coverage is invalid")
    try:
        if terminal_path.exists() or terminal_path.is_symlink():
            raise ValueError("terminal failure generation already exists")
        final_path = terminal_path.parent / "final_v1"
        if final_path.exists() or final_path.is_symlink():
            raise ValueError("terminal failure cannot coexist with final generation")
        if publication_revalidator is not None:
            publication_revalidator()
        _publish_final_directory(stage, terminal_path)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _publish_terminal_failure(
    *,
    work_stage: Path,
    terminal_path: Path,
    plan: Any,
    prior_state: _PriorAttemptState,
    failure: _GuardedWorkerFailure,
    publication_revalidator: Any = None,
) -> None:
    existing_stages = set(terminal_path.parent.glob(".task0258-terminal.*"))
    try:
        _populate_and_publish_terminal_failure(
            work_stage=work_stage,
            terminal_path=terminal_path,
            plan=plan,
            prior_state=prior_state,
            failure=failure,
            publication_revalidator=publication_revalidator,
        )
    finally:
        for stage in set(terminal_path.parent.glob(".task0258-terminal.*")) - existing_stages:
            if stage.is_dir() and not stage.is_symlink():
                shutil.rmtree(stage)


def _publish_completed_generation(
    *,
    work_stage: Path,
    final_path: Path,
    verified_inputs: Any,
    plan: Any,
    prior_state: _PriorAttemptState,
    worker_summary: Mapping[str, object],
    progress: Mapping[str, object],
    publication_revalidator: Any,
) -> Any:
    attempt_ordinal = len(prior_state.attempt_chain) + 1
    started_prefix_count = prior_state.latest_resume["completed_count"] if prior_state.latest_resume is not None else 0
    final_stage = Path(tempfile.mkdtemp(prefix=".task0258-final.", dir=final_path.parent))
    try:
        terminal_attempt = final_stage / "terminal_attempt"
        terminal_attempt.mkdir(mode=0o700)
        final_log_path = terminal_attempt / "resource_guard.jsonl"
        _copy_file_fsync(work_stage / "resource_guard.jsonl", final_log_path)
        attempt = seal_tiled_swin_attempt(
            {
                "attempt_ordinal": attempt_ordinal,
                "prior_attempt_receipt": (
                    copy.deepcopy(prior_state.attempt_chain[-1]["attempt_record"])
                    if prior_state.attempt_chain
                    else None
                ),
                "plan_receipt": {
                    "artifact_sha256": plan._artifact_sha256,
                    "file_sha256": plan._file_sha256,
                },
                "task0257_input_receipts": copy.deepcopy(plan._payload["task0257_receipts"]),
                "started_prefix_count": started_prefix_count,
                "completed_prefix_count": 45,
                "new_rows_verified": 45 - started_prefix_count,
                "cumulative_active_runtime_nanoseconds": worker_summary["active_runtime_nanoseconds"],
                "cumulative_resource_samples": worker_summary["sample_count"],
                "cumulative_resource_log_bytes": worker_summary["log_bytes"],
                "resource_log_receipt": _file_receipt(final_log_path),
                "resume_input_cas": copy.deepcopy(prior_state.latest_resume_cas),
                "resume_output_cas": None,
                "received_signal": None,
                "disposition": "completed",
                "stop_reason": None,
            }
        )
        attempt_path = terminal_attempt / "attempt_record.json"
        _write_file_fsync(attempt_path, _canonical_json_bytes(attempt))
        attempt_chain = [
            *copy.deepcopy(list(prior_state.attempt_chain)),
            {
                "attempt_ordinal": attempt_ordinal,
                "attempt_record": _stored_receipt(attempt_path),
                "resource_log": _file_receipt(final_log_path),
                "resume_output_cas": None,
            },
        ]
        fresh = _build_fresh_tiled_swin_embeddings(
            plan=plan,
            tile_embeddings=progress["tile_embeddings"],
            attempt_chain=attempt_chain,
            resource_summary={
                "attempt_count": len(attempt_chain),
                "cumulative_active_runtime_nanoseconds": attempt["cumulative_active_runtime_nanoseconds"],
                "cumulative_resource_samples": worker_summary["sample_count"],
                "cumulative_resource_log_bytes": worker_summary["log_bytes"],
                "all_sustained_limits_passed": True,
                "process_tree_reaped": worker_summary["process_tree_reaped"],
            },
        )
        generation = _build_vru_causal_temporal_final_generation(
            inputs=verified_inputs,
            plan=plan,
            tiled_embeddings=fresh,
        )
        if generation.mechanical_gate.get("decision") == "mechanical_failure":
            raise ValueError("mechanical failure cannot be published as final_v1")
        outputs = {
            "tiled_swin_embeddings.json": fresh._payload,
            "temporal_retrospective.json": generation.retrospective,
            "baseline_final_evaluator.json": generation.baseline_evaluator,
            "candidate_final_evaluator.json": generation.candidate_evaluator,
            "mechanical_gate.json": generation.mechanical_gate,
        }
        for filename, payload in outputs.items():
            output_path = final_stage / filename
            _write_file_fsync(output_path, _canonical_json_bytes(payload))
            replayed, encoded = _read_bounded_json(output_path, max_bytes=67_108_864)
            if replayed != payload or encoded != _canonical_json_bytes(payload):
                raise ValueError("final generation member failed byte replay")
        _fsync_directory(terminal_attempt)
        if {path.name for path in terminal_attempt.iterdir()} != {
            "resource_guard.jsonl",
            "attempt_record.json",
        } or {path.name for path in final_stage.iterdir()} != {
            "terminal_attempt",
            *outputs,
        }:
            raise ValueError("final generation coverage is invalid")
        publication_revalidator()
        _publish_final_directory(final_stage, final_path)
        return generation
    finally:
        if final_stage.exists():
            shutil.rmtree(final_stage)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size != 1:
        raise ValueError("batch size must be exactly 1")
    _preflight_output_state(args.output_root)
    receipt_fields = (
        "receipt_registry",
        "expected_receipt_registry_artifact_sha256",
        "expected_receipt_registry_file_sha256",
    )
    supplied_receipt_values = tuple(getattr(args, field, None) is not None for field in receipt_fields)
    final_candidate = args.output_root / "final_v1"
    if any(supplied_receipt_values) and not all(supplied_receipt_values):
        raise ValueError("external final receipt registry arguments must be supplied together")
    if (final_candidate.exists() or final_candidate.is_symlink()) and not all(supplied_receipt_values):
        raise ValueError("extractor output state contains an unreceipted final generation")
    if not final_candidate.exists() and not final_candidate.is_symlink() and any(supplied_receipt_values):
        raise ValueError("external final receipt registry requires an existing final generation")
    verified_inputs, plan, paths = _verify_preflight(args)
    with _exclusive_directory_lock(args.output_root):
        _preflight_output_state(args.output_root)
        verified_inputs, plan, paths = _verify_preflight(args)
        final_path = args.output_root / "final_v1"
        terminal_path = args.output_root / "terminal_failure_v1"
        if final_path.exists() or final_path.is_symlink():
            generation = _verify_existing_final_noop(
                args=args,
                verified_inputs=verified_inputs,
                plan=plan,
            )
            mechanical_gate = getattr(generation, "mechanical_gate", {})
            decision = mechanical_gate.get("decision") if isinstance(mechanical_gate, Mapping) else None
            print(
                json.dumps(
                    {
                        "final_generation": str(final_path),
                        "decision": decision,
                        "state": "verified_read_only_noop",
                    },
                    sort_keys=True,
                )
            )
            return 0
        prior_state = _load_prior_attempt_state(
            output_root=args.output_root,
            plan=plan,
            raw_expected_attempt_receipts=args.expected_prior_attempt_receipt,
            raw_expected_resume_cas=args.expected_resume_cas,
        )
        if final_path.exists() or terminal_path.exists() or any(args.output_root.glob(".task0258-*")):
            raise ValueError("extractor output state is not empty")

        def revalidate_terminal_publication() -> None:
            _verified_inputs, verified_plan, _paths = _verify_preflight(args)
            if (
                _load_prior_attempt_state(
                    output_root=args.output_root,
                    plan=verified_plan,
                    raw_expected_attempt_receipts=args.expected_prior_attempt_receipt,
                    raw_expected_resume_cas=args.expected_resume_cas,
                )
                != prior_state
            ):
                raise ValueError("prior attempt state changed before terminal publication")
            if final_path.exists() or final_path.is_symlink():
                raise ValueError("terminal failure cannot coexist with final generation")
            if terminal_path.exists() or terminal_path.is_symlink():
                raise ValueError("terminal failure generation already exists")
            verify_disk_write_budget(
                output_paths=(terminal_path,),
                worst_case_new_bytes=1_073_741_824,
            )

        attempt_ordinal = len(prior_state.attempt_chain) + 1
        if attempt_ordinal > 3:
            raise ValueError("Module-A extraction attempt limit is exhausted")
        work_stage_owner = tempfile.TemporaryDirectory(
            prefix=".task0258-work.",
            dir=args.output_root,
        )
        work_stage = Path(work_stage_owner.name)
        progress_path = work_stage / "private-progress.json"
        request_path = work_stage / "worker-request.json"
        started_prefix_count = 0
        if prior_state.latest_resume is not None:
            started_prefix_count = prior_state.latest_resume["completed_count"]
            _atomic_replace_private(
                progress_path,
                {
                    "schema_version": "agu.vru-causal-tiled-swin-private-progress.v1",
                    "completed_count": started_prefix_count,
                    "tile_embeddings": prior_state.latest_resume["completed_examples"],
                },
            )
        request = {
            "plan_path": str(args.plan.resolve(strict=True)),
            "plan_artifact_sha256": args.expected_plan_artifact_sha256,
            "plan_file_sha256": args.expected_plan_file_sha256,
            "source_video_paths": [str(path.resolve(strict=True)) for path in paths.source_videos],
            "checkpoint_path": str(paths.checkpoints[1].resolve(strict=True)),
            "progress_path": str(progress_path),
            "started_prefix_count": started_prefix_count,
        }
        _write_file_fsync(request_path, _canonical_json_bytes(request))
        log_path = work_stage / "resource_guard.jsonl"
        _write_file_fsync(log_path, b"")
        request_file_sha256 = _sha256_bytes(request_path.read_bytes())
        latest_record = prior_state.attempt_records[-1] if prior_state.attempt_records else None
        verify_disk_write_budget(
            output_paths=(final_path, terminal_path),
            worst_case_new_bytes=1_073_741_824,
        )
        try:
            worker_summary = _run_guarded_worker(
                command=(
                    sys.executable,
                    __file__,
                    "--worker-request",
                    str(request_path),
                    "--expected-worker-request-file-sha256",
                    request_file_sha256,
                ),
                log_path=log_path,
                attempt_ordinal=attempt_ordinal,
                prior_active_runtime_nanoseconds=(
                    latest_record["cumulative_active_runtime_nanoseconds"] if latest_record is not None else 0
                ),
                prior_resource_samples=(
                    latest_record["cumulative_resource_samples"] if latest_record is not None else 0
                ),
                prior_resource_log_bytes=(
                    latest_record["cumulative_resource_log_bytes"] if latest_record is not None else 0
                ),
                prior_consecutive_breach_count=prior_state.prior_consecutive_breach_count,
            )
        except _GuardedWorkerInterrupted as interruption:
            verified_inputs, plan, paths = _verify_preflight(args)
            reloaded_prior = _load_prior_attempt_state(
                output_root=args.output_root,
                plan=plan,
                raw_expected_attempt_receipts=args.expected_prior_attempt_receipt,
                raw_expected_resume_cas=args.expected_resume_cas,
            )
            if reloaded_prior != prior_state:
                raise ValueError("prior attempt state changed during extraction")
            progress_payload: dict[str, object] | None = None
            if progress_path.exists():
                raw_progress, _encoded = _read_bounded_json(progress_path, max_bytes=67_108_864)
                raw_count = raw_progress.get("completed_count")
                if type(raw_count) is int and 0 < raw_count <= 45:
                    progress_payload = _read_private_progress(
                        progress_path,
                        expected_completed_count=raw_count,
                    )
            completed_count = (
                progress_payload["completed_count"] if progress_payload is not None else started_prefix_count
            )
            if completed_count == 45:
                worker_summary = interruption.summary
            elif attempt_ordinal <= 2 and completed_count > started_prefix_count:
                verify_disk_write_budget(
                    output_paths=(args.output_root / "attempts" / f"attempt-{attempt_ordinal:04d}",),
                    worst_case_new_bytes=1_073_741_824,
                )
                attempt_receipt, resume_cas = _publish_recoverable_attempt(
                    work_stage=work_stage,
                    output_root=args.output_root,
                    plan=plan,
                    prior_state=prior_state,
                    worker_summary=interruption.summary,
                    completed_rows=progress_payload["tile_embeddings"],
                    signal_name=interruption.signal_name,
                )
                work_stage_owner.cleanup()
                print(
                    json.dumps(
                        {
                            "state": "interrupted_recoverable",
                            "attempt_receipt": attempt_receipt,
                            "resume_cas": resume_cas,
                        },
                        sort_keys=True,
                    )
                )
                return 130 if interruption.signal_name == "SIGINT" else 143
            else:
                failure = _GuardedWorkerFailure(
                    "attempt_limit" if attempt_ordinal == 3 else "non_advancing_prefix",
                    "external signal did not produce a recoverable prefix",
                )
                failure.signal_name = interruption.signal_name
                failure.summary = interruption.summary
                _publish_terminal_failure(
                    work_stage=work_stage,
                    terminal_path=terminal_path,
                    plan=plan,
                    prior_state=prior_state,
                    failure=failure,
                    publication_revalidator=revalidate_terminal_publication,
                )
                work_stage_owner.cleanup()
                return RESOURCE_LIMIT_EXIT_CODE
        except _GuardedWorkerFailure as failure:
            _publish_terminal_failure(
                work_stage=work_stage,
                terminal_path=terminal_path,
                plan=plan,
                prior_state=prior_state,
                failure=failure,
                publication_revalidator=revalidate_terminal_publication,
            )
            work_stage_owner.cleanup()
            print(json.dumps({"terminal_generation": str(terminal_path), "stop_reason": failure.stop_reason}))
            return RESOURCE_LIMIT_EXIT_CODE
        progress = _read_private_progress(progress_path, expected_completed_count=45)

        def revalidate_publication() -> None:
            _verified_inputs, verified_plan, _paths = _verify_preflight(args)
            if (
                _load_prior_attempt_state(
                    output_root=args.output_root,
                    plan=verified_plan,
                    raw_expected_attempt_receipts=args.expected_prior_attempt_receipt,
                    raw_expected_resume_cas=args.expected_resume_cas,
                )
                != prior_state
            ):
                raise ValueError("prior attempt state changed before publication")
            verify_disk_write_budget(
                output_paths=(final_path,),
                worst_case_new_bytes=1_073_741_824,
            )

        try:
            generation = _publish_completed_generation(
                work_stage=work_stage,
                final_path=final_path,
                verified_inputs=verified_inputs,
                plan=plan,
                prior_state=prior_state,
                worker_summary=worker_summary,
                progress=progress,
                publication_revalidator=revalidate_publication,
            )
        except Exception as error:
            if final_path.exists() or final_path.is_symlink():
                work_stage_owner.cleanup()
                raise RuntimeError("final generation became visible but publication acknowledgement failed") from error
            failure = _GuardedWorkerFailure(
                _classify_finalization_failure(error),
                "private finalization failed",
            )
            failure.summary = dict(worker_summary)
            _publish_terminal_failure(
                work_stage=work_stage,
                terminal_path=terminal_path,
                plan=plan,
                prior_state=prior_state,
                failure=failure,
                publication_revalidator=revalidate_terminal_publication,
            )
            work_stage_owner.cleanup()
            print(
                json.dumps(
                    {
                        "terminal_generation": str(terminal_path),
                        "stop_reason": failure.stop_reason,
                    }
                )
            )
            return RESOURCE_LIMIT_EXIT_CODE
        work_stage_owner.cleanup()
    print(json.dumps({"final_generation": str(final_path), "decision": generation.mechanical_gate["decision"]}))
    return 0


if __name__ == "__main__":
    if (
        len(sys.argv) == 5
        and sys.argv[1] == "--worker-request"
        and sys.argv[3] == "--expected-worker-request-file-sha256"
    ):
        raise SystemExit(_worker_main(Path(sys.argv[2]), expected_file_sha256=sys.argv[4]))
    raise SystemExit(main())
