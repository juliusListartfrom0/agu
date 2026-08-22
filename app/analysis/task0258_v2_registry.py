"""TASK-0258 Amendment-001 v2 run-history registry durable write path.

Thin durable-write functions that compose the registry schemas/validators
(`task0258_run_history`) with the atomic publication primitives
(`task0258_v2_fs`). These are the trust spine's persistence boundary: every
write validates first, then publishes no-clobber with exact basenames.
"""

from __future__ import annotations

import hashlib
import os
from contextlib import nullcontext
from pathlib import Path

from app.analysis.task0258_module_a_v2 import (
    compact_canonical_json,
    is_sha256,
    verify_internal_artifact_hash,
)
from app.analysis.task0258_run_history import (
    _registry_history_filenames_from_fd,
    claim_filename,
    completion_filename,
    history_filename,
    parse_history_filename,
    replay_run_history_registry,
    verify_completed_successor,
    verify_history_transition,
    verify_run_admission,
    verify_run_consumption_claim,
    verify_run_consumption_completed,
    verify_run_history_marker,
)
from app.analysis.task0258_v2_fs import (
    FlockHandle,
    _ensure_directory_no_follow,
    _open_existing_directory_no_follow,
    active_flock,
    atomic_write_json_at,
    exclusive_flock_at,
    seal_generation_directory,
)


def create_run_history_registry(
    *,
    registry_dir: Path,
    auth_sha256: str,
    claim_payload: object,
    admission_payload: object,
    completion_payload: object,
    output_root: Path,
    flock_path: Path,
) -> Path:
    """Open the run-history registry: claim -> output root -> completed.

    Validates every payload first, then no-clobber publishes in the exact order
    (claim.json, the new output root whose sole member is ``run_admission.json``,
    then completed.json). Returns ``output_root``. A failure after the claim
    leaves fail-closed residue that is never repaired here.
    """
    verify_run_consumption_claim(claim_payload)
    verify_run_admission(admission_payload)
    verify_run_consumption_completed(completion_payload)
    if (
        not isinstance(claim_payload, dict)
        or not isinstance(admission_payload, dict)
        or not isinstance(completion_payload, dict)
    ):
        raise ValueError("run-history payloads must be mutable JSON objects")
    for payload in (claim_payload, admission_payload, completion_payload):
        verify_internal_artifact_hash(payload)
    if not is_sha256(auth_sha256):
        raise ValueError("run-history authorization SHA is invalid")
    if claim_payload["authorization_receipt"]["artifact_sha256"] != auth_sha256:
        raise ValueError("claim authorization receipt is not bound to the registry name")
    claim_receipt = _json_artifact_receipt(claim_payload)
    if admission_payload["claim_receipt"] != claim_receipt:
        raise ValueError("admission claim receipt is not bound to the claim bytes")
    admission_receipt = _json_artifact_receipt(admission_payload)
    if (
        completion_payload["claim_receipt"] != claim_receipt
        or completion_payload["admission_receipt"] != admission_receipt
    ):
        raise ValueError("completion receipts are not bound to the claim/admission bytes")
    for payload in (admission_payload, completion_payload):
        if payload["authorization_receipt"]["artifact_sha256"] != auth_sha256:
            raise ValueError("run-history authorization receipt is not bound to the registry name")
        if payload["run_id"] != claim_payload["run_id"] or payload["nonce"] != claim_payload["nonce"]:
            raise ValueError("run-history identity is inconsistent")
    expected_root = str(output_root)
    if any(
        payload["output_root_absolute_path"] != expected_root
        for payload in (claim_payload, admission_payload, completion_payload)
    ):
        raise ValueError("run-history output root is not bound to the requested root")
    _ensure_directory_no_follow(Path(registry_dir))
    history_lock_path = Path(registry_dir) / f".{auth_sha256}.history.lock"
    registry_fd = _open_existing_directory_no_follow(Path(registry_dir))
    try:
        with exclusive_flock_at(registry_fd, history_lock_path.name, history_lock_path) as history_lock:
            seal_run_consumption_claim(
                registry_dir,
                auth_sha256,
                claim_payload,
                held_lock=history_lock,
                registry_fd=registry_fd,
            )
            admission_bytes = (compact_canonical_json(admission_payload) + "\n").encode("utf-8")
            seal_generation_directory(
                output_root.parent,
                output_root.name,
                {"run_admission.json": admission_bytes},
                ("run_admission.json",),
                flock_path=flock_path,
            )
            seal_run_consumption_completed(
                registry_dir,
                auth_sha256,
                completion_payload,
                held_lock=history_lock,
                registry_fd=registry_fd,
            )
    finally:
        os.close(registry_fd)
    return output_root


def seal_run_consumption_claim(
    registry_dir: Path,
    auth_sha256: str,
    payload: object,
    *,
    held_lock: FlockHandle | None = None,
    registry_fd: int | None = None,
) -> Path:
    """Validate and no-clobber publish ``<AUTH_SHA>.claim.json``."""
    verify_run_consumption_claim(payload)
    if not isinstance(payload, dict):
        raise ValueError("claim payload must be a mutable JSON object")
    verify_internal_artifact_hash(payload)
    registry_dir = Path(registry_dir)
    if registry_fd is None:
        _ensure_directory_no_follow(registry_dir)
    lock_path = registry_dir / f".{auth_sha256}.history.lock"
    if registry_fd is not None:
        if held_lock is None:
            raise ValueError("registry directory descriptor requires its held history lock")
        held_lock.assert_held(lock_path)
    elif held_lock is not None:
        held_lock.assert_held(lock_path)
    owns_directory_fd = registry_fd is None
    if owns_directory_fd:
        registry_fd = _open_existing_directory_no_follow(registry_dir)
    try:
        assert registry_fd is not None
        atomic_write_json_at(registry_fd, claim_filename(auth_sha256), payload)
    finally:
        if owns_directory_fd:
            os.close(registry_fd)
    final = registry_dir / claim_filename(auth_sha256)
    return final


def seal_run_consumption_completed(
    registry_dir: Path,
    auth_sha256: str,
    payload: object,
    *,
    held_lock: FlockHandle | None = None,
    registry_fd: int | None = None,
) -> Path:
    """Validate and no-clobber publish ``<AUTH_SHA>.completed.json``."""
    verify_run_consumption_completed(payload)
    if not isinstance(payload, dict):
        raise ValueError("completion payload must be a mutable JSON object")
    verify_internal_artifact_hash(payload)
    registry_dir = Path(registry_dir)
    if registry_fd is None:
        _ensure_directory_no_follow(registry_dir)
    lock_path = registry_dir / f".{auth_sha256}.history.lock"
    if registry_fd is not None:
        if held_lock is None:
            raise ValueError("registry directory descriptor requires its held history lock")
        held_lock.assert_held(lock_path)
    elif held_lock is not None:
        held_lock.assert_held(lock_path)
    owns_directory_fd = registry_fd is None
    if owns_directory_fd:
        registry_fd = _open_existing_directory_no_follow(registry_dir)
    try:
        assert registry_fd is not None
        atomic_write_json_at(registry_fd, completion_filename(auth_sha256), payload)
    finally:
        if owns_directory_fd:
            os.close(registry_fd)
    final = registry_dir / completion_filename(auth_sha256)
    return final


def append_run_history_marker(
    registry_dir: Path,
    auth_sha256: str,
    payload: object,
    prior_event: str | None,
    *,
    held_lock: FlockHandle | None = None,
) -> Path:
    """Validate and no-clobber append a history marker.

    ``prior_event`` is the immediately preceding history event, or ``None`` for
    the first marker (whose single legal predecessor is the completed ledger).
    The marker's ``event`` must be the exact legal successor; the ordinal must
    match the filename ``NN``. Publishes ``<AUTH_SHA>.history-<NN>-<EVENT>.json``.
    """
    if not is_sha256(auth_sha256):
        raise ValueError("history authorization SHA is invalid")
    verify_run_history_marker(payload)
    registry_dir = Path(registry_dir)
    lock_path = registry_dir / f".{auth_sha256}.history.lock"
    if held_lock is None:
        held_lock = active_flock(lock_path)
    if held_lock is not None:
        held_lock.assert_held(lock_path)
    _ensure_directory_no_follow(registry_dir)
    registry_fd = _open_existing_directory_no_follow(registry_dir)
    if held_lock is None:
        lock_context = exclusive_flock_at(registry_fd, lock_path.name, lock_path)
    else:
        lock_context = nullcontext(held_lock)
    with lock_context as history_lock:
        try:
            replayed = replay_run_history_registry(
                registry_dir,
                auth_sha256,
                held_lock=history_lock,
                registry_fd=registry_fd,
            )
            claim = replayed[0]
            completion = replayed[1]
            authorization_receipt = claim["authorization_receipt"]
            completion_receipt = _json_artifact_receipt(completion)
            if payload["authorization_receipt"] != authorization_receipt:
                raise ValueError("history marker authorization is not bound to the claim")
            if payload["run_identity_receipt"] != completion_receipt:
                raise ValueError("history marker run identity is not bound to completion")
            if (
                payload["run_id"] != claim["run_id"]
                or payload["nonce"] != claim["nonce"]
                or payload["output_root"] != claim["output_root_absolute_path"]
            ):
                raise ValueError("history marker run identity/root binding drifted")
            filenames = _registry_history_filenames_from_fd(registry_fd, auth_sha256)
            marker_names = filenames[2:]
            actual_prior = None
            actual_prior_receipt = completion_receipt
            expected_ordinal = 1
            if marker_names:
                expected_ordinal, actual_prior = parse_history_filename(marker_names[-1], auth_sha256)
                expected_ordinal += 1
                actual_prior_receipt = _json_artifact_receipt(replayed[-1])
            if prior_event != actual_prior:
                raise ValueError("caller prior_event does not match the durable registry head")
            if payload["prior_marker_receipt"] != actual_prior_receipt:
                raise ValueError("history marker predecessor receipt does not match the durable registry head")
            event = payload["event"]
            if actual_prior is None:
                verify_completed_successor(event)
            else:
                verify_history_transition(actual_prior, event)
            if payload["sequence_ordinal"] != expected_ordinal:
                raise ValueError("history marker ordinal does not match the durable registry head")
            final_name = history_filename(auth_sha256, expected_ordinal, event)
            atomic_write_json_at(registry_fd, final_name, payload)
            return registry_dir / final_name
        finally:
            os.close(registry_fd)


__all__ = [
    "seal_run_consumption_claim",
    "seal_run_consumption_completed",
    "append_run_history_marker",
    "create_run_history_registry",
]


def _json_artifact_receipt(payload: dict[str, object]) -> dict[str, str]:
    data = (compact_canonical_json(payload) + "\n").encode("utf-8")
    return {
        "artifact_sha256": str(payload["artifact_sha256"]),
        "file_sha256": hashlib.sha256(data).hexdigest(),
    }
