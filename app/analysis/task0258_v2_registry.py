"""TASK-0258 Amendment-001 v2 run-history registry durable write path.

Thin durable-write functions that compose the registry schemas/validators
(`task0258_run_history`) with the atomic publication primitives
(`task0258_v2_fs`). These are the trust spine's persistence boundary: every
write validates first, then publishes no-clobber with exact basenames.
"""

from __future__ import annotations

from pathlib import Path

from app.analysis.task0258_module_a_v2 import compact_canonical_json
from app.analysis.task0258_run_history import (
    claim_filename,
    completion_filename,
    history_filename,
    verify_completed_successor,
    verify_history_transition,
    verify_run_admission,
    verify_run_consumption_claim,
    verify_run_consumption_completed,
    verify_run_history_marker,
)
from app.analysis.task0258_v2_fs import (
    atomic_write_json,
    seal_generation_directory,
    verify_absent,
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
    seal_run_consumption_claim(registry_dir, auth_sha256, claim_payload)
    admission_bytes = (compact_canonical_json(admission_payload) + "\n").encode("utf-8")
    seal_generation_directory(
        output_root.parent,
        output_root.name,
        {"run_admission.json": admission_bytes},
        ("run_admission.json",),
        flock_path=flock_path,
    )
    seal_run_consumption_completed(registry_dir, auth_sha256, completion_payload)
    return output_root


def seal_run_consumption_claim(
    registry_dir: Path,
    auth_sha256: str,
    payload: object,
) -> Path:
    """Validate and no-clobber publish ``<AUTH_SHA>.claim.json``."""
    verify_run_consumption_claim(payload)
    final = registry_dir / claim_filename(auth_sha256)
    atomic_write_json(final, payload)
    return final


def seal_run_consumption_completed(
    registry_dir: Path,
    auth_sha256: str,
    payload: object,
) -> Path:
    """Validate and no-clobber publish ``<AUTH_SHA>.completed.json``."""
    verify_run_consumption_completed(payload)
    final = registry_dir / completion_filename(auth_sha256)
    atomic_write_json(final, payload)
    return final


def append_run_history_marker(
    registry_dir: Path,
    auth_sha256: str,
    payload: object,
    prior_event: str | None,
) -> Path:
    """Validate and no-clobber append a history marker.

    ``prior_event`` is the immediately preceding history event, or ``None`` for
    the first marker (whose single legal predecessor is the completed ledger).
    The marker's ``event`` must be the exact legal successor; the ordinal must
    match the filename ``NN``. Publishes ``<AUTH_SHA>.history-<NN>-<EVENT>.json``.
    """
    verify_run_history_marker(payload)
    event = payload["event"]
    if prior_event is None:
        verify_completed_successor(event)
    else:
        verify_history_transition(prior_event, event)
    final = registry_dir / history_filename(auth_sha256, payload["sequence_ordinal"], event)
    verify_absent(final)
    atomic_write_json(final, payload)
    return final


__all__ = [
    "seal_run_consumption_claim",
    "seal_run_consumption_completed",
    "append_run_history_marker",
    "create_run_history_registry",
]
