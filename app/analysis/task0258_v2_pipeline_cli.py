"""TASK-0258 Amendment-001 v2 production pipeline orchestration.

Guarded end-to-end skeleton of the v2 run: open the run-history registry
(claim -> admission -> completion), assemble and publish ``candidate_v2``
(ten members), seal the candidate receipt bundle, then publish
``verified_result_v2``. The heavy producer/verification *extraction* is injected
as payloads by the caller (the runtime extraction providers in
``task0258_v2_verification_extract`` and the parent producer path); this module
owns the sealed state-machine transitions and stays fail-closed.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from app.analysis.task0258_module_a_v2 import compact_canonical_json
from app.analysis.task0258_v2_artifacts import CANDIDATE_MEMBER_PATHS
from app.analysis.task0258_v2_pipeline import (
    seal_candidate_receipt_bundle,
    seal_candidate_v2,
    seal_verified_result,
)
from app.analysis.task0258_v2_registry import create_run_history_registry


def encode_member(value: object) -> bytes:
    """Canonical-JSON-encode a member value; pass bytes through verbatim."""
    if isinstance(value, bytes):
        return value
    return (compact_canonical_json(value) + "\n").encode("utf-8")


def assemble_candidate_members(members: Mapping[str, object]) -> dict[str, bytes]:
    """Validate exact ten-member coverage and encode every member."""
    if set(members) != set(CANDIDATE_MEMBER_PATHS):
        raise ValueError(
            f"candidate member coverage mismatch: "
            f"missing={sorted(set(CANDIDATE_MEMBER_PATHS) - set(members))} "
            f"extra={sorted(set(members) - set(CANDIDATE_MEMBER_PATHS))}"
        )
    return {rel: encode_member(value) for rel, value in members.items()}


def run_v2_pipeline(
    *,
    output_root: Path,
    registry_dir: Path,
    flock_path: Path,
    auth_sha256: str,
    claim_payload: object,
    admission_payload: object,
    completion_payload: object,
    candidate_members: Mapping[str, object],
    bundle_path: Path,
    bundle_payload: object,
    result_payload: object,
) -> dict[str, Path]:
    """Run the guarded v2 state machine to a sealed verified result.

    Phases: registry open -> candidate_v2 publish -> candidate receipt bundle ->
    verified_result_v2 publish. Every transition validates first and publishes
    no-clobber; a failure at any phase leaves fail-closed residue.
    """
    create_run_history_registry(
        registry_dir=registry_dir,
        auth_sha256=auth_sha256,
        claim_payload=claim_payload,
        admission_payload=admission_payload,
        completion_payload=completion_payload,
        output_root=output_root,
        flock_path=flock_path,
    )
    encoded = assemble_candidate_members(candidate_members)
    candidate = seal_candidate_v2(output_root, encoded, flock_path=flock_path)
    seal_candidate_receipt_bundle(bundle_path, bundle_payload)
    result = seal_verified_result(output_root, result_payload, flock_path=flock_path)
    return {"candidate": candidate, "bundle": bundle_path, "result": result}


__all__ = [
    "encode_member",
    "assemble_candidate_members",
    "run_v2_pipeline",
]
