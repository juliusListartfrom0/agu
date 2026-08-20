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

import hashlib
import os
import secrets
from collections.abc import Mapping
from pathlib import Path

from app.analysis.task0258_module_a_v2 import (
    canonical_artifact_sha256,
    compact_canonical_json,
    verify_internal_artifact_hash,
)
from app.analysis.task0258_v2_artifacts import (
    CANDIDATE_MEMBER_PATHS,
    verify_candidate_gate,
    verify_candidate_receipt_bundle,
    verify_postpublication_verification,
)
from app.analysis.task0258_v2_pipeline import (
    build_member_receipts,
    seal_candidate_receipt_bundle,
    seal_candidate_v2,
    seal_verified_result,
)
from app.analysis.task0258_v2_registry import create_run_history_registry

_PIPELINE_CONTEXT_TOKENS: set[tuple[int, bytes]] = set()


class _VerifiedV2PipelineAdmissionContext:
    """Process-private admission capability issued by the authorized runner."""

    __slots__ = ("_pid", "_token")

    def __init__(self, token: bytes) -> None:
        self._pid = os.getpid()
        self._token = token


def _issue_verified_v2_pipeline_admission_context() -> _VerifiedV2PipelineAdmissionContext:
    token = secrets.token_bytes(32)
    _PIPELINE_CONTEXT_TOKENS.add((os.getpid(), token))
    return _VerifiedV2PipelineAdmissionContext(token)


def _require_verified_pipeline_context(context: object) -> None:
    if not isinstance(context, _VerifiedV2PipelineAdmissionContext):
        raise PermissionError("v2 pipeline requires an authorized admission context")
    if (context._pid, context._token) not in _PIPELINE_CONTEXT_TOKENS or context._pid != os.getpid():
        raise PermissionError("v2 pipeline admission context is invalid or expired")


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
    encoded = {rel: encode_member(value) for rel, value in members.items()}
    for rel, data in encoded.items():
        if rel.endswith(".json"):
            import json

            payload = json.loads(data)
            if not isinstance(payload, Mapping):
                raise ValueError(f"candidate member must be a JSON object: {rel}")
            verify_internal_artifact_hash(payload)
    return encoded


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
    authorization_context: object | None = None,
) -> dict[str, Path]:
    """Run the guarded v2 state machine to a sealed verified result.

    Phases: registry open -> candidate_v2 publish -> candidate receipt bundle ->
    verified_result_v2 publish. Every transition validates first and publishes
    no-clobber; a failure at any phase leaves fail-closed residue.
    """
    _require_verified_pipeline_context(authorization_context)
    encoded = assemble_candidate_members(candidate_members)
    candidate_gate = candidate_members.get("candidate_gate.json")
    if not isinstance(candidate_gate, Mapping):
        raise ValueError("candidate members must include an object candidate_gate")
    verify_candidate_gate(candidate_gate)
    if not isinstance(bundle_payload, Mapping):
        raise ValueError("candidate bundle must be an object")
    verify_candidate_receipt_bundle(bundle_payload)
    if not isinstance(result_payload, Mapping):
        raise ValueError("verified result must be an object")
    verify_postpublication_verification(result_payload)
    create_run_history_registry(
        registry_dir=registry_dir,
        auth_sha256=auth_sha256,
        claim_payload=claim_payload,
        admission_payload=admission_payload,
        completion_payload=completion_payload,
        output_root=output_root,
        flock_path=flock_path,
    )
    candidate = seal_candidate_v2(output_root, encoded, flock_path=flock_path)
    actual_member_receipts = build_member_receipts(candidate)
    if list(bundle_payload["ordered_member_receipts"]) != actual_member_receipts:
        raise ValueError("candidate receipt bundle is not bound to the published candidate bytes")
    seal_candidate_receipt_bundle(bundle_path, bundle_payload)
    bundle_bytes = bundle_path.read_bytes()
    bound_result = dict(result_payload)
    bound_result["candidate_receipt_bundle_receipt"] = {
        "artifact_sha256": bundle_payload["artifact_sha256"],
        "file_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
    }
    bound_result["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in bound_result.items() if key != "artifact_sha256"}
    )
    verify_postpublication_verification(bound_result)
    result = seal_verified_result(output_root, bound_result, flock_path=flock_path)
    return {"candidate": candidate, "bundle": bundle_path, "result": result}


__all__ = [
    "encode_member",
    "assemble_candidate_members",
    "run_v2_pipeline",
]
