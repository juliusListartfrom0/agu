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
    is_sha256,
    verify_internal_artifact_hash,
)
from app.analysis.task0258_v2_artifacts import (
    CANDIDATE_INPUT_RECEIPT_PROVIDERS,
    CANDIDATE_MEMBER_PATHS,
    verify_candidate_gate,
    verify_candidate_receipt_bundle,
    verify_postpublication_verification,
)
from app.analysis.task0258_v2_fs import exclusive_flock
from app.analysis.task0258_v2_pipeline import (
    build_member_receipts,
    read_published_candidate_receipt_bundle,
    seal_candidate_receipt_bundle,
    seal_candidate_v2,
    seal_verified_result,
)
from app.analysis.task0258_v2_registry import create_run_history_registry

_SYNTHETIC_PIPELINE_CONTEXT_TOKENS: set[tuple[int, bytes]] = set()
_SYNTHETIC_PIPELINE_CONTEXT_TOKEN = object()


class _SyntheticV2PipelineTestContext:
    """Process-private context for temporary diagnostic pipeline tests only."""

    __slots__ = ("_marker", "_pid", "_token")

    def __init__(self, token: bytes) -> None:
        self._marker = _SYNTHETIC_PIPELINE_CONTEXT_TOKEN
        self._pid = os.getpid()
        self._token = token


def _issue_synthetic_v2_pipeline_test_context() -> _SyntheticV2PipelineTestContext:
    token = secrets.token_bytes(32)
    _SYNTHETIC_PIPELINE_CONTEXT_TOKENS.add((os.getpid(), token))
    return _SyntheticV2PipelineTestContext(token)


def _require_synthetic_pipeline_test_context(context: object) -> None:
    if (
        not isinstance(context, _SyntheticV2PipelineTestContext)
        or context._marker is not _SYNTHETIC_PIPELINE_CONTEXT_TOKEN
    ):
        raise PermissionError("synthetic v2 pipeline requires its explicit test context")
    if (context._pid, context._token) not in _SYNTHETIC_PIPELINE_CONTEXT_TOKENS or context._pid != os.getpid():
        raise PermissionError("synthetic v2 pipeline test context is invalid or expired")


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
    synthetic_test_only: bool = False,
) -> dict[str, Path]:
    """Run the guarded v2 state machine to a sealed diagnostic result.

    Phases: registry open -> candidate_v2 publish -> candidate receipt bundle ->
    verified_result_v2 publish. Every transition validates first and publishes
    no-clobber; a failure at any phase leaves fail-closed residue. The local
    implementation has no production admission issuer: callers must opt into
    ``synthetic_test_only`` explicitly, while an unimplemented production
    invocation fails before any filesystem write.
    """
    if not synthetic_test_only:
        raise PermissionError(
            "production v2 pipeline requires an externally authorized admission; "
            "use synthetic_test_only only for temporary diagnostics"
        )
    _require_synthetic_pipeline_test_context(authorization_context)
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
    if not all(isinstance(payload, Mapping) for payload in (claim_payload, admission_payload, completion_payload)):
        raise ValueError("run-history payloads must be objects")
    _require_pipeline_receipt_bindings(
        auth_sha256=auth_sha256,
        candidate_gate=candidate_gate,
        admission_payload=admission_payload,
        completion_payload=completion_payload,
        bundle_payload=bundle_payload,
        result_payload=result_payload,
    )
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
    with exclusive_flock(flock_path):
        actual_member_receipts = build_member_receipts(candidate)
    if list(bundle_payload["ordered_member_receipts"]) != actual_member_receipts:
        raise ValueError("candidate receipt bundle is not bound to the published candidate bytes")
    seal_candidate_receipt_bundle(
        bundle_path,
        bundle_payload,
        candidate_dir=candidate,
        output_flock_path=flock_path,
    )
    bundle_bytes = read_published_candidate_receipt_bundle(bundle_path, expected_payload=bundle_payload)
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


def _require_pipeline_receipt_bindings(
    *,
    auth_sha256: str,
    candidate_gate: Mapping[str, object],
    admission_payload: Mapping[str, object],
    completion_payload: Mapping[str, object],
    bundle_payload: Mapping[str, object],
    result_payload: Mapping[str, object],
) -> None:
    """Require the supplied pipeline payloads to share one receipt tuple.

    This preflight runs before registry or output publication.  It closes the
    caller-self-reporting seam where individually valid candidate, bundle, and
    result objects could otherwise describe different authorization, admission,
    history, or candidate-member bytes.
    """
    if not is_sha256(auth_sha256):
        raise ValueError("pipeline authorization SHA is invalid")
    input_receipts = candidate_gate["input_receipts"]
    authorization_index = CANDIDATE_INPUT_RECEIPT_PROVIDERS.index("exact_v2_rerun_authorization")
    admission_index = CANDIDATE_INPUT_RECEIPT_PROVIDERS.index("run_admission")
    history_index = CANDIDATE_INPUT_RECEIPT_PROVIDERS.index("run_history_ledger")
    authorization_receipt = _verified_provider_receipt(input_receipts[authorization_index])
    admission_receipt = _verified_provider_receipt(input_receipts[admission_index])
    history_receipt = _verified_provider_receipt(input_receipts[history_index])
    actual_admission_receipt = _json_artifact_receipt(admission_payload)
    actual_completion_receipt = _json_artifact_receipt(completion_payload)

    if authorization_receipt["artifact_sha256"] != auth_sha256:
        raise ValueError("candidate gate authorization is not bound to the pipeline authorization")
    if authorization_receipt != bundle_payload["authorization_receipt"]:
        raise ValueError("candidate gate and bundle authorization receipts differ")
    if admission_receipt != actual_admission_receipt:
        raise ValueError("candidate gate admission receipt is not bound to admission bytes")
    if bundle_payload["run_admission_receipt"] != actual_admission_receipt:
        raise ValueError("bundle admission receipt is not bound to admission bytes")
    if bundle_payload["static_input_contract"] != admission_payload["static_input_contract"]:
        raise ValueError("bundle static input contract is not bound to admission")
    if not isinstance(history_receipt, Mapping):
        raise ValueError("candidate gate history receipt is invalid")
    if history_receipt["run_identity_receipt"] != actual_completion_receipt:
        raise ValueError("candidate gate run identity is not bound to completion bytes")
    if bundle_payload["run_identity_receipt"] != history_receipt["run_identity_receipt"]:
        raise ValueError("bundle run identity is not bound to the candidate gate")

    result_authorization = result_payload["authorization_receipts"]
    if not isinstance(result_authorization, Mapping):
        raise ValueError("result authorization receipts are invalid")
    if result_authorization["rerun_authorization"] != bundle_payload["authorization_receipt"]:
        raise ValueError("result authorization is not bound to the bundle")
    if result_payload["run_admission_receipt"] != bundle_payload["run_admission_receipt"]:
        raise ValueError("result admission receipt is not bound to the bundle")
    if result_payload["static_input_contract"] != bundle_payload["static_input_contract"]:
        raise ValueError("result static input contract is not bound to the bundle")
    result_history = result_payload["run_history_contract_receipt"]
    if not isinstance(result_history, Mapping):
        raise ValueError("result history contract receipt is invalid")
    if result_history["run_identity_receipt"] != bundle_payload["run_identity_receipt"]:
        raise ValueError("result run identity is not bound to the bundle")
    if result_history["head_receipt"] != bundle_payload["candidate_published_history_head_receipt"]:
        raise ValueError("result history head is not bound to the bundle")
    if result_payload["candidate_member_receipts"] != bundle_payload["ordered_member_receipts"]:
        raise ValueError("result candidate member receipts are not bound to the bundle")


def _verified_provider_receipt(slot: object) -> Mapping[str, object]:
    if not isinstance(slot, Mapping) or slot.get("verification_state") != "verified":
        raise ValueError("pipeline provider slot is not verified")
    receipt = slot.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("pipeline provider receipt is invalid")
    return receipt


def _json_artifact_receipt(payload: Mapping[str, object]) -> dict[str, str]:
    data = (compact_canonical_json(payload) + "\n").encode("utf-8")
    return {
        "artifact_sha256": str(payload["artifact_sha256"]),
        "file_sha256": hashlib.sha256(data).hexdigest(),
    }


__all__ = [
    "encode_member",
    "assemble_candidate_members",
    "run_v2_pipeline",
]
