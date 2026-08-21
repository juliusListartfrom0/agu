"""TASK-0258 Amendment-001 v2 pipeline — candidate, bundle, and post-publication sealing.

Pure builders/sealers for the candidate, candidate-receipt-bundle, and
post-publication segments of the v2 production CLI. Every builder self-validates
its payload before any write; every sealer publishes no-clobber via the atomic
fs primitives.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from app.analysis.task0258_module_a_v2 import (
    CANDIDATE_GATE_SCHEMA_V2,
    CANDIDATE_RECEIPT_BUNDLE_SCHEMA,
    MODULE_ID,
    POSTPUBLICATION_FAILURE_SCHEMA_V2,
    POSTPUBLICATION_VERIFICATION_SCHEMA_V2,
    canonical_artifact_sha256,
    compact_canonical_json,
    is_sha256,
    verify_internal_artifact_hash,
)
from app.analysis.task0258_v2_artifacts import (
    CANDIDATE_MEMBER_PATHS,
    verify_candidate_gate,
    verify_candidate_receipt_bundle,
    verify_generation_member_receipt,
    verify_postpublication_failure,
    verify_postpublication_verification,
)
from app.analysis.task0258_v2_fs import (
    atomic_write_bytes,
    exclusive_flock,
    seal_generation_directory,
    verify_absent,
    verify_generation_directory,
)
from app.analysis.task0258_v2_gate import (
    CANDIDATE_PREPUBLICATION_CHECKS,
    RESULT_ORDERED_CHECKS,
)


def build_candidate_gate_payload(
    *,
    input_receipts: object,
    producer_attempt_chain: object,
    verification_attempt_receipt: object,
    error_bound_result: object,
    candidate_metric_outcome: str,
    conditional_downstream: Mapping[str, bool] | None = None,
) -> dict[str, object]:
    """Build a `agu.vru-causal-temporal-candidate-gate.v2` payload (self-validated)."""
    payload: dict[str, object] = {
        "schema_version": CANDIDATE_GATE_SCHEMA_V2,
        "module_id": MODULE_ID,
        "purpose": "development_diagnostic_only",
        "runtime_consumable": False,
        "training_consumable": False,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
        "input_receipts": input_receipts,
        "producer_attempt_chain": producer_attempt_chain,
        "verification_attempt_receipt": verification_attempt_receipt,
        "ordered_prepublication_check_results": [
            {"check_name": name, "passed": True} for name in CANDIDATE_PREPUBLICATION_CHECKS
        ],
        "error_bound_result": error_bound_result,
        "candidate_metric_outcome": candidate_metric_outcome,
        "publication_state": "not_yet_observed",
        "postpublication_verification_required": True,
        "conditional_downstream": dict(conditional_downstream or {}),
    }
    payload["artifact_sha256"] = canonical_artifact_sha256(payload)
    verify_candidate_gate(payload)
    return payload


def seal_candidate_v2(
    output_root: Path,
    candidate_members: Mapping[str, bytes],
    *,
    flock_path: Path,
) -> Path:
    """Validate exact ten-member coverage and publish ``candidate_v2``."""
    return seal_generation_directory(
        output_root,
        "candidate_v2",
        dict(candidate_members),
        CANDIDATE_MEMBER_PATHS,
        flock_path=flock_path,
    )


def build_member_receipts(candidate_dir: Path) -> list[dict[str, object]]:
    """Compute the ten ``GenerationMemberReceipt`` rows for a published candidate.

    Rows are in literal candidate order; JSONL members are ``file_only``, the
    eight canonical JSON members bind their ``artifact_sha256`` field.
    """
    rows: list[dict[str, object]] = []
    for rel in CANDIDATE_MEMBER_PATHS:
        path = candidate_dir / rel
        data = path.read_bytes()
        file_sha = hashlib.sha256(data).hexdigest()
        if rel.endswith(".jsonl"):
            row: dict[str, object] = {
                "relative_path": rel,
                "receipt_kind": "file_only",
                "size_bytes": len(data),
                "internal_sha256_field": None,
                "artifact_sha256": None,
                "file_sha256": file_sha,
            }
        else:
            payload = json.loads(data)
            if not isinstance(payload, Mapping):
                raise ValueError(f"candidate JSON member is not an object: {rel}")
            verify_internal_artifact_hash(payload)
            row = {
                "relative_path": rel,
                "receipt_kind": "json",
                "size_bytes": len(data),
                "internal_sha256_field": "artifact_sha256",
                "artifact_sha256": payload["artifact_sha256"],
                "file_sha256": file_sha,
            }
        verify_generation_member_receipt(row)
        rows.append(row)
    return rows


def build_candidate_receipt_bundle_payload(
    *,
    authorization_receipt: Mapping[str, object],
    run_identity_receipt: Mapping[str, object],
    run_admission_receipt: Mapping[str, object],
    static_input_contract: Mapping[str, object],
    candidate_published_history_head_receipt: Mapping[str, object],
    candidate_dir: Path,
    observed_at_utc: str,
) -> dict[str, object]:
    """Build a `agu.vru-causal-temporal-candidate-receipt-bundle.v1` payload.

    ``ordered_member_receipts`` is computed from the published ``candidate_v2``
    directory bytes.
    """
    payload: dict[str, object] = {
        "schema_version": CANDIDATE_RECEIPT_BUNDLE_SCHEMA,
        "module_id": MODULE_ID,
        "purpose": "development_diagnostic_only",
        "runtime_consumable": False,
        "training_consumable": False,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
        "authorization_receipt": dict(authorization_receipt),
        "run_identity_receipt": dict(run_identity_receipt),
        "run_admission_receipt": dict(run_admission_receipt),
        "static_input_contract": dict(static_input_contract),
        "candidate_published_history_head_receipt": dict(candidate_published_history_head_receipt),
        "candidate_generation_name": "candidate_v2",
        "ordered_member_receipts": build_member_receipts(candidate_dir),
        "observed_at_utc": observed_at_utc,
    }
    payload["artifact_sha256"] = canonical_artifact_sha256(payload)
    verify_candidate_receipt_bundle(payload)
    return payload


def seal_candidate_receipt_bundle(
    bundle_path: Path,
    bundle_payload: object,
    *,
    candidate_dir: Path,
    output_flock_path: Path,
) -> Path:
    """No-clobber publish the candidate receipt bundle under both phase locks.

    The candidate directory is re-opened and its member receipts are recomputed
    while holding the output-root lock. The bundle transaction uses the fixed
    authorization-bound stage basename from Amendment-001 rather than a random
    caller-independent stage.
    """
    if not isinstance(bundle_payload, Mapping):
        raise ValueError("candidate receipt bundle must be an object")
    verify_candidate_receipt_bundle(bundle_payload)
    if not bundle_path.is_absolute() or not candidate_dir.is_absolute() or not output_flock_path.is_absolute():
        raise ValueError("candidate bundle paths must be absolute")
    if bundle_path.name != "candidate-receipt-bundle.json":
        raise ValueError("candidate receipt bundle basename is not authorized")
    if candidate_dir.name != "candidate_v2":
        raise ValueError("candidate bundle output-root binding is invalid")
    if bundle_path.is_relative_to(candidate_dir.parent):
        raise ValueError("candidate receipt bundle must be outside the output root")
    if output_flock_path.is_relative_to(candidate_dir):
        raise ValueError("candidate output lock must be outside the candidate generation")
    if output_flock_path == bundle_path.parent / ".candidate-receipt-bundle.lock":
        raise ValueError("candidate output and bundle locks must be distinct")
    authorization_receipt = bundle_payload["authorization_receipt"]
    if not isinstance(authorization_receipt, Mapping) or not is_sha256(authorization_receipt["artifact_sha256"]):
        raise ValueError("candidate bundle authorization receipt is invalid")
    stage_path = bundle_path.parent / (
        f".{authorization_receipt['artifact_sha256']}.{bundle_path.name}.task0258-bundle-stage"
    )
    lock_path = bundle_path.parent / ".candidate-receipt-bundle.lock"
    bundle_bytes = (compact_canonical_json(bundle_payload) + "\n").encode("utf-8")
    with exclusive_flock(output_flock_path):
        with exclusive_flock(lock_path):
            verify_generation_directory(candidate_dir, CANDIDATE_MEMBER_PATHS)
            actual_member_receipts = build_member_receipts(candidate_dir)
            if list(bundle_payload["ordered_member_receipts"]) != actual_member_receipts:
                raise ValueError("candidate receipt bundle is not bound to the locked candidate bytes")
            verify_absent(bundle_path)
            verify_absent(stage_path)
            atomic_write_bytes(bundle_path, bundle_bytes, mode=0o600, stage_path=stage_path)
            reopened = bundle_path.read_bytes()
            if reopened != bundle_bytes:
                raise ValueError("candidate receipt bundle changed during publication")
            if stage_path.exists() or stage_path.is_symlink():
                raise ValueError("candidate receipt bundle stage remained after publication")
    return bundle_path


def build_postpublication_verification_payload(
    *,
    error_bounds_pass: bool,
    authorization_receipts: object,
    run_history_contract_receipt: object,
    run_admission_receipt: Mapping[str, object],
    static_input_contract: object,
    candidate_receipt_bundle_receipt: Mapping[str, object],
    candidate_member_receipts: object,
    prior_attempt_receipts: object,
    producer_embedding_receipt: object,
    verification_embedding_receipt: object,
    verification_attempt_receipt: object,
    error_bound_result: object,
    evaluator_receipts: object,
    publication_observation: Mapping[str, object] | None = None,
    conditional_downstream: Mapping[str, bool] | None = None,
) -> dict[str, object]:
    """Build a `agu.vru-causal-temporal-postpublication-verification.v2` payload.

    The final ``frozen_error_bounds`` check is ``error_bounds_pass``; the
    decision/stop_reason follow from it (``mechanical_pass``/null or
    ``temporal-hypothesis-rejected``/same).
    """
    checks = [{"check_name": name, "passed": True} for name in RESULT_ORDERED_CHECKS[:-1]]
    checks.append({"check_name": "frozen_error_bounds", "passed": error_bounds_pass})
    payload: dict[str, object] = {
        "schema_version": POSTPUBLICATION_VERIFICATION_SCHEMA_V2,
        "module_id": MODULE_ID,
        "purpose": "development_diagnostic_only",
        "runtime_consumable": False,
        "training_consumable": False,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
        "candidate_generation_name": "candidate_v2",
        "authorization_receipts": authorization_receipts,
        "run_history_contract_receipt": run_history_contract_receipt,
        "run_admission_receipt": dict(run_admission_receipt),
        "static_input_contract": static_input_contract,
        "candidate_receipt_bundle_receipt": dict(candidate_receipt_bundle_receipt),
        "candidate_member_receipts": candidate_member_receipts,
        "prior_attempt_receipts": prior_attempt_receipts,
        "producer_embedding_receipt": producer_embedding_receipt,
        "verification_embedding_receipt": verification_embedding_receipt,
        "verification_attempt_receipt": verification_attempt_receipt,
        "ordered_check_results": checks,
        "error_bound_result": error_bound_result,
        "decision": "mechanical_pass" if error_bounds_pass else "temporal-hypothesis-rejected",
        "stop_reason": None if error_bounds_pass else "temporal-hypothesis-rejected",
        "evaluator_receipts": evaluator_receipts,
        "publication_observation": dict(
            publication_observation
            or {
                "candidate_visible": True,
                "result_publication_state": "not_yet_observed",
                "bound_active_stage_is_only_stage": True,
            }
        ),
        "conditional_downstream": dict(conditional_downstream or {}),
    }
    payload["artifact_sha256"] = canonical_artifact_sha256(payload)
    verify_postpublication_verification(payload)
    return payload


def seal_verified_result(
    output_root: Path,
    registry_payload: object,
    *,
    flock_path: Path,
) -> Path:
    """No-clobber publish ``verified_result_v2/verification_registry.json``."""
    if not isinstance(registry_payload, Mapping):
        raise ValueError("verified result must be an object")
    verify_postpublication_verification(registry_payload)
    authorization_receipts = registry_payload["authorization_receipts"]
    if not isinstance(authorization_receipts, Mapping) or not isinstance(
        authorization_receipts["rerun_authorization"], Mapping
    ):
        raise ValueError("verified result rerun authorization receipt is invalid")
    authorization_sha256 = authorization_receipts["rerun_authorization"]["artifact_sha256"]
    if not is_sha256(authorization_sha256):
        raise ValueError("verified result rerun authorization SHA is invalid")
    registry_bytes = (compact_canonical_json(registry_payload) + "\n").encode("utf-8")
    return seal_generation_directory(
        output_root,
        "verified_result_v2",
        {"verification_registry.json": registry_bytes},
        ("verification_registry.json",),
        flock_path=flock_path,
        stage_name=f".{authorization_sha256}.verified-result-v2-stage",
    )


def build_postpublication_failure_payload(
    *,
    failed_check_name: str,
    dependency_provider_slots: object,
    conditional_downstream: Mapping[str, bool] | None = None,
) -> dict[str, object]:
    """Build a `agu.vru-causal-temporal-postpublication-failure.v2` payload."""
    state = "limit_failure" if failed_check_name == "global_resource_caps" else "computation_mismatch"
    payload: dict[str, object] = {
        "schema_version": POSTPUBLICATION_FAILURE_SCHEMA_V2,
        "module_id": MODULE_ID,
        "purpose": "development_diagnostic_only",
        "runtime_consumable": False,
        "training_consumable": False,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
        "dependency_provider_slots": dependency_provider_slots,
        "failed_check_name": failed_check_name,
        "observed_result": {
            "subject_kind": "check",
            "subject": failed_check_name,
            "observation_state": state,
        },
        "decision": "mechanical_failure",
        "stop_reason": failed_check_name,
        "final_result_published": False,
        "conditional_downstream": dict(conditional_downstream or {}),
    }
    payload["artifact_sha256"] = canonical_artifact_sha256(payload)
    verify_postpublication_failure(payload)
    return payload


def seal_postverification_failure(
    output_root: Path,
    failure_payload: object,
    *,
    flock_path: Path,
) -> Path:
    """No-clobber publish ``postverification_failure_v2/failure.json``."""
    if not isinstance(failure_payload, Mapping):
        raise ValueError("postverification failure must be an object")
    verify_postpublication_failure(failure_payload)
    dependency_slots = failure_payload["dependency_provider_slots"]
    if not isinstance(dependency_slots, (list, tuple)) or len(dependency_slots) < 4:
        raise ValueError("postverification failure authorization slot is missing")
    authorization_slot = dependency_slots[3]
    if (
        not isinstance(authorization_slot, Mapping)
        or authorization_slot.get("provider") != "exact_v2_rerun_authorization"
    ):
        raise ValueError("postverification failure authorization slot is invalid")
    authorization_receipt = authorization_slot.get("receipt")
    if not isinstance(authorization_receipt, Mapping):
        raise ValueError("postverification failure authorization receipt is invalid")
    authorization_sha256 = authorization_receipt.get("artifact_sha256")
    if not is_sha256(authorization_sha256):
        raise ValueError("postverification failure authorization SHA is invalid")

    failure_bytes = (compact_canonical_json(failure_payload) + "\n").encode("utf-8")
    return seal_generation_directory(
        output_root,
        "postverification_failure_v2",
        {"failure.json": failure_bytes},
        ("failure.json",),
        flock_path=flock_path,
        stage_name=f".{authorization_sha256}.postverification-failure-v2-stage",
    )


__all__ = [
    "build_candidate_gate_payload",
    "seal_candidate_v2",
    "build_member_receipts",
    "build_candidate_receipt_bundle_payload",
    "seal_candidate_receipt_bundle",
    "build_postpublication_verification_payload",
    "seal_verified_result",
    "build_postpublication_failure_payload",
    "seal_postverification_failure",
]
