"""TASK-0258 Amendment-001 v2 verification worker — artifact schemas.

Closed, fail-closed validators for the second empty-state extraction's three
artifacts: the verification embedding, the verification attempt, and the
verification resource-sample row (amendment §"Verification embedding",
§"Verification attempt", §"Shared resource and disk limits").
"""

from __future__ import annotations

import struct
from collections.abc import Mapping

from app.analysis.task0258_module_a_v2 import (
    VERIFICATION_ATTEMPT_SCHEMA_V2,
    VERIFICATION_EMBEDDING_SCHEMA_V2,
    VERIFICATION_RESOURCE_SAMPLE_SCHEMA,
    canonical_artifact_sha256,
    compact_canonical_json,
    is_sha256,
    verify_artifact_file_receipt,
    verify_exact_field_set,
    verify_internal_artifact_hash,
    verify_provider_slot,
    verify_v2_receipt_fields,
)
from app.analysis.task0258_v2_artifacts import verify_common_false_fields
from app.analysis.task0258_v2_read_isolation import (
    verify_read_isolation_binding,
)

ROLE = "independent_empty_state_verification"


def _float32(value: object) -> float:
    """Round a float to its IEEE-754 float32 value (round-to-nearest-even)."""
    return float(struct.unpack("!f", struct.pack("!f", float(value)))[0])


def normalize_float32_leaf(value: object) -> object:
    """Recursively normalize every float leaf to its float32 representation."""
    if isinstance(value, float):
        return _float32(value)
    if isinstance(value, (list, tuple)):
        return [normalize_float32_leaf(v) for v in value]
    if isinstance(value, Mapping):
        return {k: normalize_float32_leaf(v) for k, v in value.items()}
    return value


def compute_verification_projection(
    representation: str,
    examples: object,
) -> str:
    """Compute ``computational_projection_sha256`` (amendment §"Verification attempt").

    SHA-256 of the compact-canonical JSON
    ``{"representation": NAME, "ordered_examples": ROWS}`` where every float leaf
    in ``ROWS`` is first normalized to its float32 representation. ``ROWS`` is the
    45-element plan order; each row has exactly ``ordinal``, ``key``,
    ``tile_frame_indexes``, ``derivation_only_tile_embeddings``, ``model_input``.
    """
    payload = {
        "representation": representation,
        "ordered_examples": normalize_float32_leaf(examples),
    }
    return __import__("hashlib").sha256(compact_canonical_json(payload).encode("utf-8")).hexdigest()


def build_verification_embedding_payload(
    *,
    authorization_receipts: Mapping[str, object],
    run_identity_receipt: Mapping[str, object],
    history_head_receipt: Mapping[str, object],
    run_admission_receipt: Mapping[str, object],
    plan_receipt: Mapping[str, object],
    task0257_input_receipts: object,
    representation: str,
    producer_environment: object,
    checkpoint_receipt: Mapping[str, object],
    source_video_receipts: object,
    examples: object,
) -> dict[str, object]:
    """Build a `agu.vru-causal-tiled-swin-verification-embeddings.v2` payload.

    Normalizes every example float to float32, pins ``role`` and ``row_count=45``,
    computes ``artifact_sha256`` canonically, and self-validates.
    """
    normalized_examples = normalize_float32_leaf(examples)
    payload: dict[str, object] = {
        "schema_version": VERIFICATION_EMBEDDING_SCHEMA_V2,
        "module_id": "existing-45-temporal-retrospective",
        "purpose": "development_diagnostic_only",
        "runtime_consumable": False,
        "training_consumable": False,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
        "role": ROLE,
        "authorization_receipts": dict(authorization_receipts),
        "run_identity_receipt": dict(run_identity_receipt),
        "history_head_receipt": dict(history_head_receipt),
        "run_admission_receipt": dict(run_admission_receipt),
        "plan_receipt": dict(plan_receipt),
        "task0257_input_receipts": task0257_input_receipts,
        "representation": representation,
        "producer_environment": producer_environment,
        "checkpoint_receipt": dict(checkpoint_receipt),
        "source_video_receipts": source_video_receipts,
        "row_count": len(normalized_examples),
        "examples": normalized_examples,
    }
    payload["artifact_sha256"] = canonical_artifact_sha256(payload)
    verify_verification_embedding(payload)
    return payload


VERIFICATION_EMBEDDING_FIELDS = frozenset(
    {
        "schema_version",
        "module_id",
        "purpose",
        "runtime_consumable",
        "training_consumable",
        "formal_evaluation_eligible",
        "promotion_eligible",
        "promoted",
        "role",
        "authorization_receipts",
        "run_identity_receipt",
        "history_head_receipt",
        "run_admission_receipt",
        "plan_receipt",
        "task0257_input_receipts",
        "representation",
        "producer_environment",
        "checkpoint_receipt",
        "source_video_receipts",
        "row_count",
        "examples",
        "artifact_sha256",
    }
)

VERIFICATION_ATTEMPT_FIELDS = frozenset(
    {
        "schema_version",
        "module_id",
        "purpose",
        "runtime_consumable",
        "training_consumable",
        "formal_evaluation_eligible",
        "promotion_eligible",
        "promoted",
        "verification_ordinal",
        "authorization_receipts",
        "run_identity_receipt",
        "history_head_receipt",
        "run_admission_receipt",
        "plan_receipt",
        "task0257_input_receipts",
        "checkpoint_receipt",
        "source_video_receipts",
        "producer_attempt_chain_receipts",
        "producer_embedding_receipt",
        "worker_request",
        "child_observation",
        "worker_payload",
        "read_isolation_policy",
        "read_isolation_attestation",
        "verification_embedding_slot",
        "resource_log_receipt",
        "started_cumulative_active_runtime_nanoseconds",
        "ended_cumulative_active_runtime_nanoseconds",
        "started_cumulative_resource_samples",
        "ended_cumulative_resource_samples",
        "started_cumulative_resource_log_bytes",
        "ended_cumulative_resource_log_bytes",
        "computational_projection_sha256",
        "received_signal",
        "disposition",
        "stop_reason",
        "artifact_sha256",
    }
)

VERIFICATION_RESOURCE_SAMPLE_FIELDS = frozenset(
    {
        "schema_version",
        "attempt_role",
        "verification_ordinal",
        "attempt_sample_ordinal",
        "cumulative_sample_ordinal",
        "scheduled_cumulative_active_seconds",
        "observed_attempt_active_nanoseconds",
        "system_memory_percent",
        "available_memory_bytes",
        "free_swap_bytes",
        "system_cpu_percent",
        "consecutive_breach_count",
        "breached_limits",
    }
)

# Inner terminal stop reasons for a verification attempt with a null signal.
_INNER_STOP_REASONS = frozenset(
    {
        "receipt_failure",
        "schema_failure",
        "input_identity_failure",
        "disk_failure",
        "resource_breach",
        "runtime_cap",
        "sample_cap",
        "log_byte_cap",
        "decode_failure",
        "model_failure",
        "determinism_failure",
        "publication_failure",
    }
)

_SIGNAL_TO_REASON = {
    "SIGINT": "external_sigint",
    "SIGTERM": "external_sigterm",
    "SIGHUP": "unsupported_signal",
    "SIGQUIT": "unsupported_signal",
}


def verify_verification_resource_sample(row: Mapping[str, object]) -> None:
    """Validate a ``agu.vru-causal-verification-resource-sample.v1`` row."""
    verify_exact_field_set(row, VERIFICATION_RESOURCE_SAMPLE_FIELDS)
    if row["schema_version"] != VERIFICATION_RESOURCE_SAMPLE_SCHEMA:
        raise ValueError("verification resource sample schema_version is invalid")
    if row["attempt_role"] != ROLE or row["verification_ordinal"] != 1:
        raise ValueError("verification resource sample role/ordinal is invalid")
    for field in (
        "attempt_sample_ordinal",
        "cumulative_sample_ordinal",
        "scheduled_cumulative_active_seconds",
        "observed_attempt_active_nanoseconds",
        "available_memory_bytes",
        "free_swap_bytes",
        "consecutive_breach_count",
    ):
        value = row[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"verification resource sample {field} is invalid")
    for field in ("system_memory_percent", "system_cpu_percent"):
        value = row[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"verification resource sample {field} is invalid")
    if not isinstance(row["breached_limits"], (list, tuple)):
        raise ValueError("verification resource sample breached_limits must be a list")


def verify_verification_embedding(payload: Mapping[str, object]) -> None:
    """Validate a ``agu.vru-causal-tiled-swin-verification-embeddings.v2`` payload."""
    verify_exact_field_set(payload, VERIFICATION_EMBEDDING_FIELDS)
    verify_common_false_fields(payload, schema_version=VERIFICATION_EMBEDDING_SCHEMA_V2)
    verify_internal_artifact_hash(payload)
    if payload["role"] != ROLE:
        raise ValueError("verification embedding role is invalid")
    if payload["row_count"] != 45:
        raise ValueError("verification embedding row_count must be 45")
    verify_v2_receipt_fields(payload)
    verify_artifact_file_receipt(payload["plan_receipt"])
    verify_artifact_file_receipt(payload["checkpoint_receipt"])
    if not isinstance(payload["source_video_receipts"], (list, tuple)):
        raise ValueError("verification embedding source_video_receipts must be a list")


def _verify_failure_pairing(signal: object, stop_reason: object) -> None:
    if signal is None:
        if stop_reason not in _INNER_STOP_REASONS:
            raise ValueError("verification attempt null-signal stop_reason is invalid")
        return
    if not isinstance(signal, str) or signal not in _SIGNAL_TO_REASON:
        raise ValueError("verification attempt received_signal is invalid")
    if stop_reason != _SIGNAL_TO_REASON[signal]:
        raise ValueError("verification attempt signal/reason pairing is invalid")


def verify_verification_attempt(payload: Mapping[str, object]) -> None:
    """Validate a ``agu.vru-causal-tiled-swin-verification-attempt.v2`` payload."""
    verify_exact_field_set(payload, VERIFICATION_ATTEMPT_FIELDS)
    verify_common_false_fields(payload, schema_version=VERIFICATION_ATTEMPT_SCHEMA_V2)
    verify_internal_artifact_hash(payload)
    if payload["verification_ordinal"] != 1:
        raise ValueError("verification attempt ordinal must be 1")
    verify_v2_receipt_fields(payload)
    verify_read_isolation_binding(payload["read_isolation_policy"], payload["read_isolation_attestation"])
    slot = payload["verification_embedding_slot"]
    verify_provider_slot(slot)
    if slot["provider"] != "verification_tiled_swin_embeddings":
        raise ValueError("verification attempt embedding slot provider is invalid")
    disposition = payload["disposition"]
    if disposition == "completed":
        if payload["received_signal"] is not None or payload["stop_reason"] is not None:
            raise ValueError("completed verification attempt must have null signal/reason")
        if slot["verification_state"] != "verified":
            raise ValueError("completed verification attempt requires a verified embedding slot")
    elif disposition == "terminal_failure":
        _verify_failure_pairing(payload["received_signal"], payload["stop_reason"])
    else:
        raise ValueError(f"verification attempt disposition is invalid: {disposition!r}")
    projection = payload["computational_projection_sha256"]
    if projection is not None and not is_sha256(projection):
        raise ValueError("verification attempt projection is invalid")


__all__ = [
    "ROLE",
    "VERIFICATION_EMBEDDING_FIELDS",
    "VERIFICATION_ATTEMPT_FIELDS",
    "VERIFICATION_RESOURCE_SAMPLE_FIELDS",
    "verify_verification_resource_sample",
    "verify_verification_embedding",
    "verify_verification_attempt",
    "normalize_float32_leaf",
    "compute_verification_projection",
    "build_verification_embedding_payload",
]
