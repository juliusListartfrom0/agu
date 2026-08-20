"""TASK-0258 Amendment-001 v2 closed-evidence layer (receipt primitives).

This module defines the v2 schema versions and the exact receipt shapes that
every amended-run artifact binds. It is the shared foundation for the v2
producer/verification attempt, resume, embedding, candidate, failure, and result
artifacts. It reuses the parent ``vru_causal_temporal_retrospective``
computation contract (which is unchanged by the amendment) but never mutates it;
v2 artifacts are not interchangeable with v1.

Receipt shapes (amendment-001 §"Exact nested receipt vocabulary"):

- ``ArtifactFileReceipt`` is exactly ``{artifact_sha256, file_sha256}``.
- ``HistoryArtifactReceipt`` is the same shape restricted to run-history nodes.
- ``authorization_receipts`` is exactly the four-key object with canonical key
  order ``parent_spec_approval``, ``amendment_implementation_approval``,
  ``amended_implementation_review``, ``rerun_authorization``; each value is an
  ``ArtifactFileReceipt``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

MODULE_ID = "existing-45-temporal-retrospective"

# v2 schema versions (amendment-001). Parent v1 versions stay distinct and are
# never promoted.
TILED_SWIN_EMBEDDING_SCHEMA_V2 = "agu.vru-causal-tiled-swin-embeddings.v2"
TILED_SWIN_ATTEMPT_SCHEMA_V2 = "agu.vru-causal-tiled-swin-attempt.v2"
TILED_SWIN_RESUME_SCHEMA_V2 = "agu.vru-causal-tiled-swin-embeddings-resume.v2"
VERIFICATION_EMBEDDING_SCHEMA_V2 = "agu.vru-causal-tiled-swin-verification-embeddings.v2"
VERIFICATION_ATTEMPT_SCHEMA_V2 = "agu.vru-causal-tiled-swin-verification-attempt.v2"
VERIFICATION_RESOURCE_SAMPLE_SCHEMA = "agu.vru-causal-verification-resource-sample.v1"
CANDIDATE_GATE_SCHEMA_V2 = "agu.vru-causal-temporal-candidate-gate.v2"
MECHANICAL_FAILURE_SCHEMA_V2 = "agu.vru-causal-temporal-mechanical-failure.v2"
CANDIDATE_RECEIPT_BUNDLE_SCHEMA = "agu.vru-causal-temporal-candidate-receipt-bundle.v1"
POSTPUBLICATION_VERIFICATION_SCHEMA_V2 = "agu.vru-causal-temporal-postpublication-verification.v2"
POSTPUBLICATION_FAILURE_SCHEMA_V2 = "agu.vru-causal-temporal-postpublication-failure.v2"
RERUN_AUTHORIZATION_SCHEMA = "agu.task0258-module-a-v2-rerun-authorization.v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_RFC3339_RE = re.compile(r"\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})\Z")

# Canonical provider order for ``authorization_receipts`` (exact key set/order).
AUTHORIZATION_PROVIDER_ORDER: tuple[str, ...] = (
    "parent_spec_approval",
    "amendment_implementation_approval",
    "amended_implementation_review",
    "rerun_authorization",
)

# The four exact receipt fields every v2 artifact adds to its parent v1 allowlist.
V2_RECEIPT_FIELDS = frozenset(
    {
        "authorization_receipts",
        "run_identity_receipt",
        "history_head_receipt",
        "run_admission_receipt",
    }
)

# Extra closed objects carried only by producer/verification attempt records.
ATTEMPT_WORKER_FIELDS = frozenset(
    {
        "worker_request",
        "child_observation",
        "worker_payload",
        "read_isolation_policy",
        "read_isolation_attestation",
    }
)


def is_sha256(value: object) -> bool:
    """Return True when ``value`` is a lowercase 64-hex SHA-256 string."""
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def is_safe_slug(value: object) -> bool:
    """Return True when ``value`` is a nonempty ``[A-Za-z0-9_-]`` slug."""
    return isinstance(value, str) and bool(value) and value.replace("-", "").replace("_", "").isalnum()


def is_rfc3339(value: object) -> bool:
    """Return True when ``value`` is a UTC/offset RFC3339 timestamp."""
    return isinstance(value, str) and _RFC3339_RE.fullmatch(value) is not None


def verify_artifact_file_receipt(value: object) -> None:
    """Validate an exact ``ArtifactFileReceipt`` (``{artifact_sha256,file_sha256}``)."""
    if not isinstance(value, Mapping) or set(value) != {"artifact_sha256", "file_sha256"}:
        raise ValueError("ArtifactFileReceipt shape is invalid")
    if not is_sha256(value["artifact_sha256"]) or not is_sha256(value["file_sha256"]):
        raise ValueError("ArtifactFileReceipt hashes are invalid")


def verify_history_artifact_receipt(value: object) -> None:
    """Validate a run-history ``HistoryArtifactReceipt`` (same shape)."""
    verify_artifact_file_receipt(value)


def verify_authorization_receipts(value: object) -> None:
    """Validate the exact four-key ``authorization_receipts`` object.

    Key set and canonical order must equal
    ``(parent_spec_approval, amendment_implementation_approval,
    amended_implementation_review, rerun_authorization)``; every value is an
    ``ArtifactFileReceipt``. A self-authored or reordered object is rejected.
    """
    if not isinstance(value, Mapping):
        raise ValueError("authorization_receipts is not an object")
    if tuple(value) != AUTHORIZATION_PROVIDER_ORDER:
        raise ValueError("authorization_receipts key set/order is invalid")
    for key in AUTHORIZATION_PROVIDER_ORDER:
        verify_artifact_file_receipt(value[key])


def verify_v2_receipt_fields(payload: Mapping[str, object]) -> None:
    """Validate the four shared v2 receipt fields present in every v2 artifact.

    ``authorization_receipts`` uses :func:`verify_authorization_receipts`;
    ``run_identity_receipt``, ``history_head_receipt``, and
    ``run_admission_receipt`` each use the exact ``ArtifactFileReceipt`` shape.
    """
    verify_authorization_receipts(payload["authorization_receipts"])
    verify_history_artifact_receipt(payload["run_identity_receipt"])
    verify_history_artifact_receipt(payload["history_head_receipt"])
    verify_artifact_file_receipt(payload["run_admission_receipt"])


def _reject_missing_receipt_fields(payload: Mapping[str, object]) -> None:
    missing = V2_RECEIPT_FIELDS - set(payload)
    if missing:
        raise ValueError(f"v2 artifact missing receipt fields: {sorted(missing)}")


# The exact v2 producer-embedding allowlist: the parent v1 embedding allowlist
# plus the four shared v2 receipt fields. Mirror of parent `_EMBEDDING_FIELDS`.
V2_PRODUCER_EMBEDDING_FIELDS = (
    frozenset(
        {
            "artifact_sha256",
            "attempt_chain",
            "examples",
            "formal_evaluation_eligible",
            "module_id",
            "plan_receipt",
            "promoted",
            "promotion_eligible",
            "producer_environment",
            "purpose",
            "representation",
            "row_count",
            "runtime_consumable",
            "schema_version",
            "task0257_input_receipts",
            "training_consumable",
        }
    )
    | V2_RECEIPT_FIELDS
)

# The exact v2 verification-embedding allowlist (amendment-001 §"Verification
# embedding"): role-bound, no attempt_chain, with checkpoint/source-video receipts.
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

# Common false-eligibility fields shared by every v2 artifact.
_COMMON_FALSE_FIELDS = frozenset(
    {
        "schema_version",
        "module_id",
        "purpose",
        "runtime_consumable",
        "training_consumable",
        "formal_evaluation_eligible",
        "promotion_eligible",
        "promoted",
    }
)

# v2 producer attempt allowlist: parent `_ATTEMPT_FIELDS` + the four receipt
# fields + the five worker-isolation objects (amendment §"Acyclic producer and
# verification artifacts").
V2_PRODUCER_ATTEMPT_FIELDS = (
    _COMMON_FALSE_FIELDS
    | V2_RECEIPT_FIELDS
    | ATTEMPT_WORKER_FIELDS
    | frozenset(
        {
            "attempt_ordinal",
            "prior_attempt_receipt",
            "plan_receipt",
            "task0257_input_receipts",
            "started_prefix_count",
            "completed_prefix_count",
            "new_rows_verified",
            "cumulative_active_runtime_nanoseconds",
            "cumulative_resource_samples",
            "cumulative_resource_log_bytes",
            "resource_log_receipt",
            "resume_input_cas",
            "resume_output_cas",
            "received_signal",
            "disposition",
            "stop_reason",
            "artifact_sha256",
        }
    )
)

# v2 producer resume allowlist: parent `_RESUME_FIELDS` + the four receipt
# fields (no worker-isolation objects on resume).
V2_PRODUCER_RESUME_FIELDS = (
    _COMMON_FALSE_FIELDS
    | V2_RECEIPT_FIELDS
    | frozenset(
        {
            "artifact_sha256",
            "attempt_chain_receipts",
            "checkpoint_receipt",
            "completed_count",
            "completed_examples",
            "plan_receipt",
            "producer_environment",
            "representation",
            "row_count",
            "source_video_receipts",
            "task0257_input_receipts",
        }
    )
)

# v2 verification attempt allowlist (amendment §"Verification attempt").
VERIFICATION_ATTEMPT_FIELDS = _COMMON_FALSE_FIELDS | frozenset(
    {
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


def verify_exact_field_set(payload: Mapping[str, object], allowlist: frozenset[str]) -> None:
    """Raise when ``payload``'s key set is not exactly ``allowlist``.

    Every v2 artifact is closed: missing, extra, or renamed fields fail.
    """
    if set(payload) != allowlist:
        raise ValueError(
            f"artifact field set mismatch: "
            f"missing={sorted(allowlist - set(payload))} "
            f"extra={sorted(set(payload) - allowlist)}"
        )


def _verify_file_receipt(value: object) -> None:
    """Validate the parent ``FileReceipt`` ``{size_bytes, file_sha256}``."""
    if not isinstance(value, Mapping) or set(value) != {"size_bytes", "file_sha256"}:
        raise ValueError("FileReceipt shape is invalid")
    size = value["size_bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError("FileReceipt size_bytes is invalid")
    if not is_sha256(value["file_sha256"]):
        raise ValueError("FileReceipt file_sha256 is invalid")


def _verify_run_history_contract_receipt(value: object) -> None:
    """Validate ``RunHistoryContractReceipt`` ``{run_identity_receipt, head_receipt, marker_count}``."""
    if not isinstance(value, Mapping) or set(value) != {
        "run_identity_receipt",
        "head_receipt",
        "marker_count",
    }:
        raise ValueError("RunHistoryContractReceipt shape is invalid")
    verify_artifact_file_receipt(value["run_identity_receipt"])
    verify_artifact_file_receipt(value["head_receipt"])
    count = value["marker_count"]
    if not isinstance(count, int) or isinstance(count, bool) or not (0 <= count <= 10):
        raise ValueError("RunHistoryContractReceipt marker_count is invalid")


def verify_provider_slot(value: object, *, receipt_kind: str = "artifact") -> None:
    """Validate an exact provider slot ``{provider, verification_state, receipt}``.

    ``verification_state`` is ``verified``, ``failed``, or ``not_reached``. Only
    ``verified`` carries a non-null receipt; the other states carry ``null``.

    ``receipt_kind`` selects the role-fixed receipt shape for a ``verified`` slot:
    ``artifact`` (``ArtifactFileReceipt``), ``run_history``
    (``RunHistoryContractReceipt``), ``static_inputs``
    (``StaticInputContractReceipt``), ``jsonl`` (parent ``FileReceipt``), or
    ``generic`` (non-null only; the caller validates the role-specific shape).
    """
    if not isinstance(value, Mapping) or set(value) != {"provider", "verification_state", "receipt"}:
        raise ValueError("provider slot shape is invalid")
    if not is_safe_slug(value["provider"]):
        raise ValueError("provider slot provider is invalid")
    state = value["verification_state"]
    if state not in ("verified", "failed", "not_reached"):
        raise ValueError("provider slot verification_state is invalid")
    receipt = value["receipt"]
    if state == "verified":
        if receipt is None:
            raise ValueError("provider slot verified receipt must be non-null")
        if receipt_kind == "artifact":
            verify_artifact_file_receipt(receipt)
        elif receipt_kind == "run_history":
            _verify_run_history_contract_receipt(receipt)
        elif receipt_kind == "static_inputs":
            verify_static_input_contract(receipt)
        elif receipt_kind == "jsonl":
            _verify_file_receipt(receipt)
        elif receipt_kind == "generic":
            pass
        else:
            raise ValueError(f"unknown provider slot receipt_kind: {receipt_kind!r}")
    elif receipt is not None:
        raise ValueError("provider slot receipt must be null when not verified")


def verify_provider_slots(
    value: object,
    provider_order: tuple[str, ...],
    *,
    receipt_kinds: Mapping[str, str] | None = None,
) -> None:
    """Validate an exact ordered provider-slot list against ``provider_order``.

    ``receipt_kinds`` maps a provider name to its role-fixed ``receipt_kind``
    (default ``artifact``).
    """
    if not isinstance(value, (list, tuple)) or len(value) != len(provider_order):
        raise ValueError("provider slot list length/type is invalid")
    for slot, expected in zip(value, provider_order):
        kind = (receipt_kinds or {}).get(expected, "artifact")
        verify_provider_slot(slot, receipt_kind=kind)
        if slot["provider"] != expected:
            raise ValueError(f"provider slot order mismatch: {slot['provider']!r} != {expected!r}")


def verify_static_input_contract(value: object) -> None:
    """Validate the exact ``StaticInputContractReceipt`` shape.

    ``{temporal_plan_artifact_sha256, temporal_plan_file_sha256,
    task0257_receipts_projection_sha256}`` — three lowercase SHA-256 strings,
    no path, no artifact SHA for the opaque wrapper.
    """
    expected = {
        "temporal_plan_artifact_sha256",
        "temporal_plan_file_sha256",
        "task0257_receipts_projection_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("StaticInputContractReceipt shape is invalid")
    for key in expected:
        if not is_sha256(value[key]):
            raise ValueError(f"StaticInputContractReceipt {key} is invalid")


def compact_canonical_json(value: object) -> str:
    """Return the compact-canonical JSON of ``value``.

    Sorted object keys, ``(',', ':')`` separators, no whitespace, no trailing LF.
    This is the byte source for ``artifact_sha256`` (the internal hash excludes
    the ``artifact_sha256`` field itself, computed by the caller).
    """
    import json as _json

    def _canon(v: object) -> str:
        if isinstance(v, dict):
            return (
                "{"
                + ",".join(
                    f"{_json.dumps(k, separators=(',', ':'))}:{_canon(x)}"
                    for k, x in sorted(v.items(), key=lambda kv: _json.dumps(kv[0], separators=(",", ":")))
                )
                + "}"
            )
        if isinstance(v, (list, tuple)):
            return "[" + ",".join(_canon(x) for x in v) + "]"
        if isinstance(v, bool):
            return "true" if v else "false"
        if v is None:
            return "null"
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, str):
            return _json.dumps(v, separators=(",", ":"))
        raise TypeError(type(v))

    return _canon(value)


def canonical_artifact_sha256(value: object) -> str:
    """Return SHA-256 of the compact-canonical JSON (no trailing LF)."""
    import hashlib

    return hashlib.sha256(compact_canonical_json(value).encode("utf-8")).hexdigest()


__all__ = [
    "MODULE_ID",
    "TILED_SWIN_EMBEDDING_SCHEMA_V2",
    "TILED_SWIN_ATTEMPT_SCHEMA_V2",
    "TILED_SWIN_RESUME_SCHEMA_V2",
    "VERIFICATION_EMBEDDING_SCHEMA_V2",
    "VERIFICATION_ATTEMPT_SCHEMA_V2",
    "VERIFICATION_RESOURCE_SAMPLE_SCHEMA",
    "CANDIDATE_GATE_SCHEMA_V2",
    "MECHANICAL_FAILURE_SCHEMA_V2",
    "CANDIDATE_RECEIPT_BUNDLE_SCHEMA",
    "POSTPUBLICATION_VERIFICATION_SCHEMA_V2",
    "POSTPUBLICATION_FAILURE_SCHEMA_V2",
    "RERUN_AUTHORIZATION_SCHEMA",
    "AUTHORIZATION_PROVIDER_ORDER",
    "V2_RECEIPT_FIELDS",
    "ATTEMPT_WORKER_FIELDS",
    "V2_PRODUCER_EMBEDDING_FIELDS",
    "VERIFICATION_EMBEDDING_FIELDS",
    "V2_PRODUCER_ATTEMPT_FIELDS",
    "V2_PRODUCER_RESUME_FIELDS",
    "VERIFICATION_ATTEMPT_FIELDS",
    "is_sha256",
    "is_safe_slug",
    "is_rfc3339",
    "verify_artifact_file_receipt",
    "verify_history_artifact_receipt",
    "verify_authorization_receipts",
    "verify_v2_receipt_fields",
    "verify_exact_field_set",
    "verify_provider_slot",
    "verify_provider_slots",
    "verify_static_input_contract",
    "compact_canonical_json",
    "canonical_artifact_sha256",
]
