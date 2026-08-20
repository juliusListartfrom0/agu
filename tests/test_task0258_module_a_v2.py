"""Tests for the TASK-0258 Amendment-001 v2 receipt primitives."""

from __future__ import annotations

import pytest

from app.analysis.task0258_module_a_v2 import (
    AUTHORIZATION_PROVIDER_ORDER,
    V2_PRODUCER_ATTEMPT_FIELDS,
    V2_PRODUCER_EMBEDDING_FIELDS,
    V2_PRODUCER_RESUME_FIELDS,
    V2_RECEIPT_FIELDS,
    VERIFICATION_ATTEMPT_FIELDS,
    VERIFICATION_EMBEDDING_FIELDS,
    is_sha256,
    verify_artifact_file_receipt,
    verify_authorization_receipts,
    verify_exact_field_set,
    verify_history_artifact_receipt,
    verify_provider_slot,
    verify_provider_slots,
    verify_static_input_contract,
    verify_v2_receipt_fields,
)


def _receipt():
    return {"artifact_sha256": "0" * 64, "file_sha256": "f" * 64}


def test_is_sha256():
    assert is_sha256("0" * 64)
    assert is_sha256("a1b2" * 16)
    assert not is_sha256("A" * 64)  # uppercase rejected
    assert not is_sha256("g" * 64)  # non-hex rejected
    assert not is_sha256("0" * 63)  # wrong length
    assert not is_sha256(0)  # non-string


def test_artifact_file_receipt_valid():
    verify_artifact_file_receipt(_receipt())


def test_artifact_file_receipt_rejects_shape_and_hashes():
    with pytest.raises(ValueError):
        verify_artifact_file_receipt({"artifact_sha256": "0" * 64})
    with pytest.raises(ValueError):
        verify_artifact_file_receipt({**_receipt(), "extra": "x"})
    with pytest.raises(ValueError):
        verify_artifact_file_receipt({"artifact_sha256": "bad", "file_sha256": "0" * 64})
    with pytest.raises(ValueError):
        verify_artifact_file_receipt("not-a-mapping")


def test_history_artifact_receipt():
    verify_history_artifact_receipt(_receipt())
    with pytest.raises(ValueError):
        verify_history_artifact_receipt({"artifact_sha256": "bad", "file_sha256": "bad"})


def test_authorization_receipts_valid_and_order_locked():
    verify_authorization_receipts(
        {
            "parent_spec_approval": _receipt(),
            "amendment_implementation_approval": _receipt(),
            "amended_implementation_review": _receipt(),
            "rerun_authorization": _receipt(),
        }
    )
    assert AUTHORIZATION_PROVIDER_ORDER == (
        "parent_spec_approval",
        "amendment_implementation_approval",
        "amended_implementation_review",
        "rerun_authorization",
    )


def test_authorization_receipts_reject_reorder_missing_extra():
    base = {
        "parent_spec_approval": _receipt(),
        "amendment_implementation_approval": _receipt(),
        "amended_implementation_review": _receipt(),
        "rerun_authorization": _receipt(),
    }
    # reordered keys
    reordered = {
        "rerun_authorization": _receipt(),
        "amended_implementation_review": _receipt(),
        "amendment_implementation_approval": _receipt(),
        "parent_spec_approval": _receipt(),
    }
    with pytest.raises(ValueError):
        verify_authorization_receipts(reordered)
    # missing key
    with pytest.raises(ValueError):
        verify_authorization_receipts({k: v for k, v in list(base.items())[:-1]})
    # extra key
    with pytest.raises(ValueError):
        verify_authorization_receipts({**base, "extra": _receipt()})
    # bad value
    with pytest.raises(ValueError):
        verify_authorization_receipts(
            {**base, "rerun_authorization": {"artifact_sha256": "bad", "file_sha256": "0" * 64}}
        )
    # non-mapping
    with pytest.raises(ValueError):
        verify_authorization_receipts("nope")


def _receipt_fields():
    return {
        "authorization_receipts": {
            "parent_spec_approval": _receipt(),
            "amendment_implementation_approval": _receipt(),
            "amended_implementation_review": _receipt(),
            "rerun_authorization": _receipt(),
        },
        "run_identity_receipt": _receipt(),
        "history_head_receipt": _receipt(),
        "run_admission_receipt": _receipt(),
    }


def test_verify_v2_receipt_fields_valid():
    verify_v2_receipt_fields(_receipt_fields())


def test_verify_v2_receipt_fields_reject():
    bad = _receipt_fields()
    bad["run_identity_receipt"] = {"artifact_sha256": "bad", "file_sha256": "bad"}
    with pytest.raises(ValueError):
        verify_v2_receipt_fields(bad)


def test_v2_field_sets_are_supersets_of_receipt_fields():
    assert V2_RECEIPT_FIELDS <= V2_PRODUCER_EMBEDDING_FIELDS
    assert V2_RECEIPT_FIELDS <= VERIFICATION_EMBEDDING_FIELDS
    # verification embedding is role-bound and has no attempt_chain
    assert "attempt_chain" not in VERIFICATION_EMBEDDING_FIELDS
    assert "attempt_chain" in V2_PRODUCER_EMBEDDING_FIELDS
    assert "role" in VERIFICATION_EMBEDDING_FIELDS
    assert "checkpoint_receipt" in VERIFICATION_EMBEDDING_FIELDS
    assert "source_video_receipts" in VERIFICATION_EMBEDDING_FIELDS


def test_v2_attempt_and_resume_field_sets():
    # attempt carries the worker-isolation objects; resume and embedding do not
    assert V2_RECEIPT_FIELDS <= V2_PRODUCER_ATTEMPT_FIELDS
    assert V2_RECEIPT_FIELDS <= V2_PRODUCER_RESUME_FIELDS
    for f in (
        "worker_request",
        "child_observation",
        "worker_payload",
        "read_isolation_policy",
        "read_isolation_attestation",
    ):
        assert f in V2_PRODUCER_ATTEMPT_FIELDS
        assert f not in V2_PRODUCER_RESUME_FIELDS
        assert f not in V2_PRODUCER_EMBEDDING_FIELDS
    # verification attempt is role-bound with producer receipts
    assert V2_RECEIPT_FIELDS <= VERIFICATION_ATTEMPT_FIELDS
    assert "verification_ordinal" in VERIFICATION_ATTEMPT_FIELDS
    assert "producer_attempt_chain_receipts" in VERIFICATION_ATTEMPT_FIELDS
    assert "verification_embedding_slot" in VERIFICATION_ATTEMPT_FIELDS
    assert "computational_projection_sha256" in VERIFICATION_ATTEMPT_FIELDS


def test_verify_exact_field_set():
    verify_exact_field_set({"a": 1, "b": 2}, frozenset({"a", "b"}))
    with pytest.raises(ValueError):
        verify_exact_field_set({"a": 1}, frozenset({"a", "b"}))
    with pytest.raises(ValueError):
        verify_exact_field_set({"a": 1, "b": 2, "c": 3}, frozenset({"a", "b"}))


def test_verify_provider_slot():
    verify_provider_slot({"provider": "producer_embedding", "verification_state": "verified", "receipt": _receipt()})
    verify_provider_slot({"provider": "producer_embedding", "verification_state": "failed", "receipt": None})
    verify_provider_slot({"provider": "producer_embedding", "verification_state": "not_reached", "receipt": None})
    with pytest.raises(ValueError):
        verify_provider_slot({"provider": "x", "verification_state": "verified", "receipt": None})
    with pytest.raises(ValueError):
        verify_provider_slot({"provider": "x", "verification_state": "failed", "receipt": _receipt()})
    with pytest.raises(ValueError):
        verify_provider_slot({"provider": "x", "verification_state": "bogus", "receipt": None})


def test_verify_provider_slots_order():
    slots = [
        {"provider": "a", "verification_state": "verified", "receipt": _receipt()},
        {"provider": "b", "verification_state": "not_reached", "receipt": None},
    ]
    verify_provider_slots(slots, ("a", "b"))
    with pytest.raises(ValueError):
        verify_provider_slots(slots, ("b", "a"))
    with pytest.raises(ValueError):
        verify_provider_slots(slots[:1], ("a", "b"))


def test_verify_static_input_contract():
    verify_static_input_contract(
        {
            "temporal_plan_artifact_sha256": "0" * 64,
            "temporal_plan_file_sha256": "0" * 64,
            "task0257_receipts_projection_sha256": "0" * 64,
        }
    )
    with pytest.raises(ValueError):
        verify_static_input_contract(
            {
                "temporal_plan_artifact_sha256": "bad",
                "temporal_plan_file_sha256": "0" * 64,
                "task0257_receipts_projection_sha256": "0" * 64,
            }
        )


def test_is_safe_slug_and_rfc3339():
    from app.analysis.task0258_module_a_v2 import is_rfc3339, is_safe_slug

    assert is_safe_slug("producer_embedding")
    assert is_safe_slug("candidate-gate")
    assert not is_safe_slug("bad provider!")
    assert not is_safe_slug("")
    assert is_rfc3339("2026-08-17T13:51:02Z")
    assert is_rfc3339("2026-08-17T13:51:02+08:00")
    assert not is_rfc3339("2026-08-17")
    assert not is_rfc3339("not-a-time")


def test_verify_provider_slot_rejects_bad_slug():
    with pytest.raises(ValueError):
        verify_provider_slot({"provider": "bad provider!", "verification_state": "verified", "receipt": _receipt()})
