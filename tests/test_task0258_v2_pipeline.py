"""Tests for the TASK-0258 v2 pipeline candidate + bundle sealing."""

from __future__ import annotations

import json

from app.analysis.task0258_v2_artifacts import CANDIDATE_MEMBER_PATHS
from app.analysis.task0258_v2_pipeline import (
    build_candidate_gate_payload,
    build_candidate_receipt_bundle_payload,
    build_member_receipts,
    seal_candidate_receipt_bundle,
    seal_candidate_v2,
)


def _receipt():
    return {"artifact_sha256": "0" * 64, "file_sha256": "0" * 64}


def _trust_slots(providers):
    out = []
    for p in providers:
        if p == "run_history_ledger":
            out.append(
                {
                    "provider": p,
                    "verification_state": "verified",
                    "receipt": {"run_identity_receipt": _receipt(), "head_receipt": _receipt(), "marker_count": 0},
                }
            )
        elif p == "static_inputs":
            out.append(
                {
                    "provider": p,
                    "verification_state": "verified",
                    "receipt": {
                        "temporal_plan_artifact_sha256": "0" * 64,
                        "temporal_plan_file_sha256": "0" * 64,
                        "task0257_receipts_projection_sha256": "0" * 64,
                    },
                }
            )
        else:
            out.append({"provider": p, "verification_state": "verified", "receipt": _receipt()})
    return out


def _gate():
    return build_candidate_gate_payload(
        input_receipts=_trust_slots(
            (
                "parent_spec_approval",
                "amendment_implementation_approval",
                "amended_implementation_review",
                "exact_v2_rerun_authorization",
                "run_history_ledger",
                "run_admission",
                "static_inputs",
                "producer_embedding",
                "verification_embedding",
                "verification_attempt",
                "retrospective",
                "baseline_evaluator",
                "candidate_evaluator",
            )
        ),
        producer_attempt_chain=[],
        verification_attempt_receipt=_receipt(),
        error_bound_result={},
        candidate_metric_outcome="within_frozen_error_bounds",
    )


def _make_members():
    members = {}
    for i, rel in enumerate(CANDIDATE_MEMBER_PATHS):
        if rel.endswith(".jsonl"):
            members[rel] = b'{"schema_version":"x"}\n'
        else:
            payload = {"schema_version": "x", "artifact_sha256": "0" * 64}
            members[rel] = (json.dumps(payload) + "\n").encode()
    return members


def test_candidate_gate_payload():
    gate = _gate()
    assert gate["publication_state"] == "not_yet_observed"
    assert gate["postpublication_verification_required"] is True
    assert len(gate["ordered_prepublication_check_results"]) == 24
    assert len(gate["artifact_sha256"]) == 64


def test_seal_candidate_v2_and_member_receipts(tmp_path):
    members = _make_members()
    out = tmp_path / "out"
    out.mkdir()
    lock = out / ".lock"
    lock.write_text("")
    final = seal_candidate_v2(out, members, flock_path=lock)
    assert final == out / "candidate_v2"
    assert final.is_dir()
    rows = build_member_receipts(final)
    assert [r["relative_path"] for r in rows] == list(CANDIDATE_MEMBER_PATHS)
    jsonl = [r for r in rows if r["receipt_kind"] == "file_only"]
    json_rows = [r for r in rows if r["receipt_kind"] == "json"]
    assert len(jsonl) == 2
    assert len(json_rows) == 8
    for r in json_rows:
        assert r["artifact_sha256"] == "0" * 64
        assert r["internal_sha256_field"] == "artifact_sha256"


def test_candidate_receipt_bundle(tmp_path):
    members = _make_members()
    out = tmp_path / "out"
    out.mkdir()
    lock = out / ".lock"
    lock.write_text("")
    final = seal_candidate_v2(out, members, flock_path=lock)
    bundle = build_candidate_receipt_bundle_payload(
        authorization_receipt=_receipt(),
        run_identity_receipt=_receipt(),
        run_admission_receipt=_receipt(),
        static_input_contract={
            "temporal_plan_artifact_sha256": "0" * 64,
            "temporal_plan_file_sha256": "0" * 64,
            "task0257_receipts_projection_sha256": "0" * 64,
        },
        candidate_published_history_head_receipt=_receipt(),
        candidate_dir=final,
        observed_at_utc="2026-08-17T00:00:00Z",
    )
    assert bundle["candidate_generation_name"] == "candidate_v2"
    assert len(bundle["ordered_member_receipts"]) == 10
    bundle_path = tmp_path / "bundle.json"
    seal_candidate_receipt_bundle(bundle_path, bundle)
    assert bundle_path.is_file()
    assert json.loads(bundle_path.read_text())["candidate_generation_name"] == "candidate_v2"


def _result(error_bounds_pass=True):
    from app.analysis.task0258_v2_pipeline import build_postpublication_verification_payload

    return build_postpublication_verification_payload(
        error_bounds_pass=error_bounds_pass,
        authorization_receipts={},
        run_history_contract_receipt={},
        run_admission_receipt=_receipt(),
        static_input_contract={
            "temporal_plan_artifact_sha256": "0" * 64,
            "temporal_plan_file_sha256": "0" * 64,
            "task0257_receipts_projection_sha256": "0" * 64,
        },
        candidate_receipt_bundle_receipt=_receipt(),
        candidate_member_receipts=[],
        prior_attempt_receipts=[],
        producer_embedding_receipt=_receipt(),
        verification_embedding_receipt=_receipt(),
        verification_attempt_receipt=_receipt(),
        error_bound_result={},
        evaluator_receipts={},
    )


def test_postpublication_verification_payload():
    r = _result(error_bounds_pass=True)
    assert r["decision"] == "mechanical_pass"
    assert r["stop_reason"] is None
    assert r["ordered_check_results"][-1] == {"check_name": "frozen_error_bounds", "passed": True}
    r2 = _result(error_bounds_pass=False)
    assert r2["decision"] == "temporal-hypothesis-rejected"
    assert r2["stop_reason"] == "temporal-hypothesis-rejected"
    assert r2["ordered_check_results"][-1]["passed"] is False


def test_seal_verified_result_and_postverification_failure(tmp_path):
    from app.analysis.task0258_v2_pipeline import (
        build_postpublication_failure_payload,
        seal_postverification_failure,
        seal_verified_result,
    )

    out = tmp_path / "out"
    out.mkdir()
    lock = out / ".lock"
    lock.write_text("")
    final = seal_verified_result(out, _result(True), flock_path=lock)
    assert final == out / "verified_result_v2"
    assert (final / "verification_registry.json").is_file()

    # postverification failure: prefix + bundle slots
    slots = [
        {"provider": "parent_spec_approval", "verification_state": "verified", "receipt": _receipt()},
        {"provider": "amendment_implementation_approval", "verification_state": "verified", "receipt": _receipt()},
        {"provider": "amended_implementation_review", "verification_state": "verified", "receipt": _receipt()},
        {"provider": "exact_v2_rerun_authorization", "verification_state": "verified", "receipt": _receipt()},
        {
            "provider": "run_history_ledger",
            "verification_state": "verified",
            "receipt": {"run_identity_receipt": _receipt(), "head_receipt": _receipt(), "marker_count": 0},
        },
        {"provider": "run_admission", "verification_state": "verified", "receipt": _receipt()},
        {
            "provider": "static_inputs",
            "verification_state": "verified",
            "receipt": {
                "temporal_plan_artifact_sha256": "0" * 64,
                "temporal_plan_file_sha256": "0" * 64,
                "task0257_receipts_projection_sha256": "0" * 64,
            },
        },
        {"provider": "candidate_receipt_bundle", "verification_state": "verified", "receipt": _receipt()},
    ]
    failure = build_postpublication_failure_payload(
        failed_check_name="retrospective",
        dependency_provider_slots=slots,
    )
    assert failure["decision"] == "mechanical_failure"
    assert failure["final_result_published"] is False
    fail_dir = seal_postverification_failure(out, failure, flock_path=lock)
    assert fail_dir == out / "postverification_failure_v2"
    assert (fail_dir / "failure.json").is_file()
