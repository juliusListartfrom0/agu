"""End-to-end test of the TASK-0258 v2 production pipeline orchestration."""

from __future__ import annotations

import hashlib

import pytest

from app.analysis.task0258_module_a_v2 import MODULE_ID, canonical_artifact_sha256, compact_canonical_json
from app.analysis.task0258_v2_artifacts import CANDIDATE_MEMBER_PATHS
from app.analysis.task0258_v2_pipeline import (
    build_candidate_gate_payload,
    build_candidate_receipt_bundle_payload,
    build_member_receipts,
    build_postpublication_verification_payload,
    seal_candidate_v2,
)
from app.analysis.task0258_v2_pipeline_cli import (
    _issue_synthetic_v2_pipeline_test_context,
    assemble_candidate_members,
    run_v2_pipeline,
)


def _receipt():
    return {"artifact_sha256": "0" * 64, "file_sha256": "0" * 64}


def _member_row(path):
    is_jsonl = path.endswith(".jsonl")
    return {
        "relative_path": path,
        "receipt_kind": "file_only" if is_jsonl else "json",
        "size_bytes": 0,
        "internal_sha256_field": None if is_jsonl else "artifact_sha256",
        "artifact_sha256": None if is_jsonl else "0" * 64,
        "file_sha256": "0" * 64,
    }


def _seal(payload):
    payload["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )
    return payload


def _file_receipt(payload):
    data = (compact_canonical_json(payload) + "\n").encode()
    return {"artifact_sha256": payload["artifact_sha256"], "file_sha256": hashlib.sha256(data).hexdigest()}


def _trust_slots(
    providers,
    *,
    authorization_receipt=None,
    history_receipt=None,
    admission_receipt=None,
    static_input_contract=None,
):
    out = []
    for p in providers:
        if p == "run_history_ledger":
            out.append(
                {
                    "provider": p,
                    "verification_state": "verified",
                    "receipt": history_receipt
                    or {"run_identity_receipt": _receipt(), "head_receipt": _receipt(), "marker_count": 0},
                }
            )
        elif p == "static_inputs":
            out.append(
                {
                    "provider": p,
                    "verification_state": "verified",
                    "receipt": static_input_contract
                    or {
                        "temporal_plan_artifact_sha256": "0" * 64,
                        "temporal_plan_file_sha256": "0" * 64,
                        "task0257_receipts_projection_sha256": "0" * 64,
                    },
                }
            )
        elif p == "exact_v2_rerun_authorization":
            out.append(
                {"provider": p, "verification_state": "verified", "receipt": authorization_receipt or _receipt()}
            )
        elif p == "run_admission":
            out.append({"provider": p, "verification_state": "verified", "receipt": admission_receipt or _receipt()})
        else:
            out.append({"provider": p, "verification_state": "verified", "receipt": _receipt()})
    return out


def _claim(output_root="/x"):
    return _seal(
        {
            "schema_version": "agu.task0258-module-a-v2-run-consumption-claim.v1",
            "module_id": MODULE_ID,
            "authorization_receipt": _receipt(),
            "run_id": "run-1",
            "output_root_absolute_path": output_root,
            "nonce": "0" * 64,
            "state": "claimed",
            "created_at_utc": "2026-08-17T00:00:00Z",
        }
    )


def _admission(output_root="/x", claim_receipt=None):
    return _seal(
        {
            "schema_version": "agu.task0258-module-a-v2-run-admission.v1",
            "module_id": MODULE_ID,
            "authorization_receipt": _receipt(),
            "claim_receipt": claim_receipt or _receipt(),
            "nonce": "0" * 64,
            "run_id": "run-1",
            "output_root_absolute_path": output_root,
            "static_input_contract": {
                "temporal_plan_artifact_sha256": "0" * 64,
                "temporal_plan_file_sha256": "0" * 64,
                "task0257_receipts_projection_sha256": "0" * 64,
            },
            "maximum_run_count": 1,
            "admission_state": "admitted",
            "module_b_authorized": False,
            "created_at_utc": "2026-08-17T00:00:00Z",
        }
    )


def _completed(output_root="/x", claim_receipt=None, admission_receipt=None):
    return _seal(
        {
            "schema_version": "agu.task0258-module-a-v2-run-consumption-completed.v1",
            "module_id": MODULE_ID,
            "authorization_receipt": _receipt(),
            "claim_receipt": claim_receipt or _receipt(),
            "admission_receipt": admission_receipt or _receipt(),
            "nonce": "0" * 64,
            "run_id": "run-1",
            "output_root_absolute_path": output_root,
            "consumption_count": 1,
            "state": "completed",
            "root_identity": {"device": 1, "inode": 2},
            "admission_identity": {
                "device": 1,
                "inode": 3,
                "size_bytes": 4,
                "internal_sha256": "0" * 64,
                "file_sha256": "0" * 64,
            },
            "created_at_utc": "2026-08-17T00:00:00Z",
        }
    )


def _gate(
    *,
    authorization_receipt=None,
    history_receipt=None,
    admission_receipt=None,
    static_input_contract=None,
):
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
            ),
            authorization_receipt=authorization_receipt,
            history_receipt=history_receipt,
            admission_receipt=admission_receipt,
            static_input_contract=static_input_contract,
        ),
        producer_attempt_chain=[],
        verification_attempt_receipt=_receipt(),
        error_bound_result={},
        candidate_metric_outcome="within_frozen_error_bounds",
    )


def _json_member(name):
    payload = {"schema_version": f"agu.{name}"}
    payload["artifact_sha256"] = canonical_artifact_sha256(payload)
    return payload


def _members(candidate_gate=None):
    return {
        "terminal_attempt/resource_guard.jsonl": b'{"attempt_role":"producer"}\n',
        "terminal_attempt/attempt_record.json": _json_member("attempt"),
        "producer_tiled_swin_embeddings.json": _json_member("embeddings"),
        "verification_attempt/resource_guard.jsonl": b'{"attempt_role":"verification"}\n',
        "verification_attempt/attempt_record.json": _json_member("verification-attempt"),
        "verification_tiled_swin_embeddings.json": _json_member("verification-embeddings"),
        "temporal_retrospective.json": _json_member("retrospective"),
        "baseline_final_evaluator.json": _json_member("baseline-evaluator"),
        "candidate_final_evaluator.json": _json_member("candidate-evaluator"),
        "candidate_gate.json": candidate_gate or _gate(),
    }


def _result_payload(
    *,
    authorization_receipt=None,
    history_receipt=None,
    admission_receipt=None,
    static_input_contract=None,
    candidate_member_receipts=None,
    candidate_published_history_head_receipt=None,
):
    member_rows = candidate_member_receipts or [_member_row(path) for path in CANDIDATE_MEMBER_PATHS]
    member_by_path = {row["relative_path"]: row for row in member_rows}
    authorization_receipt = authorization_receipt or _receipt()
    history_receipt = history_receipt or {
        "run_identity_receipt": _receipt(),
        "head_receipt": candidate_published_history_head_receipt or _receipt(),
        "marker_count": 0,
    }
    admission_receipt = admission_receipt or _receipt()
    static_input_contract = static_input_contract or _admission()["static_input_contract"]
    return build_postpublication_verification_payload(
        error_bounds_pass=True,
        authorization_receipts={
            "parent_spec_approval": _receipt(),
            "amendment_implementation_approval": _receipt(),
            "amended_implementation_review": _receipt(),
            "rerun_authorization": authorization_receipt,
        },
        run_history_contract_receipt=history_receipt,
        run_admission_receipt=admission_receipt,
        static_input_contract=static_input_contract,
        candidate_receipt_bundle_receipt=_receipt(),
        candidate_member_receipts=member_rows,
        prior_attempt_receipts=[],
        producer_embedding_receipt=member_by_path["producer_tiled_swin_embeddings.json"],
        verification_embedding_receipt=member_by_path["verification_tiled_swin_embeddings.json"],
        verification_attempt_receipt=member_by_path["verification_attempt/attempt_record.json"],
        error_bound_result={},
        evaluator_receipts={
            "baseline": member_by_path["baseline_final_evaluator.json"],
            "candidate": member_by_path["candidate_final_evaluator.json"],
        },
    )


def test_run_v2_pipeline_end_to_end(tmp_path):
    reg = tmp_path / "registry"
    reg.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    root = out / "vru_causal_temporal_retrospective_v2"
    lock2 = out / ".task0258-output.lock"
    lock2.write_text("")
    bundle_path = tmp_path / "candidate-receipt-bundle.json"
    claim = _claim(str(root))
    admission = _admission(str(root), _file_receipt(claim))
    completion = _completed(str(root), _file_receipt(claim), _file_receipt(admission))
    admission_receipt = _file_receipt(admission)
    completion_receipt = _file_receipt(completion)
    candidate_published_history_head_receipt = _receipt()
    authorization_receipt = _receipt()
    static_input_contract = admission["static_input_contract"]
    history_receipt = {
        "run_identity_receipt": completion_receipt,
        "head_receipt": completion_receipt,
        "marker_count": 0,
    }
    candidate_gate = _gate(
        authorization_receipt=authorization_receipt,
        history_receipt=history_receipt,
        admission_receipt=admission_receipt,
        static_input_contract=static_input_contract,
    )
    members = _members(candidate_gate)
    encoded = assemble_candidate_members(members)

    # Pre-publish candidate_v2 in a scratch root to compute the bundle's member
    # receipts from the exact bytes the pipeline will publish.
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    lock = scratch.parent / ".task0258-output.lock"
    lock.write_text("")
    pre_candidate = seal_candidate_v2(scratch, encoded, flock_path=lock)

    bundle_payload = build_candidate_receipt_bundle_payload(
        authorization_receipt=authorization_receipt,
        run_identity_receipt=completion_receipt,
        run_admission_receipt=admission_receipt,
        static_input_contract=static_input_contract,
        candidate_published_history_head_receipt=candidate_published_history_head_receipt,
        candidate_dir=pre_candidate,
        observed_at_utc="2026-08-17T00:00:00Z",
    )
    result_payload = _result_payload(
        authorization_receipt=authorization_receipt,
        history_receipt={
            "run_identity_receipt": completion_receipt,
            "head_receipt": candidate_published_history_head_receipt,
            "marker_count": 0,
        },
        admission_receipt=admission_receipt,
        static_input_contract=static_input_contract,
        candidate_member_receipts=build_member_receipts(pre_candidate),
        candidate_published_history_head_receipt=candidate_published_history_head_receipt,
    )

    with pytest.raises(PermissionError):
        run_v2_pipeline(
            output_root=root,
            registry_dir=reg,
            flock_path=lock2,
            auth_sha256="0" * 64,
            claim_payload=claim,
            admission_payload=admission,
            completion_payload=completion,
            candidate_members=members,
            bundle_path=bundle_path,
            bundle_payload=bundle_payload,
            result_payload=result_payload,
        )
    assert not any(reg.iterdir())
    assert not root.exists()
    assert not bundle_path.exists()

    with pytest.raises(PermissionError, match="explicit test context"):
        run_v2_pipeline(
            output_root=root,
            registry_dir=reg,
            flock_path=lock2,
            auth_sha256="0" * 64,
            claim_payload=claim,
            admission_payload=admission,
            completion_payload=completion,
            candidate_members=members,
            bundle_path=bundle_path,
            bundle_payload=bundle_payload,
            result_payload=result_payload,
            authorization_context=object(),
            synthetic_test_only=True,
        )
    assert not any(reg.iterdir())
    assert not root.exists()
    assert not bundle_path.exists()

    bad_bundle = dict(bundle_payload)
    bad_bundle["run_admission_receipt"] = _receipt()
    bad_bundle["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in bad_bundle.items() if key != "artifact_sha256"}
    )
    with pytest.raises(ValueError, match="bundle admission receipt"):
        run_v2_pipeline(
            output_root=root,
            registry_dir=reg,
            flock_path=lock2,
            auth_sha256="0" * 64,
            claim_payload=claim,
            admission_payload=admission,
            completion_payload=completion,
            candidate_members=members,
            bundle_path=bundle_path,
            bundle_payload=bad_bundle,
            result_payload=result_payload,
            authorization_context=_issue_synthetic_v2_pipeline_test_context(),
            synthetic_test_only=True,
        )
    assert not any(reg.iterdir())
    assert not root.exists()
    assert not bundle_path.exists()

    bad_result = dict(result_payload)
    bad_member_rows = list(result_payload["candidate_member_receipts"])
    bad_member_rows[0] = dict(bad_member_rows[0])
    bad_member_rows[0]["file_sha256"] = "1" * 64
    bad_result["candidate_member_receipts"] = bad_member_rows
    bad_result["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in bad_result.items() if key != "artifact_sha256"}
    )
    with pytest.raises(ValueError, match="candidate member receipts"):
        run_v2_pipeline(
            output_root=root,
            registry_dir=reg,
            flock_path=lock2,
            auth_sha256="0" * 64,
            claim_payload=claim,
            admission_payload=admission,
            completion_payload=completion,
            candidate_members=members,
            bundle_path=bundle_path,
            bundle_payload=bundle_payload,
            result_payload=bad_result,
            authorization_context=_issue_synthetic_v2_pipeline_test_context(),
            synthetic_test_only=True,
        )
    assert not any(reg.iterdir())
    assert not root.exists()
    assert not bundle_path.exists()

    result = run_v2_pipeline(
        output_root=root,
        registry_dir=reg,
        flock_path=lock2,
        auth_sha256="0" * 64,
        claim_payload=claim,
        admission_payload=admission,
        completion_payload=completion,
        candidate_members=members,
        bundle_path=bundle_path,
        bundle_payload=bundle_payload,
        result_payload=result_payload,
        authorization_context=_issue_synthetic_v2_pipeline_test_context(),
        synthetic_test_only=True,
    )
    assert result["candidate"] == root / "candidate_v2"
    assert result["bundle"] == bundle_path
    assert result["result"] == root / "verified_result_v2"
