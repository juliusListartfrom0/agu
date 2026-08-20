"""Tests for the TASK-0258 Amendment-001 v2 run-history registry skeleton."""

from __future__ import annotations

import pytest

from app.analysis.task0258_run_history import (
    ADMISSION_SCHEMA,
    CLAIM_SCHEMA,
    COMPLETION_SCHEMA,
    HISTORY_EVENTS,
    MARKER_SCHEMA,
    MAX_SEQUENCE_ORDINAL,
    MODULE_ID,
    TRANSITIONS,
    claim_filename,
    completion_filename,
    history_filename,
    is_terminal_event,
    legal_successors,
    parse_history_filename,
    verify_completed_successor,
    verify_history_subject_receipt,
    verify_history_transition,
    verify_marker_scalars,
    verify_registry_listing,
    verify_run_admission,
    verify_run_consumption_claim,
    verify_run_consumption_completed,
    verify_run_history_marker,
)

AUTH = "0" * 64


def test_history_events_exact_order():
    assert HISTORY_EVENTS == (
        "producer_attempt_1_admitted",
        "producer_attempt_1_recoverable_published",
        "producer_attempt_1_completed_private",
        "producer_attempt_2_admitted",
        "producer_attempt_2_recoverable_published",
        "producer_attempt_2_completed_private",
        "producer_attempt_3_admitted",
        "producer_attempt_3_completed_private",
        "verification_attempt_admitted",
        "verification_attempt_completed_private",
        "pre_candidate_failure_published",
        "candidate_published",
        "verified_result_published",
        "postverification_failure_published",
    )


def test_terminal_events():
    assert is_terminal_event("pre_candidate_failure_published")
    assert is_terminal_event("verified_result_published")
    assert is_terminal_event("postverification_failure_published")
    assert not is_terminal_event("candidate_published")
    assert not is_terminal_event("producer_attempt_1_admitted")


def test_transitions_key_edges():
    assert legal_successors("producer_attempt_1_admitted") == frozenset(
        {
            "producer_attempt_1_recoverable_published",
            "producer_attempt_1_completed_private",
            "pre_candidate_failure_published",
        }
    )
    # attempt 3 has no recoverable-published (attempt limit exhausted)
    assert legal_successors("producer_attempt_3_admitted") == frozenset(
        {
            "producer_attempt_3_completed_private",
            "pre_candidate_failure_published",
        }
    )
    assert legal_successors("candidate_published") == frozenset(
        {
            "verified_result_published",
            "postverification_failure_published",
        }
    )
    # every event key is legal and has a defined successor set
    assert set(TRANSITIONS).issubset(set(HISTORY_EVENTS))


def test_terminal_events_have_no_successors():
    for event in ("pre_candidate_failure_published", "verified_result_published", "postverification_failure_published"):
        assert legal_successors(event) == frozenset()
        assert event not in TRANSITIONS


def test_completed_successor():
    verify_completed_successor("producer_attempt_1_admitted")
    with pytest.raises(ValueError):
        verify_completed_successor("producer_attempt_2_admitted")


def test_transition_valid_and_invalid():
    verify_history_transition("producer_attempt_1_admitted", "producer_attempt_1_completed_private")
    with pytest.raises(ValueError):
        verify_history_transition("producer_attempt_1_admitted", "candidate_published")
    with pytest.raises(ValueError):
        verify_history_transition("candidate_published", "producer_attempt_1_admitted")


def test_parse_history_filename_valid():
    nn, event = parse_history_filename(f"{AUTH}.history-03-producer_attempt_1_recoverable_published.json", AUTH)
    assert nn == 3
    assert event == "producer_attempt_1_recoverable_published"


def test_parse_history_filename_rejects_bad_auth():
    with pytest.raises(ValueError):
        parse_history_filename(f"{'f' * 64}.history-01-candidate_published.json", AUTH)


def test_parse_history_filename_rejects_bad_ordinal_and_event():
    with pytest.raises(ValueError):
        parse_history_filename(f"{AUTH}.history-00-candidate_published.json", AUTH)
    with pytest.raises(ValueError):
        parse_history_filename(f"{AUTH}.history-11-candidate_published.json", AUTH)
    with pytest.raises(ValueError):
        parse_history_filename(f"{AUTH}.history-01-bogus_event.json", AUTH)


def test_marker_scalars_valid():
    verify_marker_scalars(
        {
            "schema_version": MARKER_SCHEMA,
            "module_id": MODULE_ID,
            "sequence_ordinal": 1,
            "event": "producer_attempt_1_admitted",
        }
    )


def test_marker_scalars_reject():
    base = {
        "schema_version": MARKER_SCHEMA,
        "module_id": MODULE_ID,
        "sequence_ordinal": 1,
        "event": "producer_attempt_1_admitted",
    }
    with pytest.raises(ValueError):
        verify_marker_scalars({**base, "schema_version": "bogus"})
    with pytest.raises(ValueError):
        verify_marker_scalars({**base, "module_id": "bogus"})
    with pytest.raises(ValueError):
        verify_marker_scalars({**base, "sequence_ordinal": 0})
    with pytest.raises(ValueError):
        verify_marker_scalars({**base, "sequence_ordinal": MAX_SEQUENCE_ORDINAL + 1})
    with pytest.raises(ValueError):
        verify_marker_scalars({**base, "event": "bogus"})


def test_filenames():
    assert claim_filename(AUTH) == f"{AUTH}.claim.json"
    assert completion_filename(AUTH) == f"{AUTH}.completed.json"
    assert history_filename(AUTH, 3, "candidate_published") == f"{AUTH}.history-03-candidate_published.json"
    with pytest.raises(ValueError):
        history_filename(AUTH, 11, "candidate_published")
    with pytest.raises(ValueError):
        history_filename(AUTH, 1, "bogus")


def test_verify_registry_listing_valid():
    names = [claim_filename(AUTH), completion_filename(AUTH)]
    names.append(history_filename(AUTH, 1, "producer_attempt_1_admitted"))
    names.append(history_filename(AUTH, 2, "producer_attempt_1_completed_private"))
    names.append(history_filename(AUTH, 3, "candidate_published"))
    verify_registry_listing(names, AUTH)


def test_verify_registry_listing_rejects():
    names = [claim_filename(AUTH), completion_filename(AUTH)]
    # wrong first entry
    with pytest.raises(ValueError):
        verify_registry_listing([completion_filename(AUTH), claim_filename(AUTH)], AUTH)
    # ordinal gap
    names2 = names + [
        history_filename(AUTH, 1, "producer_attempt_1_admitted"),
        history_filename(AUTH, 3, "candidate_published"),
    ]
    with pytest.raises(ValueError):
        verify_registry_listing(names2, AUTH)
    # duplicate
    names3 = names + [
        history_filename(AUTH, 1, "producer_attempt_1_admitted"),
        history_filename(AUTH, 1, "producer_attempt_1_admitted"),
    ]
    with pytest.raises(ValueError):
        verify_registry_listing(names3, AUTH)


def test_verify_history_subject_receipt():
    verify_history_subject_receipt(
        {
            "provider": "producer_resource_log",
            "receipt_kind": "file_only",
            "artifact_sha256": None,
            "file_sha256": "0" * 64,
        }
    )
    verify_history_subject_receipt(
        {
            "provider": "producer_embedding",
            "receipt_kind": "json",
            "artifact_sha256": "0" * 64,
            "file_sha256": "0" * 64,
        }
    )
    with pytest.raises(ValueError):
        verify_history_subject_receipt(
            {
                "provider": "x",
                "receipt_kind": "file_only",
                "artifact_sha256": "0" * 64,
                "file_sha256": "0" * 64,
            }
        )
    with pytest.raises(ValueError):
        verify_history_subject_receipt(
            {
                "provider": "x",
                "receipt_kind": "bogus",
                "artifact_sha256": None,
                "file_sha256": "0" * 64,
            }
        )


def _marker(subject_kind="published_paths"):
    return {
        "schema_version": MARKER_SCHEMA,
        "module_id": MODULE_ID,
        "authorization_receipt": {"artifact_sha256": "0" * 64, "file_sha256": "0" * 64},
        "run_identity_receipt": {"artifact_sha256": "0" * 64, "file_sha256": "0" * 64},
        "run_id": "run-1",
        "output_root": "/x/vru_causal_temporal_retrospective_v2",
        "nonce": 1,
        "sequence_ordinal": 1,
        "prior_marker_receipt": {"artifact_sha256": "0" * 64, "file_sha256": "0" * 64},
        "event": "candidate_published",
        "subject_kind": subject_kind,
        "subject_receipts": []
        if subject_kind == "worker_launch_admission"
        else [
            {
                "provider": "candidate_gate",
                "receipt_kind": "json",
                "artifact_sha256": "0" * 64,
                "file_sha256": "0" * 64,
            },
        ],
        "subject_projection_sha256": None if subject_kind != "completed_private" else "0" * 64,
        "root_subject_cas": (
            [
                {
                    "relative_path": "staging",
                    "entry_kind": "directory",
                    "device": 1,
                    "inode": 2,
                    "size_bytes": None,
                    "file_sha256": None,
                    "artifact_sha256": None,
                }
            ]
            if subject_kind == "worker_launch_admission"
            else [
                {
                    "relative_path": "candidate_v2",
                    "entry_kind": "directory",
                    "device": 1,
                    "inode": 2,
                    "size_bytes": None,
                    "file_sha256": None,
                    "artifact_sha256": None,
                },
                {
                    "relative_path": "candidate_v2/candidate_gate.json",
                    "entry_kind": "json_file",
                    "device": 1,
                    "inode": 3,
                    "size_bytes": 10,
                    "file_sha256": "0" * 64,
                    "artifact_sha256": "0" * 64,
                },
            ]
        ),
        "worker_launch_claim": None,
        "cumulative_active_runtime_nanoseconds": 0,
        "cumulative_resource_samples": 0,
        "cumulative_resource_log_bytes": 0,
        "created_at_utc": "2026-08-17T00:00:00Z",
        "artifact_sha256": "0" * 64,
    }


def test_verify_run_history_marker_valid():
    verify_run_history_marker(_marker("published_paths"))
    verify_run_history_marker(_marker("completed_private"))


def test_verify_run_history_marker_worker_launch_admission():
    m = _marker("worker_launch_admission")
    m["worker_launch_claim"] = {
        "worker_role": "producer",
        "parent_pid": 100,
        "child_nonce": "0" * 64,
        "launch_secret_sha256": "0" * 64,
        "bootstrap_source_sha256": "0" * 64,
        "worker_request_artifact_sha256": "0" * 64,
        "provider_process_instance_id": 1,
        "audit_prepared_artifact_sha256": "0" * 64,
        "expected_hash_state_projection_sha256": "0" * 64,
        "private_stage_identity": {"device": 1, "inode": 2},
        "source_fd": 4,
        "liveness_fd": 5,
        "result_fd": 6,
        "admitted_marker_fd": 7,
        "audit_start_gate_fd": 8,
        "claim_envelope_fd": 9,
    }
    verify_run_history_marker(m)


def test_verify_run_history_marker_rejects():
    # missing field
    m = _marker("published_paths")
    del m["artifact_sha256"]
    with pytest.raises(ValueError):
        verify_run_history_marker(m)
    # published_paths must have null projection
    m = _marker("published_paths")
    m["subject_projection_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        verify_run_history_marker(m)
    # completed_private requires non-null projection
    m = _marker("completed_private")
    m["subject_projection_sha256"] = None
    with pytest.raises(ValueError):
        verify_run_history_marker(m)
    # non-admission must have null worker_launch_claim
    m = _marker("completed_private")
    m["worker_launch_claim"] = {"worker_role": "producer"}
    with pytest.raises(ValueError):
        verify_run_history_marker(m)


def _receipt():
    return {"artifact_sha256": "0" * 64, "file_sha256": "0" * 64}


def test_verify_run_consumption_claim():
    p = {
        "schema_version": CLAIM_SCHEMA,
        "module_id": MODULE_ID,
        "authorization_receipt": _receipt(),
        "run_id": "run-1",
        "output_root_absolute_path": "/x",
        "nonce": "0" * 64,
        "state": "claimed",
        "created_at_utc": "2026-08-17T00:00:00Z",
        "artifact_sha256": "0" * 64,
    }
    verify_run_consumption_claim(p)
    p["state"] = "completed"
    with pytest.raises(ValueError):
        verify_run_consumption_claim(p)


def test_verify_run_admission():
    p = {
        "schema_version": ADMISSION_SCHEMA,
        "module_id": MODULE_ID,
        "authorization_receipt": _receipt(),
        "claim_receipt": _receipt(),
        "nonce": "0" * 64,
        "run_id": "run-1",
        "output_root_absolute_path": "/x",
        "static_input_contract": {
            "temporal_plan_artifact_sha256": "0" * 64,
            "temporal_plan_file_sha256": "0" * 64,
            "task0257_receipts_projection_sha256": "0" * 64,
        },
        "maximum_run_count": 1,
        "admission_state": "admitted",
        "module_b_authorized": False,
        "created_at_utc": "2026-08-17T00:00:00Z",
        "artifact_sha256": "0" * 64,
    }
    verify_run_admission(p)
    p["module_b_authorized"] = True
    with pytest.raises(ValueError):
        verify_run_admission(p)


def test_verify_run_consumption_completed():
    p = {
        "schema_version": COMPLETION_SCHEMA,
        "module_id": MODULE_ID,
        "authorization_receipt": _receipt(),
        "claim_receipt": _receipt(),
        "admission_receipt": _receipt(),
        "nonce": "0" * 64,
        "run_id": "run-1",
        "output_root_absolute_path": "/x",
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
        "artifact_sha256": "0" * 64,
    }
    verify_run_consumption_completed(p)
    p["root_identity"] = {"device": 1}
    with pytest.raises(ValueError):
        verify_run_consumption_completed(p)


def test_verify_run_history_marker_rejects_bad_timestamp():
    m = _marker("published_paths")
    m["created_at_utc"] = "2026-08-17"
    with pytest.raises(ValueError):
        verify_run_history_marker(m)


def test_marker_rejects_duplicate_unsorted_subject_receipts():
    m = _marker("published_paths")
    m["subject_receipts"] = [
        {"provider": "b", "receipt_kind": "json", "artifact_sha256": "0" * 64, "file_sha256": "0" * 64},
        {"provider": "a", "receipt_kind": "json", "artifact_sha256": "0" * 64, "file_sha256": "0" * 64},
    ]
    with pytest.raises(ValueError):
        verify_run_history_marker(m)
    m["subject_receipts"] = [
        {"provider": "a", "receipt_kind": "json", "artifact_sha256": "0" * 64, "file_sha256": "0" * 64},
        {"provider": "a", "receipt_kind": "json", "artifact_sha256": "0" * 64, "file_sha256": "0" * 64},
    ]
    with pytest.raises(ValueError):
        verify_run_history_marker(m)


def test_completion_rejects_bad_nonce():
    from app.analysis.task0258_run_history import verify_run_consumption_completed

    p = {
        "schema_version": COMPLETION_SCHEMA,
        "module_id": MODULE_ID,
        "authorization_receipt": _receipt(),
        "claim_receipt": _receipt(),
        "admission_receipt": _receipt(),
        "nonce": "NOTHEX",
        "run_id": "run-1",
        "output_root_absolute_path": "/x",
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
        "artifact_sha256": "0" * 64,
    }
    with pytest.raises(ValueError):
        verify_run_consumption_completed(p)


def test_verify_worker_launch_claim():
    from app.analysis.task0258_run_history import verify_worker_launch_claim

    claim = {
        "worker_role": "verification",
        "parent_pid": 100,
        "child_nonce": "0" * 64,
        "launch_secret_sha256": "0" * 64,
        "bootstrap_source_sha256": "0" * 64,
        "worker_request_artifact_sha256": "0" * 64,
        "provider_process_instance_id": 1,
        "audit_prepared_artifact_sha256": "0" * 64,
        "expected_hash_state_projection_sha256": "0" * 64,
        "private_stage_identity": {"device": 1, "inode": 2},
        "source_fd": 4,
        "liveness_fd": 5,
        "result_fd": 6,
        "admitted_marker_fd": 7,
        "audit_start_gate_fd": 8,
        "claim_envelope_fd": 9,
    }
    verify_worker_launch_claim(claim)
    bad = dict(claim)
    bad["source_fd"] = 10
    with pytest.raises(ValueError):
        verify_worker_launch_claim(bad)
    bad = dict(claim)
    bad["worker_role"] = "bad role!"
    with pytest.raises(ValueError):
        verify_worker_launch_claim(bad)


def test_marker_worker_launch_admission_validates_claim():
    m = _marker("worker_launch_admission")
    m["worker_launch_claim"] = {
        "worker_role": "producer",
        "parent_pid": 100,
        "child_nonce": "0" * 64,
        "launch_secret_sha256": "0" * 64,
        "bootstrap_source_sha256": "0" * 64,
        "worker_request_artifact_sha256": "0" * 64,
        "provider_process_instance_id": 1,
        "audit_prepared_artifact_sha256": "0" * 64,
        "expected_hash_state_projection_sha256": "0" * 64,
        "private_stage_identity": {"device": 1, "inode": 2},
        "source_fd": 4,
        "liveness_fd": 5,
        "result_fd": 6,
        "admitted_marker_fd": 7,
        "audit_start_gate_fd": 8,
        "claim_envelope_fd": 9,
    }
    verify_run_history_marker(m)
