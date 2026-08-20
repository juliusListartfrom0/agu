"""Tests for the TASK-0258 v2 run-history registry durable write path."""

from __future__ import annotations

import json

import pytest

from app.analysis.task0258_run_history import (
    ADMISSION_SCHEMA,
    CLAIM_SCHEMA,
    COMPLETION_SCHEMA,
    MARKER_SCHEMA,
    MODULE_ID,
    claim_filename,
    completion_filename,
)
from app.analysis.task0258_v2_registry import (
    append_run_history_marker,
    seal_run_consumption_claim,
    seal_run_consumption_completed,
)

AUTH = "0" * 64


def _receipt():
    return {"artifact_sha256": "0" * 64, "file_sha256": "0" * 64}


def _claim():
    return {
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


def _completed():
    return {
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


def _marker(event, seq=1, subject_kind="published_paths"):
    return {
        "schema_version": MARKER_SCHEMA,
        "module_id": MODULE_ID,
        "authorization_receipt": _receipt(),
        "run_identity_receipt": _receipt(),
        "run_id": "run-1",
        "output_root": "/x",
        "nonce": 1,
        "sequence_ordinal": seq,
        "prior_marker_receipt": _receipt(),
        "event": event,
        "subject_kind": subject_kind,
        "subject_receipts": [
            {
                "provider": "candidate_gate",
                "receipt_kind": "json",
                "artifact_sha256": "0" * 64,
                "file_sha256": "0" * 64,
            },
        ],
        "subject_projection_sha256": None,
        "root_subject_cas": [
            {
                "relative_path": "candidate_v2",
                "entry_kind": "directory",
                "device": 1,
                "inode": 2,
                "size_bytes": None,
                "file_sha256": None,
                "artifact_sha256": None,
            },
        ],
        "worker_launch_claim": None,
        "cumulative_active_runtime_nanoseconds": 0,
        "cumulative_resource_samples": 0,
        "cumulative_resource_log_bytes": 0,
        "created_at_utc": "2026-08-17T00:00:00Z",
        "artifact_sha256": "0" * 64,
    }


def test_seal_claim_and_completed(tmp_path):
    reg = tmp_path / "registry"
    reg.mkdir()
    c = seal_run_consumption_claim(reg, AUTH, _claim())
    assert c == reg / claim_filename(AUTH)
    assert json.loads(c.read_text())["state"] == "claimed"
    done = seal_run_consumption_completed(reg, AUTH, _completed())
    assert done == reg / completion_filename(AUTH)
    # no-clobber: re-seal fails
    with pytest.raises(FileExistsError):
        seal_run_consumption_claim(reg, AUTH, _claim())


def test_append_marker_valid(tmp_path):
    reg = tmp_path / "registry"
    reg.mkdir()
    m = append_run_history_marker(reg, AUTH, _marker("producer_attempt_1_admitted"), None)
    assert m == reg / f"{AUTH}.history-01-producer_attempt_1_admitted.json"
    # second marker with a legal transition
    m2 = append_run_history_marker(
        reg,
        AUTH,
        _marker("producer_attempt_1_completed_private", seq=2),
        "producer_attempt_1_admitted",
    )
    assert m2.name == f"{AUTH}.history-02-producer_attempt_1_completed_private.json"


def test_append_marker_rejects_illegal_transition(tmp_path):
    reg = tmp_path / "registry"
    reg.mkdir()
    with pytest.raises(ValueError):
        append_run_history_marker(
            reg,
            AUTH,
            _marker("candidate_published"),
            None,
        )
    with pytest.raises(ValueError):
        append_run_history_marker(
            reg,
            AUTH,
            _marker("candidate_published", seq=2),
            "producer_attempt_1_admitted",
        )


def test_append_marker_no_clobber(tmp_path):
    reg = tmp_path / "registry"
    reg.mkdir()
    append_run_history_marker(reg, AUTH, _marker("producer_attempt_1_admitted"), None)
    with pytest.raises(FileExistsError):
        append_run_history_marker(reg, AUTH, _marker("producer_attempt_1_admitted"), None)


def _admission():
    return {
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


def test_create_run_history_registry(tmp_path):
    from app.analysis.task0258_v2_registry import create_run_history_registry

    reg = tmp_path / "registry"
    reg.mkdir()
    out = tmp_path / "out" / "vru_causal_temporal_retrospective_v2"
    lock = tmp_path / "out" / ".lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("")

    result = create_run_history_registry(
        registry_dir=reg,
        auth_sha256=AUTH,
        claim_payload=_claim(),
        admission_payload=_admission(),
        completion_payload=_completed(),
        output_root=out,
        flock_path=lock,
    )
    assert result == out
    # claim + completion in registry, admission in the output root
    assert (reg / f"{AUTH}.claim.json").is_file()
    assert (reg / f"{AUTH}.completed.json").is_file()
    assert (out / "run_admission.json").is_file()
    assert json.loads((out / "run_admission.json").read_text())["admission_state"] == "admitted"
    # second creation must fail: claim no-clobber
    with pytest.raises(FileExistsError):
        create_run_history_registry(
            registry_dir=reg,
            auth_sha256=AUTH,
            claim_payload=_claim(),
            admission_payload=_admission(),
            completion_payload=_completed(),
            output_root=tmp_path / "out2" / "vru_causal_temporal_retrospective_v2",
            flock_path=tmp_path / "out2" / ".lock",
        )
