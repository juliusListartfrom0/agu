"""Tests for the TASK-0258 v2 run-history registry durable write path."""

from __future__ import annotations

import hashlib
import json

import pytest

from app.analysis.task0258_module_a_v2 import canonical_artifact_sha256, compact_canonical_json
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


def _seal(payload):
    payload["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )
    return payload


def _file_receipt(payload):
    data = (compact_canonical_json(payload) + "\n").encode()
    return {"artifact_sha256": payload["artifact_sha256"], "file_sha256": hashlib.sha256(data).hexdigest()}


def _file_receipt(payload):
    data = (compact_canonical_json(payload) + "\n").encode()
    return {"artifact_sha256": payload["artifact_sha256"], "file_sha256": hashlib.sha256(data).hexdigest()}


def _claim(output_root="/x"):
    return _seal(
        {
            "schema_version": CLAIM_SCHEMA,
            "module_id": MODULE_ID,
            "authorization_receipt": _receipt(),
            "run_id": "run-1",
            "output_root_absolute_path": output_root,
            "nonce": "0" * 64,
            "state": "claimed",
            "created_at_utc": "2026-08-17T00:00:00Z",
        }
    )


def _completed(output_root="/x", claim_receipt=None, admission_receipt=None):
    return _seal(
        {
            "schema_version": COMPLETION_SCHEMA,
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


def _marker(event, seq=1, subject_kind="published_paths", prior_marker_receipt=None, run_identity_receipt=None):
    payload = {
        "schema_version": MARKER_SCHEMA,
        "module_id": MODULE_ID,
        "authorization_receipt": _receipt(),
        "run_identity_receipt": run_identity_receipt or _receipt(),
        "run_id": "run-1",
        "output_root": "/x",
        "nonce": "0" * 64,
        "sequence_ordinal": seq,
        "prior_marker_receipt": prior_marker_receipt or _receipt(),
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
    return _seal(payload)


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
    claim = _claim()
    completed = _completed()
    seal_run_consumption_claim(reg, AUTH, claim)
    seal_run_consumption_completed(reg, AUTH, completed)
    completed_receipt = _file_receipt(completed)
    m = append_run_history_marker(
        reg,
        AUTH,
        _marker(
            "producer_attempt_1_admitted",
            prior_marker_receipt=completed_receipt,
            run_identity_receipt=completed_receipt,
        ),
        None,
    )
    assert m == reg / f"{AUTH}.history-01-producer_attempt_1_admitted.json"
    # second marker with a legal transition
    m2 = append_run_history_marker(
        reg,
        AUTH,
        _marker(
            "producer_attempt_1_completed_private",
            seq=2,
            prior_marker_receipt=_file_receipt(json.loads(m.read_text())),
            run_identity_receipt=completed_receipt,
        ),
        "producer_attempt_1_admitted",
    )
    assert m2.name == f"{AUTH}.history-02-producer_attempt_1_completed_private.json"


def test_append_marker_rejects_illegal_transition(tmp_path):
    reg = tmp_path / "registry"
    reg.mkdir()
    completed = _completed()
    seal_run_consumption_claim(reg, AUTH, _claim())
    seal_run_consumption_completed(reg, AUTH, completed)
    completed_receipt = _file_receipt(completed)
    with pytest.raises(ValueError):
        append_run_history_marker(
            reg,
            AUTH,
            _marker(
                "candidate_published", prior_marker_receipt=completed_receipt, run_identity_receipt=completed_receipt
            ),
            None,
        )
    with pytest.raises(ValueError):
        append_run_history_marker(
            reg,
            AUTH,
            _marker(
                "candidate_published",
                seq=2,
                prior_marker_receipt=completed_receipt,
                run_identity_receipt=completed_receipt,
            ),
            "producer_attempt_1_admitted",
        )


def test_append_marker_no_clobber(tmp_path):
    reg = tmp_path / "registry"
    reg.mkdir()
    completed = _completed()
    seal_run_consumption_claim(reg, AUTH, _claim())
    seal_run_consumption_completed(reg, AUTH, completed)
    completed_receipt = _file_receipt(completed)
    append_run_history_marker(
        reg,
        AUTH,
        _marker(
            "producer_attempt_1_admitted",
            prior_marker_receipt=completed_receipt,
            run_identity_receipt=completed_receipt,
        ),
        None,
    )
    with pytest.raises(ValueError, match="durable registry head"):
        append_run_history_marker(reg, AUTH, _marker("producer_attempt_1_admitted"), None)


def _admission(output_root="/x", claim_receipt=None):
    return _seal(
        {
            "schema_version": ADMISSION_SCHEMA,
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


def test_create_run_history_registry(tmp_path):
    from app.analysis.task0258_v2_registry import create_run_history_registry

    reg = tmp_path / "registry"
    reg.mkdir()
    out = tmp_path / "out" / "vru_causal_temporal_retrospective_v2"
    lock = tmp_path / "out" / ".lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("")

    claim = _claim(str(out))
    admission = _admission(str(out), _file_receipt(claim))
    completion = _completed(str(out), _file_receipt(claim), _file_receipt(admission))
    result = create_run_history_registry(
        registry_dir=reg,
        auth_sha256=AUTH,
        claim_payload=claim,
        admission_payload=admission,
        completion_payload=completion,
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
            claim_payload=claim,
            admission_payload=admission,
            completion_payload=completion,
            output_root=out,
            flock_path=lock,
        )
