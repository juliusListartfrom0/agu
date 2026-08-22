"""Tests for the TASK-0258 v2 run-history registry durable write path."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError

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
    replay_run_history_registry,
)
from app.analysis.task0258_v2_fs import _open_existing_directory_no_follow, exclusive_flock, exclusive_flock_at
from app.analysis.task0258_v2_pipeline import output_parent_flock_path
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


@pytest.mark.parametrize(
    ("sealer", "payload_factory"),
    [
        (seal_run_consumption_claim, _claim),
        (seal_run_consumption_completed, _completed),
    ],
)
def test_registry_sealers_reject_lock_parent_fd_drift(tmp_path, sealer, payload_factory):
    registry = tmp_path / "registry"
    registry.mkdir()
    moved_registry = tmp_path / "moved-registry"
    locked_registry_fd = _open_existing_directory_no_follow(registry)
    lock_path = registry / f".{AUTH}.history.lock"
    try:
        with exclusive_flock_at(locked_registry_fd, lock_path.name, lock_path) as history_lock:
            registry.rename(moved_registry)
            registry.mkdir()
            write_registry_fd = _open_existing_directory_no_follow(registry)
            try:
                with pytest.raises(ValueError, match="lock parent"):
                    sealer(
                        registry,
                        AUTH,
                        payload_factory(),
                        held_lock=history_lock,
                        registry_fd=write_registry_fd,
                    )
            finally:
                os.close(write_registry_fd)
    finally:
        os.close(locked_registry_fd)


def test_registry_sealer_acquires_history_lock_when_not_supplied(tmp_path):
    registry = tmp_path / "registry"
    registry.mkdir()
    lock_path = registry / f".{AUTH}.history.lock"
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        with exclusive_flock(lock_path):
            future = executor.submit(seal_run_consumption_claim, registry, AUTH, _claim())
            with pytest.raises(TimeoutError):
                future.result(timeout=0.1)
        assert future.result(timeout=2) == registry / claim_filename(AUTH)
    finally:
        executor.shutdown(wait=True)


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


def test_append_marker_rejects_forged_predecessor_receipt(tmp_path):
    reg = tmp_path / "registry"
    reg.mkdir()
    completed = _completed()
    seal_run_consumption_claim(reg, AUTH, _claim())
    seal_run_consumption_completed(reg, AUTH, completed)
    completed_receipt = _file_receipt(completed)
    forged = _marker(
        "producer_attempt_1_admitted",
        prior_marker_receipt={"artifact_sha256": "1" * 64, "file_sha256": "1" * 64},
        run_identity_receipt=completed_receipt,
    )

    with pytest.raises(ValueError, match="predecessor receipt"):
        append_run_history_marker(reg, AUTH, forged, None)

    assert not (reg / f"{AUTH}.history-01-producer_attempt_1_admitted.json").exists()


def test_append_marker_rejects_identity_binding_drift(tmp_path):
    reg = tmp_path / "registry"
    reg.mkdir()
    completed = _completed()
    seal_run_consumption_claim(reg, AUTH, _claim())
    seal_run_consumption_completed(reg, AUTH, completed)
    completed_receipt = _file_receipt(completed)
    forged = _marker(
        "producer_attempt_1_admitted",
        prior_marker_receipt=completed_receipt,
        run_identity_receipt=completed_receipt,
    )
    forged["output_root"] = "/different-root"
    forged["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in forged.items() if key != "artifact_sha256"}
    )

    with pytest.raises(ValueError, match="identity/root binding"):
        append_run_history_marker(reg, AUTH, forged, None)

    assert not (reg / f"{AUTH}.history-01-producer_attempt_1_admitted.json").exists()
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


def test_append_marker_rejects_invalid_authorization_before_lock_path(tmp_path):
    reg = tmp_path / "registry"
    reg.mkdir()
    escaped_lock = tmp_path / "escaped.history.lock"

    with pytest.raises(ValueError):
        append_run_history_marker(
            reg,
            "/../escaped",
            _marker("producer_attempt_1_admitted"),
            None,
        )

    assert not escaped_lock.exists()
    assert list(reg.iterdir()) == []


def test_append_marker_rejects_symlinked_lock_without_touching_target(tmp_path):
    reg = tmp_path / "registry"
    reg.mkdir()
    claim = _claim()
    completed = _completed()
    seal_run_consumption_claim(reg, AUTH, claim)
    seal_run_consumption_completed(reg, AUTH, completed)

    target = tmp_path / "lock-target"
    target.write_text("sentinel")
    fixed_ns = 123456789000000000
    os.utime(target, ns=(fixed_ns, fixed_ns))
    lock_path = reg / f".{AUTH}.history.lock"
    lock_path.unlink()
    lock_path.symlink_to(target)

    with pytest.raises(OSError):
        append_run_history_marker(reg, AUTH, _marker("producer_attempt_1_admitted"), None)

    assert target.read_text() == "sentinel"
    assert target.stat().st_mtime_ns == fixed_ns
    assert lock_path.is_symlink()


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
    out = tmp_path / "out" / "vru_causal_temporal_retrospective_v2"
    lock = output_parent_flock_path(out)
    lock.parent.mkdir(parents=True, exist_ok=True)

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
    replay_run_history_registry(reg, AUTH)
    (reg / "unexpected.tmp").write_text("residue")
    with pytest.raises(ValueError, match="non-JSON residue"):
        replay_run_history_registry(reg, AUTH)
    (reg / "unexpected.tmp").unlink()
    completion_path = reg / f"{AUTH}.completed.json"
    completion_path.unlink()
    completion_path.symlink_to(reg / f"{AUTH}.claim.json")
    with pytest.raises(ValueError, match="without following links"):
        replay_run_history_registry(reg, AUTH)
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


def test_create_run_history_registry_rejects_noncanonical_output_lock(tmp_path):
    from app.analysis.task0258_v2_registry import create_run_history_registry

    reg = tmp_path / "registry"
    out = tmp_path / "out" / "vru_causal_temporal_retrospective_v2"
    arbitrary_lock = tmp_path / "out" / ".caller-selected.lock"
    arbitrary_lock.parent.mkdir(parents=True, exist_ok=True)

    claim = _claim(str(out))
    admission = _admission(str(out), _file_receipt(claim))
    completion = _completed(str(out), _file_receipt(claim), _file_receipt(admission))
    with pytest.raises(ValueError, match="fixed output-parent lock"):
        create_run_history_registry(
            registry_dir=reg,
            auth_sha256=AUTH,
            claim_payload=claim,
            admission_payload=admission,
            completion_payload=completion,
            output_root=out,
            flock_path=arbitrary_lock,
        )

    assert not reg.exists()
    assert not out.exists()


def test_replay_rejects_symlinked_registry_ancestor(tmp_path):
    real_parent = tmp_path / "real-parent"
    registry = real_parent / "registry"
    registry.mkdir(parents=True)
    claim = _claim()
    completed = _completed()
    seal_run_consumption_claim(registry, AUTH, claim)
    seal_run_consumption_completed(registry, AUTH, completed)

    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="without following links"):
        replay_run_history_registry(linked_parent / "registry", AUTH)
