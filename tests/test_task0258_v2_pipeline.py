"""Tests for the TASK-0258 v2 pipeline candidate + bundle sealing."""

from __future__ import annotations

import json
import os

import pytest
from task0258_candidate_fixtures import candidate_members

from app.analysis import task0258_v2_pipeline as pipeline_module
from app.analysis.task0258_module_a_v2 import canonical_artifact_sha256, compact_canonical_json
from app.analysis.task0258_v2_artifacts import CANDIDATE_MEMBER_PATHS
from app.analysis.task0258_v2_pipeline import (
    build_candidate_gate_payload,
    build_candidate_receipt_bundle_payload,
    build_member_receipts,
    read_published_candidate_receipt_bundle,
    seal_candidate_receipt_bundle,
    seal_candidate_v2,
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
    return candidate_members(_gate())


def test_candidate_gate_payload():
    gate = _gate()
    assert gate["publication_state"] == "not_yet_observed"
    assert gate["postpublication_verification_required"] is True
    assert len(gate["ordered_prepublication_check_results"]) == 24
    assert len(gate["artifact_sha256"]) == 64


def test_candidate_gate_verification_receipt_is_bound_to_member_bytes():
    members = _make_members()
    gate = json.loads(members["candidate_gate.json"])
    gate["verification_attempt_receipt"] = _receipt()
    gate["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in gate.items() if key != "artifact_sha256"}
    )
    members["candidate_gate.json"] = (compact_canonical_json(gate) + "\n").encode()

    from app.analysis.task0258_v2_pipeline_cli import assemble_candidate_members

    with pytest.raises(ValueError, match="verification attempt receipt"):
        assemble_candidate_members(members)


def test_candidate_publication_rejects_schema_valid_but_unauthorized_member(tmp_path):
    members = _make_members()
    forged = {"schema_version": "agu.test-member.v1", "artifact_sha256": "0" * 64}
    forged["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in forged.items() if key != "artifact_sha256"}
    )
    members["temporal_retrospective.json"] = (compact_canonical_json(forged) + "\n").encode()
    out = tmp_path / "out"
    out.mkdir()
    lock = tmp_path / ".task0258-output.lock"
    lock.write_text("")
    with pytest.raises(ValueError, match="artifact field set mismatch|parent artifact identity"):
        seal_candidate_v2(out, members, flock_path=lock)


def test_seal_candidate_v2_and_member_receipts(tmp_path, monkeypatch):
    members = _make_members()
    out = tmp_path / "out"
    out.mkdir()
    lock = out.parent / ".task0258-output.lock"
    lock.write_text("")
    final = seal_candidate_v2(out, members, flock_path=lock)
    assert final == out / "candidate_v2"
    assert final.is_dir()

    def fail_path_read_bytes(_path):
        raise AssertionError("candidate receipts must use bounded no-follow reads")

    monkeypatch.setattr(type(final / CANDIDATE_MEMBER_PATHS[0]), "read_bytes", fail_path_read_bytes)
    rows = build_member_receipts(final)
    assert [r["relative_path"] for r in rows] == list(CANDIDATE_MEMBER_PATHS)
    jsonl = [r for r in rows if r["receipt_kind"] == "file_only"]
    json_rows = [r for r in rows if r["receipt_kind"] == "json"]
    assert len(jsonl) == 2
    assert len(json_rows) == 8
    for r in json_rows:
        assert len(r["artifact_sha256"]) == 64
        assert r["internal_sha256_field"] == "artifact_sha256"


def test_candidate_publication_binds_authorization_stage_and_cleans_it(tmp_path):
    members = _make_members()
    out = tmp_path / "out"
    out.mkdir()
    lock = out.parent / ".task0258-output.lock"
    lock.write_text("")

    final = seal_candidate_v2(out, members, flock_path=lock)

    authorization_sha256 = _gate()["input_receipts"][3]["receipt"]["artifact_sha256"]
    assert final == out / "candidate_v2"
    assert not (out / f".{authorization_sha256}.candidate-v2-stage").exists()


def test_candidate_publication_rejects_fixed_stage_residue(tmp_path):
    members = _make_members()
    out = tmp_path / "out"
    out.mkdir()
    lock = out.parent / ".task0258-output.lock"
    lock.write_text("")
    authorization_sha256 = _gate()["input_receipts"][3]["receipt"]["artifact_sha256"]
    stage = out / f".{authorization_sha256}.candidate-v2-stage"
    stage.write_bytes(b"residue")

    with pytest.raises(FileExistsError):
        seal_candidate_v2(out, members, flock_path=lock)

    assert not (out / "candidate_v2").exists()
    assert stage.read_bytes() == b"residue"


def test_candidate_publication_rejects_existing_terminal_generation(tmp_path):
    members = _make_members()
    out = tmp_path / "out"
    out.mkdir()
    lock = out.parent / ".task0258-output.lock"
    lock.write_text("")
    (out / "verified_result_v2").mkdir()

    with pytest.raises(ValueError, match="candidate publication topology"):
        seal_candidate_v2(out, members, flock_path=lock)

    assert not (out / "candidate_v2").exists()


def test_candidate_receipt_bundle(tmp_path, monkeypatch):
    members = _make_members()
    out = tmp_path / "out"
    out.mkdir()
    lock = out.parent / ".task0258-output.lock"
    lock.write_text("")
    final = seal_candidate_v2(out, members, flock_path=lock)
    monkeypatch.setattr(
        pipeline_module,
        "exclusive_flock",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("path-based flock is not allowed")),
        raising=False,
    )
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
    bundle_path = tmp_path / "candidate-receipt-bundle.json"
    seal_candidate_receipt_bundle(
        bundle_path,
        bundle,
        candidate_dir=final,
        output_flock_path=lock,
    )
    assert bundle_path.is_file()
    assert json.loads(bundle_path.read_text())["candidate_generation_name"] == "candidate_v2"
    assert (
        read_published_candidate_receipt_bundle(
            bundle_path,
            candidate_dir=final,
            output_flock_path=lock,
            expected_payload=bundle,
        )
        == (compact_canonical_json(bundle) + "\n").encode()
    )
    with pytest.raises(ValueError, match="fixed output-parent lock"):
        read_published_candidate_receipt_bundle(
            bundle_path,
            candidate_dir=final,
            output_flock_path=out / ".caller-selected.lock",
            expected_payload=bundle,
        )
    stage_path = (
        tmp_path / f".{bundle['authorization_receipt']['artifact_sha256']}.{bundle_path.name}.task0258-bundle-stage"
    )
    assert not stage_path.exists()


def test_candidate_receipt_bundle_rejects_bundle_parent_replacement(tmp_path, monkeypatch):
    members = _make_members()
    out = tmp_path / "out"
    out.mkdir()
    lock = tmp_path / ".task0258-output.lock"
    lock.write_text("")
    final = seal_candidate_v2(out, members, flock_path=lock)
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    bundle_path = bundle_dir / "candidate-receipt-bundle.json"
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
    original_write = pipeline_module.atomic_write_bytes_at
    moved_bundle_dir = tmp_path / "bundle-moved"

    def write_then_replace_parent(*args, **kwargs):
        result = original_write(*args, **kwargs)
        bundle_dir.rename(moved_bundle_dir)
        bundle_dir.mkdir()
        return result

    monkeypatch.setattr(pipeline_module, "atomic_write_bytes_at", write_then_replace_parent)
    with pytest.raises(ValueError, match="bundle parent"):
        seal_candidate_receipt_bundle(bundle_path, bundle, candidate_dir=final, output_flock_path=lock)
    assert not bundle_path.exists()
    assert (moved_bundle_dir / bundle_path.name).is_file()


def test_candidate_receipt_bundle_replay_rejects_bundle_parent_replacement(tmp_path, monkeypatch):
    members = _make_members()
    out = tmp_path / "out"
    out.mkdir()
    lock = tmp_path / ".task0258-output.lock"
    lock.write_text("")
    final = seal_candidate_v2(out, members, flock_path=lock)
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    bundle_path = bundle_dir / "candidate-receipt-bundle.json"
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
    seal_candidate_receipt_bundle(bundle_path, bundle, candidate_dir=final, output_flock_path=lock)
    original_open = pipeline_module._open_existing_directory_no_follow
    moved_bundle_dir = tmp_path / "bundle-moved"

    def open_then_replace_parent(path):
        fd = original_open(path)
        if path == bundle_dir and not moved_bundle_dir.exists():
            bundle_dir.rename(moved_bundle_dir)
            bundle_dir.mkdir()
        return fd

    monkeypatch.setattr(pipeline_module, "_open_existing_directory_no_follow", open_then_replace_parent)
    with pytest.raises(ValueError, match="bundle parent"):
        read_published_candidate_receipt_bundle(
            bundle_path,
            candidate_dir=final,
            output_flock_path=lock,
            expected_payload=bundle,
        )
    assert not bundle_path.exists()
    assert (moved_bundle_dir / bundle_path.name).is_file()


def test_candidate_receipt_bundle_rejects_output_root_replacement(tmp_path, monkeypatch):
    members = _make_members()
    out = tmp_path / "out"
    out.mkdir()
    lock = tmp_path / ".task0258-output.lock"
    lock.write_text("")
    final = seal_candidate_v2(out, members, flock_path=lock)
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    bundle_path = bundle_dir / "candidate-receipt-bundle.json"
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
    original_open = pipeline_module._open_directory_at
    moved_out = tmp_path / "out-moved"
    calls = 0

    def open_then_replace_output_root(parent_fd, name):
        nonlocal calls
        fd = original_open(parent_fd, name)
        calls += 1
        if calls == 2:
            out.rename(moved_out)
            out.mkdir()
        return fd

    monkeypatch.setattr(pipeline_module, "_open_directory_at", open_then_replace_output_root)
    with pytest.raises(ValueError, match="output root|candidate directory"):
        seal_candidate_receipt_bundle(bundle_path, bundle, candidate_dir=final, output_flock_path=lock)
    assert not bundle_path.exists()
    assert (moved_out / "candidate_v2").is_dir()


def test_candidate_receipt_bundle_replay_rejects_output_root_replacement(tmp_path, monkeypatch):
    members = _make_members()
    out = tmp_path / "out"
    out.mkdir()
    lock = tmp_path / ".task0258-output.lock"
    lock.write_text("")
    final = seal_candidate_v2(out, members, flock_path=lock)
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    bundle_path = bundle_dir / "candidate-receipt-bundle.json"
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
    seal_candidate_receipt_bundle(bundle_path, bundle, candidate_dir=final, output_flock_path=lock)
    original_open = pipeline_module._open_directory_at
    moved_out = tmp_path / "out-moved"
    calls = 0

    def open_then_replace_output_root(parent_fd, name):
        nonlocal calls
        fd = original_open(parent_fd, name)
        calls += 1
        if calls == 2:
            out.rename(moved_out)
            out.mkdir()
        return fd

    monkeypatch.setattr(pipeline_module, "_open_directory_at", open_then_replace_output_root)
    with pytest.raises(ValueError, match="output root|candidate directory"):
        read_published_candidate_receipt_bundle(
            bundle_path,
            candidate_dir=final,
            output_flock_path=lock,
            expected_payload=bundle,
        )
    assert bundle_path.is_file()
    assert (moved_out / "candidate_v2").is_dir()


def test_candidate_receipt_bundle_closes_fds_when_candidate_open_fails(tmp_path):
    if not os.path.isdir("/dev/fd"):
        pytest.skip("/dev/fd is unavailable")
    members = _make_members()
    out = tmp_path / "out"
    out.mkdir()
    lock = tmp_path / ".task0258-output.lock"
    lock.write_text("")
    final = seal_candidate_v2(out, members, flock_path=lock)
    bundle_path = tmp_path / "candidate-receipt-bundle.json"
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
    seal_candidate_receipt_bundle(bundle_path, bundle, candidate_dir=final, output_flock_path=lock)
    moved_candidate = tmp_path / "candidate-moved"
    final.rename(moved_candidate)
    before = len(os.listdir("/dev/fd"))
    for _ in range(25):
        with pytest.raises(ValueError):
            read_published_candidate_receipt_bundle(
                bundle_path,
                candidate_dir=final,
                output_flock_path=lock,
                expected_payload=bundle,
            )
    after = len(os.listdir("/dev/fd"))
    assert after <= before + 2


def test_published_candidate_receipt_bundle_rejects_path_and_payload_drift(tmp_path):
    members = _make_members()
    out = tmp_path / "out"
    out.mkdir()
    lock = out.parent / ".task0258-output.lock"
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
    bundle_path = tmp_path / "candidate-receipt-bundle.json"
    seal_candidate_receipt_bundle(bundle_path, bundle, candidate_dir=final, output_flock_path=lock)

    drifted = dict(bundle)
    drifted["observed_at_utc"] = "2026-08-17T00:00:01Z"
    drifted["artifact_sha256"] = canonical_artifact_sha256(
        {key: value for key, value in drifted.items() if key != "artifact_sha256"}
    )
    bundle_path.write_bytes((compact_canonical_json(drifted) + "\n").encode())
    with pytest.raises(ValueError, match="does not match the expected payload"):
        read_published_candidate_receipt_bundle(
            bundle_path,
            candidate_dir=final,
            output_flock_path=lock,
            expected_payload=bundle,
        )

    replacement = tmp_path / "replacement.json"
    replacement.write_bytes((compact_canonical_json(bundle) + "\n").encode())
    bundle_path.unlink()
    bundle_path.symlink_to(replacement)
    with pytest.raises(ValueError, match="without following links"):
        read_published_candidate_receipt_bundle(
            bundle_path,
            candidate_dir=final,
            output_flock_path=lock,
            expected_payload=bundle,
        )


def test_candidate_receipt_bundle_rejects_locked_candidate_drift(tmp_path):
    members = _make_members()
    out = tmp_path / "out"
    out.mkdir()
    lock = out.parent / ".task0258-output.lock"
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
    mutated = final / "temporal_retrospective.json"
    mutated.write_bytes(mutated.read_bytes() + b" ")

    with pytest.raises(ValueError, match="candidate JSON member|locked candidate bytes"):
        seal_candidate_receipt_bundle(
            tmp_path / "candidate-receipt-bundle.json",
            bundle,
            candidate_dir=final,
            output_flock_path=lock,
        )


def test_candidate_receipt_bundle_rejects_noncanonical_output_lock(tmp_path):
    members = _make_members()
    out = tmp_path / "out"
    out.mkdir()
    canonical_lock = tmp_path / ".task0258-output.lock"
    final = seal_candidate_v2(out, members, flock_path=canonical_lock)
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
    with pytest.raises(ValueError, match="output-parent lock"):
        seal_candidate_receipt_bundle(
            tmp_path / "candidate-receipt-bundle.json",
            bundle,
            candidate_dir=final,
            output_flock_path=out / ".caller-selected.lock",
        )


def test_candidate_receipt_bundle_rejects_fixed_stage_residue(tmp_path):
    members = _make_members()
    out = tmp_path / "out"
    out.mkdir()
    lock = out.parent / ".task0258-output.lock"
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
    bundle_path = tmp_path / "candidate-receipt-bundle.json"
    stage_path = (
        tmp_path / f".{bundle['authorization_receipt']['artifact_sha256']}.{bundle_path.name}.task0258-bundle-stage"
    )
    stage_path.write_bytes(b"residue")

    with pytest.raises(FileExistsError):
        seal_candidate_receipt_bundle(
            bundle_path,
            bundle,
            candidate_dir=final,
            output_flock_path=lock,
        )

    assert not bundle_path.exists()
    assert stage_path.read_bytes() == b"residue"


def _result(error_bounds_pass=True):
    from app.analysis.task0258_v2_pipeline import build_postpublication_verification_payload

    member_rows = [_member_row(path) for path in CANDIDATE_MEMBER_PATHS]
    member_by_path = {row["relative_path"]: row for row in member_rows}

    return build_postpublication_verification_payload(
        error_bounds_pass=error_bounds_pass,
        authorization_receipts={
            "parent_spec_approval": _receipt(),
            "amendment_implementation_approval": _receipt(),
            "amended_implementation_review": _receipt(),
            "rerun_authorization": _receipt(),
        },
        run_history_contract_receipt={
            "run_identity_receipt": _receipt(),
            "head_receipt": _receipt(),
            "marker_count": 0,
        },
        run_admission_receipt=_receipt(),
        static_input_contract={
            "temporal_plan_artifact_sha256": "0" * 64,
            "temporal_plan_file_sha256": "0" * 64,
            "task0257_receipts_projection_sha256": "0" * 64,
        },
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
    lock = out.parent / ".task0258-output.lock"
    lock.write_text("")
    candidate = seal_candidate_v2(out, _make_members(), flock_path=lock)
    assert candidate == out / "candidate_v2"
    final = seal_verified_result(out, _result(True), flock_path=lock)
    assert final == out / "verified_result_v2"
    assert (final / "verification_registry.json").is_file()
    assert not (
        out
        / f".{_result(True)['authorization_receipts']['rerun_authorization']['artifact_sha256']}.verified-result-v2-stage"
    ).exists()

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
    with pytest.raises(ValueError, match="terminal publication topology"):
        seal_postverification_failure(out, failure, flock_path=lock)
    assert not (out / "postverification_failure_v2").exists()

    failure_out = tmp_path / "failure-out"
    failure_out.mkdir()
    failure_lock = failure_out.parent / ".task0258-output.lock"
    failure_lock.write_text("")
    seal_candidate_v2(failure_out, _make_members(), flock_path=failure_lock)
    fail_dir = seal_postverification_failure(failure_out, failure, flock_path=failure_lock)
    assert fail_dir == failure_out / "postverification_failure_v2"
    assert (fail_dir / "failure.json").is_file()
    assert not (failure_out / f".{slots[3]['receipt']['artifact_sha256']}.postverification-failure-v2-stage").exists()
