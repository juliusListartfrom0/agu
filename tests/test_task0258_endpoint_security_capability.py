"""Tests for the fail-closed Endpoint Security capability report."""

import plistlib
import sys
from pathlib import Path

import pytest

from scripts import task0258_endpoint_security_capability as capability
from scripts.task0258_endpoint_security_capability import derive_status


def test_endpoint_security_capability_statuses():
    assert derive_status(sdk_available=False, compile_ok=False, entitlement_present=False) == "unavailable_sdk"
    assert derive_status(sdk_available=True, compile_ok=False, entitlement_present=False) == "unavailable_build"
    assert (
        derive_status(sdk_available=True, compile_ok=True, entitlement_present=False)
        == "blocked_external_authorization"
    )
    assert (
        derive_status(
            sdk_available=True,
            compile_ok=True,
            entitlement_present=True,
            signature_kind="signed",
        )
        == "requires_external_user_approval"
    )


def test_endpoint_security_capability_rejects_entitlement_on_adhoc_artifact():
    assert (
        derive_status(
            sdk_available=True,
            compile_ok=True,
            entitlement_present=True,
            signature_kind="adhoc",
        )
        == "blocked_external_authorization"
    )


def test_inspect_signed_artifact_reads_entitlement_from_the_requested_path(monkeypatch, tmp_path: Path):
    artifact = tmp_path / "audit.systemextension"
    artifact.mkdir()

    def fake_run(*args: str):
        if args[:2] == ("codesign", "--verify"):
            return capability.subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args[:3] == ("codesign", "-dv", "--verbose=4"):
            return capability.subprocess.CompletedProcess(
                args, 0, stdout="Executable=\nAuthority=Developer ID Application: AGU\n", stderr=""
            )
        if args[:3] == ("codesign", "-d", "-r-"):
            return capability.subprocess.CompletedProcess(
                args, 0, stdout='designated => identifier "agu.audit" and anchor apple generic\n', stderr=""
            )
        if args[:4] == ("codesign", "-d", "--entitlements", ":-"):
            return capability.subprocess.CompletedProcess(
                args,
                0,
                stdout=plistlib.dumps({"com.apple.developer.endpoint-security.client": True}).decode("utf-8"),
                stderr="",
            )
        raise AssertionError(args)

    monkeypatch.setattr(capability, "_run", fake_run)
    assert capability.inspect_signed_artifact(artifact) == ("signed", True)


def test_inspect_signed_artifact_rejects_failed_strict_codesign(monkeypatch, tmp_path: Path):
    artifact = tmp_path / "audit.systemextension"
    artifact.mkdir()

    def fake_run(*args: str):
        assert args[:2] == ("codesign", "--verify")
        return capability.subprocess.CompletedProcess(args, 1, stdout="", stderr="invalid")

    monkeypatch.setattr(capability, "_run", fake_run)
    with pytest.raises(ValueError, match="verification failed"):
        capability.inspect_signed_artifact(artifact)


def test_inspect_signed_artifact_requires_true_entitlement_value(monkeypatch, tmp_path: Path):
    artifact = tmp_path / "audit.systemextension"
    artifact.mkdir()

    def fake_run(*args: str):
        if args[:2] == ("codesign", "--verify"):
            return capability.subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args[:3] == ("codesign", "-dv", "--verbose=4"):
            return capability.subprocess.CompletedProcess(args, 0, stdout="Authority=AGU\n", stderr="")
        if args[:3] == ("codesign", "-d", "-r-"):
            return capability.subprocess.CompletedProcess(
                args, 0, stdout='designated => identifier "agu.audit" and anchor apple generic\n', stderr=""
            )
        if args[:4] == ("codesign", "-d", "--entitlements", ":-"):
            return capability.subprocess.CompletedProcess(
                args,
                0,
                stdout=plistlib.dumps({"com.apple.developer.endpoint-security.client": False}).decode("utf-8"),
                stderr="",
            )
        raise AssertionError(args)

    monkeypatch.setattr(capability, "_run", fake_run)
    assert capability.inspect_signed_artifact(artifact) == ("signed", False)


def test_inspect_signed_artifact_rejects_self_signed_authority_with_entitlement(monkeypatch, tmp_path: Path):
    artifact = tmp_path / "audit.systemextension"
    artifact.mkdir()

    def fake_run(*args: str):
        if args[:2] == ("codesign", "--verify"):
            return capability.subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args[:3] == ("codesign", "-dv", "--verbose=4"):
            return capability.subprocess.CompletedProcess(args, 0, stdout="Authority=AGU\n", stderr="")
        if args[:3] == ("codesign", "-d", "-r-"):
            return capability.subprocess.CompletedProcess(
                args, 0, stdout='designated => identifier "agu.audit"\n', stderr=""
            )
        if args[:4] == ("codesign", "-d", "--entitlements", ":-"):
            return capability.subprocess.CompletedProcess(
                args,
                0,
                stdout=plistlib.dumps({"com.apple.developer.endpoint-security.client": True}).decode("utf-8"),
                stderr="",
            )
        raise AssertionError(args)

    monkeypatch.setattr(capability, "_run", fake_run)
    assert capability.inspect_signed_artifact(artifact) == ("untrusted", True)


def test_inspect_signed_artifact_rejects_missing_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        capability.inspect_signed_artifact(tmp_path / "missing.systemextension")


def test_inspect_signed_artifact_rejects_symlinked_leaf(tmp_path: Path):
    artifact = tmp_path / "real-audit.systemextension"
    artifact.mkdir()
    symlink_artifact = tmp_path / "audit.systemextension"
    symlink_artifact.symlink_to(artifact, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        capability.inspect_signed_artifact(symlink_artifact)


def test_main_passes_signed_artifact_to_report_builder(monkeypatch, capsys, tmp_path: Path):
    signed_artifact = tmp_path / "audit.systemextension"
    signed_artifact.mkdir()
    captured: dict[str, Path | None] = {}

    def fake_report(*, signed_artifact: Path | None = None):
        captured["path"] = signed_artifact
        return {"status": "requires_external_user_approval"}

    monkeypatch.setattr(capability, "build_capability_report", fake_report)
    monkeypatch.setattr(sys, "argv", ["probe", "--json", "--signed-artifact", str(signed_artifact)])
    assert capability.main() == 0
    assert captured["path"] == signed_artifact
    assert '"status":"requires_external_user_approval"' in capsys.readouterr().out


def test_require_capability_accepts_pending_user_approval(monkeypatch, capsys, tmp_path: Path):
    signed_artifact = tmp_path / "audit.systemextension"
    signed_artifact.mkdir()

    monkeypatch.setattr(
        capability,
        "build_capability_report",
        lambda *, signed_artifact=None: {"status": "requires_external_user_approval"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["probe", "--json", "--require-capability", "--signed-artifact", str(signed_artifact)],
    )
    assert capability.main() == 0
    assert '"status":"requires_external_user_approval"' in capsys.readouterr().out


def test_endpoint_security_source_bounds_output_and_process_lineage():
    source = capability.SOURCE.read_text(encoding="utf-8")

    assert "O_NOFOLLOW" in source
    assert "O_EXCL" in source
    assert "openat" in source
    assert "audit_token_to_pidversion" in source
    assert "ES_EVENT_TYPE_NOTIFY_FORK" in source
    assert "ES_EVENT_TYPE_NOTIFY_EXIT" in source
    assert "--timeout-seconds" in source
    assert "alarm(" in source
    assert "msg->action.notify" in source
    assert "global_seq_num" in source
    assert "seq_num" in source
    assert "ES_ACTION_TYPE_NOTIFY" in source
    assert "sequence_gap" in source
    assert '"result":"notify"' not in source
    assert "return 5" in source
    assert "fsync(fileno(g_out))" in source
    assert "write_final_row" in source
    assert '\\"record_type\\":\\"final\\"' in source
    assert "append_json_escaped" in source
    assert "path_truncated" in source
