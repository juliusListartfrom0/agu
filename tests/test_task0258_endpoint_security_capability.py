"""Tests for the fail-closed Endpoint Security capability report."""

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
        derive_status(sdk_available=True, compile_ok=True, entitlement_present=True)
        == "requires_external_user_approval"
    )


def test_inspect_signed_artifact_reads_entitlement_from_the_requested_path(monkeypatch, tmp_path: Path):
    artifact = tmp_path / "audit.systemextension"
    artifact.mkdir()

    def fake_run(*args: str):
        if args[:3] == ("codesign", "-dv", "--verbose=4"):
            return capability.subprocess.CompletedProcess(
                args, 0, stdout="Executable=\nAuthority=Developer ID Application: AGU\n", stderr=""
            )
        if args[:4] == ("codesign", "-d", "--entitlements", ":-"):
            return capability.subprocess.CompletedProcess(
                args,
                0,
                stdout="<key>com.apple.developer.endpoint-security.client</key>\n<true/>",
                stderr="",
            )
        raise AssertionError(args)

    monkeypatch.setattr(capability, "_run", fake_run)
    assert capability.inspect_signed_artifact(artifact) == ("signed", True)


def test_inspect_signed_artifact_rejects_missing_path(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        capability.inspect_signed_artifact(tmp_path / "missing.systemextension")


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
