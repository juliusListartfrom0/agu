#!/usr/bin/env python3
"""Probe macOS Endpoint Security build/signing readiness without sudo.

This is a capability report, not a read-isolation attestation.  It never starts
an Endpoint Security client, observes a worker, or claims user approval.
"""

from __future__ import annotations

import argparse
import json
import platform
import plistlib
import stat
import subprocess
import tempfile
from pathlib import Path

SOURCE = Path(__file__).with_name("endpoint_security") / "es_read_isolation_audit.c"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def derive_status(
    *,
    sdk_available: bool,
    compile_ok: bool,
    entitlement_present: bool,
    signature_kind: str | None = None,
) -> str:
    if not sdk_available:
        return "unavailable_sdk"
    if not compile_ok:
        return "unavailable_build"
    if not entitlement_present:
        return "blocked_external_authorization"
    if signature_kind != "signed":
        return "blocked_external_authorization"
    return "requires_external_user_approval"


def _checked_artifact_path(path: Path) -> Path:
    """Bind the selected artifact to a canonical, non-symlink path."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if candidate.is_symlink():
        raise ValueError(f"signed artifact path is a symlink: {candidate}")
    try:
        candidate = candidate.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(candidate) from None
    current = candidate
    while True:
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"signed artifact path contains a symlink: {current}")
        if current == candidate and not (stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode)):
            raise ValueError(f"signed artifact is not a file or bundle directory: {candidate}")
        if current.parent == current:
            break
        current = current.parent
    return candidate


def inspect_signed_artifact(path: Path) -> tuple[str, bool]:
    """Inspect a caller-selected signed executable or system-extension bundle."""
    path = _checked_artifact_path(Path(path))
    verification = _run("codesign", "--verify", "--deep", "--strict", "--verbose=4", str(path))
    if verification.returncode != 0:
        raise ValueError(f"codesign verification failed for {path}: {verification.stderr[-1000:]}")
    codesign = _run("codesign", "-dv", "--verbose=4", str(path))
    if codesign.returncode != 0:
        raise ValueError(f"codesign verification failed for {path}: {codesign.stderr[-1000:]}")
    details = f"{codesign.stdout}\n{codesign.stderr}"
    if "adhoc" in details or "linker-signed" in details:
        signature_kind = "adhoc"
    elif "Authority=" in details:
        signature_kind = "signed"
    else:
        signature_kind = "unknown"
    entitlements = _run("codesign", "-d", "--entitlements", ":-", str(path))
    if entitlements.returncode != 0:
        raise ValueError(f"codesign entitlement inspection failed for {path}: {entitlements.stderr[-1000:]}")
    entitlement_bytes = entitlements.stdout.encode("utf-8")
    if not entitlement_bytes.strip():
        entitlement_payload: object = {}
    else:
        try:
            entitlement_payload = plistlib.loads(entitlement_bytes)
        except (plistlib.InvalidFileException, ValueError, TypeError) as exc:
            raise ValueError(f"codesign entitlements are not a plist for {path}") from exc
    if not isinstance(entitlement_payload, dict):
        raise ValueError(f"codesign entitlements have an invalid shape for {path}")
    entitlement_present = entitlement_payload.get("com.apple.developer.endpoint-security.client") is True
    return signature_kind, entitlement_present


def build_capability_report(*, signed_artifact: Path | None = None) -> dict[str, object]:
    report: dict[str, object] = {
        "schema_version": "agu.task0258-endpoint-security-capability-report.v1",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "source_present": SOURCE.is_file(),
        "sdk_header_present": False,
        "sdk_stub_present": False,
        "compile_ok": False,
        "signature_kind": None,
        "endpoint_security_entitlement_present": False,
        "external_user_approval_observed": False,
        "kernel_read_isolation_attestation_observed": False,
        "status": "unavailable_sdk",
    }
    sdk_result = _run("xcrun", "--show-sdk-path")
    clang_result = _run("xcrun", "--find", "clang")
    if sdk_result.returncode != 0 or clang_result.returncode != 0:
        return report
    sdk_path = Path(sdk_result.stdout.strip())
    clang_path = clang_result.stdout.strip()
    header = sdk_path / "usr/include/EndpointSecurity/EndpointSecurity.h"
    stub = sdk_path / "usr/lib/libEndpointSecurity.tbd"
    report["sdk_header_present"] = header.is_file()
    report["sdk_stub_present"] = stub.is_file()
    sdk_available = bool(report["source_present"] and report["sdk_header_present"] and report["sdk_stub_present"])
    if not sdk_available:
        report["status"] = derive_status(sdk_available=False, compile_ok=False, entitlement_present=False)
        return report
    with tempfile.TemporaryDirectory(prefix="agu-task0258-es-") as temporary_directory:
        executable = Path(temporary_directory) / "es_read_isolation_audit"
        compile_result = _run(
            clang_path,
            "-O2",
            "-isysroot",
            str(sdk_path),
            "-I",
            str(sdk_path / "usr/include"),
            str(SOURCE),
            "-L",
            str(sdk_path / "usr/lib"),
            "-lEndpointSecurity",
            "-framework",
            "CoreFoundation",
            "-lbsm",
            "-o",
            str(executable),
        )
        report["compile_ok"] = compile_result.returncode == 0 and executable.is_file()
        if not report["compile_ok"]:
            report["status"] = derive_status(sdk_available=True, compile_ok=False, entitlement_present=False)
            report["compile_stderr"] = compile_result.stderr[-2000:]
            return report
        inspected_artifact = signed_artifact or executable
        try:
            signature_kind, entitlement_present = inspect_signed_artifact(inspected_artifact)
        except (FileNotFoundError, ValueError) as exc:
            report["signature_kind"] = "invalid"
            report["inspection_error"] = str(exc)
            report["status"] = "invalid_signed_artifact"
            return report
        report["signature_kind"] = signature_kind
        report["endpoint_security_entitlement_present"] = entitlement_present
        report["status"] = derive_status(
            sdk_available=True,
            compile_ok=True,
            entitlement_present=entitlement_present,
            signature_kind=signature_kind,
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a concise status line")
    parser.add_argument(
        "--signed-artifact",
        type=Path,
        help="inspect this signed executable or .systemextension instead of the temporary compile",
    )
    parser.add_argument("--require-ready", action="store_true", help="return nonzero unless entitlement is present")
    args = parser.parse_args()
    report = build_capability_report(signed_artifact=args.signed_artifact)
    if args.json:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        print(report["status"])
    return 0 if not args.require_ready or report["status"] == "requires_external_user_approval" else 1


if __name__ == "__main__":
    raise SystemExit(main())
