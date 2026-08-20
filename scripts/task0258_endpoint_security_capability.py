#!/usr/bin/env python3
"""Probe macOS Endpoint Security build/signing readiness without sudo.

This is a capability report, not a read-isolation attestation.  It never starts
an Endpoint Security client, observes a worker, or claims user approval.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import tempfile
from pathlib import Path

SOURCE = Path(__file__).with_name("endpoint_security") / "es_read_isolation_audit.c"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def derive_status(*, sdk_available: bool, compile_ok: bool, entitlement_present: bool) -> str:
    if not sdk_available:
        return "unavailable_sdk"
    if not compile_ok:
        return "unavailable_build"
    if not entitlement_present:
        return "blocked_external_authorization"
    return "requires_external_user_approval"


def build_capability_report() -> dict[str, object]:
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
        codesign = _run("codesign", "-dv", "--verbose=4", str(executable))
        details = f"{codesign.stdout}\n{codesign.stderr}"
        if "adhoc" in details or "linker-signed" in details:
            report["signature_kind"] = "adhoc"
        elif "Authority=" in details:
            report["signature_kind"] = "signed"
        else:
            report["signature_kind"] = "unknown"
        entitlements = _run("codesign", "-d", "--entitlements", ":-", str(executable))
        entitlement_text = f"{entitlements.stdout}\n{entitlements.stderr}"
        report["endpoint_security_entitlement_present"] = bool(
            re.search(r"com\.apple\.developer\.endpoint-security\.client", entitlement_text)
        )
        report["status"] = derive_status(
            sdk_available=True,
            compile_ok=True,
            entitlement_present=bool(report["endpoint_security_entitlement_present"]),
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a concise status line")
    parser.add_argument("--require-ready", action="store_true", help="return nonzero unless entitlement is present")
    args = parser.parse_args()
    report = build_capability_report()
    if args.json:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        print(report["status"])
    return 0 if not args.require_ready or report["status"] == "requires_external_user_approval" else 1


if __name__ == "__main__":
    raise SystemExit(main())
