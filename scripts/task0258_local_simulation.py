#!/usr/bin/env python3
"""Run the repository-local TASK-0258 simulation without minting evidence.

The simulation exercises the review-only synthetic contexts and the canonical
manifest loader.  It also includes the real platform capability probe, but
copies its result into an explicitly diagnostic boundary.  This command can
never report P5 readiness, user approval, or kernel read-isolation evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tempfile
from pathlib import Path

from app.analysis.task0258_module_a_v2 import canonical_artifact_sha256, compact_canonical_json
from app.analysis.task0258_v2_audit import parse_endpoint_security_transcript
from app.analysis.task0258_v2_capabilities import (
    bind_implementation_review_discovery_context,
    bind_implementation_review_sandbox_context,
    bind_synthetic_module_a_discovery_context,
    bind_synthetic_module_a_transaction_context,
    exercise_module_a_v2_state_machine_for_discovery,
    exercise_module_a_v2_state_machine_for_review,
    replay_module_a_read_traversal_for_discovery,
)
from scripts.task0258_endpoint_security_capability import build_capability_report

SCHEMA_VERSION = "agu.task0258-local-simulation-report.v1"
SIMULATION_SCENARIO = {
    "scenario_id": "claim_then_publish",
    "steps": [
        {"state": "fresh", "event": "claim"},
        {"state": "claimed", "event": "review"},
        {"state": "reviewed", "event": "publish"},
    ],
}


def _write_manifest(root: Path) -> tuple[Path, str, str]:
    payload: dict[str, object] = {"schema_version": "agu.task0258-local-simulation-manifest.v1"}
    payload["artifact_sha256"] = canonical_artifact_sha256(payload)
    raw = (compact_canonical_json(payload) + "\n").encode("utf-8")
    path = root / "manifest.json"
    path.write_bytes(raw)
    return path, payload["artifact_sha256"], hashlib.sha256(raw).hexdigest()


def _diagnostic_platform_boundary(report: dict[str, object]) -> dict[str, object]:
    """Copy only diagnostic fields; never promote the probe to P5 evidence."""
    return {
        "status": report.get("status"),
        "signature_kind": report.get("signature_kind"),
        "compile_ok": report.get("compile_ok", False),
        "endpoint_security_entitlement_present": report.get("endpoint_security_entitlement_present", False),
        "external_user_approval_observed": report.get("external_user_approval_observed", False),
        "kernel_read_isolation_attestation_observed": report.get("kernel_read_isolation_attestation_observed", False),
        "evidence_class": "diagnostic_only",
        "production_capability": False,
        "p5_ready": False,
    }


def _simulate_endpoint_security_diagnostic() -> dict[str, object]:
    """Exercise the C-shaped JSONL projection without claiming kernel evidence."""
    event_rows = (
        '{"event":"open","pid":42,"pidversion":7,"ppid":1,"seq_num":10,'
        '"global_seq_num":100,"path":"/private/tmp/simulation-input.json",'
        '"result_type":"auth","result_auth":"allow"}\n'
        '{"event":"fork","pid":43,"pidversion":8,"ppid":42,"seq_num":11,'
        '"global_seq_num":102,"path":null,"result_type":"flags","result_flags":3}\n'
    )
    finalization = {
        "record_type": "final",
        "rows": 2,
        "bytes": len(event_rows.encode("utf-8")),
        "overflow": False,
        "sequence_gap": False,
        "protocol_error": False,
        "timed_out": False,
    }
    transcript = event_rows + json.dumps(finalization, separators=(",", ":")) + "\n"
    events = parse_endpoint_security_transcript(io.StringIO(transcript))
    return {
        "event_count": len(events),
        "finalization_verified": True,
        "evidence_class": "diagnostic_only",
        "production_capability": False,
    }


def build_local_simulation_report() -> dict[str, object]:
    """Exercise local synthetic paths and return an explicitly non-authorizing report."""
    command_sha256 = hashlib.sha256(b"task0258-local-simulation").hexdigest()
    with tempfile.TemporaryDirectory(prefix="agu-task0258-simulation-") as temporary_directory:
        # macOS may expose the temporary root through a `/var` symlink. The
        # review-only context requires a canonical no-symlink path, so resolve
        # the freshly created directory before binding it.
        root = Path(temporary_directory).resolve()
        manifest_path, manifest_artifact_sha256, manifest_file_sha256 = _write_manifest(root)
        discovery = bind_implementation_review_discovery_context(
            expected_check_name="focused_pytest", expected_command_sha256=command_sha256
        )
        review = bind_implementation_review_sandbox_context(
            expected_check_name="focused_pytest", expected_command_sha256=command_sha256
        )
        discovery_observation = replay_module_a_read_traversal_for_discovery(
            discovery_context=discovery,
            operation="manifest_replay",
            operation_input_manifest_path=manifest_path,
            expected_manifest_artifact_sha256=manifest_artifact_sha256,
            expected_manifest_file_sha256=manifest_file_sha256,
        )
        discovery_context = bind_synthetic_module_a_discovery_context(
            discovery_context=discovery, isolated_temp_ancestor=root
        )
        review_context = bind_synthetic_module_a_transaction_context(review_sandbox=review, isolated_temp_ancestor=root)
        discovery_state = exercise_module_a_v2_state_machine_for_discovery(
            synthetic_context=discovery_context, scenario=SIMULATION_SCENARIO
        )
        review_state = exercise_module_a_v2_state_machine_for_review(
            synthetic_context=review_context, scenario=SIMULATION_SCENARIO
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "simulation": {
            "discovery_manifest": dict(discovery_observation),
            "discovery_state_machine": dict(discovery_state),
            "review_state_machine": dict(review_state),
            "endpoint_security_diagnostic": _simulate_endpoint_security_diagnostic(),
            "evidence_class": "review_only_synthetic",
            "production_capability": False,
        },
        "platform_boundary": _diagnostic_platform_boundary(build_capability_report()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the complete report as JSON")
    args = parser.parse_args()
    report = build_local_simulation_report()
    if args.json:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        boundary = report["platform_boundary"]
        print(f"simulation=review_only_synthetic status={boundary['status']} p5_ready=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
