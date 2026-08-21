"""Tests for the TASK-0258 repository-local simulation command."""

from __future__ import annotations

import sys

from scripts import task0258_local_simulation as simulation


def test_local_simulation_is_observation_only(monkeypatch):
    monkeypatch.setattr(
        simulation,
        "build_capability_report",
        lambda: {
            "status": "blocked_external_authorization",
            "signature_kind": "adhoc",
            "endpoint_security_entitlement_present": False,
            "external_user_approval_observed": False,
            "kernel_read_isolation_attestation_observed": False,
        },
    )

    report = simulation.build_local_simulation_report()

    assert report["schema_version"] == "agu.task0258-local-simulation-report.v1"
    assert report["platform_boundary"]["status"] == "blocked_external_authorization"
    assert report["platform_boundary"]["p5_ready"] is False
    assert report["simulation"]["discovery_manifest"]["production_capability"] is False
    assert report["simulation"]["discovery_state_machine"]["production_capability"] is False
    assert report["simulation"]["review_state_machine"]["production_capability"] is False
    endpoint_security = report["simulation"]["endpoint_security_diagnostic"]
    assert endpoint_security["event_count"] == 2
    assert endpoint_security["finalization_verified"] is True
    assert endpoint_security["evidence_class"] == "diagnostic_only"
    assert endpoint_security["production_capability"] is False


def test_local_simulation_never_marks_p5_ready_even_if_probe_is_ready(monkeypatch):
    monkeypatch.setattr(
        simulation,
        "build_capability_report",
        lambda: {
            "status": "requires_external_user_approval",
            "signature_kind": "signed",
            "endpoint_security_entitlement_present": True,
            "external_user_approval_observed": False,
            "kernel_read_isolation_attestation_observed": False,
        },
    )

    report = simulation.build_local_simulation_report()

    assert report["platform_boundary"]["status"] == "requires_external_user_approval"
    assert report["platform_boundary"]["p5_ready"] is False


def test_local_simulation_cli_emits_json(monkeypatch, capsys):
    monkeypatch.setattr(simulation, "build_local_simulation_report", lambda: {"p5_ready": False})
    monkeypatch.setattr(sys, "argv", ["task0258_local_simulation", "--json"])

    assert simulation.main() == 0
    assert capsys.readouterr().out.strip() == '{"p5_ready":false}'
