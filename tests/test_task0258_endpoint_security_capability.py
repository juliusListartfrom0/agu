"""Tests for the fail-closed Endpoint Security capability report."""

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
