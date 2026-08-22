# TASK-0258 P2 free-route audit

Date: 2026-08-22  
Status: `blocked_external_authorization`  
Scope: determine whether P2 can be completed without personally paying the
Apple Developer Program annual fee.

## Decision

There is no free individual route that satisfies the repository's real P2
acceptance gate. A free Apple Account / Xcode Personal Team remains useful for
ordinary Xcode development and limited device testing, but it does not provide
a path to hold Apple's managed Endpoint Security client entitlement. The
Endpoint Security entitlement is separately requested from Apple; without it,
`es_new_client` fails with `ES_NEW_CLIENT_RESULT_ERR_NOT_ENTITLED`.

The Apple Developer Program is currently listed as USD 99 per membership year.
Paying that fee would still not be sufficient by itself: the team must request
and receive the managed capability, then produce the signed, installed, and
approved System Extension plus the OS/kernel evidence required by P2.

## Options checked

| Route | Can it be zero new cost? | What it can prove | P2 outcome |
| --- | --- | --- | --- |
| Free Apple Account / Personal Team | Yes | Ordinary Xcode work and limited device testing; Personal Team provisioning is short-lived and limited | **No**: no managed Endpoint Security capability or P2 platform attestation |
| Ad-hoc/local self-signing | Yes | Local build, signature inspection, and repository diagnostics | **No**: no Apple-authorized entitlement, user approval, or kernel read-isolation evidence |
| SIP-disabled System Extension debugging | Yes, on a development Mac | Apple documents it as a temporary debugging route while an entitlement request is under review | **No**: diagnostic installation is not the production authorization/evidence gate |
| Apple fee waiver | Potentially | A qualifying organization's Apple Developer Program membership and subsequent capability request | **Potentially**: only for a legal nonprofit, accredited educational institution, or government entity that Apple approves; not available to an individual, sole proprietor, or single-person business |
| Existing enrolled organization sponsors AGU | No new personal fee, if the organization agrees | Real team signing, Account Holder capability request, and the organization's authorized installation/approval path | **Potentially**: the sponsoring team must actually own the signed host/extension and obtain the capability; a copied signature or borrowed artifact is not valid evidence |

The two potentially viable no-new-fee routes are therefore organization-only:
an approved fee waiver or sponsorship by an already enrolled organization.
Neither is a personal free-team workaround, and both still depend on Apple's
authorization and the P2 evidence procedure.

## Current machine evidence

The repository-local checks remain unchanged:

```text
security find-identity -v -p codesigning  -> 0 valid identities found
systemextensionsctl list                   -> 0 extension(s)
xcodebuild -version                        -> no Xcode selected/installed
signature_kind                             -> adhoc
endpoint_security_entitlement_present      -> false
external_user_approval_observed             -> false
kernel_read_isolation_attestation_observed  -> false
status                                      -> blocked_external_authorization
```

The local synthetic simulation continues to report
`evidence_class=review_only_synthetic`, `production_capability=false`, and
`p5_ready=false`. These are correct fail-closed results, not a P2 failure in
the code and not a P2 completion receipt.

## Official references checked

- [Choosing a Membership](https://developer.apple.com/support/compare-memberships/)
  — free Apple Account capabilities, Personal Team limits, and USD 99 annual
  program fee.
- [Endpoint Security entitlement](https://developer.apple.com/documentation/BundleResources/Entitlements/com.apple.developer.endpoint-security.client)
  — entitlement must be requested from Apple; absence makes `es_new_client`
  fail as not entitled.
- [System Extensions and DriverKit](https://developer.apple.com/system-extensions/)
  — Endpoint Security use requires an entitlement request; SIP-disabled local
  testing is described as a temporary development route.
- [Managed capability requests](https://developer.apple.com/help/account/capabilities/capability-requests/)
  — managed capabilities are assigned by Apple and organization requests are
  submitted by the Account Holder.
- [Developer Program fee waiver](https://developer.apple.com/help/account/membership/fee-waivers)
  — eligibility is limited to qualifying legal organizations and excludes
  individuals and sole proprietors.

## Repository decision

Continue AGU local code, pytest, FastAPI smoke, review-only simulation, and
fail-closed guard work. Keep P2 marked as externally blocked until a real
Apple-authorized team supplies the signed System Extension, Endpoint Security
entitlement, user approval, and OS/kernel read-isolation evidence. Do not use
the free routes above to mint or infer those fields.
