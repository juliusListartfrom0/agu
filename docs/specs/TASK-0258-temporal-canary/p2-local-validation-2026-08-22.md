# TASK-0258 P2 local validation handoff — 2026-08-22

Status: **externally blocked**

- Branch: `codex/agu-p2-platform-gate`
- Base: `main` merge commit `aa3858f`
- P1 prerequisite: sealed independent receipt, Critical=0, Required=0
- Local validation scope: SDK/build probe, synthetic review-only simulation,
  bounded transcript/projection checks, and repository regression tests

## Local evidence completed

The repository-local P2 work is available without Apple Developer Program
membership:

- Endpoint Security source and SDK headers compile with the strict local probe.
- The capability probe rejects the temporary ad-hoc artifact instead of
  treating it as an entitled client.
- The synthetic state machine and C-shaped diagnostic finalization pass while
  reporting `production_capability=false` and `p5_ready=false`.
- The TASK-0258 lock, replay, bootstrap, worker-cleanup, transcript, and
  capability regressions pass under the canonical `.venv`.

## Current probe results

```text
compile_ok=true
signature_kind=adhoc
endpoint_security_entitlement_present=false
external_user_approval_observed=false
kernel_read_isolation_attestation_observed=false
status=blocked_external_authorization
```

The read-only host audit also reports:

```text
security find-identity -v -p codesigning: 0 valid identities found
systemextensionsctl list: 0 extension(s)
xcodebuild -version: unavailable; active developer directory is CommandLineTools
```

These results confirm that this Mac currently has neither a usable Apple
signing identity nor an installed System Extension. They are environment
observations, not substitutes for Apple authorization.

The local simulation reports:

```text
evidence_class=review_only_synthetic
production_capability=false
p5_ready=false
```

## Missing authoritative evidence

P2 cannot be marked complete until all of the following are collected from a
real macOS platform path:

1. Apple-authorized signing and the managed
   `com.apple.developer.endpoint-security.client` entitlement.
2. Installed and externally approved System Extension state.
3. Authenticated kernel/Endpoint Security read-isolation evidence bound to the
   worker lineage and artifact receipts.
4. OS-enforced review-sandbox evidence for the trusted runner.

A free Apple Account/Personal Team, local ad-hoc signing, SIP-disabled debug,
synthetic event rows, or parsed `fs_usage` output cannot satisfy these rows.
The only no-new-fee routes identified are an eligible organization fee waiver
or sponsorship by an already enrolled organization; both still require Apple
authorization. See
[`TASK-0258-EXTERNAL-UNBLOCK-CHECKLIST.md`](../../harness/TASK-0258-EXTERNAL-UNBLOCK-CHECKLIST.md).

## Gate decision

P2 local diagnostics are complete, but the authoritative P2 platform gate is
`blocked_external_authorization`. P3 exact rerun authorization, P4 execution,
and P5 readiness remain prohibited. This handoff is not a P2 completion receipt.
Its documentation was merged into `main@ac8a2aa` only as a truthful local
handoff; that merge must not be interpreted as the platform gate passing.
