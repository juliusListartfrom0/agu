# TASK-0258 external Endpoint Security unblock checklist

This checklist is the handoff from the repository-local state to the real
macOS platform gate. A local ad-hoc compile, synthetic review context, or raw
`fs_usage` transcript is not an Endpoint Security or kernel read-isolation
attestation.

## Scope boundary

Apple Developer Program access is **not** a prerequisite for AGU repository
development, local pytest/Harness verification, FastAPI smoke tests, or the
review-only synthetic simulation. This checklist is only the handoff for the
real macOS P2 platform gate. Without an authorized team, the platform result
must remain `blocked_external_authorization`; local work may continue, but P2,
P3, P4, and P5 must not be marked complete.

## Preconditions

1. Enroll an Apple Account in the Apple Developer Program as an individual, or
   join an organization team. The team must have an Account Holder who can
   request managed capabilities.
2. Register separate App IDs for the host application and the System Extension.
3. In Certificates, Identifiers & Profiles → App ID → Capability Requests,
   request Endpoint Security access. Apple describes managed capabilities as
   entitlements assigned to the account by Apple; the Account Holder submits
   the request and enables it only after assignment.
4. Keep the signing private key on the Mac that performs the build. Never put a
   private key, provisioning profile secret, Apple password, or token in the AGU
   repository or in a Codex message.

Official references:

- [Apple Developer Program enrollment](https://developer.apple.com/programs/enroll/)
- [Apple managed capability requests](https://developer.apple.com/help/account/capabilities/capability-requests/)
- [Endpoint Security entitlement](https://developer.apple.com/documentation/BundleResources/Entitlements/com.apple.developer.endpoint-security.client)
- [Installing System Extensions and Drivers](https://developer.apple.com/documentation/systemextensions/installing-system-extensions-and-drivers)
- [Monitoring system events with Endpoint Security](https://developer.apple.com/documentation/endpointsecurity/monitoring-system-events-with-endpoint-security)

## Build and sign the real artifact

AGU's current C file is the Endpoint Security client/audit core; it is not by
itself an installable `.systemextension`. The acceptance artifact must be a
host application containing a real System Extension under
`Contents/Library/SystemExtensions`, with the appropriate host and extension
entitlements and an Apple-authorized provisioning profile.

Use Xcode's normal signing flow after the capability has been assigned. The
extension target must carry:

```text
com.apple.developer.endpoint-security.client = true
```

The host application and extension must be signed with the same authorized
team. A Developer ID or development signing identity is acceptable only when
the corresponding provisioning/capability evidence and installation path are
also recorded; an ad-hoc signature is never sufficient for this gate.

## Verify before installation

Replace `<signed-artifact>` with the actual executable or `.systemextension`
inside the built application:

```bash
codesign --verify --deep --strict --verbose=4 <signed-artifact>
codesign --display --verbose=4 <signed-artifact>
codesign --display --entitlements :- <signed-artifact>

.venv/bin/python scripts/task0258_endpoint_security_capability.py \
  --signed-artifact <signed-artifact> --json --require-ready
```

The last command must report at least:

```json
{
  "signature_kind": "signed",
  "endpoint_security_entitlement_present": true,
  "status": "requires_external_user_approval"
}
```

This is still only the capability/signing stage. It does not prove user
approval or kernel read isolation.

After the C client exits and produces its transcript, validate the bounded
diagnostic projection without upgrading it to an attestation:

```bash
.venv/bin/python scripts/inspect_endpoint_security_transcript.py \
  --transcript /absolute/path/to/audit.jsonl --json
```

The expected local result is `status=valid_diagnostic_transcript` with
`finalization_verified=true`, `evidence_class=diagnostic_only`,
`production_capability=false`, and `p5_ready=false`. A missing/unclean
finalization record, truncated row, symlinked file/ancestor, or count mismatch must
return `invalid_diagnostic_transcript` and a nonzero exit code.

## Install and collect approval evidence

1. Copy the signed host application to `/Applications`.
2. Launch the host application and request activation through the System
   Extension API; do not treat a raw executable launch as installation.
3. Approve the extension in System Settings → General → Login Items &
   Extensions → Endpoint Security Extensions. See the [Mac User
   Guide](https://support.apple.com/guide/mac-help/change-login-items-extensions-settings-mtusr003/mac).
4. Grant the host application the required Full Disk Access permission in
   Privacy & Security for the controlled test.
5. Record the installed extension:

```bash
systemextensionsctl list
```

The receipt must identify the extension bundle identifier, team identifier,
version, and approval/activation state. A zero-extension result is a failed
platform gate, not a pass.

## Complete the TASK-0258 evidence gate

The platform handoff is complete only when all of the following are retained
as auditable, hash-bound evidence:

| Evidence | Required result |
| --- | --- |
| Signed artifact | The real executable or `.systemextension` verifies and is not ad-hoc signed. |
| Entitlement | The signed artifact and authorized profile contain Endpoint Security entitlement. |
| User approval | `systemextensionsctl list` shows the installed/approved extension. |
| Read isolation | An externally authenticated Endpoint Security/syscall provider records the worker's bounded read events, process lineage, denied/unknown counts, overflow state, and exact artifact bindings. |
| Review sandbox | The OS-enforced FD layout, immutable runtime image, worker cleanup, and no-model review scenarios pass from the trusted runner. |
| Independent review | A fresh implementation review seals `Critical=0 / Required=0`. |

The repository's `task0258_endpoint_security_capability.py` probe can verify
the first two rows once the signed artifact exists. It deliberately leaves the
approval and kernel fields false; no synthetic API, caller-supplied event row,
or `fs_usage` parser output may be promoted into those fields.

Only after the P1/P2 evidence is sealed may the separately issued exact-SHA v2
rerun authorization be used to begin P3/P4. P5 remains closed until its own
rights, causal-evidence, performance, and independent-VLM gates pass.

## Current no-team result

On the current Mac, the expected local diagnostic remains:

```text
signature_kind=adhoc
endpoint_security_entitlement_present=false
external_user_approval_observed=false
kernel_read_isolation_attestation_observed=false
status=blocked_external_authorization
```

That result is the correct fail-closed behavior while no Apple Developer team
and no authorized signed System Extension are available.
