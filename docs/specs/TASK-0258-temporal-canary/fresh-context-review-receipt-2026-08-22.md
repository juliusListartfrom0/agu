# TASK-0258 Amendment-001 — independent fresh-context review receipt

Status: **sealed**

- Reviewer: Codex fresh-context independent review (read-only)
- Timestamp: `2026-08-22T19:09:57+08:00`
- Baseline: `c5157f23ab16a52399460b808a151583a2ee4a0`
- Target: `46c35e67260a4a9db4a1fe1ebef0b49427ceb2b2`
- Scope: 46 files total, 36 Python files
- Python diff SHA-256: `3a962603f008ec2214c9cb8c324c48f1c3471b6bcacd8e957199920c0c62f488`
- `scope_complete=true`
- `sealed_receipt=true`
- Critical: `0`
- Required: `0`
- Optional: `0`

## Review boundary

The reviewer inspected the complete baseline-to-target scope, including
receipt/marker replay, publication locks and races, canonical/no-follow
loaders, bootstrap and FD binding, worker process cleanup, registry topology,
and diagnostic-versus-production admission. The docs-only follow-up commit
`1a9c6db` did not change the reviewed code target.

## Closure evidence

1. No-directory-FD `FlockHandle.assert_held()` and `active_flock()` compare the
   current parent inode with the recorded `_parent_identity`; replacing the
   parent while hard-linking the old lock leaf returns `ValueError` for both
   held-handle reuse and new acquisition.
2. `_hold_exclusive_flock()` rechecks the stable parent FD and pathname after
   `flock()` and before yielding; replacing the parent at that point returns
   `ValueError` instead of entering the critical section.
3. Parent-directory xattr, `.identity` sidecar, lock leaf, paired replacement,
   registry topology, bootstrap/FD binding, worker process-group cleanup, and
   prior Critical/Required findings were reviewed with no residual finding.

## Verification

- `./.venv/bin/pytest -q tests/test_task0258*.py`: `320 passed`
- `./.venv/bin/pytest -q`: `2,188 passed, 5 skipped, 15 warnings`
- Diff-scoped Ruff and `git diff --check`: passed
- `./.venv/bin/python scripts/verify_harness.py --test-command
  './.venv/bin/pytest -q tests/test_task0258*.py'`: passed
- Local simulation: `review_only_synthetic`, `production_capability=false`
- Capability probe: ad-hoc signature, no Endpoint Security entitlement,
  approval, or kernel attestation; `blocked_external_authorization`

## Gate decision

P1 is closed by this sealed 0/0 receipt. This receipt does not authorize P2,
P3, P4, or P5. P2/P5 bypass is false; real Apple Endpoint Security
entitlement, external approval, and kernel read-isolation evidence remain
required.
