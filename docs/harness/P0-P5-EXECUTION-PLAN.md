# AGU P0–P5 Execution Plan

This plan tracks the user-authorized sequence for TASK-0258 and the broader
AGU readiness gates. Each phase is isolated on a short-lived `codex/agu-pN`
branch, verified before commit, pushed to `origin`, merged into `main`, and
then verified and pushed again. A phase does not authorize the next phase's
model execution or readiness transition.

## Current operating decision (2026-08-21)

- Continue repository-local implementation, review-only synthetic scenarios,
  capability probing, pytest, and Harness verification while Apple Developer
  Program access is unavailable.
- Treat the no-sudo ad-hoc Endpoint Security probe and all synthetic contexts as
  diagnostics only. They must not mint a kernel read-isolation attestation,
  user-approval evidence, a v2 rerun authorization, or a product result.
- Keep P1/P2 open where they require an OS-enforced review sandbox, a real
  Endpoint Security entitlement, external approval, and authenticated kernel
  observation. Keep P3/P4/P5 and readiness incomplete until those gates are
  satisfied.
- The exact external handoff procedure is recorded in
  [`TASK-0258-EXTERNAL-UNBLOCK-CHECKLIST.md`](TASK-0258-EXTERNAL-UNBLOCK-CHECKLIST.md).

## Phase gates

### P0 — Working-tree and state convergence

- [x] Preserve the existing TASK-0258 v2 implementation as the baseline.
- [x] Fix scoped Ruff/import issues and maintenance-script absolute paths.
- [x] Synchronize task board, TASK-0258 artifacts, and wiki state.
- [x] Verify focused tests, Harness, Ruff, format, and diff checks.

### P1 — Independent implementation review

- [x] Review the exact current implementation from a different fresh context.
- [ ] Verify implementation-scope baseline/delta, review inputs, resource policy,
      and `Critical=0 / Required=0`.
- [ ] Seal the amended implementation-review receipt.
- [x] Remediate the repository-local receipt, replay, path, bootstrap, worker,
      and review-only loader findings; current focused suite is 194 passed and
      full suite is 2,062 passed, 5 skipped, 15 warnings. The review-only JSON
      loaders now reopen bounded temporary files through descriptor-relative
      no-follow reads and reject symlinked temporary parents. Run-history marker
      append now validates the authorization SHA before constructing any lock
      path, and all v2 lock acquisition creates/opens a regular lock file
      without following a lock-path symlink. Candidate receipt-bundle sealing
      now reopens the candidate under the output-root lock, holds a distinct
      bundle-parent lock, uses the fixed authorization-bound stage basename,
      and reopens the published bundle. Result and postverification-failure
      directory sealing now uses authorization-bound fixed stages and reopens
      exact member coverage after rename; terminal publication now checks the
      candidate-present, mutually-exclusive topology under the root lock before
      creating any stage. Candidate publication now validates the candidate gate
      itself, binds its exact rerun authorization SHA to a fixed stage name, and
      checks candidate/terminal occupancy under the root lock before creating
      any stage. The pipeline preflight now cross-binds candidate-gate
      authorization/admission/history receipts to the actual admission and
      completion bytes, requires the bundle and result to share the same tuple
      and candidate-member receipts, and rejects mismatches before registry or
      output writes. Run-history marker append now also binds the new marker's
      authorization, run identity, root/nonce, and predecessor receipt to the
      durable ledger head under the registry lock. The review-only candidate
      bundle loader now requires an absolute `candidate_v2` directory and
      replays its exact ten member receipts, rejecting candidate drift. The
      review-only admission loader now replays claim/admission/completion
      receipt edges and physical CAS; the terminal loader replays exactly one
      result/failure topology and binds it to history, admission, bundle, and
      candidate-member receipts. The canonical JSON loader restores fixed
      provider iteration order only after canonical-byte validation. Candidate
      bundle replay also validates `candidate_gate.json` and terminal replay
      binds its trust-spine provider slots. The
      repeatable
      repository-local simulation command now passes the synthetic discovery
      and review state machine while explicitly returning `p5_ready=false`.
- [ ] Close the OS-enforced review-sandbox/Endpoint Security/admission gaps and
      repeat the independent review to obtain `Critical=0 / Required=0`.

### P2 — Platform isolation gate

- [x] Run the no-sudo Endpoint Security capability probe: SDK/build available,
      but the temporary client is only ad-hoc signed and lacks the required
      entitlement; the probe correctly remains fail-closed. Once a signed
      executable or `.systemextension` exists, pass it explicitly with
      `scripts/task0258_endpoint_security_capability.py --signed-artifact
      <path> --json`; the default probe intentionally continues to inspect its
      temporary compile.
- [ ] Complete macOS Endpoint Security/syscall read-isolation observation.
- [ ] Complete the OS-enforced FD review sandbox and verify worker/process
      cleanup. The pure bootstrap now rejects arbitrary target argv, fixes the
      request/source map to FD `202/203`, and reopens the standalone runtime
      contract through its receipt with no-follow/canonical checks, and checks
      inherited descriptor identity/type; this is not yet an OS/kernel
      attestation.
- [ ] Prove model-visible reads are logged and receipt-bound.
      The Python `fs_usage` parser and caller-supplied verified-row builder
      now fail closed; only an externally authenticated kernel provider may
      mint the attestation.

### P3 — Explicit v2 run authorization

- [ ] Issue a separate exact-SHA v2 rerun authorization.
- [ ] Keep Module B, runtime, promotion, readiness, and blind inference false.

### P4 — One guarded Module-A v2 run

- [ ] Execute exactly one authorized producer/verification pipeline.
- [ ] Publish either `verified_result_v2` or a canonical `terminal_failure_v2`.
- [ ] Verify post-publication receipts and preserve the v1 boundary.

### P5 — Product evidence readiness

- [ ] Close a rights-cleared, production-distinct continuous source.
- [ ] Complete independent label-hidden causal evidence.
- [ ] Reach per-source held-production precision and recall of at least 0.85.
- [ ] Close the independent-VLM evidence path before any runtime promotion.

## Branch and verification contract

Every phase must record its branch, commit, remote push, merge commit, and
post-merge `main` verification in `docs/harness/TASK-BOARD.md`. The canonical
Python runtime is `.venv/bin/python`. No generated media, datasets, model
weights, secrets, or unverified result artifacts may be committed.
