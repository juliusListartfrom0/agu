# AGU P0–P5 Execution Plan

This plan tracks the user-authorized sequence for TASK-0258 and the broader
AGU readiness gates. Each phase is isolated on a short-lived `codex/agu-pN`
branch, verified before commit, pushed to `origin`, merged into `main`, and
then verified and pushed again. A phase does not authorize the next phase's
model execution or readiness transition.

## Current operating decision (2026-08-22)

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
      and review-only loader findings; current focused suite is 277 passed and
      full suite is 2,145 passed, 5 skipped, 15 warnings. The review-only JSON
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
      output writes. The pipeline entry now refuses production execution
      unless an external admission issuer is present; temporary end-to-end
      tests must opt into an explicit synthetic-only context. Run-history marker append now also binds the new marker's
      authorization, run identity, root/nonce, and predecessor receipt to the
      durable ledger head under the registry lock. Registry replay now opens
      the directory once and reads members relative to an `O_NOFOLLOW` FD,
      rejecting non-JSON residue and symlinked members. The review-only candidate
      bundle loader now requires an absolute `candidate_v2` directory and
      replays its exact ten member receipts, rejecting candidate drift. The
      review-only admission loader now replays claim/admission/completion
      receipt edges and physical CAS; the terminal loader replays exactly one
      result/failure topology and binds it to history, admission, bundle, and
      candidate-member receipts. The canonical JSON loader restores fixed
      provider iteration order only after canonical-byte validation. Candidate
      bundle replay also validates `candidate_gate.json` and terminal replay
      binds its trust-spine provider slots. The
      local read-isolation policy↔attestation binder now checks the exact
      policy/provider/worker tuple, deny-before-allow matching, unique longest
      policy rows, fixed inherited-FD rows, and returned file identity; it
      validates provider output only and cannot mint kernel evidence, and the
      verification-attempt validator invokes it before accepting a worker
      attempt. The review-only capabilities module now also reopens both
      canonical policy and attestation files through the bounded no-follow
      loader and returns an opaque cross-bound review object; this remains
      diagnostic-only and cannot issue production authorization. The
      verification-attempt loader now reopens a complete canonical attempt
      artifact and invokes its nested read-isolation gate before returning an
      opaque review-only object. A review-only trust-spine binder now binds
      that attempt to the loaded admission/history/authorization and output
      root physical CAS. A review-only no-write preflight now reopens the
      output-root and registry identities, rejects reserved output/stage
      residue and bundle-path aliases, and returns only a diagnostic write
      plan; it cannot issue production authorization. The
      review-only implementation-approval loader now reopens the sealed
      amendment approval through the bounded canonical/no-follow JSON boundary,
      replays the parent approval and its three exact spec files, the amendment
      file, the Critical/Required 0/0 fresh review, and the implementation-scope
      baseline/root CAS, then returns only an opaque diagnostic artifact; it
      does not issue rerun authorization. The review-only amended-
      implementation-review loader now binds the implementation-review and
      fresh-review receipts, current closed review inputs/outputs, scope delta,
      governance observation, and Critical/Required 0/0 without issuing
      authority. The review-only rerun-authorization loader now binds that
      review object, exact approval receipts, static-input contract, fixed
      operation list, output-root/candidate-bundle absence, and registry
      identity; it also remains diagnostic-only. The
      review-only static-input loader now replays the complete parent plan and
      TASK-0257 receipt graph, including all ordered JPEG/source-video/checkpoint
      receipts, computes the caller-frozen projection, and returns an opaque
      `production_capability=false` wrapper. Its review-only replay entry point
      reopens the plan and replays the full parent graph again on every use; it
      does not authorize a run. The authorization replay now has a separate
      review-only entry point that requires this per-use static replay before
      binding the exact three-hash contract. Matching run-admission and terminal
      review loaders now apply the same replay before accepting their
      static-input-bound receipt bytes.
      The verified bootstrap launcher now performs a standard-library-only
      pre-import validation, builds the fixed three-row runtime path, and only
      then imports the sealed core; a local integration test exercises the real
      `-P -S /dev/fd/203` and FD `202/203` path. This remains local evidence,
      not an OS-enforced sandbox or kernel attestation.
      repeatable
      repository-local simulation command now passes the synthetic discovery
      and review state machine, parses a C-shaped Endpoint Security event stream
      with clean finalization, and explicitly returns `p5_ready=false`.
      The simulation canonicalizes macOS temporary paths before binding the
      no-follow synthetic context, so the system `/var` alias cannot create a
      false local failure while user-controlled symlinked ancestors remain
      rejected.
      The read-only `inspect_endpoint_security_transcript.py` handoff command
      validates a real transcript's finalization and reports only
      `valid_diagnostic_transcript`; it cannot create P2/P5 evidence.
      The pipeline now reopens the published candidate bundle under its distinct
      bundle lock through a bounded regular-file/no-follow descriptor before
      binding its file receipt into the result.
      The diagnostic `fs_usage` transcript parser now applies the v2 read-event
      row and UTF-8 byte caps before retaining parsed events, and its harness
      bounds diagnostic pipe capture; this is a local
      resource bound only and cannot mint kernel read-isolation evidence.
      The worker subprocess runner now rejects non-list argv, embedded NULs,
      and non-positive/non-integer timeouts before spawning; environment
      isolation and process-group cleanup remain local evidence only. The
      fs_usage audit entry point reuses the same launch validator, sanitized
      environment, and new-process-group boundary, and kills/reaps the worker
      group on timeout before parsing diagnostics.
      The fs_usage audit harness now drains diagnostic pipes while retaining
      at most the shared byte cap, propagates malformed-input and background
      iterator failures to the main caller, rejects unfinished drain threads,
      and fails closed on overflow; this remains a diagnostic resource bound,
      not kernel evidence.
      The fs_usage audit entry point now also starts the diagnostic observer in
      its own process group and terminates/reaps both observer and worker on
      observer-start, drain-cleanup, and worker-timeout failures; this remains
      local cleanup evidence only.
      Its CLI now reads the policy through the bounded no-follow file reader
      and writes any future provider output with no-clobber atomic JSON; a
      symlinked policy or output cannot cross that diagnostic boundary.
      The Endpoint Security capability probe now rejects entitlement claims
      from adhoc/unknown signatures and canonicalizes the selected artifact
      before strict codesign inspection. Its C client uses strict PID parsing,
      no-follow/exclusive output creation, bounded fork/pidversion lineage,
      path-truncation rejection, JSON escaping, strict decimal PID/timeout
      parsing, target EXIT handling, bounded default client lifetime, exact
      notify-result serialization, and fail-closed `seq_num`/`global_seq_num`
      gap detection; successful transcript finalization also flushes and
      `fsync`s the descriptor before close; the Python side now has a strict,
      bounded JSONL diagnostic parser that requires a clean finalization row,
      verifies event-row/byte counts, rejects duplicate fields, malformed
      result projections, invalid paths, retained-row sequence regressions, and
      rows over the C-aligned 512-byte single-row cap; it remains
      diagnostic-only and cannot mint an attestation. SDK
      compilation remains local evidence only.
      Diff-scoped Ruff over the TASK-0258 Python change set passes; a full
      repository Ruff audit still reports 52 pre-existing findings outside
      this scope and is not silently broadened into this phase.
- [ ] Close the OS-enforced review-sandbox/Endpoint Security/admission gaps and
      repeat the independent review to obtain `Critical=0 / Required=0`.

### P2 — Platform isolation gate

- [x] Run the no-sudo Endpoint Security capability probe: SDK/build available,
      but the temporary client is only ad-hoc signed and lacks the required
      entitlement; the probe correctly remains fail-closed. The client also
      passes strict `clang -Wall -Wextra -Werror` syntax compilation and has a
      bounded timeout/target-exit shutdown path and fail-closed notification
      sequence/result capture and durable transcript finalization. Once a signed
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

- [x] Exercise local review-only replay of the exact authorization graph; it
      returns `production_capability=false` and does not consume a run.
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
