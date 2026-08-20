# AGU P0–P5 Execution Plan

This plan tracks the user-authorized sequence for TASK-0258 and the broader
AGU readiness gates. Each phase is isolated on a short-lived `codex/agu-pN`
branch, verified before commit, pushed to `origin`, merged into `main`, and
then verified and pushed again. A phase does not authorize the next phase's
model execution or readiness transition.

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
- [ ] Remediate the 7 Critical and 5 Required findings recorded in
      `fresh-context-review-2026-08-20.md`, then repeat the independent review.

### P2 — Platform isolation gate

- [ ] Complete macOS Endpoint Security/syscall read-isolation observation.
- [ ] Complete the FD review sandbox and verify worker/process cleanup.
- [ ] Prove model-visible reads are logged and receipt-bound.

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
