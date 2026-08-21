# TASK-0258 Amendment-001 — v2 implementation plan

Status: **IMPLEMENTATION AUTHORIZED — PARTIALLY IMPLEMENTED** (amendment fresh
review Critical/Required 0/0, exact-SHA approval sealed 2026-08-17; see
`amendment_implementation_approval.json`).

## Progress to date (2026-08-22)

The schema/validation/grammar/persistence layer is partially implemented and
being hardened after the independent fresh-context review failed with
`Critical=7 / Required=5 / Optional=3` (see
`fresh-context-review-2026-08-20.md`). The remediation slice now has 211
focused tests passing and keeps unauthorized v2 publication fail-closed; it is
not a passing implementation-review seal.
Implemented modules:

- `app/analysis/task0258_module_a_v2.py` — receipt primitives + exact field sets.
- `app/analysis/task0258_run_history.py` — run-history events/graph/marker/claim/admission/completion schemas plus descriptor-relative no-follow durable replay.
- `app/analysis/task0258_v2_gate.py` — candidate/result check sequences + failure matrix.
- `app/analysis/task0258_v2_artifacts.py` — candidate/failure/bundle/result/failure artifact schemas + membership.
- `app/analysis/task0258_v2_fs.py` — atomic publication primitives.
- `app/analysis/task0258_v2_registry.py` — registry durable write path + `create_run_history_registry`.
- `app/analysis/task0258_v2_verification.py` — verification embedding/attempt/resource schemas, float32 projection, embedding builder, and attempt-level read-isolation binding.
- `app/analysis/task0258_v2_verification_extract.py` — empty-state extraction wiring (parent reuse).
- `app/analysis/task0258_v2_read_isolation.py` — read-isolation policy/attestation schemas and local policy binding.
- `app/analysis/task0258_v2_bootstrap.py` + `scripts/task0258_module_a_verified_bootstrap.py` — authorized pre-import bootstrap core + FD entry.
- `app/analysis/task0258_v2_pipeline.py` — candidate/bundle/post-publication builders + sealers.
- `app/analysis/task0258_v2_pipeline_cli.py` — full pipeline orchestration (registry→candidate→bundle→result).
- `app/analysis/task0258_v2_worker_runner.py` — worker subprocess isolation runner.
- `app/analysis/task0258_v2_capabilities.py` — review-only synthetic contexts,
  no-follow canonical-byte loaders, durable run-history replay, admission
  claim/admission/completion replay, candidate-gate schema/trust-spine replay,
  terminal result/failure replay, and read-isolation policy/attestation
  cross-bound loading, complete verification-attempt replay, and implementation
  approval full parent/amendment/review/baseline receipt replay, plus amended
  implementation-review/fresh-review and review-only rerun-authorization
  receipt replay; these
  contexts also bind the attempt to the loaded admission/history trust spine
  and provide a no-write preflight replay, but cannot authorize production
  publication.
- `scripts/smoke_v2_verification_extraction.py` — real empty-state Swin extraction smoke.

Current working-tree verification: 211 TASK-0258 tests pass and the AGU Harness
structural gate passes. The full repository suite reports 2,079 passed, 5
skipped, and 15 warnings. Scoped Ruff/format checks are clean after the P0 and
P1 remediation work. The existing
implementation plan records a real empty-state extraction smoke producing 45
rows (4x768) with computational projection
`3d8dfba9…3c72`, but no retained v2 result/terminal receipt has been found in
the current repository state. This smoke claim is therefore not a v2 run
result and does not authorize execution.

Remaining gates: close the fresh-context findings and repeat the independent
review, then complete the kernel-audit read-isolation production path (macOS
Endpoint Security / syscall audit), OS-enforced FD review sandbox driver,
externally authenticated verified loaders, and exact authorization-bound
production admission. The bootstrap now rejects non-fixed target commands and
FD assignments and reopens the standalone runtime contract through the
four-field receipt with descriptor-relative no-follow/canonical checks; the
runtime contract and kernel/provider evidence are still external gates. The
Python audit parser also refuses to mint an attestation from raw or caller-
supplied "verified" rows. The current review-only
  contexts/loaders are deliberately not an admission substitute. The
  worker subprocess layer now sanitizes its environment, starts a process group,
  and kills the group on timeout, but the full production proof boundary is not
  sealed. The pipeline CLI now refuses production execution unless a future
  external admission issuer is bound; its end-to-end tests require an explicit
  synthetic-only flag. Module B and the v2 rerun remain unauthorized.

This plan decomposes the amendment-001 v2 proof boundary into dependency-ordered
phases matching the spec's acyclic hash order (amendment §"Acyclic producer and
verification artifacts"). Each phase is TDD: RED tests first (per amendment
§"TDD, review and execution gates"), then production code, then GREEN.

## Delta from v1 (what already exists)

The 4,246-line `app/analysis/vru_causal_temporal_retrospective.py` already
implements the parent Module-A computation: tiled-Swin feature plan, batch-one
MPS extraction, four-game nested LOGO screening, baseline/candidate final
evaluators, mechanical gate, disk budget (`DiskWriteBudgetError`), resource
guard, atomic publication, and signal-only resume. The amendment does **not**
change the parent computation contract; it adds the closed-evidence
post-publication boundary and a second empty-state extraction on top.

## New v2 schema versions to introduce

| Kind | Schema |
| --- | --- |
| producer attempt | `agu.vru-causal-tiled-swin-attempt.v2` |
| producer resume | `agu.vru-causal-tiled-swin-embeddings-resume.v2` |
| producer embedding | `agu.vru-causal-tiled-swin-embeddings.v2` |
| verification embedding | `agu.vru-causal-tiled-swin-verification-embeddings.v2` |
| verification attempt | `agu.vru-causal-tiled-swin-verification-attempt.v2` |
| verification resource row | `agu.vru-causal-verification-resource-sample.v1` |
| candidate gate | `agu.vru-causal-temporal-candidate-gate.v2` |
| mechanical failure | `agu.vru-causal-temporal-mechanical-failure.v2` |
| candidate receipt bundle | `agu.vru-causal-temporal-candidate-receipt-bundle.v1` |
| postpublication verification | `agu.vru-causal-temporal-postpublication-verification.v2` |
| postpublication failure | `agu.vru-causal-temporal-postpublication-failure.v2` |
| run-history marker | `agu.task0258-module-a-v2-run-history-marker.v1` |
| run-history claim/completion | `agu.task0258-module-a-v2-run-history-*.v1` (exact names in §2726) |
| rerun authorization | `agu.task0258-module-a-v2-rerun-authorization.v1` (issued separately) |

## Phases

1. **Run-history registry** (foundation / trust spine) — new module
   `app/analysis/task0258_run_history.py`: claim/completion files
   `<AUTH_SHA>.claim.json` / `<AUTH_SHA>.completed.json`, 14-event history markers
   with the exact transition graph, `root_subject_cas`, `subject_receipts`, and
   cumulative counters. Exposes `create_run_history_registry`,
   `verify_run_history_marker`, `load_verified_run_history_ledger`.
2. **v2 producer schemas** — version `attempt/resume/embedding` to `.v2`, adding
   `authorization_receipts`, `run_identity_receipt`, `history_head_receipt`,
   `run_admission_receipt` (+ worker isolation objects on the attempt).
3. **Verification worker** — second empty-state extraction (role
   `independent_empty_state_verification`), `verification_embeddings.v2` +
   `verification_attempt.v2`, read-isolation policy/attestation, shared global
   resource caps, `computational_projection_sha256`.
4. **Candidate generation** — `candidate_v2/` ten members +
   `candidate_gate.v2` (24 ordered checks, `publication_state=not_yet_observed`).
5. **Failure generation** — `terminal_failure_v2/mechanical_failure.json` +
   the phase/member/check/reason matrix (§3361-3515).
6. **Candidate receipt bundle** — external sealer
   `candidate_receipt_bundle.v1` (§3517-3579).
7. **Post-publication verification** — `verified_result_v2/verification_registry.json`
   (§3639-3745) and `postverification_failure_v2/failure.json` (§3753-3806).
8. **Bootstrap + review sandbox + public API** — `scripts/task0258_module_a_verified_bootstrap.py`,
   the seven review-sandbox-only functions (§3808-3967), and the FD-based review
   driver.

## Verification

- Focused pytest per phase (RED→GREEN), then the full suite.
- `ruff check` + `ruff format --check` + `git diff --check`.
- No runtime/formal/promotion/readiness/blind-inference change; every v2
  eligibility flag is hard-false.

## Output boundary

New output root is `vru_causal_temporal_retrospective_v2` (never reuses v1).
Module B remains unauthorized. The v2 rerun authorization is a **separate**
user-issued artifact after implementation + a fresh implementation review.
