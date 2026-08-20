# TASK-0257 Gate review — Harwood additive training bridge

## Decision

**IMPLEMENTATION VERIFIED; DEVELOPMENT DIAGNOSTIC COMPLETE; NOT PROMOTED.**

The additive v2 chain, two embedding artifacts, and nested four-game probe are
implemented and strictly verified. This closes TASK-0257 only as a
development diagnostic. It does not establish a production-held evaluation,
formal acceptance, runtime promotion, readiness change, blind-inference
resume, or 85% result.

## Spec gate

- [x] v1 is an immutable parent with exact internal/file receipts.
- [x] All six parent child files have exact SHA-256 anchors.
- [x] Harwood source, selection, plan, raw frames, and review have exact SHA
  anchors and one caller-supplied expected-receipt boundary.
- [x] Counts are frozen at 48 selected, 45 exported, `17+/28-`, and three
  excluded `uncertain` rows.
- [x] Four held games and only two production families are modeled explicitly.
- [x] New v2 schema/API/CLI names are additive; v1 is neither changed nor
  overloaded.
- [x] Nested model/threshold selection is confined to outer-training games.
- [x] Resource guard, atomic IO, path aliasing, and cleanup behavior are
  specified.
- [x] Runtime/formal/promotion/readiness/blind-inference boundaries are explicit.
- [x] Cross-model review is pending explicit user authorization and is not
  implicitly authorized by this task.

## Export implementation gate

- [x] RED tests fail for missing v2 interfaces and every parent/Harwood tamper
  class in the TDD matrix.
- [x] Parent v1 verifier replay plus all six child-byte checks pass using the
  exact frozen receipts.
- [x] Every Harwood evidence edge, all 1,536 JPEGs, and the retained source
  video pass exact receipt verification.
- [x] The v2 export writes only additive artifacts and resolves exactly 45
  unique three-part-key rows with per-game counts `8/7/7/23`.
- [x] `closure-randolph-0002`, `closure-vtv-0003`, and
  `closure-84dc8a546ac128ce09978c0f-0023` are excluded, not mapped to false.
- [x] Candidate geometry is invariant to outcome/notes/reviewed positions and
  contains no target-conditioned field.
- [x] Source groups verify as Hazen/Randolph/Harwood -> HCTV and VTV -> VTV;
  no filename inference or third production-family claim exists.
- [x] A new four-video `agu.training-annotation.v1` manifest binds the v2 index
  and all four label files without changing the TASK-0254 manifest.
- [x] Direct/symlink/hardlink/case-fold/ancestor aliases fail before destructive
  work; multi-file write failure leaves no partial generation.
- [x] Every v1 file still matches its pre-task SHA-256.

## Probe implementation gate

- [x] A frozen probe plan binds the verified v2 export, source-group manifest,
  embedding sources, checkpoint, sampling, feature dimensions, variants,
  regularization, preprocessing, thresholds, metrics, and tie-breaks.
- [x] Exactly 45 complete embeddings join one-to-one by
  `(source_video_sha256, candidate_bundle_sha256, event_id)`.
- [x] Exactly four outer game-held folds run; every fold reports held-game
  counts and selected inner settings.
- [x] Held-label perturbation tests prove that outer labels cannot alter
  representation, preprocessing, regularization, or threshold selection.
- [x] Pooled and per-game metrics are reported; HCTV/VTV aggregation is labeled
  descriptive and no production-held gate is claimed.
- [x] Export, manifest, embeddings, plan, and probe all verify with
  `runtime_consumable=false`, `formal_evaluation_eligible=false`,
  `promotion_eligible=false`, and `promoted=false`.

## Resource and quality gate

- [x] `.venv` / Python 3.11 is the only environment used; no `venv/` exists.
- [x] Batch size is one; the preflight rejects unsafe disk/memory or receipt
  state before model loading.
- [x] Resource JSONL shows configured memory/available-memory/swap/CPU limits,
  no unhandled sustained breach, bounded process-tree RSS, and a reaped child.
- [x] Guarded stop behavior is tested: exit 75, synced log, no orphan, no
  unverified final artifact, verified resume only.
- [x] Focused tests, full pytest, scoped Ruff check, scoped Ruff format check, artifact
  verifier replay, and `git diff --check` pass with exact commands recorded.
- [x] Fresh-context adversarial review returns PASS with no Critical/Required
  issue, or all such issues are fixed and re-reviewed.
- [x] Cross-model interactive review is either explicitly authorized and
  recorded, or explicitly recorded as not run; no external model is invoked
  without user authorization.

## Permanent non-promotion gate

- [x] The final record states that three of four games share HCTV production.
- [x] The final record states that this prior-stratified set is not exhaustive
  continuous full-game truth.
- [x] No checkpoint/default pointer/runtime API/readiness/blind-inference state
  changes as a consequence of TASK-0257 metrics.
- [x] No statement equates development metrics with the formal 85% gate.

Evidence summary: extractor/backbone tests `132 passed`; broader export/probe
tests `205 passed`; guarded full suite `1712 passed, 5 skipped, 15 warnings in
52.03s`; scoped Ruff and artifact verifier replay passed; fresh-context public
verifier reviews returned PASS. The repository-wide Ruff diagnostic still
fails on unrelated historical baseline debt and was not mass-formatted. The
cross-model interactive review was not run because the user has not explicitly
authorized it.

The gate is closed as a verified development diagnostic and remains permanently
closed for downstream promotion on this evidence alone.
