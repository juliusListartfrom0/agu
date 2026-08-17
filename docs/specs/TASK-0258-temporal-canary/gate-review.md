# TASK-0258 Module A Gate review — existing-45 temporal retrospective

## Decision

**MODULE-A MAP APPROVAL RECORDED; CLOSING-REVIEW REMEDIATIONS COMPLETE; FRESH
SPECIFICATION RE-REVIEW PASSED (CRITICAL 0 / REQUIRED 0); EXACT-SHA USER APPROVAL PENDING;
IMPLEMENTATION NOT STARTED; MODULE B NOT AUTHORIZED.**

The Phase-0 capability map was approved on 2026-08-16. This gate reviews only
the completeness and future acceptance conditions of
`existing-45-temporal-retrospective`. Checked items under “Specification gate”
mean the requirement is written, not that code, tests, or real artifacts exist.
The approved map's opening scope statement includes Module-A implementation
conditionally, but its build order and global stop condition require approval
of the current module specification. Map approval alone therefore authorizes
drafting at this point. Implementation remains blocked after fresh review until
the user separately approves the exact stored-file SHA-256 tuple of
`requirement.md`, `solution.md`, and this `gate-review.md`.

## Specification gate

- [x] The spec traces to the approved stable module id and preserves the one-way
  dependency to `rights-cleared-independent-descriptive-canary`.
- [x] Scope is exactly the existing TASK-0257 45-row resolved view with
  `17+/28-`, game order Hazen/Randolph/VTV/Harwood, denominators `8/7/7/23`,
  and the same three excluded `uncertain` IDs.
- [x] TASK-0257, parent-v1, child, review-plan, Harwood, source-video,
  checkpoint, embedding, manifest, group, plan, and probe receipts are frozen;
  expected values must be externally supplied.
- [x] The sole variable is exactly
  `swin3d-t-tiled-4x2s-mean-delta-v1`: four ordered two-second tiles, 16 frames
  per tile, retained Swin3D-T checkpoint/preprocessing, and one 1,536-vector
  formed as overall mean plus later-minus-earlier mean.
- [x] The sampling authority is the verified original review-plan list of 64
  frame indexes, sliced by positions `[0:16]`, `[16:32]`, `[32:48]`, and
  `[48:64]`. Candidate event endpoints may not generate a new linspace.
- [x] The plan is sealed before extraction, contains no labels/review outcomes/
  probabilities/metrics, and is byte-invariant to target perturbation.
- [x] The retrospective has exactly four outer game-held folds; threshold
  selection uses only three-game inner LOGO OOF, the exact 21-value grid,
  train-split-only standardization, no PCA, and balanced logistic `C=0.01`.
- [x] Held-label invariance covers the full non-truth computation state,
  including features, inner predictions, fitted state, threshold, held
  probabilities, and decisions—not only a settings dictionary.
- [x] Metric zero-denominator semantics and ordered balanced-accuracy/F1/
  precision/recall/higher-threshold tie-break are exact.
- [x] The baseline final evaluator independently rebuilds complete four-game
  LOGO OOF for both original ordered variants; it cannot infer them from the
  archived nested probe's selected-fold predictions.
- [x] The candidate final evaluator uses only the tiled variable. Both
  evaluators select from LOGO OOF and each refits a fresh scaler/model exactly
  once on all 45 rows, in separately addressed artifacts.
- [x] The all-45 evaluator contract freezes representation, ordered keys and
  labels, scaler statistics, coefficients, intercept, class order, threshold,
  estimator/library state, OOF evidence, and every provider receipt.
- [x] Error bounds are exactly Hazen/Randolph/VTV/Harwood `0/1/2/3` total
  errors, with Harwood additionally `FP<=2` and `FN<=1`; pooled errors are at
  most six.
- [x] A bound-only miss seals `temporal-hypothesis-rejected`; any other required
  failure is `mechanical_failure`; neither permits an alternate variable or
  Module B.
- [x] Every Module-A schema hard-codes `runtime_consumable=false`,
  `training_consumable=false`, `formal_evaluation_eligible=false`,
  `promotion_eligible=false`, and `promoted=false`.
- [x] Batch size one, sustained memory/available-memory/swap/CPU guard, exit
  `75`, process-tree reaping, and no-success-residue behavior are specified.
- [x] Every model/write boundary enforces
  `free_bytes - declared_worst_case_new_bytes >= 3,758,096,384` per filesystem.
- [x] Exact schema/receipt/path-alias/TOCTOU/atomic-generation/resume/
  deterministic-replay RED tests are specified.
- [x] `preflight_refusal` is a zero-new-artifact diagnostic before attempt
  admission; only an admitted non-bound failure may publish terminal
  `mechanical_failure`.
- [x] The mutually exclusive state machine permits recovery only for external
  `SIGINT`/`SIGTERM` on attempt 1 or 2 with a strictly advanced verified prefix
  shorter than 45 rows and all receipt/resource/disk/determinism checks still
  passing. A signal after row 45 follows private finalization. Guard, cap,
  decode, model, receipt, CAS, non-progress, unsupported-signal, and attempt-3
  failures are terminal.
- [x] At most three immutable no-clobber attempt generations are allowed. Each
  binds its exact JSONL/record/optional resume and predecessor receipts; the
  final embedding and terminal record bind the complete ordered chain and
  cumulative runtime/sample/log/disk budget.
- [x] A completed attempt and embeddings remain private until the same atomic
  `final_v1` publication as the retrospective, evaluators, and gate; a handled
  failure uses the mutually exclusive atomic `terminal_failure_v1`. No
  standalone `embeddings_complete` restart state exists.
- [x] Resource rows have an exact schema and cumulative schedule
  `t=2,4,...,43,200` (at most `21,600` rows); the combined log limit is
  `16,777,216` bytes and no partial row may be published.
- [x] The fit identity freezes SciPy `1.17.1`, exact `scipy.show_config()` SHA,
  Accelerate BLAS/LAPACK, threadpoolctl `3.6.0`, the two normalized `libomp`
  backends, and one CPU numerical thread before/after every producer/verifier
  fit. NumPy mean semantics are exact without asserting an extra reduction
  order.
- [x] A non-writing module-exclusive existing-directory lock, under-lock and
  publication-time revalidation, absent-target no-clobber publication, and
  resume device/inode/internal/file CAS are frozen.
- [x] Public plan/embedding consumers accept only opaque
  `VerifiedTemporalFeaturePlan`/`VerifiedTiledSwinEmbeddings` constructed from a
  path plus independently expected internal/file hashes; raw mappings fail.
- [x] The first atomic producer uses a distinct private, transaction-local
  produced-embedding capability because no external receipt can predate new
  bytes. It cannot cross the public boundary; after publication, externally
  frozen receipts are mandatory to construct the public verified token and
  independently trust the result.
- [x] Checkpoint verification and `torch.load` use the same open no-follow file
  descriptor. Video decoding is guarded by full pre/post hashes with an
  explicit concurrent mutate-and-restore exclusion and matching mutation tests,
  not an overstated stable-descriptor claim.
- [x] Reusable open-source components, threat boundaries, and AGU's thin
  offline-only adapter boundary are documented; no new dependency is proposed.
- [x] Module B appears only as a conditional downstream name and receipt gate;
  no candidate, rights, acquisition, window, label, or implementation design is
  advanced.
- [x] The user's 2026-08-16 TASK-0258 capability-map approval freezes the module
  boundary and conditional implementation scope, but does not satisfy the map's
  current-spec stop condition. Exact-SHA Module-A specification approval remains
  a separate open gate before implementation can start.

No implementation plan, task list, code, test, real artifact, runtime change,
or Module-B work may begin until the fresh-context specification re-review has
no unresolved Critical or Required issue and the user then approves the exact
three-file SHA tuple.

## Fresh-review findings — remediation record

- [x] The first review's TASK-0257 receipt-graph gap is closed:
  `Task0257InputPaths`, `Task0257ExpectedReceipts`, tuple orders, and full
  `validated_input_paths` coverage include parent provenance/all JPEGs/six
  children, complete v2/Harwood chains, old probe inputs, videos, and
  checkpoints.
- [x] The first review's final-evaluator gap is closed: verification receives
  complete verified rows/features and independently reruns baseline two-variant
  and candidate LOGO selection plus fresh all-45 refits; producer fit-once is
  scoped only to production.
- [x] The first review's decoder/determinism/serialization gap is closed with an
  exact backend/device/hardware/cast/numerical/canonical-byte contract and a
  bounded claim of reproducibility.
- [x] The closing review's Critical authorization finding is corrected: the
  map's general implementation scope remains conditional on its current-spec
  stop condition; a passing fresh re-review and later exact-SHA user approval
  are both mandatory before implementation starts. Semantic edits reopen both
  gates, while any byte edit invalidates the exact approval receipt.
- [x] The closing review's preflight/failure ambiguity is corrected:
  `preflight_refusal` writes no artifact, while only a post-admission terminal
  error may publish `mechanical_failure`.
- [x] The closing review's resume ambiguity is corrected with an exact
  mutually exclusive state machine, three-attempt ceiling, signal-only
  recoverable whitelist, strict prefix progress, terminal guard/cap/decode/
  model/receipt failures, and immutable per-attempt receipt chaining.
- [x] The closing review's numerical-stack gap is corrected with SciPy/config,
  Accelerate, threadpoolctl/backend and one-thread producer/verifier fit checks;
  unsupported reduction-order wording was removed.
- [x] The closing review's concurrency/TOCTOU/API gaps are corrected with a
  no-write module lock, publication revalidation, no-clobber targets, resume
  CAS, opaque path-plus-receipt verified inputs, same-descriptor checkpoint
  loading, and an explicit pre/post-hash video threat boundary.
- [x] The closing review's post-extraction crash window is corrected by keeping
  the completed attempt and embeddings private until one atomic final or
  terminal-failure generation. Only qualifying signal attempts are durable and
  resumable; private finalization is never a restart input.
- [x] Determinism equality is limited to an exact computational projection;
  attempt/resource/signal/CAS/publication receipts are audit fields. Clean and
  resume artifacts may differ bytewise but each must verify its own external
  receipt.
- [x] Optional closing-review precision items are frozen: exact approval-record
  shape, exact cumulative resource-sample schedule and row schema, global caps,
  and no partial JSONL row.

## RED and implementation gate — future work, all currently open

- [ ] RED tests fail because the new plan, tiled embedding, retrospective,
  evaluator, and gate interfaces/schemas do not yet exist.
- [ ] Map-only, missing-review, missing-user-approval, wrong/reordered exact-SHA
  tuple, self-authored approval, and post-approval drift all refuse before any
  output/model work and create no new artifact.
- [ ] The complete TASK-0257 receipt graph and old nested probe replay from
  caller-supplied internal/file receipts before plan/extraction/model work.
- [ ] Every immutable TASK-0257 file retains its pre-Module-A SHA after every
  success and injected failure path.
- [ ] The label-hidden plan has exactly 45 canonical keys and exact 4x16 slices
  of the two verified review plans; label perturbation yields identical bytes.
- [ ] All 45 rows have four finite 768-dimensional derivation vectors and one
  exactly recomputed finite 1,536-dimensional model input; no other selectable
  representation exists.
- [ ] Two empty-state real extractions under the frozen environment are
  identical on the exact computational projection. A valid signal-only
  interruption/resume chain matches the clean computational projection without
  recomputing the verified prefix; audit fields and full artifact bytes may
  differ and each artifact verifies its own external receipt. All
  non-whitelisted failures are terminal and cannot resume.
- [ ] Exactly four nested outer folds and their three-game inner OOF selections
  replay; held-label perturbation changes only truth-derived fields.
- [ ] Every stored confusion count and metric recomputes exactly using the
  frozen zero-denominator rules and threshold decision.
- [ ] Baseline complete LOGO OOF exists independently for both original
  variants in original order; candidate complete LOGO OOF exists only for the
  tiled variant; all selection tie-breaks replay.
- [ ] Each evaluator's all-45 scaler/model is fit exactly once, has correct
  dimensions/state, and a verifier given complete verified rows/features runs
  its own fresh LOGO fits and all-45 refit to replay every field from external
  receipts. Producer fit-once does not suppress verifier refits. Every fit
  verifies the exact SciPy/config/Accelerate/threadpool stack and one-thread
  backends before and after fitting.
- [ ] The final decision exactly matches the per-game/Harwood/pooled error
  bounds and cannot emit pass when another ordered mechanical check fails.
- [ ] Every eligibility flag mutation, runtime/generic-training/default-pointer/
  readiness/blind-inference consumer, and formal/promotion attempt fails.
- [ ] Direct, symlink, hardlink, case-fold, ancestor/descendant, cross-output,
  and input/output aliases fail before destructive work or model loading.
- [ ] Raw mappings/forged wrappers cannot cross the public plan or embedding
  boundary; only path-plus-independent-internal/file-receipt constructors create
  opaque verified values accepted by extraction/evaluation.
- [ ] A competing invocation fails the non-writing module lock. Injected
  temporary-write, file-sync, publication revalidation, no-clobber rename,
  generation-sync, existing-target, and resume-CAS failures expose no partial
  stable generation and preserve every prior immutable generation.
- [ ] A crash/SIGKILL after row 45 but before terminal publication exposes
  neither a standalone completed attempt nor embeddings; only private staging
  may remain, and it is never accepted as restart evidence.
- [ ] Checkpoint hash/load/stat use the same verified no-follow descriptor.
  Observable source-video mutations fail the full pre/post hash boundary and
  discard staging; tests preserve the explicit mutate-and-restore exclusion.
- [ ] Disk accounting subtracts all simultaneous worst-case bytes before
  enforcing the `3.5 GiB` reserve at every frozen boundary. Current free space
  already includes retained attempts, so they are verified for cumulative caps
  but not subtracted twice; declared new bytes cover maximum remaining attempts,
  logs, resume/record/embedding members, and the larger mutually exclusive
  final or terminal-failure staging generation.
- [ ] Each of at most three attempts publishes an immutable exact log/record and
  only a qualifying attempt-1/2 signal adds a resume; predecessor and final
  receipts replay as a complete chain with strictly increasing prefix/counters.
- [ ] Resource sampling occurs only at cumulative `t=2,4,...,43,200`, emits at
  most `21,600` complete exact-schema rows and `16,777,216` total bytes, and
  never publishes a partial row.
- [ ] Sustained resource or cumulative-cap failures exit `75`, sync or discard
  the current untrusted log, reap the process group, leave no final/random
  temporary, publish terminal failure without a resume, and leave no worker.
- [ ] Focused tests, full pytest, scoped Ruff check, scoped Ruff format check,
  all real verifier replays, and `git diff --check` pass with exact commands and
  output receipts recorded.

## Mechanical outcome gate — future real evidence

Exactly one of these outcomes may be sealed after every implementation item
above has evidence:

### `mechanical_pass`

- [ ] Hazen support 8 has `FP+FN=0`.
- [ ] Randolph support 7 has `FP+FN<=1`.
- [ ] VTV support 7 has `FP+FN<=2`.
- [ ] Harwood support 23 has `FP+FN<=3`, `FP<=2`, and `FN<=1`.
- [ ] Pooled errors are at most six.
- [ ] All non-metric ordered mechanical checks pass.

A checked pass does not promote anything. It only creates the exact receipt
precondition that a later, separately reviewed Module-B spec would need.

### `temporal-hypothesis-rejected`

- [ ] Every non-bound mechanical check passes.
- [ ] At least one frozen error bound fails and the exact observed counts are
  preserved.
- [ ] TASK-0258 stops before Module B with no feature/threshold/model/source
  retry.

### `mechanical_failure`

- [ ] A mandatory non-bound check fails only after a no-write preflight has
  passed and an attempt has been admitted. A preflight failure remains
  artifact-free `preflight_refusal`, not this outcome.
- [ ] Receipt/schema/sampling/leakage/evaluator/path/atomic/resume-CAS/resource/
  cumulative-cap/disk/process, decode/model/determinism, non-progress,
  unsupported-signal, or incomplete-attempt-3 failure is terminal and cannot
  publish a resume.
- [ ] No pass receipt or trusted partial final generation is published.
- [ ] An atomic `terminal_failure_v1/mechanical_failure.json` records only
  receipts that actually verified, binds the complete published immutable
  attempt chain without fabricating a failed-publication receipt, states
  `final_generation_published=false`, contains no evaluator receipt, and its
  parent terminal generation does not coexist with `final_v1`.
- [ ] The exact failure is recorded without fabricating a verified provider
  receipt or changing a frozen choice.

## Fresh-review gate

- [x] Current remediated specification receives a fresh-context adversarial
  re-review with no Critical or Required issue.
- [ ] After that re-review, the user approves the exact ordered stored-file
  SHA-256 tuple for `requirement.md`, `solution.md`, and `gate-review.md`.
- [ ] The resulting `agu.module-a-spec-approval.v1` record and its human
  statement are externally receipt-bound without editing any of those three
  files. Any byte drift requires user reapproval; semantic drift first requires
  another fresh specification review.
- [ ] After implementation, a different fresh-context review reads the approved
  capability map/spec, exact diff, RED/GREEN evidence, external receipt replay,
  real artifacts, immutable attempt/resource logs, and atomic/resume tests.
- [ ] Every implementation-review Critical or Required issue is fixed and the
  review is rerun before any outcome is treated as closed.

Fresh specification re-review has passed. The exact-SHA user approval, including
its external approval record, is the only remaining pre-implementation gate;
map approval does not satisfy it. The later implementation-review checkboxes
cannot pass from documentation alone.

## Permanent non-promotion and downstream boundary

- [x] The spec states that the 45 rows are prior-stratified/non-exhaustive
  development evidence across only two production families.
- [x] The archived TASK-0257 result is a fixed comparator, not a selectable
  variable or formal baseline.
- [x] No Module-A metric changes a checkpoint/default pointer, API, runtime
  behavior, readiness, blind inference, or the formal 85% gate.
- [x] A successful gate does not by itself authorize Module B, independent
  media access, or any promotion claim.

Current evidence summary: capability map approved, with its Module-A
implementation scope still conditional on the current-spec stop condition;
first and closing fresh-review findings are remediated in the Module-A
requirement, solution, and this gate; fresh re-review passed with Critical 0 and
Required 0; exact-SHA user approval and its external approval record remain
pending and unchecked; implementation has not started; Module B is not
authorized; no Module-A output receipt exists.
