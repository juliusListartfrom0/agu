# TASK-0258 Phase-0 Capability Map — temporal diagnostic and independent canary

Status: **APPROVED by the user on 2026-08-16.** This document freezes capability
boundaries, dependency direction, build order, receipts, and stop conditions.
Approval authorizes specification and implementation of module A only. Module B
remains conditional on module A's mechanical pass and its own rights/disk gates.

## Why this must be decomposed

TASK-0257 is a verified development diagnostic over 45 determinate rows
(`17+/28-`) from four games but only two production families. Its pooled
precision/recall/F1 is `0.777778/0.823529/0.800000`; Harwood is the weakest
held game at `0.400000/0.666667`. The observed errors are consistent with both
sparse temporal sampling and production/selection shift. Those hypotheses
cannot be tested by one capability without letting development evidence leak
into an allegedly independent check.

TASK-0258 therefore has two independently testable modules. The first may look
back only at the already-known 45 rows. The second may inspect genuinely new
media only after the first module is mechanically valid, and the second can
never feed changes back into the first.

## Capability map

| Module id | Responsibility | Depends on | Output boundary |
| --- | --- | --- | --- |
| `existing-45-temporal-retrospective` | Test one pre-registered tiled Swin temporal variable on the exact TASK-0257 45-row development set, preserving the old rows and baseline receipts. | — | One externally receipt-bound, development-only diagnostic plus separately frozen baseline and candidate final evaluators; no runtime/formal/promotion eligibility. |
| `rights-cleared-independent-descriptive-canary` | Apply both already-frozen final evaluators once to the same sealed 24 windows from one rights-cleared game in a finite, receipt-bound candidate universe and a production family distinct from HCTV and VTV. Report paired descriptive results only. | `existing-45-temporal-retrospective` mechanical pass | One externally receipt-bound descriptive canary record whose primary outcome is paired `delta balanced_accuracy`; the new media, labels, errors, and metrics are forbidden training/tuning inputs. |

Dependency direction is one-way:

`existing-45-temporal-retrospective` → `rights-cleared-independent-descriptive-canary`

There is no reverse edge. A later desire to tune from the canary would require
a new task, a new development set, and another untouched independent canary;
TASK-0258 cannot reinterpret module B as module A input.

## Frozen build order and review gates

1. The user reviews and approves this Phase-0 map.
2. Specify, review, plan, task, implement, and mechanically verify
   `existing-45-temporal-retrospective` only.
3. Freeze the module-A baseline final evaluator and candidate final evaluator,
   with separate independent expected receipts, before any module-B metadata or
   media is inspected. Module-A metrics may be reported, but may not trigger
   another feature, checkpoint, tiling, aggregation, classifier, threshold-grid,
   or preprocessing attempt inside TASK-0258.
4. Only after the module-A mechanical gate passes, specify
   `rights-cleared-independent-descriptive-canary`.
5. Before selecting or downloading any media, create a finite metadata candidate
   universe and a separate rights receipt. Freeze their internal and file
   SHA-256 values and a metadata-only candidate order.
6. If and only if one candidate passes the frozen metadata, rights,
   independence, and disk gates, acquire that one game and freeze a label-hidden
   24-window canary plan.
7. Run and seal one baseline prediction and one candidate prediction for every
   one of the same 24 windows before labels are revealed. Reveal labels once,
   compute the paired descriptive result, and stop. Do not tune, retry on a
   different candidate because of labels/metrics, promote, or resume blind
   inference.

No `requirement.md`, `solution.md`, implementation plan, or task breakdown may
be written for either module before the immediately preceding human review gate
has passed.

## Module A — `existing-45-temporal-retrospective`

### Exact responsibility

- Consume exactly the verified TASK-0257 resolved view: 45 rows, four games,
  `17` positives, `28` negatives, and the same three excluded `uncertain` rows.
- Keep every TASK-0257 file and SHA byte-for-byte unchanged. The old 45 rows are
  permanently `development_diagnostic_only`, non-exhaustive, non-formal, and
  non-promotable; no later result can upgrade their status.
- Introduce exactly one new representation variable:
  `swin3d-t-tiled-4x2s-mean-delta-v1`.
- For each existing half-open eight-second window, divide the window into four
  contiguous, ordered two-second tiles. Uniformly sample exactly 16 frames per
  tile with the existing TASK-0257 Swin3D-T preprocessing and the retained
  `swin3d_t-7615ae03.pth` checkpoint (file SHA-256
  `7615ae035996b65eb38dad437ae533d2dfcd36f9f89d28c0f0fa7bfb8e6b3130`).
  Let their 768-dimensional embeddings be `e0..e3` in temporal order. Produce
  exactly one 1,536-dimensional variable:
  `concat((e0+e1+e2+e3)/4, (e2+e3)/2 - (e0+e1)/2)`. The second half is the
  later-two-tile mean minus the earlier-two-tile mean; it is not `e3-e0`.
- Permit no alternate tile count, offset, overlap, frame count, pooling,
  aggregation/difference formula, attention head, backbone, checkpoint, crop,
  prompt, VLM answer, or feature concatenation. The archived TASK-0257 result
  is a fixed comparator, not another selectable variant.
- Reuse the TASK-0257 nested outer-game-held protocol for this sole variable:
  train-split-only standardization, no PCA, logistic `C=0.01`, the pre-existing
  `0.00..1.00` step-`0.05` threshold grid, and selection from inner game-held
  OOF predictions only. Outer-held labels may affect metrics only.
- After the retrospective replay passes, freeze two distinct non-runtime final
  evaluators for module B. Both use four-game LOGO OOF over only the old 45 rows,
  `StandardScaler` fit inside each LOGO training split, no PCA, balanced
  logistic regression (`C=0.01`, `class_weight="balanced"`, `max_iter=5000`,
  `random_state=0`), and the frozen `0.00..1.00` step-`0.05` threshold grid.
- The **baseline final evaluator** considers exactly two variants in this
  original order: `[swin3d_t, mvit_v2_s+swin3d_t]`. Pool the four held-game OOF
  predictions for each `(variant, threshold)` and select exactly one pair by
  maximizing, in order, balanced accuracy, F1, precision, and recall; remaining
  ties prefer the earlier original variant and then the higher threshold.
- The **candidate final evaluator** considers only
  `swin3d-t-tiled-4x2s-mean-delta-v1`. Use the same four-game LOGO OOF fitting,
  metric order, threshold grid, and higher-threshold tie-break; there is no
  candidate-variant choice.
- For both selections, precision is zero when `TP+FP=0`; recall is zero when
  `TP+FN=0`; specificity is zero when `TN+FP=0`; F1 is zero when precision plus
  recall is zero; balanced accuracy is `(recall+specificity)/2`. The old 45-row
  supports contain both classes, so the final selection metric is computable.
- Refit each selected representation exactly once on all 45 rows with its own
  all-45 `StandardScaler` and balanced logistic model. Freeze representation,
  scaler statistics, coefficients, intercept, threshold, label/key order, and
  every input receipt in two separately addressed artifacts. Neither artifact
  may be changed after module B's candidate-universe work begins.

### Mechanical go/no-go (still not a promotion claim)

Module A passes mechanically only when all of the following are true:

1. Every frozen TASK-0257 external receipt, all four source videos, the Swin
   checkpoint, and the exact 45-row join replay successfully.
2. A label-hidden tiled-feature plan was sealed before extraction; it contains
   exactly the single variable above and hard-false runtime/formal/promotion
   flags.
3. All 45 rows produce exactly one finite 1,536-dimensional vector with exact
   key, order, sampling, checkpoint, and plan bindings; deterministic replay,
   atomic publication, verified resume, and no residual staging process/file
   are proven.
4. Exactly four outer game-held folds complete, held-label perturbation cannot
   change any fitted/selected setting, and all reported confusion counts and
   metrics recompute exactly.
5. The frozen baseline and candidate final evaluators independently replay
   their four-game LOGO OOF selection, exact ordered tie-breaks, and all-45
   refit from external receipts; neither can be consumed by AGU runtime or any
   formal/promotion path.
6. On the fixed per-game denominators Hazen/Randolph/VTV/Harwood=`8/7/7/23`,
   held errors (`FP+FN`) are no greater than `0/1/2/3`, respectively; Harwood
   additionally has `FP<=2` and `FN<=1`, so pooled errors are at most six. If
   any bound fails, seal `temporal-hypothesis-rejected` and stop before module
   B; no alternate representation or retry is allowed.
7. Resource and disk guards pass. At every write/model boundary, free space
   after the declared worst-case new bytes must remain at least `3.5 GiB`
   (`3,758,096,384` bytes); extraction remains batch size one under the existing
   sustained memory/available-memory/swap/CPU guard.

This is a pre-registered retrospective diagnostic gate, not a formal accuracy
or promotion gate. A mechanically valid result that misses the error bounds is
still informative, but it rejects the temporal hypothesis for TASK-0258 and
must not be hidden by trying another variable. Any failed item stops TASK-0258
before module B.

## Module B — `rights-cleared-independent-descriptive-canary`

### Preconditions and responsibility

- Require the exact module-A mechanical-pass receipt and both separately frozen
  final evaluators. Missing or drift in either evaluator stops before candidate
  selection or media acquisition.
- First enumerate a finite metadata-only candidate universe. Each row must bind
  a stable candidate/game id, canonical publisher page and revision, media URL,
  duration/size metadata, production family, game identity, continuous 5x5
  status, license text/URL/revision, attribution, permitted AGU research/open-
  source use, and a literal eligibility decision. Open-ended search after seeing
  canary labels or metrics is forbidden.
- Seal the candidate universe and a separate rights decision before selecting
  or downloading pixels. A self-declared platform license, missing underlying
  media permission, non-redistributable/gated terms, ambiguous attribution, or
  unverifiable revision fails closed.
- Select at most one game using only the frozen metadata order and eligibility
  rules. It must be a genuinely different game and a production family distinct
  from both HCTV and VTV; a re-encode, mirror, clip, or derivative of an existing
  game is not independent.
- Before acquisition, require that current free disk minus the declared
  worst-case download, temporary, extracted-evidence, and final-artifact bytes
  remains at least `3.5 GiB` (`3,758,096,384` bytes). If this cannot be proven,
  stop before download rather than cleaning protected evidence opportunistically.
- After media verification, freeze exactly 24 deterministic geometry/time-only,
  label-hidden, half-open eight-second canary windows and one review plan.
  Window selection may not use either evaluator's score, VLM answers, PBP,
  review labels, or known shot times.
- Apply the frozen baseline final evaluator and frozen candidate final evaluator
  exactly once to every one of the same 24 window keys. Seal both complete,
  ordered 24-prediction vectors and their feature/input receipts before any
  review label is revealed. Missing, extra, duplicate, reordered, or unequal
  window coverage fails before label reveal.
- Reveal and seal the labels only after both prediction seals exist. Exclude an
  `uncertain` label symmetrically from both metric calculations; all determinate
  rows must remain paired. The primary canary outcome is
  `paired_delta_balanced_accuracy = candidate_balanced_accuracy - baseline_balanced_accuracy`
  on that one identical determinate subset. Report both confusion matrices,
  class denominators, balanced accuracies, and the paired delta; all other
  metrics are secondary descriptive fields.
- After label reveal, neither evaluator may be refit, recalibrated, reselected,
  rethresholded, or rerun with changed preprocessing/features. The new media
  and its labels may not change sampling, label policy, the candidate universe,
  or any follow-up acquisition.
- If the canary has insufficient determinate positive or negative support,
  report the affected metric as not computable and stop. Do not choose another
  source based on the observed class balance or result.

### Permanent output boundary

The canary must be hard-coded and verified as
`runtime_consumable=false`, `training_consumable=false`,
`formal_evaluation_eligible=false`, `promotion_eligible=false`, and
`promoted=false`. It is not an 85% gate, does not establish production
readiness, does not update a checkpoint/default pointer, and cannot resume
blind inference. A future formal evaluation requires a separately specified,
untouched, rights-cleared and sufficiently powered dataset.

## External receipt boundary

An “external receipt” means the verifier's caller supplies an expected
internal SHA-256 and, for stored artifacts, file SHA-256 from a previously
frozen registry/plan. A verifier must never learn the expected value from the
artifact it is currently checking. Self-consistent resealing is not evidence.

| Provider | Required externally frozen receipts | Consumer |
| --- | --- | --- |
| TASK-0257 | v2 export, immutable v1 parent and six children, source groups, four-video training manifest, MViT/Swin embeddings, probe plan/probe, all four source videos, and all linked Harwood evidence | Module A input verifier |
| Module-A pre-registration | tiled-feature plan, ordered baseline variants, exact MViT/Swin checkpoints, sampling/preprocessing contracts, four-game LOGO/tie-break protocol, threshold grid, balanced-logistic contract, and disk/resource policy | Module-A extractor, retrospective probe, and final-evaluator freezer |
| Module-A result | tiled embeddings, retrospective diagnostic, mechanical-pass record, frozen baseline final evaluator, and frozen candidate final evaluator, each with exact OOF-selection/all-45-refit receipts | Module B preflight |
| Module-B metadata gate | finite candidate-universe snapshot plus independent per-candidate rights decision and deterministic order | Module-B selector/downloader |
| Module-B acquisition | selected source manifest, license/attribution receipt, exact media size/file SHA, and production-family independence record | Module-B plan/materializer |
| Module-B label-hidden evidence | 24-window selection, review plan, retained-frame manifest, and all materialized-frame hashes | Both final evaluators and prediction sealers |
| Module-B prediction freeze | exact baseline and candidate 24-row prediction seals, identical ordered keys, evaluator receipts, feature receipts, and proof that no label was revealed | Label reveal and paired-result verifier |
| Module-B revealed evidence | sealed review plus its external receipt and the exact common determinate-row subset | Paired `delta balanced_accuracy` result verifier |

Every downstream receipt must bind its immediate providers by both expected
content identity and file identity. Missing, extra, reordered, aliased,
non-finite, path-escaping, or receipt-drifted inputs fail before output or model
loading.

## Global stop conditions

Stop without advancing the dependency graph when any of these occurs:

- the user has not approved this capability map or the current module spec;
- free-space headroom would fall below `3.5 GiB` after declared worst-case new
  bytes;
- an external receipt, source byte, license revision, game identity, production
  family, or independence claim cannot be verified exactly;
- module A introduces more than the single frozen tiled Swin variable, reads a
  held label for selection, or changes any TASK-0257 byte/status;
- module A lacks complete finite embeddings, exact nested replay, a verified
  baseline final evaluator, a verified candidate final evaluator, the frozen
  per-game error bounds, or a clean resource/process/staging exit;
- module B begins before module A mechanically passes, lacks a finite sealed
  metadata universe or separate rights receipt, or attempts a second source
  because the first result is inconvenient;
- either final evaluator changes after candidate-universe work starts, the two
  evaluators do not cover the same sealed 24 keys exactly once, or any review
  label is revealed before both prediction vectors are sealed;
- any new-media label, score, error, or metric changes a model, threshold,
  preprocessing rule, selection policy, prompt, or follow-up acquisition;
- any artifact attempts to become runtime, training, formal, promotion,
  readiness, default-checkpoint, or blind-inference input.

Stopping is a valid outcome. It must be recorded as `not_ready`/descriptive or
not-computable evidence, not repaired through post-hoc tuning inside TASK-0258.

## Reusable open-source capability and AGU boundary

- Reuse the already-retained torchvision Swin3D-T implementation/checkpoint,
  OpenCV/video IO, canonical `.venv` Python 3.11, existing receipt verifiers,
  atomic artifact patterns, nested evaluation utilities, and sustained resource
  guard. No new dependency, model family, language stack, or external runtime
  service is in scope.
- Use official publisher/repository APIs and license pages only to form the
  future finite metadata/rights universe; payload acquisition is downstream of
  that gate.
- AGU may implement only the thin tiled-sampling/receipt/evaluation adapters
  needed to expose stable offline artifacts. It may not alter the v3 inference
  preprocessing contract, service/API behavior, default model registry,
  runtime VLM answers, or BFF boundary.

## Phase-0 approval record

The user approved the two module ids, the one-way dependency, the sole
`swin3d-t-tiled-4x2s-mean-delta-v1` variable, the mechanical module-A gate, the
separately frozen baseline/candidate final evaluators, the finite rights-first
module-B gate, same-24-window paired `delta balanced_accuracy`, the no-tuning
canary boundary, and the `3.5 GiB` disk reserve on 2026-08-16. Module-A
specification may now proceed; no independent media may be acquired until all
later module-B gates are satisfied.
