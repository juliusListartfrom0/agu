# TASK-0267 Phase-0 Capability Map — closed independent-VLM evidence

Status: **PROPOSED — AWAITING USER APPROVAL.** This document defines module
boundaries and dependency direction only. No module specification, production
code, model run, label reveal, fusion run, runtime change, or promotion action
is authorized by this draft.

## Why this must be decomposed

AGU's generic independent-shot VLM v1 contract is deliberately
`open_world_structural_only`, `legacy_provenance_only`, and permanently
non-promotable. That is the correct compatibility boundary, but it cannot prove
that arbitrary producer fields are label-free or independent of Codex. The
existing nested scene/video auxiliary contract is closed and receipt-bound, yet
its current consumer still accepts legacy v1 VLM predictions and emits a
non-promotable diagnostic. A trusted independent-VLM path therefore needs
separate capabilities for a shared model-visible plan, independent VLM
predictions, independently frozen base-model predictions, label-free fusion,
delayed evaluation, and eventual runtime adoption.

Current artifact evidence confirms that this is not a paperwork-only gap. The
frozen nested v4 plan contains 32 exact three-part keys and has plan SHA-256
`4d3573b89d33b302431caa56c3cef6443290080dc84e678e995b146aebd279bf`.
The current repository contains 33 legacy prediction artifacts, but none binds
that plan SHA; the largest key overlap is 21/32. Rebinding any older prediction
is forbidden. The nested auxiliary v4 artifact remains a structural,
development-only diagnostic and does not make missing VLM evidence appear.

This initiative advances one necessary component of the full-game statistics
goal: trustworthy shot-validity evidence. It does not claim that shot validity
alone completes all box-score fields or proves the requested >=85% full-game
accuracy.

## Assumptions frozen for map review

1. Generic plan/prediction/evaluation v1 stays byte-compatible, open-world,
   legacy-only, and permanently non-promotable. TASK-0267 is additive v2.
2. The independent VLM may receive only pixels plus a fixed prompt and sampling
   contract. The base-evidence runner may receive only pixels/features plus an
   already frozen policy. Training annotations, review notes, target labels,
   selection-class quotas and held metrics are not model-visible input.
3. Codex may create offline annotations or perform human-style review only
   after independent prediction file receipts have been frozen. Codex never
   supplies an AGU/VLM runtime answer or a prediction row.
4. A self-consistent internal artifact SHA is integrity evidence, not an
   external receipt. Every state transition that matters to evaluation or
   adoption requires caller-supplied expected internal and stored-file SHA-256
   values from an earlier frozen registry.
5. Current 32-row and 45-row development data stay non-formal. New schemas do
   not upgrade their data eligibility, metrics, or readiness status.
6. No new dependency, model family, network service, or language stack is
   required for the pre-runtime modules. Eventual runtime adoption is a separate
   gate and must preserve canonical `.venv` Python 3.11 and resource guards.

## Capability map

| Module id | Responsibility | Depends on | Output boundary |
| --- | --- | --- | --- |
| `label-hidden-shared-evidence-plan` | Project an externally receipt-bound parent evidence plan into an exact, model-visible, label-hidden plan containing only media geometry, three-part keys, fixed input contracts and non-runtime policy. | Existing closed parent-plan verifier | One exact-schema v2 shared plan plus external internal/file receipts; no labels, class quotas, annotation hashes, training manifest, review text or evaluation metrics. |
| `receipt-bound-independent-vlm-predictions` | Run one declared local independent VLM against only the shared plan and seal a closed, typed prediction vector with complete exact coverage and full model/prompt/preprocess receipts. | `label-hidden-shared-evidence-plan` | One exact-schema v2 VLM prediction artifact and immutable file receipt; never generic-v1 promotion or direct runtime consumption. |
| `receipt-bound-base-evidence-predictions` | Apply one previously frozen, externally receipt-bound base-evidence policy to exactly the same shared plan without selecting/refitting on target labels. | `label-hidden-shared-evidence-plan`, an independently approved frozen base-policy provider | One exact-schema v2 base prediction/probability artifact with policy, feature and input receipts. Existing OOF artifacts qualify only for development diagnostics, not deployment/formal use. |
| `label-free-base-vlm-fusion` | Combine the exact VLM and base evidence vectors under one pre-frozen rule without reading truth, held metrics or target-derived fields. | `receipt-bound-independent-vlm-predictions`, `receipt-bound-base-evidence-predictions` | One distinct fused-evidence v2 schema with exact shared-plan/VLM/base/policy receipts; no generic prediction schema and no promotion decision. |
| `prediction-first-delayed-evaluation` | Verify base, VLM and fusion bytes plus their external receipts before opening separately sealed truth, then compute fixed-policy overall and per-source metrics under an independently frozen data-eligibility contract. | Both prediction modules and `label-free-base-vlm-fusion` | One closed evaluation artifact that distinguishes observed metrics, data eligibility, metric gate and promotion/adoption eligibility. Existing development data can only emit non-formal diagnostics. |
| `resource-guarded-runtime-adoption` | Adopt a previously frozen base+VLM fusion policy into AGU only after a separate formal dataset passes every data, metric, provenance, performance, and rollback gate. | `prediction-first-delayed-evaluation` formal pass | A versioned runtime policy/default pointer with resource monitoring and rollback evidence. This module is unreachable until a future formal gate passes and does not authorize Codex runtime answers. |

Dependency direction is acyclic and one-way:

`label-hidden-shared-evidence-plan`
→ (`receipt-bound-independent-vlm-predictions`,
`receipt-bound-base-evidence-predictions`)
→ `label-free-base-vlm-fusion`
→ `prediction-first-delayed-evaluation`
→ `resource-guarded-runtime-adoption`

The delayed evaluator may also consume unfused v2 predictions as an explicit
baseline, but neither evaluation nor runtime may feed labels, scores, errors, or
threshold changes back into the shared plan or frozen predictions.

## Cross-module invariants

- Every schema is exact and additive v2; generic v1 remains permanently
  non-promotable and cannot consume v2 or fused evidence.
- Every meaningful edge binds caller-supplied expected internal and stored-file
  SHA-256 values. Self-resealing cannot establish provenance.
- The model-visible shared plan excludes labels, review material and
  label-derived parent fields. Both evidence vectors cover the same ordered
  keys; prediction rows contain only bounded typed data and no decision-bearing
  free text.
- Predictions are complete and externally frozen before any truth path opens.
  Evaluation cannot feed labels, scores, errors or thresholds back upstream.
- Plan/key/geometry coverage is exact and ordered. Duplicate, missing, extra,
  non-finite, unknown-schema or path-aliased input fails closed.
- The VLM is independent from Codex; the base runner cannot refit/select on
  target data. Both are batch/resource guarded and atomically published. Codex
  is limited to post-freeze offline annotation.
- Runtime adoption requires a later formal dataset pass and explicit approval;
  a shot-validity pass alone cannot establish full-game >=85% readiness.

## Frozen build order and human gates

1. The user reviews and approves this capability map.
2. Specify, review, plan, task, implement and independently review
   `label-hidden-shared-evidence-plan` only.
3. Repeat the gated cycle for `receipt-bound-independent-vlm-predictions`.
4. Independently specify and gate `receipt-bound-base-evidence-predictions`;
   target data may not select or refit its provider policy.
5. Only after both complete prediction vectors exist may the fusion module be
   specified and implemented.
6. Only after base, VLM and fusion receipts are frozen may the
   delayed-evaluation module open or consume truth.
7. Runtime adoption requires a separate formal pass and an explicit later user
   approval; it is not authorized by approving this map or earlier modules.

No `requirement.md`, `solution.md`, implementation task list, or production code
for TASK-0267 may be written before this Phase-0 map is approved. Approval of the
map authorizes specification of the first module only, not model execution,
truth reveal, fusion evaluation, runtime adoption, or readiness changes.

## Reusable open-source capability and AGU boundary

- Reuse Python stdlib strict JSON/hashing, the existing nested-plan/auxiliary
  verifiers as development provenance, local frame/native-video runners,
  base-feature adapters, OpenCV/video IO, retained open-weight model adapters,
  atomic artifact patterns and resource guards.
- No new framework or dependency is justified for the closed-contract layer;
  exact Python verifiers match the repository's established artifact style and
  avoid a second schema/runtime stack.
- Future model selection still follows AGU's open-source license/revision/weight
  receipt gate. This map does not authorize a new download or remote inference
  service.
- AGU owns only the Python plan/prediction/fusion/evaluation adapters and later
  runtime policy. It does not add a BFF, another language stack, identity
  processing, or unrelated statistics behavior.

## Phase-0 approval record

Pending. Record the exact capability-map file SHA and the user's approval only
after the final reviewed bytes are stable. Until then TASK-0267 is planning-only,
readiness remains `not_ready`, and blind inference remains paused.
