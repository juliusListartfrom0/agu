# TASK-0258 Module A Requirement — existing-45 temporal retrospective

Status: **MODULE-A SPECIFICATION DRAFTING AUTHORIZED; FRESH-CONTEXT
SPECIFICATION RE-REVIEW AND USER APPROVAL OF THE EXACT SPECIFICATION BYTES ARE
REQUIRED BEFORE IMPLEMENTATION.** The user approved only the Phase-0 capability
map on 2026-08-16. Although the map's opening scope statement conditionally
includes Module-A implementation, its build order and global stop condition
still require approval of the current module specification. Map approval alone
therefore permits drafting these documents but does not approve their current
contents or authorize implementation to start. Implementation remains blocked
until a fresh-context review reports no unresolved Critical or Required issue
and the user then approves the exact stored-file SHA-256 tuple of
`requirement.md`, `solution.md`, and `gate-review.md`. It does not authorize
Module B metadata or media work, runtime use, formal evaluation, or promotion.

## Capability trace and objective

This document specifies only capability-map module
`existing-45-temporal-retrospective` (Module A). It must test one and only one
pre-registered temporal representation on the exact verified TASK-0257
development rows, then freeze two separately addressed offline final evaluators:

- the sole candidate representation is
  `swin3d-t-tiled-4x2s-mean-delta-v1`;
- the frozen comparator is the archived TASK-0257 result, not a selectable
  representation in the retrospective;
- the baseline final evaluator selects from the original ordered variants
  `[swin3d_t, mvit_v2_s+swin3d_t]`;
- the candidate final evaluator uses only the sole tiled representation.

Success means that the complete receipt graph, tiled extraction, nested replay,
four-game LOGO final-evaluator selection/refit, resource guard, and immutable
publication are mechanically valid. It is not success evidence for runtime,
formal accuracy, the 85% gate, production-family generalization, readiness, or
promotion. A valid replay that misses the frozen error bounds is a required
negative result named `temporal-hypothesis-rejected`, not permission to try a
second representation.

Module B, `rights-cleared-independent-descriptive-canary`, is only a conditional
downstream consumer named by the approved capability map. This specification
does not define its candidate universe, rights decision, acquisition, windows,
labels, metrics beyond the map, commands, schemas, or implementation.

## Frozen assumptions and invariants

1. The input population is exactly the TASK-0257 resolved view: 45 unique rows
   ordered by Hazen, Randolph, VTV, Harwood and then `event_id`; per-game
   denominators are `8/7/7/23`; class support is `17` positive and `28`
   negative. The excluded rows remain exactly `closure-randolph-0002`,
   `closure-vtv-0003`, and
   `closure-84dc8a546ac128ce09978c0f-0023`. An `uncertain` row can never become a
   negative row.
2. The row key is exactly
   `(source_video_sha256, candidate_bundle_sha256, event_id)`. Missing, extra,
   duplicate, aliased, or reordered keys fail before extraction or fitting.
3. The original TASK-0254/TASK-0257 artifacts, source videos, checkpoints, and
   every byte named by their receipts are immutable inputs. Module A writes
   only to its new output directory and a same-directory temporary/resume area.
4. The authoritative temporal sampling evidence is the verified TASK-0254 and
   Harwood review-plan `frame_indexes`, which contain exactly 64 ordered samples
   for each original half-open eight-second review window. Candidate-event
   `start_frame/end_frame` are only the first and last materialized review
   samples (about 7.875 seconds apart at 8 samples/second); they are forbidden
   as endpoints for a new linspace.
5. Every new Module-A artifact has purpose
   `development_diagnostic_only` and exact hard-false fields
   `runtime_consumable=false`, `training_consumable=false`,
   `formal_evaluation_eligible=false`, `promotion_eligible=false`, and
   `promoted=false`. “Training consumable” here means eligible for any AGU
   training route outside this sealed Module-A diagnostic; the explicitly
   specified internal fits below remain allowed.
6. TASK-0257 contains four games but only two production families: Hazen,
   Randolph, and Harwood are HCTV; VTV is VTV. No result may be described as
   production-held or independent.
7. All expected internal and stored-file SHA-256 values are supplied by the
   caller from this approved specification, the frozen Module-A plan, or a
   previously frozen registry. A verifier may not obtain its expected value
   from the artifact it is currently verifying.
8. Module-A implementation authorization is an external human receipt over the
   ordered tuple `(requirement.md file SHA-256, solution.md file SHA-256,
   gate-review.md file SHA-256)` after the fresh specification re-review passes.
   Approval is recorded without editing these three files. Any byte change
   invalidates that exact approval receipt and requires user reapproval; any
   semantic change also clears the fresh-review result and requires a new fresh
   review before the user can approve the new tuple.

## Frozen TASK-0257 receipt boundary

The implementation must replay these receipts before reading labels into an
evaluation or loading a model. Internal SHA is `n/a` only where the provider
contract has no internal canonical hash.

The caller-facing path and receipt structures are closed contracts, not asset
roots from which the implementation may discover inputs. They must explicitly
name the parent selection, source manifest, review plan, raw-frame manifest,
every manifest-referenced JPEG, sealed review, all six parent children, parent
v1 export, complete v2 export and both v2 children, source groups, four-video
training manifest, both old embedding files, old probe plan and probe, the full
Harwood selection/source-manifest/review-plan/raw-manifest/JPEG/sealed-review
chain, all four source videos, and both checkpoints. Every leaf path must occur
exactly once in `validated_input_paths`; no validated input may be represented
only by a directory. The exact executable structures and tuple orders are
defined in the solution.

### Core TASK-0257 outputs

| Provider artifact | Internal SHA-256 | File SHA-256 |
| --- | --- | --- |
| v2 training export | `9dbbffc81828d053154dd2e3adeadfb80869e5e1d7c6a03659df5ebc38fd7b35` | `6fd7680694122ac878154886550de6d0070ec32985ae37e038e894cd86b359aa` |
| source groups | `029cb1b4c56eebe2cf36495618e12966944883e52a706a5bb05feb0e81e674b0` | `d48f990790b57b11389186b407078f52fce3d34a6410a9101001cd5d3221c05e` |
| four-video training manifest | `ddf1af3b23224b7fb99e96cb06226938686062cee5bf28bef153822f23847aba` | `e8261c550bf30da4b36d9b6773717769775e829d3c3c26f7cd750d2d7407ac3b` |
| MViT-V2-S embeddings | `d6aa1d9bbb478972d5880bf944da4b0fa9ef1d7f1aa1640f79e76281c23b201f` | `01dbd913a5fdda3a7b9173cbd2509aa191edf50d1010d00785a0c52a9d3b9a51` |
| Swin3D-T embeddings | `3dc7caacfbc2cb3d8d8df856094131bde4fab3c3615811f363725490b7fea40b` | `1c849096276c2e60f9d01cc064f54aa7426128634e85e7288faf526e76b031b5` |
| frozen nested-probe plan | `de1444c74672a827af2660a825af1d2792a4c8f61fdf47558ddc372261290333` | `78cc7dc64ea0089a431d362485ed85faee2b75863d797492c8a787d1420729c2` |
| nested probe | `92d5d8045b2f39834eac8c57a662343f98cb9ba9e2049e2edbabe704c398ae21` | `7d1c763656f3e940f6ca9db1bdad4e40f1edb0112d756fd549183fb68008d720` |

The nested probe must be recomputed from its externally receipt-bound inputs,
not merely self-verified, and must reproduce its expected internal identity.
Its archived pooled TP/FP/FN/TN is `14/4/3/24`; precision/recall/F1 is
`0.777778/0.823529/0.800000`. Those values are a fixed descriptive comparator
and may not select or reject a Module-A setting.

### Immutable parent and window-provenance inputs

| Provider artifact | Internal SHA-256 | File SHA-256 |
| --- | --- | --- |
| parent v1 export | `1b0bcef2c67f0d0e1fa35b79042584c43d938f05e893e89a4a64266ab9d0e464` | `afb406ef84ae57b8d6cbe565a848b03bcefc2fdd449d8c223374e04d7f7dc3e0` |
| parent source selection | `dfe9798aef33dfa993ced6a33192d30cd3e1d97c51ed3df7837008b732bcd38d` | `a5274bad5c6e3df3fb5e1255dc56004c60a05eaac2c1529da4df89ce149deb1c` |
| parent review plan | `a8566eb3bb172035a24cedf18c33bd8fda4fda7ee04b647cd030816e1956691c` | `1dae313160a728c41d7f4e66fa88921ad2c4943653c29144aa45446d6530d74a` |
| parent raw-frame manifest | `cf645285d1a4e20beed8deb6e01f921cfc749191038484c37941aa47ddddc742` | `ba939d8f4330a078b2e5e0a725fb3c3e55dc0c89181d0acd70bfc4fc381365b8` |
| parent sealed review | `37f85d6078a26dc5862f755de96228e18a315558afba187bdf2f51d4d6189447` | `6c773946a6fcb1f1397e9a674305cf719dec29b151d88e9c4f897caba1fd0eec` |
| parent source manifest | `n/a` | `7d4fc69d2e7aa258e7601f44add84daeb38655265d8a136b7e961411f6a3a76e` |

The six parent children retain these exact file SHA-256 values:

| Child | File SHA-256 |
| --- | --- |
| `hazen_candidate_bundle_v1.json` | `378a08604962a8733898312478d9e3ad762791563d00f7608cd4353cf2dbaaa9` |
| `hazen_shot_validity_labels_v1.json` | `acabe7e07ad8f98350ef492622bea8d7230ce9dfe1b8d278a4919248faad8dda` |
| `randolph_candidate_bundle_v1.json` | `2d7d0d3b6d872cb1b94c43c8c595df1d98c1485b82faa72ba43f3157dafc8cd8` |
| `randolph_shot_validity_labels_v1.json` | `44b1588e583edb1e4e66ab270511245938c19e8197cd96df0a09aec3da8dda09` |
| `vtv_candidate_bundle_v1.json` | `65c0fb60b8338512d50ee3c33ca01b93c3c9d6f4c126a27ca311f8e8709f22f2` |
| `vtv_shot_validity_labels_v1.json` | `5274a0ce6b2aae635b763b138a88487d28da61f9945f5e779526efbb52ba1b98` |

### Harwood evidence chain

| Provider artifact | Internal SHA-256 | File SHA-256 |
| --- | --- | --- |
| source manifest | `n/a` | `8786b64fc02f4d6105acbe097617a733f7276469a60a721b883ac64062d51ded` |
| continuous selection | `5a7eaa1ecefa295c5d97632e05e7aea2187eafac0082ba6ea6341fc6b372a80f` | `35126baa4ae15a888391ff69840c5410c4195f48a0608b25f7ec1a680a576138` |
| causal review plan | `87c9b92c548000527f253082c6190ec82022934d9c3db244be32e50505244911` | `bbf12a7f7de808d467c6057b2e93170765e0098b47fd4465180ac8c9b9575c24` |
| raw-frame manifest | `0757e339ffd5acb97d5b8f42790cfced2de825e44deea20095abd7bedfeae776` | `984365ba89c71d781a1e36566a5c5327006349c9c0c6744d0246449933a5bcca` |
| sealed review | `5f2219a4cdcc7e8d4e40dde060c8ba82b5070eb6db471c1ce2855cea9d3e5cb6` | `ce9ad5d00db77baed53d874cc11ee7deb1ba960e75592ddf2c591a2490319fff` |
| Harwood candidate child | `bundle 3b69b074dbd313129b2a943255f462500bd0f18eb8e38e1cbcc2c56a2e466ea0` | `12e675e15d278854098f7eef2cd5f81fc711e024fb15faaab4ed0d0f79f2c8c5` |
| Harwood label child | `n/a` | `c786841d4ddca430ff010ff882ba5da55f9985ea1b5a33ea0f077ab40284a770` |

Both raw-frame manifests and every referenced JPEG hash must verify even though
model input is decoded from the exact retained source videos, not from the JPEG
copies. The review plans provide the authoritative 64 frame indexes.

### Source-video and checkpoint receipts

| Input | Size bytes | File SHA-256 |
| --- | ---: | --- |
| `hctv_hazen_lyndon_2023.webm` | `1,416,244,469` | `54dedaee0b04a0bab210b6fe730f857edaf193ec5c28a03a84bf005aa64f5bf9` |
| `hctv_randolph_2026.webm` | `2,152,005,777` | `f234d5bfadd0b182c9a15a6d7b206335fcb371e9971cd2bf67313f544dc6a102` |
| `vtv_spartans_trotamundos_2026.webm` | `2,228,583,591` | `e5cdcdfd1e49f5a71765d4cece56a4810a563938fca4d407836ea6c12409fbec` |
| `hctv_harwood_2026.webm` | `1,376,342,882` | `bf7f3135774027aec3c2827cfa282c95ac4f958dd2bccd93e42f370d4b1f8b81` |
| `mvit_v2_s-ae3be167.pth` | verified local file | `ae3be16733081f6d1cd40e4ab980ca23d6df6dc6486d15ada05a5e8ab8c9b975` |
| `swin3d_t-7615ae03.pth` | verified local file | `7615ae035996b65eb38dad437ae533d2dfcd36f9f89d28c0f0fa7bfb8e6b3130` |

MViT is not re-extracted for the candidate. Its checkpoint receipt is frozen
because the baseline evaluator representation contract must remain replayable.

## Sole temporal representation contract

For each of the 45 ordered keys:

1. Locate the matching verified review-plan example by `review_id=event_id` and
   exact source SHA. It must contain exactly 64 ordered integer frame indexes.
2. Define tiles by position only:
   `t0=frame_indexes[0:16]`, `t1=frame_indexes[16:32]`,
   `t2=frame_indexes[32:48]`, and `t3=frame_indexes[48:64]`. The concatenation
   of the four slices must equal the original list byte-for-value. No frame may
   be inserted, removed, reordered, shifted, overlapped, or regenerated.
3. Decode each group from the exact source video and run the existing
   `torchvision/swin3d_t/kinetics400_v1` transform and retained checkpoint with
   batch size one. Each tile yields one finite float32 vector `e0..e3` of
   dimension 768.
4. Produce exactly one selectable/model-input vector of dimension 1,536:

   `concat(mean(e0,e1,e2,e3), mean(e2,e3) - mean(e0,e1))`.

   The delta is later-two-tile mean minus earlier-two-tile mean. It is never
   `e3-e0`. Exact float32 NumPy call semantics and serialization are frozen by
   the plan; no additional manual reduction order is asserted.

The numerical contract is part of the representation, not producer metadata.
It freezes the OpenCV/FFmpeg decoder and indexed-seek procedure, `mps:0` on the
specified Apple M4 hardware/OS build, PyTorch deterministic-mode/thread flags,
the exact uint8-to-transform-to-float32 cast path, the exact NumPy float32
`mean(..., axis=0, dtype=np.float32)` calls, float64 sklearn fitting, SciPy
`1.17.1` and its build-summary receipt, threadpoolctl/backend identity with CPU
numerical threads limited to one, and canonical JSON byte encoding listed in
the solution. Producer and verifier must validate that complete fit environment
immediately before every fit. Any mismatch stops before extraction or fitting.
Byte identity is claimed only on that complete identity and must be proved by
two empty-state computational replays; it is not claimed across a changed
decoder, hardware, device, OS, or numerical-library build. The equality claim
is limited to the frozen computational projection (ordered keys/frame indexes,
tile vectors, 1,536-vectors, probabilities, fitted numerical state, decisions,
counts, metrics, and terminal decision). Attempt chains, signals, resume CAS,
resource observations, elapsed time, publication receipts, and artifact hashes
are audit fields outside that projection and may differ. Every complete
artifact must instead verify against its own external internal/file receipt. No
manual reduction order beyond the frozen NumPy call semantics is claimed.

Raw tile embeddings may be retained only as `derivation_only` evidence so the
verifier can recompute the 1,536 values. They are not variants, selectable
features, evaluator inputs, or downstream training inputs. There is no alternate
tile count, duration, offset, overlap, frame count, backbone, checkpoint,
preprocessing/crop, pooling, delta formula, attention head, prompt, VLM answer,
old embedding concatenation, or retry representation.

The label-hidden feature plan must be atomically sealed before extraction. It
contains keys, source/game receipts, the exact four frame-index arrays,
representation arithmetic, environment/resource policy, and all hard-false
flags. It contains no `event_present`, review outcome, shot sequence, confidence,
notes, release/rim position, PBP, model probability, old probe metric, or prior
AGU/Codex answer. Perturbing any label while holding the verified 45-key set
fixed must produce byte-identical plan content.

## Retrospective nested evaluation contract

Run exactly four outer folds in order Hazen, Randolph, VTV, Harwood. In each
fold, the held game is excluded from all scaler fitting, logistic fitting, and
threshold selection. On the remaining three games:

1. create inner leave-one-game-out probabilities using the sole 1,536-dimension
   representation;
2. fit `StandardScaler` only on each inner training split; use no PCA;
3. fit balanced logistic regression with `C=0.01`,
   `class_weight="balanced"`, `max_iter=5000`, `random_state=0`, and the frozen
   estimator semantics in the solution;
4. pool the complete inner OOF predictions and evaluate thresholds
   `0.00,0.05,...,1.00` using decision `probability >= threshold`;
5. maximize balanced accuracy, then F1, then precision, then recall; any
   remaining tie uses the higher threshold;
6. refit the scaler/model on all three outer-training games and evaluate the
   held game once at the selected threshold.

Outer-held labels may affect only the held confusion counts and metrics. A
held-label perturbation must leave frame selection, feature bytes, inner OOF
probabilities, scaler/model state, selected threshold, outer-held probabilities,
and decisions unchanged. It is not sufficient merely to compare a settings
dictionary.

Metric definitions are exact: precision is zero when `TP+FP=0`; recall is zero
when `TP+FN=0`; specificity is zero when `TN+FP=0`; F1 is zero when precision
plus recall is zero; balanced accuracy is `(recall+specificity)/2`.

## Final evaluator selection and all-45 refit

After the retrospective computation is complete, independently freeze two
files using four-game LOGO OOF over only the same 45 rows. These are offline
serialized evaluators, not runtime model checkpoints.

### Baseline final evaluator

- Candidate variants, in tie-break order, are exactly `swin3d_t` (the frozen
  TASK-0257 768-vector) and `mvit_v2_s+swin3d_t` (MViT 768 followed by Swin
  768, for 1,536 dimensions).
- For each variant, each held-game probability comes from a scaler and balanced
  logistic model fit on the other three games. Pool all four held-game vectors.
- Select one `(variant, threshold)` by balanced accuracy, F1, precision, recall,
  then earlier variant order, then higher threshold.
- Do not infer this OOF vector from TASK-0257's nested probe; that artifact only
  retains each outer fold's selected variant. Refit both baseline variants to
  obtain their complete LOGO OOF predictions.

### Candidate final evaluator

- The only variant is `swin3d-t-tiled-4x2s-mean-delta-v1`.
- Use the identical four-game LOGO fitting, threshold grid, metric order, and
  higher-threshold tie-break. No candidate-variant choice exists.

For each selected evaluator, fit a fresh `StandardScaler` and balanced logistic
model exactly once on all 45 rows. Freeze its representation contract, ordered
45 keys and labels, all LOGO predictions and threshold metrics, selected pair,
scaler mean/variance/scale and sample count, coefficients, intercept, class
order, threshold, estimator parameters and library versions, and every input
internal/file receipt. Baseline and candidate artifacts must be separately
addressed and independently verifiable. Neither may be overwritten or
reselected after any Module-B candidate-universe work begins.

Verification is computational, not structural. A final-evaluator verifier must
receive the complete externally verified 45 ordered rows and complete verified
feature matrices for every allowed representation. It independently reruns
all four LOGO fits, all 21 thresholds, baseline two-variant or candidate
single-variant selection, and a fresh all-45 refit, then compares every frozen
field and canonical byte projection. The producer's “fit exactly once” rule
applies to one production build invocation; it never prevents a verifier from
performing its own fresh fits.

## Mechanical decision and error bounds

The Module-A decision uses the sole candidate retrospective outer-held
predictions. It is `mechanical_pass` only if every receipt, schema, extraction,
determinism, leakage, evaluator, atomic/resume, process-cleanup, disk, and
resource check passes and these held-error bounds all hold:

| Held game | Support | Maximum `FP+FN` | Additional bounds |
| --- | ---: | ---: | --- |
| Hazen | 8 | 0 | — |
| Randolph | 7 | 1 | — |
| VTV | 7 | 2 | — |
| Harwood | 23 | 3 | `FP<=2` and `FN<=1` |

Pooled errors must therefore be at most six. If all mechanical checks pass but
any error bound fails, atomically seal `temporal-hypothesis-rejected` with the
observed counts and stop. If any other mandatory check fails, stop as
artifact-free `preflight_refusal` before attempt admission or terminal
`mechanical_failure` after admission; do not publish a pass receipt, alter a
threshold, switch hardware to chase a result, or try another feature.

A refusal during the complete no-write preflight is `preflight_refusal`, not
`mechanical_failure`: it returns a nonzero exit and a bounded stderr/returned
diagnostic, creates no new file or directory, and does not modify an existing
artifact. Only after no-write preflight, module-exclusive lock acquisition, and
under-lock revalidation all pass may the supervisor open the first attempt
staging file. From that transition onward, a mandatory non-bound failure is a
terminal `mechanical_failure` and is recorded inside one atomic
`terminal_failure_v1` generation, containing only receipts that actually
verified, the immutable attempt chain, and
`final_generation_published=false`; it cannot contain evaluator receipts or
coexist with a trusted `final_v1`. This separates a refusal that produced no
evidence from a failure observed during an admitted attempt.

The extraction state machine is mutually exclusive. Its only nonterminal
durable state is `interrupted_recoverable`, and that state is allowed solely for
an externally delivered `SIGINT` or `SIGTERM` during attempt 1 or 2. Every
receipt, input identity, disk check, deterministic-environment check, and
resource threshold must still pass, and the newly verified prefix must be a
strict extension of the preceding prefix by at least one row while remaining
strictly shorter than all 45 rows. A signal after row 45 follows the completed
private-finalization path and cannot publish a resume. Receipt/schema
drift, a resource-guard breach, any runtime/sample/log/disk cap, decode/model or
determinism error, failed resume CAS, a non-advancing prefix, any other signal,
or exhaustion of attempt 3 is terminal `mechanical_failure`; none is resumable.
Terminal states are exactly `mechanical_failure`,
`temporal-hypothesis-rejected`, and `mechanical_pass`, and no two may coexist.

## Resource, disk, and publication requirements

- Use only `.venv/bin/python` on Python 3.11. Extraction batch size is exactly
  one. The sustained guard is no weaker than TASK-0257: maximum system memory
  `90%`, minimum available memory `2 GiB`, minimum free swap `0.25 GiB`, maximum
  system CPU `95%`, two-second samples, and stop after three consecutive
  breaches with exit code `75`.
- The frozen CPU fit stack is SciPy `1.17.1`, raw `scipy.show_config()` stdout
  SHA-256
  `23f41e614b4eaa40a20a60411c0700c189258aa52c47569b60b59b6683e78438`,
  BLAS/LAPACK `Accelerate`, and threadpoolctl `3.6.0`. All CPU numerical-thread
  environment limits are set to `1` before imports; every producer and verifier
  fit runs inside `threadpool_limits(limits=1)` and verifies immediately before
  and after fitting that both normalized sklearn and PyTorch `libomp` backends
  report exactly one thread. Environment or backend drift is terminal.
- Extraction permits at most three admitted attempts total. Across all attempts
  combined, guarded active runtime is capped at `43,200` seconds, resource
  sampling at exactly the scheduled cumulative points `t=2,4,...,43,200`
  seconds and therefore at `21,600` rows, and all published plus currently
  staged resource-log bytes are capped at `16,777,216`. Attempt gaps do not
  consume guarded active runtime. Every cap is terminal, never resumable.
- Each admitted attempt publishes a new immutable, no-clobber attempt
  placement containing an exact resource JSONL and attempt record. A valid
  recoverable attempt-1/2 signal publishes that generation under `attempts/`
  with a resume; the current completed or terminal attempt remains private until
  it is published inside the mutually exclusive `final_v1` or
  `terminal_failure_v1` generation without a resume. Attempt ordinals 1 through
  3 are never appended, replaced, or deleted and form an immediate-provider
  internal/file receipt chain. A resumed attempt must externally supply and
  verify the complete prior chain plus the latest resume's device/inode and
  internal/file SHA pair. The final embedding and every terminal record bind
  the complete ordered published chain and never fabricate a failed-publication
  receipt.
- At preflight and immediately before every model load, resume write, final-file
  write, and generation publication, compute per-filesystem
  `free_bytes - declared_worst_case_new_bytes`. It must remain at least
  `3,758,096,384` bytes (`3.5 GiB`). Current free space alone is not sufficient,
  and the same filesystem must not be charged or counted twice. Current free
  space already reflects retained attempt generations, whose exact sizes are
  verified for cumulative caps but are not subtracted again. Declared new bytes
  include the maximum remaining attempts, remaining cumulative-log allowance,
  resume/record snapshots, private staging, embedding output, and the larger of
  the mutually exclusive final or terminal-failure generations.
- Any resource-guard breach or cap reaps the entire process group, completes or
  discards the current untrusted log staging without a partial JSONL row,
  removes random temporaries, leaves no unverified final, and seals terminal
  `mechanical_failure` after the admitted preflight. Only a qualifying external
  `SIGINT`/`SIGTERM` may publish a new immutable resume generation. Success
  publishes the completed attempt, verified embedding, retrospective, both
  evaluators, and gate together in one `final_v1`; it retains prior immutable
  signal attempts but leaves no random staging path or worker process.
- Every JSON write uses a same-directory unpredictable temporary, file flush and
  `fsync`, no-replace atomic rename, and parent-directory `fsync`. New stable
  targets must be absent and publication must fail rather than overwrite. The
  completed attempt, embedding, final retrospective, baseline evaluator,
  candidate evaluator, and gate record are staged as one immutable generation
  and become visible together. A handled terminal failure likewise publishes
  the current attempt, optional already-verified embedding, and failure record
  only through one `terminal_failure_v1` directory rename.
- Direct, symlink, hardlink, case-fold, ancestor/descendant, cross-output, and
  input/output aliases fail before large reads or model loading. Stored paths
  are safe relative names; absolute paths, `..`, NUL, unsafe IDs, and
  environment-specific path strings are rejected.
- The legacy guard that opens its log before caller preflight is not invoked
  directly. A Module-A supervisor must finish receipt replay, all path/alias and
  required-absence checks, exact-spec-approval verification, per-volume disk
  accounting, exclusive advisory locking of the already-existing canonical
  output-parent directory, and a complete under-lock revalidation before
  creating an output directory or opening any output, resume, staging, failure,
  or log path for write. Lock acquisition writes no lock artifact. Publication
  revalidates the lock, all input identities/receipts, resume CAS, disk budget,
  and stable-target absence before a no-clobber rename.
- Any checkpoint consumed by `torch.load` is opened once with no-follow
  semantics; size, identity, streamed file hash, and
  `torch.load(weights_only=True)` all use that same verified open file
  descriptor, followed by an `fstat` identity check. The Swin checkpoint follows
  this path; the receipt-only MViT checkpoint is hashed but not model-loaded.
  OpenCV cannot consume the already-verified video descriptor in the frozen
  API, so
  every source video instead has a full pre-decode hash/stat and a full
  post-decode hash/stat before staged outputs become trusted. This detects
  persistent mutation or replacement at those boundaries but does not claim
  protection against an adversarial concurrent mutate-and-restore between
  checks; the allowed environment therefore excludes such a writer, and tests
  must prove that observable mid-run mutations fail and discard staged output.

Public artifact consumers never accept an already parsed raw mapping. The only
public construction boundaries take a regular-file path plus caller-supplied
expected internal and stored-file SHA-256 and return an opaque immutable
`VerifiedTemporalFeaturePlan` or `VerifiedTiledSwinEmbeddings`. Public replay
and evaluation accept those verified values only. A raw mapping, missing
expected hash, self-derived expected hash, stale file identity, or token
constructed outside its verifier fails before model load or fit.

The first atomic producer cannot consume an external embedding receipt before
those bytes exist. It may pass a separate opaque, non-serializable
transaction-local produced-value capability from the extractor to the private
same-transaction final builder after strict plan/source/checkpoint/schema/
finite/formula/staging checks. That value is never a public
`VerifiedTiledSwinEmbeddings` and cannot escape or be reconstructed from a
mapping. After atomic publication, the caller freezes the embedding's returned
internal/file receipts externally; only then does the independent public
verifier construct `VerifiedTiledSwinEmbeddings` from path plus those expected
values and replay the computational result before trusting the terminal gate.

## RED-first test matrix

Implementation starts by proving these tests fail for the missing behavior.
Synthetic fixtures cover failures; one receipt-bound real replay covers the
frozen evidence.

| Area | Required RED cases and acceptance |
| --- | --- |
| Authorization | Map approval alone, a missing fresh-review pass, a missing user approval, any wrong/reordered requirement/solution/gate SHA tuple, a self-authored approval, or any post-approval byte drift blocks before output/model work. A semantic edit clears both review and approval gates. |
| Receipt graph | Wrong/missing v2 export, parent, any of six parent children, source group, manifest, old embedding, old plan/probe, parent/Harwood review-plan, raw-frame, review, source-manifest, video, checkpoint internal/file receipt fails before labels/model/output. The old nested probe is recomputed and must match its expected receipt. Self-resealed tampering still fails the caller receipt. |
| Label-hidden plan | Unknown/missing field, label/review/probability field, non-45 coverage, three excluded-ID drift, key/order drift, plan-receipt drift, non-64 review indexes, or any tile other than exact 16-position slices fails. Changing all labels leaves plan bytes unchanged. Candidate event endpoints are ignored and a regression proves they cannot regenerate indexes. |
| Sole representation | Anything other than 4x16 Swin tiles and the exact float32 768+768 mean/delta result fails. Unit vectors distinguish later-minus-earlier from `e3-e0`; dimension, order, dtype, NaN/Inf, backbone, checkpoint, transform, crop, or formula drift fails. Derivation-only tile embeddings recompute the stored 1,536-vector exactly and cannot enter an evaluator separately. |
| Join/schema | Exact 45 three-part keys and `8/7/7/23`, `17+/28-` counts are required. Missing/extra/duplicate/reordered/mislabeled rows, boolean-as-number fields, unknown keys, duplicate JSON keys, NaN/Infinity constants, unsafe IDs, absolute/escaping paths, oversized snapshots, or source mutation between stat/read/use fails closed. |
| Nested leakage | Exactly four outer and three inner game-held folds. Held-label perturbation cannot change any plan/feature/fit/selection/probability/decision field; only truth-derived counts/metrics may change. A deliberately leaky selector must fail the test. |
| Threshold and metrics | Exact 21 thresholds, `>=` decision, zero-denominator rules, balanced-accuracy formula, ordered metric tie-break, and higher-threshold final tie are tested with adversarial ties and recomputed confusion matrices. No outer-held row may appear in a fit. |
| Final evaluators | Baseline independently produces complete four-game LOGO OOF for both ordered variants; candidate has exactly one. Variant/threshold tie-breaks, MViT-then-Swin concatenation, all-45 scaler/refit, coefficient/intercept dimensions, library/parameter recording, separate addresses, and receipt replay are exact. A verifier supplied complete verified rows/features must freshly rerun every LOGO fit, threshold result, selection and all-45 refit and compare every field; merely checking stored algebra or reusing selected-fold TASK-0257 predictions fails. Producer fit-once instrumentation may not disable verifier refits. SciPy/config/backend/threadpool drift or any pre/post-fit numerical thread count other than one fails. |
| Error gate | Boundary cases `0/1/2/3`, Harwood `FP=2/FN=1`, and pooled six pass; each one-step violation rejects. A mechanically valid bound miss seals only `temporal-hypothesis-rejected`; any other failure cannot emit `mechanical_pass`. |
| Eligibility | Every new schema rejects any eligibility/promoted flag other than false. Runtime model loaders, default-pointer writers, readiness, blind inference, formal evaluators, and generic training routes reject these artifacts. |
| Verified public API | Raw parsed mappings and forged verified wrappers are rejected. Only path plus independently expected internal/file hashes may construct public `VerifiedTemporalFeaturePlan` and `VerifiedTiledSwinEmbeddings`. The new-byte producer uses only a distinct private transaction capability, which cannot serialize or cross the public boundary; after publication, independent replay requires externally frozen receipts. |
| Paths/atomicity | Direct/symlink/hardlink/case-fold/nested/cross-output aliases fail before destructive work. A competing invocation cannot pass the module-exclusive directory lock. Injected write, `fsync`, no-replace rename, publication-time input/resume revalidation, and directory-publication failures preserve prior immutable generations, expose no partial generation, never clobber a target, and clean random temporaries. Checkpoint mutation is caught on the same verified no-follow descriptor; persistent video mutation between pre/post full hashes discards staging, while the documented concurrent mutate-and-restore exclusion is not misrepresented as covered. |
| Resume/determinism | At most three immutable attempt generations are allowed. Each has an exact log/attempt record and optional signal-only resume, binds its predecessor by external internal/file receipts, and is never appended or replaced. Only `SIGINT`/`SIGTERM` on attempt 1 or 2 with `0 < new_rows_verified` and `completed_count < 45` plus clean receipt/resource/disk/determinism revalidation is recoverable; a signal after row 45 follows private finalization, while guard/cap/decode/model/receipt/CAS/non-progress/other-signal/attempt-3 failures are terminal. Resume requires the external device/inode/internal/file CAS tuple, complete chain replay, and the immediate prior prefix. Clean, empty-state replay, and valid signal/resume paths produce an identical frozen computational projection without recomputing a verified resume prefix. Attempt/resource/signal/CAS/publication audit fields and complete artifact bytes are expected to differ by path; each artifact must independently match its own external receipt. |
| Disk/resources | Exact batch size one is required. The supervisor proves authorization/receipt/alias/absence/disk preflight plus lock and under-lock revalidation precede every output-directory creation or output/log open; a preflight refusal emits no artifact and the legacy log-first guard is never called directly. Across all attempts, scheduled resource rows are exactly cumulative `t=2,4,...,43,200` (maximum `21,600`) and complete canonical JSONL rows stay within `16,777,216` bytes with no partial row. Disk accounting includes all retained/max-remaining attempt generations and simultaneous staging before enforcing the `3.5 GiB` reserve. Synthetic guard/cap traces prove terminal exit `75`, atomic attempt/failure publication, a reaped process tree, no resumable guard failure, and no unverified final. |
| Immutability | Snapshot every frozen TASK-0257 file SHA before/after success and every injected failure; all remain identical. Module A creates files only under its new output root. |

## Commands and project structure

Proposed implementation locations are frozen for later planning only. The
map's general Module-A implementation scope remains conditional and map
approval alone does not satisfy the current-spec stop condition. No file below
may be created or changed for Module A until fresh specification re-review
passes and the user approves the exact three-file SHA tuple:

- `app/analysis/vru_causal_temporal_retrospective.py`: schemas, verifiers,
  feature combination, evaluation, evaluator freeze, and decision logic;
- `scripts/seal_vru_causal_temporal_feature_plan.py`: plan CLI;
- `scripts/extract_vru_causal_tiled_swin_embeddings.py`: guarded resumable
  extractor CLI;
- `scripts/screen_vru_causal_temporal_retrospective.py`: diagnostic and atomic
  final-generation CLI;
- `tests/test_vru_causal_temporal_feature_plan.py`,
  `tests/test_vru_causal_tiled_swin_embeddings.py`,
  `tests/test_vru_causal_temporal_retrospective.py`,
  `tests/test_vru_causal_final_evaluator.py`, and
  `tests/test_task0258_module_a_cli.py`.

Canonical verification commands after implementation are:

```zsh
.venv/bin/python -m pytest -q \
  tests/test_vru_causal_temporal_feature_plan.py \
  tests/test_vru_causal_tiled_swin_embeddings.py \
  tests/test_vru_causal_temporal_retrospective.py \
  tests/test_vru_causal_final_evaluator.py \
  tests/test_task0258_module_a_cli.py
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check \
  app/analysis/vru_causal_temporal_retrospective.py \
  scripts/seal_vru_causal_temporal_feature_plan.py \
  scripts/extract_vru_causal_tiled_swin_embeddings.py \
  scripts/screen_vru_causal_temporal_retrospective.py \
  tests/test_vru_causal_temporal_feature_plan.py \
  tests/test_vru_causal_tiled_swin_embeddings.py \
  tests/test_vru_causal_temporal_retrospective.py \
  tests/test_vru_causal_final_evaluator.py \
  tests/test_task0258_module_a_cli.py
.venv/bin/python -m ruff format --check \
  app/analysis/vru_causal_temporal_retrospective.py \
  scripts/seal_vru_causal_temporal_feature_plan.py \
  scripts/extract_vru_causal_tiled_swin_embeddings.py \
  scripts/screen_vru_causal_temporal_retrospective.py \
  tests/test_vru_causal_temporal_feature_plan.py \
  tests/test_vru_causal_tiled_swin_embeddings.py \
  tests/test_vru_causal_temporal_retrospective.py \
  tests/test_vru_causal_final_evaluator.py \
  tests/test_task0258_module_a_cli.py
git diff --check -- \
  docs/specs/TASK-0258-temporal-canary \
  app/analysis/vru_causal_temporal_retrospective.py \
  scripts/seal_vru_causal_temporal_feature_plan.py \
  scripts/extract_vru_causal_tiled_swin_embeddings.py \
  scripts/screen_vru_causal_temporal_retrospective.py \
  tests/test_vru_causal_temporal_feature_plan.py \
  tests/test_vru_causal_tiled_swin_embeddings.py \
  tests/test_vru_causal_temporal_retrospective.py \
  tests/test_vru_causal_final_evaluator.py \
  tests/test_task0258_module_a_cli.py
```

Concrete materialization commands and observed receipts belong in a later
approved implementation record, not in this pre-implementation specification.

## Boundaries

### Always

- Verify the entire frozen receipt graph, all review-frame hashes, all four
  source videos, both checkpoint contracts, and the exact 45-row join.
- Seal the label-hidden plan before extraction; use only the 64 review-plan
  indexes sliced into four consecutive groups of 16.
- Keep fit/selection state inside the proper training games and recompute every
  stored count, probability decision, metric, coefficient shape, and receipt.
- Enforce the `3.5 GiB` post-worst-case reserve and exact batch-one sustained
  guard at every required boundary.
- Stop and preserve negative evidence when a frozen check or bound fails.

### Ask first

- Any dependency addition, estimator/backbone/checkpoint/preprocessing change,
  output-schema change, or change to a frozen receipt or threshold rule.
- Any runtime/API/config/default-pointer/readiness/blind-inference integration.
- Any Module-B specification or work beyond citing it as conditional downstream.

### Never

- Modify, overwrite, migrate, copy as a trusted replacement, or reinterpret a
  TASK-0257/TASK-0254 artifact.
- Read an outer-held label for sampling, feature construction, fitting, or
  threshold selection; convert `uncertain` to a binary target.
- Try another tiling, model, feature, threshold grid, crop, prompt, or source
  after seeing errors or metrics.
- Claim independence, production-held validity, an 85% result, readiness, or
  promotion; write a runtime checkpoint or resume blind inference.
- Inspect, enumerate, download, label, or otherwise begin Module-B media work.

## Stop conditions and success criteria

Stop immediately without advancing the dependency graph when the exact
three-file specification tuple lacks the required fresh-review result or user
approval, or on any receipt, path, schema, row, sampling, finite-value,
label-invariance, fold, metric, refit, atomicity, resume, resource, disk, or
process-cleanup failure. A no-write admission failure is `preflight_refusal`;
an admitted-attempt non-bound failure is terminal `mechanical_failure`. Stop as
`temporal-hypothesis-rejected` on an error-bound-only miss. Stopping is valid
evidence and cannot be repaired within TASK-0258 by a retry variant.

The approved capability map names Module-A implementation as conditional scope,
but its global stop condition means map approval alone currently authorizes only
specification drafting. Implementation may begin only after the current
specification passes fresh-context re-review with no unresolved Critical or
Required issue and the user separately approves its exact three-file SHA tuple.
The implementation is complete only when:

1. the exact tiled plan and 45 finite 1,536-vectors replay deterministically;
2. all four nested folds and both independently selected/refit final evaluators
   recompute from externally supplied receipts;
3. a correct immutable gate record reports either `mechanical_pass` or the
   exact mandatory stop decision, with every eligibility field false;
4. old files remain byte-identical, publication/resume/process/resource/disk
   checks pass, and focused/full/lint/format/diff checks are recorded; and
5. a fresh-context adversarial implementation review reports no Critical or
   Required issue, or every such issue is fixed and the review is rerun.

Even a `mechanical_pass` only permits a later user-reviewed specification of
Module B. It does not authorize Module B implementation or any independent
media access.

## Open questions

None inside Module A. Any proposed semantic change to authorization, receipts,
sampling, representation, evaluation, error bounds, resources, schemas, or
output boundaries reopens fresh specification review and exact-SHA user
approval instead of becoming an implementation-time choice. Any byte-only edit
still invalidates the exact approval receipt and requires user reapproval.
Whether Module B is later specified remains a separate conditional user
decision.
