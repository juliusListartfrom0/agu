# TASK-0258 Module A Solution — tiled Swin retrospective and frozen evaluators

Status: **DESIGN DRAFT AUTHORIZED; FRESH-CONTEXT SPECIFICATION RE-REVIEW AND
USER APPROVAL OF THE EXACT THREE-FILE SHA TUPLE ARE REQUIRED BEFORE
IMPLEMENTATION.** The approved capability map's opening scope statement includes
Module-A implementation only conditionally; its build order and global stop
condition still require approval of the current module specification. Map
approval alone therefore authorizes drafting this design, not starting its
implementation. This solution covers only the specification boundary for
`existing-45-temporal-retrospective`; no implementation has started.

## Decision

Add a new offline Module-A implementation beside TASK-0257. Do not widen
`agu.shot-validity-video-embeddings.v1`: that schema requires one native
backbone vector of dimension 768 and carries labels. Do not use
`agu.shot-validity-temporal-model.v1` or `TemporalShotValidityModel`: those are
runtime-facing contracts and would violate the permanent hard-false boundary.

The new module will:

1. replay the complete TASK-0257 receipt graph and its nested-probe result;
2. seal one label-hidden 45-row tiled-feature plan from the original review-plan
   frame indexes;
3. extract exactly the sole 1,536-dimensional Swin mean-plus-delta feature;
4. run the frozen nested outer-game-held retrospective;
5. independently run four-game LOGO selection and one all-45 refit for the
   baseline and candidate final evaluators; and
6. atomically publish the completed attempt, embeddings, diagnostic, both
   evaluators, and the mechanical gate as one immutable generation.

This is additive and rollback-safe. Deleting the new output directory removes
Module A without changing any prior file. An existing byte-identical generation
is a verified no-op; an existing drifted or partial generation fails closed and
is never repaired in place.

## Output layout and schema names

The proposed real-output root is:

```text
analysis_outputs/public_research/vru_causal_temporal_retrospective_v1/
├── temporal_feature_plan.json
├── attempts/                                  # recoverable signals only
│   ├── attempt-0001/                       # immutable atomic generation
│   │   ├── resource_guard.jsonl
│   │   ├── attempt_record.json
│   │   └── resume.json                       # qualifying signal only
│   └── attempt-0002/                       # optional, same exact layout
├── terminal_failure_v1/                       # mutually exclusive with final_v1
│   ├── terminal_attempt/
│   │   ├── resource_guard.jsonl
│   │   └── attempt_record.json               # no resume
│   ├── tiled_swin_embeddings.json          # only if already verified
│   └── mechanical_failure.json
└── final_v1/                                 # one immutable atomic generation
    ├── terminal_attempt/
    │   ├── resource_guard.jsonl
    │   └── attempt_record.json                   # completed, no resume
    ├── tiled_swin_embeddings.json
    ├── temporal_retrospective.json
    ├── baseline_final_evaluator.json
    ├── candidate_final_evaluator.json
    └── mechanical_gate.json
```

Only qualifying attempt-1/2 signal generations may appear under root
`attempts/`; attempt 3 can never appear there. The current completed or terminal
attempt is kept private until it is published inside exactly one terminal
generation. `terminal_failure_v1` contains `tiled_swin_embeddings.json` only if
that artifact verified before the later failure. `terminal_failure_v1` and
`final_v1` are mutually exclusive absent-target, immutable, no-clobber
publications. Thus neither a successful attempt nor embeddings are exposed as a
standalone durable nonterminal state. Random same-directory temporary names are
not stable interfaces and must be absent after either successful publication or
handled failure. A `preflight_refusal` creates none of these paths. The
following new schemas are exact one-version contracts:

```python
TEMPORAL_FEATURE_PLAN_SCHEMA = "agu.vru-causal-temporal-feature-plan.v1"
TILED_SWIN_EMBEDDING_SCHEMA = "agu.vru-causal-tiled-swin-embeddings.v1"
TILED_SWIN_RESUME_SCHEMA = "agu.vru-causal-tiled-swin-embeddings-resume.v1"
EXTRACTION_ATTEMPT_SCHEMA = "agu.vru-causal-tiled-swin-attempt.v1"
RESOURCE_SAMPLE_SCHEMA = "agu.vru-causal-resource-sample.v1"
TEMPORAL_RETROSPECTIVE_SCHEMA = "agu.vru-causal-temporal-retrospective.v1"
FINAL_EVALUATOR_SCHEMA = "agu.vru-causal-final-evaluator.v1"
MODULE_A_GATE_SCHEMA = "agu.vru-causal-temporal-mechanical-gate.v1"
MODULE_A_SPEC_APPROVAL_SCHEMA = "agu.module-a-spec-approval.v1"

MODULE_ID = "existing-45-temporal-retrospective"
REPRESENTATION = "swin3d-t-tiled-4x2s-mean-delta-v1"
RESERVE_BYTES = 3_758_096_384
MAX_JSON_SNAPSHOT_BYTES = 67_108_864
MAX_CHECKPOINT_SNAPSHOT_BYTES = 209_715_200
RESOURCE_SAMPLE_INTERVAL_SECONDS = 2.0
MAX_GUARDED_RUNTIME_SECONDS = 43_200
MAX_RESOURCE_SAMPLES = 21_600
MAX_RESOURCE_LOG_BYTES = 16_777_216
MAX_EXTRACTION_ATTEMPTS = 3
```

`MAX_JSON_SNAPSHOT_BYTES` is also the hard encoded-byte ceiling for every new
JSON plan, attempt record, resume, embedding, retrospective, evaluator, gate,
and failure member. Disk preflight charges one full ceiling for each such member
that can still exist simultaneously under the three-attempt state machine; it
does not estimate from current content length. JSONL uses its separate global
cap. A writer that reaches its member ceiling fails before a partial stable
publication.

Every schema has an exact field allowlist. Unknown and missing fields fail. The
feature-plan, embedding, resume, attempt, retrospective, final-evaluator, and
mechanical-gate schemas share common fields `schema_version`, `module_id`,
`purpose`, `runtime_consumable`, `training_consumable`,
`formal_evaluation_eligible`, `promotion_eligible`, and `promoted`; all five
consumption/eligibility/promotion fields are literally false and purpose is
`development_diagnostic_only`. The external specification-approval record and
resource JSONL row instead use only their separately listed exact allowlists.

Stored provider receipts use only these two shapes:

```text
StoredArtifactReceipt = {
  schema_version, internal_sha256_field, internal_sha256,
  file_sha256, filename, size_bytes
}
FileReceipt = {file_sha256, filename, size_bytes}
```

`internal_sha256_field` preserves the provider's actual contract name, such as
`artifact_sha256`, `manifest_sha256`, or `bundle_sha256`; verifiers may not
silently normalize one provider field into another. Receipts never store
absolute paths. A caller maps safe filenames to input paths and supplies the
expected hashes independently.

### Exact Module-A specification approval receipt

The map's conditional implementation scope is not approval of the current
module specification and is not this receipt. After a fresh-context
specification review has no unresolved Critical or Required finding, the user
must approve the ordered stored-file SHA-256 tuple for exactly these three
files. The approval record has the exact allowlist:

```text
ModuleASpecApproval = {
  schema_version = agu.module-a-spec-approval.v1,
  module_id = existing-45-temporal-retrospective,
  approved_files = [
    {filename = requirement.md, file_sha256, size_bytes},
    {filename = solution.md, file_sha256, size_bytes},
    {filename = gate-review.md, file_sha256, size_bytes},
  ],
  fresh_review_receipt = {internal_sha256, file_sha256},
  approval_scope = module_a_implementation_only,
  approval_statement_sha256,
  approved_at_utc,
  artifact_sha256,
}
```

The order above is literal. `approval_statement_sha256` binds the exact human
approval statement that names the same three hashes. The implementation caller
must supply the approval record's expected internal/file SHA pair from the
human-controlled review registry; copying values from the record itself does
not establish approval. Recording approval never edits these three spec files.
Any byte drift invalidates their tuple and needs a new user approval; any
semantic change additionally requires a new fresh-context specification review
before the new approval can be issued.

### `agu.vru-causal-temporal-feature-plan.v1`

Top-level fields are exactly the common fields plus:

```text
label_hidden
representation
task0257_receipts
ordered_examples
evaluation_protocol
resource_policy
environment_contract
artifact_sha256
```

`label_hidden` is true. `task0257_receipts` has exact named entries for the v2
export, parent export and six children, parent evidence, Harwood evidence,
source groups, training manifest, two old embedding artifacts, old probe plan,
old nested probe, four source videos, and two checkpoint contracts. Arrays use
their frozen provider order and cannot be reordered.

`representation` freezes:

```text
name = swin3d-t-tiled-4x2s-mean-delta-v1
backbone = torchvision/swin3d_t/kinetics400_v1
checkpoint_sha256 = 7615ae035996b65eb38dad437ae533d2dfcd36f9f89d28c0f0fa7bfb8e6b3130
window_seconds = 8.0
tile_count = 4
tile_seconds = 2.0
clip_frames_per_tile = 16
review_plan_frames = 64
tile_position_slices = [[0,16],[16,32],[32,48],[48,64]]
tile_embedding_dimension = 768
model_input_dimension = 1536
model_input_dtype = float32
frame_source = verified_review_plan_frame_indexes
preprocessing = torchvision/Swin3D_T_Weights.KINETICS400_V1.transforms
aggregation = concat(mean(e0,e1,e2,e3),mean(e2,e3)-mean(e0,e1))
```

Each `ordered_examples` row has exactly `ordinal`, `game_id`,
`production_family`, `source_video_sha256`, `candidate_bundle_sha256`,
`event_id`, `review_plan_receipt`, and `tile_frame_indexes`. The latter is four
ordered arrays of 16 integers. There is no target, review answer, anchor, model
score, metric, or free-form text.

`evaluation_protocol` freezes game order/counts, outer/inner LOGO, scaler,
logistic constructor, threshold grid, metric definitions, and tie-breaks.
`resource_policy` freezes batch one, the sustained thresholds, process cleanup,
worst-case-byte accounting, and `RESERVE_BYTES`. `environment_contract` records
the canonical `.venv` and the exact frozen identity: Python `3.11.15`, macOS
`26.5.2` build `25F84`, `arm64`, Mac mini `Mac16,10`, Apple M4 with 10 CPU and
10 GPU cores and 16 GiB unified memory, device `mps:0`, OpenCV `4.13.0` with
explicit `cv2.CAP_FFMPEG` backend and `cv2.getBuildInformation()` SHA-256
`6fa24b2683931738b55c6b8a2d2fe42ff5663b9d8b4eca4fc53fb1803a55e46e`,
NumPy `2.4.6` with `numpy.show_config()` text SHA-256
`470ee8aedb85b9fa43dcd664ea25e13f2fad1bf6c822d5b9dddebcff81d2a3a7`,
scikit-learn `1.9.0`, PyTorch `2.13.0` with `torch.__config__.show()` SHA-256
`1943041ad11240ec18f123ae0f5aa98ed16a9ebf53b479bb4f0c11ea4c6f12f1`,
torchvision `0.28.0`, SciPy `1.17.1` with raw `scipy.show_config()` stdout
SHA-256
`23f41e614b4eaa40a20a60411c0700c189258aa52c47569b60b59b6683e78438`,
and threadpoolctl `3.6.0`. The SciPy summary must identify both BLAS and LAPACK
as `Accelerate`. With all frozen modules imported and
`threadpool_limits(limits=1)` active, the normalized threadpool backend list is
exactly two entries: sklearn `.dylibs/libomp.dylib` and PyTorch
`lib/libomp.dylib`; each has `user_api=openmp`, `internal_api=openmp`,
`prefix=libomp`, `version=null`, and `num_threads=1`. Absolute backend paths are
not serialized. Serial numbers, UUIDs, display identity, and other personal
machine identifiers are forbidden. A changed frozen field must produce a new
reviewed plan, not silently reuse this one.

### `agu.vru-causal-tiled-swin-embeddings.v1`

Top-level fields are exactly the common fields plus `plan_receipt`,
`task0257_input_receipts`, `representation`, `producer_environment`,
`attempt_chain`, `row_count`, `examples`, and `artifact_sha256`. It deliberately
has no label field.

Each example contains exactly:

```text
ordinal
key = {source_video_sha256, candidate_bundle_sha256, event_id}
tile_frame_indexes
derivation_only_tile_embeddings = [e0, e1, e2, e3]
model_input
```

Every `e` is 768 finite float32 values; `model_input` is 1,536 finite float32
values. The verifier recomputes `model_input` from the derivation-only values
and rejects any mismatch. The four tile vectors exist only to prove the frozen
formula: no public evaluator getter exposes them and no variant may select or
concatenate them. Only `model_input` enters the retrospective or candidate
final evaluator.

The embedding top level also carries `attempt_chain`, an ordered array of one
to three immutable attempt-generation receipts. Every entry binds the attempt
record by internal/file SHA, the resource log by file SHA, and the optional
resume by internal/file SHA plus publication device/inode. The chain starts at
attempt 1, has no gap, and each attempt record binds its immediate predecessor.

### Attempt, resource-log, and resume contracts

Each resource-log line is one canonical JSON object plus one LF and has exactly:

```text
{
  schema_version = agu.vru-causal-resource-sample.v1,
  attempt_ordinal,
  attempt_sample_ordinal,
  cumulative_sample_ordinal,
  scheduled_cumulative_active_seconds,
  observed_attempt_active_nanoseconds,
  system_memory_percent,
  available_memory_bytes,
  free_swap_bytes,
  system_cpu_percent,
  consecutive_breach_count,
  breached_limits,
}
```

Ordinals are positive integers; byte counts and nanoseconds are nonnegative
integers; percentages are finite JSON numbers; `breached_limits` is an ordered
subset of `memory_percent`, `available_memory`, `free_swap`, and `system_cpu`.
`scheduled_cumulative_active_seconds` is exactly
`2 * cumulative_sample_ordinal`, so the complete allowed schedule is precisely
`2,4,...,43_200` and contains at most `21_600` rows across all attempts. The
supervisor encodes a complete row and checks the cumulative
`16_777_216`-byte allowance before any write. A row that would exceed a cap is
not partially written; the cap is a terminal failure. A published JSONL ends
after an LF and has no truncated or extra row.

`agu.vru-causal-tiled-swin-attempt.v1` uses the common fields plus exactly
`attempt_ordinal`, `prior_attempt_receipt`, `plan_receipt`,
`task0257_input_receipts`, `started_prefix_count`, `completed_prefix_count`,
`new_rows_verified`, `cumulative_active_runtime_nanoseconds`,
`cumulative_resource_samples`, `cumulative_resource_log_bytes`,
`resource_log_receipt`, `resume_input_cas`, `resume_output_cas`,
`received_signal`, `disposition`, `stop_reason`, and `artifact_sha256`.
`disposition` is exactly `completed`, `interrupted_recoverable`, or
`terminal_failure`. `prior_attempt_receipt` is null only for attempt 1;
otherwise it is the immediately preceding attempt record's externally supplied
internal/file receipt. Cumulative totals are monotone and include the current
attempt. `new_rows_verified` equals completed minus started prefix.
`received_signal` is null, `SIGINT`, `SIGTERM`, `SIGHUP`, or `SIGQUIT`; only
`SIGINT`/`SIGTERM` can accompany a recoverable disposition, while handled
`SIGHUP`/`SIGQUIT` are terminal `unsupported_signal`. An uncatchable crash or
`SIGKILL` can leave only untrusted staging and can never produce a trusted
resume. `resume_input_cas` is null only on attempt 1 and
`resume_output_cas` is non-null only for `interrupted_recoverable` on attempt 1
or 2. `stop_reason` is null for `completed`, `external_sigint` or
`external_sigterm` for a recoverable disposition, or exactly one terminal code
from `receipt_failure`, `schema_failure`, `input_identity_failure`,
`resume_cas_failure`, `disk_failure`, `resource_breach`, `runtime_cap`,
`sample_cap`, `log_byte_cap`, `decode_failure`, `model_failure`,
`determinism_failure`, `non_advancing_prefix`, `unsupported_signal`,
`attempt_limit`, or `publication_failure`.

The resume schema is
`agu.vru-causal-tiled-swin-embeddings-resume.v1`. Its exact top-level allowlist
is the common fields plus `plan_receipt`, `task0257_input_receipts`,
`representation`, `producer_environment`, `checkpoint_receipt`,
`source_video_receipts`, `attempt_chain_receipts`, `row_count`,
`completed_count`, `completed_examples`, and `artifact_sha256`. The examples
are the exact plan prefix and independently recompute as finite vectors.
`attempt_chain_receipts` contains exactly the already published predecessor
attempts and is empty for attempt 1; it never names its own current attempt
record. The current attempt record binds the new resume after the resume bytes
and publication identity are known, avoiding a circular receipt.

`ResumeCAS` is external metadata with exactly `device`, `inode`, `size_bytes`,
`internal_sha256`, and `file_sha256`. It is never learned from the resume being
checked. A qualifying attempt 1 or 2 signal may publish a resume only when its
completed prefix strictly extends the preceding prefix by at least one row and
remains strictly less than `row_count=45`,
all current inputs and numerical/resource/disk contracts still verify, and the
resume is atomically re-read and verified. The attempt record stores the
published resume CAS. A next attempt must receive the complete prior attempt
chain and that exact CAS from an external registry/caller; it opens the resume
no-follow, matches device/inode/size, file hash, strict schema/internal hash and
prefix, then repeats the CAS immediately before model load. Attempt 3 cannot
publish a recoverable resume.

### `agu.vru-causal-temporal-retrospective.v1`

Top-level fields are exactly the common fields plus `input_receipts`,
`selection_protocol`, `source_selection_limitation`, `outer_folds`,
`observed_metrics`, `archived_task0257_comparator`, `error_bound_result`,
`temporal_hypothesis_decision`, and `artifact_sha256`.

Each outer fold stores its held game/source/family, exact fit and inner-game
lists, inner per-game OOF rows, every threshold candidate and recomputed metric,
the selected threshold, fitted outer-training scaler/model receipt, held
probabilities/decisions/truth, and held metrics. `observed_metrics` contains
pooled, per-game, and descriptive per-production-family counts/metrics. The
archived comparator stores only the externally bound TASK-0257 receipt and its
verbatim reported counts/metrics; it is absent from candidate selection.

`temporal_hypothesis_decision` is `within_frozen_error_bounds` or
`temporal-hypothesis-rejected`. It is not the overall mechanical decision;
receipt/resource/atomic checks belong to the gate record.

### `agu.vru-causal-final-evaluator.v1`

Both evaluator files have the same exact top-level keys: common fields plus
`evaluator_role`, `input_receipts`, `representation_candidates`,
`ordered_training_rows`, `logo_selection`, `selected_evaluator`,
`all_45_refit`, `conditional_downstream`, and `artifact_sha256`.

- `evaluator_role` is `baseline` or `candidate`.
- Baseline `representation_candidates` is exactly
  `[swin3d_t, mvit_v2_s+swin3d_t]`; candidate has exactly the tiled name.
- `ordered_training_rows` stores the 45 exact keys and boolean targets only for
  receipt replay; it is not an exported training dataset.
- `logo_selection` stores four held-game fits and predictions for every allowed
  representation, all 21 pooled threshold results, metric order, original
  representation positions, and the selected pair.
- `selected_evaluator` stores the exact representation and threshold.
- `all_45_refit` stores the one all-row scaler/model state: feature dimension,
  sample count, scaler `mean_`, `var_`, `scale_`, `n_samples_seen_`, classifier
  class order, effective balanced class weights, coefficient vector, intercept,
  iteration count, and the full observed estimator parameter map and
  numerical-library versions.
- `conditional_downstream` says only that the file may be referenced by the
  separately specified Module B after an exact Module-A `mechanical_pass`
  receipt. It grants no runtime, training, formal, promotion, or media access.

The logistic estimator is instantiated exactly as TASK-0257 did:

```python
LogisticRegression(
    C=0.01,
    class_weight="balanced",
    max_iter=5000,
    random_state=0,
)
```

No newly explicit estimator option may change those semantics. The complete
`get_params(deep=False)` result is sealed so a changed scikit-learn default
cannot masquerade as a replay. `StandardScaler()` likewise uses its TASK-0257
defaults and records the observed parameter map.

### `agu.vru-causal-temporal-mechanical-gate.v1`

Top-level fields are exactly the common fields plus `input_receipts`,
`attempt_chain`, `ordered_check_results`, `error_bound_result`,
`resource_summary`, `publication_summary`, `decision`, `stop_reason`,
`conditional_downstream`, and `artifact_sha256`.

`decision` is exactly one of `mechanical_pass`,
`temporal-hypothesis-rejected`, or `mechanical_failure`. A pass requires every
ordered check true and every frozen error bound satisfied. A hypothesis
rejection requires every non-bound mechanical check true and at least one exact
bound false. `mechanical_failure` cannot embed or bless an unverified provider
receipt. The file names Module B only as conditional downstream and states that
its own spec and user review are still required.

`input_receipts` has the same frozen ordered provider slots in every decision.
Each slot is exactly `{provider, verification_state, receipt}` where state is
`verified`, `failed`, or `not_reached`; `receipt` is present only for `verified`
and is otherwise null. This keeps the schema exact without copying an
unverified expected value into a failure record.

`mechanical_pass` and `temporal-hypothesis-rejected` records live only inside a
complete `final_v1` generation with both independently verified evaluators.
When a non-bound failure occurs after an attempt has been admitted but before
such a generation can be trusted, the same schema is written atomically as
`terminal_failure_v1/mechanical_failure.json` with the complete published
immutable attempt chain, including the terminal attempt record when that record
can be verified,
`publication_summary.final_generation_published=false`, only the provider
receipts that actually verified, and no evaluator receipt. It cannot coexist
with `final_v1`; its parent directory is one atomic terminal generation, not a
final-evaluator generation. Receipt, schema, input, resume-CAS,
disk/resource/cumulative-cap, decode/model, determinism, non-advancing-prefix,
unsupported-signal, or third-attempt interruption errors are all terminal
failures.

`preflight_refusal` is deliberately not a gate-record decision or schema. It is
the bounded diagnostic returned when exact-spec authorization, receipts,
paths/aliases/required absence, disk budget, the non-writing module lock, or
under-lock revalidation fails before an attempt staging path is opened. It
creates no new file/directory and changes no existing artifact. Thus the only
durable nonterminal state after an attempt is admitted is an immutable
attempt-1/2 signal generation with `disposition=interrupted_recoverable`.
The pre-registered plan is an immutable prerequisite, not extraction progress.
The two terminal directories are mutually exclusive, and the three terminal
decisions are exactly `mechanical_failure`, `temporal-hypothesis-rejected`, and
`mechanical_pass`.

## Exact sampling and feature derivation

The review-plan 64-index list is the already-frozen uniform eight-second
sampling grid. It is the only defensible source for four two-second tiles:

```python
groups = (
    frame_indexes[0:16],
    frame_indexes[16:32],
    frame_indexes[32:48],
    frame_indexes[48:64],
)
```

The candidate-bundle event bounds are forbidden for derivation. Their end is
the last sampled frame, not the original half-open right boundary; running a
new endpoint-inclusive linspace would silently change the sampling variable.

For each group, decode the exact source-video frames, convert BGR to RGB, form
the same TCHW uint8 clip consumed by the existing torchvision transform,
construct the same Swin3D-T architecture, load the verified official state dict
from the already verified checkpoint descriptor, remove the classification
head as TASK-0257 did, and run inference with batch size one. The existing
path-only `load_video_backbone` helper may not be used if it would reopen the
checkpoint; equivalent architecture/state-dict logic must remain in the
reviewed public Module-A primitive. No retained review JPEG is used as model
input; those JPEGs remain provenance evidence and must still hash correctly.

Arithmetic is explicit and float32:

```python
tiles = np.asarray([e0, e1, e2, e3], dtype=np.float32)
overall = np.mean(tiles, axis=0, dtype=np.float32)
early = np.mean(tiles[0:2], axis=0, dtype=np.float32)
late = np.mean(tiles[2:4], axis=0, dtype=np.float32)
model_input = np.concatenate((overall, late - early)).astype(
    np.float32,
    copy=False,
)
```

Decoder and numerical behavior are exact:

- open each verified WebM with `cv2.VideoCapture(path, cv2.CAP_FFMPEG)`, require
  reported backend `FFMPEG`, seek independently with
  `CAP_PROP_POS_FRAMES=int(frame_index)`, call `read()` once, require a
  three-channel C-contiguous `uint8` BGR frame of the source's verified
  dimensions, then use `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`; no sequential
  decode fallback, alternate backend, resize, or retry is allowed;
- require `PYTHONHASHSEED=0` in the environment before interpreter startup; then
  before model construction set Python/NumPy/PyTorch seed `0`,
  `torch.use_deterministic_algorithms(True)`,
  `torch.set_num_threads(1)`, and `torch.set_num_interop_threads(1)`; require
  deterministic algorithms to report enabled and use exactly `mps:0`, with no
  autocast, TF32, reduced-precision accumulation, CPU, or device fallback;
- stack RGB frames as C-contiguous `uint8` TCHW, apply exactly
  `Swin3D_T_Weights.KINETICS400_V1.transforms()`, run inference at batch one,
  move each result once to CPU, cast to contiguous IEEE-754 binary32, and apply
  exactly the three shown `np.mean(..., axis=0, dtype=np.float32)` calls under
  the frozen NumPy build; no manual summation order is asserted. Subtraction and
  concatenation remain float32 and each stored feature is obtained through
  `np.float32(...).item()`;
- create every sklearn input matrix by converting the verified stored
  binary32 values to C-contiguous `np.float64`; scaler statistics, transformed
  matrices, logistic fitting, probabilities, coefficients, intercepts, and
  metric arithmetic are float64 under the frozen NumPy/SciPy/sklearn stack.
  Before importing those numerical modules, require
  `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=BLIS_NUM_THREADS=VECLIB_MAXIMUM_THREADS=NUMEXPR_NUM_THREADS=1`.
  Wrap every producer and verifier fit in
  `threadpoolctl.threadpool_limits(limits=1)` and immediately before and after
  each fit require the frozen two-entry normalized `libomp` backend list with
  `num_threads=1`; and
- canonical JSON is UTF-8 from
  `json.dumps(ensure_ascii=False, allow_nan=False, sort_keys=True,
  separators=(",", ":"))`. Internal SHA-256 hashes those bytes with the
  artifact's own SHA field absent. Stored-file bytes are the same encoding
  after inserting that SHA field plus exactly one trailing LF, with no BOM or
  indentation. Resume, final, and verifier all use this one encoder.

The real producer plus independent verifier replay from empty state under this
exact decoder/device/hardware/numerical identity must match the frozen
computational projection. That projection is exactly:

```text
embedding = representation + ordered(
  ordinal, key, tile_frame_indexes,
  derivation_only_tile_embeddings, model_input
)
retrospective = selection_protocol + source_selection_limitation + outer_folds
  + observed_metrics + archived_task0257_comparator values
  + error_bound_result + temporal_hypothesis_decision
evaluators = evaluator_role + representation_candidates + ordered_training_rows
  + logo_selection + selected_evaluator + all_45_refit + conditional_downstream
gate = ordered non-audit check outcomes + error_bound_result + decision
```

Attempt-chain receipts, resource rows/summaries, received signals, resume CAS,
elapsed times, publication summaries, file/internal hashes, and path-specific
receipts are audit fields excluded from that equality projection. A clean run
and a signal/resume run are expected to have different complete artifact bytes
and chain cardinality; each complete artifact must independently verify against
its own caller-frozen internal/file receipt. Nondeterminism in the computational
projection or an unsupported deterministic operation is a stop condition, not
permission to disable the flag, average runs, switch hardware, or weaken the
projection. No cross-environment equality claim is made.

## Interfaces

The implementation should keep logic in
`app/analysis/vru_causal_temporal_retrospective.py`; CLIs only parse paths and
externally expected receipts, call the public boundary, and publish verified
bytes. Suggested type-friendly interfaces are:

```python
@dataclass(frozen=True)
class Task0257InputPaths:
    parent_selection: Path
    parent_source_manifest: Path
    parent_review_plan: Path
    parent_raw_frame_manifest: Path
    parent_review_jpegs: tuple[Path, ...]
    parent_sealed_review: Path
    parent_v1_export: Path
    parent_candidate_children: tuple[Path, Path, Path]
    parent_label_children: tuple[Path, Path, Path]
    v2_export: Path
    v2_candidate_child: Path
    v2_label_child: Path
    source_groups: Path
    four_video_training_manifest: Path
    old_embedding_files: tuple[Path, Path]
    old_nested_probe_plan: Path
    old_nested_probe: Path
    harwood_source_manifest: Path
    harwood_selection: Path
    harwood_review_plan: Path
    harwood_raw_frame_manifest: Path
    harwood_review_jpegs: tuple[Path, ...]
    harwood_sealed_review: Path
    source_videos: tuple[Path, Path, Path, Path]
    checkpoints: tuple[Path, Path]


@dataclass(frozen=True)
class Task0257ExpectedReceipts:
    parent_selection: StoredArtifactReceipt
    parent_source_manifest: FileReceipt
    parent_review_plan: StoredArtifactReceipt
    parent_raw_frame_manifest: StoredArtifactReceipt
    parent_review_jpegs: tuple[FileReceipt, ...]
    parent_sealed_review: StoredArtifactReceipt
    parent_v1_export: StoredArtifactReceipt
    parent_candidate_children: tuple[
        StoredArtifactReceipt, StoredArtifactReceipt, StoredArtifactReceipt
    ]
    parent_label_children: tuple[FileReceipt, FileReceipt, FileReceipt]
    v2_export: StoredArtifactReceipt
    v2_candidate_child: StoredArtifactReceipt
    v2_label_child: FileReceipt
    source_groups: StoredArtifactReceipt
    four_video_training_manifest: StoredArtifactReceipt
    old_embedding_files: tuple[StoredArtifactReceipt, StoredArtifactReceipt]
    old_nested_probe_plan: StoredArtifactReceipt
    old_nested_probe: StoredArtifactReceipt
    harwood_source_manifest: FileReceipt
    harwood_selection: StoredArtifactReceipt
    harwood_review_plan: StoredArtifactReceipt
    harwood_raw_frame_manifest: StoredArtifactReceipt
    harwood_review_jpegs: tuple[FileReceipt, ...]
    harwood_sealed_review: StoredArtifactReceipt
    source_videos: tuple[FileReceipt, FileReceipt, FileReceipt, FileReceipt]
    checkpoints: tuple[FileReceipt, FileReceipt]


@dataclass(frozen=True)
class VerifiedTask0257TemporalInputs:
    training_chain: ContinuousCausalTrainingChain
    ordered_rows: tuple[Mapping[str, object], ...]
    parent_review_plan: Mapping[str, object]
    harwood_review_plan: Mapping[str, object]
    source_groups: Mapping[str, object]
    old_embeddings: tuple[Mapping[str, object], Mapping[str, object]]
    replayed_nested_probe: Mapping[str, object]
    validated_input_paths: tuple[Path, ...]


def verify_task0257_temporal_inputs(
    *,
    paths: Task0257InputPaths,
    expected_receipts: Task0257ExpectedReceipts,
) -> VerifiedTask0257TemporalInputs: ...


def seal_vru_causal_temporal_feature_plan(
    *,
    inputs: VerifiedTask0257TemporalInputs,
    environment_contract: Mapping[str, object],
) -> dict[str, object]: ...


@dataclass(frozen=True, init=False)
class VerifiedTemporalFeaturePlan:
    """Opaque value constructible only by load_verified_temporal_feature_plan."""


@dataclass(frozen=True, init=False)
class VerifiedTiledSwinEmbeddings:
    """Opaque value constructible only by load_verified_tiled_swin_embeddings."""


@dataclass(frozen=True, init=False)
class FreshlyProducedTiledSwinEmbeddings:
    """Private, transaction-local producer value; never a public trust token."""


@dataclass(frozen=True)
class ResumeCAS:
    device: int
    inode: int
    size_bytes: int
    internal_sha256: str
    file_sha256: str


def load_verified_temporal_feature_plan(
    *,
    plan_path: Path,
    expected_artifact_sha256: str,
    expected_file_sha256: str,
) -> VerifiedTemporalFeaturePlan: ...


def load_verified_tiled_swin_embeddings(
    *,
    embeddings_path: Path,
    expected_artifact_sha256: str,
    expected_file_sha256: str,
    plan: VerifiedTemporalFeaturePlan,
) -> VerifiedTiledSwinEmbeddings: ...


def combine_tiled_swin_embeddings(
    tile_embeddings: Sequence[Sequence[float]],
) -> list[float]: ...


def extract_vru_causal_tiled_swin_embeddings(
    *,
    plan: VerifiedTemporalFeaturePlan,
    source_video_paths: Sequence[Path],
    checkpoint_path: Path,
    expected_checkpoint_file_sha256: str,
    attempts_root: Path,
    expected_prior_attempt_receipts: Sequence[StoredArtifactReceipt],
    expected_resume_cas: ResumeCAS | None,
    private_embeddings_staging_path: Path,
    batch_size: Literal[1],
    device: torch.device,
) -> FreshlyProducedTiledSwinEmbeddings: ...


@dataclass(frozen=True)
class ModuleAFinalGeneration:
    retrospective: Mapping[str, object]
    baseline_evaluator: Mapping[str, object]
    candidate_evaluator: Mapping[str, object]
    mechanical_gate: Mapping[str, object]


def _build_vru_causal_temporal_final_generation(
    *,
    inputs: VerifiedTask0257TemporalInputs,
    plan: VerifiedTemporalFeaturePlan,
    tiled_embeddings: FreshlyProducedTiledSwinEmbeddings,
) -> ModuleAFinalGeneration: ...


def verify_vru_causal_final_evaluator(
    *,
    evaluator_path: Path,
    expected_artifact_sha256: str,
    expected_file_sha256: str,
    verified_inputs: VerifiedTask0257TemporalInputs,
    tiled_embeddings: VerifiedTiledSwinEmbeddings,
) -> dict[str, object]: ...


def verify_vru_causal_temporal_final_generation(
    *,
    generation_dir: Path,
    expected_file_receipts: Mapping[str, str],
    expected_artifact_receipts: Mapping[str, str],
    verified_inputs: VerifiedTask0257TemporalInputs,
    tiled_embeddings: VerifiedTiledSwinEmbeddings,
) -> ModuleAFinalGeneration: ...
```

The two `Verified*` classes have no public constructor, deserializer, or
mapping-based adapter. Their private instance token binds the canonical capped
byte snapshot, caller-expected internal/file hashes, no-follow file identity,
and verifier version. Public artifact replay/evaluation APIs accept those
instances only and revalidate their bound identity immediately before model
load, fit, and verification. `typing.cast`, subclassing,
pickle/copy reconstruction, or passing a structurally similar mapping is not a
trust boundary and is rejected at runtime. A producer's in-memory mapping is
untrusted until written no-clobber and loaded back through the path-plus-two-
expected-hashes constructor. For newly produced embeddings that path is the
exclusive private terminal-generation staging member; verification does not
make it a stable output. Only the later atomic terminal-directory rename can do
that.

The atomic producer is not an artifact consumer and cannot possess an external
receipt for bytes it has not created. Its extractor therefore returns the
separate non-public, non-serializable
`FreshlyProducedTiledSwinEmbeddings` capability after strict schema, finite,
dimension, formula, plan/input/checkpoint/source, and staging-identity checks.
Only the private same-transaction `_build_*` function accepts that capability;
it cannot be reconstructed from a mapping or used by a public replay API. This
does not relabel the producer's own hash as external evidence. After terminal
publication, the caller freezes the returned embedding internal/file hashes in
an external registry; only then can public verification construct
`VerifiedTiledSwinEmbeddings` from the final/terminal member path and those
expected values. The independent verifier reruns the frozen computational
projection before any terminal decision is trusted.

The tuple orders above are contract fields, not hints. Parent children are
Hazen/Randolph/VTV; old embeddings and checkpoints are MViT/Swin; videos are
Hazen/Randolph/VTV/Harwood. Each caller-supplied JPEG receipt tuple must have
the exact manifest order and cardinality and is compared independently with
the verified manifest's same ordered entries. `validated_input_paths` is the
canonical de-aliased tuple of every leaf field above, including every JPEG,
both v2 children, every video, and both checkpoints; equality with that full
set and cardinality is verified before a model load or label access.
Final-generation and evaluator verifiers derive the exact `mvit_v2_s` and
`swin3d_t` matrices from `VerifiedTask0257TemporalInputs` and the tiled matrix
from `VerifiedTiledSwinEmbeddings`, each in the verified 45-row order. The
verifier itself derives `mvit_v2_s+swin3d_t` by MViT-then-Swin concatenation;
callers cannot supply any raw or pre-concatenated feature matrix as expected
truth.

`verify_task0257_temporal_inputs` first takes capped byte snapshots and verifies
caller-supplied file hashes, then calls the existing public v2-chain verifiers.
It must independently read and verify the parent review plan because the current
v2 chain verifier replays the parent index and six children but does not return
or reread the parent review plan. It reruns
`screen_vru_causal_video_embeddings_nested_v2(...)` from exact old artifacts
and compares both the recomputed internal identity and canonical stored bytes to
the external TASK-0257 internal/file receipts.

The current v2 verifier alone is insufficient for this boundary because its
returned path set does not cover the retained parent selection/source
manifest/review plan/raw-frame manifest/JPEGs/sealed review or videos and
checkpoints. Module A therefore verifies those named paths and their external
receipts directly, cross-checks their graph edges against the full v2 replay,
and only then constructs `VerifiedTask0257TemporalInputs`.

No new module imports TASK-0257 private helpers such as `_fit_probability_model`
or `_select_candidate`. Equivalent logic is promoted into reviewed public
Module-A primitives or implemented locally with its own exact tests; private
cross-module coupling would make the receipt contract unstable.

## Evaluation algorithms

Every producer and verifier scaler/classifier fit first validates Python,
NumPy, SciPy, raw SciPy-config SHA, sklearn, threadpoolctl and normalized
backend identity, enters `threadpool_limits(limits=1)`, and rechecks that both
frozen `libomp` entries report one thread immediately before and after the fit.
This applies independently to every inner, outer, LOGO, and all-45 fit. A
mismatch fails rather than being recorded as harmless provenance.

### Nested retrospective

For outer held game `g`, construct masks before examining its target values.
For each of the other three inner held games, fit a new scaler and classifier on
the remaining two games and predict the inner-held rows. The frozen data have
both classes in every required fit; a single-class or empty split fails instead
of introducing a fallback. Pool inner probabilities in canonical row order,
select the threshold from the 21-value grid by the exact score tuple:

```text
(balanced_accuracy, f1, precision, recall, threshold)
```

using maximum lexicographic order, then refit on all three outer-training games
and predict `g`. Only after probability and decision seals exist are held truth
and metrics joined for the diagnostic record.

The held-label invariance test runs the same fold with an adversarially changed
held target vector and compares a canonical projection that excludes only
truth-derived fields. Plan, features, inner rows/probabilities, threshold grid
results, selected threshold, scaler state, coefficients/intercept, held
probabilities, and held decisions must remain byte-identical.

### Final evaluators

For each allowed representation and each held game, fit on the other three and
store the complete held probabilities. Pool the four folds, evaluate all 21
thresholds, and select baseline by:

```text
(
  balanced_accuracy,
  f1,
  precision,
  recall,
  -original_representation_position,
  threshold,
)
```

Candidate uses the same tuple without a representation position. After
selection, the producer instantiates a new scaler and classifier and calls each
`fit` exactly once on all 45 rows. Tests instrument producer fit-call counts; a
refit per producer serialization is forbidden. This constraint does not apply
to verification. Given the full verified rows and feature matrices, the
verifier starts from fresh estimator instances, independently reruns all four
LOGO folds for both baseline variants or the sole candidate variant, recomputes
all 21 threshold rows and the selected pair, performs a fresh all-45 fit, and
compares every `logo_selection`, `selected_evaluator`, `all_45_refit`, receipt,
parameter, probability, metric, and canonical byte field. Stored OOF rows or
stored coefficients are never the verifier's source of expected truth.

## Receipt, path, atomicity, and resume design

All JSON readers reject duplicate keys and non-standard constants, open with
no-follow semantics, require a regular file, cap size at `67,108,864` bytes
before allocation, and read/hash/parse one byte snapshot from that descriptor.
They bind device/inode/size/mtime and the external internal/file receipt in a
verified token, then revalidate those values at every model/fit/publication use.
Every file SHA comparison precedes JSON parsing whose contents could influence
a path, model load, or label decision.

Checkpoint files are capped at `209,715,200` bytes. Each is opened with
`O_RDONLY|O_NOFOLLOW`; regular-file type, expected size and device/inode are
checked and its file SHA is streamed from that descriptor. For the Swin
checkpoint actually consumed by the extractor, that same descriptor is rewound
and `torch.load(..., weights_only=True)` consumes its backed file object; a
final `fstat` must match before close. Path re-open between verification and
load is forbidden. The MViT checkpoint remains receipt-only and is never passed
to `torch.load` because its retained embeddings are reused.

OpenCV's frozen `VideoCapture(path, cv2.CAP_FFMPEG)` interface cannot consume
that verified descriptor. Therefore each exact-size video receives a full
streamed hash plus lstat/stat identity check immediately before decode and a
second full hash/identity check after its last decode and before any staged
embedding/attempt output is trusted. A persistent mutation or replacement at a
boundary fails terminally and discards staging. This is not a stable-descriptor
guarantee: the threat model excludes an adversarial concurrent writer capable
of changing bytes and restoring the original full hash between the two checks.
Tests cover observable mutation/replacement before, during, and after decode
and explicitly do not claim coverage of mutate-and-restore.

Before any read/write combination, resolve identities and reject:

- identical paths;
- symlink or hardlink aliases;
- case-fold-equivalent paths;
- either path being an ancestor/descendant of the other;
- output paths crossing each other or any immutable input root; and
- unsafe stored names, absolute paths, `..`, or NUL.

Attempt selection is not based on directory discovery alone. Attempt 1 requires
an externally approved empty-chain mode and absent `attempt-0001`. Attempt 2 or
3 requires the externally receipt-bound complete prior chain, an exact latest
`ResumeCAS`, a consecutive absent next target, and a strict prefix extension.
The resume is opened no-follow and its device/inode/size/internal/file tuple is
checked from the same capped snapshot; immediately before model load and next-
attempt publication, a compare-and-swap re-open must still match that exact
external tuple. A copied value from the resume itself, a missing tuple member,
a chain gap, an old rather than latest resume, or a self-resealed resume fails
before model load.

Every file is encoded in a private same-directory staging generation, flushed,
file-`fsync`ed, re-read and verified, and the staging directory is `fsync`ed.
Stable file/directory publication uses macOS `renameatx_np(..., RENAME_EXCL)`
or a tested equivalent with identical no-replace semantics; `os.replace` and
plain overwrite-capable rename are forbidden. Immediately before publication,
the supervisor revalidates its module lock, exact spec approval, all verified
input identities/receipts, source-video post-hashes, checkpoint descriptor
identity, resume CAS, cumulative counters, disk budget, and absence of the new
target. It then publishes once and `fsync`s the parent. The same rule applies to
each recoverable-signal attempt directory and the mutually exclusive
`terminal_failure_v1`/`final_v1` terminal directory. Embeddings and the current
completed/failed attempt are never published as separate stable targets. Any
failure exposes no partial stable terminal target and never modifies a previous
generation.

The current completed attempt, embeddings, and four final records are encoded
and cross-verified in one new random generation directory and become visible
only through the final no-clobber directory publication. A handled terminal
failure similarly publishes its current attempt/log, optional already-verified
embeddings, and failure record in one `terminal_failure_v1` rename. If an
externally receipt-bound terminal directory already exists, the only allowed
path is capped read-only verification and a no-op; partial, drifted,
unreceipted, dual-terminal, or conflicting output fails without repair.

## Resource and disk design

Do not invoke `scripts/run_guarded_training.py` directly: its log-first
lifecycle can create or open output state before Module-A receipt/path/disk
validation. Module A instead defines an offline supervisor that may reuse only
its side-effect-free resource sampling and process-tree termination primitives.
The supervisor performs, in order:

1. verify the externally receipt-bound exact three-file user approval and its
   passing fresh-review receipt;
2. take capped input snapshots and replay every external receipt;
3. verify the complete leaf-path identities, aliases, safe names, required
   absence, attempt-chain continuity, and optional resume CAS;
4. compute per-filesystem worst-case disk accounting from the nearest existing
   ancestors without creating a directory;
5. open the already-existing canonical output-parent directory no-follow and
   acquire a nonblocking exclusive `fcntl.flock` on that descriptor; and
6. while holding the lock, repeat authorization, receipts/identities, CAS,
   absence, cumulative counters, and disk accounting.

These are the complete no-write preflight. Failure at any step returns
`preflight_refusal` through a bounded diagnostic and nonzero exit, creates no
new file or directory, and does not alter an existing artifact. The directory
lock is an advisory kernel lock on an existing descriptor and leaves no lock
file. Only after all six steps pass does opening the random attempt staging
directory admit an attempt; the same lock remains held through child reaping
and every attempt/final publication.

If `final_v1` already exists, the supervisor performs only the capped read-only
verification path. An externally receipt-bound byte-identical complete
generation returns as a no-op without creating or opening any path for write; a
partial, drifted, or unreceipted generation fails. The absence rule applies to
a new materialization after that branch. Disk grouping for an absent target
uses its nearest existing ancestor's device and never creates a directory to
discover the device.

After admission, the supervisor exclusively creates the current attempt's
private resource-log staging file with `O_NOFOLLOW`, mode `0o600`, and no
overwrite. The global active-runtime clock resumes from the prior attempt
record; time between attempts is excluded. Samples are scheduled at cumulative
active `t=2,4,...,43_200` seconds, never at `t=0`, yielding exactly `21,600`
possible rows. Before writing a fully encoded row, both the row-count and the
cumulative published-plus-staged `16,777,216`-byte limit are checked. Reaching
the byte limit is terminal before that row is written. At `t=43_200`, the final
allowed complete row is written if it fits; if work is not already complete,
the runtime/sample cap then terminates with exit `75` and no `t=43_202` row.
A three-consecutive resource breach is likewise terminal exit `75`. None of
these conditions is converted to a resume.

The exact state transitions are:

| Current state | Event and guard | Next state / durable output |
| --- | --- | --- |
| `no_attempt` | no-write preflight fails | invocation ends as `preflight_refusal`; no new artifact |
| `no_attempt` or `interrupted_recoverable` | preflight passes, next ordinal `<=3`, next target absent | `attempt_running`; private staging only |
| `attempt_running` on attempt 1 or 2 | external `SIGINT`/`SIGTERM`; `0 < new_rows_verified` and `completed_count < 45`; every input/CAS/resource/disk/determinism check still passes | immutable attempt generation with log, record, resume; `interrupted_recoverable` |
| `attempt_running` | all 45 rows verified and video post-hashes pass | `finalizing_private`; completed attempt and embeddings remain in private staging |
| `attempt_running` | guard/cap/decode/model/receipt/CAS/determinism/non-progress/unsupported-signal failure, or incomplete attempt 3 | atomic `terminal_failure_v1` with terminal attempt/log and failure record when publication succeeds; terminal `mechanical_failure` |
| `finalizing_private` | all retrospective/evaluator/publication checks pass and bounds pass | one atomic `final_v1` containing completed attempt, embeddings, and four records; terminal `mechanical_pass` |
| `finalizing_private` | all non-bound checks pass and a frozen bound fails | the same atomic `final_v1`; terminal `temporal-hypothesis-rejected` |
| `finalizing_private` | any non-bound check or signal fails | atomic `terminal_failure_v1` with completed attempt, embeddings, and failure record; terminal `mechanical_failure` |

Only `interrupted_recoverable` is resumable. Attempt ordinals and cumulative
counters strictly increase; no transition returns to an earlier state. The
`finalizing_private` state has no stable file or directory and cannot be
restarted after a crash. The two terminal directories are mutually exclusive.
`SIGINT` exits `130` and
`SIGTERM` exits `143` only when the recoverable generation publishes; otherwise
the applicable terminal-failure exit is used. The supervisor launches the
extractor child with `PYTHONHASHSEED=0` and every frozen CPU thread environment
variable already set; changing them after interpreter startup is forbidden.
If terminal-generation publication itself fails, no current attempt,
embeddings, or failure record becomes stable. A bounded diagnostic must name the
failed attempt ordinal and `publication_failure`; it may never fabricate a
receipt for unpublished members. The process still returns a terminal nonzero
exit with no trusted partial terminal output.

The extractor also defines a stricter disk primitive:

```python
def verify_disk_write_budget(
    *,
    output_paths: Sequence[Path],
    worst_case_new_bytes: int,
    reserve_bytes: int = 3_758_096_384,
) -> None: ...
```

It groups targets by filesystem and checks each filesystem once. Current free
space already reflects retained immutable signal attempts; their exact sizes
are verified for cumulative counters but are not subtracted again. Declared new
bytes include every remaining attempt up to three, the remaining portion of the
global `16,777,216`-byte resource-log allowance, one `67,108,864`-byte attempt
record for each remaining ordinal, at most one `67,108,864`-byte resume for
each remaining ordinal 1 or 2, and the larger mutually exclusive terminal
generation: current attempt/log plus optional embedding and one failure member,
or current completed attempt/log plus embedding and the four final records, all
JSON members at the same ceiling. Private staging allocations are charged while
they exist; a same-filesystem rename does not invent a second copy. No
filesystem or byte category is double-counted.

The check runs before model load and immediately before every material write or
publication. It does not opportunistically delete protected evidence to make
the check pass.

The extractor receives `batch_size=1` as a literal contract, clears batch-local
MPS tensors/cache after every tile, and never queues a second clip. A handled
signal first stops new rows, completes verification of the current row or
discards it, validates `0 < new_rows_verified` and `completed_count < 45`, syncs
a complete JSONL, reaps the entire process group, and only then may publish the
immutable signal-only attempt/resume generation. All other failures reap first
and publish a terminal attempt/failure record without a resume. Resource-log and
process-tree assertions are acceptance evidence, not advisory telemetry.

## Threat model and hardening

Trust boundaries are the JSON artifacts, caller paths, large source videos,
model checkpoints, resume state, and generated evaluator parameters.

- **Tampering/spoofing:** exact-spec user approval, independent internal/file
  receipts, exact schemas, complete child replay, source/checkpoint hashes, and
  immutable generations.
- **Path traversal/overwrite:** safe relative stored names, identity/alias and
  ancestor checks before reads, same-directory random temporaries, a
  module-exclusive existing-directory lock, no-clobber publication, and no
  input path derived from free-form artifact text.
- **TOCTOU:** capped immutable JSON snapshots and checkpoint loading use the
  same verified no-follow descriptors; resume uses external dev/inode/hash CAS.
  Video uses explicit pre/post full hashes and excludes adversarial concurrent
  mutate-and-restore rather than claiming stable-descriptor coverage.
- **Resource denial:** 45-row/4-tile/16-frame caps, batch one, JSON size caps,
  three attempts, cumulative runtime/sample/log limits, post-worst-case disk
  reserve, sustained guard, and process-group reaping.
- **Information leakage/overreach:** plan and embedding artifacts contain no
  labels or review prose; all downstream eligibility is false; no secrets or
  personal data are introduced.
- **Repudiation:** exact resource rows, the immutable attempt/resume chain,
  ordered receipt graph, deterministic plan, complete OOF predictions, and
  immutable gate decision retain an audit trail.

Model output and stored floats are untrusted data. Non-finite, wrong-dimension,
or schema-drifted values never reach sklearn or a path/shell operation. CLIs use
argument arrays and never interpolate artifact text into a shell command.

## Reusable open-source capability and AGU boundary

- Reuse the retained torchvision Swin3D-T and MViT-V2-S implementations and
  Kinetics-400 checkpoints, NumPy float32 arithmetic, scikit-learn
  `StandardScaler`/`LogisticRegression`, OpenCV decoding, existing TASK-0257
  verifiers, and the sustained resource guard.
- torchvision and scikit-learn are already retained dependencies. The plan
  records implementation/weight provenance and preserves TASK-0257's warning
  to verify upstream dataset/weight terms. No new package, model family,
  download, external service, language stack, or runtime adapter is needed.
- AGU owns only a thin offline receipt/sampling/evaluation adapter and immutable
  diagnostic artifacts. It does not change v3 preprocessing, service/API
  behavior, configuration, BFF boundaries, model registries, default pointers,
  VLM answers, readiness, or blind-inference state.

## Alternatives rejected

### Re-linspace each candidate event from `start_frame` to `end_frame`

Rejected because those endpoints are the first and last of 64 half-open-window
review samples, not the original eight-second boundary. It would create an
unapproved sampling variable and can duplicate tile boundaries.

### Reuse the existing 768-dimensional video-embedding schema

Rejected because it validates native backbone dimension 768, contains labels,
and cannot prove the four-tile derivation or the sole 1,536-vector contract.

### Reuse the temporal runtime model schema

Rejected because it is consumable by the runtime shot gate. Module A must be
unreachable from runtime/default/readiness/promotion paths.

### Derive baseline LOGO predictions from the archived nested probe

Rejected because each archived outer fold retains only its inner-selected
variant. It does not contain four-game OOF predictions for both baseline
variants and therefore cannot support the approved final selection.

### Publish the two evaluators and gate as unrelated files

Rejected because a crash could expose an unmatched evaluator pair or a gate
that references an absent file. One immutable generation provides a clear
atomic boundary while retaining separately addressed files.

### Continue with another variable after a bound miss

Rejected as post-hoc tuning. The required output is
`temporal-hypothesis-rejected`, and TASK-0258 stops before Module B.

## Review and downstream gate

Before implementation, a fresh-context reviewer must read the approved
capability map and the exact requirement/solution/gate bytes. Any Critical or
Required specification issue blocks until fixed and re-reviewed. After that
review passes, the user must separately approve the exact three-file SHA tuple;
the approval is recorded externally without editing those files. Any byte drift
invalidates the approval, and any semantic drift also reopens fresh review.

After implementation and all verification, a different fresh-context reviewer
must read the approved map/spec receipts, exact diff, RED/GREEN evidence, output
schemas, receipts, immutable attempt chain, and resource/atomic/resume logs. Any
Critical or Required implementation issue blocks until fixed and re-reviewed.

Only a verified `mechanical_pass` plus external internal/file receipts for both
final evaluator files can become inputs to a later Module-B specification. This
solution neither approves that specification nor authorizes inspection or
acquisition of independent media.
