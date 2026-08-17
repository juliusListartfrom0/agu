# TASK-0257 Requirement — Harwood additive training bridge

## Objective

Create an additive, development-only v2 training chain that resolves the frozen
three-game TASK-0254 v1 export together with the frozen TASK-0256 Harwood review.
The resolved view must contain four held-game groups, two production families,
and exactly 45 binary shot-validity rows (`17` positive, `28` negative), while
excluding all three `uncertain` reviews. Run a nested outer-game-held video
representation probe over that resolved view without changing any v1 schema,
API, file, or byte.

This task is one capability with two sequential stages: build and verify the
additive training chain, then consume that verified chain in the nested probe.
The probe cannot exist safely without the chain, so a separate capability map
is not required.

## Assumptions

1. The TASK-0254 export is an immutable parent, not an input to rebuild:
   - schema `agu.vru-causal-shot-validity-training-export.v1`;
   - internal SHA-256
     `1b0bcef2c67f0d0e1fa35b79042584c43d938f05e893e89a4a64266ab9d0e464`;
   - file SHA-256
     `afb406ef84ae57b8d6cbe565a848b03bcefc2fdd449d8c223374e04d7f7dc3e0`;
   - 22 rows: Hazen `6+/2-`, Randolph `4+/3-`, VTV `4+/3-`, with two
     `uncertain` rows already excluded.
2. The TASK-0256 Harwood chain is frozen at 24 selected windows: `3` shot,
   `20` not-a-shot, and one `uncertain`
   (`closure-84dc8a546ac128ce09978c0f-0023`). Only the 23 determinate rows may
   cross the training boundary.
3. Hazen, Randolph, and Harwood are three distinct games in one HCTV production
   family. VTV is one game in the VTV production family. Four outer held-game
   folds are possible, but there are only two production families and only one
   game in VTV; this cannot establish production-held generalization.
4. All inputs are prior-stratified development evidence rather than continuous,
   exhaustive full-game truth. No output from this task is a formal benchmark,
   an 85% acceptance result, a runtime answer, or a promotion decision.

## Frozen input receipts

### Parent v1 child assets

The v2 verifier must replay the parent v1 verifier and bind the parent index to
all six child files. It must not accept the index receipt alone.

| Asset | File SHA-256 |
|---|---|
| `hazen_candidate_bundle_v1.json` | `378a08604962a8733898312478d9e3ad762791563d00f7608cd4353cf2dbaaa9` |
| `hazen_shot_validity_labels_v1.json` | `acabe7e07ad8f98350ef492622bea8d7230ce9dfe1b8d278a4919248faad8dda` |
| `randolph_candidate_bundle_v1.json` | `2d7d0d3b6d872cb1b94c43c8c595df1d98c1485b82faa72ba43f3157dafc8cd8` |
| `randolph_shot_validity_labels_v1.json` | `44b1588e583edb1e4e66ab270511245938c19e8197cd96df0a09aec3da8dda09` |
| `vtv_candidate_bundle_v1.json` | `65c0fb60b8338512d50ee3c33ca01b93c3c9d6f4c126a27ca311f8e8709f22f2` |
| `vtv_shot_validity_labels_v1.json` | `5274a0ce6b2aae635b763b138a88487d28da61f9945f5e779526efbb52ba1b98` |

### Harwood v2 evidence chain

| Artifact | Internal SHA-256 | File SHA-256 |
|---|---|---|
| source manifest | n/a | `8786b64fc02f4d6105acbe097617a733f7276469a60a721b883ac64062d51ded` |
| continuous selection | `5a7eaa1ecefa295c5d97632e05e7aea2187eafac0082ba6ea6341fc6b372a80f` | `35126baa4ae15a888391ff69840c5410c4195f48a0608b25f7ec1a680a576138` |
| causal review plan | `87c9b92c548000527f253082c6190ec82022934d9c3db244be32e50505244911` | `bbf12a7f7de808d467c6057b2e93170765e0098b47fd4465180ac8c9b9575c24` |
| raw-frame manifest | `0757e339ffd5acb97d5b8f42790cfced2de825e44deea20095abd7bedfeae776` | `984365ba89c71d781a1e36566a5c5327006349c9c0c6744d0246449933a5bcca` |
| sealed review | `5f2219a4cdcc7e8d4e40dde060c8ba82b5070eb6db471c1ce2855cea9d3e5cb6` | `ce9ad5d00db77baed53d874cc11ee7deb1ba960e75592ddf2c591a2490319fff` |

The retained source video must independently match SHA-256
`bf7f3135774027aec3c2827cfa282c95ac4f958dd2bccd93e42f370d4b1f8b81`
and size `1,376,342,882` bytes. The verifier must validate all 1,536 retained
JPEG hashes. Caller-supplied expected internal receipts are required for the
selection, plan, raw-frame manifest, and sealed review; they must not be
learned from the artifacts under verification.

## Contracts and project structure

- New export schema:
  `agu.vru-causal-shot-validity-training-export.v2`.
- New production grouping schema: `agu.vru-causal-source-groups.v1`.
- New additive in-memory boundary: `ContinuousCausalTrainingChain`.
- Reuse the existing `agu.training-annotation.v1` manifest schema for a new,
  four-video manifest; do not edit the TASK-0254 manifest.
- New builder:
  `build_vru_causal_shot_validity_training_extension_export(...)`.
- New nested consumer: `screen_vru_causal_video_embeddings_nested_v2(...)`.
- Owning Python code belongs under `app/analysis/`; thin CLIs belong under
  `scripts/`; unit/contract tests belong under `tests/`.
- Real outputs belong in a new directory under
  `analysis_outputs/public_research/`; v1 files remain in
  `vru_causal_closure_training_v1/` and are read-only inputs.

The v2 export is a chain, not a copied v1 index. It stores an exact parent v1
receipt, verifies and references the six parent child assets, adds only the
Harwood candidate/label children, and exposes a deterministic resolved 45-row
view through its verifier. A consumer must not concatenate JSON files on its
own or infer production families from filenames.

## Commands

- Environment: canonical `.venv` / Python 3.11 only.
- Focused tests:
  `.venv/bin/python -m pytest -q <TASK-0257 focused test files>`.
- Full tests: `.venv/bin/python -m pytest -q`.
- Lint/format:
  `.venv/bin/python -m ruff check <changed Python files>` and
  `.venv/bin/python -m ruff format --check <changed Python files>`.
- Embedding extraction must run batch size one through
  `scripts/run_guarded_training.py`; the concrete export, extraction, and probe
  CLI commands must be recorded in `development.md` when implemented.

## Testing strategy

Implementation must follow RED then GREEN. Tests use tiny synthetic files for
contract failures and the frozen real artifacts for one end-to-end replay.

| Area | Required failing-first cases |
|---|---|
| Parent immutability | Wrong parent internal/file SHA, changed parent schema, missing/reordered/renamed child, child content drift, bundle/label mismatch, or any write target aliasing a v1 path fails before output. A regression asserts every v1 file SHA above remains unchanged. |
| Harwood provenance | Wrong source/selection/plan/raw/review receipt, source byte/size drift, missing/extra/changed JPEG, plan coverage drift, selection-plan mismatch, or self-resealed label swap fails closed. |
| Mapping | Exact extension counts `24/23/3/20/1` and resolved counts `48/45/17/28/3`; `uncertain` cannot become negative; outcome, notes, confidence, reviewed release/rim anchors, and prior AGU answers never enter candidate geometry. |
| Source groups and training manifest | Exact four unique game IDs map to `hctv` or `vtv`; three HCTV games and one VTV game; unknown games, duplicate games, missing games, extra groups, mutable filename inference, or an externally expected source-group SHA mismatch fails. A new training manifest binds four source videos, the v2 index, and all four label files; overlap, receipt, or count drift fails. |
| Exact schema | Unknown/missing fields, non-canonical ordering where order is contractual, duplicate three-part keys, unsafe IDs, absolute/escaping paths, and inconsistent counts fail. Deterministic inputs produce byte-identical children/index. |
| File safety | Direct, symlink, hardlink, case-fold, ancestor/descendant, or cross-output aliasing fails before reads/model loading; writes use same-directory random temporaries, file `fsync`, and atomic replace. |
| Embedding join | Exactly one embedding per `(source_video_sha256, candidate_bundle_sha256, event_id)`; missing, extra, duplicate, mislabeled, wrong-dimension, checkpoint/sampling drift, or export/source-group receipt drift fails. |
| Nested evaluation | Exactly four outer game-held folds. Outer labels do not influence variant, regularization, normalization/PCA, or threshold selection; perturbing held labels cannot change those selections. Inner folds are game-held within outer training only. Report each held game plus pooled metrics and production-family summaries. |
| Non-promotion | Export, manifest, embeddings, and probe reject any value other than `runtime_consumable=false`, `formal_evaluation_eligible=false`, `promotion_eligible=false`, and `promoted=false`; readiness and blind-inference state are untouched. |
| Resource safety | Batch size other than one fails; preflight insufficient disk/memory fails before model loading; three consecutive breaches of the configured memory/available-memory/swap/CPU limits stop with exit 75, reap the process tree, sync the log, remove partial output, and allow only verified resume. |

## Boundaries

### Always

- Verify the parent index, every parent child file, the complete Harwood chain,
  every retained JPEG, and all source media before constructing labels.
- Keep parent and extension provenance distinct while exposing one verified
  resolved view.
- Freeze the source-group manifest and probe plan before extracting or reading
  held labels.
- Use batch size one, release MPS batch tensors/cache between flushes, and run
  the sustained resource guard with defaults no weaker than 90% system memory,
  2 GiB available memory, 95% system CPU, and three consecutive breaches.
- Treat all annotation/model JSON as untrusted external input at the verifier
  boundary and use exact allowlisted schemas.

### Ask first

- Adding a dependency, changing the embedding checkpoint/sampling contract, or
  using a cross-model interactive reviewer.
- Any change to runtime configuration, checkpoint pointers, readiness, blind
  inference, formal evaluation, or promotion state.

### Never

- Modify, overwrite, migrate, relax, or reinterpret v1 schema/API/artifacts.
- Copy `uncertain` into a binary class or export reviewed outcome/release/rim
  anchors as model input.
- Use Codex labels as AGU runtime answers or VLM predictions.
- Select a model, threshold, or preprocessing variant from an outer-held label.
- Describe four games as four production domains, or claim this task satisfies
  the 85% gate.

## Success criteria

1. The v2 chain verifies the exact frozen parent and Harwood receipts and
   resolves 45 unique rows with exact class/source counts.
2. The source-group manifest reports four games but exactly two production
   families and is externally SHA-bound by the export, embedding, and probe. A
   separate four-video training manifest binds the resolved annotation inputs.
3. The nested probe runs four outer game-held folds, selects all tunable choices
   exclusively inside each outer training partition, and reports per-game,
   pooled, and production-family diagnostics.
4. Every artifact is deterministic, atomic, self-verifying, externally
   receipt-bound where trust must cross files, and permanently development-only.
5. Resource logs and process checks show no sustained threshold violation and
   no surviving training process; all listed focused/full/lint/diff checks pass.
6. A fresh-context adversarial review finds no Critical or Required issue.
   Cross-model interactive review is recorded as pending unless the user gives
   explicit authorization; it must not be invoked implicitly.

## Delivered evidence (2026-08-16)

The development-only implementation now satisfies the additive-chain and
nested-probe requirements. The frozen receipts are:

| Artifact | Internal SHA-256 | File SHA-256 |
|---|---|---|
| v2 training export | `9dbbffc81828d053154dd2e3adeadfb80869e5e1d7c6a03659df5ebc38fd7b35` | `6fd7680694122ac878154886550de6d0070ec32985ae37e038e894cd86b359aa` |
| four-video training manifest | `ddf1af3b23224b7fb99e96cb06226938686062cee5bf28bef153822f23847aba` | `e8261c550bf30da4b36d9b6773717769775e829d3c3c26f7cd750d2d7407ac3b` |
| frozen probe plan | `de1444c74672a827af2660a825af1d2792a4c8f61fdf47558ddc372261290333` | `78cc7dc64ea0089a431d362485ed85faee2b75863d797492c8a787d1420729c2` |
| MViT-V2-S embeddings | `d6aa1d9bbb478972d5880bf944da4b0fa9ef1d7f1aa1640f79e76281c23b201f` | `01dbd913a5fdda3a7b9173cbd2509aa191edf50d1010d00785a0c52a9d3b9a51` |
| Swin3D-T embeddings | `3dc7caacfbc2cb3d8d8df856094131bde4fab3c3615811f363725490b7fea40b` | `1c849096276c2e60f9d01cc064f54aa7426128634e85e7288faf526e76b031b5` |
| nested probe | `92d5d8045b2f39834eac8c57a662343f98cb9ba9e2049e2edbabe704c398ae21` | `7d1c763656f3e940f6ca9db1bdad4e40f1edb0112d756fd549183fb68008d720` |

The probe resolves 45 rows (`17+/28-`) and reports pooled
TP/FP/FN/TN=`14/4/3/24`, precision=`0.777778`, recall=`0.823529`, and
F1=`0.800000`. These are prior-stratified, non-exhaustive development
diagnostics across four games but only two production families. Every
eligibility flag remains false; readiness remains `not_ready`, blind inference
remains paused, and the formal 85% gate is unchanged.

Focused extractor tests (`132`) and the broader export/probe tests (`205`)
passed. The guarded full suite passed with `1712 passed, 5 skipped, 15
warnings`; scoped Ruff checks passed. A repository-wide Ruff diagnostic still
finds unrelated historical formatting/lint debt and was not mass-fixed as part
of this task.

## Resolved choices and remaining question

- The frozen plan predeclared Swin3D-T alone, then MViT-V2-S + Swin3D-T, with
  train-split-only standardization, no PCA, `C=0.01`, and a `0.00..1.00`/`0.05`
  threshold grid before held-label evaluation.
- Whether a future second non-HCTV game can make production-family-held
  evaluation mechanically meaningful. TASK-0257 cannot answer that question.
