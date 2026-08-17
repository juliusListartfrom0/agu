# TASK-0257 Solution — additive v2 chain and game-held probe

## Decision

Add a new v2 chain beside the frozen v1 implementation. Do not widen v1's
hard-coded three-source/22-row contract, overload its CLI, or edit any v1
artifact. The v2 index is an additive receipt graph: its parent edge points to
the exact verified v1 export and six children, and its extension edge points to
the exact Harwood review chain plus two new Harwood training children.

This design preserves the useful fail-closed v1 behavior and avoids silently
changing existing consumers under the same schema. It also makes rollback
trivial: removing the v2 directory leaves v1 byte-for-byte intact.

## Resolved data model

`ContinuousCausalTrainingChain` owns the distinction between stored edges and
the resolved training view:

```python
@dataclass(frozen=True)
class ContinuousCausalTrainingChain:
    index: dict[str, Any]
    extension_candidate_bundles: dict[str, dict[str, Any]]
    extension_label_files: dict[str, dict[str, Any]]
    source_groups: dict[str, Any]
    resolved_examples: tuple[dict[str, Any], ...]
    validated_input_paths: tuple[Path, ...]
```

The stored v2 index uses schema
`agu.vru-causal-shot-validity-training-export.v2` and includes exact fields for:

- hard-false eligibility flags and `development_diagnostic_only` purpose;
- the parent v1 schema, internal SHA, file SHA, and six child file/bundle
  receipts;
- the Harwood source manifest, source video, selection, plan, raw-frame
  manifest, sealed review, and 1,536-JPEG aggregate evidence;
- the two Harwood extension children and their file/bundle receipts;
- mapping policy, exact extension/resolved counts, the three excluded review
  IDs, training routes, and the source-group-manifest receipt;
- a canonical internal `artifact_sha256` computed with that field omitted.

The resolved rows use the existing minimal four-field shape:
`source_video_sha256`, `candidate_bundle_sha256`, `event_id`, and
`event_present`. Parent rows are returned from the verified v1 children; they
are not copied into new candidate or label files. Resolution sorts by a frozen
game order and then event ID, rejects duplicate three-part keys, and checks
exact totals:

| Game | Production family | Positive | Negative | Exported |
|---|---|---:|---:|---:|
| Hazen | HCTV | 6 | 2 | 8 |
| Randolph | HCTV | 4 | 3 | 7 |
| VTV | VTV | 4 | 3 | 7 |
| Harwood | HCTV | 3 | 20 | 23 |
| **Total** | **2 families** | **17** | **28** | **45** |

The three excluded rows remain explicit provenance:
`closure-randolph-0002`, `closure-vtv-0003`, and
`closure-84dc8a546ac128ce09978c0f-0023`.

## Additive interfaces

The new builder has a separate name and requires independent expected receipts:

```python
def build_vru_causal_shot_validity_training_extension_export(
    *,
    parent_export_path: Path,
    expected_parent_artifact_sha256: str,
    expected_parent_file_sha256: str,
    parent_asset_root: Path,
    harwood_selection_path: Path,
    expected_harwood_selection_artifact_sha256: str,
    harwood_review_plan_path: Path,
    expected_harwood_review_plan_artifact_sha256: str,
    harwood_sealed_review_path: Path,
    expected_harwood_sealed_review_artifact_sha256: str,
    harwood_raw_frame_manifest_path: Path,
    expected_harwood_raw_frame_manifest_artifact_sha256: str,
    harwood_source_manifest_path: Path,
    source_groups_path: Path,
    expected_source_groups_artifact_sha256: str,
) -> ContinuousCausalTrainingChain: ...
```

The public v2 verifier accepts the same external receipts and returns a verified
`ContinuousCausalTrainingChain`. It delegates parent validation to the existing
v1 verifier, then independently verifies each parent child byte and every
Harwood edge. It never calls the v1 builder and never derives an expected SHA
from the file being checked.

The source grouping file uses exact schema `agu.vru-causal-source-groups.v1`.
It maps stable game IDs to source-video SHA values and the allowlisted production
families `hctv` and `vtv`; it records four game groups, two production families,
and the HCTV/VTV membership explicitly. Its internal and file SHA values become
required receipts in the v2 export and later embedding/probe artifacts. Build
and freeze it first with a separate deterministic source-group CLI; the export
builder consumes its caller-supplied expected internal receipt rather than
generating and immediately trusting its own grouping input.

Use a new CLI for v2 export. It emits only the v2 index, the Harwood candidate
bundle, and the Harwood binary labels. The CLI preflights every input/output
identity before reading large media, stages all outputs in same-directory random
temporary files, syncs them, and replaces only after the entire set verifies. A
failure removes temporaries and leaves both v1 and the last complete v2
generation untouched.

After the export verifies, reuse the existing `agu.training-annotation.v1`
sealer to create a new manifest rather than editing the TASK-0254 manifest. The
new manifest binds the four exact source videos, the v2 index file, all four
label files (the three immutable parent labels plus the Harwood label), the
shot-validity task type, and the existing disjoint acceptance-video receipt.
Embedding extraction and the nested probe require caller-supplied internal and
file receipts for this manifest as well as for the source-group manifest.

## Label and leakage boundary

The Harwood candidate bundle is built from the label-hidden frozen selection
and plan. It must not contain `shot_sequence`, outcome, confidence, reviewer,
notes, release/rim positions, reviewed anchors, PBP, model probabilities, or a
Codex/AGU prior answer. Labels are written separately with the unchanged policy:

- `shot -> true`;
- `not_a_shot -> false`;
- `uncertain -> excluded`;
- outcome is unused for shot validity;
- target-conditioned anchors are forbidden.

The v2 verifier compares the exact review set against the exact plan, checks the
expected `3+/20-/1 uncertain` partition, and refuses any other count or excluded
ID even when a modified review has been self-resealed.

## Nested probe

Add `screen_vru_causal_video_embeddings_nested_v2(...)` rather than changing
the v1 probe. It consumes only a verified v2 chain, an externally SHA-bound
source-group manifest, the new externally SHA-bound four-video training
manifest, a frozen embedding plan, and complete verified embedding artifacts.

The evaluation protocol is nested outer leave-one-game-out:

1. Freeze representation variants, sampling, dimensions, regularization grid,
   normalization/PCA rules, metric, threshold grid, and tie-breaks before any
   outer-held result is read.
2. Hold out exactly one game. The remaining three games are the only data
   available to preprocessing, model fitting, variant selection, and threshold
   selection.
3. Within the outer-training partition, produce inner game-held predictions.
   Select all tunable choices only from those inner OOF predictions. If an
   inner split is single-class or a candidate cannot be evaluated, apply the
   predeclared deterministic fallback; never inspect the outer labels.
4. Refit the selected pipeline on all three outer-training games and evaluate
   once on the held game.
5. Repeat for Hazen, Randolph, VTV, and Harwood. Report confusion counts,
   precision, recall, F1, balanced accuracy, support, and selected settings for
   every fold, then pooled metrics and descriptive HCTV/VTV summaries.

Production-family summaries are diagnostic aggregation only. With only one VTV
game, TASK-0257 must not run or claim a statistically independent
leave-one-production-family-out gate.

The probe schema hard-codes `runtime_consumable=false`,
`formal_evaluation_eligible=false`, `promotion_eligible=false`, and
`promoted=false`. Its verifier rejects attempts to flip any flag, regardless of
metric values. It cannot write a checkpoint, update a default pointer, change
readiness, or resume blind inference.

## Resource plan

Reuse retained local video checkpoints; no new backbone download is part of
this task unless separately approved. Extract exactly 45 rows with batch size
one. Run the extractor under `scripts/run_guarded_training.py` with the existing
sustained guard defaults: memory at most 90%, at least 2 GiB available memory,
CPU at most 95%, and stop after three consecutive breaches; record swap and
aggregate process-tree RSS as well.

Before model loading, verify required inputs, available disk, output
disjointness, checkpoint SHA, and resume artifact integrity. Flush each completed
row atomically to a verified resumable checkpoint, release batch tensors, empty
the MPS cache, and keep the final embedding artifact absent until all 45 unique
rows verify. A guarded stop returns exit code 75, syncs the resource log,
terminates and reaps the process group, removes unverified partial output, and
does not run the probe.

Expected local cost for full MViT/Swin re-extraction is approximately 12–25
minutes and about 1 GiB peak process-tree RSS; this is an estimate, not an
acceptance threshold.

## Implemented result

The additive v2 chain was materialized without changing the frozen v1 files.
Its export resolves 48 selected rows into 45 determinate rows (`17+/28-`) and
keeps the three uncertain IDs explicit and excluded. The source-group manifest
has internal/file receipts `029cb1b4c56eebe2cf36495618e12966944883e52a706a5bb05feb0e81e674b0` /
`d48f990790b57b11389186b407078f52fce3d34a6410a9101001cd5d3221c05e`.
The export, four-video manifest, and frozen probe-plan internal/file receipts
are respectively:

- `9dbbffc81828d053154dd2e3adeadfb80869e5e1d7c6a03659df5ebc38fd7b35` /
  `6fd7680694122ac878154886550de6d0070ec32985ae37e038e894cd86b359aa`;
- `ddf1af3b23224b7fb99e96cb06226938686062cee5bf28bef153822f23847aba` /
  `e8261c550bf30da4b36d9b6773717769775e829d3c3c26f7cd750d2d7407ac3b`;
- `de1444c74672a827af2660a825af1d2792a4c8f61fdf47558ddc372261290333` /
  `78cc7dc64ea0089a431d362485ed85faee2b75863d797492c8a787d1420729c2`.

The plan froze Swin3D-T alone and MViT-V2-S + Swin3D-T, train-split-only
standardization, no PCA, `C=0.01`, and thresholds from `0.00` through `1.00`
in `0.05` increments. Both retained Kinetics-400 checkpoints produced 45
768-dimensional rows. MViT internal/file receipts are
`d6aa1d9bbb478972d5880bf944da4b0fa9ef1d7f1aa1640f79e76281c23b201f` /
`01dbd913a5fdda3a7b9173cbd2509aa191edf50d1010d00785a0c52a9d3b9a51`;
Swin internal/file receipts are
`3dc7caacfbc2cb3d8d8df856094131bde4fab3c3615811f363725490b7fea40b` /
`1c849096276c2e60f9d01cc064f54aa7426128634e85e7288faf526e76b031b5`.
Successful publication removed both resumable staging files.

The four-fold nested probe is a 254,401-byte artifact with internal/file
receipts `92d5d8045b2f39834eac8c57a662343f98cb9ba9e2049e2edbabe704c398ae21` /
`7d1c763656f3e940f6ca9db1bdad4e40f1edb0112d756fd549183fb68008d720`.
Pooled TP/FP/FN/TN is `14/4/3/24`, giving precision `0.777778`, recall
`0.823529`, and F1 `0.800000`. Per-game precision/recall is Hazen `1/1`,
Randolph `1/0.75`, VTV `0.75/0.75`, and Harwood `0.4/0.666667`.
HCTV `0.785714/0.846154` and VTV `0.75/0.75` are descriptive aggregates only.

These results do not change the design boundary: the 22 parent rows are
prior-stratified, the 23 Harwood rows are geometry-only/label-hidden but not
exhaustive continuous truth, and there are only two production families.
Every runtime/formal/promotion/promoted flag is false, readiness remains
`not_ready`, blind inference remains paused, and the formal 85% gate is
unchanged.

## Alternatives rejected

### Modify v1 constants from three sources/22 rows to four sources/45 rows

Rejected because it changes observable v1 behavior, invalidates frozen hashes,
and makes existing TASK-0254/TASK-0255 evidence unreplayable.

### Copy all parent rows and files into a standalone v2 dataset

Rejected because duplicate bytes can drift, provenance becomes ambiguous, and
verifiers may accidentally trust the copy rather than the frozen parent.

### Treat each game as a distinct production domain

Rejected because Hazen, Randolph, and Harwood share the HCTV production family.
This would overstate independence and could create a false promotion claim.

### Tune on the four outer-held results

Rejected as label leakage. All choices are frozen or selected by inner
game-held predictions inside the relevant outer-training partition.

## Documentation and review boundary

Implementation must add development, testing, and code-review records and
update the normal task/dataset/project memory files in the owning task, but this
spec phase does not claim those steps are complete. Fresh-context adversarial
review is mandatory before archiving. Cross-model interactive review may be
useful, but it remains pending explicit user authorization and must not be
called automatically.
