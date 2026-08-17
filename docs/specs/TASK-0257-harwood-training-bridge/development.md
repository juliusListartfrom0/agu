# TASK-0257 Development

## Additive chain

The implementation added a v2 receipt graph beside the frozen three-game v1
chain. It never rewrites the v1 index or its six child files. The v2 export
atomically publishes exactly three files in an immutable generation directory:
the index, Harwood candidate bundle, and Harwood binary labels. A complete
byte-identical target is a no-op; partial, drifted, extra-file, symlink, hardlink,
case-fold, ancestor, or descendant aliases fail closed.

The source-group manifest explicitly maps Hazen, Randolph, and Harwood to HCTV
and VTV to VTV. The verified resolved view contains 45 rows (`17+/28-`) from 48
selected reviews; the three uncertain rows are excluded. The four-video
training manifest binds the v2 index, all four label files, the four exact
source videos, task type `shot_validity`, and a disjoint benchmark receipt.

## Probe plan and extraction

The probe plan was frozen before extraction and held-label evaluation. It
declares two variants in order: Swin3D-T alone and MViT-V2-S + Swin3D-T. Each
inner fit uses train-split-only `StandardScaler`, no PCA, `C=0.01`, and a
threshold grid from `0.00` through `1.00` in `0.05` increments. The outer
protocol holds out exactly one game and confines all choices to inner
leave-one-game-out predictions from the remaining three games.

Both already-retained torchvision Kinetics-400 checkpoints were used with 16
frames, batch size one, MPS, preflight thresholds, and the sustained resource
guard. Each final embedding artifact has 45 rows of dimension 768. Successful
atomic publication removed the MViT and Swin staging checkpoints; neither
staging file remains.

## Reproducible command forms

The commands below are the normalized replay forms used by the implementation.
All paths are repository-relative and all Python commands use canonical
`.venv` Python 3.11.

```zsh
V2=analysis_outputs/public_research/vru_causal_harwood_training_v2
V1=analysis_outputs/public_research/vru_causal_closure_training_v1
HW=analysis_outputs/public_research/wikimedia_hctv_harwood_causal_review_v2

.venv/bin/python scripts/seal_vru_causal_source_groups.py \
  --spec "$V2/source_groups_spec.json" \
  --output "$V2/source_groups.json"

.venv/bin/python scripts/export_vru_causal_shot_validity_training_v2.py \
  --parent-export "$V1/training_export.json" \
  --expected-parent-artifact-sha256 1b0bcef2c67f0d0e1fa35b79042584c43d938f05e893e89a4a64266ab9d0e464 \
  --expected-parent-file-sha256 afb406ef84ae57b8d6cbe565a848b03bcefc2fdd449d8c223374e04d7f7dc3e0 \
  --parent-asset-root "$V1" \
  --harwood-selection "$HW/selection.json" \
  --expected-harwood-selection-artifact-sha256 5a7eaa1ecefa295c5d97632e05e7aea2187eafac0082ba6ea6341fc6b372a80f \
  --harwood-review-plan "$HW/review_plan.json" \
  --expected-harwood-review-plan-artifact-sha256 87c9b92c548000527f253082c6190ec82022934d9c3db244be32e50505244911 \
  --harwood-sealed-review "$HW/sealed_review.json" \
  --expected-harwood-sealed-review-artifact-sha256 5f2219a4cdcc7e8d4e40dde060c8ba82b5070eb6db471c1ce2855cea9d3e5cb6 \
  --harwood-raw-frame-manifest "$HW/raw_frames_manifest.json" \
  --expected-harwood-raw-frame-manifest-artifact-sha256 0757e339ffd5acb97d5b8f42790cfced2de825e44deea20095abd7bedfeae776 \
  --harwood-source-manifest "$HW/source_manifest.json" \
  --source-groups "$V2/source_groups.json" \
  --expected-source-groups-artifact-sha256 029cb1b4c56eebe2cf36495618e12966944883e52a706a5bb05feb0e81e674b0 \
  --output "$V2/export_v2/training_export.json"
```

The four-video manifest was sealed with producer
`codex-assisted-offline-causal-review-v2`, all four retained source videos, the
v2 export plus four label files, task type `shot_validity`, and
`analysis_outputs/public_research/mem_okc_benchmark_game6/empty_raw_candidate_bundle.json`.

```zsh
.venv/bin/python scripts/seal_training_annotation_manifest.py \
  --producer codex-assisted-offline-causal-review-v2 \
  --source-video dataset/public_sources/wikimedia_hctv_fullgame_v1/raw/hctv_hazen_lyndon_2023.webm \
  --source-video dataset/public_sources/wikimedia_hctv_randolph_v1/raw/hctv_randolph_2026.webm \
  --source-video dataset/public_sources/wikimedia_vtv_fullgame_v1/raw/vtv_spartans_trotamundos_2026.webm \
  --source-video dataset/public_sources/wikimedia_hctv_harwood_v1/raw/hctv_harwood_2026.webm \
  --annotation "$V2/export_v2/training_export.json" \
  --annotation "$V1/hazen_shot_validity_labels_v1.json" \
  --annotation "$V1/randolph_shot_validity_labels_v1.json" \
  --annotation "$V1/vtv_shot_validity_labels_v1.json" \
  --annotation "$V2/export_v2/harwood_shot_validity_labels_v2.json" \
  --task-type shot_validity \
  --benchmark-bundle analysis_outputs/public_research/mem_okc_benchmark_game6/empty_raw_candidate_bundle.json \
  --output "$V2/training_manifest.json"
```

The common extraction inputs were the training manifest; the four candidate
bundles, labels, and videos in Hazen/Randolph/VTV/Harwood order; manifest
internal/file receipts
`ddf1af3b23224b7fb99e96cb06226938686062cee5bf28bef153822f23847aba` /
`e8261c550bf30da4b36d9b6773717769775e829d3c3c26f7cd750d2d7407ac3b`;
and resource preflight values `90%` memory, `2 GiB` available, and `2 GiB`
output free space. Each invocation used this guarded form:

```zsh
.venv/bin/python scripts/run_guarded_training.py \
  --max-memory-percent 90 --min-available-gib 2 --min-free-swap-gib 0.25 \
  --max-cpu-percent 95 --consecutive-breaches 3 --sample-interval-sec 2 \
  --log "$V2/<backbone>_embedding_resource_guard.jsonl" -- \
  .venv/bin/python scripts/extract_shot_validity_video_embeddings.py \
  --manifest "$V2/training_manifest.json" \
  --candidate-bundle "$V1/hazen_candidate_bundle_v1.json" \
  --candidate-bundle "$V1/randolph_candidate_bundle_v1.json" \
  --candidate-bundle "$V1/vtv_candidate_bundle_v1.json" \
  --candidate-bundle "$V2/export_v2/harwood_candidate_bundle_v2.json" \
  --annotation "$V1/hazen_shot_validity_labels_v1.json" \
  --annotation "$V1/randolph_shot_validity_labels_v1.json" \
  --annotation "$V1/vtv_shot_validity_labels_v1.json" \
  --annotation "$V2/export_v2/harwood_shot_validity_labels_v2.json" \
  --video dataset/public_sources/wikimedia_hctv_fullgame_v1/raw/hctv_hazen_lyndon_2023.webm \
  --video dataset/public_sources/wikimedia_hctv_randolph_v1/raw/hctv_randolph_2026.webm \
  --video dataset/public_sources/wikimedia_vtv_fullgame_v1/raw/vtv_spartans_trotamundos_2026.webm \
  --video dataset/public_sources/wikimedia_hctv_harwood_v1/raw/hctv_harwood_2026.webm \
  --backbone <torchvision-backbone> \
  --backbone-checkpoint <verified-local-checkpoint> \
  --expected-backbone-sha256 <checkpoint-sha256> \
  --expected-training-manifest-sha256 ddf1af3b23224b7fb99e96cb06226938686062cee5bf28bef153822f23847aba \
  --expected-training-manifest-file-sha256 e8261c550bf30da4b36d9b6773717769775e829d3c3c26f7cd750d2d7407ac3b \
  --resume-checkpoint "$V2/<backbone>_embeddings.staging.json" \
  --output "$V2/<backbone>_embeddings.json" \
  --clip-frames 16 --batch-size 1 --device mps \
  --max-memory-percent 90 --min-available-gib 2 --min-output-free-gib 2
```

The two substitutions were
`torchvision/mvit_v2_s/kinetics400_v1`,
`model_checkpoints/mvit_v2_s-ae3be167.pth`, SHA
`ae3be16733081f6d1cd40e4ab980ca23d6df6dc6486d15ada05a5e8ab8c9b975`;
and `torchvision/swin3d_t/kinetics400_v1`,
`model_checkpoints/swin3d_t-7615ae03.pth`, SHA
`7615ae035996b65eb38dad437ae533d2dfcd36f9f89d28c0f0fa7bfb8e6b3130`.

The nested probe CLI was invoked with both embedding paths and their external
internal/file receipts, the v2 export internal/file receipts, the complete v1
and Harwood chain arguments shown above, source-group internal/file receipts,
training-manifest internal/file receipts, frozen plan internal/file receipts,
and output `$V2/nested_probe_v2.json`:

```zsh
.venv/bin/python scripts/screen_vru_causal_video_embeddings_nested_v2.py \
  --embeddings "$V2/mvit_v2_s_embeddings.json" \
  --expected-embedding-artifact-sha256 d6aa1d9bbb478972d5880bf944da4b0fa9ef1d7f1aa1640f79e76281c23b201f \
  --expected-embedding-file-sha256 01dbd913a5fdda3a7b9173cbd2509aa191edf50d1010d00785a0c52a9d3b9a51 \
  --embeddings "$V2/swin3d_t_embeddings.json" \
  --expected-embedding-artifact-sha256 3dc7caacfbc2cb3d8d8df856094131bde4fab3c3615811f363725490b7fea40b \
  --expected-embedding-file-sha256 1c849096276c2e60f9d01cc064f54aa7426128634e85e7288faf526e76b031b5 \
  --training-export "$V2/export_v2/training_export.json" \
  --expected-training-export-artifact-sha256 9dbbffc81828d053154dd2e3adeadfb80869e5e1d7c6a03659df5ebc38fd7b35 \
  --expected-training-export-file-sha256 6fd7680694122ac878154886550de6d0070ec32985ae37e038e894cd86b359aa \
  --extension-asset-root "$V2/export_v2" \
  --parent-export "$V1/training_export.json" \
  --expected-parent-artifact-sha256 1b0bcef2c67f0d0e1fa35b79042584c43d938f05e893e89a4a64266ab9d0e464 \
  --expected-parent-file-sha256 afb406ef84ae57b8d6cbe565a848b03bcefc2fdd449d8c223374e04d7f7dc3e0 \
  --parent-asset-root "$V1" \
  --harwood-selection "$HW/selection.json" \
  --expected-harwood-selection-artifact-sha256 5a7eaa1ecefa295c5d97632e05e7aea2187eafac0082ba6ea6341fc6b372a80f \
  --harwood-review-plan "$HW/review_plan.json" \
  --expected-harwood-review-plan-artifact-sha256 87c9b92c548000527f253082c6190ec82022934d9c3db244be32e50505244911 \
  --harwood-sealed-review "$HW/sealed_review.json" \
  --expected-harwood-sealed-review-artifact-sha256 5f2219a4cdcc7e8d4e40dde060c8ba82b5070eb6db471c1ce2855cea9d3e5cb6 \
  --harwood-raw-frame-manifest "$HW/raw_frames_manifest.json" \
  --expected-harwood-raw-frame-manifest-artifact-sha256 0757e339ffd5acb97d5b8f42790cfced2de825e44deea20095abd7bedfeae776 \
  --harwood-source-manifest "$HW/source_manifest.json" \
  --source-groups "$V2/source_groups.json" \
  --expected-source-groups-artifact-sha256 029cb1b4c56eebe2cf36495618e12966944883e52a706a5bb05feb0e81e674b0 \
  --expected-source-groups-file-sha256 d48f990790b57b11389186b407078f52fce3d34a6410a9101001cd5d3221c05e \
  --training-manifest "$V2/training_manifest.json" \
  --expected-training-manifest-sha256 ddf1af3b23224b7fb99e96cb06226938686062cee5bf28bef153822f23847aba \
  --expected-training-manifest-file-sha256 e8261c550bf30da4b36d9b6773717769775e829d3c3c26f7cd750d2d7407ac3b \
  --plan "$V2/probe_plan.json" \
  --expected-plan-artifact-sha256 de1444c74672a827af2660a825af1d2792a4c8f61fdf47558ddc372261290333 \
  --expected-plan-file-sha256 78cc7dc64ea0089a431d362485ed85faee2b75863d797492c8a787d1420729c2 \
  --output "$V2/nested_probe_v2.json"
```

The CLI requires every one of those receipts; it cannot consume a
hand-constructed or partially verified chain.

## Resource observations

| Run | Samples | Max memory | Max CPU | Peak tree RSS | Min available | Min free swap | Sustained breach |
|---|---:|---:|---:|---:|---:|---:|---:|
| MViT-V2-S | 63 | 84.5% | 94.0% | 0.9756 GiB | 2.474 GiB | 0.4233 GiB | 0 |
| Swin3D-T | 52 | 84.1% | 83.2% | 1.0922 GiB | 2.543 GiB | 0.4940 GiB | 0 |

Both guards finished with exit code zero. The guarded full regression also
finished with exit code zero: 27 samples, max memory 78.6%, max CPU 40.8%, peak
tree RSS 1.24884 GiB, min available 3.43158 GiB, min free swap 0.42279 GiB,
and zero sustained breaches.

## Runtime boundary

No checkpoint was trained or promoted. No default pointer, runtime API,
readiness record, or blind-inference state changed. Cross-model interactive
review was not run because the user did not explicitly authorize it.
