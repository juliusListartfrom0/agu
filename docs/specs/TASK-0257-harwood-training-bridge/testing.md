# TASK-0257 Testing

## Frozen artifact receipts

| Artifact | Internal SHA-256 | File SHA-256 | Size |
|---|---|---|---:|
| source groups | `029cb1b4c56eebe2cf36495618e12966944883e52a706a5bb05feb0e81e674b0` | `d48f990790b57b11389186b407078f52fce3d34a6410a9101001cd5d3221c05e` | — |
| v2 export | `9dbbffc81828d053154dd2e3adeadfb80869e5e1d7c6a03659df5ebc38fd7b35` | `6fd7680694122ac878154886550de6d0070ec32985ae37e038e894cd86b359aa` | 6,020 bytes |
| training manifest | `ddf1af3b23224b7fb99e96cb06226938686062cee5bf28bef153822f23847aba` | `e8261c550bf30da4b36d9b6773717769775e829d3c3c26f7cd750d2d7407ac3b` | 2,081 bytes |
| probe plan | `de1444c74672a827af2660a825af1d2792a4c8f61fdf47558ddc372261290333` | `78cc7dc64ea0089a431d362485ed85faee2b75863d797492c8a787d1420729c2` | 2,645 bytes |
| MViT embeddings | `d6aa1d9bbb478972d5880bf944da4b0fa9ef1d7f1aa1640f79e76281c23b201f` | `01dbd913a5fdda3a7b9173cbd2509aa191edf50d1010d00785a0c52a9d3b9a51` | 1,038,316 bytes |
| Swin embeddings | `3dc7caacfbc2cb3d8d8df856094131bde4fab3c3615811f363725490b7fea40b` | `1c849096276c2e60f9d01cc064f54aa7426128634e85e7288faf526e76b031b5` | 1,032,110 bytes |
| nested probe | `92d5d8045b2f39834eac8c57a662343f98cb9ba9e2049e2edbabe704c398ae21` | `7d1c763656f3e940f6ca9db1bdad4e40f1edb0112d756fd549183fb68008d720` | 254,401 bytes |

The public verifiers replayed the complete v1 parent and six children, the
Harwood source/selection/plan/1,536-JPEG/raw/sealed chain, the four-video
manifest, both 45-row embedding joins, the frozen plan, and the final probe.
Fresh-context adversarial re-reviews returned PASS with no Critical or Required
finding.

## Automated regression

- Extractor and backbone contract suite:
  `.venv/bin/python -m pytest -q tests/test_extract_shot_validity_video_embeddings.py tests/test_shot_validity_video_backbone.py`
  -> `132 passed`.
- Broader v1/v2 export and probe suite:
  `.venv/bin/python -m pytest -q tests/test_vru_causal_video_probe_v2.py tests/test_vru_causal_training_export_v2.py tests/test_vru_causal_source_groups.py tests/test_vru_causal_training_export.py tests/test_vru_causal_video_probe.py`
  -> `205 passed`.
- Guarded full suite:
  `.venv/bin/python -m pytest -q` -> `1712 passed, 5 skipped, 15 warnings in
  52.03s`, exit code zero.
- Full-suite guard log:
  `task0257_full_regression_resource_guard.jsonl`, file SHA-256
  `0ae49af1f0e9bc58cf9c99a260ffd1cd2b3aea376c51153077531fa7157f50f6`;
  27 samples, max memory 78.6%, max CPU 40.8%, peak process-tree RSS
  1.24884 GiB, min available memory 3.43158 GiB, min free swap 0.42279 GiB,
  zero sustained breaches, exit code zero.
- TASK-0257 scoped Ruff check and format check passed. An additional
  repository-wide `ruff check .` / `ruff format --check .` diagnostic failed on
  substantial historical baseline debt outside TASK-0257; no unrelated mass
  formatting was performed.
- After the completed full and focused regressions, the inactive pytest scratch
  tree was the only large disposable target removed. The two `du` observations
  total `5,824,397,312` allocated bytes; APFS block sharing/asynchronous reclaim
  means this is not a claim of equal `df` recovery. The audits are
  `task0257_pytest_temp_cleanup.json` and
  `task0257_focused_pytest_temp_cleanup.json`, with file SHA-256
  `6c38ec66c9c1e941f43d174bf4b27e08f7a72e4f650df3d3a7ed86a396e4c302`
  and `11be66744e24d68b7eb4eacbff3793a135b8d7978929a631289ee7f71db26d9b`.
  The final observation was `4,226,068 KiB` free, above TASK-0258's 3.5 GiB
  disk preflight; source artifacts, `.venv`, and model weights were not touched.

The suites cover exact/missing/unknown fields, duplicate keys, non-finite JSON,
count drift, all input/output alias classes, crash-atomic generation, checkpoint
snapshot verification, staging/resume corruption, resource preflight and exit
75 behavior, complete-manifest enforcement, held-label selection invariance,
inner LOGO selection, and hard-false eligibility flags.

## Probe observations

The probe contains four outer game-held folds over 45 rows (`17+/28-`).

| Held game | TP/FP/FN/TN | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Hazen | `6/0/0/2` | 1.000000 | 1.000000 | 1.000000 |
| Randolph | `3/0/1/3` | 1.000000 | 0.750000 | 0.857143 |
| VTV | `3/1/1/2` | 0.750000 | 0.750000 | 0.750000 |
| Harwood | `2/3/1/17` | 0.400000 | 0.666667 | 0.500000 |
| **Pooled** | **`14/4/3/24`** | **0.777778** | **0.823529** | **0.800000** |

The HCTV descriptive aggregate has precision/recall
`0.785714/0.846154`; VTV has `0.750000/0.750000`. These are not
production-held estimates: HCTV contains three games while VTV contains only
one. The 22 parent rows are prior-stratified, the 23 Harwood rows are
geometry-only/label-hidden, and neither source is exhaustive continuous
full-game truth.

Every artifact remains non-runtime/non-formal/non-promotable. The 85% gate is
unchanged, readiness remains `not_ready`, and blind inference remains paused.
