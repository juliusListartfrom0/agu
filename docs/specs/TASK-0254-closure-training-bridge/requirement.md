# TASK-0254 Requirement

## Objective

Convert the sealed 24-window causal-closure review into a separate, explicit,
SHA-bound, training-only derivative and run a resource-guarded MViT
representation diagnostic with nested source-held evaluation.

## Scope

- Verify the frozen selection, review plan, sealed review, raw-frame manifest,
  all retained review JPEGs, and the three original source videos.
- Export label-hidden candidate geometry plus binary shot-validity labels.
- Map `shot` to positive and `not_a_shot` to negative; exclude `uncertain`.
- Bind the derivative labels through a training annotation manifest.
- Extract fixed MViT embeddings and evaluate them with nested source-held folds.

## Non-scope

- No ExtraTrees training from synthetic/default detector features.
- No made/missed outcome model.
- No runtime/default/checkpoint/readiness change and no blind-inference resume.
- No claim that the prior-stratified 24-window sample is a formal 85% benchmark.

## Acceptance

- Exactly 22 examples: 14 positive, 8 negative, and two named exclusions.
- Candidate geometry contains no review label, outcome, notes, or reviewed
  release/rim anchors.
- Every source/provenance or frame mismatch fails closed.
- The selection, review plan, sealed review, and raw-frame manifest each require
  a caller-supplied, externally frozen internal SHA-256 receipt.
- Output artifacts are atomic, input-disjoint, reproducible, and self-verifying.
- Child asset filenames are globally unique, and direct, symlink, hardlink, or
  case-only output aliases are rejected before reads or model loading.
- Resource-guarded nested OOF is recorded as development diagnostic only.

## Context

The llm-wiki pre-hook used the causal-closure selection-freeze, continuous
annotation, base-model audit, and nested-fusion records. Repository code and
sealed artifact bytes remain authoritative.
