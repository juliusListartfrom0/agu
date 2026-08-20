# TASK-0255 Solution

## Fixed experiment

The experiment reuses the existing TASK-0254 exporter, training manifest,
resource guard, video embedding extractor, and nested source-held probe without
changing model code or hyperparameters. The only new model input is the local
official Swin3D-T checkpoint.

## Evaluation order

1. Verify the frozen export, manifest, three child pairs, three source videos,
   and Swin checkpoint before extraction.
2. Extract 16-frame, batch-one Swin3D-T embeddings under the sustained guard.
3. Verify the complete embedding artifact and resource log.
4. Produce the Swin-only probe.
5. Produce the predeclared MViT+Swin concatenated probe.
6. Compare observed metrics only as a development diagnostic; neither result
   is eligible for selection or promotion.

## Decision rule

- If both probes retain low held-source recall, prioritize independent causal
  truth and calibration rather than adding more generic video backbones.
- If Swin materially improves every source, freeze a separate future fusion
  contract before any broader evaluation.
- In all cases, readiness remains `not_ready` until an independent continuous
  full-game gate reaches precision and recall of at least 0.85 per source.
