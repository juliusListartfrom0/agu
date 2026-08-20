# TASK-0254 Solution

## Approach

1. Add an owning analysis module for the closure-to-training export contract.
2. Add one CLI that validates the full evidence chain before writing a batch
   index, three raw-only geometry bundles, and three binary label files.
3. Reuse the existing training annotation manifest sealer for benchmark
   disjointness.
4. Reuse the retained local MViT checkpoint and existing video embedding
   extractor; do not download another backbone.
5. Add a nested outer-source/inner-source linear probe whose threshold and
   model selection never read the outer-held labels.
6. Require external receipts for the plan, review, and raw-frame manifest in
   addition to the existing selection receipt. Keep the output schema stable.

## Data boundary

The source selection remains non-training-consumable. The exporter is the only
authorized boundary: it records the source flag, the explicit derivative
policy, every input SHA, permitted downstream routes, and formal-evaluation
ineligibility. Outcomes and reviewed causal positions stay only in the sealed
review and are never projected into candidate geometry.

## Verification

- Focused RED/GREEN tests for contract, mapping, tampering, path safety, and
  held-label invariance.
- Real artifact rebuild and verifier replay.
- Resource-guard log for embedding extraction.
- Focused and broader pytest, Ruff, and `git diff --check`.
- Fresh-context adversarial review and llm-wiki post-hook.
