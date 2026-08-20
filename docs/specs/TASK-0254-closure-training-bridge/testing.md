# TASK-0254 Testing

## TDD evidence

- Export contract: initial module-missing RED, then full-chain mapping,
  tampering, path, alias, and deterministic-byte GREEN; final file has 25 tests.
- Embedding CLI safety: 18 initial alias/atomic RED cases, then expanded
  hardlink/case-only coverage; final direct file has 28 tests.
- Nested probe: held-label invariance, external bindings, duplicate receipts,
  metrics reconciliation, and CLI safety; final direct file has 36 tests.
- Resource guard: log-open/write/fsync and sampling failures reproduced child
  lifecycle leaks; final direct file has 13 tests.
- Training-manifest CLI: eight alias/atomic failures reproduced before GREEN;
  final direct file has 13 tests.

## Final verification

- Targeted closure/export/embedding/probe/guard suite: `114 passed`.
- Full canonical `.venv` suite after all safety changes: `1392 passed`, with 15
  pre-existing sklearn/Starlette warnings and no failures.
- Target Ruff, `pip check`, CLI help, `git diff --check`, and untracked-file
  whitespace checks passed.
- Real export was rebuilt in memory: index object, pretty bytes, and all six
  child files were byte-identical. Real training manifest was regenerated in a
  temporary directory and remained 1,727 bytes with the same file SHA.
- `README.md` and `docs/api.md` were reviewed. No update or local-service curl
  hook is required because this slice adds only offline training/diagnostic
  contracts and does not alter the service/API/runtime.

## Diagnostic result

The MViT nested probe reports TP/FP/FN/TN `2/1/12/7`, pooled
precision/recall/F1 `0.666667/0.142857/0.235294`. Per-source recall is
`0`, `0.25`, and `0.25`; therefore the 85% gate fails.
