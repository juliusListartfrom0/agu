# TASK-0254 Gate Review

## Decision

PASS for a development-only vertical slice.

## Reasons

- Project documentation explicitly permits closure labels to enter a later
  offline experiment only through an explicit SHA-bound training manifest.
- All three sources contain both binary classes after excluding uncertain rows,
  so nested source-held diagnostics are mechanically possible.
- The retained MViT checkpoint avoids a new download and is the strongest
  currently observed local representation path.

## Risks and controls

- Prior-label stratification and only three sources make the result non-formal:
  force formal evaluation and runtime flags false.
- Synthetic geometry has no real detector features: disallow the traditional
  ExtraTrees route.
- Reviewer positions can leak labels: never export release/rim/outcome/notes or
  use them as sampling anchors.
- Tiny folds invite threshold leakage: select threshold and any variant only in
  inner folds, then evaluate the untouched outer source.
- Local training may pressure memory/CPU: use `.venv`, batch size one, and the
  sustained resource guard.
- Self-resealed upstream labels or frames can counterfeit provenance: require
  independent expected SHA values for selection, plan, review, and raw frames.
- Artifact writers can overwrite evidence through filesystem aliases: compare
  resolved names, case-folded names, and existing `(device, inode)` identities,
  then write through same-directory random temporaries and atomic replace.
