# TASK-0254 Development

## Implemented boundary

- Added `agu.vru-causal-shot-validity-training-export.v1` and its public
  verifier. The builder validates the frozen selection, review plan, sealed
  review, raw-frame manifest, 1,536 JPEGs, source manifest, and three full
  source videos.
- Required external expected internal SHA-256 values for the selection, plan,
  sealed review, and raw-frame manifest. These receipts are caller inputs and
  are never inferred from the artifacts being verified.
- Exported 24 label-hidden geometry rows but only 22 binary label/index rows:
  `shot -> true`, `not_a_shot -> false`; the two `uncertain` rows are excluded.
  Reviewed outcome, release, rim, notes, and candidate-event anchors never
  enter the training geometry.
- Reused the training-annotation manifest for three-source and acceptance-video
  disjointness. Traditional ExtraTrees training rejects the synthetic geometry
  before feature extraction; only scene/video representation diagnostics are
  authorized.
- Added the fixed-C nested source-held probe. Each outer-held source is absent
  from inner threshold selection and model fitting, and all outputs are
  permanently runtime/formal/promotion ineligible.

## File and process safety

- Export, embedding, probe, and training-manifest CLIs reject direct, symlink,
  hardlink, and case-only output aliases before destructive work. They freeze
  resolved paths and use same-directory random temporary files, file `fsync`,
  and atomic replace.
- The resource guard opens and syncs its log before/through supervision and
  always terminates and reaps the child process group on logging, sampling, or
  interrupt failures.
- MPS batch tensors are released and the accelerator cache is emptied after
  each embedding flush.

## Real artifacts

- Training export: internal `1b0bcef2c67f0d0e1fa35b79042584c43d938f05e893e89a4a64266ab9d0e464`,
  file `afb406ef84ae57b8d6cbe565a848b03bcefc2fdd449d8c223374e04d7f7dc3e0`.
- Training manifest: internal `56cbcda7ab0b1586e5284d40d0010c02f4c26042ca95aa78187e3462e44f71e0`,
  file `cd9a765b8fbd3111d94d79021f43a3c876e90a573eb4c6d175cf262e2bf51b43`.
- MViT embeddings: internal `0572f7d897b8ca16a955b2bfca36177f1c5a16f2121ee87da2b6c88ac5bab405`,
  file `90858d6aa3702e3e52ed0746ff35bd31059b7364d920d4818b4ac9e00a41bc6e`.
- MViT nested probe: internal `d2039499c1d8455fdbf25c3d50f8b959475a4e8668acc49240bff91e757cf177`,
  file `4e10f28aab6ee5aff1a72325d39b3bb3c454752e5ccb55802e04ee86f2c02120`.

No deployable checkpoint or runtime pointer was produced.
