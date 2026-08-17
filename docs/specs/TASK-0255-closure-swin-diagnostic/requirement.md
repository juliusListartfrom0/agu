# TASK-0255 Requirement

## Objective

Determine whether the weak cross-source recall observed with the retained MViT
representation is specific to that backbone or persists with an independently
pretrained video encoder. Reuse the already retained official torchvision
Swin3D-T checkpoint and the frozen TASK-0254 22-row training export.

## Assumptions

- The 22 rows remain prior-stratified development data, not a formal benchmark.
- The checkpoint is the existing local
  `model_checkpoints/swin3d_t-7615ae03.pth`, with SHA-256
  `7615ae035996b65eb38dad437ae533d2dfcd36f9f89d28c0f0fa7bfb8e6b3130`.
- No threshold, regularization, sampling, or representation choice is tuned
  after seeing an outer-held label.

## Commands and artifacts

- Extract one 768-dimensional Swin3D-T embedding per verified example through
  `scripts/extract_shot_validity_video_embeddings.py` under
  `scripts/run_guarded_training.py`.
- Run exactly two predeclared non-promotable probes:
  1. Swin3D-T alone (768 dimensions).
  2. MViT-V2-S plus Swin3D-T concatenated (1,536 dimensions).
- Store the embedding, resource JSONL, and two probe JSON files beside the
  TASK-0254 artifacts in
  `analysis_outputs/public_research/vru_causal_closure_training_v1/`.

## Testing strategy

- Reuse the verified TASK-0254 export, manifest, embedding, and probe contracts.
- Require exact three-part-key, label, source, manifest, sampling, checkpoint,
  and external export-receipt binding.
- Re-run both probe verifiers, artifact SHA checks, focused tests, Ruff, and
  resource/process checks after extraction.

## Boundaries

- Always use canonical `.venv`, batch size one, MPS cache release, and the
  sustained CPU/memory/swap guard.
- Never use the two outer-held results to authorize model selection, runtime,
  default-pointer, readiness, or blind-inference changes.
- Never map the two excluded `uncertain` rows into either class.
- Stop safely with exit code 75 and resume from the verified checkpoint if a
  resource threshold is sustained.

## Success criteria

- A complete, sealed 22-row Swin3D-T embedding artifact and resource log exist.
- Both predeclared nested source-held probes verify and report pooled/per-source
  metrics while keeping `runtime/formal/promotion/promoted=false`.
- The result identifies whether independent representation or calibration is
  the more likely next bottleneck without claiming the 85% gate is closed.

## Open questions

- Whether Swin3D-T improves held-source recall enough to justify a later,
  independently frozen fusion contract.
- Whether a future formal evaluation set can be obtained from rights-cleared,
  continuous, exhaustive full-game labels rather than this stratified pilot.
