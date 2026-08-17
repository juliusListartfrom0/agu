# TASK-0256 Requirement — HCTV Harwood fourth-game development expansion

## Objective

Add one previously unseen, rights-recorded continuous basketball game as a fourth
game-held development group. The slice must increase real label-hidden causal
training evidence without re-tuning the existing 22-row held labels.

## Assumptions

1. The selected Commons payload is the complete HCTV Hazen–Harwood game from
   2026-01-22, not a clip: `4,125.088` seconds and `1,376,342,882` bytes.
2. Commons declares CC BY 4.0 and the original YouTube metadata independently
   declares `Creative Commons Attribution license (reuse allowed)`.
3. HCTV–Harwood is a new game but not a new production domain. It may improve
   development training and game-held diagnostics, but cannot close the
   held-production or formal 85% gate by itself.

## Stack and commands

- Environment: `.venv` / Python 3.11 only.
- Tests: `.venv/bin/python -m pytest -q <focused tests>`.
- Lint: `.venv/bin/python -m ruff check <changed Python files>`.
- Media inspection: `ffprobe`; download from the canonical Commons file URL
  with resume support and exact byte/SHA verification.

## Project structure

- Source manifest/media: `dataset/public_sources/wikimedia_hctv_harwood_v1/`.
- Selection/review artifacts: `analysis_outputs/public_research/` under a
  dedicated HCTV Harwood pilot directory.
- Selection logic: `app/analysis/` with a thin CLI in `scripts/`.
- Tests: `tests/`.

## Testing strategy

- RED before GREEN for deterministic selection and verifier behavior.
- Verify labels, model scores, PBP, outcome, review notes, and prior AGU answers
  cannot influence the frozen selection.
- Verify exactly 24 non-overlapping eight-second windows, eight per temporal
  third, sampled at 8 FPS with 64 half-open frames.
- Later review/export tests must preserve uncertain as excluded and keep every
  derivative non-runtime and non-promotable.

## Boundaries

- Always: bind the source video, license metadata, selection, plan, materialized
  frames, review, and any derivative by SHA-256.
- Always: freeze the full 24-window batch before any frame is reviewed.
- Never: use Codex review decisions as AGU runtime answers or VLM predictions.
- Never: select windows from labels, PBP, current model probabilities, or
  outcome/release annotations.
- Never: update runtime defaults, readiness, or blind-inference state from this
  same-production development game.

## Success criteria

1. The canonical payload matches the official declared byte count and a locally
   computed SHA-256; `ffprobe` confirms continuous long-form video.
2. A deterministic, exact-schema selection artifact freezes 24 label-hidden,
   non-overlapping windows with no target or model fields.
3. The review chain can be materialized and sealed using existing strict causal
   review contracts.
4. Any subsequent four-game nested diagnostic remains development-only and
   reports every held game separately.
