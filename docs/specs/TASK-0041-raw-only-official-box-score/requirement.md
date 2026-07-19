# TASK-0041 Requirement

## Objective

Implement the six phases defined by `docs/complete-box-score-plan.md` and prove that AGU can produce auditable player statistics when inference receives raw game video only.

## Six delivery areas

1. Raw-only benchmark, immutable event/review/official-box-score contracts, and leakage audit.
2. Ball, rim/backboard, court and pose adapter contracts plus temporal tracks.
3. Possession and shot state machines, including release, made/missed and 2/3-point classification.
4. Rebound, block, turnover/steal, foul and assist event relationships.
5. Full-game player identity, team/jersey/roster binding and an append-only official event ledger.
6. Score reconciliation, isolated Codex annotation/acceptance packages,
   deterministic AGU-only aggregation and two-game blind evaluation.

## Raw-only rule

The inference command may read only declared raw video files and versioned model/configuration assets that are shared across games. Reference CSV files, edited highlights/misses/errors, exported box scores, ground-truth events, and their paths or hashes must not be present in the inference manifest or process arguments. Ground truth is opened only by a separate evaluation step after predictions have been finalized and hashed.

## Acceptance criteria

- Preserve the v3 action preprocessing contract and all existing API behavior.
- Every official count traces to an accepted event revision and evidence range.
- Strict event evaluation requires event type, time, primary player, outcome and shot value whenever those fields apply.
- Report precision, recall and F1 per event type and per game; do not substitute field-match rate, scene localization, scoreboard accuracy or pooled accuracy.
- Each of two games must achieve strict micro F1 >= 0.85. Also report macro F1, actor accuracy, box-score exact-field accuracy, score reconciliation and unresolved workload.
- Predictions must be finalized before the evaluator reads reference truth, with SHA-256 provenance proving the boundary.
- Candidate, automatic-confirmed and Codex/human-reviewed metrics are reported separately.
- Codex/human-reviewed metrics are audit-only and cannot satisfy the autonomous
  85% acceptance gate.
- Codex may create detector, tracker, OCR or action-model training annotations
  as a substitute for manual labeling. Eligible autonomous models must bind
  those annotations to a SHA-256 training manifest and declare zero raw-video
  hash overlap with acceptance games. The annotations are training inputs only:
  they cannot be loaded as runtime event/statistic sidecars or provide official
  event answers to AGU.

## Available validation assets

- Game A: six raw MOV files in `20260128_原片`; complete exported reference package in `20260128_早鸟jq3`.
- Game B: `第一节.mov`; 17 manually labeled events and ten-player roster. This is a one-quarter, partial-category benchmark, not yet a complete second-game truth set.
- Nine `2026-07-04` MOV files have scoreboard validation but no complete player event truth, so they are calibration/smoke assets only.

## Non-scope

- Reference-assisted inference advertised as independent recognition.
- Hard-coded private paths in service or library code.
- A new language stack or BFF behavior.
- Silent conversion of action clips or scoreboard deltas into official player statistics.

## Context read

Pre-development context came from the llm-wiki pages `agu-complete-box-score-plan-2026-07-13`, `agu-ground-truth-baseline-evaluator-2026-07-02`, and `agu-reference-assisted-game-audit-2026-07-15`, plus the current code and task board.
