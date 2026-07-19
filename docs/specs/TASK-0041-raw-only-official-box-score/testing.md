# TASK-0041 Testing

## 2026-07-19 image-bound cache, jersey consensus and enrollment enforcement

- Focused identity/official/candidate verification passed: `74 passed`; the
  enrollment/actor subset passed `59 passed`, and touched-file Ruff passed.
- Including `tests/test_hybrid_analysis.py` produced `170 passed, 3 failed`.
  All three failures are environment capability failures in the installed
  OpenCV build (`CascadeClassifier` and `legacy.MultiTracker_create` absent),
  outside the changed official identity/VLM/candidate paths.
- Fresh qwen3-vl:8b four-overlay diagnostic bundle
  `4e450fb06b1c47f78a0711fd6f89926340169a5c8e147ab7657955142cfb64b2`
  confirmed four of five candidates and rejected the made-shot rebound, but
  still assigned shot 3 to the foreground official.
- Fresh role-gated diagnostic bundle
  `3d9e60c201ac376db1a3dbd9b286e848586056f0067a5dbcc7d2c588d78838ac`
  proved the configured VLM returned `primary_actor_role=player` for that
  official. No accuracy promotion is claimed from the role prompt.
- The current identity artifact readiness audit reports 68 canonical identities
  and zero face-gallery anchored players. The new official enrollment gate
  therefore fails closed pending a benchmark-disjoint annotated face list.
- Harness verification with the focused official/identity suite, touched-file
  Ruff and `git diff --check` passed. Local service on port 8877 returned `ok`
  and `ready`; task `26c9c8aa5b654e5cbc8ff2dc0f38fb25` completed at 100% with
  no error. (`uv run` lacks the service-only uvicorn dependency in this
  workspace, so the documented `venv/bin/python` runtime was used.)

## 2026-07-19 annotated face gallery identity gate

- `uv run pytest -q tests/test_face_gallery.py tests/test_face_identity.py tests/test_identity_graph.py tests/test_official_identity.py tests/test_official_autonomous_inference.py tests/test_official_autonomous_script.py tests/test_official_event_candidates.py`
  passed: `63 passed`.
- `uv run python scripts/verify_harness.py --test-command "uv run pytest -q tests/test_face_gallery.py tests/test_face_identity.py tests/test_identity_graph.py tests/test_official_identity.py"`
  passed.
- Touched-file Ruff and `git diff --check` passed.
- Local service on port 8876 returned `ok` and `ready`; task
  `a4dcbf1b0da14dd496b2cf97b53877f5` completed at 100%.
- The repository contains the YuNet and SFace model assets but no legitimate
  benchmark-disjoint annotated face list for the 20260128 players. The existing
  team PNGs contain only statistics. No complete-game identity or six-stat
  accuracy claim is made until independent enrollment photos are provided.

## 2026-07-18 open-vocabulary/trajectory increment

- Focused perception, candidate and action-owner suite: 29 passed.
- Full regression with the OpenCV 4 compatibility override: 274 passed with 15
  existing warnings.
- Ruff passed all touched perception/candidate scripts, modules and tests;
  `scripts/verify_harness.py` passed.
- Local service on port 8793 returned `ok` and `ready`; a 60-frame VLM-off run
  returned pending task `584f4cac50aa45b7a0bb0ecd079a8b17` and status polling
  completed at progress 100 with no error.
- README and `docs/api.md` were checked. The changes add internal/offline
  adapter and candidate CLI options only; the public run/status contract and
  startup documentation did not change.
- Autonomous acceptance remains open: the new 11-action model is not integrated
  because cross-video Top-1 is below 0.85, and neither complete benchmark game
  has independently passed the strict raw-only gate.

## Interim verification (2026-07-16)

- `venv/bin/python -m pytest tests/test_perception_adapters.py tests/test_official_stats.py tests/test_game_state.py tests/test_identity_graph.py -q`: 20 passed.
- `venv/bin/python -m pytest`: 181 passed, 15 pre-existing warning messages.
- `venv/bin/python scripts/verify_harness.py`: passed.
- Ruff over all new/changed Python modules and tests: passed.
- Local service on port 8791: `/health` returned `ok`, `/ready` returned
  `ready`, `POST /api/v1/analysis/run` returned the expected pending task shape,
  and task `d74a0d962d2f4de1995dd9fac67a77d6` completed at progress 100.
- First-quarter strict legacy baseline: TP 0 / FP 648 / FN 17 / F1 0.0 with a
  90-frame tolerance.
- Same 50-frame detector sample: COCO two ball detections; E-BARD nine ball and
  ten rim detections.
- Dense-review importer focused verification:
  `venv/bin/python -m pytest tests/test_dense_codex_review.py tests/test_official_stats.py -q`
  passed 12 tests. It covers exhaustive-window enforcement, raw/manifest hash
  provenance, evidence-window bounds, sealed aggregation and mandatory partial
  coverage downgrade.
- Ruff passed for the dense importer, package exports, CLI and focused tests.
- Applying the importer to the real 18-window first-quarter package failed
  closed because `codex_events.jsonl` is still empty; it reported windows
  `window_00001` onward as unreviewed.
- After the coarse raw-only review, the same importer accepted 18/18 decisions,
  sealed 17 unresolved candidates, and forced `needs_review` because the review
  covers only 120--300 seconds of the 1033-second raw video.
- Tiered development-quarter evaluation at a 90-frame tolerance: candidate
  localization TP 17 / FP 0 / FN 0 / F1 1.0; automatic strict TP 0 / FN 17 /
  F1 0.0; reviewed strict TP 0 / FN 17 / F1 0.0.
- Focused verification after adjacent-window, dense-sheet and tiered-evaluator
  changes: `venv/bin/python -m pytest tests/test_official_stats.py
  tests/test_dense_codex_review.py -q` passed 15 tests; Ruff passed.
- Hash-bound review/finalization and identity-alignment focused tests passed.
  The real reviewed-quarter bundle contains 16 accepted and one unresolved
  event; strict identity-aligned evaluation at 90 frames is TP 16 / FP 0 /
  FN 1 / F1 0.9697.
- Six-period dense composition tests cover raw/evidence source-ID remapping and
  multi-video bundle ordering. Game A package generation produced 336 zero-gap
  windows; attempting combination before decisions were present failed closed
  with `unreviewed windows remain`.
- Full verification before the latest candidate-fusion increment reached 194
  passed tests and a passing harness gate. The new candidate generator/adapter,
  temporal evidence fusion and relation-remapping focused suite passes 4 tests;
  the combined official-stats focused run passes 17 tests, and Ruff passes over
  the touched modules, CLI and tests.
- The first fused sealed development bundle exposed a greedy matching defect in
  the evaluator. A regression test proves maximum-cardinality one-to-one
  matching for adjacent events. Re-evaluation of the unchanged bundle is TP 14
  / FP 49 / FN 3 rather than the previously under-counted TP 13.
- Post-fusion full regression: `venv/bin/python -m pytest -q` passed 198 tests
  with 15 pre-existing warnings; `venv/bin/python scripts/verify_harness.py`
  passed.
- Causal candidate bundle
  `1a53fa2d398689569c1b7c11d68bb9b7d54e4c2b64667f1421ff81240872ac4b`
  explicitly contains 12 paired steal/turnover candidates. Within the declared
  partial truth category scope, candidate localization is TP 15 / FP 48 / FN 2,
  recall 0.8824 and F1 0.375.
- Causal Codex-reviewed bundle
  `06a13c74945cf1b1fab0a1badadc463da6ca73a3e23393153ab8dee36b95478b`
  passes ledger and score reconciliation with 18/18 accepted events. Scoped
  identity-aligned strict evaluation is TP 17 / FP 0 / FN 0 / F1 1.0.
- Latest full regression: 203 passed, 15 pre-existing warnings; harness passed.
- Coverage-provenance and reviewed-reseal regressions:
  `uv run --with pytest --with ruff --with opencv-contrib-python-headless
  pytest -q tests/test_dense_codex_review.py tests/test_official_stats.py`
  passed 24 tests; Ruff passed over the touched review modules and tests.
- Reapplying Game A period 1's 12/65 dense decisions produced coarse bundle
  `75eff31a0521847a48a070409731df2ed58094b0af833f785db5784823521b0d`
  with explicit incomplete-coverage provenance. Applying eight hash-bound fine
  decisions produced reviewed bundle
  `ca2ce8296dc62e0acfc722c2e9b32effd896af76b20fdb6684d305a1116bb3f4`;
  its box score remains `needs_review`, accepts two made twos and reports 48
  unresolved candidates plus `incomplete_video_coverage`.
- Post-change full regression:
  `uv run --extra dev --extra inference --extra training pytest -q` passed
  205 tests with the same 15 warning messages; `uv run --extra dev --extra
  inference --extra training python scripts/verify_harness.py` passed. The
  first full-test attempt lacked the declared training extra and stopped during
  collection on missing `matplotlib`; installing that project extra resolved
  the environment-only failure.
- Fine-review event insertion focused verification:
  `uv run --with pytest --with ruff --with opencv-contrib-python-headless
  pytest -q tests/test_official_stats.py tests/test_official_bundle_script.py`
  passed 20 tests, and Ruff passed over the schema, ledger, review package and
  tests. Reapplying 25 Game A decisions produced reviewed bundle
  `6f2a6bcf85ec61bfd3e9c69c4bb51e007c11c0191fc240048aea902309ae1197`
  with 13 accepted events, 32 unresolved candidates and the expected
  incomplete-coverage gate.
- Post-insertion full regression:
  `uv run --extra dev --extra inference --extra training pytest -q` passed
  207 tests with 15 existing warnings; `uv run --extra dev --extra inference
  --extra training python scripts/verify_harness.py` passed.
- Local service hook on port 8792 passed with an ephemeral
  `opencv-contrib-python-headless<5` compatibility override: `/health` and
  `/ready` returned 200, `POST /api/v1/analysis/run` returned pending task
  `d42fb5eae65440148c21e957561ad674`, and status polling reached `completed`
  at progress 100. Without the override, the locked OpenCV 5.0 runtime lacks
  `CascadeClassifier`; the first environment attempt also omitted the optional
  Ultralytics tracking extra. These were runtime dependency issues, not analysis
  assertion failures. README was checked; the new internal review decision is
  documented in `docs/api.md` and does not change the public run/status fields.
- Fine review through candidate pair 18 now applies 37 hash-bound decisions.
  Partial bundle
  `56ef8d796e6577d8730efd2e4580e3487e117335ecfe5e4fe6c1ba5893010fb7`
  contains 19 accepted events and 20 unresolved candidates; its only
  reconciliation issue is the expected `incomplete_video_coverage` at 12/65.
  This batch corrected a field-goal candidate to a missed free throw and linked
  the ensuing offensive rebound. A regression found and fixed the ledger rule
  that previously accepted rebounds only after missed field goals. `uv run
  pytest tests/test_official_stats.py -q` passes 19 tests.
- Post-fix full regression under the same OpenCV 4.x compatibility override
  passes 208 tests with 15 existing warnings, and the AGU harness passes. The
  project-resolved OpenCV 5 run predictably reports the three already documented
  missing-cascade/legacy-tracker failures while the other 205 tests pass.
- Autonomous-boundary focused verification passes 38 tests across official
  inference, sealing/evaluation, event candidates, perception adapters and
  scripts. Ruff passes all touched modules. Regressions prove Codex-prefilled
  events are refused, reference/reviewer provenance cannot enter autonomous
  acceptance, VLM cannot invent identities outside traditional candidates, and
  incomplete events remain unresolved.
- Real raw-video smoke artifacts are under
  `analysis_outputs/raw_only_official/agu_autonomous_smoke_0_60/`: perception
  found only two basketball detections and the candidate/autonomous bundles
  contain zero events. This is a failed quality smoke, not an accuracy result.
- The follow-up AGU accurate task `dae20ce7492c4ba28962b94bb49f6de3`
  completed `/health`/`/ready`/run/status runtime verification and wrote 465
  R(2+1)D records. Fusion still emitted zero events. The separately opened
  four-event Codex acceptance CSV produced autonomous candidate/strict F1 0.0
  in `autonomous_evaluation.json`.
- Dense AGU VLM artifacts add `dense_vlm_candidates_v2.json` (29 candidates,
  pre-adjudication recall 0.75/F1 0.1818), resumable VLM decision cache, and
  `dense_vlm_autonomous_predictions_v2.json` (13 rejected, 9 automatic, 7
  unresolved). Strict autonomous evaluation remains F1 0.0.
- Latest OpenCV-4 full regression passes 220 tests with 15 existing warnings;
  Ruff and the focused harness gate pass.
- Applying all 59 first-batch Game A decisions produced reviewed bundle
  `3f67effb76efa58c53f7d5f21c84bde9e5a1f54cfec7490eca746a43b56f2071`:
  28 accepted events, zero unresolved events, raw-black/raw-white event points
  6--6, and exactly one reconciliation issue, `incomplete_video_coverage`.
- Re-importing the expanded Game A period 1 coarse log after 24 decisions passed
  with `reviewed_window_count=24`, `total_window_count=65`, 102 events, and
  bundle `4d05ec3d9c0e0d4b62c6800320794da9ed30e196048f07ff6cb02087bf25a16c`.
  The preceding run intentionally failed on out-of-window evidence for the
  459.3-second cross-boundary candidate, confirming the support-window gate.
- Re-importing after 40 append-only decisions passed with
  `reviewed_window_count=40`, `total_window_count=65`, 159 events, and bundle
  `88c947b6bf8720d5a6be02a260cc83436dedd44373fad8aebffaf564ad55c85c`.
  Window 33 is an explicit no-event decision and every boundary candidate names
  only a reviewed supporting window.
- Fine adjudication through candidate pair 32 now applies 71 hash-bound
  decisions. Reviewed bundle
  `acb3bc0ed7d12644e3efbe8755fd6129c9edb90971297e0ac644100d62d44c5e`
  contains 37 accepted events and 72 unresolved candidates. Accepted additions
  include a black 21 putback, paired NDP 0 turnover / EBBA 627 steal, and FK 7
  assist on black 21's made layup; incomplete coverage remains enforced.
- Importing all 65 period-1 decisions without `--allow-incomplete-windows`
  passed with `complete_video_coverage=true`, 256 candidates, and bundle
  `a55d914b036a3e0474df865d8635b6ea8e8f1c6f957fc540c590606392fef49c`.
  The generated full fine package contains all 256 candidate media folders.
- `cmp` over canonicalized event JSON proved the first 136 full-bundle events
  exactly match the prior 0--640-second bundle. Rebinding only the input hash of
  the 71 prior decisions and reapplying them produced 37 accepted and 192
  unresolved events in reviewed bundle
  `de502943b437a07a6a513214ead8677ba1297cf306fa3edf67ac174c40af20ff`.
  Reconciliation is valid with no issues; status remains `needs_review` because
  unresolved fine candidates remain.
- `jq -c` validated all 105 decision rows, and
  `scripts/apply_official_review.py` under `opencv-contrib-python-headless<5`
  applied them against source bundle `a55d914...` without a stale-hash or
  ledger error. Review audit reports 105 decisions, 50 accepted events, 160
  unresolved events, valid reconciliation, and reviewed bundle
  `4b7eea20918b6b2e19582cf8b031a1acba7792014c0e6093118eba7d4ed3ea70`.
  The task remains open because this is a partial period-1 result, not either
  complete-game strict-F1 evaluation.
- The first focused harness command failed before collection because it named
  two nonexistent files (`tests/test_official_box_score.py` and
  `tests/test_official_review.py`). Re-running with the repository's actual
  suites (`test_official_stats.py`, `test_dense_codex_review.py`, and
  `test_official_bundle_script.py`) passed the harness verification gate.
- `jq -s` validated 138 rows as 138 unique decision IDs and 138 unique event
  IDs. Reapplying them with `scripts/apply_official_review.py` under
  `opencv-contrib-python-headless<5` produced reviewed bundle
  `12ec531cc979013e2d764161f249502e2f80ed1870b96b53cc09baf512995204`.
  Review audit reports 138 decisions, 394 immutable revisions, 266 latest
  events, 65 accepted and 128 unresolved; reconciliation is valid with no
  issues. Status correctly remains `needs_review`, so this partial-period
  progress cannot satisfy the two-complete-game strict-F1 gate.
- The post-pair-64 `agu-test` / `agu-verify` gate passed with
  `scripts/verify_harness.py` running `test_official_stats.py`,
  `test_dense_codex_review.py`, and `test_official_bundle_script.py` under the
  pinned OpenCV 4.x runtime.
- `jq -s` validates the post-pair-72 ledger as 154 rows, 154 unique decision
  IDs, and 154 unique event IDs. Applying it against source bundle `a55d914...`
  produces reviewed bundle
  `8bc6f4e0cee00b309f2006ebc2bf0a26b48bee2e439c34077f5fa3e22ddefac7`.
  Review audit reports 154 decisions, 410 immutable revisions, 266 latest
  events, 77 accepted and 112 unresolved. Reconciliation is valid with no
  issue, and status correctly remains `needs_review`; this is not evidence for
  either complete-game accuracy gate.

The task must not move to Done until the new pipeline is connected end to end
and both games independently satisfy strict F1 >= 0.85. Full second-game truth
is still missing.

Latest autonomous Game A 0--60 second verification:

- Multi-detector traditional candidate bundle before semantic adjudication:
  TP 4 / FP 1 / FN 0, F1 0.8889.
- Final autonomous localization after temporal overlay, trajectory priority and
  causal rebound gating: TP 4 / FP 0 / FN 0, F1 1.0.
- Raw strict identity F1: 0.0; track IDs are not yet stable roster identities.
- Hash-bound evaluation-only identity alignment: TP 3 / FP 0 / FN 1, precision
  1.0, recall 0.75, F1 0.8571. This is explicitly not a full-game result.
- `uv run --with 'opencv-contrib-python-headless<5' pytest -q`: 226 passed, 15
  existing warnings. Touched-file Ruff passed and `python
  scripts/verify_harness.py` passed. Full-repository Ruff still reports 71
  pre-existing issues in legacy service/manual-test files.
- Local service hook passed on port 8765: `/health` returned `ok`, `/ready`
  returned `ready`, task `d038956ef0ca4c1c917c502b0ae0aeac` progressed from
  pending/45% processing to completed/100% in 13.87 seconds.

Post-identity-graph verification on 2026-07-18:

- Identity graph, raw-artifact verification, canonical observation mapping,
  VLM alias mapping and pre-rim shooter-window focused suites passed.
- After sharing canonicalization with dense all-event VLM discovery, the
  identity/discovery/candidate/adjudication focused suite passed 22 tests and
  touched-file Ruff passed.
- `uv run --with 'opencv-contrib-python-headless<5' pytest -q`: 233 passed with
  15 existing warnings.
- Touched-file `uv run ruff check ...`: passed.
- `uv run python scripts/verify_harness.py --run-tests`: passed.
- Local runtime hook on port 8876 passed: `/health` returned `ok`, `/ready`
  returned `ready`, and task `012a9713ae8d4304af210ef460b80efe`
  completed at 100%. The default `uv` environment lacked `uvicorn`, so the
  repository's documented `venv/bin/python` runtime was used.
- These structural/runtime results do not satisfy the business accuracy gate.
  Game A has only a 60-second autonomous diagnostic and the available Game B
  truth is a 17-event partial first-quarter set, not a second complete game.
- After dense identity and causal relation integration, the full OpenCV-4 suite
  passes 236 tests with the same 15 existing warnings. The new regressions cover
  type-compatible parent linking, parent-first adjudication, assist/make team
  consistency and rebound-type/team consistency.
- A planned qwen3-vl:8b comparison could not run: two `ollama pull` attempts
  resumed the 6.1 GB layer to roughly 11% and then failed with upstream EOF/max
  retries. The existing qwen3-vl:4b result remains the latest semantic result;
  model-download failure is not treated as an accuracy result or code failure.

2026-07-18 8B/two-pass and annotation-boundary verification:

- `qwen3-vl:8b` completed its resumable 6.1 GB pull and Ollama digest check.
- Game A two-pass 8B sealed bundle:
  `db020384772eecbae3ef92c5eb7e7d1b098903a5ddd93a321041bb96e1a0995f`.
  Candidate localization is TP 3 / FP 0 / FN 1, F1 0.8571. Global post-seal
  identity alignment is TP 1 / FP 2 / FN 3, strict F1 0.2857; raw strict F1 is
  0. The result is a 60-second diagnostic and cannot satisfy a full-game gate.
- Game B role-routed candidate bundle on the declared partial truth is TP 14 /
  FP 49 / FN 3, recall 0.8235 and F1 0.35. No automatic strict success is
  claimed, and the truth is not a complete second game.
- Focused annotation, frame-provider, two-pass reviewer, causal and inference
  tests passed. `uv run --with 'opencv-contrib-python-headless<5' pytest -q`
  passed 254 tests with 15 existing warnings. Touched-file Ruff passed.
- `uv run --with 'opencv-contrib-python-headless<5' python
  scripts/verify_harness.py` passed. Full-repository Ruff still reports the 71
  pre-existing legacy import/unused-code issues; no bulk unrelated rewrite was
  performed.
- Local hook on port 8877 passed: `/health` returned `ok`, `/ready` returned
  `ready`, and task `02f92577266c4079af8e0b62829cc2f2` completed at 100% in
  8.37 seconds. The documented `venv/bin/python` was used because the temporary
  `uv --with opencv` environment does not include `uvicorn`.

2026-07-18 open-pretraining/action-owner follow-up:

- Focused action ownership, candidate generation, review-sheet, game-state and
  open-dataset suites passed 30 tests; touched-file Ruff passed.
- Full `uv run --with 'opencv-contrib-python-headless<5' pytest -q` passed 270
  tests with 15 existing warnings. `python scripts/verify_harness.py` and
  `git diff --check` passed.
- Local service hook on port 8878 passed: `/health` returned `ok`, `/ready`
  returned `ready`, and task `7b385d0d6b7845c7afc7e7e385fe688e`
  completed at 100% in 7.20 seconds. The documented `venv/bin/python` was used;
  `.venv` does not include uvicorn.
- The latest 8-action cross-video model scored pairwise F1 0.7544 and Top-1
  0.375. These are negative gate results: no model was promoted to runtime and
  neither independent complete-game accuracy requirement is satisfied.

2026-07-19 action-owner training and independent acceptance verification:

- Corrected manifest v11 hash:
  `b894c564fb1f0f801aab70a6ae4f92179c5d0290d81947030e618e645627d219`.
  Promoted Extra Trees model hash:
  `c98e2af56480e41390415fb0f9d880ae1a876f3ec2625035ec45b144d9dd0c22`.
  It has 155 candidate rows, 16 actions, eight videos, candidate F1 0.6316,
  and leave-one-video-out Top-1 0.875. Candidate F1 is reported for
  calibration transparency; Top-1 is the declared actor-selection subgate.
- Touched-file Ruff passed. Focused action-owner, embeddings, official
  inference and overlay suites passed 35 tests. `scripts/verify_harness.py
  --run-tests` passed.
- A direct `uv run pytest -q` produced 277 passed and three unrelated failures:
  the current uv OpenCV build lacks `CascadeClassifier` and
  `legacy.MultiTracker_create`. No unrelated face/tracker fallback code was
  changed. Prior full validation with the repository's pinned OpenCV 4.x
  environment remains recorded above.
- Local runtime hook on port 8765 passed using `venv/bin/python`: `/health`
  returned `ok`, `/ready` returned `ready`, and task
  `7b7c08c5efb445fe92795b6ac78623cb` completed at 100%. `.venv` still does not
  include uvicorn.
- Sealed independent diagnostic bundle `5cbb1758...` used absolute model-prior
  overlays and scored TP 0 / FP 3 / FN 4, strict F1 0.0. Balanced-prior bundle
  `068d5077...` always retained the ball-nearest candidate, scoring TP 1 / FP 2
  / FN 3, strict F1 0.2857. It recovered but did not exceed the prior baseline.
  Both runs passed `--require-agu-autonomous` and exposed the training manifest
  plus `benchmark_overlap=false` provenance before truth was opened.
- The independent gate failure supersedes the 0.875 LOVO promotion decision.
  The artifact and adapter remain experimental/default-off. Neither
  complete-game six-stat strict-F1 result has run, so TASK-0041 and the user
  goal remain active.
- A follow-up actor prompt that explicitly traced the airborne ball backward
  also scored strict F1 0.0 and increased `needs_review`; it was reverted.
  This negative result narrows the next work to canonical-ID duplicate
  suppression, release-anchor accuracy and rebound first-control coverage
  rather than more prompt tuning.

2026-07-19 face/causal-rebound verification:

- Focused identity, face-gallery, action-owner, candidate, exact-frame,
  two-pass inference and overlay suites: `78 passed`.
- Touched-file Ruff and `git diff --check`: passed.
- `uv run python scripts/verify_harness.py --run-tests`: passed.
- Local service hook on port 8879 passed: `/health` returned `ok`, `/ready`
  returned `ready`, and task `ae512497bfd34bed9ddcd8d124cd8b9e`
  completed at 100% with no error.
- The sealed distinct-overlay predecessor retained localization F1 0.8571 but
  strict F1 0.0. The later causal team-constrained autonomous bundle hash is
  `e160a39d9eb7f65b8068db94a8e5af97664ee98ea0c774dd9d32e3240b4f04ac`;
  it reaches localization TP 4 / FP 0 / FN 0 (F1 1.0), including the rebound,
  while raw strict TP 0 / FP 4 / FN 4 (F1 0.0) exposes the remaining identity
  fragmentation. Neither complete-game gate has run or passed.

2026-07-19 annotated face enrollment verification:

- `uv run --python 3.11 pytest -q tests/test_face_enrollment.py
  tests/test_face_enrollment_approval.py tests/test_face_gallery.py
  tests/test_official_identity.py`: 16 passed.
- Touched-file Ruff passed.
- Full July candidate manifest hash:
  `c908e13f7d39fddc15609e5c067781413f45b575566e5ee20ef398d29daa1ab1`;
  face gallery hash:
  `27d771a5cef3cc79c6149c9696e42672f6550f3c5c746fc8a4455e0cc81fc304`.
- Game A identity artifact hash:
  `03e424e291acd676a0184d94cd5b7af271b26bf7ba2b751997586be06268e5d0`;
  required-gallery candidate bundle hash:
  `dec3e0e8953c5ded47c39dbd49beda3b18d15a257a7d1656e71d11f0eca0b3a7`.
  It contains three field-goal and three rebound candidates, all with empty
  actor lists because no registered identity intersects those windows.
- `uv run --python 3.11 python scripts/verify_harness.py --test-command
  "uv run --python 3.11 pytest -q tests/test_face_enrollment.py
  tests/test_face_enrollment_approval.py tests/test_face_gallery.py
  tests/test_official_identity.py"` passed; `git diff --check` passed.
- Local runtime hook on port 8880 passed: `/health` returned `ok`, `/ready`
  returned `ready`, and task `cb4892cdd2424772904857b89f6ee6ba`
  completed at 100% with a result and no error.

2026-07-19 enrolled identity propagation correction:

- `uv run --python 3.11 pytest -q tests/test_face_enrollment.py
  tests/test_face_enrollment_approval.py tests/test_face_enrollment_merge.py
  tests/test_face_gallery.py tests/test_identity_graph.py
  tests/test_official_identity.py`: 32 passed; touched-file Ruff passed.
- Regression tests prove that body-only similarity cannot inherit a gallery ID,
  while direct quality-gated face evidence still can.
- Strict graph artifact hash:
  `dbe0e32d15bcd6093db270386348ae920204e165c4144ad23f54c76370125ec5`;
  strict candidate bundle hash:
  `1b260be237951c73abb9ab775ce1244a8152aa191922cecf6eef32f534ce03da`;
  autonomous bundle hash:
  `4e5e507948bd84043e51e5fe61096a945ff3f0086d4451e3aeb12cf34ba7582e`.
- The corresponding independent 0--60 s evaluation is localization F1 0.8889
  and automatic strict F1 0.0. This explicitly supersedes the earlier statement
  that all six actor candidate lists were empty.
- January cross-game gallery hash:
  `e858d9f7112bb2e02fe665c392151fe089eb0567c9eeb39d3251332b6ded9623`;
  combined July plus independent-first-quarter gallery hash:
  `086d52ac2ec389ef320c7391b6e347a2f6b57e10db7cbc60c1427c38e6335fb0`.
  The 11-label July development audit has exact enrolled action-owner coverage
  1/11; it is not an acceptance result because those videos are training data.
- Harness verification with the 32-test focused command and `git diff --check`
  passed. Local runtime hook on port 8881 passed `/health`, `/ready`, task
  submission and completed status; task `3d224b59181049b5969931f9fd62b3dd`
  reached 100% with a result and no error.
- The new low-quality gallery regression proves a 0.60-quality query cannot
  receive an enrolled ID under the 0.65 default. Focused enrollment/identity
  tests remain 32 passed and touched-file Ruff passes.
- Quality-gated identity artifact hash:
  `9ceeb75fb782a87eadb39c0ab7f1e3ccbef1fc49dc732fee83d7426b02ca2525`;
  candidate bundle hash:
  `5940133a665604b750b9058ed7a6c3860240458732859bfd6aea6901c03008fa`.
  All six action candidates have empty enrolled actor lists after the observed
  cross-actor false match is removed.
- Post-quality-gate harness plus the 32-test command and `git diff --check`
  passed. Local service hook on port 8882 passed health/ready and task
  `6fa1109d8bc548c9b19449f75a0b5d4d` completed at 100% with no error.
A 2026-07-19 repository consolidation retained the raw-only official pipeline,
face-enrollment and identity graph, Codex training/acceptance firewall, FastAPI
compatibility path and v3 preprocessing contract while removing redundant legacy
entry points and stale training plans. Verification after cleanup: `322 passed`,
source `ruff --select F` passed, `git diff --check` passed, harness `--run-tests`
passed, and local task `45dc5762f00e4939a718f517fd28640b` completed at 100%.
This structural cleanup does not change the unresolved accuracy conclusion.
