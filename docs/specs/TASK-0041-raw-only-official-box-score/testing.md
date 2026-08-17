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
  workspace, so the documented `.venv/bin/python` runtime was used.)

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

- `.venv/bin/python -m pytest tests/test_perception_adapters.py tests/test_official_stats.py tests/test_game_state.py tests/test_identity_graph.py -q`: 20 passed.
- `.venv/bin/python -m pytest`: 181 passed, 15 pre-existing warning messages.
- `.venv/bin/python scripts/verify_harness.py`: passed.
- Ruff over all new/changed Python modules and tests: passed.
- Local service on port 8791: `/health` returned `ok`, `/ready` returned
  `ready`, `POST /api/v1/analysis/run` returned the expected pending task shape,
  and task `d74a0d962d2f4de1995dd9fac67a77d6` completed at progress 100.
- First-quarter strict legacy baseline: TP 0 / FP 648 / FN 17 / F1 0.0 with a
  90-frame tolerance.
- Same 50-frame detector sample: COCO two ball detections; E-BARD nine ball and
  ten rim detections.
- Dense-review importer focused verification:
  `.venv/bin/python -m pytest tests/test_dense_codex_review.py tests/test_official_stats.py -q`
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
  changes: `.venv/bin/python -m pytest tests/test_official_stats.py
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
- Post-fusion full regression: `.venv/bin/python -m pytest -q` passed 198 tests
  with 15 pre-existing warnings; `.venv/bin/python scripts/verify_harness.py`
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
  repository's documented `.venv/bin/python` runtime was used.
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
  8.37 seconds. The documented `.venv/bin/python` was used because the temporary
  `uv --with opencv` environment does not include `uvicorn`.

2026-07-18 open-pretraining/action-owner follow-up:

- Focused action ownership, candidate generation, review-sheet, game-state and
  open-dataset suites passed 30 tests; touched-file Ruff passed.
- Full `uv run --with 'opencv-contrib-python-headless<5' pytest -q` passed 270
  tests with 15 existing warnings. `python scripts/verify_harness.py` and
  `git diff --check` passed.
- Local service hook on port 8878 passed: `/health` returned `ok`, `/ready`
  returned `ready`, and task `7b385d0d6b7845c7afc7e7e385fe688e`
  completed at 100% in 7.20 seconds. The documented `.venv/bin/python` was used;
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
- Local runtime hook on port 8765 passed using `.venv/bin/python`: `/health`
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

2026-07-20 verification note:

- Focused multi-prototype enrollment and identity suites passed 22 tests;
  official event/inference suites passed 51 tests; candidate-probe tests passed
  3 tests; and semantic/script suites passed 41 tests during iteration.
- The sealed ATL-CHI gallery passes 19/23 strict roster entries. The canonical
  team-map artifact contains 1570 ATL/CHI tracklets, but only 13 direct named
  tracklets across four people; team normalization is therefore not equivalent
  to player identity success.
- Full-game linked-trajectory rebuilds retained 145 ATL-CHI and 182 LAL-BOS
  field-goal candidates. ATL-CHI outcomes were 142 unknown/2 made/1 missed;
  LAL-BOS outcomes were 179 unknown/2 made/1 missed. This is insufficient for a
  frozen acceptance run.
- The truth-free 8B 3-event temporal probe completed with one evidence-complete
  missed result and two conservative unknown results. No truth or Codex event
  answer was available to the inference process.
- Final verification passed: touched-path Ruff, `git diff --check`, all
  `360 passed` repository tests, and `scripts/verify_harness.py --run-tests`.
  Local service port 8891 returned `ok` and `ready`; task
  `b2da833b21e64c67bd8605a4e3eb5eda` progressed from pending/processing to
  completed at 100% with a result and no error.

2026-07-20 semantic retention and rim-detail probe:

- Focused official inference/candidate/script regression coverage reached 70
  passing tests. It proves partial labels remain unresolved, low-confidence
  labels are discarded, cached three-point grounding survives, contradictory
  made/miss evidence abstains, cluster midpoint sampling is deterministic, and
  the optional rim inset preserves frame shape.
- ATL-CHI v14 probe bundle
  `c5aab07635cbf3a52f13bc5941e57a6bb1cc80df9a471cb3eb16339e35e3cf09`
  returned missed/unknown/missed. LAL-BOS v14 probe bundle
  `8ac22ecefe5490daed61f03aaba8dc95fe195e090948cbf845db1ecd0e82a25c`
  returned made/made/unknown. All six remain `needs_review` and were produced
  without truth access.
- Independent post-freeze Codex contact-sheet inspection did not reliably
  validate the two LAL makes, so no precision, recall or F1 is claimed and the
  full-game semantic batch was not started.
- Final verification passed touched-path Ruff, `git diff --check`, all
  `368 passed` repository tests, and the harness with `--run-tests`. Local
  service port 8892 returned `ok`/`ready`; task
  `d8796b0c921b42a9841c8ce19374a48d` completed at 100% with a result and no
  error.

2026-07-20 scoreboard verification:

- Focused scoreboard reader/runner tests: `17 passed`; touched-path Ruff passed.
- LAL-BOS evidence SHA:
  `80d0b6d95fb5e0f43f435ab8bf0d3784945618cdfb8149c595e22ebc8f5dc419`;
  bundle SHA: `3aa06aba6cf8fc85fe25f7094d095c97b87e3b898ac65ae6618349cbbc11e95a`.
  It emitted 87 deltas (34 uniquely linked, 16 ambiguous, 37 unlinked) and
  scored team made-value component micro F1 `0.8600`.
- ATL-CHI evidence SHA:
  `5630c8d693d6844d6f89b4198d178955663ba10dbe8ba9f80e0a0b5f565df54d`;
  bundle SHA: `e98ea51c0d736f742c258d51692e86c57b22ea51e5db46fd7efb617085b263a0`.
  It emitted 98 deltas (21 uniquely linked, 11 ambiguous, 66 unlinked) and
  scored team made-value component micro F1 `0.9703`.
- ATL closed at the raw final score `83-95`; LAL's last reliable v3 state was
  `88-86`, missing the second team's final 12 points. Neither artifact has a
  confirmed actor, and neither is a complete-game player six-stat result.
- Final repository verification: `385 passed`, touched-path Ruff passed,
  `git diff --check` passed, and `scripts/verify_harness.py --run-tests` passed.
  Local service port 8893 returned `ok`/`ready`; task
  `abf060ed36ab4410a6fd8c283187cc3e` completed at 100% with no error.

2026-07-20 audio and live-broadcast verification:

- Both full enrollment scoreboard scans completed and matched the exact final
  scores LAL 102–BOS 108 and CHI 99–ATL 82. Base-English ASR artifacts completed
  for both enrollment and benchmark videos.
- The unpromoted score-candidate calibration artifact has pooled precision
  0.875, recall 0.0383 and coverage 0.0437; neither game passed the declared
  minimum predictions/coverage gates.
- Semantic secondary probes reviewed 9/9 LAL events and 8/8 ATL events. One ATL
  block was visually absent and is now `rejected`; the remaining events stay
  unresolved because actor/causal requirements are incomplete.
- Registered-candidate probes produced six LAL events and three ATL events. All
  four proposed rebounds were rejected. The apparent ATL registered-player make
  was independently inspected at video time 54:40 and shown to be a `HALFTIME`
  replay. After the broadcast-gate change, the one-event frozen rerun is
  `rejected` with bundle SHA
  `57c9d5a76bdbbd12d179c3b7b7d58d474dd352756fab751032e98ce7b690a484`.
- Focused inference/audio/probe checks pass 64 tests; the expanded autonomous
  inference/script/probe set passes 56 tests, and touched-file Ruff passes.
  The harness `--run-tests` gate passed all 397 collected repository tests;
  touched-path Ruff and `git diff --check` passed. Local service port 8894
  returned `ok`/`ready`; task `8a908975c4ac40c3b48693c0863f6ad5`
  completed at 100% with a result and no error.

2026-07-20 registered-jersey verification:

- `.venv/bin/python -m pytest`: `405 passed`, 15 existing warnings.
- `.venv/bin/ruff check` passed for the face-gallery, identity, jersey-enrollment
  scripts and their focused tests; `git diff --check` passed.
- `.venv/bin/python scripts/verify_harness.py --run-tests` passed.
- The corrected full ATL identity graph has SHA
  `893e1891201981481824d7ad48894374c04f6b28bd1b6cfca5c81ff227c96955`;
  the jersey registration gallery has SHA
  `7528b1a90f8d9a6cce9c35da9ec8d1160ddf7872ff8be22e4e41480a76a7cc0f`.
- The first runtime jersey graph probe has SHA
  `eb859f7d6697a0d674caed2357abcc8b034a05d8ece89b39c533ee3c675c7621`.
  It maps only the two consensus reads and preserves the two rejected reads as
  anonymous.
- Local service port 8765 returned `ok`/`ready`; task
  `ff46ea6c01354d90a2a33caa7d984da4` progressed through processing and completed
  at 100% with a result and no error.
- The authoritative-player-source regression passes in the focused event
  candidate/bundle suite (`24 passed`) and verifies that a colliding primary
  player track is absent whenever a dedicated player artifact is supplied.

2026-07-20 authoritative actor/jersey verification:

- Focused autonomous-inference and identity tests pass `53 passed`; after the
  prompt/image-bound cache regression was added, the autonomous-inference file
  passes `43 passed`. Touched-path Ruff passes.
- The actor-scoped registered identity graph has SHA
  `03025e7e618983c9ec9f3b5765c993d161ab61f8fa59ac6ec493f18515a63a19`.
  It maps raw actor tracks to `201565` (jersey 1) and `201149` (jersey 13), and
  preserves the conflicting and sub-threshold tracks as anonymous ATL IDs.
- The rebuilt full candidate bundle has SHA
  `1927d900b88cbc2d859d15509b5f3f8598b27e1313223bb7d8275e205ae7da9e`
  and retains exactly 145 field-goal plus 145 rebound candidates with unchanged
  frame intervals. The six-event registered probe has SHA
  `7b33fd05fd15f45c876b1f7d6743ce0dfd32542a127618b31c078a34eec5f143`.
- This remains a truth-free capability probe, not a full-game strict-F1 result.
  Both complete-game freezes and the declared accuracy gate remain open.

2026-07-22 focused verification:

- Candidate-window pose, model sealing/tamper rejection, feature extraction,
  rejected-shot/rebound cascading, manifest-bound training, leave-one-video-out
  threshold calibration, truth-free annotation export and optional candidate
  integration pass focused pytest coverage.
- The implementation does not yet have enough independently labeled development
  games to promote a checkpoint. No two-game player six-stat strict-F1 run was
  performed and no 85% or 95% accuracy claim is made.
- Touched-path Ruff and `git diff --check` passed. Full pytest passed `420`
  tests with 15 existing warnings, and `verify_harness.py --run-tests` passed.
- Local service port 8766 returned `ok`/`ready`; task
  `d9ae50652c4347d5b801e50566c92fe2` progressed through processing and completed
  at 100% with a result and no error.
- Two full candidate-window pose artifacts were hash-bound to their raw video,
  source bundle and YOLO11 pose weight. Both pose-enriched rebuilds retained
  exact prior event geometry and generated 145/182 truth-free label templates.
- A sparse-frame regression covers the optimized pose attachment index; focused
  perception/candidate tests pass 30 tests. The post-optimization full suite
  passed 420 tests and the harness `--run-tests` gate passed again.

2026-07-22 full-window temporal shot-gate verification:

- Independent contact-sheet review first produced 55 training-only windows: 19 true
  shots and 36 false candidates. The sealed manifest reports no overlap with
  the frozen benchmark set.
- Pose/geometry Extra Trees leave-one-video-out P/R/F1 was
  1.000/0.053/0.100. The frozen Kinetics R(2+1)D-18 full-window embedding plus
  logistic head improved that to 1.000/0.368/0.538 at the same 0.95 precision
  floor. These are development calibration metrics, not blind acceptance.
- The ATL complete-candidate runtime smoke consumed only raw video, a sealed
  candidate bundle, the frozen backbone and sealed head. It retained 67/145
  shots plus 67 dependent rebounds; all 12 labeled positives in that same
  development subset were retained with no labeled false positives. The latter
  is an in-sample implementation consistency check, not generalization.
- Touched-path Ruff and focused pytest passed (`56 passed`). The project
  `.venv/bin/python -m pytest -q` suite passed `430 passed, 15 warnings`, and
  `.venv/bin/python scripts/verify_harness.py --run-tests` passed. A secondary
  `.venv` suite reached `427 passed` but failed three pre-existing OpenCV API
  tests because that environment lacks `CascadeClassifier` and
  `legacy.MultiTracker_create`; the focused new tests pass there.
- Local service port 8767 returned `ok`/`ready`; task
  `73b793e6778748efa75aec1098995aa1` progressed through processing to
  `completed`, progress 100%, with a result and no error.

2026-07-23 causal-window verification:

- Detector/model screening stayed raw-only. Codex viewed only generated
  detection montages and did not supply any runtime event answer.
- LAL processed 5,146 samples in 66 merged windows; ATL processed 2,051 samples
  in 71 windows. Pose runs processed 4,864/2,621 samples and emitted
  21,214/7,911 player poses.
- Unit tests cover raw-hash tamper rejection, candidate-window binding,
  object filtering, montage validation, scoreboard-semantic preservation,
  identical ball/rim class-confusion rejection, evidence-only label rebinding
  and timing-change rejection.
- The 518-row causal manifests exclude both reserved MEM-OKC raw hashes.
  Neither the causal-only nor causal+pose model passes the four-video
  precision/recall promotion gate; both remain generated, non-runtime
  checkpoints.
- Final verification passed 458 tests with 15 existing warnings, 37 focused
  tests, touched-path Ruff, `git diff --check` and
  `verify_harness.py --run-tests`. The local service returned healthy/ready;
  task `f6ed00d58e0741d2af9c0ef323212c9e` completed at progress 100 with a
  result and no error.
- Sixteen additional independently reviewed windows expanded the sealed set to
  71 (25 positive/46 negative). The earlier temporal score did not reproduce:
  default leave-one-video-out P/R/F1 is 0/0/0, and the best precision-1.0 grid
  point recovers only 2/25 positives. The new per-video precision and recall
  promotion gate rejects the artifact.
- A frozen eight-window ATL semantic probe measured Qwen3-VL 8B contact-sheet
  P/R/F1 at 0/0/0 and separate-frame P/R/F1 at 0.429/1.000/0.600. These are
  event-presence metrics from the raw-only VLM cache, not final box-score
  accuracy. Requiring the explicit three-stage release chain did not change the
  measured result because 8B asserted all stages on the same four negatives.
  The causal release-chain regression suite passes 52 tests.
- A third training game added 24 raw-only labels (11 positive/13 negative).
  The sealed 95-window embedding artifact SHA is
  `9ca814317dd6cc8691574e96bd0eed6f98937cb8d1ebdad9764a5242a8c4381d`.
  C=0.001--10 all failed the per-video promotion gate; best pooled P/R/F1 was
  1.000/0.056/0.105 and two held-out videos had zero recall.
- Cross-game temporal recall remains below the required floor, so the gate stays default-off. No
  two-unseen-game player six-stat strict-F1 run occurred and neither 85% nor
  95% is claimed.

2026-07-23 four-game fine-tuning verification:

- The fourth development annotation has 23 rows (2 positive/21 negative), is
  `runtime_consumable=false`, and is bound to source SHA
  `248470c4dac88a4c046faccd37be23f23ecb804f46b39f4ebeebe57cf212dc81`.
  The four-game manifest SHA is
  `341a032f3b12c936b2d0006ec86a75197a83f9c306b17d1198ee15067005e3bc`.
- The Kinetics end-to-end OOF artifact rejected promotion at P/R/F1
  0.500/0.026/0.050. The AGU v3/SpaceJam initialization rejected promotion at
  0.200/0.026/0.047. Neither checkpoint is runtime-consumable.
- New benchmark video hashes are
  `1caf38b7b9c1997805cf5e66a37a05af974fdb5d99dfaa8816f328710523bbb8`
  and `9651a64355839780dffac2d404fa4715a08668b192a254cf87cbaba2d70c7f2e`;
  both are absent from all training source hashes. The enrollment-only video
  hash is `69bdd21fcc33f804403328180386819f0bd869bc0e31accd03bffcc673eab137`.
- Focused fine-tuning and frozen-temporal tests pass 8 tests; touched-path Ruff
  and `git diff --check` pass. The project Python 3.12 environment passes the
  complete suite (`439 passed, 15 warnings`) and
  `verify_harness.py --run-tests` passes. The secondary `.venv` reaches
  436 passes and retains its three known OpenCV-build failures because it lacks
  `CascadeClassifier` and `legacy.MultiTracker_create`.
- Local port 8768 returned `ok`/`ready`; task
  `b26fda4ae634449988f947f6c3aa9060` progressed to `completed`, progress 100%,
  with a result and no error.

2026-07-23 complete 374-window shot-gate verification:

- Supplemental labels contain 109/147 unique events, overlap neither prior
  annotation file, and their per-game unions exactly equal the 145/182 sealed
  field-goal candidate sets. The LAL supplemental label SHA is
  `2c28ca2511b68d1c89f15a1719c8f22c9e099823749a146c8cedd651d5ddd702`.
- The combined training manifest has 374 rows (173 positive/201 negative), four
  source videos and zero raw-hash overlap with the two reserved benchmark games.
- `shot_validity_r2plus1d_full374_layer4_e3.json` records `promoted=false` and
  `runtime_consumable=false`; `shot_validity_extra_trees_full374.json` records
  `minimum_precision_gate=false`. Neither artifact is eligible for runtime.
- The four rejected R(2+1)D training checkpoints were deleted after a reference
  audit found no code or runtime consumer. Their small JSON evaluation records
  remain as the durable result.

2026-07-23 511-window open-backbone verification:

- The two canonical scoreboard annotation files contain 66 and 71 unique
  timestamps (50/16 and 65/6 positive/negative). Their union produces a
  511-row manifest with SHA
  `8e2686c6a0ccad063bc6e9900964c2ad3211d52e17d1eb7d5996f7f334488f9e`;
  reserved MEM-OKC hashes remain excluded.
- Swin3D-T embedding SHA is
  `916ddf43e876bc511f3fb0d62bf69b118d43f9691ee0dc3c51c11f04ad42dd86`;
  MViT-v2-S embedding SHA is
  `65f1dac5eb5808dc0a82c7ec0fe44717fec3e8ef483fd9917e1e9a9d60630123`.
  Both verify as training-only.
- The best Swin per-game precision at recall >=0.85 is 0.734/0.779/0.838/0.570.
  The best Swin+MViT fusion result is 0.742/0.825/0.845/0.570. No configuration
  passes promotion, so no acceptance truth or downstream statistic was opened.
- Focused Ruff and `tests/test_shot_validity_video_backbone.py` pass. The tests
  cover open-source registry metadata, SHA tamper rejection, game-level OOF,
  aligned fusion and fail-closed rejection of mismatched examples.
- The v2 traditional trainer consumed all eight annotation/bundle pairs across
  four source videos (511 rows) and rejected promotion for every tested tree
  depth/leaf configuration. Best per-game P@R>=0.85 was
  0.686/0.765/0.953/0.557; only one fold crossed 0.95 precision.
- Regression coverage verifies that multiple bundle SHAs may bind one raw video,
  every held video must pass the joint precision/recall gate, causal wrist-ball
  features are extracted, and existing v1 model artifacts remain readable.
- Final touched-path Ruff, `git diff --check`, 22 focused shot/candidate tests and
  the AGU harness verification all pass. No API, configuration or service
  contract changed, so the local service curl hook was not required.

2026-07-23 short-window scoreboard verification:

- Unit coverage verifies that a configured scoreboard localization lookback
  bounds event geometry, preserves the complete score-delta evidence interval,
  and rejects non-positive values.
- The two new annotation files contain 71 and 73 unique field-goal IDs and
  exactly cover their short-window candidate bundles. The combined manifest is
  SHA-bound, training-only and excludes both reserved MEM-OKC raw hashes.
- The 518-row causal v2 run records `promoted=false`; per-game P@R>=0.85 is
  0.683/0.803/0.985/0.553. No runtime model or acceptance truth was opened.
- Focused pytest passed 24 tests, touched-path Ruff passed, and the complete
  `verify_harness.py --run-tests` gate passed. Local port 8765 returned
  `ok`/`ready`; task `9d8f2aaf209541c0a5be24005c24aa10` progressed to
  `completed`, progress 100%, with a result and no error.

2026-07-23 uniform window-causal verification:

- ATL benchmark enrichment binds 145 attempts and adds physical proposals to 34
  events; LAL binds 182 attempts and adds proposals to 46. Both pose artifacts
  were accepted only after their source candidate identity was proven equivalent
  to the final actor/pose bundle.
- The version-scoped causal evaluator reports ATL 16 TP / 18 FP / 64 FN and LAL
  22 TP / 24 FP / 58 FN across the complete reviewed benchmark labels. This
  falsifies using the fixed geometry rule as a hard candidate gate.
- The sealed uniform manifest contains 518 rows (300 positive/218 negative),
  four source-video hashes, eight annotation files and zero overlap with the two
  reserved acceptance bundles. Manifest SHA is
  `fd11b82c694369f9d37c16025dc96a3581e66e86ef7c7aaef356a312a1be6f6e`.
- Depth/leaf configurations 3/4, 4/4, 6/2, 6/4, 6/8 and 8/4 all fail closed.
  Depth 3 / leaf 4 is best overall: held-game precision at recall >=.85 is
  .692/.816/.971/.562. It improves every fold over the prior best v2 ranking
  (.673/.769/.968/.540), but three folds remain below .95, so all v3 artifacts
  record `promoted=false` and no acceptance truth was opened.
- Touched-path Ruff, `git diff --check`, 16 focused tests, all 462 repository
  tests (15 existing warnings) and `verify_harness.py --run-tests` pass. Local
  port 8769 returned `ok`/`ready`; minimal raw-video task
  `f78860bb913242a0819a4656f9a347b6` completed at 100% with a result and no
  error.
- RF-DETR 1.8.3 passes `pip check` without changing core inference packages.
  Adapter regression tests verify BGR-to-RGB conversion, four-class mapping,
  source-shape boxes and fail-closed removal of the extra background class.
  On the frozen 23-window ATL probe, RF-DETR's causal decision is 1 TP / 8 FP /
  1 FN / 13 TN at evidence thresholds .25, .40, .50 and .70. The YOLO decision
  is 2 TP / 10 FP / 0 FN / 11 TN, and the RF set is a strict subset, so detector
  agreement cannot improve precision or recall on this sample.

2026-07-23 independent-VLM and normalization verification:

- Granite 3.2 Vision 2B was evaluated only against the frozen eight-event raw
  bundle. Twelve-frame requests failed closed because the model's 16K context
  was exceeded. Contact-sheet and six-frame retries completed, but all eight
  final events remained `needs_review` because the required release-chain
  fields were absent; no missing response was converted into a negative label.
- The normalized-feature sweep computes ranks and z-scores independently
  within each video and never uses held-video labels. Its best worst-fold
  precision is .535 at recall .850, below the raw-feature best of .562. The
  experiment is rejected and no runtime-consumable artifact was emitted.
- Final task-path Ruff, `git diff --check`, `pip check`, all 464 repository
  tests (15 existing warnings), and `verify_harness.py --run-tests` pass. The
  repository-wide Ruff command still reports 49 pre-existing baseline issues in
  untouched legacy paths, chiefly import ordering; no bulk rewrite was made.
- Local port 8770 returned `ok`/`ready`. Minimal raw-video task
  `a26521878a714bab9582875ae0bb8051` completed at 100% with a result and no
  error; the service was then shut down cleanly.
- The post-gate temporal-feature probe used only cached training-game
  detections and preserved reserved-game isolation. Fifteen trajectory and
  box-shape candidates cover 148/518 examples. The best fused model has
  held-game precision/recall .700/.875, .811/.857, .971/.957 and .562/.850;
  the worst fold is unchanged from the raw baseline. No source/schema change
  followed, so the completed regression and service results remain applicable.
- The official Online BootsTAPIR checkpoint verifies at 218,887,028 bytes and
  SHA-256
  `87c1e752cf5ce56e3e2f7da460aeb4d40fc826d04ef2939bade86a5c7495377f`.
  MPS produces NaN coordinates and zero visibility on the same correctly scaled
  seeds that CPU tracks with .877--1.000 visibility. The sealed bidirectional
  CPU probe contains 12/12 completed events, two positives, 417.862 seconds of
  inference and `runtime_consumable=false`. No one-dimensional trajectory gate
  approaches the required precision at full probe recall; best is .667.
- `pip check` and `git diff --check` still pass after the isolated dependency
  and artifact screen. No AGU source, API, configuration or runtime dependency
  changed, so the already completed 464-test/Harness/service gate remains the
  applicable source verification.
- A later disk audit confirmed that the BootsTAPIR checkpoint had no source or
  runtime consumer after rejection. Manual crop review had already shown that
  9/12 seeds were genuine basketballs while five false shot events still
  contained a real ball, so correcting seed placement cannot supply the missing
  shot-causality signal. The 218,887,028-byte checkpoint and download metadata
  were deleted; the sealed probe artifact and exact source revision/SHA remain.

2026-07-24 offline seed-verifier verification:

- Candidate-manifest internal hash
  `f6e1f0d8d8a6c829439da709024e4309901027f3fe1166821f337990086cc202`
  recomputes exactly; file SHA-256 is
  `887b3b5c1e4c1fa50e29ba76130d79eebf9ce97eb7c2403d71214a312d9d0efe`.
  Counts are 1,800 images, 60 games, 3,675 examples, 1,348 positives and 2,327
  negatives; duplicate image hashes are zero.
- Whole-list MPS inference exhausted unified memory and the first whole-list
  CPU stream was killed with exit 137. Single-image MPS inference with periodic
  cache release completed all 1,800 images without changing thresholds. These
  failures are experiment-runner constraints, not AGU runtime changes.
- Frozen-feature CV artifact file SHA-256 is
  `0b074478fca3a19f5bf7f848fd25a3286eecccb51db107f7c97e2307a54425d8`;
  internal artifact hash is
  `49b55ade3251f1cbb4cc7bf82cd903b1a619d09dff78fea20322dfc768c6f272`.
- The blind crop-review file SHA-256 is
  `de6818931768f6222cfb68d88d44ecf78f58af785614519d50edd45b94fd7f28`.
  The domain-transfer file SHA-256 is
  `cf4f55ddf4938f8b40a840cc685fd5e6fd509b46314c75c964959bb209c4cc6d`
  with internal artifact hash
  `338c63979d8d5837becd1ee404be044c6da00d17203e0896e27a47552467bcb1`.
- No source, schema, API, configuration or runtime dependency changed. The
  prior task-path Ruff, `git diff --check`, `pip check`, 464-test, Harness and
  completed local-service verification therefore remains the applicable source
  gate; this increment requires artifact and documentation checks only.

2026-07-24 release-chain and RTMPose research verification:

- Object-panel manifest file SHA-256 is
  `03066e3805023e68d7e9ddf813d37149805b388b282da7630c8d8811529ba4cb`.
  The Gemma and Qwen result file SHAs are
  `04d96771487508ecb3c491b017ca7315da4200c8ef90e83a3cbbd51f64b5ba6a`
  and
  `869a3db0162ccca71eeb6e2af5438bd4460e0ad893600ee9ea2605cc9371388a`;
  their internal artifact hashes recompute as
  `f9d8e961e447f821a2cacff9cb6c641b2ba3e8ade8e16f106786dc0a76ce6d90`
  and
  `42e66e154d871e65a2c235165df61f731e19c1dc3b7fe464d9b7f20f2ce5e84b`.
- The adaptive-anchor result file SHA-256 is
  `b8ec3a7e9b99e7365382770764942d2d5ef71c782321133d7d4b011e4316f8be`;
  internal artifact hash is
  `ea65da274273006c7d1bafe4f7c63a8601296b259e41b256769babd9cbb4062f`.
  Coverage is 106/518 and the best fused worst-fold precision is .544 at the
  recall gate, below its .557 reproduced baseline.
- The RTMPose archive is 52,089,522 bytes with SHA-256
  `55b81170e236040b59fc792ad0a8315301ac4c079a3bdb1095d838aad3088d18`;
  `unzip -tq` passes. The extracted ONNX is 55,685,444 bytes with SHA-256
  `26f3a19e61304a600dfb82d1001d41d24343b89fc70a33ffc84657e0b0bf2ecf`.
- OpenCV DNN CPU processed 91 unique frames and 397 player-box observations in
  7.7 seconds. The probe file SHA-256 is
  `0ca7b6cc40638ae878a40dd6161f94ba9577fda3b81718a6eb4104fb93b3a080`;
  internal artifact hash is
  `1330e62ef9b65e4bfeb1cbc45ead10de8637f73c64049fb8ccbd74b7a8cb4717`.
- The model-source manifest file SHA-256 is
  `2ea698547f6ff11996ee3fe4c22ba7fef23da8ed1c87c9120f3463bc05544608`;
  its canonical content hash is
  `7a933076df93fe348c80e6346f3343acaa2e926bfb0fab88eebe4bf7329d3468`.
- Every new result records `runtime_consumable=false`. Because this increment
  changed only offline research artifacts and documentation, it does not
  supersede the existing 464-test, Harness and completed local-service source
  gate.
- The SmolVLM2 source-manifest file SHA-256 is
  `26bfbfc1ceaf0c37870cc1f6bc88a92269e4ba9b42bbb7fae35e9243d73706a3`;
  internal artifact hash is
  `ca4f72bc3d967e7c20a8057be0b7eec356ad259cc3fc63ac9ba202ff5747c8e1`.
  The downloaded weight size and hash agree with the pinned Hugging Face asset.
- Raw-frame and short-video manifest file SHAs are
  `5433dd22637d7892951820f3be60e2feda5eb41a2a3c7db141563606144ee201`
  and
  `fa960cd7308aa853034c5e5be8f95b7283a646827245b51c2edd67736f7c22d6`;
  their internal hashes are
  `6787921d382b239955c33842e41f182f59b521a610d760b885feb5f02f88a3a3`
  and
  `7072d24bc3a3f1e1f171ef213dae7c31678be2fb9320c4a0f7ee651733edcfa8`.
- The four-event SmolVLM2 result file SHA-256 is
  `2e9610668241b573540f9635197616275640e91bbcf88cac4acb7fa1431a817c`;
  internal artifact hash is
  `eac0a042faab8bf93ea6c891fecfb8f3ea9fd525471effd64c8d78e27448ee31`.
  All four rows have `schema_key_mismatch`, so the strict score is 0 TP,
  0 FP and 2 FN rather than a natural-language reinterpretation.
- JSON parsing, five research self-hashes, both RTMPose weight hashes, ZIP
  integrity, `git diff --check` and the artifact-level Harness pass. No source
  or runtime contract changed, so a new service run is not applicable.

2026-07-24 WASB temporal-association research verification:

- Official checkpoint size is 6,102,633 bytes; SHA-256 is
  `8d1ba9870d0a6ab37b06ab82bed593c6c09133e713810bac475d0c000bb7e948`.
  PyTorch loads all 428 state entries and the standalone HRNet probe completes
  all 90 frozen frames on MPS.
- Review-package, offline annotation, WASB result and association-result file
  SHA-256 values are respectively
  `888f4b3e00610d0b30a356fe19983d69d1000f2126c759af7d5b74bf5ed7ea47`,
  `0b4ffa1cd4bb734c4b63ddbd102ac60469be5d3de1b75d941fea56272e1645bf`,
  `630b8e75986a7e91519d9c435b455418e0fa412c7e9045577ffccf73ce8a4ee9`
  and
  `a6cbc2c130a42b5ac005bf74297efbbfc2fa01995e1adc5b0f3f3a97208d6f1a`.
- The annotated event has eight visible-ball frames. Official-threshold
  recovery is 0/8; local-peak Top-1/Top-2/Top-10 coverage is
  5/8, 7/8 and 8/8. Frozen scene-reset/velocity association scores 7/8
  (precision=.875, recall=.875) and explicitly records
  `expansion_gate=false`.
- Every new artifact parses as JSON and records `runtime_consumable=false`.
  This increment changed no AGU source, API, configuration, dependency or
  runtime contract, so the prior 464-test/Harness/completed-service source gate
  remains applicable; artifact checks and the Harness gate are sufficient.

2026-07-24 expanded ball-fusion research verification:

- Offline visual-ranking, v2 annotation and frozen fusion-result file SHA-256
  values are respectively
  `0df208933468549e6794d8bf9e24e0b70e784ed34234628ab6d9a45fcba782ed`,
  `62a6269bc1a4f0fe94205d774a7f3cc0e5d43772d4224a33da6055bfded57f73`
  and
  `047b2397e3b93cf45b43068da9c02422a735ccc45d39528b9b22680dc2408759`.
  The evaluator hash stored before annotation write and recomputed afterward is
  `eda73e722b8a0789265c1f12751d98afa5313d4e1ab906159592fafa1d272375`.
- The v2 artifact contains five events, 34 reviewed frames, 30 visible boxes
  and four excluded occluded/not-visible frames. Its video, review, WASB and
  offline-ranking input hashes recompute exactly.
- Metric recomputation yields official WASB .300 recall; WASB raw-peak Top-2
  .900; BODD candidate Top-2 .867; offline visual candidate Top-2 .867; frozen
  WASB association .733; frozen fusion association .700. The result records
  `expansion_gate=false` and `decision=do_not_integrate`.
- JSON/schema invariants, file hashes, `git diff --check` and the artifact-level
  Harness gate are the applicable checks. This research-only increment changes
  no source/runtime contract, so it does not supersede the completed 464-test
  and local-service source gate.

2026-07-24 dense global-path validation and adapter verification:

- Every-frame development inference and the 540-configuration development
  search are sealed at
  `a2a074cd6cbc8af467126220d59dee1fd026c475c00657ae945189c75a995148`
  and
  `de2a47c3f7103c7edc2f9aa4c8d2896a2b1f20774bc7ba608bb67493d749b03c`.
  The selected Top-10/log-probability/40 px-per-frame soft-cap path reaches
  29/30 on the five-event development labels.
- Before new labels existed, the validation plan, dense inference and sealed
  path selections were written with SHA-256 values
  `7f2f22e024744dedb4d8d8a2b5ae66de6799da120881a683e317087e337e1a00`,
  `bd1a303cf485d7bea02744fb591f1f14b917393f3c28e6aa8bc2cf968bed0c15`
  and
  `54614c0b983f55877a0bd9a452a61db8adebbe6a323db3f0a2dbb05df2903cf8`.
- Codex then froze only ball visibility/location for 16 frames: ten visible and
  six excluded occluded/not-visible. Annotation and score-result SHAs are
  `b2c59a2bfb22f931aec8319123a28f5b3f8f8fee915de4a49f0832ea5e7a0386`
  and
  `7814974dc920fad7fd900a0a8a09d4c24d91666d05bad91545754b5768ced22d`.
  The sealed path hits 10/10 across two validation events, passing the
  predeclared .85/default-off-adapter gate.
- TDD first produced the expected import failure, then three focused global
  path tests passed. The final implementation uses backpointers rather than
  copying full paths, bounding state to Top-10 per frame. Touched-path Ruff,
  20 related tests, all 467 tests (15 existing warnings), `git diff --check`
  and `verify_harness.py --run-tests` pass.
- Local port 8771 returned `ok`/`ready`; task
  `5a78dd2ba4d3417bb83f19f0e318fa78` completed at 100% with a result and no
  error. README/API review found no public request, response, environment or
  default-runtime change.

2026-07-24 cross-game global-path verification:

- v1 recomputation gives 10/11 (.909), accuracy gate true, sample-size gate
  false and overall validation false. Per-event visible counts are 2, 4 and 5,
  proving the predeclared per-event failure is retained.
- v2 recomputation verifies all plan/selection hashes before scoring and gives
  12/13 (.923). Per-event results are 4/5, 4/4 and 4/4; all three have at least
  three visible frames, the total exceeds 12, and both sample and accuracy
  gates are true.
- The v2 annotation/result hashes are
  `bf2e4f188ab0a56b1ac59830779b64fb9f59bbeee806800a1963e5dc98c623d0`
  and
  `057f87965016f41c72a1036f16e2d8b38162116391ef1446c6e8b2da37c35aa2`.
  JSON parsing and `git diff --check` pass. This artifact-only increment changes
  no source or runtime contract, so the prior 467-test, Harness and completed
  local-service source gate remains applicable.
- TDD for the downstream bridge first failed on the absent
  `ball_path_selector` argument, then passed after the opt-in implementation.
  Ruff and all 35 ball-path/candidate tests pass. The project-resolved OpenCV 5
  environment reports the three already documented missing
  `CascadeClassifier`/legacy tracker failures (465 other tests pass); the
  established `opencv-contrib-python-headless<5` compatibility run passes all
  468 tests with 15 existing warnings.
- Harness with the compatibility-backed full-test command and
  `git diff --check` pass. Local service port 8772 returned `ok`/`ready`; task
  `72d40206be2446c5ac75712f9af605ab` reached `completed` at 100%, with a result
  and no error. README and API review found no public request, response,
  environment variable or default-runtime change.
- Default-off downstream diagnostic and evaluation artifacts parse as JSON and
  have SHA-256 values
  `8139f11c63e3caf0b490204a6094b47a17c05dffb491952fd16b7b3a9077d08e`
  and
  `d93a6ae5b0c9b05133d1db91110a5be4820e5121d72f29d343d17218fe358c71`.
  The evaluation records its non-blind limitation, 1/3 FGA precision and zero
  resolved outcomes on the one true FGA; it is not an acceptance metric.

2026-07-24 MEM-OKC component-score and VLM-fusion verification:

- The frozen blind-plan, selection, score-evidence and evaluation artifact
  SHA-256 values are
  `6b14be07f84695422f0828ba909ee5b99064b7cb3bec325866e908990db8128e`,
  `9fa6274dff1c9a09bfd084fed96c2190a130bed6cd22693732934c3c26c20989`,
  `fd6ca7140a831fb6ab4d759b61c90fdba0a79321b3e5e57ca1fd47596b04ffc9`
  and
  `f04a62902ad2d6d7828432625c3f44b343a75bd762e1887a01b681fb447ae916`.
- Frozen predictions were made/FT/+1, unknown FGA and unknown FGA. Post-freeze
  review found a made first free throw, a live missed field goal and a replay/
  dead-ball duplicate. The score gate is 1/1 correct at 1/3 coverage; live-FGA
  outcome coverage is 0/1, so the blind gate does not promote the path.
- The non-blind score-first plus Qwen3-VL evaluation SHA-256 is
  `006462cf4a44a4fe5b9bc3898e96bdf5baa99ddf604cf420abe7bf7e48a48646`.
  It records 2/2 correct selective decisions at 2/3 coverage and preserves the
  replay as unresolved.
- TDD coverage for component reconciliation and the targeted candidate runner
  passes 22 focused tests. The full OpenCV<5 compatibility suite passes all
  474 tests with 15 existing warnings; Harness with the same full-test command
  passes.
- Local service port 8772 returned `ok` and `ready`; task
  `14af9bedac0e4456ae43928daf432dc2` completed at 100% with a result and no
  error. README/API review found no public request, response, configuration or
  default-runtime change.

2026-07-24 replay-logo proxy and OpenCV compatibility verification:

- Replay-transition/logo focused tests pass 15/15 and Ruff passes for the new
  offline modules, scripts and tests.
- The sealed four-video evaluation covers 63 explicit windows: 38 live FGA and
  25 replay/highlight. It reports TP=4, FP=3, FN=21, TN=35, precision 0.571,
  recall 0.160 and `promotion_eligible=false` against the 0.95 precision floor.
- The three false positives are namespaced by raw-video hash in the evaluation
  artifact, preventing same-named candidate IDs in different games from
  colliding.
- An unconstrained OpenCV 5 environment reproduced three unrelated missing-API
  failures. After pinning both OpenCV wheel families below 5 and refreshing the
  lock, normal `uv run pytest -q` passes all 490 tests with 15 existing
  warnings. Ruff and Harness pass.
- Local service port 8765 returned `ok` and `ready`; task
  `bf0377f2fc8d409ab686dc84b0a8e875` completed at 100% with a result and no
  error. README/API review found no request, response or default-runtime
  behavior change.
- The downloader regression adds a finite-inner-retry and curl/IPv4 command
  contract test. Three downloader-focused tests and eight public-research tests
  pass. Both native and curl live attempts fail before receiving bytes from the
  current Googlevideo CDN, so the new plan's media gate correctly stays false.

2026-07-24 broadcast-clock replay verification:

- Nine focused unit tests cover split/merged clock OCR, period confusion,
  invalid clocks, the one-way frozen-prefix rule, fail-closed counterexamples,
  hash tampering and SHA-bound label evaluation. Touched-path Ruff passes.
- Four independently sealed evidence artifacts cover 17, 11, 19 and 16
  windows. Their evaluation SHA-256 is
  `c382346d5f00b431532a367f03ad350af73e980452e1f4681408b99632f0385c`
  and records `2/0/23/38` TP/FP/FN/TN.
- The evaluator reports per-video counts and requires at least five replay
  predictions in addition to precision `>=0.95`; only two exist, so promotion
  remains false.
- The separate MEM-OKC evidence artifact
  `78588fb0b509734b386f0b0c2feb190d02e5a40342449b619a5114de203dec67`
  abstains on all three post-freeze windows. No reference or label is consumed
  by the evidence runner.
- Touched-path Ruff, all 499 tests (15 existing warnings), Harness and
  `git diff --check` pass. Repository-wide Ruff still reports 49 pre-existing
  style findings in untouched legacy files and was not auto-fixed.
- Local port 8765 returned `ok` and `ready`; task
  `ebe7cc2550854096ac51a6e338170567` completed at 100% with a result and no
  error.
- A fifth IPv4/external-curl media attempt again failed before receiving bytes
  with Googlevideo `SSL_ERROR_SYSCALL`; the plan-bound download state remains
  failed and the blind media gate remains false.

2026-07-24 EBQwen independent raw-frame verification:

- Five focused tests cover plan leakage rejection, deterministic stratified
  selection, canonical SHA tamper detection, exact prediction coverage and the
  fixed per-video promotion gate.
- Frozen plan, prediction and evaluation file SHA-256 values are
  `357010d2567de5a3b6f2c94ca469039e2cf13ad36f9d0e169dad0eac83fb9691`,
  `2effd63912e33677ffc171f7f08c938a44dda9308c863dd7f1f1d92a75b70301`
  and
  `7e2d779c96c1edd8388883ec02e0e039bdd156a3df10750b1e6d9adf13404312`.
  Their canonical internal artifact hashes are independently verified by the
  readers.
- All 32 planned windows have predictions. Evaluation records
  TP/FP/FN/TN `16/12/0/4`, precision `0.5714`, recall `1.0000`, no unknowns,
  and `promotion_eligible=false`.
- Touched-path Ruff passes. The full suite passes 504 tests with 15 existing
  warnings. All experiment artifacts remain runtime-ineligible and explicitly
  record that Codex supplied no runtime answers.
- The sixth blind-media attempt again failed with zero received bytes and
  Googlevideo `SSL_ERROR_SYSCALL`; no blind evaluation was claimed.

2026-07-24 local-training resource guard verification:

- Seven focused tests cover invalid thresholds, sustained-pressure debounce,
  recovery reset, low available memory, structured JSON, child exit-code
  propagation and termination of the supervised child on a forced limit.
- The supervisor records host CPU/memory, available bytes and process-tree RSS
  as JSONL. It terminates only its own process group after three consecutive
  default breaches and returns 75 on a resource stop or critical sampler error.
- A macOS sandbox denial while enumerating descendants now falls back to the
  directly supervised process RSS; this was reproduced by the integration test
  before the fix.
- Current no-training sample: 75.7% host memory, 4,176,723,968 available bytes,
  21,889,024 direct process-tree RSS and 13 GiB filesystem free. CPU's first
  non-blocking sample is intentionally a baseline and sustained decisions use
  later samples.
- Touched Ruff, all 511 tests with 15 existing warnings and Harness pass.
  Local port 8765 returned `ok`/`ready` and accepted task
  `29feb7998fe941609c9685e1df2d1079`. Subsequent status polling could not cross
  the execution sandbox's loopback namespace; a requested non-sandbox rerun was
  not authorized, so the task is recorded only as accepted, not completed.

2026-07-25 EBQwen native-video verification:

- Fourteen focused tests cover sealed native-plan derivation, exact
  annotation-source linkage, processor pixel limits, deterministic temporal
  sampling, malformed-output abstention, single model loading, native tensor
  API shape, generation-error preservation and exact prediction coverage.
- The 32-window prediction/evaluation/resource-log SHA-256 values are
  `7aa704bb9eaf7768f545f6b4f03b66bb01a42f0fc07e371e114f4b89bb9b94ec`,
  `302696e745ca0649ca3ae56f21d89908a25e98a23f13095e9a58ed0eb6dd6c26`
  and `30ca1ab50decc1717a0d07e7272bf73eb84338fbced4d7ccb4c89627692eb7cb`.
- The guarded run exited 0 with 85.2% peak system memory, 2,549,530,624
  minimum available bytes, 19.9% peak CPU and 1,850,474,496 bytes peak
  process-tree RSS. Evaluation reports TP/FP/FN/TN `16/16/0/0`,
  precision `0.5000`, recall `1.0000`, and rejects promotion.
- Touched-path Ruff and the combined public-research/native-VLM suite pass
  (`29 passed`). The full suite passes `527 passed` with 15 existing warnings;
  Harness, `pip-audit` and `git diff --check` also pass. A local service smoke
  returned HTTP 200 with `{"status":"ok"}` and `{"status":"ready"}` before
  clean shutdown.

2026-07-25 staged jersey-cache verification:

- TDD first reproduced the missing prefill module as a collection error.
  Focused verification now passes 14 tests across the new selector, resumable
  cache and official identity graph; touched-path Ruff also passes.
- The post-change repository suite passes `541 passed` with 15 existing
  warnings; Harness, touched-path Ruff and `git diff --check` pass.
- Tests require namespace-bound event counts, duplicate suppression within one
  event, one best segment per raw source, exclusion of existing gallery
  identities and quality/coverage ranking.
- The 2B cache/resource SHA-256 values are
  `8041f37a8534f8e4d6b7b4da1ad62be7195de62dc60f8837100ad13aff4b8338`
  and
  `95f22b7a61804cdf397ae05418e6408abcb8444c510b3f124d0ca3fa5b5f222d`.
  It completed 6 selected tracks with 0 trusted reads and no resource breach.
- The 4B 8192-context resource log SHA-256 is
  `d8ec18727c337e19518ad8b2c0694a32253d97c7eeb14802a9d65862972a395c`;
  the guard stopped it after three sustained low-memory samples. Ollama was
  explicitly released afterward. These are failure diagnostics, not accuracy
  evidence or a promotion result.

2026-07-25 conservative face-anchor verification:

- The team-binding regression fails against the prior two-anchor default and
  passes after the default support floor is raised to four. The related
  identity/candidate suite passes 51 tests.
- The final identity graph file/resource SHA-256 values are
  `983551d49064b48505e81fbfa2d31b2c65b62f64913bf22685dcea6449105244`
  and
  `d283771d34dfc3aafb473ede256a77764e71247f20c0bebe81d0f3caeb5b1cc8`.
  It has 14 anchors/11 people, no ambiguity and no inferred team binding.
- The final candidate file/resource SHA-256 values are
  `8fe345857a6bdedcc32f54ada34f776d6caa4ca511b42751fe6511d8097375ff`
  and
  `aa63213987cf25af03c6c22562c87b5223c8ad3d1ed991f5ba357160e113a66d`.
  Event counts remain 197 FGA + 193 rebound.
- Frozen v5 canonical SHA verification, touched-path Ruff and all 541 tests
  pass. No benchmark answer or Codex actor label was opened.

2026-07-25 cache fallback and swap-guard verification:

- RED tests first required a cache-aware reviewer to return an exact primary
  hit without invoking either backend, route only misses to fallback, and
  preserve distinct primary/fallback model provenance. The focused official
  inference/script suite passes 70 tests.
- The 2B 512/8192 actor probe produced one exact fallback cache entry with
  cache SHA
  `3f13796af8f1cf6e2c8b247e2c354b971a244ac333c941bfa512d07f8dbbbecb`
  and resource-log SHA
  `0150664a9202123f709c38407c9b24f67ad68a6d821a746882bdf35b8a99eb98`.
  It returned `available=true`, confidence 0.95 and both actor/team labels,
  without reading truth.
- RED tests also required a low-free-swap breach and swap-aware restart wait.
  The resource-guard/resumable-supervisor suite passes 13 tests; snapshots now
  include swap bytes/percent and lifecycle logs include the configured swap
  threshold.
- Touched-path Ruff passes. The complete suite after both changes passes
  546 tests with 15 pre-existing warnings.
- The candidate-104 semantic probe returned `available=true`, confidence 0.95,
  `live_game_action=true` and `replay_or_highlight=false`; its cache and
  resource-log SHA values are
  `8ac85d2244da4b92c9a064c7aad2435fef933b84ce5221a8f85dd6fcfc8c0beb`
  and
  `830dfda96b3008c7d37420c5449dd00b0f7f83ad48ac313914f0ff376377ac37`.
- Frozen v10 canonical SHA is
  `543a0b9342de046cde1c61e22d80696e76a014d8fdec2f779a4ae670cd7d7edb`.
  The first-game prediction remains unsealed and truth remains closed while the
  resumable run fills exact semantic/actor cache misses.

2026-07-26 blind-count and free-throw correction verification:

- RED tests first reproduced both failures: an explicit live free throw was
  rejected instead of retained under its correct event type, and missed
  free-throw parents did not expose bounded rebound context. Separate
  counterexamples require rejection when the explanation lacks subtype
  evidence, the scene is dead-ball, or the clip is a replay.
- Focused official inference/script/count-audit verification passes `78` tests;
  touched-path Ruff passes. The complete repository suite passes `554` tests
  with 15 existing warnings.
- The post-reveal development replay output SHA-256 is
  `99dbbe97b2173c71467ff813d8089bc70fdc9b2b4317e3e61d8fe86550472ad0`.
  Its guard-log SHA-256 is
  `1fbddb1407603e1c60f6989e846162e03f0acaf176d2e0f04da860d553b92430`;
  22 samples recorded zero breaches, at least 4,804,214,784 available bytes,
  at least 1,340,735,488 free-swap bytes, at most 61.6% CPU, at most 72.0%
  system memory and at most 703,856,640 process-tree RSS.
- Blind baseline and development complete-scope audit SHA-256 values are
  `3ab2ff4179a4ff9b42ac29900f19fe376e8997a279ae9fbff9d81754a5ee8bfb`
  and
  `cfca744e34acf9c2a75142317aea765188069b282b11534b0ceaedb344a28be5`.
- Local `/health` and `/ready` returned `ok`/`ready`; task
  `226a37933ea64a159632da63b9d8b40f` completed at progress 100 with no error.
  README and API docs were checked; no public request, response, startup,
  configuration or checkpoint contract changed.

## 2026-07-26 HOU-ORL enrollment and canonical environment

- 33 focused identity/roster tests passed.
- The reviewed gallery covers 17/17 players in the hash-bound G3/G4 observable
  scope; G1/G2 were not opened, decoded or inspected.
- `.venv/bin/python -m pip check` reported no broken requirements.

2026-07-27 explicit shot-state and native EBQwen backend verification:

- Label-free 15/30/60-second MViT/Swin neighborhood pooling failed to improve
  the weakest held game beyond precision .548 at recall .85.
- Training-only multiclass state screening (real shot, free throw, replay,
  stoppage and no release) reached only .602 weakest-game precision at recall
  .85 and was not promoted.
- Requiring every EBQwen causal observable improved the frozen 32-window
  precision only from .571 to .588 while recall fell from 1.000 to .625; the
  parser and runtime contract were not loosened or promoted.
- Added a Transformers native-video reviewer for the verified 7 GiB BF16
  EBQwen weights, with MLX remaining the default backend. A one-example plan
  is mechanically marked `resource_probe_only=true`.
- The BF16 load did not start with 3.51 GiB available RAM and 1.25 GiB free
  swap, below the predeclared 6/2 GiB start line.
- Focused independent-VLM tests passed 16 tests. The complete suite passed
  `600 passed, 15 warnings`; Harness, pip check, touched-path Ruff and
  `git diff --check` passed.
- Local service port 8765 returned healthy/ready. Task
  `6ca6433a4e8146bea2c4d1fd61fdf7de` completed at progress 100 with no error,
  and the service shut down cleanly.
- `.venv` imports `mlx_vlm`, `mlx_whisper` and `yt_dlp`; the project-level
  `venv/` path is absent.

2026-07-27 BasketEvent and three-phase scene-state verification:

- A RED test first proved no governed BasketEvent entry existed. The catalog
  now records fixed Git/Hub revisions, missing licenses/media, the 1.8 GB
  checkpoint and a fail-closed no-download/no-import decision.
- Four scene-state regressions cover deterministic 15%/50%/85% frame indexes,
  training-only hash verification, game-held OOF provenance and rejection of
  misaligned fusion artifacts. Together with the existing video-backbone tests,
  `9 passed`; touched-path Ruff passes.
- The sealed scene artifact contains 511 examples, 288 positives and four
  development games. Artifact SHA is
  `ec4f4b881dc0398de2dba6d1eccbac327a0c800b09ed5e2724a57b0f0ec3a3a3`;
  file SHA-256 is
  `e0af956658ca8950204482fcd29c1e6f0d231cf0bf000386f97c0f62985a9f6c`.
- The fixed PCA-16/C=0.01 fusion result SHA is
  `9b2b6b2f8888046286f867b22af8f839aee1a2bdaf5bfd803730a6e3b445111c`;
  file SHA-256 is
  `3cef3e5a80818a72cca12ef935436aaa1d2b5bcd30e9f72e6250ecdddf222a5c`.
  Best scene+MViT weakest-game precision/recall is 0.6422/0.8525, so
  `promoted=false`.
- Extraction recorded zero guard breaches, at most 82.8% system memory, 71.2%
  CPU and 1,213,054,976 bytes process-tree RSS. The screen likewise recorded
  zero breaches. HOU-ORL G1/G2 were not opened.
- The complete canonical `.venv` suite passes `604 tests` with 15 existing
  warnings. Harness, dependency consistency, touched-path Ruff and
  `git diff --check` pass.
- Local `/health` and `/ready` passed, and analysis task
  `b5602a7ec68844cdb65b0137143a6ebf` completed at 100% with no error.

## 2026-07-27 ball-release review verification

- Six focused annotation/review-sheet tests pass and touched-path Ruff passes.
  Contracts reject hash mismatches, label exposure, plan overlap, non-rendered
  path labels and non-conservative correction inputs.
- Batch plan hashes are `63230c622a4cb5074387ab0ec6bf63243215b50815cc2f5a06f39d2c5812f875`,
  `9a63e8ae831a5135c27f0f1f75aaf17fcfa034dcce10e24d1b73a494c26c5733`
  and `3b047ee2eca79c639f74499fe9f63f86e1335014b52aac28efb2d200887da7cf`;
  each contains 32 examples and later plans exclude all prior rows.
- The final corrected fusion artifact SHA is
  `fb279bd51e5aa626e45a17af60811e1f16694427f03766c14a75b6f134ab74d9`.
  Pooled precision/recall/F1 is 0.7705/0.9144/0.8363 and `promoted=false`.
- Final-screen process-tree RSS peaked below 0.80 GiB, system memory below
  81%, CPU below 38%, and no resource threshold was breached.
- The full canonical `.venv` suite passes `610 tests` with 15 existing
  warnings; Harness, dependency consistency, touched-path Ruff and
  `git diff --check` pass.
- This work is offline training/evaluation only; API behavior is unchanged and
  no additional service smoke is required.
- The full `.venv` regression suite passed `577 tests`; repository Harness
  passed after the environment consolidation.

## 2026-07-26 guarded secondary-event audio verification

- RED tests first proved that foul calls were absent, retrospective foul
  counts were accepted, action-scoped candidate generation was unavailable,
  and offline reviews lacked a complete hash-bound contract.
- Ten focused audio, roster and review tests pass; touched-path Ruff passes.
- The full `.venv` suite passes `583 tests` with 15 existing warnings.
- Tiny ASR completed in 24 guard samples. Base-English first stopped safely at
  exit 75 on three sustained free-swap breaches, then completed after resource
  recovery without changing the 2 GiB RAM, 0.5 GiB swap, 90% memory or 95% CPU
  limits.
- Base-English output has 454 events and all 8 required event types. The
  optimistic complete-scope count-only result is TP<=263, FP>=191, FN>=115,
  precision<=0.5793, recall<=0.6958 and F1<=0.6322.
- RapidOCR period/clock matching read only 2/57 foul candidates, so its apparent
  one match was explicitly rejected as insufficient evaluation evidence.
- The sealed 57-row Codex review contains 36 live-current, 16 non-event and
  five uncertain labels and remains disconnected from runtime inference.
- Local `/health` and `/ready` returned their expected states, and VLM-off
  analysis task `23fc738fea9a478c998744b79376f678` completed at 100% with no
  error. README/API consistency review documents the repeatable
  `--candidate-action` scope and the non-runtime review contract.

## 2026-07-26 one-to-one commentary evidence verification

- Two new RED-to-GREEN regressions cover live field-goal result grammar,
  rejected non-event contexts, deterministic best-overlap assignment, and
  evidence-only shot outcomes.
- Focused audio/roster/review verification passes `11 tests`; touched-path
  Ruff and `git diff --check` pass.
- The full canonical `.venv` suite passes `585 tests` with 15 existing
  warnings; repository Harness passes.
- The v7 candidate bundle SHA-256 is
  `b1a2b55f3a4083bfc544aff91b4c5a06cfc46132961cb7828d26ad5589b4aa96`.
  It keeps 454 events and the fixed complete-scope count-only F1 upper bound
  at 0.632212 while increasing audio-linked FGA from 11 to 17.
- Local `/health` and `/ready` pass. VLM-off task
  `49c398afe31e4a48876d6d5caaf22d6b` completes at 100% in 25.91 seconds with
  no error; the service is then shut down cleanly.
- The 518-row interaction diagnostic preserves four source-video folds and the
  original 300-positive/218-negative labels. Coverage is 370/518 for original
  player-ball observations and 146/518 for uniform-window observations.
  Interaction-only worst-fold precision is 0.557 at recall >=0.85, versus the
  existing approximately 0.562 baseline; no source or model artifact is
  promoted.
- TOTNet loads the exact pinned tennis checkpoint successfully in `.venv`.
  MPS fails closed on unsupported 3D adaptive pooling. CPU inference completes
  one window in 17.88 seconds; a three-window batch is stopped after four
  minutes at about 2.2 GB RSS. No benchmark media, labels, or runtime path were
  exposed to the model.

## 2026-07-27 compact Qwen3-VL independent-review verification

- A RED import failure preceded implementation of the frame-sampled derived
  plan. The new regression proves source-plan binding, exact example
  preservation, label omission, explicit 4-frame/512-pixel sampling and
  rejection of fewer than two frames.
- All 17 independent-shot-VLM tests pass. Touched-path Ruff and
  `git diff --check` pass; the full canonical `.venv` suite passes
  `618 tests` with 15 existing warnings.
- The 32-row prediction artifact exactly covers the derived frozen plan and
  was sealed before the eight SHA-bound training annotations were supplied to
  the evaluator.
- Evaluation precision/recall is 0.4545/0.3125. All four per-game gates fail,
  so `promotion_eligible=false`.
- Resource logs record safe stops for infeasible 8192/4096 configurations and
  resumable 3072-context batches. The successful final batch stayed below the
  88% system-memory ceiling and above the 2 GiB available-memory floor.
- This is offline-only code and artifact generation; API behavior did not
  change, so the local service smoke requirement does not apply.

## 2026-07-27 continuous reason-evidence verification

- RED first failed on the absent motion/temporal feature keys. Three evidence
  tests and one nested-fusion test now pass; the synthetic fusion test proves
  both the outer held game and the scored game are excluded from reason-model
  training provenance.
- The evidence artifact contains 511 examples and 39 features under schema
  `agu.shot-reason-evidence.v2`; its internal SHA-256 is
  `ab5e5fb4d63eec11d0438a101deaeab8867518aa8384365b64e1cd260957ef9f`.
  The fusion artifact SHA-256 is
  `9efea8da9bcf72c04e021a4b7cdb7627c12b5b727c63be29ca22f764f923901c`.
- The frozen screen rejects all four variants. The best selected raw-feature
  variant has pooled precision/recall/F1 0.7655/0.9144/0.8333 and weakest-game
  precision 0.7191; `promoted=false`.
- Guarded extraction and fusion recorded zero breaches. Final extraction
  peaked at about 1.60 GiB process-tree RSS and 73.9% system memory; fusion
  peaked at about 0.66 GiB RSS and 73.7% system memory.
- Touched-path Ruff, dependency consistency and repository Harness pass. The
  full canonical `.venv` suite passes 622 tests with 15 existing warnings.
  This is offline-only behavior, so the API smoke requirement does not apply.

## 2026-07-27 Basketball-51 transfer and fourth-review verification

- RED-to-GREEN regressions prove inverse-frequency sampling equalizes expected
  binary class mass and that target transfer evaluation fails closed on
  per-game gate failure or prediction-length mismatch.
- Guarded balanced MViT training reaches source-fold precision/recall/F1
  1.000/0.927/0.962 with zero resource breaches. The checkpoint remains
  training-only.
- The frozen 95-row target-domain screen rejects transfer at
  precision/recall/F1 0.175/0.625/0.274, ROC AUC 0.496 and average precision
  0.168. No threshold satisfies both 0.85 precision and 0.85 recall.
- A fourth disjoint 32-window review seals all decisions against plan and
  review-sheet hashes. Conservative correction resolves 29 rows, changes 10
  and leaves three edge-of-window releases unresolved.
- The updated base scene+Swin+MViT game-held screen reaches pooled
  precision/recall/F1 0.780/0.919/0.844, but per-game precision bottoms at
  0.727. Nested reason fusion is worse at F1 0.831; neither is promoted.
- Eighteen focused tests, touched-path Ruff, dependency consistency,
  `git diff --check` and Harness pass. The full canonical `.venv` suite passes
  633 tests with 15 existing warnings.
- This increment changes only offline training/evaluation artifacts. No API,
  runtime model, HOU-ORL blind media, prediction or truth is touched, so the
  local service smoke hook does not apply.

## 2026-07-27 label-free broadcast-state verification

- RED first failed on absent broadcast-state and fusion modules, then on the
  absent resume loader. Fifteen focused clock/state/fusion/script tests pass.
- Six sealed RapidOCR artifacts cover the scene manifest exactly: 511 rows,
  zero missing, extra or duplicate `(video SHA, candidate bundle SHA,
  event_id)` keys. The derived 21-feature artifact SHA-256 is
  `a002ca21a132970c85b0f2b78c5b81ca80d284e617e3c25e5bb029dc726473bb`.
- A real zero-work resume smoke restored all 24 rows in the smallest shard,
  processed zero rows and reproduced artifact SHA-256
  `e61c30bc958d34ea695028cf904eb74d9ec7391e539a50f4813be5920a28a8e3`.
- The frozen four-game screen improves base P/R/F1
  `0.7801/0.9190/0.8439` to `0.7897/0.9271/0.8529`, but weakest-game
  precision is `0.7326`; the fusion artifact SHA-256 is
  `0c8776d61fd50cf449cc675a8bcc0f49e5db306ab972f7476c06d268eaa40164`
  and `promoted=false`.
- Guarded extraction peaked at 1,979,072,512 bytes process-tree RSS, 80.9%
  system memory and 92.3% CPU, with zero breaches. Guarded fusion also
  recorded zero breaches.
- Touched-path Ruff, dependency consistency, `git diff --check` and Harness
  pass. The full canonical `.venv` suite passes 639 tests with 15 existing
  warnings.
- This increment is offline-only. It changes no API or runtime model and
  never reads HOU-ORL blind media, prediction or truth, so the local service
  smoke hook does not apply.

## 2026-07-27 causal overlay and independent ball-detector verification

- RED first failed because overlay state/fusion modules were absent. A real
  fusion CLI run then exposed a wrong `best_variant.promoted` lookup; its
  regression failed with the same `KeyError` before the code was corrected to
  read `best_variant.gate.promoted`. Five focused tests pass.
- The 22-feature artifact exactly covers 511 examples and has SHA-256
  `608685aa2fff94644fab5782dc2623c3a714be8d7ebb546480621a0e2ffe6484`.
  Label fields, duplicate keys, missing videos, malformed score timelines,
  ambiguous shot-clock suffixes and hash changes fail closed.
- Fixed game-held fusion gives P/R/F1 `0.7747/0.9190/0.8407` for
  `base+overlay_raw` and `0.7801/0.9190/0.8439` for
  `base+broadcast+overlay_raw`. Both lose to `base+broadcast_raw` at
  `0.7897/0.9271/0.8529`; all variants remain unpromoted. Fusion artifact
  SHA-256 is
  `d8456995c6b6b36b55a59eb6fd7cf2e990a5f2d8b15e8d45e3c42880f53dde01`.
- The missing LAL-BOS score timeline resumed from its frame cache and sealed
  619 reads. Guarded execution peaked at 646,004,736 bytes process-tree RSS,
  74.9% system memory and 34.5% CPU with zero breaches.
- The single-game MIT YOLO11m ball weight first stopped safely after three
  consecutive samples below 2 GiB available memory at 1280/batch 8. The
  704/batch-2 retry completed on all 180 E-BARD test images at
  precision/recall/mAP50 `0.1213/0.0395/0.0149`. The rejected weight and
  generated label cache were deleted.
- Full canonical `.venv` verification passes 644 tests with 15 existing
  warnings. Touched-path Ruff, dependency consistency, `git diff --check` and
  Harness pass.
- This increment is offline-only and changes no API/runtime model. HOU-ORL
  blind media, predictions and truth remain untouched, so the local service
  smoke hook does not apply.

## 2026-07-27 PBP visual-state verification

- RED collection initially failed because `app.analysis.pbp_visual_state` did
  not exist. Fourteen focused tests now cover strict clock parsing, state and
  ambiguity rules, closest-read selection, full event accounting, legacy blank
  action types, game/action validation, blind-video rejection and tamper
  detection.
- Four real builds produce 12/37/9/79 determinate examples for ATL enrollment,
  ATL benchmark, LAL enrollment and LAL benchmark respectively. Their total is
  137 with 26 field goals, 84 free throws and 27 foul-only states.
- The first real build exposed blank `actionType` values in legacy NBA steal
  and block rows. They are retained as auditable `Unknown` actions and can only
  yield the excluded `other` state; descriptions are never parsed.
- Small held-game feature screens used `/usr/bin/time -l`: the largest screen
  used 324,075,520 bytes maximum RSS with no swap. Two three-row EBQwen probes
  also used no swap and were stopped after demonstrating contradictory or
  invalid state outputs.
- This work is offline training/evaluation only, so the local FastAPI smoke
  hook does not apply.

## 2026-07-28 EBQwen adaptation and temporal-state verification

- Eight focused tests cover plan-order selection, exact correction/sheet
  coverage, blind/mismatch rejection, contact-sheet row geometry, manifest
  sealing/tamper rejection, five-panel splitting, model pooling, image-hash
  loading and preprocessing.
- The generated dataset contains 81 examples with exact class counts
  41/40 and four source counts 27/5/6/43. Manifest verification reproduces
  SHA-256
  `5a00875f6611558cc9309623ff84be99b03538d7331107e915da6d227f25f1e8`.
- The one-step EBQwen LoRA feasibility probe exited 75 under the resource
  guard before writing an adapter. Peak system memory was 92.5%, minimum
  available memory was about 1.2 GiB and minimum free swap was zero.
- A one-epoch head-only smoke completed with worst-game balanced accuracy
  0.500. The fixed 12-epoch last-block run completed with worst-game balanced
  accuracy 0.6291 and `accepted=false`; screen artifact SHA-256 is
  `6d68c4365efd2569f853c78c91542fbe34a943d813a6b55eb8e0eaa75a6b9f57`.
  Peak process-tree RSS was about 0.741 GiB, peak system memory 71.0% and no
  guard limits were breached.
- Smoke/final fold checkpoints were deleted after rejection and both output
  directories now retain only `screen.json`. This work changes no API or
  runtime behavior, so the local service smoke hook does not apply. HOU-ORL
  G1/G2 remain sealed.
- Focused Ruff and all eight focused tests pass. The canonical `.venv` full
  suite passes `690 tests` with 15 existing warnings; `pip check`,
  `git diff --check` and repository Harness verification also pass.

## 2026-07-29 expanded review and domain-screen verification

- RED collection first failed with the expected missing
  `app.analysis.visual_state_domain_screen` import. Four focused domain-screen
  tests now cover follow-up-only overlays, exact unresolved coverage, blind
  rejection, embedding-dimension mismatch, nested game isolation, acceptance
  calculation and artifact tamper rejection.
- Eight existing review tests cover follow-up parent provenance, fixed
  nine-frame offsets, unresolved-only selection and rendered sheet geometry.
  The combined focused set passes 12 tests and touched-path Ruff passes.
- The real screen verifies 256 source and 108 target rows, four target groups,
  identical 768-dimensional backbone identity and both correction hashes.
  Artifact SHA-256 is
  `6a1da45851cb9200f95ec38fda1b7c5942950cdd5907dfd6d20d0fe6d3acde0c`;
  `accepted=false`.
- `/usr/bin/time -l` reports 512,950,272 bytes maximum RSS and zero swaps for
  the reproducible MViT screen. The 39-feature and corrected pose-layout
  diagnostics also use zero swap. No checkpoint or runtime artifact is
  produced.
- This increment changes only offline training/review code and documentation;
  the FastAPI local-service smoke hook does not apply. HOU–ORL G1/G2 remain
  sealed.
- The canonical `.venv` full suite passes 697 tests with 15 existing warnings.
  `pip check`, full `git diff --check` and repository Harness verification
  also pass.

## 2026-07-29 court-topology source verification

- Catalog RED first failed because the two new fail-closed sources were
  absent. The focused catalog suite passes after adding immutable governance
  entries for the YOLO11n court-keypoint release and KaliCalib/DeepSport.
- A 12-frame label-hidden risk probe produced 9 detections and exposed
  high-confidence closeup false positives. The complete 108-row screen
  produced 98 detections, 57 conservative valid rows and four held-game
  balanced accuracies `0.562/0.583/0.750/0.597`.
- `/usr/bin/time -l` reports 758,824,960 bytes maximum RSS and zero swaps for
  the corrected full screen. The first attempt stopped before fitting because
  the asserted empty-detection feature length was wrong; it wrote no model or
  result and the corrected run used the same frozen protocol.
- No API/runtime behavior changed, so the local service smoke hook does not
  apply. HOU–ORL G1/G2 remain sealed.

## 2026-07-30 whole-game E-BARD + MUVY detector verification

- RED collection first failed on missing manifest seal/verify functions. Four
  focused tests now cover canonical game-ID parsing, deterministic disjoint
  44/8/8 grouping, ball-only filtering including the known ignored zero-height
  player row, and hash-bound manifest verification.
- The materializer verified both source manifests, all 1,969 output file
  hashes, every hard-linked image inode, 44/8/8 game disjointness and exact
  1,458/271/240 split counts.
- One-epoch smoke completed with no corrupt images and validation
  `0.543/0.498/0.459/0.186` P/R/mAP50/mAP50-95. Formal training early-stopped
  at epoch 30 and selected epoch 21 at
  `0.845/0.586/0.682/0.326`.
- Independent whole-game test is `0.801/0.729/0.771/0.375`; MUVY validation
  is `0.605/0.303/0.268/0.129`. Frozen BODD references are respectively
  `0.854/0.733/0.814/0.425` and `0.452/0.364/0.232/0.109`.
- The exact LAL-BOS enrollment causal protocol sampled 405 frames and emitted
  zero detections/tracks, proving failure of the 0.85 worst-game gate without
  running the other three games.
- Across 1,417 guarded samples there were zero breaches. Peaks were 85.7%
  system memory, 66.7% CPU and 1,034,665,984 bytes process-tree RSS.

## 2026-07-29 DINOv2 temporal-context verification

- RED first failed because the centered-wide offset protocol was absent;
  12 frame-protocol tests now pass. A second RED failed because temporal joins
  did not accept a wide artifact; four temporal-screen tests now cover exact
  duplicate-offset equality, correction coverage, blind/tamper rejection,
  nested held-game isolation and wide representations.
- The 110-row wide extraction reproduced the existing DINOv2 model hash and
  emitted artifact SHA-256
  `ceedfbc19c8ca7faf2a7b4001b91680bfd30f6ac2873baa6f7056552ae11a3fc`.
  The final screen artifact SHA-256 is
  `a05095d7371180f33d9019cebbc7232958130f3680781f2fc0ea054886dc4c79`;
  `accepted=false`.
- Extraction used 627,965,952 bytes maximum RSS and zero swaps. The nested
  screen used 573,456,384 bytes maximum RSS and zero swaps. No guard limit was
  breached.
- This increment changes only offline training/evaluation code; the local API
  smoke hook does not apply. HOU–ORL G1/G2 remain sealed.
- The canonical `.venv` full suite passes 702 tests with 15 existing warnings.
  Touched-path Ruff, dependency consistency, `git diff --check` and repository
  Harness verification also pass.

## 2026-07-29 event-centered cut-aware verification

- RED first established that event-plan transition selection and the new scene
  screen did not exist. Eighteen focused tests now cover plan binding, exact
  coverage, sealed-blind rejection, Codex runtime-answer rejection, semantic
  field independence, nested held-game isolation and artifact hash sealing.
- Four boundary artifacts contain 33/9/63/5 events and 79/19/161/7 cuts. Their
  SHA-256 values are
  `b937380bfb1a3ac374c8b7c9eebc41cb4cde64747ab65df872d40b754374f51a`,
  `259bdfc6e9a1429d99b8454c8a69db9585834bf5e79f5524fb83213025fa90a2`,
  `bdb33152b3aa3aadb8ee568d6f21b2ba574df56da1b00a9268c464a75d1de6b4`
  and
  `591b94382dba6d480a96ff0575b9192347613eb689633f84dcfed4ade1802f5b`.
- The rejected screen artifact SHA-256 is
  `e8bae471b101266eb77c1aafe0b949a14d2c471c9c5e88c811f88d32cc1cb697`.
  Boundary extraction peaked at 80.5% system memory, 64.4% CPU and about
  1.05 GiB process-tree RSS; screening peaked at 75.1%, 28.9% and about
  530 MiB. Every resource log records zero guard breaches.
- This is offline training/evaluation work only, so the local API smoke hook
  does not apply. HOU–ORL G1/G2 remain sealed.
- The canonical `.venv` full suite passes 708 tests with 15 existing warnings.
  Touched-path Ruff, `pip check`, `git diff --check` and repository Harness
  verification also pass. One resource-guard test initially sampled an
  unrelated system-CPU spike in addition to its forced memory condition; its
  isolated rerun and the complete clean rerun both pass.

## 2026-07-29 dense cut-aligned MViT verification

- RED failures established the missing clip planner, artifact contract,
  extractor, extraction CLI, nested screen, screen CLI, source catalog entries
  and empty-backbone provenance rejection. The combined focused set passes
  17 tests and touched-path Ruff passes.
- Four verified embedding artifacts contain 110 events and 256 available
  clips. Their sealed SHA-256 values are
  `a1865836999872571e644910e75b6b0aa368535b569fcfc396afcb646601b5f5`,
  `694e18d3e5c9e99e2a1a0bcb0b1f93169f1a64df3ece98b05084e4e45e6261f3`,
  `f9c4dbd9b9b190cc6e2d863bca63597c6cb5c1cb520de9af716afae0ae2a17f8`
  and
  `fb962a91629aac3605b83f83b6c433ab5aa2ecb2827e60f53eb1757a3c41a611`.
- The verified screen artifact SHA-256 is
  `69c692acec1871b98903f771966d6307b721e64521ee6a186648e1ce2ec05abd`;
  `accepted=false`.
- All guarded commands exit zero with no breach. Extraction maxima are
  736,428,032 bytes process-tree RSS, 85.6% system memory and 57.2% CPU;
  minimum available memory is 2,481,979,392 bytes and minimum free swap is
  1,579,614,208 bytes. Screening peaks at 491,470,848 bytes RSS, 76.7%
  memory and 27.1% CPU.
- This increment is offline training/evaluation and source governance only.
  It changes no API/runtime behavior, so the local service curl hook does not
  apply. HOU–ORL G1/G2 remain sealed.

## 2026-07-30 Transformers RF-DETR adapter verification

- RED first established the absent adapter, factory backend and causal-script
  model-directory/hash behavior. Eighteen focused adapter, factory, script and
  causal-window tests pass after implementation.
- The real guarded run completed 405/405 frames with artifact SHA-256
  `ac911d31ee923dc6bb09ae79611da355caec9ab192acb83f1fa3de8b1339222a`.
  All 31 resource samples are within limits; peaks are 84.3% system memory,
  35.0% CPU and 617,414,656 bytes process-tree RSS.
- The sealed review artifact SHA-256 is
  `fb15efb6943a4dc753aef5bba8ab40bdb51803d58d04d67c1ae442b16ab5c6af`;
  it explicitly records `runtime_consumable=false` and
  `codex_runtime_answer_used=false`.
- The adapter is opt-in and local-only. It is not promoted because reviewed
  precision remains below 0.85; runtime and HOU–ORL blind assets are
  unchanged.
- The canonical `.venv` local service hook passed on loopback port 8794:
  `/health` and `/ready` returned HTTP 200, task
  `1625618136d34ced8a6748580f9f829c` was accepted and completed at progress
  100 with `error=null`, and the service shut down cleanly.
- The canonical `.venv` full suite passes 756 tests with 15 existing warnings.
  Touched-path Ruff, `pip check`, `git diff --check` and Harness
  `--run-tests` also pass.

## 2026-07-30 ball-candidate verifier transfer verification

- RED established missing recall-floor selection and stratified population
  metrics; the two focused tests pass after implementation.
- Every E-BARD image and all source/review/perception artifacts are hash
  verified. GroupKFold uses `game_id`; LAL–BOS decisions cannot enter model or
  threshold fitting.
- The MPS run stopped safely with exit 75 after three free-swap breaches and
  emitted no result. CPU four-thread visual+confidence and visual-only runs
  completed with zero sustained breaches.
- Complete-run peaks were 84.2% system memory, 95.8% transient CPU and
  4,849,975,296 bytes process-tree RSS. No classifier checkpoint, runtime
  change or blind access occurred.
- The canonical `.venv` suite passes 758 tests with 15 existing warnings;
  touched-path Ruff, `pip check`, `git diff --check` and Harness
  `--run-tests` pass.

## 2026-07-30 same-detector review/verifier verification

- RED established deterministic confidence-band sampling and underfilled-band
  rejection; two focused review-contract tests pass.
- Plan, perception, raw-video pixels, candidate IDs/order and sealed decisions
  are hash-bound. The review scope explicitly forbids event/statistic answers.
- Source and target video hashes must differ. Source fitting uses nine causal
  window groups; target decisions join only after source model/C/threshold
  selection.
- ATL inference, exact-crop transfer and 2× context transfer all completed
  under the resource guard with zero sustained breaches. Peaks were 71.2%
  memory, 67.1% CPU and 808,157,184 bytes process-tree RSS.
- No checkpoint, runtime path or blind source changed.
- Four focused review/verifier tests and the canonical 760-test `.venv` suite
  pass with 15 existing warnings. Ruff, `pip check`, `git diff --check` and
  Harness `--run-tests` also pass.

## 2026-07-30 multi-source verifier verification

- RED first established missing selected-detection decoding, uncertain-row
  exclusion, game-scoped causal groups and visible-track feature binding.
  The focused verifier/review set passes seven tests after implementation.
- The second review plan contains exactly 108 ordered IDs. Its decisions CSV
  SHA-256 is
  `7dfb88bad2e50b4602b00200b44f24aded1ba721e463e50c65988cc37e6827c9`;
  the sealed review file SHA-256 is
  `56e1965eb790adb01808c6986fbca22247f19c33ebbbb0f0ac8c5748e73f77e3`,
  and its canonical artifact SHA-256 is
  `11e6e51141ac2861085cf9b9978572966f360f37f434aa6ffa8fa6592dc8be3f`.
- Both source games are mutually disjoint from each other and from LAL–BOS.
  Source labels and thresholds are fixed before the target review joins.
- Lower-bound external P/R is `0.547/0.770` for visual-only,
  `0.542/0.735` for visual+confidence, `0.382/0.965` for track geometry and
  `0.572/0.766` for visual+track geometry. The final artifact is
  `accepted=false`.
- Two guarded four-thread CPU runs collected 33 samples with no breach. Peak
  system memory is 75.0%, peak CPU 64.0%, peak process-tree RSS 811,188,224
  bytes and minimum available memory 4,295,639,040 bytes.
- The canonical `.venv` suite passes 763 tests with 15 existing warnings.
  Touched-path Ruff passes. No runtime behavior changed, so a service smoke
  hook is not required. HOU–ORL G1/G2 remain sealed.

## 2026-07-30 EBQwen candidate-panel verification

- RED established label-free balanced selection, fail-closed JSON parsing,
  weighted uncertain-label bounds and bounded MLX image-processor
  configuration. Four focused tests pass after implementation.
- The first prediction/evaluation artifact canonical hashes are
  `bc6855a96f13149cb14e0995d459dc8cd27790ce49d63206e4611250cd8d6398`
  and
  `d3881ce34acfde6e2b4fb8779ae03fd74b471f002ab44837bd3c604768b236b4`.
  The single revised-prompt hashes are
  `ced1985f441954d3ccccb5fb26ac01a2b126172fbf543f566293f615e7da0118`
  and
  `7fc6aecd75486a74a5773a9681e53d7f508f3d25dafcf92bcbe8f6bec2bdab80`.
- Both runs produce 12 uncertain states, zero kept candidates and zero
  weighted recall. The second evaluation is `accepted=false`; LAL–BOS is not
  opened by this branch.
- Across 55 resource samples, the guard stopped three sustained-pressure runs
  with exit 75 and resumable cache recovery completed both prompts. Peak
  memory is 89.6%, startup CPU 100%, process-tree RSS 1,404,600,320 bytes,
  minimum available memory 1,794,703,360 bytes and minimum free swap
  940,965,888 bytes. Safety thresholds were not relaxed.
- One full-suite attempt captured an unrelated system CPU spike in the
  intentionally forced memory-failure test, yielding one extra reason. Its
  isolated rerun passed, followed by a clean `767 passed, 15 warnings` full
  suite. Touched-path Ruff, `pip check`, diff and Harness `--run-tests` gates
  all pass after documentation updates.

## 2026-07-30 frozen RF-DETR query-verifier verification

- RED first failed because `app.analysis.rfdetr_query_verifier` did not
  exist. GREEN covers post-processor object-index reconstruction, fail-closed
  bbox mismatch and exact query-vector composition; three focused tests pass.
- Touched-path Ruff and `git diff --check` pass. The CLI help entry loads
  successfully from the canonical Python 3.11 `.venv`.
- The guarded MPS run uses two ATL–CHI source games and a disjoint LAL–BOS
  target. Its result canonical SHA-256 is
  `3624e90b7be00ee1546ce76c8f313b018c7cdf4ec7c722d2e1d1d27b81b657a5`;
  the file SHA-256 is
  `cff363bcc432353cb3db5304f00ee774064842ed8b5db7c616a83cb688733f55`.
- Best external lower-bound P/R is `0.659286/0.858863`; every variant has
  `accepted=false`, and the artifact records `checkpoint_saved=false`.
- Five resource samples have zero stops or breaches. Peak memory is 79.9%,
  peak CPU 33.7%, peak process-tree RSS 571,129,856 bytes, minimum available
  memory 3,450,961,920 bytes and minimum free swap 1,410,400,256 bytes.
- The canonical full suite passes `770 tests` with 15 existing warnings.
  Touched-path Ruff, `pip check`, diff check and Harness all pass.

## 2026-07-30 third-domain RF-DETR supervision verification

- RED first rejected `transformers_rfdetr` in the generic window runner.
  GREEN accepts the backend, resolves a single local safetensors artifact and
  preserves the injected-adapter test boundary.
- A second RED proved generic perception output lacked a canonical hash.
  GREEN seals the return payload, and downstream `verify_artifact` succeeds.
  Six focused window/adapter tests pass.
- The CHI–UTA perception canonical/file SHA-256 values are
  `41bfd2d89e165d8eed00c6ed2ef12844052f5dcca1cdb7138be7f541b5fe2200`
  and
  `8f24429487a196117b25444bc66c047393b49c29a203609ea819db66bedc4da1`.
  The 108-row review has 50 valid, 56 false and two uncertain decisions.
- The three-source screen canonical/file SHA-256 values are
  `ba3369e73ab2de53b895b67255d1fbe25d232aee68246d866479d385822920d8`
  and
  `9cdccf890adeb976901abe3f55ab33e51911feda530cb55d3f9b23b6160ec4f2`.
  Best lower-bound P/R is `0.660091/0.936992`; `accepted=false`.
- Detection and screening resource guards record 43 samples and zero stops.
  Peak system memory/CPU/RSS across them are 78.9%, 45.3% and 909,426,688
  bytes. The canonical full suite passes `771 tests` with 15 existing
  warnings; Ruff, `pip check`, diff check and Harness pass.

## 2026-07-30 temporal query and LAL–BOS Game 1 verification

- Five focused query-verifier tests cover strict same-track expansion,
  visible/non-predicted member filtering and deterministic 36-dimensional
  temporal aggregation. Together with the generic perception tests, the
  focused verification is `9 passed`.
- The temporal Game 2 screen canonical/file SHA-256 values are
  `696fafb1071d49cf29f1b11c3537dd08298cec8716932f6e129f64571581b0e8`
  and
  `f29731450096615f0001544db5614e92c1fb88294ef784f60929b8be8d15fd90`.
  Best temporal lower-bound P/R is `0.637263/0.902268`; `accepted=false`.
- LAL–BOS Game 1 perception contains exactly 5,016 samples, 10,566
  detections and 2,441 tracks. The 108-row review order matches the plan
  exactly and seals 20 valid, 86 false and two uncertain decisions.
- The frozen three-source Game 1 screen canonical/file SHA-256 values are
  `5825cfabe9e75817995b5c3420715ccecc688c19a66e6d112064dbf00aea849f`
  and
  `7008f959e9defed0d833861ffbe44df48975b7c07cbc93fbb467df1a00a463f3`.
  Best lower-bound P/R is `0.371835/0.629407`; all variants are rejected.
- The four-source reverse-development screen canonical/file SHA-256 values
  are
  `2eb8a5a571bfc082a4bfab1d43df9945ead6e648f256383c82a76ad869d2b081`
  and
  `0d9bdde75c56d702dda190d15be550af8b7508a10d62b409ecebb2cb1791e164`.
  Best lower-bound P/R is `0.664548/0.936992`; all variants are rejected.
- The Game 1 detector guard has 24 samples and zero stops, peaking at
  79.1% memory, 40.0% CPU and 769,638,400-byte RSS. The three query screens
  have 55 samples and zero stops, with combined peaks of 80.5% memory,
  51.7% CPU and 834,650,112-byte RSS.
- `.venv/bin/python -m pytest -q` passes `773 tests` with 15 existing
  warnings. Focused Ruff, `pip check`, `git diff --check` and repository
  Harness all pass.

## 2026-07-30 WASB adapter and entity-relation verification

- RED first: `tests/test_wasb_ball.py` initially failed because the adapter
  module did not exist. Six tests now cover the official affine round trip,
  three-frame channel/normalization contract, mixed-shape rejection,
  confidence-weighted component decoding, strict tensor-only loading and
  source/checkpoint hash rejection.
- A guarded real MPS smoke loaded source SHA
  `862ce90279a44ad19dd0590e073a8089c1cf4bd79eca0930189dda9106fa83f5`
  and checkpoint SHA
  `8d1ba9870d0a6ab37b06ab82bed593c6c09133e713810bac475d0c000bb7e948`;
  the output was finite with shape `[1,3,288,512]`.
- RED first: the entity relation tests initially failed on the missing module.
  Five tests now cover player-order and horizontal-mirror invariance, uniform
  pixel-scale stability, finite missing-entity defaults, canonical hash
  tamper rejection, exact joins and blind-source rejection.
- `17 passed` across the two new suites plus the existing dense-temporal
  suite; touched-path Ruff passes.
- Real artifact verification reloaded both canonical hashes:
  `32733ec21d35a00ad0d19c04bd1dd559696e94b817dd69fbf67b6ad50a9b0feb`
  and
  `876619506d009d2ee5ca235114d362c0d49179313f897e6858aa3c34d9998fc7`.
- Guard logs:
  `causal_phase_entity_relations_build_guard_v1.jsonl` sampled 3 times with
  no stop; `causal_phase_dense_entity_relations_screen_guard_v1.jsonl`
  sampled 5 times with no stop; the WASB smoke sampled twice with no stop.
- The canonical `.venv/bin/python -m pytest -q` regression passes
  `784 tests` with 15 existing warnings. Repository Harness verification
  passes with the 17-test focused command; touched-path Ruff, `pip check`
  and `git diff --check` also pass.

## 2026-07-30 BARD embedded-validation verification

- Nine importer tests cover safe CSV parsing, event-state classification,
  deterministic game-balanced selection, blind-game exclusion, URL binding,
  Git blob verification and duplicate-media rejection.
- Five transfer tests cover normalized frame sampling, fixed-dimensional
  temporal pooling, artifact tamper rejection, label-free target prediction
  and exact delayed-label joins.
- The 38-video subset has 38 distinct SHA-256 values and passes ffprobe as
  H.264, 1280x720, 60 FPS, with durations from 3.1167 to 13.05 seconds.
- Canonical SHA-256 values are
  `38205c205f1aba5c013b10023791bba746f5477b5cbe687fe558d09732da8f25`
  for the plan,
  `fb6fe27cb03174b2f63f90131ea4a4e6a428c6e93eb27229973d3e94e182c50c`
  for source embeddings,
  `bfc1c737495d305997b90aa2fa8da05ea7291653d48fe4d811640fa1bc1748d6`
  for target embeddings,
  `f813d4943afbacd33292d3f124ee36a835a8728f3a8ff3d53e53d297643da02b`
  for frozen predictions, and
  `94e50874dd88acf98db8a2beb6c6bfea38b372431cac823955ba0a4acbbb1c19`
  for rejected evaluation.
- Download, embedding and screen guards sampled 20/17/4 times with zero stop
  events. Peak memory is 74.0/81.8/74.4%, peak CPU is 39.5/60.4/20.3%, and
  peak process-tree RSS is 68,665,344/730,742,784/680,935,424 bytes.
- The BARD importer, transfer and public-catalog focus passes `22 tests`.
  The canonical Python 3.11 `.venv` full suite passes `798 tests` with 15
  existing warnings. Touched Ruff, `pip check`, `git diff --check` and the
  repository Harness all pass.

## 2026-07-30 exact-candidate base/VLM fusion verification

- RED first exposed the missing fusion module. Five tests now cover player
  selection, bounded frames, label-free sealing, clip/availability validation,
  exact plan coverage and unknown propagation.
- A first guarded real run failed safely because variable-size crops were
  stacked before preprocessing. Each crop is now preprocessed before batching;
  that failed run did not create a sealed prediction artifact.
- The final base artifact canonical/file SHA-256 values are
  `ae4231450f3686773536d0fd372ac1aa30fbe4fb13e180b1e638e02648d31a70`
  and
  `3abab80a8d1507bedbc3e297c129aecc87b60eb4ca76d017e9b06c6f91af99c4`.
- Fixed-rule evaluation canonical SHA-256 values are
  `0c6316a9945f592a71425e58c104b99d71b9ae0ff642c0bd2677bf94be6da6b8`
  (base),
  `e121a7e61811079c59f9aec851b63bcb348fac74ee81400159e078f2001c86a7`
  (VLM),
  `ffe00be483777eb284763a290ee0d65b035db5959841f15c9f581c9317a15755`
  (AND), and
  `6d563a5090df47bc60d3e0bd55db8b005c02a7d9bd88030a1b079c7ae99e20e7`
  (OR).
- The final resource log has 15 samples and zero stops. Peak memory/CPU/RSS
  are `82.8%/52.2%/1,121,959,936` bytes; minimum available memory/free swap
  are `2,949,693,440/1,133,707,264` bytes.
- The focused slice passes `30 tests`; the canonical Python 3.11 `.venv` full
  suite passes `803 tests` with 15 existing warnings. Touched Ruff,
  `pip check`, `git diff --check` and the repository Harness all pass.
- HOU–ORL G1/G2 hashes, frames, predictions and truth were not read.

## 2026-08-01 same-detector broadcast hard-negative expansion verification

- Sealed the 180-row review with exact plan order and candidate IDs: 55
  `valid_ball`, 98 `false_positive`, 27 `uncertain`; the review reloads with
  plan SHA `93a9062caf41b0ef3f6aa89540562c4eb00fdf17ea750c492686b605c02468d3`.
- Ran the game-disjoint verifier under `scripts/run_guarded_training.py` with
  CPU and memory limits. The artifact is
  `analysis_outputs/public_research/atl_chi_continuous_review_v1/same_detector_ball_verifier_screen_v1.json`;
  its SHA-256 is `f1a179b00e0712909471325aedcfa14b7fc5495e971d8f09055fdec3eab53c40`.
- Resource guard: 10 samples, zero stops, peak system memory/CPU/process-tree
  RSS `85.1%/62.1%/494,698,496` bytes, minimum available memory/free swap
  `2,556,510,208/1,077,870,592` bytes.
- Focused Python tests pass (`14 passed`); Ruff and `py_compile` pass. The
  screen is rejected and no checkpoint, runtime or blind-game input changed.

## 2026-08-01 Open Images V7 ball-box transfer verification

- Official source URLs and license terms were checked before download. The
  materializer produced 240 validation images and 358 ball-related boxes;
  manifest reload and every image SHA-256 were checked during extraction.
- RED→GREEN tests cover normalized-box clamping and overlap-safe background
  crop selection. The focused Open Images/catalog slice passes `10 tests`.
- The guarded CPU screen ran with four threads and zero resource stops. Its
  max samples were system memory `84.6%`, process-tree RSS `834,289,664`
  bytes, and CPU `61.7%`; minimum available memory was `2,646,802,432` bytes
  and free swap `1,119,092,736` bytes was above the configured 0.5 GiB floor.
- The source OOF AP is about `0.993`, while the best frozen external lower
  bound is P/R `0.559/0.356`; screen SHA-256 is
  `53a6d52248c63ff04c30220c0b8b7e2f3ca2320d0263051153fafb9801446639` and
  `accepted=false`. No checkpoint, runtime or blind asset changed.
- Touched Ruff, `py_compile`, `git diff --check` and the source-catalog tests
  pass. This screen is offline-only; Open Images remains a small pretraining
  candidate, not a production detector or answer channel.

## 2026-07-30 MUVS formation geometry and local-view verification

- RED first: the missing formation module failed import. Four geometry tests
  now cover offline/hash binding, grouped OOF acceptance, multi-artifact exact
  deduplication, conflicting duplicate rejection and invalid normalized boxes.
- The first real multi-artifact screen failed closed because duplicate images
  inferred in different YOLO batch compositions differed slightly. The
  extractor now requires singleton batches. Canonical reruns agree exactly and
  reload at detection SHA-256 values
  `887db98d3684366e6c2810d1a04ff51373db1dc995834d16c9ffaa7c3e579032`
  and
  `718eee74e6bcf06e7f38f3f3f5f78acd8a3e4084035a47c9fe087745c3da5d8a`.
- Geometry screen SHA-256
  `0712ff7a8d3d4a4aee8555a40d607a8fe5e4fc121e384c995e298924fed5b6af`
  reloads with `accepted=false`, balanced accuracy `0.712` and worst positive
  event recall `0`.
- Four local-view tests cover DINOv2 artifact tampering, fixed fold-local PCA,
  multi-artifact deduplication/conflicts and deterministic rim-first/player-
  density view selection. Base/professional embedding hashes are
  `1b8907789e625691e59ef7ec744b6a078ba772f122c5f495359a8e77ae777d67`
  and
  `8b40af78fca302ea05238a05244e132bec72df1393b6af10cfea2341af1e27b4`.
- Local-view screen SHA-256
  `7f2acae7781271c9bb382534c68c2a1b408c6fcffb2ccc6ddd9b14b93c5f2759`
  reloads with `accepted=false`, balanced accuracy `0.686` and worst positive
  event recall `0`.
- Canonical BODD singleton extraction logged eight resource samples, zero
  stops, peak memory/CPU/RSS
  `74.4%/31.8%/865,026,048` bytes, minimum available memory/free swap
  `4,405,788,672/1,097,400,320` bytes. DINOv2 logged eight samples, zero
  stops, peak `73.8%/17.8%/675,594,240` bytes and minimum
  `4,492,673,024/1,130,954,752` bytes.
- The focused MUVS/perception slice passes `26 tests`; the canonical `.venv`
  Python 3.11 full suite passes `825 tests` with 15 existing warnings. Touched
  Ruff, `pip check`, `git diff --check` and repository Harness all pass.
- No service/API/runtime/config behavior changed, so the local FastAPI curl
  hook does not apply. HOU–ORL G1/G2 remained sealed.

## 2026-07-30 MUVS source-state verification

- RED→GREEN tests cover balanced/selected event plans, source-hash drift,
  exact frame/review coverage, embedding tamper rejection, separately sealed
  source combination, duplicate-frame removal and leave-one-event-out OOF.
- The MUVS/public-catalog/resource/harness focused slice passes `25 tests`.
  Touched-path Ruff, `pip check`, `git diff --check` and repository Harness
  all pass.
- The canonical Python 3.11 `.venv` full suite passes `817 tests` with 15
  existing warnings.
- Final review canonical SHA-256 values are
  `65a0c426c374c1952173dc2f53039644473a5d65ad1ef8ca2a78549a1d9e76b4`
  (48-row base) and
  `545073fa21d24ddd389b8468846772743bca5d91a214c0b1e2a2d297fcf4c6dc`
  (24-row professional supplement).
- Embedding canonical SHA-256 values are
  `06a5b97e066f8c18bb108efae6a75ffcb3ee01b363e7e672c86f956df3e86438`
  and
  `b371c6aca1a71db0c0e1ac766f182dd14e75e25a93a85d2a3ef6fd803946447f`;
  rejected screen SHA-256 is
  `1c3c38d839a98d40ab8f68fd529c3c92b9471b10dad40ca694c5008edd345ef7`.
- Final extraction/screen guards record zero stops, peak 74.6% memory and
  565,903,360-byte process-tree RSS. HOU–ORL G1/G2 remain sealed.

## 2026-07-30 sparse-gated temporal verification

- RED first: the classifier shape test rejected `temporal_operator="gated"`.
  It now verifies main/phase output shapes and that all 24 learned attention
  weights are normalized per example.
- The existing strict artifact, game-leakage and nested-screen tests also pass
  with the expanded operator validator.
- The real screen reloads at canonical SHA-256
  `a6c7b41a1a09ca047217b07573999c1da4bc1dde977e6cc9cf0b072c0717f10b`;
  `accepted=false`.
- Its guard logged 14 samples and zero stops. Peak memory/CPU/RSS are
  `74.5%/33.3%/735,363,072` bytes; minimum available memory/free swap are
  `4,381,818,880/1,142,095,872` bytes.
- The regenerated 31-source catalog canonical SHA-256 is
  `d5c4ad0701e35f403da7d1cf1ff689e9a82d94dd537963a49477cc1a93c82fe3`.
- The focused suites pass `14 tests`; the canonical Python 3.11 `.venv` full
  suite passes `803 tests` with 15 existing warnings. Touched Ruff,
  `pip check`, `git diff --check` and Harness all pass.

## 2026-07-30 independent formation VLM verification

- RED first covered the missing formation module and later the missing fixed
  veto evaluator. Seven tests now cover two-frame label hiding and tamper
  detection, exact offset derivation, fail-closed parsing, cache provenance,
  exact delayed-label coverage, fixed-veto behavior and rejection of
  unexpected resident Ollama models.
- The 4B probe guard recorded one resource stop, peaking at 89.4% memory with
  no cache or prediction artifact. The completed 2B artifact has 110/110
  available rows and canonical SHA-256
  `69701908d186876334f2085388c7d8448c3400ef15137f37cf94f78426aca9af`.
- The final successful clean-residency segment has 49 resource samples, zero
  stops, peak memory/CPU/RSS of
  `83.4%/41.9%/393,920,512` bytes, and minimum available memory/free swap of
  `2,850,258,944/1,450,246,144` bytes. Earlier stopped segments are retained
  in the append-only log, including the diagnosed unexpected 4B residency.
- The delayed evaluation and fixed-veto artifacts reload at canonical
  SHA-256 values
  `caa6b7b178533fd44ad3aed5dc6c95a9e2783d8dcb7a15389f75cdc20d454dc9`
  and
  `759679df80dd80edf3f62a6960a2048f2f77ef720e12b7ef28c223576468f166`;
  both are rejected.
- The focused VLM/resource slice passes `37 tests`; the canonical Python 3.11
  `.venv` full suite passes `810 tests` with 15 existing warnings. Touched
  Ruff, `pip check`, `git diff --check` and repository Harness all pass.
- HOU–ORL G1/G2 hashes, frames, predictions and truth were not read.
