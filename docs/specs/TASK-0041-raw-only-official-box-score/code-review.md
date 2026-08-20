# TASK-0041 Code Review

## Interim W5 review (2026-07-16)

- Existing `action_proxy_v1` and v3 preprocessing remain unchanged.
- Official aggregation accepts only confirmed event states and rejects broken
  assist/block/steal/rebound relations.
- The sealed prediction manifest contains raw-video content identity but not
  reference paths; edited/reference filename markers and non-video inputs fail.
- External detector loading is lazy and opt-in. Absence/failure cannot create an
  official event.
- Residual risk: E-BARD/COCO ball recall is insufficient; official contracts
  are not yet wired into the public analysis result; strict actor recognition
  remains unproven; Game B is partial-quarter truth only.

## Dense global-path review (2026-07-24)

Verdict: approve as an opt-in experimental selector; do not enable by default.

- Correctness: behavior tests cover smooth-path preference, explicit scene
  reset and invalid thresholds. The frozen offline objective maps directly to
  the selector's Top-K/log-confidence/soft-speed dynamic program.
- Required finding resolved: the first implementation copied complete path
  lists for every DP state, causing avoidable quadratic path-memory growth.
  It now stores per-frame backpointers with bounded Top-K state.
- Architecture: the selector lives beside the existing ball linker, returns the
  existing `BallTrackResponse`, adds no dependency/schema/configuration, and has
  no default call site. The legacy tracker and v3 preprocessing remain intact.
- Security: thresholds are validated, inputs are existing typed perception
  detections, and no path, network, secret or external process enters the new
  code.
- Performance: time is `O(frames * K^2)` and state is bounded by frame
  backpointers with default `K=10`; candidate count cannot grow unbounded per
  frame.
- Residual risk: development and validation events share one broadcast, scene
  reset frames must be supplied by the caller, and ball localization success
  does not imply shot or complete-statistics correctness.

## Cross-game ball-path review (2026-07-24)

Verdict: approve continuation into default-off shot/outcome research; do not
enable production runtime behavior.

- The unchanged ATL-developed objective passes a game-disjoint LAL-BOS
  supplemental gate at 12/13 (.923) after inference and selections were sealed
  before ball-only annotation.
- The first cross-game plan's sample-floor failure is preserved, preventing
  post-hoc frame replacement. The supplemental plan has a separate selection
  policy and artifact chain.
- Codex labels only visibility/location and every artifact remains
  `runtime_consumable=false`; no shot, outcome, player or statistic labels were
  introduced.
- Residual risk remains substantial: only six cross-game events have been
  reviewed, the v2 frames are detector-enriched and correlated within events,
  scene-reset discovery is still caller-supplied, and no downstream event or
  complete-game box-score metric has passed.
- The downstream bridge is an optional keyword argument and preserves the
  original candidate path when omitted. It retains source detection IDs rather
  than reconstructing detections from track centers, so rim, trajectory and
  actor evidence remain auditable. No production caller opts in.
- The first downstream diagnostic rejects promotion: the rim-plane heuristic
  mistakes a foul continuation for a made FGA, admits a free-throw sequence as
  an FGA, and leaves the true made three unresolved. The next design needs
  independently observed whistle/free-throw/game-state evidence followed by a
  new blind, game-disjoint gate.

## Component-score and score-first VLM review (2026-07-24)

Verdict: retain as an offline, fail-closed research path; do not promote.

- Component consensus is conservative by construction: it independently
  reconciles each team and requires exactly one unique +1/+2/+3 change.
  Negative transitions, two plausible scorers and large jumps expose
  unresolved team IDs instead of manufacturing an event.
- The targeted runner decodes bounded candidate windows and SHA-binds the raw
  evidence bundle. It deliberately neutralizes candidate outcome, shot value
  and team before score or VLM fusion, preventing a coarse trajectory guess
  from overriding independently observed evidence.
- The VLM prompt uses only observable raw-frame cues. It does not receive
  play-by-play, statistics or Codex labels. The v18 cache version ensures old
  prompt results cannot be silently reused.
- The preserved blind artifact emitted one correct free throw and abstained
  twice. The post-truth two-of-two fusion diagnostic is useful evidence for
  ordering, but is ineligible for promotion.
- Residual risk is decisive: the local VLM still accepts a replay/dead-ball
  duplicate, the emitted free throw has no player attribution, and no rebound
  or secondary-event relation was validated. Public APIs, defaults, schemas,
  environment variables and the v3 preprocessing contract are unchanged.

## Replay-transition/logo proxy review (2026-07-24)

Verdict: reject runtime integration; retain only reproducible offline evidence.

- TransNetV2 is appropriate for scene boundaries, not replay semantics. The
  wrapper verifies the raw video, source revision, module and checkpoint hashes
  and uses safe weights-only loading.
- Repeated no-person graphics are auditable broadcast-transition evidence, but
  they occur after genuine live makes as well as before replays. The four-game
  result (precision 0.571, recall 0.160) invalidates the initially encouraging
  MEM single-game observation.
- The evaluator is fail-closed: it accepts only SHA-bound offline labels, checks
  that the negative subset explicitly says replay/highlight, namespaces event
  IDs by video and emits a promotion decision against the fixed 0.95 floor.
- No public schema, configuration flag, service path or VLM prompt consumes
  this proxy. Codex labels remain evaluation-only, and v3 preprocessing is
  untouched.
- Pinning OpenCV below 5 is required compatibility maintenance, not a model
  change; OpenCV 5 lacks APIs exercised by existing identity/tracker tests.
- The new blind plan passes metadata, playback and 100% enrollment-coverage
  gates but not the media gate. Transport retries are now finite within each
  resumable outer attempt, and curl/IPv4 are explicit options rather than
  implicit environment behavior. Current CDN TLS failure is recorded; it is
  not converted into a completed or SHA-sealed download.

## Broadcast-clock replay proxy review (2026-07-24)

Verdict: retain as a default-disconnected offline veto candidate; do not
promote.

- The parser preserves clock punctuation, requires a period marker on the same
  OCR row and validates basketball clock ranges. The runner binds video and
  candidate hashes and never reads labels or reference statistics.
- The rule is asymmetric by design. A frozen prefix followed by three missing
  reads can flag replay; the reverse pattern is rejected because development
  examples show it occurs after genuine live shots.
- Four-game precision is 2/2, but recall is only 2/25 and neither ATL-CHI game
  produces support. A minimum five-prediction floor prevents a two-example
  perfect result from authorizing promotion.
- The independent MEM check adds no false positive but misses its replay.
  This confirms the intended fail-closed behavior and the decisive coverage
  limitation.
- All artifacts explicitly forbid runtime consumption and Codex runtime
  answers. No public service path, environment variable, VLM prompt or v3
  preprocessing contract changed.

## EBQwen independent raw-frame review (2026-07-24)

Verdict: reject the GGUF/Ollama multi-image path from runtime integration.

- The plan builder is deterministic and keeps labels/review fields out of the
  inference artifact. Prediction coverage must exactly equal plan coverage;
  missing, duplicated or extra windows fail closed.
- Canonical hashes protect the plan, predictions and evaluation. The resumable
  cache fingerprint also binds the prompt, model revision and weight hash,
  frame count, image width and context length.
- The runner receives only raw frames. It never reads annotation files,
  play-by-play, scoreboard answers or Codex labels. The evaluator is the first
  component that opens the separately supplied annotations.
- Four-game precision of 0.5714 decisively fails the 0.95 floor despite perfect
  recall. No threshold tuning or prompt iteration was performed after truth
  reveal, and no runtime wiring was added.
- The result is scoped correctly: multi-image GGUF inference is not equivalent
  to EBQwen's official native-video input. A future native-video experiment
  needs a new frozen plan and game-disjoint evaluation, not reuse of this
  revealed development set.

## EBQwen native-video review (2026-07-25)

Verdict: keep the MLX native-video path offline and reject promotion.

- The implementation now exercises Qwen2.5-VL's native video processor and
  temporal-position path rather than multi-image emulation. Model provenance,
  source-video hashes, sampling FPS and pixel budget are sealed.
- Inference cannot read annotations. The evaluator separately requires the
  exact source plan referenced by the derived native plan before accepting its
  annotation hashes.
- Resource limits are enforced externally and the model is loaded once. Model
  parse/generation failures fail closed to `unknown` with bounded diagnostics.
- Review tightened three boundaries before handoff: the CLI verifies the sealed
  native contract before allocating model memory; any unavailable reviewer
  result is forcibly normalized to `unknown`; and the MLX extra is platform
  marked for Apple Silicon so Linux all-extras environments do not attempt an
  unsupported install.
- The four-game development screen has perfect recall only because every
  positive and every negative is predicted positive. Precision `0.5000`
  decisively fails the `0.9500` gate, so no fusion or service wiring is
  justified.

## Ball-release annotation review (2026-07-27)

Verdict: retain the training-only correction tooling, reject promotion, and
move the next increment to atomic candidate construction.

- Plans hide labels and OOF probabilities; every derived artifact is
  hash-bound and explicitly non-runtime.
- Corrections are fail-closed: free throws and replay/stoppage rows become
  negative, clear live release toward the rim becomes positive, confident
  no-release becomes negative, and all other cases remain unresolved.
- Original embedding artifacts are never overwritten. Each correction pass
  emits separately sealed scene, Swin and MViT copies.
- Three disjoint review rounds improve pooled F1 but leave two games below
  0.80. This rules out blind promotion and shows diminishing returns from
  further label-only iteration.
- The principal residual risk is semantic mixing inside long scoreboard
  windows, not an unbounded need for a larger classifier.

## First blind reveal and subtype-preservation review (2026-07-26)

Verdict: retain the conservative free-throw reclassification, but do not
promote the overall pipeline.

- Simple temporal de-duplication was rejected: the existing 3-second clustering
  already leaves no adjacent field-goal candidates within that interval, and
  widening it would erase legitimate putbacks.
- Reclassifying from one structured boolean alone was also rejected after an
  initial development replay exposed ten conversions, including one dead-ball
  contradiction and four explanations with no free-throw evidence.
- The final rule requires structured, textual and scene consistency. It
  preserves five reviewable FTA candidates, rejects contradictions, and never
  upgrades incomplete actors or outcomes merely because the category changed.
- Fixed complete-scope accounting is the comparable metric: coverage improves
  from 2/8 to 3/8 and the optimistic candidate upper bound rises by about 1.30
  points to 0.5469. The older prediction-scope number falls because adding FTA
  correctly adds 39 missed official attempts to that denominator; it is not a
  regression in fixed-scope coverage.
- Residual risk remains decisive: only 5/44 official FTA are proposed, 195/197
  original FGA outcomes were unknown, 192 rebounds were blocked by unresolved
  misses, and five event types remain absent. CHI-UTA is revealed development
  data and cannot count toward the final two-game blind gate.

## Continuous reason-evidence review (2026-07-27)

Verdict: retain the offline evidence and leakage tests; reject runtime
promotion.

- The feature artifact is label-free, hash-bound and explicitly non-runtime.
  Its schema was advanced to v2 because the vector changed from 28 to 39
  dimensions; old artifacts cannot be silently interpreted as the new
  contract.
- Nested provenance prevents the subtle leakage mode where an outer-training
  row is scored by a reason model trained on that same game's review labels.
- Missing detections fail to finite neutral defaults, while malformed artifact
  dimensions, labels, hashes or source coverage fail closed.
- Formation and temporal features are explainable but do not establish player
  identity or event outcome. Their small replay benefit cannot justify
  promotion after the main held-game metric failed to improve.
- HOU-ORL remains untouched. Reusing these development labels for further
  unplanned threshold/model search would overfit the four revealed games and
  is explicitly rejected.

## Dense cut-aligned MViT review (2026-07-29)

Verdict: retain the training-only extraction/evaluation contract and compact
embeddings; reject model or runtime promotion.

- Planning uses only exact video/candidate/event identities and TransNet cut
  geometry. It ignores transition semantics and rejects sealed blind hashes
  and artifacts declaring Codex runtime answers.
- Sampling is bounded inside the previous, anchor and next shot. Missing
  neighboring shots remain unavailable instead of duplicating anchor frames.
  The full-shot uniform protocol does not claim to reproduce torchvision's
  fixed-frame-rate Kinetics recipe; that distinction is documented.
- Every embedding artifact binds the review plan, transition artifact, source
  video, candidate bundle, backbone name/hash, dimension, frame count and exact
  event coverage. The screen rejects incomplete or mismatched artifacts.
- Review found one public-entry weakness: direct callers could pass an empty
  backbone string or non-positive dimension/frame provenance. A RED regression
  now covers the empty backbone case, and both construction and artifact
  verification fail closed on all three fields.
- Outer held games never enter model selection. Representation, PCA dimension
  and logistic regularization are chosen only through leave-one-remaining-game
  inner folds.
- The result is well below the 0.85 gate and DINO fusion is weaker. Further
  tuning of the same 108 labels is rejected as development-set overfitting.
- MultiSports/FineSports terms and SHOT7M2 task mismatch are fail-closed in the
  catalog. No external agreement was signed and no dataset asset was
  downloaded.

## Frozen RF-DETR temporal-query review (2026-07-30)

Verdict: retain the exact query/track bindings and sealed box annotations as
offline research evidence; reject every classifier and runtime promotion.

- Track expansion admits only visible, non-predicted detections sharing the
  exact track ID of a reviewed source row. Missing tracks stay explicit rather
  than borrowing nearest detections.
- Target decisions are loaded only after source fits, thresholds and target
  scores freeze. The Game 1 screen is therefore game-disjoint, though not a
  new-domain result because both holdouts are from the same LAL–BOS series.
- The reverse Game 1→Game 2 run is correctly labeled a development diagnostic:
  Game 2 variants had already been inspected and cannot become a fresh gate.
- Sparse candidate reviews are not exhaustive COCO annotations. They must not
  be used to fine-tune the full detector by treating every unreviewed query as
  background.
- The large precision collapse on Game 1 and unchanged Game 2 ceiling after
  adding hard negatives are sufficient negative evidence to stop tuning this
  frozen linear representation.
- No HOU–ORL G1/G2 frame, truth or prediction was opened; blind inference
  remains gated.

## WASB adapter and entity-relation review (2026-07-30)

Verdict: retain the reproducible offline adapter and sealed relation artifact;
reject the classifier and stop further hand-crafted relation tuning.

- The adapter verifies both source and checkpoint hashes, uses
  `torch.load(weights_only=True)`, loads the state dict strictly and avoids the
  upstream CUDA-only wrapper. Its dependency on an explicitly supplied pinned
  source path keeps third-party code outside the service boundary.
- RGB/BGR ownership is explicit: the adapter accepts RGB uint8 triplets; video
  readers must convert OpenCV BGR before calling it. Existing v3 preprocessing
  and default perception factories are untouched.
- Relation features are permutation and horizontal-mirror invariant and use a
  player-height scale. Empty or partially missing entity sets produce finite
  neutral values rather than fabricated tracks.
- Artifacts bind all four source videos, perception files, the causal review
  plan and every event identity while mechanically excluding blind hashes.
- The nested held-game result and fixed fusion diagnostic both fail the worst
  game gate. No checkpoint exists and no runtime flag can activate this
  research path.
- BasketEvent's missing license/source media is a hard governance boundary,
  not an invitation to download only its checkpoint. The paper may inform an
  independent architecture, but larger compatible supervision is required.

## BARD embedded event-state transfer review (2026-07-30)

Verdict: retain the bounded source subset and reproducible import contracts;
reject model/runtime promotion and close direct linear transfer.

- CSV labels are parsed with `ast.literal_eval`, never `eval`, and selected
  media must match exact Git blob SHA-1, size and unique SHA-256.
- The duplicate-content check correctly prevented 48 different NBA URLs from
  being mistaken for 48 training examples. The invalid 1.41 GiB directory was
  removed while its compact failure manifest was retained.
- Source configuration and threshold selection use only BARD labels with
  whole-game grouping. Target embeddings contain no labels, and predictions
  freeze before the delayed reference join.
- The all-free-throw target collapse and label-free feature-shift diagnostic
  establish a framing/domain mismatch. Source OOF `0.974` is not evidence of
  AGU-domain accuracy.
- Now that target labels are open, any new representation or threshold chosen
  in response would be development-informed and cannot become promotional
  evidence. A future attempt requires a newly specified, game-disjoint
  protocol and broader causal-window-aligned training data.
- BARD's repository license does not by itself prove redistribution rights for
  NBA broadcast footage. Keep the selected clips local and training-only.
- HOU-ORL G1/G2 remain mechanically excluded and were not inspected.

## Exact-candidate AGU v3 plus EBQwen review (2026-07-30)

Verdict: retain the reproducible offline diagnostic; reject all fusion rules
and do not promote either source.

- Extraction verifies exact plan, bundle, video and checkpoint hashes and uses
  tensor-only strict checkpoint loading.
- Artifacts forbid labels, target fields and runtime consumption. Availability,
  clip limits, action labels, confidences and unique player IDs fail closed.
- Four fusion rules are explicit and fixed; exact candidate coverage prevents
  comparison across different event universes.
- V3 has no positive confirmation signal. A non-shoot veto destroys recall;
  treating EBQwen as sufficient preserves its twelve false positives.
- The 32 rows are opened development diagnostics and cannot support another
  promotional tuning cycle. New whole-game-disjoint supervision is required.
- The deleted SpaceJam archive is identified by a retained audit manifest and
  is redownloadable; no unique user data was removed.
- Blind HOU–ORL assets remain sealed.

## Sparse-gated temporal review (2026-07-30)

Verdict: implementation is a valid offline research operator; empirical result
is rejected and closes further pooling-only tuning on the current labels.

- The gated operator is independently implemented from the published concept;
  no unlicensed BasketEvent code, annotation, checkpoint or trajectory was
  copied or downloaded.
- Attention weights are normalized across the exact 24 causal positions.
  Main and phase losses retain the existing training-only standardization,
  class balancing and auxiliary-label boundaries.
- Architecture and hyperparameters are selected only inside the three
  development games for each outer fold. Held games never enter training or
  selection.
- Adding operator choice increases selection variance and does not meet either
  pooled or weakest-game gates. No checkpoint exists to activate accidentally.
- New catalog entries fail closed on missing licenses and wrong event targets.
- Blind-game hashes and assets remain excluded.

## Independent formation VLM review (2026-07-30)

Verdict: retain the reproducible offline contracts and Ollama residency guard;
reject both local-model routes and do not promote the formation veto.

- The plan exposes only two exact raw frames and provenance. Phase labels,
  review notes, shot outcomes and blind-game sources cannot enter predictions.
- 4B is not locally feasible under the established resource thresholds even
  for one two-frame request. Lowering the guard would violate the user resource
  constraint and is not an acceptable model-quality trade.
- 2B is resource-feasible with 384px/context 2048 only when unexpected models
  are absent, but its 1.7% free-throw recall is too weak for evidence fusion.
- The fixed veto evaluator demonstrates no hidden benefit: both emitted
  free-throw signals occur where the upstream model is already negative, so
  the final confusion matrix is unchanged.
- The `/api/ps` preflight fails closed on extra resident models. It does not
  stop unrelated models automatically, preserving process ownership while
  preventing accidental co-residency.
- All 110 phase labels are now development-open. Further prompt, frame or
  threshold adjustment would be target-informed and cannot serve as an
  independent promotion gate.
- No checkpoint, runtime flag, API response, service path or blind-game asset
  changed.
