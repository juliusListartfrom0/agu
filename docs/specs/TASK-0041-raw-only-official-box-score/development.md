# TASK-0041 Development Log

## 2026-07-19 gallery-required official identity and image-bound VLM audit

- Bound official VLM cache entries to the rendered JPEG hash, context length,
  seed, prompt version and event payload. Overlay changes can no longer reuse a
  stale actor answer.
- Added deterministic AGU jersey-number reads with a 16k context, image-bound
  cache and two-independent-crop consensus. Identity graph edges now prioritize
  face-gallery and trusted-jersey evidence over body ReID and enforce globally
  unique canonical IDs.
- Added wrist-to-ball pose distance to action-owner evidence and fixed rebound
  actor validation to use the same raw-derived causal context shown to the VLM.
- Added an AGU VLM player/referee/non-player role gate. On the development
  Game A 0--60 second diagnostic, qwen3-vl:8b still labeled the foreground
  official as a player, so this is retained only as a secondary fail-closed
  guard rather than claimed as a solved classifier.
- Added `--require-face-gallery-identity` to official candidate generation.
  Only canonical IDs independently anchored by the sealed enrollment face list
  can enter actor candidates; jersey/body evidence may propagate that enrolled
  identity, while anonymous tracks are excluded. The current Game A identity
  artifact has 68 identities and zero gallery anchors, so it now fails the
  official-person readiness gate instead of emitting false per-player stats.
- Game A remains development data. Named highlights, misses and box-score files
  were not admitted as its runtime face gallery. The two-complete-game six-stat
  F1 >= 0.85 acceptance gate remains open.

## 2026-07-19 annotated face gallery identity dedup

- Added the sealed `agu.face-gallery.v1` runtime asset and strict loader/matcher.
- Added `scripts/build_face_gallery.py` for Codex-assisted face boxes on an
  explicitly benchmark-disjoint `agu.annotated-face-list.v1` source manifest.
- Wired YuNet + SFace track-face embeddings into the official identity graph.
  Same-gallery tracklets can merge despite body-ReID drift, different gallery
  identities cannot merge, and ambiguous or low-quality faces remain unknown.
- Added configurable gallery path, cosine threshold, and nearest-neighbor margin.
- No acceptance labels, highlight clips, missed-shot clips, or box-score answers
  are read by runtime inference. The Game A reference package is ineligible as
  the Game A gallery source.

## 2026-07-16

- Inventoried local validation assets and separated complete, partial and scoreboard-only datasets.
- Froze the raw-only inference/evaluation boundary and strict per-game acceptance metric.
- Started P0 contract, ledger and evaluator implementation.
- Added normalized detection/ball-track/court/possession schemas, an optional
  Ultralytics adapter and factory, constant-velocity short-gap ball tracking,
  possession/shot state machines and homography-based 2/3 classification.
- Added causal rebound, block, turnover/steal, assist and foul event builders;
  conservative raw-video identity stitching; append-only review decisions;
  official aggregation/reconciliation; and raw-only Codex review packages.
- Evaluated the existing first-quarter full-VLM result against 17 strict truth
  events: 648 legacy predictions, TP 0, FP 648, FN 17, F1 0.0.
- COCO YOLO detected two ball boxes in a 50-frame labeled-play sample. The
  CC-BY-4.0 E-BARD YOLOv8n checkpoint detected nine ball and ten rim boxes in
  the same frames, but coverage remains below the phase gate and the model card
  documents basketball recall 0.566 plus NBA-view bias.
- Added opt-in `BASKETBALL_OFFICIAL_*` configuration. No private checkpoint path
  or downloaded weight is committed.
- Added a continuous dense raw-video review package with timestamped contact
  sheets and zero-gap windows. A 120--300 second first-quarter package contains
  18 non-overlapping windows; all 18 sheets were inspected from the raw video.
- Added a strict dense-review importer and CLI. Every manifest window must have
  exactly one append-only decision; manifest/raw-video hashes, evidence times,
  event relationships and confirmed-event fields are validated before events
  enter the official ledger. Partial-video coverage is forcibly reported as
  `needs_review` with an `incomplete_video_coverage` reconciliation error.
- The real first-quarter empty decision file was rejected with all 18 windows
  listed as unreviewed. Coarse 2 FPS review visually localizes nine event
  sequences in 120--300 seconds, but cannot prove every outcome, touch or
  duplicate-number player identity. Those sequences remain unconfirmed pending
  fine-frame review rather than being promoted to improve the metric.
- Completed one raw-only coarse Codex pass over all 18 windows. It emitted 17
  `needs_review` events and a sealed bundle with hash
  `f99ba53fca517437ca520eecb792d9db3652afd05a9c1b0bedb88f438ceeca41`.
  The box score correctly contains zero accepted and 17 unresolved events and
  remains `needs_review` because coverage is only 120--300 seconds.
- Added adjacent-window evidence support for events that cross review-window
  boundaries. Every supporting window must exist, be reviewed and form a
  continuous time range; all supporting contact sheets are retained in event
  evidence.
- Replaced fixed 2 FPS/eight-frame candidate sheets with bounded full-window
  timestamped sheets (default 6 FPS, up to 72 frames). This avoids silently
  omitting the end of longer shot/rebound sequences.
- Added three explicitly separated evaluation tiers: type/time candidate
  localization, automatically confirmed strict events, and Codex/human-reviewed
  strict events. On the 17-event development quarter, the coarse pass localized
  17/17 with candidate F1 1.0, while both strict tiers remain F1 0.0 because no
  player/team/outcome/value labels have yet been accepted. The candidate result
  is not a blind 85% claim.
- Added hash-bound fine-review finalization. Every ReviewDecision must name the
  source sealed-bundle hash; accepted decisions create immutable revision 2
  events, the complete revision history is exported separately, and latest
  state is resealed with `reviewed_from_bundle_sha256` provenance.
- Fine review can now append a hash-bound revision-1 event when the raw evidence
  exposes an action missing from the coarse candidates. Existing IDs cannot be
  overwritten, and the full event payload is schema-validated before it enters
  the immutable ledger. This preserves miss/rebound/putback sequences instead
  of forcing a reviewer to discard one of the three statistics.
- Added evaluation-only global one-to-one identity alignment for raw-visible
  anonymous IDs. Alignment files are bound to both prediction-bundle and truth
  hashes, cannot alter time/type/outcome/value, reject many-to-one mappings and
  report unmapped identities. This permits fair per-player evaluation when the
  inference input contains no roster IDs without hiding the unaligned metric.
- Completed fine raw review of 16/17 partial-quarter events. The reviewed bundle
  `b7535bb6948786c70a0e26ad68e60a81952f15809e68fbd75197380a0c6400b2`
  has 16 accepted events, one unresolved steal, 33 immutable revisions and
  eight raw-visible player identities. Event points reconcile to 3--0.
- The separately bound evaluation produced strict identity-aligned TP 16 / FP 0
  / FN 1, precision 1.0, recall 0.9412 and F1 0.9697. All nine shots and seven
  rebounds matched; the steal remained unresolved because the required linked
  turnover was not yet present. This exceeds 0.85 for the partial development
  quarter only, not for a complete second game.
- Added six-video dense game review composition. Each period retains its own
  manifest/raw hash and complete-coverage status while event/evidence source IDs
  are deterministically remapped into one sealed game bundle. Any unreviewed or
  partial period fails closed.
- Generated raw-only Game A packages at 20-second windows and 1.5 FPS for all
  six original videos: 65 + 49 + 16 + 61 + 61 + 84 = 336 continuous windows,
  covering about 6,650 seconds. No reference CSV, highlights or edited clips
  were read during package generation. Combination correctly rejected the
  still-empty decisions at the first unreviewed window.
- Added a raw-perception ball/rim proximity candidate generator and a bounded
  adapter for AGU's existing raw-derived action/event candidates. Both sources
  remain `needs_review`; temporal fusion preserves the higher-priority
  ball/rim timestamp, retains corroborating evidence and remaps causal event
  relations when a duplicate is suppressed.
- On the partial Game B development interval, ball/rim candidates alone reached
  candidate-localization TP 9 / FP 29 / FN 8 (F1 0.327 at the best tested
  distance setting). A greedy one-to-one evaluator initially under-counted the
  fused result; deterministic maximum-cardinality matching corrected it to TP
  14 / FP 49 / FN 3. After preserving the control-acquisition frame for paired
  steal/turnover candidates, the sealed raw candidate bundle reached TP 15 / FP
  48 / FN 2, recall 0.8824 and F1 0.375. This is candidate coverage only and is
  not promoted to official statistics.
- Dense review now deterministically creates a provisional opponent turnover
  whenever an unresolved steal lacks its mandatory companion. Both remain
  `needs_review`; a standalone confirmed steal is rejected. Raw fine review of
  the 277--280 second sequence confirmed JOKIC 15's steal and the dark-team
  black-shirted headband player's turnover.

## 2026-07-18 open-vocabulary and flight-gate increment

- Tested YOLO-World v2 through AGU's Ultralytics adapter on disjoint July 4 raw
  footage. A `basketball hoop` prompt raised 120-second rim observations from
  11--44 with the specialist checkpoint to 625--711 before geometry filtering.
  A top-half camera prior and minimum 1.3 width/height ratio reduced ball-shaped
  hoop false positives while preserving true hoop boxes.
- Added repeatable class prompts and sealed camera-geometry parameters to the
  offline perception scan. `basketball rim` is normalized to AGU's stable rim
  schema. The model and prompt remain replaceable inputs; no private path or
  event answer is hard-coded.
- Added an optional backward ball-approach chain requiring configurable point
  count, rise, lookback and maximum inter-frame speed. The gate uses only raw
  detections and stores its evidence in the candidate bundle.
- Fusing open-vocabulary ball/rim observations with specialist ball detection,
  COCO player tracking, pose and ReID yielded seven shot candidates in the
  150151 120--240-second window. Codex accepted only green 35's drive/release
  and black 13's jump/release. A duplicated #16 action, merged identities,
  delayed rim contacts and incomplete release tracks were rejected.
- Sealed training manifest v6 binds nine annotation files from seven
  benchmark-disjoint videos and contains 11 action-owner examples. The linear
  pairwise model reaches leave-one-video-out pairwise F1 0.8384 and Top-1
  0.3636. Experimental nonlinear pairwise tournament sweeps peak at Top-1
  0.7273, below the 0.85 integration gate, so no autonomous runtime model was
  changed.
- The causal reviewed bundle
  `06a13c74945cf1b1fab0a1badadc463da6ca73a3e23393153ab8dee36b95478b`
  has 18 accepted events, 36 immutable revisions, zero unresolved events and a
  valid 3--0 reconciliation. Under the explicitly declared partial-category
  scope (shots, rebounds and steals), identity-aligned strict evaluation is TP
  17 / FP 0 / FN 0 / F1 1.0. It remains a development interval, not the final
  complete-game or blind two-game gate.
- Corrected dense coverage accounting so a manifest that spans the full video
  is still incomplete until every declared window has a decision. Coverage,
  reviewed-window count and total-window count are now sealed into bundle
  provenance and survive fine-review resealing. A fully resolved subset can no
  longer lose the coverage error and become `official` accidentally.
- Started Game A fine review from raw period 1 only. The first 12/65 windows
  cover 0--240 seconds and emitted 56 conservative coarse candidates. Fine
  review through candidate pair 12 now contains 25 hash-bound decisions. It
  rejects passing/stoppage false positives, corrects shifted event windows,
  identifies ten raw-visible players, and preserves a miss/offensive-rebound/
  putback sequence by adding the putback omitted by coarse review.
- The resealed partial Game A bundle
  `6f2a6bcf85ec61bfd3e9c69c4bb51e007c11c0191fc240048aea902309ae1197`
  contains 13 accepted events, 32 still-unresolved candidates and immutable
  provenance `review_complete_video_coverage=false`, `12/65`. It correctly
  remains `needs_review`; event points are raw black 6 and raw white 4, but this
  is progress evidence rather than an accuracy claim.
- Full regression now passes 207 tests plus the harness gate. The local API
  hook completed end to end on task `d42fb5eae65440148c21e957561ad674`
  after explicitly constraining the ephemeral runtime to OpenCV 4.x; the
  project-resolved OpenCV 5.0 package does not expose `CascadeClassifier`.
  This compatibility issue remains an environment/dependency follow-up and is
  not treated as evidence toward the two-full-game accuracy gate.
- Continued the same raw-only fine review through candidate pair 18. The ledger
  now contains 37 decisions, 19 accepted events and 20 unresolved candidates.
  It rejects stoppage and passing false positives, records the gray-shirt
  player's missed free throw, FK 7's offensive rebound and subsequent miss,
  black 21's offensive rebound, and beige 14's missed three with the gray-shirt
  player's defensive rebound. The resealed partial bundle is
  `56ef8d796e6577d8730efd2e4580e3487e117335ecfe5e4fe6c1ba5893010fb7`.
- Fixed reconciliation so a rebound may link to either a missed field-goal or
  missed free-throw attempt. The new regression passes in the 19-test focused
  official-statistics suite. The partial result still fails closed solely on
  `incomplete_video_coverage`; it is not an accuracy claim.
- Full regression with the OpenCV 4.x runtime override passes 208 tests and the
  harness gate. The unoverridden OpenCV 5 environment again fails only the three
  known cascade/legacy-tracker compatibility tests (205 pass).
- Completed fine adjudication for all candidates emitted by Game A period 1's
  first 12/65 dense windows. The 59 hash-bound decisions leave zero unresolved
  events and 28 accepted events. This batch rejects the remaining inbound/pass/
  dribble false positives, records EBBA 111's missed right-wing three and FK 7's
  offensive rebound, and preserves the previously added paired FK 7 turnover /
  James 6 steal. Reviewed bundle
  `3f67effb76efa58c53f7d5f21c84bde9e5a1f54cfec7490eca746a43b56f2071`
  remains `needs_review` solely because complete-video coverage is still false.
- Extended the append-only coarse review to 40/65 windows (0--800 seconds).
  The importer accepted 159 conservative shot/free-throw/rebound candidates and
  sealed bundle
  `88c947b6bf8720d5a6be02a260cc83436dedd44373fad8aebffaf564ad55c85c`.
  Window 33 is explicitly marked `no_event`; all cross-window candidates name
  only already-reviewed support windows, preserving the evidence-window gate.
  These items remain recall-oriented candidates, not accepted statistics or an
  accuracy claim.
- Extended hash-bound fine adjudication through candidate pair 32 against the
  earlier 0--640-second bundle. The 71 decisions yield 37 accepted events and
  72 unresolved candidates. Fine frames reject pass/inbound false positives,
  add black 21's made putback, pair NDP 0's turnover with EBBA 627's steal, and
  link FK 7's assist to black 21's made layup. Reviewed bundle
  `acb3bc0ed7d12644e3efbe8755fd6129c9edb90971297e0ac644100d62d44c5e`
  remains `needs_review` because candidate adjudication and video coverage are
  incomplete.
- Completed all 65/65 dense windows for Game A period 1. The strict importer
  accepted 256 recall-oriented candidates with `complete_video_coverage=true`
  and sealed bundle
  `a55d914b036a3e0474df865d8635b6ea8e8f1c6f957fc540c590606392fef49c`.
  Windows 62--65 are explicit no-event decisions covering the stopped/end-of-
  period footage; no incomplete-coverage override is used.
- Generated the complete 256-candidate, 525 MB fine-review package. Exact JSON
  comparison proved that its first 136 events are unchanged from the previous
  0--640-second bundle, allowing the 71 existing decisions to be mechanically
  rebound to the new bundle hash without changing labels or evidence. Applying
  them sealed reviewed bundle
  `de502943b437a07a6a513214ead8677ba1297cf306fa3edf67ac174c40af20ff`:
  37 accepted events, 192 unresolved events, and no coverage reconciliation
  issue. The box score remains `needs_review` solely because fine adjudication
  is incomplete.
- Advanced full-package fine adjudication through candidate pair 48. The 105
  hash-bound decisions seal reviewed bundle
  `4b7eea20918b6b2e19582cf8b031a1acba7792014c0e6093118eba7d4ed3ea70`
  with 50 accepted events and 160 unresolved events. The batch confirms James
  6's missed layup and the black gray-shirt headband rebound, FK 7's missed
  three and NDP 0 rebound, rejects a missed shooting-foul action as an FGA,
  rejects FK 7's dead-ball practice shot, and retypes the two official Dallas
  77 attempts as made free throws. Reconciliation is valid with no issues;
  `needs_review` remains fail-closed until all 256 candidates are adjudicated.
- Advanced full-package fine adjudication through candidate pair 64. The ledger
  now contains 138 unique hash-bound decisions and seals reviewed bundle
  `12ec531cc979013e2d764161f249502e2f80ed1870b96b53cc09baf512995204`
  with 65 accepted events and 128 unresolved events. Raw-only evidence adds a
  causally proven FK 7 turnover without inventing a steal, identifies the newly
  visible black player as stable ID `raw_black_ebba_85`, records James 6's made
  and-one free throw, and handles EBBA 85's two missed free throws, offensive
  rebound, missed putback, and later defensive rebound. Reconciliation remains
  valid with raw-black/raw-white event points 10/16 and no issues. This is still
  a partial first-period result and is not an accuracy-gate result.
- Advanced full-package fine adjudication through candidate pair 72. The ledger
  contains 154 unique, source-bundle-hash-bound decisions and seals reviewed
  bundle `8bc6f4e0cee00b309f2006ebc2bf0a26b48bee2e439c34077f5fa3e22ddefac7`
  with 77 accepted events and 112 unresolved events. Raw-only evidence confirms
  six missed tries and their six rebounds, while rejecting two pass-only
  shot/rebound pairs. It adds appearance-derived stable IDs
  `raw_black_ebba_24`, `raw_white_gray_long_sleeve`, and
  `raw_white_white_bib_green_shorts`; none is mapped from external statistics.
  Reconciliation remains valid at raw-black/raw-white event points 10/16 with
  no issues. This remains an incomplete first-period audit and is not a full-
  game strict-F1 result.
- Corrected the architecture boundary after explicit user review: Codex is now
  treated only as an isolated annotation/acceptance-truth producer. Added an
  autonomous bundle gate requiring `producer=agu` and
  `inference_mode=traditional_cv+vlm`; it rejects Codex/human statuses,
  non-edge reviewers, and reviewer/reference markers in provenance/evidence.
- Added AGU bounded official-event VLM adjudication. It can only select
  player/team IDs emitted by traditional tracking evidence, confirms only
  schema-complete events, and otherwise retains `needs_review`. Added a strict
  evaluator mode `--require-agu-autonomous` so reviewed-label F1 cannot satisfy
  the autonomous 0.85 gate.
- Extended traditional perception with ByteTrack IDs, deterministic raw player
  IDs, two-cluster torso-Lab team assignment, and nearby player/team candidate
  attachment to shot/rebound proposals. Configuration remains under the
  `BASKETBALL_` settings layer.
- A real Game A 0--60 second E-BARD smoke sampled 258 frames and emitted 1495
  player, 71 rim, 104 referee, but only 2 basketball detections. Tracking
  fragmented into 75 raw player IDs and ball/rim fusion emitted zero events.
  The resulting empty autonomous bundle is valid structurally but is explicit
  evidence that current perception quality is below the 85% target; Codex was
  not used to fill the missing events.
- AGU accurate R(2+1)D/tracking rerun for the same interval completed in
  127.09 seconds with 465 action records, three segments and 51 long-video
  player IDs. It emitted no `shoot` action (dominant labels were dribble,
  ball-in-hand and defense) and only two block candidates, which were excluded
  from the official shot/rebound scope. Fusing it with ball/rim perception still
  produced zero official-event candidates.
- Only after sealing the empty AGU bundle, the isolated Codex acceptance truth
  was opened: the interval contains three field-goal attempts and one rebound.
  `--require-agu-autonomous` evaluation therefore reports candidate and strict
  TP 0 / FP 0 / FN 4 / F1 0.0. This is the first honest autonomous result for
  the corrected architecture and establishes dense AGU VLM recall plus stable
  identity as the next implementation target.
- Added dense AGU-owned VLM candidate discovery over overlapping raw-video
  windows, with explicit context-length/timeout controls and fail-closed HTTP/
  parse handling. The first run silently converted Ollama HTTP 400 context
  overflow into empty windows; this was corrected so backend failure aborts.
  A 16K context and resumable cache now support bounded multi-frame discovery
  and adjudication.
- On Game A 0--60 seconds, dense qwen3-vl discovery raised type/time recall from
  0/4 to 3/4 (75%) but emitted 29 candidates, 26 false positives (precision
  10.34%, candidate F1 18.18%). A second AGU VLM pass rejected 13, confirmed 9
  and left 7 unresolved; post-adjudication localization was TP 2 / FP 14 / FN 2
  (F1 0.20), while strict automatic F1 remained 0 with 9 FP and 4 FN.
- Tightened deterministic semantic constraints after this run: selected team
  must be bound to the selected traditional player observation, non-shot events
  cannot receive outcome/shot-value labels, and the prompt now requires visible
  release/flight, made result and three-point-line evidence. The current local
  qwen3-vl:4b remains overconfident on dribble/gather frames, so model quality,
  ball trajectory recall and identity continuity—not Codex review volume—are
  the active blockers.
- Added 15-fps multi-detector fusion using the specialist basketball/rim model
  plus the optional generic sports-ball detector. Distance/confidence filtering
  and trajectory-based made-shot rebound suppression raised the sealed
  pre-adjudication 0--60 second candidate result to TP 4 / FP 1 / FN 0 (F1
  0.8889) without opening truth during inference.
- Fixed temporal identity overlays so a track box appears only near its actual
  observation frame. Traditional trajectory outcomes and outcome frames now
  take precedence over conflicting VLM guesses; linked rebounds require an
  automatic confirmed miss; and unsupported three-point claims remain
  unresolved. Actor candidates are restricted to a configurable five-second
  lookback before a resolved outcome.
- The resulting sealed Game A 0--60 second autonomous bundle localizes all four
  truth events (TP 4 / FP 0 / FN 0, F1 1.0). It automatically confirms three
  semantically correct events and leaves one made shot unresolved because its
  2/3-point evidence is incomplete. Raw unaligned strict F1 remains 0 because
  transient track IDs are not roster identities. A post-seal, hash-bound,
  one-to-one identity-aligned diagnostic is TP 3 / FP 0 / FN 1 (F1 0.8571);
  each player appears only once, so this is a semantic diagnostic, not proof of
  cross-window identity or either complete-game gate.
- Connected the previously isolated identity embedding and graph capabilities
  to the official raw-only path. Perception artifacts and identity artifacts
  are now bound to raw filename/SHA-256; track fragments are embedded from raw
  crops, merged with a conservative complete-link graph, and mapped to
  canonical IDs before event candidate construction. Same-video simultaneous
  spatially distinct players are hard conflicts, while overlapping duplicate
  detector boxes can merge.
- Evaluated the existing optional open-source identity backends on the Game A
  0--60 second interval. MobileNetV3 at threshold .92 reduced 136 tracklets to
  79 identities; OSNet x0.25 at the safer .80 threshold reduced them to 42.
  A lower .55 OSNet threshold produced 22 identities but is not accepted as the
  default because same-time negative pairs reached high cosine similarity and
  therefore expose false-merge risk. These counts demonstrate improved
  continuity but not roster-level identity correctness.
- Changed VLM evidence overlays to deterministic short aliases (`P01`, `P02`,
  ...) that map back only to traditional canonical player IDs. Also restricted
  shooter evidence to the pre-outcome/pre-first-rim window so rebounders cannot
  be selected merely because they appear after a shot. On the same interval,
  qwen3-vl:4b still resolved only two of five candidate events automatically
  and assigned one rebound to the wrong team. This identifies semantic model
  quality as a current bottleneck without using Codex to repair predictions.
- Reused the same raw-hash-bound canonicalization helper in dense AGU VLM
  discovery, not only the shot/rebound builder. This closes an identity
  provenance gap for assist, block, steal, turnover, foul and free-throw
  proposals discovered by the all-event VLM path; the sealed candidate bundle
  now records the identity artifact hash and embedding model.
- Added deterministic causal linking for dense all-event candidates and
  topological parent-first adjudication. Rebounds link to preceding shot/free-
  throw attempts, assists and blocks to nearby field-goal attempts, and steals
  to nearby turnovers. Links are hypotheses only: assist requires an automatic
  confirmed same-team make, block an automatic confirmed opponent shot, steal
  an automatic confirmed opponent turnover, and rebound an automatic confirmed
  miss with team-consistent offensive/defensive type. Missing or contradictory
  parents fail closed instead of producing official statistics.
- Added a hash-bound training-annotation firewall for the newly authorized
  Codex labeling role. Codex may generate detector/tracker/OCR/action-model
  labels only through an immutable training manifest that is proven disjoint
  from sealed acceptance-video hashes and marked `runtime_consumable=false`.
  Autonomous bundles may disclose this training provenance, while runtime
  Codex review, prefilled events and Codex evidence remain rejected.
- Game B's raw 120--300 second interval now has independent specialist and
  generic perception, plus a raw-bound OSNet identity graph. Role-routed
  traditional candidates reached TP 14 / FN 3 on the available 17-event
  partial truth (recall 0.8235), but with 49 FP (candidate F1 0.35). This is a
  partial-category candidate result, not an autonomous or complete-game gate.
- Added detector-role routing so an auxiliary artifact may contribute tracked
  players without letting its lower-quality ball/rim detections bridge event
  clusters. Legacy action candidates are now rebound to current canonical raw
  player IDs rather than retaining stale IDs. Dense VLM discovery writes an
  atomic, fingerprinted per-window cache and resumes safely after interruption.
- Installed and evaluated AGU's local `qwen3-vl:8b` backend. A 12B MLX backend
  was rejected as protocol-incompatible after three null decisions. For the
  8B path, event-anchor-centered frame sampling fixed long-window dilution.
  Continuous actor observations were preserved, while each displayed frame is
  limited to the two ball-nearest aliases to avoid overlay clutter.
- Split official VLM inference into two AGU-owned passes: clean raw frames
  determine occurrence/outcome/value, then release/control-centered overlaid
  frames determine actor/team. Displayed frame indexes are now mapped against
  the actual sampled bounds, and a single grounded alias named in the VLM
  reason can repair a malformed alias field without accepting an invented ID.
- On sealed Game A 0--60 second data, the two-pass 8B path automatically
  confirmed all three field-goal attempts and rejected both proposed rebounds.
  Localization is TP 3 / FP 0 / FN 1 (F1 0.8571); the missing event is the true
  rebound. After the predeclared global roster mapping, only one shot has the
  correct actor, yielding strict TP 1 / FP 2 / FN 3 (F1 0.2857). Raw strict F1
  remains 0. These results prove better event semantics but also prove that
  shooter/rebound ownership and identity continuity remain below the target.
- Strict evaluation also requires the affected/related player. For automatic
  assists, blocks and steals, AGU now derives `secondary_player_id` only from
  the already automatically confirmed parent event (scorer, shooter or player
  charged with the turnover). The VLM cannot invent this identity.

2026-07-18 open-pretraining and action-owner increment:

- Added a BARD annotation adapter and imported a sealed 14,676-record catalog
  without downloading linked NBA media. The catalog is pretraining-only,
  non-runtime-consumable, records CC BY 4.0 attribution, and explicitly marks
  media rights as unverified. It normalizes field-goal attempts, rebounds,
  free throws, fouls, turnovers, steals, blocks and violations.
- Added ball center/bbox propagation into actor observations and four
  wrist-to-ball temporal features. A pairwise ranking objective replaced
  independent-row classification, but the small dataset did not meet the
  autonomous integration gate.
- Fixed cross-backend tracker collisions by namespacing non-primary track and
  player IDs. Fixed a second integration defect by applying a declared identity
  graph to its raw perception source before namespacing; otherwise the graph
  artifact was valid but could never match additional-source IDs.
- Added conservative multi-frame trajectory NMS so duplicate boxes from two
  detection backends do not consume all actor candidates. Candidate capacity
  is configurable and defaults to 16. On a disjoint training interval, the
  actual black-number-13 shooter moved from rank 21 (only visible with 32
  candidates) into the bounded 16-candidate set at rank 14 using AGU perception
  alone.
- Added per-candidate temporal crop sheets. Codex selected the existing
  `raw-player-track-144` only for verified frames 3938--3942; the label builder
  bounded observations to that segment to exclude a later tracker identity
  switch. This label is SHA-bound, benchmark-disjoint and
  `runtime_consumable=false`.
- The updated pairwise linear experiment contains 8 actions across 7 source
  videos (65 rows / 114 pairwise rows). Leave-one-video-out pairwise F1 is
  0.7544 and Top-1 actor accuracy is 0.375. The model remains experimental and
  is not connected to AGU runtime; more clean actions and a nonlinear ranking
  model with independent acceptance are required.

2026-07-19 action-owner label audit and independent acceptance:

- Candidate crop sheets proved that the old 151732 P03 label followed black
  number 527 in the foreground; AGU candidate P04, white number 0, visibly
  lifts, jumps and releases. A bounded superseding label covers frames
  3334--3350. The older file remains immutable audit evidence but is excluded
  from training. The older 153430 duplicate action is also excluded. The
  separate 160824 white-number-13 label was visually re-audited and retained.
- Manifest v11 seals 16 actions from eight non-acceptance videos and retains
  `benchmark_overlap=false`. Kinetics-400 R(2+1)D embeddings were extracted by
  a reusable, training-only adapter, but embedding-only and fused experiments
  did not improve cross-video Top-1, so they are not used at runtime.
- An Extra Trees classifier over AGU's 13 temporal ball/wrist/pose features
  reaches strict leave-one-video-out Top-1 14/16 = 0.875. The 100-tree,
  depth-3, minimum-leaf-2 model is serialized to a hash-sealed,
  sklearn-independent runtime schema with training-manifest provenance.
- An experimental two-pass adapter lets this traditional model score AGU-generated
  candidates after the semantic VLM supplies a release anchor. Scores only
  prioritize the actor overlays; they do not prefill `primary_player_id`, and
  the actor VLM must still validate the visible player. This preserves the
  Codex-training-only boundary while making AGU traditional CV + VLM the
  actual recognizer.
- The independent Game A 0--60 second acceptance run invalidated promotion.
  Absolute model-prior overlays selected unrelated foreground players and
  produced aligned strict F1 0.0. A conservative fusion that always keeps the
  ball-nearest player alongside model Top-1 restored one correct shot and F1
  0.2857, exactly the prior baseline, but did not improve it. The artifact
  remains default-off and cannot be accepted for production inference.

2026-07-19 exact-frame, raw-face and causal-rebound follow-up:

- Mapped VLM `release_frame_index` to the exact raw frames actually displayed
  after stride sampling and even selection. This fixes an inference contract
  defect, but a sealed independent run still scored strict F1 0.0, proving that
  frame-index interpolation was not the dominant ownership error.
- Added raw SFace cosine evidence to the identity graph even when no enrollment
  gallery is configured. The reproducible MobileNetV3 + SFace artifact contains
  136 tracklets, 72 identities, 66 face-bearing tracklets and 14 identity groups
  with face-cosine evidence. It improves evidence coverage but cannot assign a
  roster identity to distant/back-facing tracks without an independent gallery.
- Simultaneous same-video tracklets with no aligned spatial observations now
  hard-conflict instead of silently merging. Close aligned detector duplicates
  remain mergeable. Existing OSNet artifacts cannot currently be rebuilt because
  `torchreid` is absent from all local Python environments; no fallback result is
  represented as OSNet.
- Added repeated-contact control features for rebound candidates. The first
  stable run of at least two player/ball observations ranks ahead of a single
  near-ball tip. Added an optional maximum ball/rim cluster span; its sealed
  2-second probe reached recall 1.0 but candidate F1 only 0.7273 because of three
  false candidates, so the option remains disabled by default.
- The clean semantic pass can now constrain rebound actor candidates causally:
  a defensive rebound after a confirmed dark-team miss exposes only light-team
  tracked players, and vice versa for an offensive rebound. This uses no truth
  or Codex answer.
- Sealed autonomous bundle
  `e160a39d9eb7f65b8068db94a8e5af97664ee98ea0c774dd9d32e3240b4f04ac`
  confirmed all three field goals plus the true defensive rebound and rejected
  the made-shot rebound. Candidate localization is TP 4 / FP 0 / FN 0, F1 1.0.
  Raw strict F1 remains 0: the second white shooter and subsequent white
  rebounder were assigned swapped/fragmented canonical IDs. This is direct
  evidence that the benchmark-disjoint annotated face list requested by the
  user is now the active identity prerequisite, not permission to use Game A
  reference clips as runtime answers.

2026-07-19 annotated-face-list identity dedup increment:

- Added direct full-frame YuNet face candidate extraction so enrollment no
  longer depends on contaminated player-track boxes. Complete-link clustering
  uses same-frame non-merge constraints and emits review sheets plus a sealed
  source manifest.
- Added exhaustive face-only approval. Codex may merge repeated views of the
  same person and reject officials/invalid faces, but supplies no action or
  statistic labels. Every source cluster must be approved or rejected exactly
  once before an `agu.annotated-face-list.v1` can be produced.
- The full nine-video July scan detected 371 faces and retained 25 strong
  clusters. Face-only review merged repeated clusters and rejected six official
  clusters, producing 12 registered players and sealed gallery
  `27d771a5cef3cc79c6149c9696e42672f6550f3c5c746fc8a4455e0cc81fc304`.
- Rebuilding Game A 0--60 seconds produced 136 tracklets / 68 identities and
  four unambiguous gallery anchors. All six shot/rebound windows still have no
  registered actor nearby, so mandatory enrollment filtering returns empty
  actor lists instead of falling back to anonymous IDs. This quantifies roster
  coverage as the next data prerequisite; it is not an inference success.
- Official identity construction now rejects explicitly untracked perception
  (`player_tracking=false`) before crop extraction, preventing generic detector
  artifacts with one shared temporary player ID from silently entering dedup.

2026-07-19 enrolled-identity propagation correction and cross-game coverage:

- The first coverage read incorrectly queried a nonexistent `player_candidates`
  field. The actual v19 bundle had propagated one registered identity into five
  action candidates. Raw truth contains different teams and actors, so this was
  a real identity false-positive rather than successful coverage.
- Identity graph v3 now prevents a registered identity from absorbing an
  anonymous track through body ReID alone. Every added track needs direct
  quality-gated face similarity to an enrolled member or a trusted matching
  jersey on both sides. Body ReID remains available only for anonymous groups.
- With this rule the same Game A window no longer assigns one person to the
  second and third shots. The v25 autonomous result retains localization
  TP 4 / FP 1 / FN 0 (F1 0.8889), but strict TP 0 / FP 1 / FN 4 (F1 0.0):
  abstention prevents invented identities but independent enrollment still
  lacks the true white and black action owners.
- A full January scan produced 695 faces / 63 clusters. Exhaustive face-only
  review merged repeated views and rejected 11 officials, yielding 20 players
  for cross-game development. Across 11 existing July action-owner labels, all
  raw tracks resolve but only one has an exact unambiguous enrolled identity.
  The gallery therefore overlaps the venue population but not the active action
  roster sufficiently for an 0.85 gate.
- Added exhaustive cross-list face merging. A separate 2024 first-quarter video
  yielded four people and added two identities after cross-source deduplication;
  the combined 14-person gallery reaches more Game A windows, but one low-quality
  identity still spans two truth actors at 0.55--0.65 SFace confidence. It is
  diagnostic only and is not promoted.
- Asset audit found one complete January game with full six-stat reference, one
  July multi-file game already used for action-owner training, and an independent
  `第一节.mov` with only a partial three-category truth interval. These do not
  constitute two independent complete-game six-stat acceptance sets.
- Added a configurable minimum enrolled-face quality gate (default 0.65) to
  both direct gallery matching and registered-ID propagation. It removes the
  combined-gallery false identity whose query quality was 0.50--0.56 and whose
  SFace score was 0.55--0.65. The v31 candidate bundle then leaves all six
  action-owner lists empty. This is the intended high-precision abstention but
  confirms that stronger registration coverage, not a looser threshold, is
  required for recall.
