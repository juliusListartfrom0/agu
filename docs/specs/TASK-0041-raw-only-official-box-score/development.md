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

2026-07-20 multi-prototype enrollment and truth-free semantic probe:

- Replaced single averaged enrollment embeddings with up to four independently
  supported face prototypes per player. Each prototype requires at least two
  mutually compatible observations; old v1 galleries remain readable. ATL-CHI
  strict roster coverage improved from 13/23 to 19/23 without lowering the
  `quality>=0.65` gate. Two players remain unenrolled and two remain low quality.
- Added confidence-weighted, one-to-one inference from raw uniform teams to
  canonical roster teams. It requires at least two direct face anchors per
  assignment and a decisive global margin. On ATL-CHI it mapped all 1570
  event-related tracklets into ATL/CHI namespaces, while registered names still
  require direct face or trusted jersey evidence.
- Added conservative short-segment ball trajectory linking around the rim. It
  reduced insufficient local trajectory fragments but did not materially change
  full-game outcome recall on either public game, so no accuracy claim is made.
- Added a deterministic, raw-only, time-stratified candidate probe and a
  semantic-only local VLM mode. `qwen3-vl:8b` with reasoning disabled avoided
  the prior 4B all-made degeneration. A 9-shot contact-sheet probe preserved
  the 2 made/1 missed traditional results and abstained on six unknowns. A
  16-frame temporal probe then resolved 1/3 unknown shots as missed from the
  explicit miss-plus-rebound observables and abstained on 2/3.
- These are calibration results, not acceptance results. Full-game predictions
  have not been frozen, truth has not been opened for isolated evaluation, and
  neither two-game strict F1 >=0.85 nor the final 0.95 target is established.

2026-07-20 evidence-preserving semantic adjudication:

- Fixed the decision layer so high-confidence, schema-valid partial semantics
  survive on `needs_review` revisions when actor identity is incomplete. Such
  revisions remain excluded from official aggregation. Low-confidence or
  unavailable results still discard all proposed labels.
- Fixed two grounding-loss defects: observables now accompany the second label
  validation, and cached three-point results reuse the exact feet/line/beyond-
  arc booleans instead of silently deleting a previously grounded three.
- Candidate evidence now carries a detector-cluster midpoint plus raw ball/rim
  boxes. The semantic runner can optionally render a yellow-bordered same-frame
  rim detail inset. This is a raw-derived visualization, default-off, and adds no
  made/missed label.
- Outcome v14 derives made/missed only from non-conflicting observables. A full
  above/inside/below sequence is made; visible rim exit, clear rim miss or a
  subsequent rebound is missed; simultaneous made and miss evidence abstains.
  The prompt defines rim exit as mutually exclusive with downward passage.
- Truth-free three-shot probes on both ATL-CHI and LAL-BOS each resolved 2/3
  prior unknowns while retaining all events as `needs_review` because identity
  was incomplete. Post-freeze Codex inspection supported the conservative ATL
  outcomes but could not reliably verify both LAL makes from ordinary raw-frame
  contact sheets. Full-game VLM expansion was therefore intentionally withheld;
  the 4/6 resolvability result is not an accuracy result.

2026-07-20 raw-broadcast scoreboard evidence:

- Added a RapidOCR adapter for the lower-third score bug. Reads require both
  expected teams, a live period marker, geometrically adjacent score digits and
  repeated stable states. Replay decreases, one-off states and jumps above
  three remain rejected.
- Added a resumable full-video runner. Its OCR cache binds raw video and reader
  configuration; candidate linking is rebuilt separately and records the exact
  source-candidate hash, so one raw scan can safely follow identity-graph updates.
- Sparse intervals retain each team's independent 1/2/3-point increase. A delta
  binds only to exactly one traditional shot candidate; otherwise AGU emits an
  actor-null `needs_review` event.
- Full v3 scans froze 619 readable LAL-BOS frames and 1228 readable ATL-CHI
  frames. Post-freeze team-row comparison produced made-value component micro
  F1 0.8600 and 0.9703 respectively. This is not player-level six-stat F1.
- Reuse with identity-mapped candidates found only one LAL scoring event with a
  unique named candidate and none on ATL, so named identity coverage is now the
  primary blocker.

2026-07-20 raw-audio, registered-candidate and live-broadcast increment:

- Completed both enrollment scoreboard scans and base-English raw-audio runs.
  Speech scoring calibration was not promoted: pooled precision was 7/8 =
  0.875, but per-game coverage was only 0.0761 and 0.0110 and the minimum
  predictions gate failed. ASR remains candidate evidence and never writes an
  actor ID.
- Added truth-free registered-candidate probe selection against a sealed face
  gallery, including candidates found in traditional evidence rather than only
  a prefilled primary ID. Causal parents can be included without reading PBP or
  box-score answers.
- Fixed two runtime defects exposed by real probes: high-confidence visual
  absence now rejects a dependent event even if its parent is unresolved, and
  an optional frame-bounds resolver can no longer crash combined inference by
  returning `None`.
- LAL–BOS registered-primary probing rejected all three proposed Odom rebound
  candidates. ATL–CHI connected registered candidate `2586` to a made shot, but
  post-freeze raw-frame inspection proved it was a halftime replay. Added a
  mandatory semantic live-broadcast gate; replay/highlight/studio/break evidence
  wins over a contradictory live flag and clears all official labels. The same
  frozen ATL candidate is now rejected.
- These changes improve precision and auditability only. LAL–BOS has three
  canonical registered primary events in the full candidate bundle; ATL–CHI has
  none, although IDs 2586 and 2746 occur among candidate observations. Identity
  recall and live-play-aware event localization remain the blocking paths.

2026-07-20 registered-jersey identity increment:

- Fixed inferred raw-uniform team bindings so they can never overwrite the
  canonical team supplied by a direct face-gallery match. The corrected ATL
  graph keeps `2586` on CHI and removes cross-team `2037`/`201565` corruption.
- Added optional jersey-number registration metadata to the hash-sealed face
  gallery plus a separate `agu.face-jersey-annotation.v1` boundary. Labels may
  contain only person/team/number enrollment metadata; every person must already
  exist in the face gallery and jersey numbers must be unique within a team.
- Added two adapters: Codex-assisted public roster labels can be attached by
  `attach_face_jersey_annotations.py`, while
  `enroll_face_gallery_jersey_numbers.py` can independently read expanded torso
  crops from benchmark-disjoint enrollment video with AGU VLM partition
  consensus. Neither path accepts game event or statistic answers.
- Runtime jersey propagation is fail-closed. AGU must read the same number from
  two crop partitions at confidence >=0.90, normalize the raw uniform team from
  multiple direct face anchors, find exactly one matching registered face entry,
  and find no duplicate same-team jersey component in the identity graph.
- A four-track ATL probe read `15`/`15` and `13`/`13`, mapping live raw tracks to
  registered IDs `201143` and `201149`. A `3`/`43` conflict and a 0.85 read stayed
  anonymous. These two tracks now appear in four traditional event windows each.
  This demonstrates the identity path but is not a complete-game accuracy result.
- A subsequent actor probe exposed colliding raw Track IDs across the primary
  BODD source and the dedicated BoT-SORT player source. Candidate construction
  now treats explicitly supplied player-perception artifacts as authoritative
  for every player/actor observation; primary and other additional artifacts
  continue to contribute ball/rim detections only. Actor identity therefore
  follows `actor raw track -> runtime jersey -> registered face entry`, never
  `any named player visible in the event window -> actor`.

2026-07-20 authoritative actor-to-registered-identity increment:

- The six-event ATL shot probe now uses only the dedicated BoT-SORT player
  artifact for actor observations. AGU's two-pass VLM selected four raw actor
  tracks across five live attempts and rejected the halftime replay before any
  identity propagation.
- Actor-track jersey reads were then scoped to those four tracks. Two passed
  independent crop-partition consensus (`1/1` and `13/13`, each at 0.95), one
  failed closed on `9/34`, and one failed closed because its only visible read
  was 0.85, below the 0.90 gate.
- Registered-jersey propagation now counts only competing runtime jersey-only
  identities. A pre-existing direct face anchor for the same enrolled person
  no longer blocks a unique runtime mapping, while two runtime tracks claiming
  the same team/number still block one another. This maps the accepted tracks
  to registered IDs `201565` and `201149` without weakening duplicate defense.
- Official VLM result caches are now keyed by the exact rendered prompt and
  image bytes. Identity-only graph changes therefore preserve clean semantic
  results, while changed actor aliases/overlays still invalidate actor review.
- The rebuilt 290-event candidate bundle has exactly the same event frames and
  confidences as the prior authoritative bundle. Only its raw-grounded identity
  fields changed; no box-score or play-by-play answer entered the rebuild.

2026-07-22 real-shot precision gate increment:

- Added candidate-window pose batching. It verifies both the sealed candidate
  bundle and raw-video SHA, merges padded overlapping windows, loads the pose
  adapter once, and emits a raw-bound `agu.official-pose.v1` artifact.
- Added an 18-feature traditional shot-validity contract spanning ball/rim
  proximity, linked trajectory, player distance, window shape and temporal pose.
  A rejected shot also removes its synthetic rebound; an accepted shot remains
  `needs_review` and does not receive actor, outcome or statistic answers.
- Added SHA-bound training and truth-free annotation export. Extra Trees uses
  leave-one-video-out probabilities to choose the highest-recall threshold that
  satisfies the configured precision floor. Training requires at least two
  disjoint games and both classes in every training fold.
- The model is optional in candidate construction and is not promoted by
  default. ATL–CHI and LAL–BOS are development/calibration assets after prior
  inspection; final acceptance requires two newly frozen unseen full games.
- Full development pose extraction then completed: ATL produced 53,394 poses
  over 6,534 samples and LAL produced 53,068 poses over 8,608 samples. Candidate
  geometry remained exactly unchanged. Pose/wrist features cover 138/145 ATL
  and 174/182 LAL shot candidates, with mean best-pose coverage 0.934/0.923.
- The first full rebuild exposed quadratic pose attachment. Replacing the global
  pose scan with a frame index preserves deterministic IoU matching while making
  the two complete candidate rebuilds practical.
- Generated 145 ATL and 182 LAL timestamped contact sheets directly from the
  pose-enriched, raw-bound candidate windows. They are marked training-only and
  paired with `event_present=null` templates; no label has been guessed from a
  box score or fed into runtime inference.
- Independently reviewed 71 distributed development windows (25 real shots
  and 46 false candidates) and sealed their source-video, candidate,
  pose, annotation and manifest hashes. The labels remain training-only.
- The pose/geometry Extra Trees gate reached leave-one-video-out precision
  1.000, recall 0.053 and F1 0.100 at the 0.95 precision floor. Coherent-player
  pose features did not recover the missing releases, so this representation
  was not promoted.
- Added a full-event-window temporal adapter backed by a locally frozen
  Kinetics-400 R(2+1)D-18 encoder and a sealed standardized logistic head. It
  verifies the raw video, candidate bundle, backbone and model hashes and never
  reads the training embedding/label artifact at runtime.
- The initial 55-window temporal leave-one-video-out result was precision
  1.000, recall 0.368 and F1 0.538, but it did not survive the 71-window
  expansion. The default head abstains completely (0/0/0); alternative
  regularization recovers at most 2/25 positives at precision 1.000. Promotion
  now additionally requires the precision floor and minimum recall on every
  held-out video, preventing pooled-score promotion.
- On eight newly frozen ATL windows, Qwen3-VL 8B with one contact sheet missed
  all three positives and accepted two negatives (0/0/0). Twelve separate raw
  frames recovered all positives but accepted four negatives (P/R/F1
  0.429/1.000/0.600). The semantic gate now requires independently visible
  control, hand separation, and post-separation progress toward the rim. The
  rerun still hallucinated the complete chain on the same four negatives, so
  metrics remained 0.429/1.000/0.600 and the VLM cannot be the primary gate.
- The retry pipeline completed one additional training game with 34 shot
  candidates and YOLO11 pose windows. Codex independently labeled 24 raw-only
  contact sheets (11 positive/13 negative), then sealed a three-game,
  95-window manifest (36 positive/59 negative). Frozen-backbone heads across
  C=0.001--10 all failed promotion; the best was P/R/F1
  1.000/0.056/0.105, with zero recall on two held-out games. The next model
  iteration must fine-tune temporal backbone layers with game-held-out
  evaluation and broadcast-style augmentation.

2026-07-23 end-to-end temporal fine-tuning increment:

- Codex reviewed all 23 truth-free contact sheets from the fourth development
  video, yielding 2 positives and 21 hard negatives. The new four-game sealed
  manifest contains 118 windows (38 positive/80 negative) and excludes two
  newly reserved MEM-OKC benchmark hashes.
- Added end-to-end R(2+1)D fine-tuning with SHA-authorized record loading,
  dense raw-window decoding, temporal jitter, crop/flip/color augmentation,
  broadcast-overlay masking, frozen batch-normalization statistics and
  configurable late-stage unfreezing.
- Leave-one-game-out calibration now derives each fold threshold only from its
  training games and applies normalized logit margin to the unseen game. This
  avoids comparing uncalibrated logits from four separately trained models.
- The Kinetics layer4+fc five-epoch run reached outer-fold TP/FP/FN=1/1/37
  (P/R/F1=0.500/0.026/0.050). Initializing from the basketball-specific AGU v3
  SpaceJam checkpoint while preserving its BGR/[0,255] contract reached
  1/4/37 (0.200/0.026/0.047). Both artifacts fail closed and remain
  `runtime_consumable=false`.
- Downloaded and sealed unseen 2011-05-11 and 2011-05-13 MEM-OKC full games,
  plus a disjoint 2011-05-15 face-enrollment game. No truth asset has entered
  training or inference. Outcome/rebound and secondary-event work remains
  gated on per-game shot P>=0.95 and R>=0.85.
- Generated 44 labeled montage pages for the remaining 256 raw-only ATL/LAL
  contact sheets so Codex-assisted active annotation can continue without
  exposing box score or play-by-play answers.

2026-07-23 complete development-window annotation increment:

- Codex reviewed all 256 remaining raw-only contact-sheet windows without box
  score, play-by-play, highlight, miss or turnover references. ATL added 109
  labels (65 positive/44 negative); LAL added 147 (70 positive/77 negative).
  Each supplemental file is `runtime_consumable=false`, SHA-bound to its raw
  video and candidate bundle, disjoint from the earlier labels, and the union
  exactly covers all 145 ATL plus 182 LAL candidates.
- Combined with the two enrollment subsets, the new four-video training set has
  374 windows (173 positive/201 negative). Manifest SHA is
  `60c9bb7d11af4386e6ca3676dd95ad0188ea7742d04d5d7edd47ec369eaa2236`;
  both reserved MEM-OKC benchmark hashes remain absent from training sources.
- A three-epoch Kinetics R(2+1)D layer4+fc game-held-out run remained below the
  promotion gate. Per-game raw probability AUC was 0.704, 0.760, 0.615 and
  0.476. At recall >=0.85 the best attainable per-game precision was 0.683,
  0.625, 0.478 and 0.133, respectively. No fold can satisfy P>=0.95/R>=0.85;
  checkpoint and metadata remain non-runtime.
- The existing 18-feature pose/trajectory Extra Trees path was retrained on the
  same sealed 374 rows. Leave-one-video-out calibration could not find a 0.95
  precision operating point and also remains non-runtime. More labels alone did
  not remove the cross-broadcast representation gap.
- The next admissible model iteration must improve the open-source video
  representation and/or fuse independently inferred raw-only VLM evidence.
  It may not tune against reserved MEM-OKC truth. Outcome, rebound and secondary
  statistics remain blocked by the unchanged shot gate rather than by missing
  annotation work.

2026-07-23 open-video-backbone screening increment:

- Added a replaceable torchvision registry for the official Kinetics-400
  Swin3D-T and MViT-v2-S checkpoints. AGU owns local checkpoint/SHA validation,
  raw-window extraction, artifact schemas and gates; torchvision remains the
  BSD-3-Clause model provider. No new mandatory runtime dependency was added.
- Added sealed frozen-feature extraction and game-level leave-one-out screening.
  Multi-backbone fusion aligns examples by raw SHA, candidate-bundle SHA and
  event ID, rejects manifest or label disagreement, and remains training-only.
- Codex reviewed 137 additional enrollment windows from raw-derived contact
  sheets. Duplicate scoreboard hypotheses at the same timestamp were collapsed
  to one canonical event. The resulting 511-window manifest has 288 positives,
  223 negatives, and remains SHA-disjoint from both reserved MEM-OKC games.
- Frozen Swin and MViT heads both failed the per-game P>=0.95/R>=0.85 gate.
  Swin+MViT fusion improved the two enrollment folds but the hardest LAL
  benchmark fold reached only P=0.570 at R=0.863. Neither backbone should be
  fine-tuned from this evidence; the next implementation must add independently
  inferred hand-ball-rim causal trajectories and keep generic appearance as
  supporting evidence only.
- Extended the traditional model with four already-computed causal ownership
  signals: minimum wrist-ball distance, player-ball exit slope, wrist-ball exit
  slope and maximum wrist speed. The v2 model schema remains able to load v1
  artifacts, while newly trained models bind the expanded feature contract.
- The trainer now accepts multiple independently sealed candidate bundles from
  one source video and requires pooled precision, per-video precision and
  per-video recall to pass together. On all 511 rows, one ATL enrollment fold
  reached P=0.953/R=0.910, but the other folds reached only
  P=0.686/0.765/0.557 at recall >=0.85. This local success does not justify
  promotion; causal evidence coverage must be generated for the supplemental
  windows before another classifier comparison.

2026-07-23 scoreboard-localization correction:

- Score-delta evidence still retains the full interval between two stable OCR
  states, but candidate event geometry can now be bounded to a configurable
  lookback (12 seconds by default). This separates provenance from the raw
  window consumed by pose, causal features, VLM review and training.
- Codex independently reviewed all 144 short-window field-goal sheets from the
  two enrollment games. ATL has 68 positive/3 negative labels; LAL has 59/14.
  Labels are bound to the new candidate bundle hashes, are training-only, and
  contain no box-score or play-by-play answer.
- The review exposed the earlier failure mode: long OCR intervals included
  unrelated possessions, while delayed OCR recovery and multi-score jumps
  produced same-window team duplicates, free-throw/dead-ball windows and
  statistics graphics. Seven shared ATL decisions changed after localization.
- The new four-game manifest contains 518 rows (300 positive/218 negative),
  remains disjoint from both sealed MEM-OKC games, and has SHA
  `f083be3b8e997dbac604e5b55330d255400255ccca5a6ed36aa32b4e92fe0516`.
- Causal Extra Trees remains non-runtime. At recall >=0.85, per-game best
  precision is 0.683/0.803/0.985/0.553. Short-window correction materially
  improves both enrollment folds, but the benchmark folds still require
  better ball/wrist/rim evidence rather than another classifier-only sweep.

2026-07-23 window-scoped causal-evidence increment:

- Added SHA-bound detector and pose runners that decode only merged candidate
  windows, plus a raw-verified detection montage for offline model screening.
  A permissively licensed single-game YOLO11m checkpoint was rejected after
  its montage showed scoreboard, logo and head false positives.
- Reused the official E-BARD YOLOv8n checkpoint with a strict physical gate:
  ball confidence >=.03, rim confidence >=.10, at least two backward approach
  points, at least eight pixels of rise, normalized ball/rim distance <=.5,
  and ball/rim box IoU <=.8. The IoU guard rejects near-identical class
  confusion boxes.
- The fixed gate generalizes across both enrollment games: LAL has 25 TP/2 FP
  (precision .926, recall .424), ATL has 20 TP/1 FP (precision .952, recall
  .294). It is supporting positive evidence, not a complete attempt detector
  or an answer channel.
- Full score-window pose extraction produced 21,214 LAL and 7,911 ATL poses.
  The enriched bundles expose pose on 27/16 score candidates and ball/wrist
  exit slopes on 8/5 and 6/2 respectively.
- Added a fail-closed label rebind tool. It permits existing training labels to
  bind an enriched bundle only when raw SHA, event set, timing and semantics
  are identical and only evidence, reason or provenance changed.
- The best non-pose causal tree run remains non-runtime. Its per-video
  precision at recall >=.85 is .673/.769/.968/.540; the pose-fused variant
  reaches .660/.750/.861/.540. Sparse enrollment-only evidence creates a
  cross-broadcast coverage mismatch, so no checkpoint is promoted. Next work
  must generate the same 10 FPS ball and pose evidence for both benchmark
  candidate sets before another fusion decision.

2026-07-23 uniform four-game causal-feature correction:

- Generated the same E-BARD basketball observations at 10 FPS for both complete
  benchmark candidate sets and both enrollment probe sets. The two benchmark
  artifacts contain 6,025/8,078 decoded samples and 10,813/8,563 basketball
  detections; the two probe artifacts contain 1,081/1,998 samples and
  2,048/1,805 detections.
- The enrichment path now accepts a separately SHA-bound pose source bundle
  only after proving identical raw video, event IDs, event types and event
  geometry. Rebound labels are moved to enriched bundles only through the
  existing evidence-only identity invariant.
- Window-derived evidence is marked
  `agu_window_ball_rim_pose_v1`. The shot-validity v3 contract preserves the
  original 22 base features and adds 11 separate window features instead of
  allowing new detector rows to replace stronger pre-existing evidence. V1 and
  v2 model artifacts remain readable.
- A direct causal-rule classifier does not generalize from score windows to
  detector-preselected benchmark candidates. ATL produced 16 TP/18 FP
  (P=.471, R=.200); LAL produced 22 TP/24 FP (P=.478, R=.275). These hard
  negatives remain in training, and the physical rule is supporting evidence
  rather than a runtime filter.
- The new 518-row manifest SHA is
  `fd11b82c694369f9d37c16025dc96a3581e66e86ef7c7aaef356a312a1be6f6e`
  and remains raw-SHA disjoint from both reserved MEM-OKC games. The best small
  Extra Trees sweep is depth 3 / leaf 4. At recall >=.85 its four held-game
  precisions are .692/.816/.971/.562, a small improvement over the prior
  .673/.769/.968/.540 baseline but far below the joint P>=.95/R>=.85 gate.
  The artifact is training-only and not promoted.
- Downloaded the official E-BARD RF-DETR Nano best-total checkpoint as a
  possible independent high-precision detector. Its SHA-256 is
  `759968ecf8f83663c85de3d881713e072f4a9dd0ea2f136b082878b1e4fe2400`
  and size is 120,811,190 bytes. Added an optional Apache-2.0 RF-DETR 1.8.3
  adapter after dependency resolution proved it would not replace AGU's Torch,
  torchvision, NumPy, OpenCV or Pydantic versions. It remains outside the
  service/runtime extra.
- The 128-frame ATL montage confirmed mostly stable rim localization but
  basketball/rim duplication near close-up baskets at low confidence. A full
  1,081-sample ATL probe at detector confidence .25 emitted 609 basketball and
  673 rim rows. The fixed causal gate matched nine candidates at every tested
  .25/.40/.50/.70 evidence threshold, scoring 1 TP/8 FP. Those nine are a
  strict subset of the YOLO gate's twelve hits; RF-DETR adds no independent
  positive evidence on this probe and therefore is not expanded to four games
  or promoted.

2026-07-23 independent-VLM and normalized-feature screening:

- Screened the Apache-2.0 `granite3.2-vision:2b` Ollama model as an independent
  raw-frame reviewer rather than another Qwen-family vote. The local model
  digest is
  `3be41a661804ad72cd08269816c5a145f1df6479ad07e2b3a7e29dba575d2669`.
  The frozen eight-event bundle is unchanged and remains bound to source bundle
  SHA `52aa64348900da90d4e21e9c9c885db98c98f11bd1499f9cb03e394111bfdad9`.
- Twelve separate 640-pixel frames exceeded Granite's 16K context and failed
  closed with HTTP 400 for all eight events. A contact-sheet run returned seven
  parseable responses, and a six-separate-frame run returned eight, but neither
  produced the complete release-chain observables required by AGU. All events
  therefore remained `needs_review`; no semantic vote or runtime behavior was
  opened. Artifact SHAs are
  `91069c0a0d8535d68093b21964e0bcb01fb163d62724745ee2a83cc95b344b67`,
  `d3d0406b585c5b119eba0b2959d5fb2c04e7a1661ffd5995a352c66674f7ec41`
  and
  `31ca01e6f67ad15d578253c856ff5c6127f2a98c1fcf2a112a6205e4e66dcc3f`.
- Tested within-video percentile ranks and z-scores to remove broadcast-scale
  shift without using held-game labels. Extra Trees, Random Forest and
  logistic variants all remained below the promotion gate. The best normalized
  configuration's held-game precision/recall pairs were
  .621/.900, .811/.857, .953/.871 and .535/.850. Its worst precision (.535)
  is below the raw-feature sweep's .562, so no normalized feature contract or
  model artifact was added.
- Granite's conservative abstention is safe but not useful evidence, and
  within-game normalization does not explain the hardest broadcast. The next
  justified model work is explicit temporal ball/wrist/rim state estimation on
  hard negatives or a stronger independently licensed VLM under a separately
  budgeted local-model evaluation; neither result justifies relaxing gates.
- A follow-up offline probe derived 15 sequence/shape statistics from cached
  window observations: unique-frame persistence, duplicate factor, temporal
  span/run length, ball/rim IoU and area/aspect similarity, center-distance
  change, rise/direction consistency and motion roughness. These observations
  exist for only 148/518 examples (88 positive, 60 negative). Conditional on
  coverage, negatives do show more detector duplication and overlap
  (median maximum IoU .379 versus .070), but this signal is too sparse.
  Base-plus-temporal Extra Trees reaches the same worst held-game precision
  .562 at recall .850 as the raw feature baseline, so the proposed v4 contract
  is rejected before implementation.
- Open-source point-tracker research rejected CoTracker because the official
  repository licenses most of the project CC-BY-NC. Google DeepMind TAPNet is
  Apache-2.0 and publishes official PyTorch Online BootsTAPIR weights, so its
  repository was inspected at revision
  `c2cbab81cc06092b5f05bfe2da7bfec54e2079c9`. The 218,887,028-byte checkpoint
  was downloaded from the official `google/tapnet` mirror and sealed at SHA-256
  `87c1e752cf5ce56e3e2f7da460aeb4d40fc826d04ef2939bade86a5c7495377f`.
- Only the PyTorch inference requirements missing from AGU (`einshape` and
  `dm-tree`, plus their small transitive dependencies) were installed for an
  isolated screen; the package's broad JAX/notebook dependency set was not
  added to AGU or `pyproject.toml`. MPS inference loads but deterministically
  emits NaN tracks. CPU inference is valid but slow.
- A frozen 12-event ATL probe seeded five points around each E-BARD basketball
  box and tracked seven frames backward plus seven forward. All 12 outputs are
  finite on CPU, but the run takes 417.862 seconds. At full recall over the two
  positives, the best single trajectory statistic reaches only 2 TP/1 FP
  (precision .667); displacement reaches 2 TP/9 FP. The artifact SHA is
  `9d747911546a70a825d284c5ed5d4e9a8c29d118583073af6cf9bd481fc1c413`.
  Point tracking faithfully preserves some false detector seeds and therefore
  is not expanded or integrated.

2026-07-24 E-BARD seed-verifier screening:

- Generated 3,675 detector-distribution crop candidates from all 1,800 E-BARD
  images: 1,348 matched basketball seeds and 2,327 non-ball hard negatives
  across 60 game IDs. The manifest is training-only, hash-bound to the source
  archive and detector checkpoint, and contains no duplicate image hashes.
- The archive's nominal train/validation/test partitions share nearly all game
  IDs, so model evaluation uses five-fold `GroupKFold` by game. A frozen
  torchvision MobileNetV3-small screen tested 1x, 2x and 3x context without
  fine-tuning. The best 1x visual-plus-confidence variant reaches mean AP .936,
  but its worst held-game precision/recall is .826/.834 when thresholds are
  fitted for 95% training recall. Threshold transport is not reliable enough
  for promotion.
- Codex performed an allowed offline crop-only review of the frozen 12-event
  BootsTAPIR seeds without deciding shot presence or statistics. Nine boxes
  contain a basketball and three do not. The crop review is separately sealed
  from event truth.
- A visual-only model calibrated on pooled game-disjoint E-BARD out-of-fold
  predictions reaches OOF P=.573/R=.950. On the 12 reviewed ATL seeds it
  reaches P=1.000/R=.778, rejecting two genuine basketball seeds. As a
  downstream diagnostic only, candidate-shot precision rises from .167 to
  .286 at full two-shot recall because five false shot events still contain a
  real basketball.
- The verifier therefore cannot decide a shot, cannot meet the 85% target, and
  is not integrated into source or runtime. The next blocker is causal
  ball/hand/rim state and release-chain evidence, not crop class identity.

2026-07-24 WASB temporal-ball association screening:

- Audited the current AGU ball path. Both `BallTracker` and candidate
  trajectory linking are greedy center-distance/constant-velocity procedures;
  they have no appearance, global path, scale or scene-cut evidence. The
  frozen ATL artifact contains 2,048 detections and 472 fragmented tracks;
  516/1,081 sampled frames have multiple ball candidates.
- Built a 12-event, 90-frame, 257-candidate offline review package. Codex
  reviewed basketball visibility/position only; it did not label shots,
  outcomes, actors or statistics, and every artifact is
  `runtime_consumable=false`.
- Screened official temporal sources. APIDIS was rejected for non-commercial
  terms; VRU_Basketball has reusable video but no ball coordinates; BASKET is
  gated highlight/skill data; TrackNetV3/TOTNet public weights are
  racket-sport-specific. The MIT-licensed official WASB-SBDT basketball
  checkpoint was the only directly runnable basketball temporal baseline.
- Downloaded only `wasb_basketball_best.pth.tar` (6,102,633 bytes), pinned
  repository revision `923462cacdeb3353b84ddebdedb3f4b7a8553b0f`, and sealed
  SHA-256
  `8d1ba9870d0a6ab37b06ab82bed593c6c09133e713810bac475d0c000bb7e948`.
  A standalone MPS probe used centered contiguous triplets at 512x288 without
  modifying AGU source or preprocessing.
- On one independently annotated eight-frame release/post-rim event, the
  official 0.5 threshold recovered 0/8 balls. The raw local-peak distribution
  was useful: Top-1/Top-2/Top-10 ball-box coverage was 5/8, 7/8 and 8/8.
  BODD candidate coverage on the same frames was 6/8, 7/8 and 8/8.
- A truth-blind frozen rule reset on HSV histogram correlation below .8 and
  then selected the highest-probability peak within 10 px/frame. It selected
  7/8 annotated balls (P=R=.875). Truth was applied only after the path was
  frozen.
- This passes only the single-event 85% diagnostic, not the expansion or
  complete-statistics gate. One eight-frame event cannot justify integration,
  and the official threshold also produced high-confidence human-body false
  detections elsewhere. No source, dependency, schema, API or runtime model was
  changed. Next work is broader target-domain ball-box annotation and the same
  frozen association evaluation across the remaining panel events.

2026-07-24 object-centered release-chain and pose-coverage screening:

- Built a truth-free 12-event panel set from raw frames plus AGU's ball, rim,
  player-box and wrist proposals. The evidence package is offline-only and does
  not expose acceptance statistics to either VLM.
- A four-event risk screen rejected both local VLMs before a full 12-event run.
  Gemma 4 12B returned the same all-positive observable payload for all four
  panels (P=.500/R=1.000), while Qwen3-VL 8B abstained on the complete release
  chain for all four (P=0/R=0). Object crops did not solve cross-frame
  correspondence, so neither model was promoted.
- On the sealed 518-row four-game manifest, an adaptive contact anchor derived
  11 wrist/ball/rim sequence features. Only 106/518 rows had usable sequences.
  The best depth-3/leaf-4 fused model reached held-game precision
  .686/.816/.971/.544 at recall >=.85, below the corresponding raw baseline
  .692/.816/.971/.557. The proposed feature path was rejected before source
  implementation.
- Pinned `rtmlib` revision
  `2f36e3b62d73ed27a0b492a0d0fee1bba5fe3ace` and downloaded the official
  RTMPose-m body7+Halpe26 256x192 ONNX archive. The Apache-2.0 source, archive
  and extracted ONNX are SHA-sealed; OpenCV DNN CPU reused existing player
  boxes, so no detector or project dependency was added.
- On the frozen 12 events, RTMPose increased valid double-wrist observations
  from 220/397 to 280/397, but events with at least three wrist+ball frames
  improved only from 10/12 to 11/12. Some already clear events lost points and
  the hardest event retained only one frame. This coverage delta does not
  justify four-game extraction or runtime integration.
- BasketHAR was rejected because its Apache-2.0 release contains inertial
  arrays rather than synchronized video. DeepSport and SportsHHI were rejected
  for non-commercial license restrictions. No source, schema, API, runtime
  model, dependency or acceptance gate changed.
- A final 16 GB-local independent VLM screen selected the Apache-2.0,
  video-trained SmolVLM2-2.2B rather than an infeasible 30B model. The
  4,493,651,795-byte MLX BF16 weight was pinned at revision
  `844516024a1c4400d34489b89ee067d794e432ed` and SHA-256
  `ed6c59250704f09f921dce1a25e0d4eff611b6c9c53e382a7eb04ce9113f2773`.
  Dependencies were isolated with `uv`; AGU's environment did not change.
- Current MLX-VLM video input failed before inference because 30 decoded images
  were paired with zero image tokens. The same model then received eight
  exact, chronologically ordered raw frames per frozen event through its
  supported multi-image path. On all four risk events it emitted a repeated
  per-frame `pass` structure with string values, wrong keys and no valid
  event-level tri-state object. Strict parsing therefore abstained on all four
  (P=0/R=0; two false negatives). SmolVLM2 was rejected without expansion.
- Molmo2-8B was not downloaded: its unquantized local weight does not fit the
  remaining disk/memory budget, and its official model card warns that some
  training sources are academic/non-commercial even though the checkpoint is
  Apache-2.0. The independent-VLM branch is now exhausted for this machine;
  the next useful work must improve continuous visual association rather than
  add another event-level reviewer.

2026-07-24 expanded ball-association gate:

- Reused the frozen E-BARD MobileNetV3-small embedding as an offline
  annotation-ordering aid over 257 BODD candidates. On the original annotated
  event it ranked the true containing candidate first on 7/8 frames, but one
  true candidate remained rank five. This score never became a runtime answer
  or shot label.
- Codex reviewed exact raw frames plus local detail crops and froze ball-only
  annotations for five events: 34 reviewed frames, 30 visible-ball boxes and
  four explicitly occluded/not-visible frames. The annotation artifact
  prohibits shot, make/miss, identity and statistic targets and records
  `runtime_consumable=false`.
- The evaluator was SHA-frozen before the v2 annotations were written. Across
  the five events, official-threshold WASB recall is .300. Raw WASB local-peak
  coverage is .733/.900/.933/1.000 at Top-1/2/5/10; BODD candidate coverage is
  .800/.867/.900 at Top-1/2/5; offline visual-ranked coverage is
  .833/.867/.900.
- Candidate recall does not solve association. The frozen 12 px/frame WASB
  greedy path reaches .733, while the fixed
  `.6*relative-WASB + .3*offline-visual + .1*detector-confidence` fusion reaches
  .700. The low-sideline event is the dominant failure (.200), showing that an
  initially wrong body peak poisons later greedy state.
- The expansion gate requires at least four events, 30 visible frames and .85
  fusion recall. Sample-size requirements pass but fusion recall does not, so
  `decision=do_not_integrate`. No AGU source, dependency, schema, API or
  runtime model changed.
- The next justified model direction is target-domain heatmap fine-tuning with
  these ball-only labels plus a global/beam path objective that can recover
  from bad initial peaks. It must be validated game-disjointly and remain
  default-off until the same frozen multi-event gate passes.

2026-07-24 dense global-path development and blind validation:

- The official WASB online tracker assumes consecutive frames and only adds its
  acceleration prediction after three visible frames. The prior sparse panel
  spacing therefore could not represent official SBDT behavior. AGU ran the
  same centered-triplet heatmap on every frame in five development intervals
  (287 frames total) without changing the model or preprocessing.
- A bounded development search compared 540 label-scored global paths over
  scene threshold, Top-K, acceleration and speed costs. The selected rule uses
  Top-10 raw peaks, log probability, HSV histogram reset below .2 and a soft
  40 px/frame speed cap. It scores 29/30 (.967) on the development labels.
- The rule and two new event intervals were frozen before validation labels.
  Label-free dense inference covered 73 frames for event 00018 and 163 for
  event 00035; the resulting paths were hash-sealed before Codex reviewed
  ball-only crops. Ten unambiguous visible frames were retained and six
  occluded/not-visible frames excluded. The sealed selections hit 10/10.
- This passes the predeclared .85 gate for an experimental default-off adapter,
  not a full-game accuracy gate. Both validation events are from the same ATL
  broadcast, visible-frame count is small, and no shot/statistic correctness
  was scored.
- Added opt-in `GlobalBallPathSelector` beside the unchanged greedy
  `BallTracker`. It caps candidates per frame, supports explicit scene-reset
  frames, maximizes the frozen confidence/speed objective, and returns standard
  `BallTrackResponse` objects. No existing call site invokes it, so default
  service behavior is unchanged.
- Review replaced the initial full-path copies with DP backpointers to avoid
  quadratic path-memory growth. No dependency, environment variable, API,
  schema or v3 preprocessing contract changed.

2026-07-24 cross-game dense global-path validation:

- Used the disjoint 2008-06-08 LAL-BOS enrollment broadcast, with different
  teams, season and broadcast from ATL. The video SHA-256 is
  `1cb92b2d62c8ddf0c33f759f220cbbe8116a5d55521750f7c9437132c6148d9b`.
- In v1, three event intervals, dense WASB inference and every frozen path
  choice were sealed before Codex reviewed the 24 predeclared frames. The path
  hit 10/11 visible balls (.909), but the plan's 12-frame total and
  three-per-event floors failed (11 total; first event two). The result remains
  a recorded validation failure and was not repaired by adding frames.
- A distinct v2 supplemental plan excluded the v1 events, selected the
  highest-BODD-count remaining field-goal event in each temporal third, and
  predeclared all unique BODD hit frames. The plan, label-free dense inference
  and sealed selections have SHA-256 values
  `e92c7f0b6f2a3ee6a7b5b1d79a1aa1c3aa2c91c44d51d2e4375ff4d1f394385f`,
  `f9c6a88efeb48f3a28d64edfc9f73fce6e28fea236e0779f82b3ec0273dc7c6a`
  and
  `3351177e83400b4a931fb7a7096bd4f084f9f5c9c7cbcc18b0686cae14b70222`.
- Only after sealing did Codex label basketball visibility/location on the 22
  predeclared frames. Thirteen were unambiguous; the unchanged path hit 12/13
  (.923), with event recalls 4/5, 4/4 and 4/4. Both the sample and .85 accuracy
  gates pass.
- Annotation/result SHA-256 values are
  `bf2e4f188ab0a56b1ac59830779b64fb9f59bbeee806800a1963e5dc98c623d0`
  and
  `057f87965016f41c72a1036f16e2d8b38162116391ef1446c6e8b2da37c35aa2`.
  Every artifact is `runtime_consumable=false`; the result authorizes only a
  default-off downstream research adapter, not production activation or a
  complete-statistics claim.
- Added an explicit default-off bridge from `GlobalBallPathSelector` into
  `propose_shot_and_rebound_candidates`. The selector can now return the
  original chosen detections, allowing the candidate generator to use only the
  global path for ball/rim hits, trajectory outcome and actor proximity.
  Evidence records `ball_path_backend=agu_ball_global_path_v1`. No existing
  caller passes the selector, so default service behavior and public schemas
  remain unchanged.
- A non-blind downstream mechanism diagnostic then used the six sealed
  cross-game paths plus frozen BODD rim observations. Five windows produced
  FGA candidates. In v2, a foul sequence was incorrectly called a made FGA, a
  free-throw sequence was admitted as an FGA, and the true Fisher made three
  remained outcome `unknown`.
- Reference inspection preceded the prediction artifact, so this diagnostic is
  explicitly ineligible for promotion or an accuracy claim. It records 1 TP/2
  FP for FGA validity and zero resolved outcomes on the one true v2 FGA. Ball
  localization has not solved event validity or outcome.

2026-07-24 MEM-OKC component-score blind gate and VLM fusion:

- Downloaded and sealed the previously reserved 2011-05-11 MEM-OKC broadcast
  at SHA-256
  `1caf38b7b9c1997805cf5e66a37a05af974fdb5d99dfaa8816f328710523bbb8`.
  The 2011-05-13 game was removed from blind eligibility after event-level
  play-by-play was inadvertently exposed during search.
- Frozen BODD inference produced 313 shot and 308 rebound candidates. Before
  event truth, a temporal-thirds rule selected three high-hit shot windows and
  sealed the plan, selections, candidate probe and score predictions.
- Added team-component score consensus. Each team is reconciled independently,
  and the gate emits only one unique positive +1/+2/+3 component change while
  failing closed on negative, multi-team or greater-than-three changes.
- Added a candidate-local RapidOCR runner that binds its cache to raw video and
  candidate inputs and clears coarse trajectory outcome/value/team fields
  before attaching score evidence.
- The frozen gate emitted only `OKC made free_throw +1`. Post-freeze play-by-
  play confirmed Durant's first free throw; the other two selections were a
  live missed field goal and a replay/dead-ball duplicate. Selective precision
  is 1/1 at 1/3 selected-window coverage, but live-FGA outcome coverage is 0/1.
- Post-truth development exposed a fusion-order defect: VLM observables could
  identify the miss, but pre-resolved coarse labels took precedence. Prompt
  contract v18 adds observable free-throw and replay cues, and score-first
  neutralization lets Qwen3-VL correctly resolve the miss. Combined selective
  decisions are 2/2 at 2/3 coverage, while replay rejection still fails.
- The latter result is explicitly non-blind and authorizes no promotion.
  Actor/team identity, replay rejection, rebounds and secondary events remain
  blocked. Codex was used only for post-freeze offline review and never as a
  runtime answer source.

2026-07-24 replay-transition and recurring-logo rejection experiment:

- Evaluated the official MIT TransNetV2 implementation and an MIT PyTorch
  checkpoint port as raw-video scene-cut evidence. AGU loads the checkpoint
  with `weights_only=True`, verifies source/checkpoint hashes and keeps every
  artifact `runtime_consumable=false`. PySceneDetect was retained as a cut
  boundary reference only; SoccerNet replay grounding was not reused because
  its learned task is soccer-domain and its replay annotations are not a
  basketball runtime answer source.
- Added an offline full-game recurrence probe: TransNetV2 cut frames are
  screened with YOLOv8n person detections, then matched against a 2 FPS,
  48x27 full-game timeline outside a 30-second exclusion window. A repeated
  no-person full-screen graphic is recorded as a replay-logo proxy, never as
  live-action proof.
- The MEM post-truth three-window diagnostic appeared promising: only the known
  replay window triggered. The required four-video, game-disjoint development
  check rejected the proxy. Across 63 clear windows (38 live FGA, 25 replay or
  highlight), TP/FP/FN/TN was 4/3/21/35: precision 0.571 and recall 0.160.
  All three false positives were genuine live releases followed by ordinary
  broadcast transition graphics.
- The sealed evaluation has SHA-256
  `99edc2b64e0ea0dbb31536ad568db098362a197dfa3be2caf488c78cc04d81c0`
  and `promotion_eligible=false`. Neither the scene-cut rule nor recurring-logo
  proxy is connected to AGU runtime fusion. Replay rejection remains open.
- OpenCV 5.0 removed APIs still used by the identity/tracker compatibility
  path. Both OpenCV wheel families are now constrained to `<5`; the refreshed
  environment resolves 4.13 and restores `CascadeClassifier` and legacy
  `MultiTracker_create` without changing v3 preprocessing.
- Reserved a new two-benchmark public-research plan without opening benchmark
  event answers: 2007-04-25 GSW-DAL and 2009-05-08 LAL-HOU, with four
  same-series enrollment videos. Both benchmark rosters have 100% enrollment
  coverage, structured metadata is complete and every URL passes yt-dlp
  playback probing. The plan SHA-256 is
  `6616a16ec1d3aa53cb375789036f8820a1de2d8e9c8a545ca5fa3b63495b0ad3`.
- Actual media download remains incomplete. Both yt-dlp's native Python
  transport and external curl fail TLS negotiation to the resolved
  Googlevideo CDN, including a one-megabyte range request against the
  progressive format. The media gate remains false and no blind claim is made.
- The resumable downloader now bounds yt-dlp's inner retries so outer attempt
  limits work, and supports optional IPv4 and external-curl transport. This
  preserves the plan-bound state and SHA-sealing contract without silently
  hanging forever inside one attempt.

2026-07-24 broadcast-clock disappearance experiment:

- Added a raw-only RapidOCR clock parser that retains punctuation, requires a
  same-row period marker and rejects invalid game clocks. The evidence runner
  samples `-6/-3/0/+3/+6` seconds around SHA-bound candidate anchors.
- The frozen replay rule is intentionally one-way: at least two pre-anchor
  reads at or before -3 seconds must have exactly the same period and clock,
  and all three non-negative samples must be missing. Missing clocks alone and
  post-anchor frozen clocks remain unknown.
- On the four-game 63-window development subset, the rule reports
  TP/FP/FN/TN `2/0/23/38`, precision `1.000` and recall `0.080`. Both hits come
  from different LAL-BOS broadcasts; neither ATL-CHI game produces a hit.
- A fixed five-prediction support floor keeps
  `promotion_eligible=false` despite the observed precision. The evaluation
  artifact SHA-256 is
  `c382346d5f00b431532a367f03ad350af73e980452e1f4681408b99632f0385c`.
- A post-freeze MEM-OKC check abstains on two live windows and one replay
  window. The replay has only one recoverable pre-anchor clock, confirming
  sparse coverage without creating a false positive. Evidence SHA-256 is
  `78588fb0b509734b386f0b0c2feb190d02e5a40342449b619a5114de203dec67`.
- The module, runners and artifacts remain offline and default-disconnected.
  No public schema, v3 preprocessing behavior or VLM prompt changed.

2026-07-24 EBQwen independent raw-frame screen:

- Researched the official CC BY 4.0 BARD, E-BARD and
  EBQwen2.5-VL-3B releases and recorded exact revisions. MultiSports was
  rejected because the dataset is CC BY-NC 4.0; FineSports requires a signed
  release agreement and was not treated as an immediately open asset.
- Downloaded and hash-verified the EBQwen Q4_K_S GGUF and Q8 vision projector,
  then imported them as local Ollama model `agu-ebqwen25-vl-3b:q4ks`.
- Added a sealed independent-VLM plan/prediction/evaluation contract. The plan
  forbids labels, review notes, targets and ground truth; the runner sees only
  six raw frames per window and a neutral observable question. Cache identity
  includes every sampling and model-provenance input.
- The SHA-stable plan selected 32 windows, eight per raw video. Labels were
  hidden until all 32 predictions were sealed. Evaluation reports
  TP/FP/FN/TN `16/12/0/4`, precision `0.5714`, recall `1.0000` and
  `promotion_eligible=false`; per-video precision is only `0.5000` to
  `0.6667`.
- Rejected this path from runtime integration. The quantized Ollama runner uses
  multi-image input, not the official native-video temporal path, so the result
  does not reject a future properly frozen native-video experiment. Codex was
  used only for orchestration and post-freeze evaluation, never as a runtime
  answer source.
- A sixth bounded IPv4/external-curl attempt for the reserved GSW-DAL/LAL-HOU
  blind pair again failed before receiving bytes. The media gate remains false.

## 2026-07-29 dense cut-aligned clip increment

- Added a training-only plan/extractor that binds the existing 110 event rows
  to TransNet boundaries and samples 16 frames independently inside each
  available previous, anchor and next shot. Missing neighbors are explicit;
  replay semantics, corrected labels and Codex runtime answers are not inputs.
- Reused the existing hash-pinned torchvision MViT-v2-S checkpoint and
  produced four complete artifacts containing 33/9/63/5 events and
  80/20/147/9 available clips. No external weight was downloaded.
- Added a strict nested game-held screen over anchor, neighbor-summary and
  role-concatenated representations. The public screen entry point now rejects
  empty backbone or invalid dimension/frame provenance before fitting.
- The sealed result reaches pooled balanced accuracy/F1 `0.677/0.653` and
  weakest-game balanced accuracy `0.645`; it is rejected. A cut-MViT plus
  existing-DINO diagnostic is weaker at `0.648/0.661`, weakest `0.625`.
- Primary-source review rejects MultiSports (CC BY-NC 4.0) and FineSports
  (signed scientific-research-only agreement) for open/commercial training.
  SHOT7M2 is public and ECL-2.0 as declared by the Hub, but its 4.67 GB
  single-agent synthetic pose release does not supply broadcast-state
  supervision. All three are fail-closed catalog entries; no asset download
  was started.

2026-07-25 EBQwen native-video development screen:

- Completed and verified the official BF16 checkpoint, then converted it under
  the local resource guard with MLX-VLM 0.6.7 to affine 4-bit/group-size 64.
  The derived 3,073,721,056-byte safetensors contains 1,330 readable tensors.
- Added a default-disconnected native-video reviewer and CLI. It loads the
  model once, applies a frozen video-pixel budget, supplies raw temporal tensors
  with FPS metadata, caches only model decisions and fails closed to `unknown`.
- Added an explicit derived-plan-to-annotation-plan verifier so evaluation
  cannot silently open labels from a different frozen source plan.
- The sealed 32-window development run completed within the resource guard but
  predicted every window positive. Evaluation is `16/16/0/0`,
  precision `0.5000`, recall `1.0000`, and
  `promotion_eligible=false`. No runtime integration was added.

2026-07-25 staged full-game jersey-cache probe:

- Added `scripts/prefill_official_jersey_cache.py`, an offline-only adapter that
  reconstructs the exact saved identity-tracklet crops and pre-populates the
  existing raw-crop-bound jersey cache without rebuilding the full ReID graph.
- Selection is truth-free and ranks one best unregistered segment per raw
  source by candidate-event coverage times median crop height. Cache scopes and
  consensus partitions match `build_official_identity_artifact`, so successful
  reads would be reusable without importing Codex answers.
- Qwen3-VL 2B returned no parseable candidate on the first partition for each
  of six selected tracks (0/6 trusted reads). The run exited 0 with 2.217 GiB
  minimum available memory.
- Qwen3-VL 4B at 512 px/context 4096 also returned no candidate on one track.
  Its 768 px/context 8192 high-visibility diagnostic crossed the resource gate
  for three consecutive samples and stopped at exit 75 with 1.540 GiB minimum
  available memory. No runtime promotion or benchmark identity label resulted.
- Pre-development context was re-read from llm-wiki query
  `agu-complete-box-score-plan-2026-07-13`; the two-game complete-statistics
  gate and the prohibition on Codex runtime answers remain unchanged.

2026-07-25 conservative face-anchor revision:

- A truth-free detector/quality sweep showed that detector score 0.45,
  enrolled quality 0.55, gallery score 0.50 and margin 0.08 retain 14 direct
  tracklet anchors across 11 registered people with zero ambiguous identities.
  The old graph had 10 anchors across nine people.
- Exact roster evidence in the unchanged 390 candidate events rises from
  13 events/seven people to 22 events/ten people, and one rebound candidate
  gains registered primary actor `204`.
- The first score-0.50 graph exposed unsafe cluster propagation: two
  raw-light anchors mapped 980 anonymous tracklets to CHI and inflated any-team
  evidence to 358/390. A RED regression test captured this before changing the
  default team-binding support from two to four independent face anchors.
- The rebuilt min-anchors=4 graph retains all 11 registered people, has zero
  ambiguous identities and zero inferred uniform-team bindings. Its guarded
  build completed with 4.162 GiB minimum available memory and 2.846 GiB peak
  process-tree RSS.
- Frozen plan v5 records the new identity/candidate hashes before truth,
  inherits all v4 VLM behavior, and allows only exact prompt/image cache-key
  reuse. Freeze SHA is
  `14ff5375add24cfb4370a6cbe5e6b4f5db5c7cf5578b801e5da96c2634645f87`.

2026-07-25 resource-aware blind-inference continuation:

- The apparent actor-cache progress was corrected: the JSON key count includes
  historical candidate revisions and is not a current-run denominator. Exact
  current hits cover only the early part of the 390-candidate traversal.
- Added `CacheAwareFallbackOfficialEventReviewer`. It asks the primary reviewer
  for an exact input-bound cache hit without invoking Ollama; only a miss is
  sent to a separately keyed fallback reviewer. Sealed output records fallback
  model, image width, context length and dispatch count.
- A 4B 512/4096 fallback returned unavailable/HTTP 400 and was rejected before
  any prediction file existed. A bounded truth-blind 2B 512/8192 probe on the
  first miss returned a schema-valid actor/team decision at confidence 0.95,
  with 2.35 GiB minimum available RAM and 85.3% maximum system memory.
- Sustained 2B residency was then stopped when macOS swap reached 9.77/10 GiB.
  The 12 completed fallback results remain resumable. The final v9 execution
  unloads 2B after uncached reviews and retains exact 4B/2B cache separation.
- `TrainingResourceGuard` now samples swap and supports
  `min_free_swap_gib`; the resumable supervisor independently waits for both
  available RAM and free swap. v9 uses inner limits of 1.75 GiB RAM and
  0.75 GiB swap, plus restart floors of 5 GiB RAM and 1 GiB swap.
- An exact cache check proved candidate 104 has no primary 4B semantic result.
  A bounded 2B 512/8192 semantic probe returned a live, non-replay rebound
  decision at confidence 0.95 within the same RAM/swap limits. Frozen v10 now
  applies the exact-hit-first fallback independently to both semantic and
  actor passes; their fallback caches and dispatch counts remain separate.
- No benchmark truth or Codex runtime answer was opened throughout v6-v9.

2026-07-25 second-game source selection:

- NBA official metadata binds 1995 Finals HOU-ORL Games 1-4 to IDs
  `0049400070` through `0049400073` and identifies all as full games.
- Internet Archive item
  `nba-finals-1995-game-1-houston-rockets-vs.-orlando-magic-olajuwon-vs.-o-neal-the-0.03-tip-in`
  contains four original MP4 files totaling about 2.34 GB. Game 1 is reserved
  as the second benchmark and Games 2-4 as same-series enrollment.
- The truth-free source manifest canonical SHA is
  `1a404d0e954f4feff7773d9349f1a3af137d151666b4f0a476a5a98c442e31d6`.
  Download waits until the first blind inference releases memory and swap.

2026-07-26 first blind reveal and free-throw category correction:

- CHI-UTA frozen v11 completed all 390 candidates and was SHA-sealed before
  official NBA truth was opened. The preserved blind output contains 197 field
  goal attempts and 193 rebounds, but only one automatic confirmation.
- A fixed eight-event-type count-only diagnostic now complements the older
  prediction-scope view. The blind candidate upper bound is F1 `0.533854`
  (`TP<=205`, `FP>=185`, `FN>=173`); the automatic-confirmed upper bound is
  `0.005277`. These are optimistic diagnostics, not strict temporal/identity
  acceptance scores.
- The semantic pass already emitted a structured `free_throw_attempt=true`
  signal for some field-goal proposals, but the prior gate discarded the whole
  event. AGU now reclassifies only when three evidence layers agree: the
  structured subtype is true, the explanation explicitly mentions a free
  throw/foul line/lane slots, and live-scene fields do not also identify a
  replay, break, jump ball, dead ball or inbound.
- Reclassification changes the immutable revision's event type to
  `free_throw_attempt`, fixes `shot_value=1`, and remains `needs_review` unless
  the ordinary completeness gate is independently satisfied. Missed free
  throws now feed the same bounded post-release rebound context as missed field
  goals.
- An exact-cache, truth-revealed development replay recovered five FTA
  candidates, kept one dead-ball conflict and four explanation-inconsistent
  outputs rejected, expanded candidate scope from 2/8 to 3/8 event types, and
  raised the fixed complete-scope optimistic upper bound from `0.533854` to
  `0.546875`. It added no automatic confirmation and is ineligible as a blind
  result.
- The open-source boundary is unchanged: AGU owns the evidence consistency and
  ledger revision logic while Qwen3-VL remains an interchangeable local
  reviewer. No API, environment variable, v3 preprocessing contract, checkpoint
  or runtime Codex answer path changed.

## 2026-07-26 HOU-ORL enrollment and environment consolidation

- Kept G1/G2 sealed and used only G3/G4 plus public official sources for
  enrollment. Corrected the series roster to 20 and defined a hash-bound
  17-player G3/G4 observable scope.
- Merged six reviewed identities, including a separately verified Charles
  Jones crop set, into an 18-person gallery. Scoped coverage is 17/17 and the
  three metadata-only nonparticipants are excluded from the denominator.
- Consolidated all local AGU dependencies into `.venv` (Python 3.11), added
  `audio-mlx` and `research-download` optional groups, and removed the sibling
  `venv/`.
- Added a fail-closed harness check for the canonical environment and retained
  both EBQwen native and MLX 4-bit weight directories.

## 2026-07-27 BasketEvent release audit

- Located the upstream code at Git revision
  `8a313f3ad4476735ddac38543578e19c1bccebd5` and the Hub artifacts at revision
  `ca4af13e7754e0f2fe9653a604b5182080b6a62a`.
- The Hub release contains 32,162 trajectory JSON files and a
  1,813,273,995-byte PlayNet checkpoint, but no source videos, dataset card or
  license. The GitHub repository also declares no license.
- A deterministic 70-file schema sample loaded successfully. It exposed mixed
  event naming, duplicate player-name tracks in 36 clips and about 9.6% missing
  trajectory boxes. These annotations remain audit-only and were not imported.
- The upstream runtime targets Python 3.12, CUDA 12.6, a CUDA-only SAM3 stack,
  machine-specific checkpoint paths and a hard-coded `cuda:4` inference device.
  It is not compatible with AGU's sole Python 3.11 `.venv` contract.
- The paper's reported PlayNet macro F1 is 0.682, below the AGU acceptance
  objective. The 1.8 GB checkpoint was therefore not downloaded.
- Added a fail-closed public-research catalog entry that allows the paper to
  inform an independent design but forbids importing code, annotations or
  weights until compatible terms and source media are supplied.

## 2026-07-27 three-phase scene-state fusion

- Added a training-only full-frame state encoder using the already cached
  official torchvision MobileNetV3-Small ImageNet checkpoint. Each candidate
  is sampled deterministically at 15%, 50% and 85% of its event window.
- The artifact is bound to the canonical 511-row training manifest, all source
  videos, candidate bundles, annotations and the exact checkpoint SHA. It is
  explicitly `runtime_consumable=false`; labels and Codex review text are never
  supplied to the encoder.
- The screen derives phase mean, standard deviation, last-minus-first and all
  three ordered states, then applies StandardScaler, PCA-16 and balanced
  logistic regression inside each game-held training fold. PCA and scaling are
  never fitted on the held game.
- Four variants were predeclared: scene only, scene+MViT, scene+Swin and
  scene+MViT+Swin. No hyperparameter expansion was performed after seeing the
  scores.
- Scene+MViT is best at a single OOF threshold: pooled precision/recall/F1 is
  0.7064/0.8854/0.7858 and weakest-game precision/recall is 0.6422/0.8525.
  The three-backbone fusion is slightly worse at 0.6379/0.8500 on its weakest
  game.
- Both fail the fixed 0.85 precision and 0.85 recall requirement. No runtime
  model, threshold, event revision or blind prediction was produced. HOU-ORL
  G1/G2 remain sealed.

## 2026-07-27 hash-bound ball-release correction development

- Added training-only contracts and CLIs for balanced hard-example selection,
  label-free review-sheet rendering, hash-bound Codex decisions, conservative
  correction overlays and corrected embedding copies.
- Three disjoint 32-window batches reviewed release, rim-directed motion, rim
  arrival, free-throw formation and replay/stoppage state. The 96 rows produced
  43 label changes; uncertain rows remained unchanged.
- Pooled game-held F1 improved from 0.7858 to 0.8091, 0.8336 and 0.8363.
  The third pass reaches per-game F1 0.7871, 0.8908, 0.9016 and 0.7952, so it
  still fails the all-games 0.85 gate.
- Remaining errors are dominated by non-atomic scoreboard windows spanning
  15–35 seconds and multiple phases. Further same-policy label correction is
  rejected as the primary next step; candidate splitting/recentering is next.
- All three retraining screens ran under the unchanged CPU/memory guard with
  zero breaches. No runtime path changed and HOU-ORL G1/G2 stayed sealed.

## 2026-07-26 guarded secondary-event audio development

- Added `foul` to raw-audio action extraction, including explicit rejection of
  foul-line, foul-trouble, no-call, future-foul and retrospective contexts.
  Every speech event remains `needs_review`; speech names never become actors.
- Added a selectable unmatched-action scope so the experiment can add only
  assist/block/foul/steal/turnover without worsening existing shot and rebound
  overgeneration.
- Added deterministic registration-roster conversion and a sealed offline
  audio-review contract. Codex decisions are development labels with
  `runtime_consumable=false` and `codex_runtime_answer_used=false`.
- The first Base-English run was correctly stopped at exit 75 after three
  consecutive free-swap breaches below 0.5 GiB. After memory recovery, Tiny
  and Base-English completed under the unchanged guard.
- Base-English expands CHI-UTA development candidate scope from 3/8 to 8/8 and
  raises complete-scope count-only F1 upper bound from 0.546875 to 0.632212.
  This is not strict accuracy: assist/steal/turnover remain 1/36, 1/15 and
  1/28, and only two of four block mentions link to a shot candidate.
- Codex reviewed all 57 broad foul mentions: 36 live-current, 16 non-event and
  five uncertain. The review artifact SHA-256 is
  `07c6fc042321bf332d8aab506772a0eda491d8f5cd8da64c9061bf0ed968480e`.
  A same-game lexical rule was deliberately rejected after cross-game examples
  exposed last-foul, team-count and historical-replay false positives.

## 2026-07-26 one-to-one commentary shot evidence

- RED tests showed that common live `hit`, `miss`, `connect`, `bury`, and
  `off target` inflections were absent and that one broad speech window attached
  to every overlapping same-action visual candidate.
- The parser now accepts those result inflections while rejecting free throws,
  foul/technical contexts, missed calls, ordinal-only phrases, action metaphors,
  conditionals, and retrospective multiple-shot summaries.
- Each mention deterministically selects one same-action event by maximum
  temporal overlap, minimum midpoint distance, shorter window, higher
  confidence, then stable event ID.
- Speech shot outcomes are stored only in
  `details.speech_shot_outcome_candidate`; they do not mutate event outcome,
  status, actor, or official aggregation.
- CHI–UTA development v7 retains 454 events and increases audio-linked FGA from
  11 to 17 (11 missed / 6 made result candidates). Its fixed complete-scope
  count-only F1 upper bound remains 0.632212.
- An alternative that added every unmatched audio FGA/rebound produced 520
  events and worsened that optimistic upper bound to 0.585746. It was rejected,
  and about 91 MB of superseded intermediate artifacts were removed.
- Current source screening retained the existing MIT `choucsan/NBA_Games`
  clone, placed BasketEvent/PlayNet on a release watchlist, and rejected
  GameCommBench, BASKET, and `saveerjain/basketball-events` for license, scope,
  access, size, or schema reasons.
- A development-only 518-row probe derived 34 player-ball interaction and
  coarse global-layout features from the same sealed four-game bundles. Base
  interaction observations cover 370/518 rows and uniform-window observations
  only 146/518. Interaction-only Extra Trees reaches worst held-game precision
  0.557 at recall >=0.85; fusion is worse. This is below the existing roughly
  0.562 baseline and was rejected before any runtime feature contract.
- The MIT TOTNet repository was pinned at revision
  `8a757f63391b262c14d18b4095486336852dbeef` and its 94 MB tennis checkpoint
  was screened as a possible occlusion-aware ball tracker. MPS fails on an
  unsupported `adaptive_max_pool3d`; one CPU window takes 17.88 seconds and a
  three-window batch remained incomplete after four minutes while using about
  2.2 GB RSS. The guarded process was terminated, and the downloaded source,
  checkpoint, and temporary `einops` dependency were removed.

## 2026-07-27 PBP-bound visual-state supervision

- Added a training-only PBP-to-video manifest with exact source hashes, strict
  NBA clock parsing, closest-to-anchor OCR selection and a maximum ±1-second
  match. Same-clock free-throw plus field-goal actions are excluded as
  ambiguous.
- Every one of the 374 development clock candidates is either a determinate
  example or has a fail-closed exclusion reason. The 137 examples contain only
  action IDs/types needed for provenance and omit players, descriptions and
  scores.
- The CLI requires the sealed HOU-ORL acquisition manifest and rejects both
  blind benchmark video hashes before labels can be built.
- Existing held-game scene/video/clock features do not generalize sufficiently:
  three-state macro F1 is at most 0.619 and free-throw-vs-other balanced
  accuracy is 0.710. A bounded EBQwen state probe also fails, so no runtime
  model or blind-game path changes.
- The official MIT WASB-SBDT source is retained at revision
  `923462cacdeb3353b84ddebdedb3f4b7a8553b0f`; its CUDA-only training runner
  is not adopted into AGU's Mac-local path.

## 2026-07-27 Qwen3-VL 2B compact independent-review screen

- Reused the already local Apache-2.0 `qwen3-vl:2b` Ollama weight rather than
  downloading another model. The official Hugging Face revision is
  `89644892e4d85e24eaac8bacfd4f463576704203`; the local quantized blob SHA-256
  is `ebabfa59b71a5b96e0281ec2994977e785284e0939807a99fc340dec3c6f10de`.
- The original frozen 32-example, 6-frame, 704-pixel plan was not modified.
  Instead, a label-free derived plan preserves the exact example set and binds
  its source-plan hash while declaring a 4-frame, 512-pixel sampling contract.
  The derived plan SHA-256 is
  `3ed7bd41be0fa9ec28748fc219a2496e16ed23a337bfd725f0f32854a7136982`.
- Context 8192 exceeded the guarded memory budget, context 2048 could not hold
  the visual prompt, and context 4096 fell below 2 GiB available memory.
  Context 3072 completed through resumable cache batches while retaining the
  88% system-memory and 2 GiB available-memory hard stops.
- Predictions were sealed before training annotations were read. The prediction
  artifact SHA-256 is
  `5918b69ff6868e6aecd43daa41a650eb07ba247a5084b9fa5ee9c38ef4b8ef15`.
  Post-seal evaluation produced TP=5, FP=6, FN=11 and TN=10: precision 0.455,
  recall 0.312 and no unknowns. Per-game recall was 0.50, 0.50, 0.00 and 0.25.
- The model is therefore rejected for independent shot-validity fusion. No
  runtime path, threshold or blind-game prediction changed; HOU-ORL G1/G2
  remain sealed. The result reinforces continuous formation/release evidence
  as the next direction rather than another event-level VLM.

## 2026-07-27 continuous reason-evidence screen

- Added a label-free 39-feature evidence contract over the existing 511
  development windows. It combines pose formation geometry and stability,
  candidate-local wrist/ball separation, and ball/rim temporal deltas without
  importing review labels into the evidence artifact.
- The reason classifiers use only 95 determinate rows from the three already
  sealed review batches. Free-throw formation, replay/stoppage and causal
  release are scored with nested game isolation: an outer-held game is absent
  from every fit, and each outer-training game's score is produced by a model
  trained on the other two outer-training games.
- Four variants were predeclared with unchanged PCA-16, logistic C=0.01 and
  threshold selection: base; base plus raw continuous evidence; base plus
  nested reason scores; and base plus both.
- Raw continuous evidence reaches pooled precision/recall/F1
  0.7655/0.9144/0.8333 and weakest-game precision 0.7191. This does not beat
  the 0.7705/0.9144/0.8363 base and remains below the all-game gate.
- Replay/stoppage auxiliary F1 improves from 0.7143 to 0.7475, but free-throw
  and causal-release F1 are only 0.2703 and 0.1600. The experiment is retained
  for diagnosis and rejected for runtime promotion. No blind media, frames,
  truth, prediction, threshold or runtime behavior changed.

## 2026-07-28 EBQwen adaptation and temporal-state development screen

- Added a training-only EBQwen visual-state data contract and builder. It
  verifies the existing review-plan, sheet and correction hashes, requires
  exact row coverage, rejects sealed blind video hashes, crops individual
  five-frame rows and binds every resulting JPEG by SHA-256.
- The resulting 81-example manifest is balanced at 41 free throws and 40 live
  field goals across four source videos. Twenty-nine unresolved review rows
  remain excluded. The manifest SHA-256 is
  `5a00875f6611558cc9309623ff84be99b03538d7331107e915da6d227f25f1e8`.
- The exact installed `mlx-vlm==0.6.7` supports multi-image `VisionDataset`
  rows and lower-level LoRA training. Its convenience CLI cannot run without
  the optional `datasets` training extra, so the feasibility check used the
  already-installed lower-level trainer rather than changing the environment.
- A bounded rank-4, one-example, one-step EBQwen MLX 4-bit probe was stopped
  by the resource guard before any adapter existed. System memory reached
  92.5%, available memory fell to about 1.2 GiB and free swap reached zero.
  Full-layer 3B LoRA is therefore not safe on this 16 GB host. The thresholds
  were not relaxed.
- Added a lightweight alternative using the official cached
  MobileNetV3-small ImageNet checkpoint. Five panels share one encoder; mean,
  standard deviation and last-minus-first embeddings feed a binary head.
  Training uses four fixed leave-one-source-video-out folds and a fixed 0.5
  threshold.
- The 12-epoch last-block screen reached pooled balanced
  accuracy/precision/recall/F1/AUC
  `0.7168/0.7500/0.6585/0.7013/0.8073`, but per-source balanced accuracy was
  `0.6291/0.7500/1.0000/0.7175`. The weakest source fails the 0.85 acceptance
  gate, so the model is rejected.
- All eight rejected fold checkpoints from the smoke and final runs were
  deleted; screen JSONs and resource logs remain reproducible evidence.
  EBQwen native and MLX weights remain because they still support independent
  inference. No runtime threshold, service behavior or blind prediction
  changed.

## 2026-07-29 expanded visual-state transfer screen

- Added a parent-hash-bound, label-hidden wide review plan for only the 29
  unresolved first-pass rows. Offline review resolved 27 and retained two as
  unknown; the combined target supervision is 108 rows with class counts
  55/53.
- Added `app/analysis/visual_state_domain_screen.py` and
  `scripts/screen_visual_state_domain_transfer.py`. The boundary verifies both
  correction chains, inherited blind hashes, exact event joins, identical
  source/target backbone identity and four target groups.
- Hyperparameters are selected with leave-one-game-out validation inside each
  outer training split. The outer-held game is excluded from every fit and
  every selection metric.
- Same-weight Basketball-51 assistance is rejected at pooled balanced
  accuracy/F1 `0.695/0.692` and weakest-game balanced accuracy `0.629`.
  Existing 39-feature reason evidence and a seven-timepoint pose-layout
  diagnostic are also rejected at weakest-game `0.575` and `0.333`.
- No deployable checkpoint was created. The next representation must preserve
  cross-broadcast court topology and continuous dead-ball/free-throw state
  rather than retuning frozen appearance or aggregate pose features.

## 2026-07-29 court-topology candidate screen

- Audited a 48-keypoint YOLO11n basketball-court model at immutable revision
  `6cb899251439982067a81baa225b33f45f335981` and verified the downloaded
  weight SHA-256
  `68e5faf5fb5388cf83477238abe22e9824a69fb33a4cf34f79ab61858f410064`.
- The release lacks semantic keypoint/world-coordinate mapping, has an
  AGPL-3.0 model license, conflicting MIT/CC BY 4.0 dataset metadata and
  unverified underlying NBA broadcast rights. It was constrained to a
  disposable offline research probe.
- The 108-row nested held-game screen reached pooled balanced accuracy/F1
  `0.602/0.598` and weakest-game balanced accuracy `0.562`; only 57 rows met
  the conservative court-evidence coverage gate.
- High-confidence closeup false positives invalidate detection confidence as
  a court-presence gate. No runtime topology feature was added. The downloaded
  weight and temporary overlay montage were deleted.
- KaliCalib was also denied because its trained basketball calibration assets
  derive from noncommercial/no-derivatives DeepSport data.

## 2026-07-29 DINOv2 temporal-context screen

- Added a fixed centered-wide `-8..+8s` anchor protocol and a training-only
  DINOv2 temporal screen with exact correction/event joins and nested
  target-game isolation.
- Existing pre/anchor/post sequences reach pooled balanced accuracy/F1
  `0.711/0.744`, but the 62-row LAL–BOS development game remains at `0.645`.
- Label-hidden review of its 22 errors found injury closeups, graphics, bench
  cuts and post-score transitions at the event anchor. The matching centered
  nine-frame representation was therefore extracted rather than changing the
  backbone or labels.
- Centered-wide selection worsens pooled balanced accuracy/F1 to
  `0.674/0.715`; the weakest outer fold chooses the old narrow representation.
  An exploratory DINO+MViT+reason fusion also leaves the weakest game at
  `0.645`.
- No model, threshold or runtime path was promoted. The temporary model weight
  and inspection images were deleted; sealed embeddings and screens remain.

## 2026-07-29 event-centered cut-aware state screen

- Extended the existing TransNetV2 runner with a hash-bound, label-hidden
  event-plan mode. It rejects sealed blind videos and any artifact declaring a
  Codex runtime answer, and joins only exact video/candidate/event identities.
- Generated boundary geometry for all 110 planned development events, up from
  11/108 resolved events previously covered. Transition semantics and
  `broadcast_state` are excluded from the feature boundary.
- Added `app/analysis/visual_state_scene_screen.py` and
  `scripts/screen_visual_state_scene_segments.py`. They build wide, anchor-shot
  and neighboring-shot DINOv2 summaries under strict outer and inner
  leave-one-game-out selection.
- The screen reaches pooled balanced accuracy/F1/AUC
  `0.684/0.702/0.720`; held-game balanced accuracies are
  `0.750/0.750/1.000/0.597`. It is rejected below the 0.85 gate and below the
  previous 0.645 weakest-game temporal result.
- No classifier checkpoint, runtime threshold or service behavior changed.
  Reusable boundary assets remain; no new model weight was downloaded.
  HOU–ORL G1/G2 remain sealed.

## 2026-07-30 whole-game-isolated ball detector follow-up

- Replaced E-BARD's game-leaking frame split with a deterministic 44/8/8
  whole-game split and combined it with cleaned MUVY using hard links.
- Retained 335 genuine no-ball frames as hard negatives. The sealed combined
  set contains 1,969 frames, 1,673 ball boxes and has artifact SHA-256
  `894dc1c4f5d23314680eb00daf2581a3cf9839bc2c16c186bd6a8fa296140f17`.
- A guarded COCO YOLOv8n run stopped at epoch 30; best epoch 21 reached
  `0.845/0.586/0.682/0.326` P/R/mAP50/mAP50-95 on validation and
  `0.801/0.729/0.771/0.375` on eight whole-game-held test games.
- The MUVY event-held result improved over BODD in mAP, but the first frozen
  development-video gate produced zero detections across all 405 LAL-BOS
  enrollment samples. Worst-game recall is therefore zero, so no geometry
  rebuild or remaining-game inference was justified.
- No runtime path was changed. Four rejected checkpoints were moved to Trash;
  compact evidence remains and HOU–ORL G1/G2 were not accessed.

## 2026-07-30 offline Transformers RF-DETR candidate screen

- Added a local-only `TransformersRFDETRDetectorAdapter` and an explicit
  `transformers_rfdetr` factory/script backend. Defaults remain unchanged and
  no hosted download or runtime integration is permitted.
- The pinned 121 MB Nano weight restores candidate output on the allowed
  LAL–BOS enrollment slice: 900 boxes and 173 tracks over 405 frames, versus
  zero for the prior whole-game E-BARD+MUVY detector.
- A sealed, six-band 72-candidate review finds 25 valid balls, 45 false
  positives and two uncertain. The best reviewed weighted threshold precision
  is only 0.667, so the head is rejected for direct evidence use.
- Keep the model only as a candidate proposer. The next justified experiment
  is a lightweight hard-negative verifier trained from the existing 3,675-row
  E-BARD candidate manifest, with game/window-disjoint evaluation.
- HOU–ORL G1/G2 were not accessed.

## 2026-07-30 E-BARD candidate-verifier transfer

- Added leakage-safe recall-threshold and stratified-review metrics plus an
  offline MobileNetV3-small verifier screen.
- Training uses only 3,675 E-BARD candidates grouped by 60 game IDs. The
  sealed 72-row LAL–BOS review is joined only after OOF threshold selection.
- Visual+confidence reaches E-BARD OOF P/R `0.874/0.851` but external
  P/R only `0.559..0.615/0.614..0.636`. Visual-only external transfer is
  `0.576..0.657/0.441..0.474`.
- Both routes are rejected without checkpoints. The next viable route needs
  same-RF-DETR hard negatives from a disjoint development game.

## 2026-07-30 same-detector ATL–CHI verifier screen

- Produced 2,014 RF-DETR candidates over 729 samples in nine ATL–CHI
  enrollment windows with the exact LAL–BOS detector protocol.
- Built a deterministic 108-row, six-band offline review. Codex labeled only
  visible-ball box validity: 29 valid, 68 false and 11 uncertain.
- Trained on 97 determinate ATL rows grouped by causal window and evaluated
  only after threshold freezing on the sealed LAL–BOS review.
- Exact crop plus confidence reaches only `0.646..0.676/0.677..0.688`
  external P/R. 2× context and geometry diagnostics collapse to all-positive.
- No checkpoint is retained. Broader same-detector multi-window supervision
  is required before another verifier or detector-head training attempt.

## 2026-07-30 multi-source same-detector verifier

- Ran the frozen RF-DETR protocol over the independent opened ATL–CHI
  benchmark and generated 4,424 candidates over 2,610 frames in 28 causal
  windows.
- Built and sealed a second 108-row six-band box-only review: 42 valid balls,
  52 false positives and 14 uncertain. Codex did not label event outcome,
  player identity or statistics.
- Extended `_broadcast_features` with exact detection-ID filtering so source
  training decodes only reviewed determinate candidates rather than all 6,438
  source candidates.
- Extended the verifier CLI to accept repeated disjoint source bundles,
  namespace causal-window groups by video hash, reject duplicate or
  source/target-equal videos, and aggregate 191 determinate rows across two
  games.
- Added detector/track-only features for box shape, same-frame density,
  visible/total point counts, span, confidence, velocity variation and net
  displacement. Exact visible track points are bound back to detections
  without opening additional frames.
- The four frozen variants all fail the sealed LAL–BOS external gate. No
  checkpoint, threshold or runtime adapter is promoted.

## 2026-07-30 independent EBQwen candidate-panel probe

- Added `app/analysis/ball_candidate_vlm.py` with label-free stratified probe
  selection, strict response parsing and population-weighted uncertainty
  bounds.
- Added a local MLX runner that renders full-frame red-box panels with enlarged
  insets, verifies the raw video and model weight hashes, applies a bounded
  image pixel budget and writes a per-candidate resumable cache.
- Added a separate evaluator so prediction cannot load review decisions.
  Predictions and evaluations remain `runtime_consumable=false` and
  `codex_runtime_answer_used=false`.
- Both the initial and single revised development prompts produce 12/12
  `uncertain` states. The route is rejected before LAL–BOS, runtime or blind
  execution.

## 2026-07-30 frozen RF-DETR query-verifier screen

- Audited the installed Transformers 5.14.1 RF-DETR output and official
  object-detection contract before choosing the adaptation boundary. Full
  detector training was rejected because the available reviews label sampled
  candidates, not every object in every frame.
- Added `app/analysis/rfdetr_query_verifier.py` to reproduce RF-DETR
  post-processing and fail closed unless the exported object index,
  confidence and pixel box reconstruct from one original query.
- Added a query-vector contract containing the final 256-dimensional decoder
  state, three raw class logits and normalized `cxcywh`.
- Added `scripts/screen_rfdetr_query_verifier.py` with exact raw-video and
  model-weight hashing, optional hash-keyed feature caches, repeated disjoint
  source bundles, MPS batching, causal-window OOF fitting and delayed target
  review loading.
- The 191 determinate ATL rows train four frozen variants. All fail the
  independent LAL–BOS lower-bound 0.85/0.85 gate; no checkpoint or runtime
  adapter is promoted.

## 2026-07-30 third-domain RF-DETR supervision

- Extended `run_official_perception_windows.py` with the already tested
  local-only Transformers RF-DETR adapter and directory-weight provenance.
- Fixed the generic perception-window contract to seal its canonical artifact
  SHA, so downstream review fails closed instead of accepting an unbound
  detector output.
- Ran the opened CHI–UTA candidate windows at 10 FPS and generated 13,108
  proposals over 8,004 sampled frames.
- Codex reviewed 108 deterministic six-band panels under the existing
  box-validity-only boundary, yielding 106 determinate rows.
- Three-source fitting now has 297 rows, but LAL–BOS precision remains 0.660.
  The single-frame linear-query route is closed; next work must aggregate
  decoder state across candidate tracks and reserve a fourth game externally.

## 2026-07-30 multi-frame RF-DETR query and second LAL–BOS game screen

- Added exact reviewed-detection expansion to visible same-track members and
  a fail-closed 36-dimensional temporal query contract. It captures track
  availability/span, current-to-mean and adjacent query similarity, and
  aggregate logits/box statistics without loading event or statistic labels.
- Added temporal-only, context+temporal and full-query+temporal variants to
  the existing source-OOF/late-target-review screen.
- On LAL–BOS Game 2, none improves over the non-temporal
  `query_context_logits_box` result of `0.660/0.937`; the best temporal result
  is `0.637/0.902`.
- Generated a separate RF-DETR artifact for opened LAL–BOS Game 1:
  5,016 samples, 10,566 proposals and 2,441 tracks in 58 merged windows.
  Codex performed box-validity-only review and sealed 20 valid, 86 false and
  two uncertain decisions.
- With all three-source fits and scores frozen before opening the Game 1
  review, the best lower-bound P/R is `0.372/0.629`. This is an independent
  game holdout, not an independent broadcast-domain claim.
- A development-only reverse diagnostic adds Game 1's 106 determinate rows
  and evaluates on already opened Game 2. Best P/R remains `0.665/0.937`;
  explicit same-series hard negatives do not fix the precision ceiling.
- All variants are rejected, no checkpoint is saved and runtime is unchanged.
  Further frozen query-statistic tuning is stopped in favor of exhaustive
  detector supervision or rim/court-constrained trajectory state.

## 2026-07-30 WASB reproducibility and entity-relation state screen

- Re-audited the existing dense WASB evidence before running a duplicate
  annotation pass. The frozen cross-game path already reaches 12/13 visible
  ball centers, while its downstream diagnostic confuses a foul and a free
  throw with field-goal attempts and leaves the genuine FGA outcome unknown.
- Added a pinned offline WASB adapter that reproduces the official three-frame
  affine/ImageNet/heatmap contract without importing the CUDA-only detector
  wrapper. Safe strict checkpoint loading and a real guarded MPS forward pass
  produce a finite `[1,3,288,512]` output. It is not a runtime backend.
- The source rescreen did not download BasketEvent/PlayNet. Its public code
  and Hub release still declare no license and provide no source video; the
  1.81 GB checkpoint therefore remains ineligible. FineSports, MultiSports
  and KaliCalib remain excluded under their recorded access/license bounds.
- Implemented a training-only, permutation- and horizontal-mirror-invariant
  entity relation representation over each player, the ball and rim. Player
  height normalizes camera scale; pair, team, rim, lane, ball-possession and
  ball-rim relations remain ordered across the existing 24 causal positions.
- The sealed 110-example relation artifact has 51 features per position and
  SHA-256
  `32733ec21d35a00ad0d19c04bd1dd559696e94b817dd69fbf67b6ad50a9b0feb`.
  It rejects blind-source overlap and joins the 108 determinate visual-state
  rows only by exact video/bundle/event/review identity.
- Under the unchanged nested game-held protocol, MobileNet frame embeddings
  plus entity relations reach balanced accuracy/F1 `0.730/0.752` and worst
  game `0.656`. Screen SHA-256 is
  `876619506d009d2ee5ca235114d362c0d49179313f897e6858aa3c34d9998fc7`;
  `accepted=false`.
- A fixed, non-promotional 50/50 OOF probability diagnostic reaches
  `0.758/0.772` pooled but only `0.688` on the weakest game. This confirms
  complementary signal without cross-broadcast reliability.
- Build/screen resource guards record zero stops, peak system memory
  `75.7%/74.4%`, peak CPU `33.3%/25.6%`, and maximum process-tree RSS
  `740,835,328` bytes. No checkpoint, runtime, API or blind-game path changed.
- Stop hand-engineering additional relations on the same 108 labels. The next
  entity-token/attention model requires materially broader, compatibly
  licensed event supervision or a future license-complete BasketEvent release.

## 2026-07-30 BARD embedded-validation event-state transfer

- Fixed the official BARD repository at revision
  `add4109bf8b2034a32f3b6a83fa0d8c0ae638473` and imported its actual embedded
  2024/2025 validation MP4 blobs rather than the dynamic NBA event links.
- Added strict benchmark parsing, game-balanced subset planning, exact Git
  blob/size/SHA verification, duplicate-media rejection, normalized frame
  sampling and source-only grouped transfer screening.
- Materialized 38 unique videos over 33 games, balanced 19/19 between live
  field goals and free throws. The bounded source is 252,967,937 bytes.
- The initial 48-URL dynamic route returned one identical 31,580,089-byte
  payload for every event. It failed the new duplicate guard and its invalid
  1.41 GiB directory was deleted.
- Source-only five-fold OOF selected quarter pooling and `C=0.01` at balanced
  accuracy/F1 `0.974/0.974`. Target predictions were frozen before the existing
  four-game labels were opened.
- External evaluation is rejected: 108/108 target rows become free throws,
  balanced accuracy/F1/AUC are `0.500/0.675/0.644`, and every game's balanced
  accuracy is `0.500`.
- No checkpoint, runtime path, API or blind-game asset changed. Further
  threshold/representation search on the now-open four games is prohibited.
  The next source must match AGU's anchor-centred causal-window framing and
  retain whole-game/broadcast isolation.

## 2026-07-30 exact-candidate AGU v3 plus EBQwen fusion

- Fixed one 32-row plan over four opened development games, balanced to four
  positive and four negative rows per game. Both sources use the same video,
  candidate bundle, event and frame bounds.
- Added strict base extraction with plan/bundle/video/checkpoint hash binding,
  at most three ball-proximal unique players and unchanged v3 preprocessing.
- Added label-free, non-runtime base artifacts and four fixed, non-learned
  rules: `base_only`, `vlm_only`, `both_confirm`, `either_confirms`.
- Current v3 emits 22 `not_field_goal`, ten `unknown` and zero
  `live_field_goal` states. Its maximum shoot-probability AUC/AP are
  `0.512/0.564`.
- Delayed evaluation rejects every rule. EBQwen and OR remain at
  precision/recall `0.571/1.000`; AND and base-only have zero recall.
- No checkpoint or runtime path changed. Replace the SpaceJam action proxy
  with causal, broadcast-aligned player+ball+rim event-state supervision from
  whole-game-isolated sources.

## 2026-07-30 sparse-gated temporal pooling

- Rechecked `yerx/bb`, MUVS and the newly updated BasketEvent repositories.
  None combines explicit compatible terms, source media and relevant event
  labels, so no external asset was downloaded.
- Added an independently implemented `gated` temporal operator to the existing
  24-position dense model. A temporal convolution feeds a normalized learned
  gate; weighted sparse evidence, global max and start-to-end change feed the
  main and auxiliary heads.
- Kept the unchanged nested leave-one-game-out protocol and included conv,
  GRU and gated variants only in training-game configuration selection.
- Two of four outer folds select gated, but pooled balanced accuracy/F1 are
  `0.703/0.719` and worst-game balanced accuracy is `0.583`.
- Reject promotion. Further pooling variants on the same 108 opened labels
  are stopped; improved continuous ball/rim evidence or new game-diverse
  causal supervision is required.

## 2026-07-30 independent two-frame formation VLM

- Added a separate label-hidden formation contract rather than weakening the
  complete-shot prompt to fit local memory. It binds the causal phase plan,
  exact two-frame indexes, source/bundle/event identities, model revision,
  weight hash, prompt hash, sampling settings and cache fingerprint.
- Added deterministic derivation, Ollama execution, delayed evaluation and a
  fixed `upstream_positive AND NOT free_throw_setup` fusion evaluator. Every
  artifact is offline-only, rejects blind-source overlap and requires exact
  plan coverage.
- A 4B two-frame probe was resource-stopped before producing predictions.
  The final 2B protocol used two 384px frames and context 2048, sealed all
  110 predictions before loading the existing phase review, and returned no
  unavailable rows.
- Delayed evaluation rejects the classifier: true free-throw support is 58,
  TP/FP/FN are `1/1/57`, precision/recall are `0.500/0.017`, live-play
  retention is `0.979`, and promotion eligibility is false.
- Fixed veto fusion changes zero upstream decisions and leaves
  precision/recall at `0.682/0.857`; no checkpoint, API, service configuration
  or runtime inference path changed.
- Resource debugging found a separately resident 4B Ollama process during
  early 2B resumptions. The runner now fails before video IO when `/api/ps`
  reports any model other than the requested target. The clean final segment
  stayed below 83.4% memory and 41.9% CPU.
- Stop prompt, offset and threshold tuning on these opened labels. A future
  formation experiment requires a newly frozen, whole-game-disjoint target
  set or broader license-compatible formation supervision.

## 2026-07-30 MUVS source-state development

- Corrected the public-source catalog after the official MUVS Zenodo record
  was found to declare CC-BY-4.0. The package README license placeholder is
  retained as an explicit release-time caveat.
- Added deterministic event selection, balanced label-hidden plans, remote
  two-frame materialization, review sheets, exact review sealing, MobileNet
  embedding extraction and multi-artifact source screening.
- Codex annotated only public MUVS source frames. Every artifact is
  training-only, hash-bound and explicitly excludes Codex runtime answers.
- A 48-row all-event review and a 24-row NBA/WNBA supplement deduplicate to
  60 determinate rows. One duplicate-frame label conflict caused a fail-closed
  screen; enlarged review corrected the first-pass label before resealing.
- Fixed leave-one-event-out screening rejects all MobileNet feature variants.
  Best balanced accuracy is `0.641` with free-throw recall `0.364`; no target
  labels, blind games, runtime path or checkpoint were touched.

## 2026-07-30 MUVS formation geometry and local-view development

- Added `agu.muvs-formation-detections.v1` and
  `agu.muvs-formation-geometry-screen.v1` contracts. They bind exact source
  plan/frame/review hashes, BODD model/classes/settings, normalized boxes and
  paired-frame hashes, and are permanently training-only.
- A fail-closed duplicate join exposed batch-composition-dependent YOLO
  outputs. The extractor now requires batch size 1 instead of weakening the
  equality gate. Canonical base/professional detection artifacts are
  `887db98d3684366e6c2810d1a04ff51373db1dc995834d16c9ffaa7c3e579032`
  and
  `718eee74e6bcf06e7f38f3f3f5f78acd8a3e4084035a47c9fe087745c3da5d8a`.
- Fixed aggregate, nearest-player-slot and combined paired geometry were
  evaluated leave-one-event-out. Best balanced accuracy is `0.712`, below the
  `0.85` gate, and worst positive-event recall is `0`; no target evaluation
  followed.
- Added `agu.muvs-local-visual-embeddings.v1` and
  `agu.muvs-local-visual-screen.v1`. Before extraction, fixed left/center/right
  65% views, detector-selected context, symmetric context and fold-local PCA8.
  The pinned Apache-2.0 DINOv2-small representation reaches only `0.686`
  balanced accuracy and `0.455` free-throw recall.
- Both routes are rejected. No runtime/service/config/checkpoint behavior
  changed, HOU–ORL remained sealed, and the unused 84 MiB DINOv2 checkpoint
  was deleted after its hash-bound embeddings were sealed.

## 2026-08-01 Open Images V7 ball-box transfer screen

- Added an official-source, offline-only Open Images V7 validation slice with
  240 images and 358 generic/sports-ball boxes. The manifest records each
  source URL, landing page, per-image license and pixel SHA-256; 14 images also
  carry verified basketball/basketball-player image labels.
- Added a MobileNetV3-small exact-crop verifier with one non-overlapping
  background crop per image and GroupKFold by image ID. Source OOF AP is about
  `0.993`, but the frozen independent LAL–BOS transfer reaches only lower-bound
  P/R `0.559/0.356`; screen SHA-256 is
  `53a6d52248c63ff04c30220c0b8b7e2f3ca2320d0263051153fafb9801446639`.
- The screen is `accepted=false`. The guard recorded zero stops (peak process
  RSS `834,289,664` bytes; system memory `84.6%`). No checkpoint, detector,
  runtime or blind-game asset changed; the small source slice remains only a
  future offline pretraining candidate.

## 2026-08-01 same-detector broadcast hard-negative expansion

- Built a six-band, 180-candidate RF-DETR review plan on the opened 2011-05-10
  ATL–CHI development broadcast. The plan binds the exact perception artifact,
  raw-video hash, frame pixel hashes and fixed confidence bands; no event,
  outcome, identity or statistic fields are exposed to the reviewer.
- Codex sealed 55 visible balls, 98 false positives and 27 uncertain crops.
  Plan SHA-256 is `93a9062caf41b0ef3f6aa89540562c4eb00fdf17ea750c492686b605c02468d3`;
  review SHA-256 is
  `a46062ade1a9b883f2c136dda4c968e836f3f715144a8cc70797c4956f2991aa`.
- Trained the existing offline MobileNet verifier on three game-disjoint
  opened source reviews and evaluated a sealed LAL–BOS G2 target. The four
  variants fail the lower-bound 0.85/0.85 gate; the best is visual plus track
  geometry at `0.542` precision and `0.677` recall. No checkpoint is saved and
  no runtime or blind asset is touched.
