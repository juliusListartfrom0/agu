# TASK-0041 Solution

## Architecture

Add stable Pydantic contracts in `app/analysis/schemas.py` and isolated Python modules under `app/analysis/perception`, `game_state`, `box_score`, and `review`. External detectors, pose models, OCR and VLM remain optional adapters selected through `app/config.py`. AGU owns normalized observations, state transitions, event relationships, immutable revisions, reconciliation and aggregation.

The implementation order follows the six delivery areas in the requirement. Official output is added alongside the existing `action_proxy_v1` estimate; the estimate is not promoted or removed.

## Evaluation design

An inference run creates a sealed prediction bundle containing raw-video hashes, configuration/model provenance, predictions and a bundle hash. The evaluator rejects bundles that list non-video inputs or any ground-truth/reference path. It then loads truth in a separate process/step and computes strict one-to-one event matching and box-score metrics per game.

The 0.85 gate applies independently to Game A and Game B. Game B cannot satisfy the full-game/all-category gate until its complete truth is available; until then it is explicitly reported as a partial-quarter gate.

Dense raw-only Codex review is an isolated annotation and acceptance-truth
workflow, never an AGU recognition fallback. Its importer binds every decision
to the immutable manifest/raw-video hashes, but `codex_confirmed` and
`human_confirmed` events are forbidden from an autonomous prediction bundle.
Only `vision_confirmed` and AGU-configured `edge_vlm_confirmed` events may enter
the autonomous 0.85 acceptance metric.

Codex may also replace manual labor for model-training labels such as player,
ball and rim boxes, tracks, jersey/team appearance, possession, release and
first-control annotations. These labels are accepted only through
`agu.training-annotation.v1`: source and label files are SHA-256 bound, the
manifest is checked against sealed benchmark video hashes, and it is explicitly
non-runtime-consumable. Models may learn from such disjoint labels; AGU may not
load the labels as an event/statistics sidecar during an acceptance run.

Official VLM adjudication uses two separable adapters. The semantics pass sees
clean, anchor-centered raw frames and decides event occurrence/outcome/value.
Only after a positive semantic result does the actor pass see sparse canonical
identity overlays centered on release or first control. The merged decision
still passes schema completeness, trajectory priority and causal-parent gates.

The autonomous path is traditional detector/tracker/team-color/action-model
candidate generation, followed by bounded local VLM adjudication and the
deterministic ledger. The VLM may select only player/team IDs proposed by
traditional tracking evidence; it cannot invent reference roster IDs.
Incomplete actor/team/outcome/value evidence remains `needs_review` and scores
as a miss in autonomous acceptance.

Fusion prefers resolved traditional ball/rim trajectories over contradictory
VLM outcome guesses. Dependent rebounds fail closed unless their linked shot is
an automatically confirmed miss, and a made shot rejects the rebound
deterministically. Three-point labels require grounded line, feet and release
geometry; long noisy shot clusters restrict actor observations to a bounded
lookback before the trajectory outcome.

When no roster is supplied at inference, AGU emits stable raw-visible player
identities instead of guessing private evaluator IDs. A separate post-seal
evaluation artifact may declare a global one-to-one mapping to truth roster
IDs. The artifact is bound to prediction and truth hashes and is reported as an
identity-aligned metric alongside the unaligned strict metric. It cannot change
event labels or timestamps and cannot use per-event/many-to-one mappings.

The reusable identity stage consumes only raw-video-bound perception artifacts
and their matching raw videos. It extracts crop embeddings through the existing
adapter, splits long tracker gaps into tracklets, applies team/temporal/spatial
hard conflicts and complete-link similarity merging, then seals canonical IDs,
tracklet maps, model provenance and an artifact hash. Candidate generation
verifies that artifact and resolves an observation by source-video/player/frame
before attaching an actor. MobileNetV3 remains the lightweight default; OSNet
is an optional local ReID backend. Both remain replaceable adapters, and unsafe
thresholds fail through fragmentation or `unknown` rather than forced
roster-sized merges.

Multi-period games use one dense manifest per raw video. Import validates every
period independently, then remaps local `video_001` event/evidence references to
the deterministic game-level video order before sealing. A complete game is
eligible for official status only when every period has zero-gap full-video
coverage.

## Open-source capability assessment

- Detection/pose: MMDetection/RTMDet and MMPose/RTMPose (Apache-2.0) behind adapters; ONNX Runtime is the preferred release runtime. Existing Ultralytics support remains an opt-in AGPL/commercial-license path.
- Open-vocabulary calibration: YOLO-World can be selected through the existing
  Ultralytics adapter with explicit class prompts. Its upstream implementation
  is GPLv3 and the Ultralytics integration remains AGPL/commercial-license
  governed, so it is an opt-in open-source research backend rather than the
  default redistribution choice. Camera-geometry and temporal-flight filters
  are AGU-owned, dependency-light fallbacks; missing evidence yields no
  candidate rather than a fabricated event.
- Tracking and geometry: existing AGU tracking plus OpenCV for homography, optical flow and Kalman-style trajectory operations.
- OCR: existing RapidOCR adapter, optional PaddleOCR.
- Review: existing Ollama/Qwen adapter for bounded event windows; CVAT Community and FiftyOne remain offline optional tooling.
- Action pretraining: BARD annotations are imported through a hash-sealed
  `agu.open-action-dataset.v1` catalog. The annotation repository is CC BY 4.0,
  but the linked NBA media rights are not established by that license, so AGU
  imports metadata/labels only and requires separately authorized media. BARD
  is useful for weak event-semantic pretraining; it does not provide actor
  boxes and cannot replace in-domain action-owner labels.

Action-owner candidates retain ball center/bbox evidence, pose keypoints and
ball-to-player/wrist temporal features. Multiple perception sources namespace
their tracker IDs before fusion. A selected raw-only identity graph is applied
to its declared source before namespacing, and persistent multi-frame box
overlaps are suppressed as duplicate backends. The default bound is 16 distinct
player candidates so a ten-player scene remains covered under fragmentation.
Codex training review receives whole-event sheets plus per-candidate temporal
crops; it may select only an existing candidate and may bound a verified track
segment around the action anchor.

Shot proposals may additionally require a bounded backward ball chain with a
minimum number of points and vertical rise. This raw-detection-only gate rejects
stationary/held-ball proximity hits before action-owner ranking. Thresholds are
explicit CLI/config inputs and are sealed into candidate provenance.

Fallbacks may emit candidates or `unknown`, but never fabricated official events. Model paths, backends and thresholds are configuration fields with environment documentation.

## Verification

- Unit tests for schemas, event revisions, relationships, state transitions, aggregation, reconciliation and leakage rejection.
- Mechanical rejection of Codex/human/reference provenance in autonomous bundles and evaluation.
- Synthetic trajectory tests for made/missed shots, shot value, rebound, block, steal and assist chains.
- Existing analysis/evaluation tests, then full pytest.
- Harness gate and local service health/ready/run/status smoke after runtime-facing integration.
- Raw-only sealed runs and separate evaluation reports for both available games.
