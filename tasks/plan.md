# Implementation Plan: NBA-Identity primary-release audit

## Overview

Audit a newly discovered basketball event/identity source against AGU's
license, reproducibility and media-rights gates. The source may be useful at
paper level, but no data should be materialized until the official release has
explicit terms and a verifiable file manifest.

## Architecture decisions

- Keep the source in the research catalog as a paper/reference entry only when
  raw media, data terms or file provenance are incomplete.
- Treat the paper's event taxonomy as useful context, not as downloadable
  supervision; never infer a license from a paper's “publicly available” claim.
- Use the existing `.venv` Python 3.11 for catalog tests; do not add a model,
  dependency or downloaded archive.

## Task list

### Phase 1: Audit

- [x] Read the official NBA-Identity paper and record the event/identity scope.
- [x] Read the official repository README and inspect the release links and
      repository license metadata.

### Phase 2: Governance

- [x] Add a fail-closed catalog entry and regression assertions.
- [x] Update dataset/current-solution/task-board records; retain no source data.

### Checkpoint

- [x] Verify the focused catalog test and keep `accepted=false` and
      `runtime_consumable=false`; no download is justified.

### Phase 3: Existing-source completeness

- [x] Audit the complete MUVY basketball ZIP prefix by HTTP Range without
      reading video bytes.
- [x] Seal the 26-entry inventory and confirm the local 13-video subset already
      covers all basketball media/annotation entries.

### Phase 4: Rebound-source availability

- [x] Audit the official NBA Rebounds paper's public-availability statement.
- [x] Record the pending NBA-permission gate; do not download unavailable clips.

### Final checkpoint

- [x] Update AGU catalog/docs/task board and Wiki; run focused/full tests,
      Ruff, diff check and Wiki lint.

## Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Publicly described dataset is not reproducibly downloadable | High | Require official file inventory and immutable hashes before materialization. |
| Underlying broadcast rights are unclear | High | Keep the source paper/reference-only and do not train or redistribute. |
| Feature archive terms differ from source-video terms | High | Treat code/data/media rights as separate catalog fields and fail closed. |

## Follow-up: nonlinear ball-candidate calibration (2026-08-02)

### Screen

- [x] Bind the three disjoint RF-DETR source perception/plan/review bundles and
      the LAL–BOS target by artifact and raw-video SHA-256.
- [x] Add RED/GREEN tests for the nonlinear detector/track feature contract.
- [x] Fit grouped ExtraTrees and Logistic calibration variants with a source-only
      recall-floor threshold; reveal target labels only for the external screen.

### Decision

- [x] Verify artifact SHA, `pixel_decode=false`, `runtime_consumable=false` and
      `accepted=false`; no checkpoint or runtime path was changed.
- [x] Record the next blocker: cross-broadcast continuous ball supervision with
      hard negatives remains the limiting factor; more classifier nonlinearity is
      not sufficient.

## Follow-up: E-BARD ObjectClassification crop-role source (2026-08-02)

### Acquisition and audit

- [x] Verify the official Hub card, CC BY 4.0 license, immutable revision and
      82.5 MB archive hash before downloading.
- [x] Download one archive copy into
      `dataset/public_sources/e_bard_object_classification_v1/all.zip`.
- [x] Add RED/GREEN tests for JSON manifest labels, image references, JPEG
      decoding, class counts and game-overlap detection.
- [x] Seal the offline audit artifact; do not extract a duplicate copy.

### Decision

- [x] Keep the source as an offline object-role/VLM crop-training candidate.
- [x] Reject it as a cross-game detector/temporal/盲测 source because all three
      splits share most game IDs; runtime remains `runtime_consumable=false`.

## Follow-up: E-BARD crop-role transfer screen (2026-08-02)

### Screen

- [x] Add a CPU-only RGB/HSV/gradient baseline with game-ID-disjoint source
      sanity split and source-only precision threshold selection.
- [x] Score only the hash-bound ATL–CHI review-sheet insets; exclude 27
      `uncertain` decisions and do not decode new video or touch blind assets.
- [x] Run the fit under `run_guarded_training.py`; record zero resource-guard
      stops and retain the JSONL monitor log.

### Decision

- [x] Source holdout P/R is `0.857/0.728`, but ATL–CHI transfer is only
      `0.326/0.255` (F1 `0.286`) on 153 determinate candidates.
- [x] Seal `accepted=false`, `runtime_consumable=false`; do not save a
      checkpoint or connect the baseline to detector/VLM/runtime paths.
- [x] Keep cross-broadcast continuous ball supervision and hard-negative
      precision as the active blocker; require more independent sequential
      ball/non-ball labels before another gate attempt.

## Follow-up: cross-broadcast full-frame detector increment (2026-08-02)

### Dataset and training

- [x] Add hash-checked full-frame materialization from four reviewed broadcast
      bundles; keep the 417-row dataset truth-free and runtime-ineligible.
- [x] Add RED/GREEN tests for frame-pixel binding, label policy, duplicate
      decisions, IoU metrics and conservative external scoring.
- [x] Run the guarded CPU YOLOv8n experiment in `.venv` and seal checkpoint,
      results and resource-monitor hashes before deleting unpromoted weights.

### Decision

- [x] Re-screen the disjoint LAL–BOS development broadcast with the final best
      checkpoint at confidence 0.10/0.25/0.50; all three fixed points fail the
      `0.85/0.85` gate and remain `accepted=false`.
- [x] Retain the manifest, review bindings, screen JSON and resource logs, but
      remove `best.pt` and `last.pt`; do not change detector defaults, VLM,
      runtime schemas or HOU–ORL blind assets.
- [x] Keep the active blocker explicit: more continuous cross-broadcast
      ball/non-ball hard negatives and causal ball–hand–rim trajectory evidence
      are required before another model or blind-inference attempt.

## Follow-up: rejected-source media cleanup (2026-08-02)

- [x] Delete the sealed, unconsumed DeepSportRadar raw/derived media, APIDIS
      raw/range/derived media and the rejected Play-by-Play ZIP.
- [x] Retain only README/manifest and result/resource/hash evidence; verify no
      code, test, runtime or blind-game asset references the removed payloads.
- [x] Record the approximately 1.8 GiB recovery and keep the current blocker
      unchanged: no new blind-inference or runtime promotion is justified.

## Independent EBQwen evidence-gate screen (2026-08-03)

- [x] Add RED/GREEN tests and an offline evidence gate that requires continuous
      live play, controlled ball before release and ball separation; veto replay
      or free-throw evidence and abstain when auxiliary evidence is missing.
- [x] Select each held-game threshold only from the other games' OOF rows, then
      screen the frozen 32-window EBQwen Q4KS prediction set against the exact
      hash-bound annotation bundles.
- [x] Seal TP/FP/FN/TN `13/9/3/7` and pooled P/R/F1
      `0.591/0.813/0.684` as `promotion_eligible=false`; keep the module and
      artifact offline-only with no runtime or blind-game change.
- [x] Re-audit the `saveerjain/basketball-events` research-only release; keep
      its incompatible/no-compatible-license 7.87 GB payload unmaterialized.
- [x] Register the closer Roboflow `ball-tracking-udopu/2` candidate as
      CC BY 4.0/700-image `ball/goal`, but keep it unmaterialized after the
      export endpoint's Cloudflare 403; do not bypass authentication.

## Follow-up: public ball-sample audit and conservative cleanup (2026-08-03)

- [x] Temporarily download the public `emirsahin/basketball-ball` archive to
      `/tmp`, verify its SHA/file count/label-file inventory, and remove it
      after finding a single augmented shoot sample with no YOLO labels.
- [x] Record the Hub-MIT versus embedded-upstream-CC-BY-4.0 rights conflict
      and missing-label boundary in the public source catalog and regression
      test; keep it research-reference-only.
- [x] Remove the rejected Open Images generic pixel/raw-CSV payload while
      retaining its manifest, README and screen/resource evidence; do not
      touch E-BARD/BODD, EBQwen, accepted sources or HOU–ORL blind assets.

## Infactory auxiliary tiny-ball audit and cleanup (2026-08-04)

- [x] Download only the metadata-referenced Infactory CC BY-NC 4.0 subset at a
      pinned source revision; exclude the repository's unreferenced image
      payload and seal deterministic whole-clip splits and labels.
- [x] Run the guarded canonical `.venv` CPU YOLOv8n auxiliary experiment and
      record training metrics, resource envelope and all checkpoint hashes.
- [x] Screen the held Infactory clips and an independent E-BARD basketball
      test set; reject the candidate because no confidence reaches `0.85/0.85`
      and the basketball screen has zero true positives.
- [x] Delete the unpromoted best/last checkpoints and generated label caches
      after sealing retention evidence; keep only the licensed source subset,
      reproducible manifests and offline screen artifacts.
- [x] Preserve the blocker: non-basketball auxiliary data cannot replace
      cross-broadcast basketball ball/non-ball hard negatives or causal
      ball-hand-rim evidence; runtime and blind inference remain unchanged.

## UniqueData basketball-tracking mirror audit (2026-08-04)

- [x] Download the fixed public mirror revision to `/tmp` and hash the CSV,
      image archive, boxes archive and loader code.
- [x] Verify all 70 CSV images decode, then check the 106 `boxes/frame_*.PNG`
      entries and reject the positional image/box pairing as unbound.
- [x] Seal the compact audit JSON, delete the temporary payload, and keep the
      catalog/docs/Wiki decision fail-closed.
- [x] Preserve the active blocker: this tiny CC BY-NC-ND sample cannot replace
      licensed cross-broadcast continuous ball/non-ball and causal supervision.

## CHI–UTA hard-negative review increment (2026-08-04)

- [x] Manually review 36 non-blind CHI–UTA candidate panels and seal 11
      `valid_ball` plus 25 `false_positive` decisions.
- [x] Materialize and hash-bind the de-duplicated 441-row dataset (354 train,
      87 val; 170 positive, 271 hard-negative rows).
- [x] Run the guarded canonical `.venv` CPU YOLOv8n experiment and retain
      args/results/resource evidence with checkpoint hashes.
- [x] Screen disjoint LAL–BOS at confidence 0.10/0.25/0.50; reject because
      the best fixed P/R/F1 is only `0.250/0.050/0.083`.
- [x] Delete the unpromoted best/last checkpoints and generated label caches
      after writing retention evidence; keep dataset and offline review traces.
- [x] Keep detector/runtime/EBQwen/HOU–ORL blind assets unchanged and preserve
      the cross-broadcast hard-negative plus ball–hand–rim causal blocker.

## Licensed basketball COCO ball/rim candidate (2026-08-05)

- [x] Pin and audit the CC BY 4.0 Hugging Face/Roboflow source at revision
      `92dea3042e1ec707be5b0d4f7ac474a2b314293`; retain only the 654-frame
      source payload and hash-bound audit/contact sheet.
- [x] Materialize the ball/rim-only YOLO view and verify split/clip disjointness,
      image decoding, category counts, watermark risk and action-label semantics.
- [x] Run source-only and 441-row cross-broadcast hard-negative YOLOv8n routes
      under the canonical `.venv` CPU resource guard, then screen disjoint
      LAL–BOS at fixed confidence points.
- [x] Reject runtime/external promotion: source-only best external P/R/F1 is
      `0.12/0.15/0.1333`, combined best is `0.13636/0.15/0.14286`; seal
      training/screen/resource hashes and delete unpromoted weights, caches and
      temporary materializations.
- [x] Keep the source and derived view offline-only; preserve the blocker of
      game-/broadcast-diverse continuous ball/non-ball hard negatives and
      ball–hand–rim causal sequences. Do not tune or inspect HOU–ORL blind data.

## Deduplicated 2025 basketball COCO supplement (2026-08-05)

- [x] Pin revision `d0b72a61490311431c00774db5277cdca737dbe5` and audit the
      MIT/CC BY 4.0 rights conflict; enforce the stricter attribution-required
      offline-only boundary.
- [x] Group augmentation variants by normalized source identity, retain the
      first upstream variant, and manually review the resulting 139 frames
      (99 ball boxes, 98 rim boxes, 19 empty frames).
- [x] Combine the de-duplicated supplement with the 2026 ball/rim view and
      441-row cross-broadcast hard-negative set; run the guarded canonical
      `.venv` CPU screen and seal training/resource/external hashes.
- [x] Reject promotion: internal best mAP50 `0.75056`, independent LAL–BOS
      best P/R/F1 `0.25/0.25/0.25` at confidence 0.20, below the `0.85/0.85`
      gate.
- [x] Delete temporary combined data, training directories, unpromoted
      checkpoints and label caches; retain source/audit/screen evidence and
      keep detector, runtime, EBQwen and HOU–ORL blind assets unchanged.

## Independent EBQwen ball-candidate VLM screen (2026-08-05)

- [x] Run a label-hidden 108-candidate LAL–BOS screen from the hash-bound
      RF-DETR perception/review plan using the retained local MLX 4-bit EBQwen;
      do not expose review decisions or statistics to the model.
- [x] Resume the initial resource-guard stop from cache with a 1.5 GiB
      available-memory floor while retaining 90% memory, 95% CPU and three-sample
      termination limits; complete and hash the prediction artifact.
- [x] Evaluate strict `ball`/`uncertain` states against the sealed review:
      precision `0.444060`, recall lower/upper `0.113961/0.102302`,
      `accepted=false`.
- [x] Keep the result offline-only; do not rewrite decisive reason text into
      labels, change detector/VLM/runtime defaults, or open HOU–ORL blind data.
      The next useful input remains cross-broadcast hard negatives and causal
      ball–hand–rim evidence.

## ORL–CLE offline supplement and cross-game verifier (2026-08-05)

- [x] Download the user-authorized official NBA Games ORL–CLE media into an
      isolated source directory, verify SHA-256 and whole-file seek/decode
      integrity, and bind its offline-only rights boundary.
- [x] Run the canonical BODD 1 FPS full-game scan and object-only 108-panel
      review; seal the 49/22/37 valid/uncertain/false-positive counts and
      reject runtime/cross-game-training promotion.
- [x] Run the label-hidden 12-panel EBQwen MLX 4-bit probe; record weighted
      P=`1.0`, R lower/upper=`0.500617/0.459419`, and reject the gate.
- [x] Run the same-detector LAL–BOS→ORL–CLE verifier; best conservative
      P/R=`0.714836/0.750198`, below the dual `0.85/0.85` gate.
- [x] Keep the video and all evidence offline-only; do not modify detector,
      VLM/runtime defaults, EBQwen weights or HOU–ORL blind assets. The next
      useful input remains licensed game-/broadcast-diverse continuous
      ball/non-ball hard negatives and causal ball–hand–rim sequences.

## SportsGrounding metadata-only causal semantics candidate (2026-08-05)

- [x] Pin revision `1fb03e7786228ee6d1d7909eb7011e7e3fb7eb2c` and verify the
      CC BY-NC 4.0 boundary, split-disjoint video IDs, 25 FPS schema and exact
      bbox/frame-span binding.
- [x] Download only train/val JSON metadata (23,792,105 bytes); do not fetch
      the approximately 14.7 GB video archives or bypass any access terms.
- [x] Seal 520 videos, 4,243 person tubes and 3,633 ball-mentioned captions
      as an offline causal/VLM semantics candidate; explicitly reject it as
      ball-box training, runtime input or external acceptance truth.
- [x] Keep detector/VLM/runtime defaults, EBQwen weights and HOU–ORL blind
      assets unchanged. The source helps language/tube semantics only and
      does not remove the continuous ball/non-ball and ball–hand–rim blocker.

## Shot-causal support screen (2026-08-06)

- [x] Add RED/GREEN tests for exact `(source_video_sha256,event_id)` joins,
      unknown-feature rejection, offline-only output and duplicate coverage.
- [x] Screen the existing 511-row frozen base OOF artifact with seven
      continuous causal-support features using outer game-held logistic fits
      and training-only thresholds.
- [x] Seal the baseline versus causal-support metrics: baseline pooled
      P/R/F1=`0.791667/0.846154/0.818004`; causal-support pooled
      P/R/F1=`0.572104/0.979757/0.722388`; both fail the dual cross-game gate,
      with causal support adding false positives.
- [x] Run the bounded 39-feature reason-evidence ablation; pooled
      P/R/F1=`0.5648/0.9352/0.7043` and Bdc held-game precision=`0.6500`.
      Seal `shot_causal_support_screen_all39_v1.json` as rejected offline
      evidence rather than expanding the runtime feature surface.
- [x] Keep the result offline-only and do not change detector, runtime, VLM,
      EBQwen or HOU–ORL blind assets. The blocker remains licensed,
      game-/broadcast-diverse continuous ball/non-ball hard negatives plus
      independently verifiable ball–hand–rim causal sequences.

## Broadcast false-positive diagnosis and guarded MViT probe (2026-08-06)

- [x] Compare class-weighted logistic, ExtraTrees, HistGradientBoosting,
      broadcast/overlay gates and the existing outer-game-held features; no
      candidate raised the worst-game precision/recall pair to `0.85/0.85`.
- [x] Inspect all 23 high-confidence false positives in nonblind
      `BdcP-XwUCk8.mp4`; the dominant failure is a normal live possession
      without a complete visible release, not replay/free-throw confusion.
- [x] Seal the offline diagnosis as
      `analysis_outputs/public_research/shot_broadcast_fp_audit_v1.json`
      (SHA-256 `7ae5e18136888e903c5ab396ca5037ae0a05980e732f439be63911afe72f88e2`).
- [x] Start a one-epoch, one-block local MViT tail probe under the resource
      guard; stop it after two consecutive available-memory breaches
      (`<2.5 GiB`), emit no checkpoint, and leave runtime defaults unchanged.
- [ ] Resume raw-video temporal fine-tuning only after memory headroom is
      restored and the experiment can run under the same guard; otherwise
      acquire licensed cross-broadcast non-release hard negatives and causal
      ball–hand–rim supervision first.

## BARD event metadata hard-negative index (2026-08-06)

- [x] Pin BARD revision `add4109bf8b2034a32f3b6a83fa0d8c0ae638473` and download
      only `dataset.csv`, `dataset_paths.csv`, the upstream license and README;
      no NBA source video was fetched.
- [x] Verify 14,676 event rows across 60 games, including 2,631 rows without a
      field-goal or free-throw action (rebound/foul/turnover/steal/block/
      violation) as a candidate non-release hard-negative acquisition index.
- [x] Seal the metadata-only bundle at
      `dataset/public_sources/bard_event_metadata_v1/` and
      `analysis_outputs/public_research/bard_event_metadata_audit_v1.json`
      (manifest SHA-256 `315f7fa1ad4c59c4cad806b2cd26170d23754eb9ec7be19178274a1257f8a75b`).
      Keep `runtime_consumable=false`, `training_media_eligible=false` and
      `codex_runtime_answer_used=false`.
- [ ] Do not fetch or train on NBA-hosted clips until a separate media-rights
      grant is established; if rights-cleared clips become available, build a
      small game-disjoint non-shot subset and screen it before promotion.

## BARD non-shot validation clips (2026-08-06)

- [x] Pin the same BARD revision and materialize only the 18 benchmark clips
      whose labels contain no 2PT Shot, 3PT Shot or Free Throw action (14
      Turnover, 11 Steal, 6 Foul labels; 15 source games; 138,530,775 bytes).
- [x] Verify every clip by Git blob SHA-1, SHA-256 and OpenCV decode (60 FPS,
      1280×720), and correct the local README's source-game count to 15.
- [x] Manually review three label-hidden quantile frames per clip; the clips
      are live possessions/stoppage hard negatives, not replay/free-throw-only
      material. Seal
      `analysis_outputs/public_research/bard_nonshot_validation_audit_v1.json`
      with source manifest SHA `aefebdea2605b90652a85fe0668c348f8958aae05029d0f040a4271fad059e02`.
- [x] Keep the subset offline-only and runtime/blind/promotion-disabled; run the
      low-memory MobileNetV3 frozen-feature probe in
      `analysis_outputs/public_research/bard_nonshot_scene_probe_v1.json`
      (artifact SHA `ea5e59563da9da3915bbea4609dad00dc57b0ff7549e05099a6db3fc7bec85ca`).
- [x] Reject the first augmentation: adding all 18 BARD clips reduced the
      held-game worst precision from `0.5862` to `0.5455` and recall from
      `0.9091` to `0.8636`; do not promote this data/model.
- [x] Run the guarded MViT frozen-feature follow-up (18/18 clips, peak process
      RSS about 2.16 GiB in the smoke and no guard stop) and screen scene+MViT
      augmentation; worst precision fell from `0.5856` to `0.5234`, so reject
      this path as well. Seal
      `analysis_outputs/public_research/bard_nonshot_mvit_augmentation_screen_v1.json`
      (SHA `8e95dd0d896f1f7874d3885b89f7ab08ddc17b6f412c3398042ccd013220d46d`).
- [ ] Acquire or label more rights-cleared, broadcast-diverse clips with
      independently verified ball-hand-rim causal supervision before rerunning
      the gate; require worst-game precision and recall ≥0.85.

## VRU_Basketball continuous-scene subset (2026-08-06)

- [x] Pin revision `d256fa0b5fab474663f595abe4f386ac4d5adcc6` and retain only
      eight individually extracted clips (four Dongguan/four Guangzhou,
      `176,838,471` bytes) under the declared CC BY 4.0 source boundary.
- [x] Verify the retained clips as 25 FPS, 1920×1080, ten-second continuous
      scenes and seal `dataset/public_sources/vru_basketball_v1/source_manifest.json`.
- [x] Run the existing BODD detector on 400 sampled frames and manually review
      48 detector-selected basketball candidates (`44 valid_ball`, `4 false_positive`).
- [x] Retain the source only as an offline continuous-scene/hard-negative
      candidate; reject training, runtime, VLM default and blind-game use because
      the candidate review is biased and no ball–hand–rim truth is supplied.
- [x] Seal `analysis_outputs/public_research/vru_basketball_audit_v1.json`,
      remove temporary extraction/review files, and leave the active causal
      and cross-broadcast blocker unchanged.

## Play by Play basketball annotation slice (2026-08-06)

- [x] Download and MD5-verify the CC BY-NC 4.0 Zenodo archive; retain only the
      basketball annotation tree (381 clips, 56,578 frames) and record the
      source URL/hash in `dataset/public_sources/play_by_play_v1/README.md`.
- [x] Add a tested offline audit that counts standardized player/ball points,
      sparse visible-ball frames, stopped-ball points and coordinate ranges;
      seal `analysis_outputs/public_research/play_by_play_basketball_audit_v1.json`.
- [x] Delete the verified but redundant 162,983,009-byte source ZIP; keep the
      extracted basketball labels and audit as the reproducible local subset.
- [x] Reject runtime, blind-truth and causal promotion: there is no raw video,
      pixel ball box, hand label or rim label, so the existing blocker remains.

## VRU raw-frame ball–hand–rim causal review (2026-08-06)

- [x] Add tested, label-hidden plan/seal contracts and `.venv` OpenCV frame-hash
      binding for continuous VRU source clips.
- [x] Seal six windows: three visible release-to-rim chains and three
      cross-scene non-shot hard negatives; keep all shot outcomes and rim contact
      unresolved where the original frames do not prove them.
- [x] Write plan/review artifacts with source, frame, and plan SHA-256 bindings:
      `vru_causal_review_plan_v1.json` and `vru_causal_review_v1.json`.
- [x] Keep the annotation training/calibration-only
      (`runtime_consumable=false`, `codex_runtime_answer_used=false`); do not
      promote detector/VLM/runtime/weights or open HOU–ORL blind media.
- [ ] Expand to licensed, game-/broadcast-diverse independent causal sequences
      and pair them with exhaustive ball/non-ball hard-negative recall before
      attempting a new promotion screen.

## Rejected ORL–CLE media cleanup and candidate-source audit (2026-08-06)

- [x] Verify the rejected ORL–CLE MP4 against the sealed size/SHA before removal;
      retain the 564-byte download-state provenance and all audit/review artifacts.
- [x] Delete only `GxQV8JhJCMs.mp4` from the offline supplement, reclaiming about
      1.02 GiB; keep HOU–ORL blind media and both EBQwen weight variants.
- [x] Audit TeamTrack, Basketball Tracking Dataset and Dryad sport-ball candidates;
      do not download the 17.7 GB player-MOT corpus or non-basketball/static sources.
- [x] Recheck MUVY v1 and the unlicensed BasketEvent HF mirror; keep the existing
      bounded MUVY subset, delete the four-file BasketEvent transient probe, and
      reject the 7.8/6.73 GB full payloads for rights or causal-label gaps.
- [ ] Obtain a licensed, continuous, game-/broadcast-diverse source with independent
      ball/non-ball and ball–hand–rim truth before resuming blind inference.

## SPIROUDOME byte-range candidate audit (2026-08-06)

- [x] Inspect the 19,629,442,701-byte Kaggle ZIP by range only; enumerate 192
      members and confirm calibration/reference files plus 80 one-minute MJPEG
      members, with no ball/event annotation members.
- [x] Extract one bounded camera-1 member (929 frames, 1600×1200 at 25 FPS),
      run BODD at 2 FPS and confidence `0.01`, and visually review all ten
      candidates: `0 valid_ball`, `10 false_positive`, `0 uncertain`.
- [x] Seal `analysis_outputs/public_research/spiroudome_camera1_210000_audit_v1.json`
      and add the source to the catalog as official non-commercial research-only,
      offline hard-negative evidence; do not import it into runtime or training.
- [x] Delete the 396,074,466-byte temporary member, index, frames and range
      audit files; retain only the hash-bound audit and catalog entry.
- [ ] Continue seeking a licensed continuous broadcast source with independent
      ball/non-ball and ball–hand–rim truth; SPIROUDOME does not remove that blocker.

## Targeted disk cleanup (2026-08-06)

- [x] Verify and remove only the redundant 50 MiB RTMPose source ZIP; the
      extracted `end2end.onnx` and hash-bound source manifest remain.
- [x] Preserve EBQwen native/MLX weights, the `r2plus1d_v3` runtime checkpoint,
      accepted offline media and all HOU–ORL blind assets.

## Qlean continuous-video candidate (2026-08-06)

- [x] Audit the public [Qlean Basketball Match Videos](https://huggingface.co/datasets/qleandataset/video-basketball-match)
      card: six games, 28 MP4 files, about 26.40 GB, multiple camera positions,
      gated academic research access.
- [x] Register the source with a fail-closed metadata-only adapter boundary and
      seal `qlean_video_basketball_match_audit_v1.json`; no payload was fetched.
- [ ] Obtain authorized access, then independently verify the governing Japanese
      license, per-file hashes, annotation availability and game-disjoint causal
      coverage before considering any bounded download.
- [ ] Keep the model/runtime and blind inference paused until an independent
      continuous ball/non-ball plus ball–hand–rim source clears the cross-game gate.

## SportsAction / MultiSports gated action-tube candidate (2026-08-06)

- [x] Verify the official SportsAction card and its basketball action taxonomy,
      including offensive/defensive rebound and shot classes.
- [x] Record the person-tube annotation boundary and the absence of ball/hand/rim
      or made/miss truth; no payload was downloaded.
- [x] Add the fail-closed catalog entry and seal
      `sports_action_multisports_audit_v1.json` (SHA-256
      `23c19cec33442ea0c0845fb833323106322fa956bed1e16f201c55dee7371642`).
- [ ] If authorized noncommercial access becomes available, audit source-media
      terms, file hashes and causal coverage before any bounded extraction.
- [ ] Do not promote the candidate, alter runtime/VLM defaults or resume blind
      inference from this metadata-only registration.

## MEM--OKC development PBP coverage probe (2026-08-06)

- [x] Run a resource-guarded 10-second scoreboard OCR and seal the PBP alignment
      and candidate-join coverage probe.
- [x] Confirm 470/477 PBP events map to 200 shot labels, while existing candidates
      cover only 63/200 at +/-90 frames.
- [x] Reject the probe for training/runtime use; +/-300-frame joins are too loose
      for causal supervision, so no embeddings or labels were written.
- [ ] Obtain more densely timed candidate proposals or an independent sub-second
      causal anchor before attempting another feature extraction.
- [ ] Keep the 85% cross-game gate and blind inference paused.

## VRU additional continuous clips and causal review (2026-08-06)

- [x] Extract and verify eight additional CC BY 4.0 VRU members by bounded HTTP
      Range requests (179,162,740 bytes); no full archive was materialized.
- [x] Keep `source_manifest_extra_v1.json` hash-bound separately from the original
      eight-clip manifest and its detector audit.
- [x] Seal the eight-window causal review: one shot (outcome unknown), one
      uncertain pass and six not-a-shot windows.
- [x] Seal the extraction/review audit with `runtime_consumable=false` and
      `accepted_for_training=false`; no runtime, weights or blind assets changed.
- [x] Run and seal the CPU/resource-guarded BODD visibility probe; reject
      detector-only causal fusion after observing candidates in pass and non-shot
      windows.
- [x] Build a fixed-seed, non-overlapping 32-window review set from the non-blind Bdc
      development video; no HOU–ORL media was opened.
- [x] Seal raw-only candidate/label/manifest provenance and review 8 positives plus
      24 hard negatives without score truth or runtime output.
- [x] Append frozen scene and MViT embeddings and rerun the four-game held-out screen;
      the increment remains below the 0.85 gate and is not promoted.
- [x] Remove temporary review images after sealing the source-frame specification.
- [ ] Replace same-game augmentation with an independently licensed, game-/broadcast-
      diverse ball/non-ball plus ball–hand–rim dataset before changing runtime defaults
      or resuming blind inference.
- [ ] Obtain independent cross-game/broadcast outcome truth and exhaustive
      hard negatives before any training promotion or blind-inference recovery.

## Bdc broadcast-state fusion follow-up (2026-08-06)

- [x] Fix and test generic `candidate_event_frame` anchoring for the offline
      broadcast-clock extractor.
- [x] Seal raw-only OCR for all 32 Bdc increment windows and merge it with the
      six existing 511-row clock artifacts.
- [x] Run the outer game-held scene+MViT+`base+broadcast_raw` screen; pooled
      P/R/F1 `0.7778/0.9059/0.8370`, weakest precision `0.7037`, unpromoted.
- [x] Preserve runtime/VLM/EBQwen/blind boundaries and retain only sealed JSON
      evidence; no checkpoint was created.
- [ ] Replace same-game augmentation with independently licensed, cross-game /
      cross-broadcast ball–hand–rim truth and exhaustive hard negatives.

## Lakers–Magic cross-broadcast development increment (2026-08-06)

- [x] Download the bounded NBA Games Lakers–Magic video and seal SHA/decode/source
      metadata; keep external media non-redistributable and runtime-disabled.
- [x] Build a fixed-seed 32-window raw-only review set, hide labels during selection,
      and seal 7 live attempts plus 25 broadcast hard negatives.
- [x] Append frozen MobileNet/MViT/Swin3D embeddings and run the five-game game-held
      OOF screen; the best scene+MViT result remains below the 0.85 gate.
- [x] Record the negative result and preserve the runtime/盲测 boundary; no checkpoint
      or default fusion change is authorized by this screen.
- [ ] Replace sparse same-game augmentation with independently licensed cross-game /
      cross-broadcast causal outcomes and exhaustive non-ball hard negatives.

## Post-audit media cleanup (2026-08-06)

- [x] Verify the SHA-256 values before deleting obsolete MEM–OKC development and
      enrollment media; keep state-file provenance and sealed coverage audits.
- [x] Preserve the remaining MEM–OKC blind source, HOU–ORL blind media and both
      EBQwen weight variants; remove only repository bytecode/test/lint caches.
- [x] Reclaim about 1.60 GiB without changing runtime, default VLM/detector,
      weights or blind-test boundaries.
- [ ] Repeat this exact-file, hash-first cleanup after each future data audit.

## Basketball Events pinned annotation audit (2026-08-06)

- [x] Pin the upstream Hugging Face revision and download a bounded annotation/sample slice;
      do not materialize the approximately 7.87 GB full media collection.
- [x] Verify SHA-256 and ffprobe metadata for the four sample clips and seal the manifest.
- [x] Register the research-only, event-semantic source with an explicit no-runtime/no-training
      boundary and document the lack of frame-level causal labels.
- [ ] Continue seeking a licensed continuous cross-game/broadcast source with sub-second
      ball--hand--rim outcomes and exhaustive non-ball hard negatives before resuming blind inference.

## HOU–SAC tiled-ball cross-broadcast screen (2026-08-06)

- [x] Correct the HOU–SAC source manifest to official game ID `0022400059` and retain the
      640×360 media only as an offline development copy.
- [x] Run a guarded BODD full-frame baseline and a two-overlapping-tile ball-only pass on
      600 samples; bind both outputs and the candidate bundle to the source SHA.
- [x] Review raw candidate sheets before opening the local PBP and seal the screen artifact
      `analysis_outputs/public_research/hou_sac_2024_raw_review_v1/tiled_ball_screen_v1.json`.
- [x] Confirm the tile route recovers detections where full-frame BODD emits zero balls, but
      first-quarter P/R/F1 is only `0.5385/0.5385/0.5385`; reject promotion.
- [x] Implement and test the screen-only tile geometry, box projection and same-frame overlap
      deduplication helpers; keep them outside the runtime detector path.
- [x] Run and seal a guarded independent YOLO11 pose-evidence follow-up; it leaves the first-
      quarter screen at `7/6/6` and is rejected for promotion.
- [ ] Implement tile-aware detector training and trajectory constraints on an independently
      licensed cross-broadcast hard-negative set.
- [ ] Keep runtime defaults and blind inference paused until the cross-game 0.85 gate clears.

## Gated shot-outcome source audit (2026-08-06)

- [x] Audit the current Hub card, access gate, size and rights boundary for
      `leharris3/basketball-shot-test-dataset`; do not download its 423 MB payload.
- [x] Keep the already-audited `yerx/bb` 1.72 GB free-throw source catalog-only because its
      license and media provenance are absent.
- [x] Add both fail-closed decisions to the public-research catalog; catalog SHA is
      `86b2cac9d1e47abfe85da05995d152cedcb47ff6ff1c253e89453c15be3928ee`.
- [ ] Continue seeking a licensed, game-diverse source with continuous ball/hand/rim outcomes
      and exhaustive non-ball hard negatives.

## E-BARD/MUVY tile-aware ball training screen (2026-08-06)

- [x] Materialize the pinned CC BY 4.0 E-BARD/MUVY ball source into 3,458 hash-bound overlapping
      tiles (`train=2,916`, `val=542`) without making the generated data runtime-consumable.
- [x] Run 3 guarded CPU epochs in canonical `.venv`; final validation P/R/mAP50/mAP50-95 is
      `0.7280/0.5648/0.589/0.267`, with no resource-guard stop.
- [x] Run the trained specialist on the fixed HOU–SAC window and seal candidate alignment
      `7/3/6`, P/R/F1 `0.700/0.5385/0.6087`.
- [x] Run the same frozen specialist on the 32 fixed independent Lakers–Magic windows; seal
      `7/19/0/6`, P/R/F1 `0.2692/1.000/0.4242`.
- [x] Reject the route for the 0.85 gate, seal
      `analysis_outputs/public_research/tiled_ebard_ball_train_v1/screen_v1.json`, and delete
      the generated tiles/checkpoint/external run after hash verification.
- [ ] Obtain independently licensed continuous ball–hand–rim outcomes plus exhaustive hard
      negatives before another detector/VLM promotion or resuming blind inference.

## Official DeepBall-Large basketball baseline screen (2026-08-06)

- [x] Pin the official WASB-SBDT revision, DeepBall-Large basketball model-zoo URL,
      MIT code boundary, source/config hashes and 4,131,456-byte checkpoint SHA-256.
- [x] Add a hash-checked offline CPU adapter and fixed-window screen with RED/GREEN
      tests; keep it outside AGU runtime detector/VLM defaults.
- [x] Run the HOU–SAC development threshold comparison under the resource guard;
      lock `0.50` from the development windows (`6/1/1/2`, P/R/F1 `0.8571/0.8571/0.8571`).
- [x] Run the locked threshold on the independent Lakers–Magic windows
      (`6/14/1/11`, P/R/F1 `0.3000/0.8571/0.4444`) and seal
      `analysis_outputs/public_research/deepball_large_v1/screen_v1.json`.
- [x] Reject runtime/VLM/blind promotion; retain only the 3.9 MiB official baseline,
      small evidence, scripts and tests. Remove the temporary smoke copy and duplicate
      `/tmp` download.
- [ ] Replace the detector-only baseline with licensed, broadcast-diverse causal
      ball–hand–rim outcomes plus exhaustive non-ball hard negatives before another
      detector/VLM promotion or resuming blind inference.

## Lakers–Magic independent VLM and DeepBall fusion screen (2026-08-06)

- [x] Derive and seal a 32-window label-hidden raw-frame plan directly from the Lakers–Magic
      candidate bundle; use only 4 chronological frames at 512 px for the Ollama reviewer.
- [x] Run `qwen3-vl:2b` with the strict-release prompt under the canonical `.venv` resource guard;
      all 32 predictions are sealed and independent of Codex/runtime answers.
- [x] Open the raw-review labels only after prediction sealing; VLM-only is `7/25/0/0`,
      P/R/F1 `0.2188/1.0000/0.3590`, so it fails the 0.95/0.85 gate.
- [x] Evaluate the predeclared `VLM AND DeepBall-Large` rule; it is identical to DeepBall-only
      at `6/14/1/11`, P/R/F1 `0.3000/0.8571/0.4444`, and reject promotion.
- [x] Preserve all plans, predictions, evaluation, fusion and resource artifacts as
      `runtime_consumable=false`; no runtime/VLM default, checkpoint or blind asset changed.
- [ ] Acquire licensed, broadcast-diverse continuous ball–hand–rim outcomes and exhaustive
      hard negatives before another VLM/detector fusion attempt or blind-inference recovery.

## BasketEvent player-grounded trajectory audit (2026-08-06)

- [x] Pin BasketEvent revision `85aaa3ce62bc096e3995c39ecfa6773fcc9fe5e1` and register the
      source with an explicit no-license/no-raw-video boundary.
- [x] Download and SHA-256 seal all 557 `valid` JSON annotations (`85,794,901` bytes) plus
      a 120-file round-robin `test` sample across 37 games (`18,911,750` bytes).
- [x] Audit player/ball trajectory shape, missing-ball clips and event taxonomy; retain the
      manifests and two audit JSONs as offline structured auxiliary evidence only.
- [x] Download `playnet.pt` only as a bounded compatibility probe, verify its 1,813,273,995-byte
      SHA-256, run the published model under the canonical `.venv` resource guard, then delete the
      unlicensed checkpoint; retain only compact provenance/results. Raw video, runtime defaults,
      EBQwen assets and blind inference remain untouched, with `runtime_consumable=false` and
      `training_media_eligible=false`.
- [ ] If a compatible media/rights release appears, re-audit the same game-held protocol;
      otherwise continue seeking licensed raw ball–hand–rim truth and exhaustive hard negatives.

## LAL–ORL Game 3 cross-broadcast screen (2026-08-06)

- [x] Download and hash-bind the bounded 640×350 NBA Games Game 3 stream; keep the external
      media and PBP under the no-redistribution/offline-only boundary.
- [x] Fix seeded uniform-window slack allocation and add deterministic/non-overlap/coverage tests.
- [x] Review 32 label-hidden raw windows and seal one visible field-goal attempt plus 31 hard
      negatives in `analysis_outputs/public_research/lal_orl_game3_v2/`.
- [x] Extract MobileNet/MViT features and run the new-game-held screen (`1/9/0/22`,
      P/R/F1 `0.1000/1.0000/0.1818`); screen the locked DeepBall baseline (`1/14/0/17`,
      P/R/F1 `0.0667/1.0000/0.1250`).
- [x] Test the hard-negative augmentation against the existing Lakers–Magic corpus; held-game
      P/R/F1 remains `0.4000/0.8571/0.5455`, so no promotion is allowed.
- [x] Run the full label-hidden independent `qwen3-vl:2b` strict-release probe on all 32 windows;
      it is all-positive (`1/31/0/0`, P/R/F1 `0.03125/1.0000/0.0606`). The fixed
      `VLM AND DeepBall-Large` rule remains `1/14/0/17` (P/R/F1 `0.0667/1.0000/0.1250`),
      with zero resource-guard stops; seal the result as offline-only.
- [ ] Obtain licensed, broadcast-diverse sub-second ball–hand–rim outcomes and exhaustive
      non-shot hard negatives before another fusion attempt or blind-inference recovery.

## LAC–DAL 2024 independent cross-broadcast screen (2026-08-06)

- [x] Download and hash/decode-audit the bounded H.264 640×360 external stream; keep the
      NBA metadata/PBP binding and no-redistribution boundary in a source manifest.
- [x] Select 32 label-hidden, non-overlapping windows and seal 9 visible releases plus 23
      context/replay/setup/no-release hard negatives in a raw-only bundle and training manifest.
- [x] Add the generic hash/boolean-label artifact builder and RED/GREEN tests; keep all output
      offline (`runtime_consumable=false`, `training_eligible=false`).
- [x] Run locked DeepBall-Large (`7/15/2/8`, P/R/F1 `0.3182/0.7778/0.4516`) and independent
      Qwen3-VL (`9/23/0/0`, P/R/F1 `0.2813/1.0000/0.4390`) screens plus fixed AND/OR fusion;
      both fail the gate and are not promoted.
- [x] Verify both guarded runs completed without resource stops, remove only sealed temporary
      review images, and leave runtime, weights, `.venv` and blind inference unchanged.
- [ ] Obtain licensed, broadcast-diverse sub-second ball–hand–rim outcomes and exhaustive
      non-shot hard negatives before any detector/VLM promotion or blind-inference recovery.

## Basketball Events shot-positive VLM audit increment (2026-08-06)

- [x] Download and SHA/size/decode-seal a 16-clip, four-game make/miss subset under the pinned
      research-only revision; retain the full source boundary and no-runtime/no-training flags.
- [x] Freeze a label-free 4-frame/512px strict-release plan and run `qwen3-vl:2b` under the
      canonical `.venv` resource guard; all 16 event-centered positives were available and
      predicted `live_field_goal`.
- [x] Merge those predictions with the existing LAC–DAL 9-positive/23-hard-negative plan;
      the combined screen is TP/FP/FN/TN `25/23/0/0`, P/R/F1 `0.5208/1.0000/0.6849`, so it is
      rejected and cannot be used as a promotion or runtime answer.
- [x] Seal plans, per-clip labels, predictions, evaluation and resource trace under
      `analysis_outputs/public_research/basketball_events_shot_vlm_v1/`; guard exit is 0 with
      zero stops. Update the source catalog to point to the bounded media and screen summary.
- [ ] Obtain rights-cleared, broadcast-diverse continuous ball–hand–rim outcomes and exhaustive
      non-shot hard negatives before changing defaults or resuming blind inference.

## Closed CHI–UTA media handoff cleanup (2026-08-06)

- [x] Verify the three legacy Internet Archive media files against the sealed pair handoff
      before deletion (`3,350,552,015` bytes).
- [x] Delete only the exact legacy benchmark/enrollment files; retain their state manifest,
      sizes and SHA-256 values for provenance.
- [x] Verify that blind media, current external screens, EBQwen native/MLX weights, `.venv`
      and runtime inputs are unchanged.
- [ ] Keep the cross-game causal blocker open; this cleanup is not a promotion or blind-recovery
      event.

## LAC–DAL EBQwen native-video resource probe (2026-08-06)

- [x] Derive a four-window native-video prefix from the frozen, label-hidden LAC–DAL plan;
      keep labels and review notes outside the model input.
- [x] Run the local EBQwen MLX 4-bit model with the pinned upstream revision, native temporal
      encoding, 2 FPS and a 151,200-pixel budget under the canonical `.venv` resource guard.
- [x] Delay truth opening and seal `1/3/0/0` (P/R=`0.25/1.00`) as an offline diagnostic;
      do not expand to 32 windows or change runtime/fusion defaults.
- [x] Correct an invalid first-run model-revision provenance value, delete its four invalid
      output/log/cache files, and retain only the re-run with manifest-bound provenance.
- [ ] Obtain rights-cleared continuous cross-broadcast causal outcomes and exhaustive hard
      negatives before any further VLM/detector promotion or blind-inference recovery.

## Online candidate gate: VC_NBA_2022 / FineSports / NSVA (2026-08-06)

- [x] Recheck primary paper/project pages and immutable repository state before any payload download.
- [x] Confirm VC_NBA_2022 is paper-described/commercial-source material without a public data
      archive or compatible dataset license; keep its nine shot/rebound labels as reference only.
- [x] Confirm FineSports requires a signed release agreement and has no separate code license;
      do not request or download the 10,000-clip release.
- [x] Confirm NSVA's CC BY-NC/data fair-use boundary; retain action lists/taxonomy only and do
      not duplicate-download raw NBA media or features.
- [x] Add `vc-nba-2022` to the fail-closed public source catalog, regenerate its SHA, update docs
      and wiki provenance, and keep runtime/weights/blind assets unchanged.
- [ ] Continue searching for rights-cleared continuous cross-broadcast ball–hand–rim outcomes
      with exhaustive non-ball hard negatives before any detector/VLM promotion or blind recovery.

## BQwen2.5-VL-3B BARD independent probe and cleanup (2026-08-06)

- [x] Pin the public CC BY 4.0 BQwen revision and download both BF16 shards with exact size/SHA
      verification; keep the source manifest fail-closed and outside AGU runtime.
- [x] Reuse only matching EBQwen/Qwen2.5-VL-3B processor files, convert under `.venv` resource
      guard to MLX 4-bit, and record the quantized weight hash before any evaluation.
- [x] Run four LAC–DAL native-video windows with raw clips only, 2 FPS, 151,200 pixels and
      delayed labels; BQwen is `1/3/0/0` (P/R=`0.25/1.00`), exactly matching EBQwen.
- [x] Delete the BF16 and MLX payloads after the failed independent gate; retain only manifest,
      compact configuration and offline prediction/evaluation/resource evidence.
- [ ] Continue only with rights-cleared, cross-broadcast sub-second ball–hand–rim outcomes and
      exhaustive non-ball hard negatives; do not change runtime/default VLM or resume blind inference.

## Basketball-51 bounded outcome sample and MViT screen (2026-08-06)

- [x] Audit the Kaggle card and extract only 8 labels × 4 source-game tokens with ZIP64 Range;
      verify CRC/size/SHA and ffprobe, retaining 19,867,852 bytes under the research boundary.
- [x] Extract canonical MViT-V2-S embeddings in `.venv`; field-goal vs free-throw worst-fold
      P/R=`0.8571/1.00`, but made-vs-missed worst-fold P/R/F1=`0.50/0.25/0.3333`.
- [x] Run the guarded 3-epoch one-block source-domain tail fine-tune; held `v112` P/R/F1=`0/0/0`,
      guard exit 2, and no checkpoint was emitted.
- [x] Keep only bounded clips, embeddings, screen/resource traces and rejected metadata; remove
      the derived contact sheet and keep all artifacts out of runtime/defaults/blind inference.
- [ ] Obtain rights-cleared cross-broadcast sub-second ball–hand–rim outcomes and exhaustive
      non-ball hard negatives before any shot-outcome promotion or blind recovery.

## F-16-NBA shot-test online candidate gate (2026-08-06)

- [x] Pin the Hugging Face revision and audit the Apache-2.0 card, 1,970 shot `Yes/No` clips,
      16 source games, JSON hash and three tar-shard object IDs.
- [x] Use one bounded Range prefix only; extract and ffprobe eight 1280×720 H.264/60-fps clips,
      and seal their hashes after the labels were kept out of any model input.
- [x] Reject further import: the candidate lacks sub-second ball–hand–rim timing, exhaustive
      non-shot hard negatives and an explicit underlying NBA broadcast redistribution grant;
      delete the temporary prefix and keep only compact provenance.
- [ ] Obtain a rights-cleared continuous cross-broadcast source with causal outcomes before
      attempting another shot-outcome model or blind recovery.

## Basketball-51 expanded source-group screen (2026-08-06)

- [x] Extract a fixed-seed 16-per-class extension from the same Kaggle ZIP via HTTP Range;
      verify 128 clips, 46 source groups, `80,328,593` bytes, CRC/size/SHA and ffprobe.
- [x] Reuse the canonical `.venv` MViT extractor and seal the source-group screen: field-goal vs
      free-throw worst-fold P/R/F1=`0.9565/0.9565/0.9565`, while made/missed is
      `0.5455/0.375/0.48` and eight-class macro-F1=`0.1667`.
- [x] Keep the expanded sample and embeddings only for the narrow free-throw auxiliary screen;
      skip a second tail fine-tune and emit no checkpoint because outcome transfer still fails.
- [ ] Obtain rights-cleared cross-broadcast sub-second ball–hand–rim outcomes and exhaustive
      non-ball hard negatives before any shot-outcome promotion or blind recovery.

## Basketball_Detection static candidate gate (2026-08-07)

- [x] Pin the GitHub repository commit and inspect the complete tree plus `dataset.yaml` without
      downloading image bytes; record split counts, classes and the absent license file.
- [x] Add the fail-closed source manifest/audit and catalog regression assertion; regenerate the
      public research catalog (`54` sources, SHA `bd18e4fbd6cd344994b726f94b39562b6326fed9c81e203c5788fdb7cfa92339`).
- [x] Reject import because the source is static image detection with no temporal causal labels,
      exhaustive hard negatives or underlying media redistribution terms; retain manifest/audit
      only and leave runtime, weights and blind assets unchanged.
- [ ] Continue only when a rights-cleared, broadcast-diverse ball–hand–rim temporal source is
      available; do not use this candidate for detector training or blind recovery.

## Qwen3-VL-4B resource-fit screen and cleanup (2026-08-07)

- [x] Pull the official Apache-2.0 Ollama `qwen3-vl:4b` asset and bind its model revision and
      weight SHA to the frozen label-hidden LAC–DAL 32-window plan.
- [x] Start the strict-release 4-frame/512-pixel probe under the canonical `.venv` resource guard;
      the guard stopped before the first prediction after three consecutive samples below the
      2.5 GiB available-memory floor (exit code 75).
- [x] Seal the zero-window screen and raw guard trace as offline-only evidence, then remove the
      3.3 GB local Ollama payload; leave Qwen3-VL-2B, EBQwen native/MLX, runtime and blind assets
      unchanged.
- [ ] Continue only with a rights-cleared, broadcast-diverse temporal source containing causal
      ball–hand–rim outcomes and exhaustive non-ball hard negatives before another VLM screen or
      blind-inference recovery.

## SportsShot online candidate gate (2026-08-07)

- [x] Pin the official Hugging Face revision and inspect its card, declared CC BY-NC 4.0 license,
      1,200-video/183 GB scope and shot-segmentation annotation contract.
- [x] Probe an annotation endpoint without credentials; record the `401 GatedRepo` boundary and
      download no media or annotation archive.
- [x] Add the fail-closed `sportsshot` catalog entry and compact audit; regenerate the catalog to
      55 sources with SHA `0dd4e24c7502a22775d3110e1fcf3d0d83849064a586db2698c005a16fbd4eb6`.
- [ ] Continue only with an explicitly authorized source that adds causal ball–hand–rim outcomes
      and exhaustive hard negatives; do not use shot-boundary labels for AGU runtime promotion.

## strict-release-v3 evidence consistency screen (2026-08-07)

- [x] Add the label-free `strict_release_v3` prompt with ordered control/release/toward-rim
      evidence and an explicit contradiction check; lock it with RED→GREEN tests.
- [x] Complete the frozen LAC–DAL 32-window Qwen3-VL-2B run under the canonical `.venv` resource
      guard; delayed labels and provenance stayed outside model input.
- [x] Seal raw evaluation (`9/23/0/0`, P/R/F1=`0.2813/1.0000/0.4390`) and the conservative
      offline consistency audit (`9/20/0/3`, P/R=`0.3103/1.0000`); reject the prompt for
      promotion and keep it out of runtime/training truth.
- [x] Verify guard/hash/diff state and retain only compact predictions, evaluation, consistency
      audit and resource trace; no new checkpoint or model payload was produced.
- [ ] Continue only with a rights-cleared, broadcast-diverse temporal source containing causal
      ball–hand–rim outcomes and exhaustive non-ball hard negatives before another model change.

## NUS Basketball Detection online candidate gate (2026-08-07)

- [x] Pin Hub revision `3ee1a0decfde199117b3a99d79bcf136c902ebc5`; inventory 719 MP4 clips and
      class/byte totals without downloading the full 2.97 GB release.
- [x] Download a fixed 12-clip, 37,378,249-byte sample; verify every size/SHA against Hub LFS and
      decode all files with ffprobe; inspect contact sheets for shot/background content.
- [x] Reject import: no README, license, card, source-game provenance, frame-level causal labels,
      or exhaustive non-shot hard negatives; folder paths also leak labels.
- [x] Move the sample to `/Users/ppt/.Trash/agu-nus-basketball-20260807`; retain only
      `analysis_outputs/public_research/nus_basketball_detection_audit_v1.json` and the 56-source
      fail-closed catalog entry (internal SHA
      `f947fb5c702aa36de6ec631ddd5fca32efb63f6435e893860f85885e0cbc94ca`).
- [ ] Continue only with a rights-cleared, broadcast-diverse temporal source containing causal
      ball–hand–rim outcomes and exhaustive non-ball hard negatives; do not use this candidate for
      training, runtime promotion or blind recovery.

## PL-NBA possession-level temporal annotation gate (2026-08-07)

- [x] Pin `holhouse/PL-NBA-Dataset` main commit `0b3d5013b5828004062ed27b10c9eddfca313711` and
      download only the 4,461,082-byte `PL-NBA-JSON.zip` annotation archive; do not fetch original
      NBA games or the separately hosted trimmed videos.
- [x] Verify archive SHA/CRC and parse all 6,408 possession JSON files: 32 games, 31,964 events,
      3,607 shot events, outcome labels and temporal spans; record four reversed intervals, four
      `nan` rows and filename/possession-id label leakage.
- [x] Reject media/training/runtime import because the archive has no paired video, no independently
      verified NBA broadcast rights, no pixel ball–hand–rim causality and no exhaustive full-game
      hard negatives; retain the compact annotation archive, manifest and audit only.
- [x] Register `pl-nba-v1` as `runtime_consumable=false` and `training_media_eligible=false`, add
      catalog regression coverage and regenerate the 57-source catalog (internal SHA
      `93542b43e62169f7c25946f3dc910fbb46f9cfe8f76f643247566f88d03efb9f`).
- [ ] Continue only when a rights-cleared, paired continuous-video source adds sub-second causal
      ball–hand–rim outcomes and exhaustive non-shot hard negatives; do not use PL-NBA annotations
      as runtime answers or blind labels.

## TAL3x3 temporal action annotation gate (2026-08-07)

- [x] Pin `open-starlab/TAL3x3` commit `f1093d62818c3e8930561fda5939db19c0981d95`; inspect the
      repository and public Drive inventory without downloading video or the separate skeletons.zip.
- [x] Download and SHA/CRC-verify only the 5,094,624-byte `dataset.zip`; parse 318 clips from 10
      3x3 source videos, 1,881 frame-bounded events, 387 shot/free-throw outcome events and 105,589
      frames of normalized person boxes.
- [x] Reject runtime/media training import: no LICENSE or dataset rights statement, no paired video,
      ball boxes/causal ball–hand–rim order or exhaustive full-game hard negatives; paper CC BY 4.0 is
      not assumed to license the underlying archive/source video. Retain annotation-only reference.
- [x] Register `tal3x3-v1` as `runtime_consumable=false` and `training_media_eligible=false`, add
      catalog regression coverage and regenerate 58 sources (internal SHA
      `460aca7b4a3f72d33666776dcc371854a2741d21290f176dfcf27b072836c5f4`).
- [ ] Continue only with an explicitly licensed paired continuous 5x5 video source containing
      sub-second causal ball–hand–rim outcomes and exhaustive non-shot hard negatives.

## MUVY multi-view basketball auxiliary sample gate (2026-08-07)

- [x] Pin Zenodo record `10.5281/zenodo.13883315` (2024-10-03, CC BY 4.0) and inspect the
      7,758,643,869-byte ZIP central directory without downloading the full archive.
- [x] Extract 26 basketball metadata files plus the two `basketball_event_01` camera videos via
      bounded ZIP64 Range requests; verify SHA/size/ffprobe and retain 34,211,695 media bytes.
- [x] Audit object-box coverage (232/48 ball rows), highlight/title leakage and the diagnostic
      32.7-second audio-energy offset; keep the result out of runtime/training truth.
- [x] Register `muvy-basketball-v1` with `runtime_consumable=false` and
      `training_media_eligible=false`; regenerate 59-source catalog with internal SHA
      `d9b5b85feedf1309e7b5aa0d2c9b2bc978b3e93fee88ff63eb29133b61ba18c2`.
- [ ] Continue only when an explicitly authorized, broadcast-diverse continuous 5x5 source adds
      sub-second causal ball–hand–rim outcomes and exhaustive non-shot hard negatives; do not
      turn MUVY highlight titles or audio alignment into labels.

## Basketball Events annotation re-audit (2026-08-07)

- [x] Re-pin `saveerjain/basketball-events` revision `26d3775286f542b41daf4a94a53190440d426111`;
      inventory the 564-entry/543-MP4 upstream tree without downloading the 7.86 GB media set.
- [x] Reconcile four game JSONs (543 clips/897 events), shot results and 58 duplicate event tuples;
      verify 776/897 integer clock-index matches and quantify event-centered coverage.
- [x] Audit train/val path versus game overlap: paths are disjoint but all four games are shared, so
      the split is not an independent broadcast/game gate; confirm no frame-causal labels or exhaustive
      non-event hard negatives.
- [x] Hash and ffprobe eight bounded sample clips (8/8 decode), then permanently clean duplicate
      metadata/media and stale rejected candidate copies. Keep the existing bounded manifests and the
      re-audit `analysis_outputs/public_research/basketball_events_reaudit_v1.json` (SHA
      `f4e19cbf3958fb5ee9543943ecee97abc75165c2affc52df4ee0d1c93943b776`).
- [ ] Continue only if an explicitly rights-cleared, broadcast-diverse continuous 5x5 source adds
      sub-second causal ball–hand–rim outcomes and exhaustive non-shot hard negatives; keep this source
      annotation-only and blind reasoning sealed.

## NBA Games full-game video/PBP index gate (2026-08-07)

- [x] Pin `choucsan/NBA_Games` revision `cf59e3a42413e3ab91f6fc0b1618f280df9a7024`; retain the
      189-game index and structured metadata under its declared MIT metadata license, without downloading
      linked YouTube videos.
- [x] Parse 5,194 box-score rows and 81,355 PBP rows: 166 games have non-empty PBP, 23 are empty;
      record 12,659 made-shot and 15,030 missed-shot action rows plus the `videoAvailable` flag boundary.
- [x] Reject runtime/training import: no local video, independently verified underlying-video rights,
      frame-level causal timing, exhaustive non-shot negatives or cross-broadcast split. Keep manifest and
      `analysis_outputs/public_research/nba_games_fullgame_audit_v1.json` (SHA
      `7fc8df8aeb002962743f08bed240c27890f31e714183d699f57ae954bee9091a`) as alignment reference.
- [ ] Continue only if a rights-cleared continuous 5x5 video source can be paired to the PBP index and
      independently verified for sub-second ball–hand–rim outcomes and hard negatives.

## Independent VLM evidence-gate integrity hardening (2026-08-07)

- [x] Make the offline evidence gate fail closed when any evaluated VLM event lacks an auxiliary OOF
      row; preserve extra training rows only for held-group threshold selection.
- [x] Reject duplicate required observables and non-finite/out-of-range auxiliary probabilities;
      abstain invalid live VLM confidence as `unknown`.
- [x] Add regression tests for coverage, numeric bounds and duplicate observables; run targeted Ruff,
      canonical `.venv` full pytest (`913 passed, 15 warnings`), Harness, compile and diff checks.
- [x] Record that the existing EBQwen + Swin/MViT screen remains `13/9/3/7`,
      P/R/F1 `0.5909/0.8125/0.6842`, `promotion_eligible=false`; no runtime or blind boundary changed.
- [ ] Continue only with rights-cleared, broadcast-diverse continuous 5x5 causal data and exhaustive
      non-shot hard negatives; this code hardening does not resolve the data gate.

## Shot-causal support threshold protocol correction (2026-08-07)

- [x] Identify and reproduce the v1 defect where the causal threshold was selected from a zero-filled
      outer training score buffer.
- [x] Implement nested leave-one-training-game-out causal OOF scores and fail-closed regression coverage
      in `app/analysis/shot_causal_support.py` and `tests/test_shot_causal_support.py`.
- [x] Seal `analysis_outputs/public_research/shot_causal_support_screen_v2.json`; corrected causal
      support is P/R/F1 `0.557562/1.000000/0.715942`, weakest-game precision `0.414634`,
      `accepted=false`.
- [ ] Continue with a new cross-broadcast ball/non-ball evidence increment; do not promote this screen,
      alter runtime defaults, or use the historical v1 artifact for decisions.

## HOU–SAC tiled ball candidate VLM probe (2026-08-07)

- [x] Build a label-hidden 48-candidate plan from the existing HOU–SAC tiled perception artifact and
      seal Codex raw-frame decisions (`11 valid_ball / 2 uncertain / 35 false_positive`).
- [x] Run 12 local EBQwen MLX-4bit candidate probes under the canonical `.venv` resource guard; all
      12 returned `uncertain`. Resource envelope: no stop/retry, RSS peak ~584 MiB, system memory peak
      `86.6%`, CPU peak `33.4%`, minimum available memory ~`2.15 GiB`.
- [x] Seal the evaluation; lower/upper precision and recall are `0/0`, `accepted=false`. Keep all
      candidate/VLM artifacts offline-only and do not change detector or runtime defaults.
- [ ] Replace the rejected candidate-VLM route with a rights-cleared, broadcast-diverse continuous
      5×5 ball/non-ball and ball–hand–rim source; the data gate remains the active blocker.

## E-BARD candidate verifier transfer to HOU–SAC (2026-08-07)

- [x] Reuse the sealed 3,675-row E-BARD manifest and HOU–SAC 48-candidate plan without downloading
      new model or dataset bytes.
- [x] Run paired MobileNetV3-small exact-crop screens with and without detector confidence under the
      canonical `.venv` resource guard and 5-fold game-held OOF threshold selection.
- [x] Seal both artifacts: confidence P/R lower/upper `0.003423/0.245232` and `0.004259/0.267303`;
      visual-only lower/upper `0.303030/0.408719` and `0.347475/0.410501`; both `accepted=false`.
- [ ] Stop threshold-only verifier iteration and obtain a rights-cleared, broadcast-diverse
      continuous 5×5 ball/non-ball source with sub-second ball–hand–rim outcomes and exhaustive
      non-shot hard negatives.

## MUVY full standardized subset cross-event ball review (2026-08-07)

- [x] Reconcile the newly extracted ZIP64 subset against the existing standardized
      `dataset/public_sources/muvy_basketball_v1` by file size and SHA-256; delete the 41-file
      duplicate extraction and retain the canonical 39-file/13-video subset.
- [x] Run complete ffprobe/frame-count and annotation inventory checks over all 13 videos and five
      events: 513 `sports ball` rows, no frame-count mismatches, and no hand/rim/outcome/continuous
      5×5 labels.
- [x] Review all 373 geometry candidates from hash-bound contact sheets; seal `74 valid_ball`,
      `5 uncertain`, `294 false_positive` decisions and materialize a 74-image/74-box offline YOLO
      auxiliary split (57 train/17 validation, validation event held out).
- [x] Seal `analysis_outputs/public_research/muvy_basketball_cross_event_review_v1.json` with
      artifact SHA `430ea5e5951b1eef6312423b1cbe1fa98dcaadf2e9770424653d2fc2ddf1c49d`; keep the
      auxiliary set offline-only and do not count it as causal shot-outcome truth.
- [ ] Use the MUVY positives only for auxiliary detector/hard-negative experiments after a new
      cross-broadcast source is found; the active promotion gate still requires licensed continuous
      5×5 ball–hand–rim outcomes and exhaustive non-shot hard negatives.

## E-BARD+MUVY v2 detector short fine-tune and independent screen (2026-08-07)

- [x] Add a sealed-manifest compatibility regression for the 74-box MUVY v2 auxiliary set while
      preserving the explicit historical 169-image v1 allow-list; focused TDD tests pass.
- [x] Materialize `dataset/public_sources/e_bard_muvy_ball_v2` from 1,800 E-BARD frames plus 74
      reviewed MUVY boxes. Keep the combined manifest `runtime_consumable=false` and record artifact
      SHA `cf51edd127d7ed9f8aea22e9bac05af18e8ffa458e715755b79d2a0a0366d0aa`.
- [x] Run a guarded five-epoch YOLOv8n MPS fine-tune under `.venv`; internal validation reached
      P/R/mAP50 `0.747/0.591/0.649`. The run completed with no resource-guard stop, and its model
      hashes were sealed before cleanup.
- [x] Fix the external screen's sealed-plan compatibility bug so legacy `frame_sha256` detector
      records are not mistaken for OpenCV pixel hashes; add regression coverage.
- [x] Screen the checkpoint on two disjoint reviewed broadcasts: LAL–BOS P/R/F1
      `0.352941/0.240000/0.285714` and ATL–CHI `0.090909/0.172414/0.119048`. Both are
      `accepted=false`; audit `analysis_outputs/public_research/e_bard_muvy_ball_v2_detector_screen_v1.json`
      has internal SHA `813b0f5ecbcda30513fa15a9d01137da977f15e9b8f95ae76804e8ae005dd655`.
- [x] Delete the unpromoted `best.pt` and `last.pt` checkpoints after sealing their SHA-256 values;
      retain only compact results, resource logs, predictions and audit metadata.
- [ ] Do not retry another detector-only fine-tune until a rights-cleared, broadcast-diverse
      continuous 5×5 source provides sub-second ball–hand–rim outcomes and exhaustive non-shot
      hard negatives; no runtime, default VLM, EBQwen or blind asset changed.

## OKC–CLE cross-broadcast development screen (2026-08-07)

- [x] Download the 640×360/29.97-fps OKC–CLE stream only after metadata/format verification; seal
      source SHA `e9955b698de9894276de90a0f79ff2dba21322ae04476287103bc55accb5414a`, then run 11
      decode spot checks.
- [x] Review 32 label-hidden uniform windows under the strict live-release protocol. The result is
      `0/32` qualifying positives because windows were no-release, free-throw, replay, stoppage or
      postgame segments.
- [x] Seal `analysis_outputs/public_research/okc_cle_2025_cross_broadcast_screen_v1.json` (file SHA
      `5de8f68f965a46e8b11f8b2a03360262d60ff694f2081c6e`) and delete the 399 MB video after the
      negative-only screen; retain only the ~2.4 MB hash-bound audit package.
- [ ] Do not treat this negative-only screen as training truth or a model increment. Resume only with
      a rights-cleared, broadcast-diverse continuous 5×5 source containing sub-second ball–hand–rim
      outcomes and exhaustive non-shot hard negatives.

## Current readiness audit (2026-08-08)

- [x] Seal the cross-experiment readiness audit at
      `analysis_outputs/public_research/agu_readiness_audit_2026-08-08.json` and bind the
      corrected causal, detector-transfer, OKC–CLE, and official-count evidence to it.
- [x] Confirm the canonical local environment is `.venv` Python 3.11.15, that `venv` is absent,
      and that no AGU training/runtime process is active.
- [x] Keep blind inference paused and preserve both EBQwen native/MLX assets; no runtime/default
      VLM, detector, training truth or blind asset changed.
- [x] Build and seal the OCR-clock aligned, training-only cross-game review at
      `analysis_outputs/public_research/pbp_clock_training_review_audit_v1.json`: 48 windows per game,
      conservative visual labels `20/48` (LAL–BOS) and `17/48` (ATL–CHI); delete superseded interpolation
      reviews/montages. No training or model/runtime promotion was started because the traditional
      feature preflight was all-zero and continuous ball–hand–rim supervision is absent.
- [ ] Resume only after a rights-cleared, broadcast-diverse continuous 5×5 source supplies
      sub-second ball–hand–rim–outcome labels and exhaustive non-shot hard negatives, then require
      independent per-game P/R >= 0.85.

## SVI-Bench gate and E-BARD team-color auxiliary (2026-08-08)

- [x] Pin the SVI-Bench Hub API at revision `aee244344c6ad5bcf0caee298ed53daecc4cff4a`; record the
      manual `.edu` gate, no-redistribution terms and HTTP 401 payload probes in
      `analysis_outputs/public_research/svi_bench_online_gate_v1.json` (artifact SHA
      `3c1d46d72b29cdd138bfcab85939ab45932d922370d47906ef3ec1de466c722b`).
- [x] Download the CC-BY-4.0 E-BARD Team Attribution archive, verify ZIP integrity and SHA
      `f25c2bb8d8527992e68fc952d840f6c379b4ce24412d0dd764bdcc9c95134be7`, and register its
      `runtime_consumable=false` / `training_media_eligible=false` boundary in the source catalog.
- [x] Run a group-held RGB/HSV baseline: accuracy `0.783279`, macro-F1 `0.732430`; retain the
      15,295 color crops as offline auxiliary only because labels are not team/player/event truth.
- [ ] Do not download SVI-Bench payloads or promote the color baseline without approved access,
      causal-label audit and independent broadcast P/R >= 0.85.

## APIDIS multi-view/event alignment restoration and detector screen (2026-08-08)

- [x] Restore only the 52 hash-bound APIDIS entries through the public file endpoint; do not fetch the
      28.56 GB archive; verify the seven 800×600/25 FPS/60-second AVIs.
- [x] Implement and test explicit `184700+02` to `164700Z` frame mapping and event-XML extraction;
      seal 4,219 mapped ball frames and five Q2 event times.
- [x] Run the camera-disjoint five-epoch YOLOv8n screen under `.venv` resource guard and evaluate the
      held camera plus independent LAL–BOS/ATL–CHI reviews.
- [x] Seal results/resource/checkpoint hashes and delete generated frames/caches/weights. Keep
      `runtime_consumable=false`, `accepted=false`, and no detector/VLM/runtime/blind promotion.
- [ ] Resume only after a rights-cleared, broadcast-diverse continuous 5×5 source provides sub-second
      ball–hand–rim–outcome labels and exhaustive non-shot hard negatives; then require independent
      per-game P/R >= 0.85 before blind inference.

## Independent Qwen3-VL negative-first prompt screen (2026-08-08)

- [x] Add the label-hidden `negative_first_v4` prompt and bind it to the runner cache fingerprint;
      require the visible live-play → control → release → rim-direction chain and fail closed on any
      missing/ambiguous link. Targeted prompt/cache regression passes.
- [x] Derive and seal the 4-frame/256-pixel plan
      `analysis_outputs/public_research/independent_shot_vlm_development_plan_compact256_v1.json`
      (plan SHA `ab36c2ec4eb66a0f431a49a6230e28a3f1947fbb2a7a748a6a37306009cf6b7c`); run local
      `qwen3-vl:2b` with context 3072 under the canonical `.venv` resource guard. Three low-memory
      stops were resumed safely; all 32 windows eventually completed.
- [x] Delay label access until evaluation and seal predictions/evaluation/resource evidence. The
      result is TP/FP/FN/TN `16/16/0/0`, pooled and per-game P/R `0.500000/1.000000`,
      `promotion_eligible=false`; prediction/evaluation internal SHAs are
      `fd8230aae39bb6b92b3b67eeeb985d08e3a609612cd5271068f02b04b80d7c4a` and
      `9dbd7094fd5743727f2c420fd52805629009f876630de95533a586b6cb648546`.
- [x] Delete only the resumable/intermediate VLM caches and failed context probe outputs after SHA
      capture; retain the compact plan, prediction, delayed evaluation and resource log. No runtime,
      default VLM, detector, EBQwen weight, training truth or blind asset changed.
- [ ] Stop further prompt-only/base-VLM fusion iteration until a rights-cleared, broadcast-diverse
      continuous 5×5 source supplies sub-second ball–hand–rim–outcome labels and exhaustive non-shot
      hard negatives; then require independent per-game P/R >= 0.85 before blind inference.

## Online continuous-causal source re-audit (2026-08-08)

- [x] Recheck BASKET, NBA Games, MUVY and BASKET-Multiview against the active source gate using
      official project pages/papers; distinguish player-highlight skill labels, metadata-only PBP,
      spatial user-video detections and synthetic short plays from real continuous broadcast truth.
- [x] Do not download new payloads: none simultaneously supplies rights-cleared continuous 5×5 video,
      sub-second ball–hand–rim–outcome labels and exhaustive non-shot hard negatives. Keep only the
      existing metadata/compact audits and leave runtime, detector, EBQwen and blind assets unchanged.
- [ ] Resume source acquisition only when an upstream release or explicit rights/label grant clears all
      three conditions; then require independent per-game P/R >= 0.85 before blind inference.

## Independent VLM + frozen scene/video OOF gate (2026-08-08)

- [x] Add and test a label-free fusion path that applies only the stored game-held auxiliary thresholds;
      do not fit on target rows, reopen target labels, or allow missing OOF scores to become positives.
- [x] Run it on the 32-window negative-first Qwen3-VL screen. Coverage is 31/32; the missing event is
      `unknown`. Delayed evaluation is `TP/FP/FN/TN=10/6/6/10`, unknown `16`, pooled P/R
      `0.625/0.625`, worst-game P/R `0.400/0.500`, `promotion_eligible=false`.
- [x] Keep the prediction/evaluation artifacts offline-only and leave runtime/default VLM, detector,
      EBQwen, blind inference and readiness unchanged.
- [ ] Do not continue prompt/threshold-only iteration on this 32-row set; resume after a rights-cleared,
      cross-broadcast continuous 5×5 causal source with exhaustive hard negatives is available.

## Continuous causal source gate (2026-08-08)

- [x] Add a fail-closed metadata-first gate requiring explicit rights, broadcast diversity, continuous 5×5
      video, sub-second ball–hand–rim–outcome labels, exhaustive non-shot hard negatives, immutable URL/
      revision and declared size.
- [x] Audit nine current candidates without opening raw-media payloads; seal
      `analysis_outputs/public_research/continuous_causal_source_gate_2026-08-08.json` with internal audit
      SHA `29b3b9190f4ad9bdd56a73dde926190d9cfdc4f0c40b01e2e50d8543b81af607`. The result is
      `eligible_source_count=0`, `payload_downloads_performed=0`; NSVA and GCB are metadata-only.
- [ ] Resume acquisition only when one source returns `download_decision=allow`; keep all current sources
      metadata-only and do not alter runtime/default VLM, EBQwen, blind assets or acceptance truth.

## NSVA event ontology metadata increment (2026-08-08)

- [x] Pin and download only the two NSVA subset Parquet tables (23,167 bytes; 1,316 rows) at revision
      `97c211f85beec54209b44faea2bff73544a16125`; verify atomic writes, per-file hashes and manifest hash.
- [x] Add the TDD RED/GREEN intent parser and sealed audit with field-goal/free-throw, rebound, foul,
      turnover, jump-ball, period-boundary, violation and ejection kinds; unknown segment count is zero.
- [x] Inspect the upstream downloader source in a bounded archive; do not execute its unbounded raw-video
      collector. Keep both NSVA artifacts `runtime_consumable=false` and `training_media_eligible=false`.
- [ ] Do not use NSVA metadata as video training truth or runtime evidence; only reconsider after a separate
      rights-cleared continuous 5×5 source supplies sub-second causal labels and exhaustive hard negatives.

## GCB/GameCommBench event metadata increment (2026-08-08)

- [x] Pin GCB revision `728f3a67839c10c54a2aba792d8659f94b8fa6ad` and download only the 6,519,620-byte
      basketball metadata JSONL plus 2,045-byte dataset summary; do not download the declared 21.761 GB
      video payload (`license=other`).
- [x] Add and test the event audit and downloader: composite labels, period/timeout/technical source text,
      source outcome/game scope, traversal-safe resolution, atomic writes and self-verifying hashes all pass.
- [x] Seal 2,981 rows across 132 games (`unknown_segment_count=0`) in
      `analysis_outputs/public_research/gcb_basketball_event_audit_v1.json` (internal SHA
      `dae4b926ae034c885cf497569a16066512c0dbeb82d1deb56565701a1a11d0c4`) and register the source in the
      nine-candidate gate. Keep it `runtime_consumable=false`, `training_media_eligible=false` and
      `media_downloaded=false`.
- [x] Regenerate `analysis_outputs/public_research/source_catalog.json`: 62 sources, internal SHA
      `4921a2cdc65d4b30b2c3471c1dd6064f30fb8cde83d5435c8ac17b3122ec270f`.
- [x] Recheck the two unresolved label-hidden visual-state sheets and seal
      `analysis_outputs/public_research/pbp_visual_state_codex_followup_v2/manual_recheck_v1.json`
      (internal SHA `b4fe543ea6e1849b17d72a483df91c4ebbf09e9835e0adb12f4df4d66a2d80af`); no labels are
      promoted to training truth.
- [x] Replay the frozen AGU v3 + independent VLM `both_confirm` rule under `.venv`; the 32-row artifact
      hash matches `8ee0dfc577641e30ef67442ed1cc820eeb1fde025b6365aaf0138bb4772d2491`, max RSS is
      339,755,008 bytes, and pooled/per-game P/R remains `0.000000/0.000000`.
- [ ] Do not use GCB metadata as video training truth or runtime evidence; resume acquisition only after a
      rights-cleared continuous 5×5 source supplies sub-second causal labels and exhaustive hard negatives.

## Obsolete rendered-review cleanup (2026-08-08)

- [x] Scan current code, tests, docs, task records and compact evidence for references to the four
      completed `ball_release_hard32*_review_sheets_v1/` directories.
- [x] Seal per-file SHA-256/size evidence in
      `analysis_outputs/public_research/obsolete_review_sheet_cleanup_2026-08-08.json`, then delete
      the 260 unreferenced rendered JPGs (`158,086,309` bytes); retain compact decisions, labels,
      screen artifacts and source videos.
- [x] Remove only the re-creatable root `.ruff_cache` and stale Hugging Face `.lock` files under the
      two complete EBQwen weight trees; preserve the weights and metadata caches needed for resume.

## Hugging Face basketball directory sweep (2026-08-08)

- [x] Query the latest-updated Hugging Face `basketball` dataset index with a bounded 100-result
      request; seal 72 returned dataset IDs and their immutable revisions/metadata.
- [x] Reconcile 9 existing catalog matches and triage 11 high-relevance entries without opening
      dataset payloads; the new event mirror is a duplicate of the already audited research-only
      Basketball Events release, while robotics/tabular/player-detection entries lack continuous
      5×5 causal supervision.
- [x] Seal `analysis_outputs/public_research/hf_basketball_directory_sweep_2026-08-08.json` with
      `payload_downloads_performed=0` and internal audit SHA
      `8deef5dcbfaffcb8cc0c97a965d97b725adb27ff94262595009c8cad72b0f587`; no candidate clears the
      five-check causal gate.
- [ ] Keep the 61 unreviewed directory-only entries metadata-only; reopen only when a candidate's
      card provides explicit rights, continuous 5×5 broadcast coverage, sub-second causal labels
      and exhaustive non-shot hard negatives.

## PBP/OCR causal coverage diagnostic (2026-08-08)

- [x] Aggregate the two sealed enrollment PBP-clock alignments without reopening target labels or
      reading blind media; bind each input file SHA and alignment artifact SHA.
- [x] Quantify event/shot coverage and mapping bias: pooled event coverage `341/873=0.390607`,
      field-goal coverage `163/300=0.543333`, made-shot coverage `0.823129` versus missed-shot
      coverage `0.274510` (difference `0.548619`).
- [x] Seal `analysis_outputs/public_research/pbp_causal_coverage_audit_2026-08-08.json` with
      internal audit SHA `9d0b0313526378f293e1e8c3b01555e514837bbcc56d637ed92250485d55d0df`;
      keep `runtime_consumable=false`, `training_media_eligible=false` and `accepted=false`.
- [ ] Do not promote OCR-clock anchors to causal training labels; obtain a rights-cleared continuous
      5×5 source before another model-training or blind-inference attempt.

## UVY basketball detector auxiliary subset (2026-08-08)

- [x] Pin the Zenodo UVY record `10.5281/zenodo.21303900`, confirm its CC-BY-4.0 terms and inspect
      the ZIP central directory without downloading the 3.27 GB archive.
- [x] Add the bounded Range extractor and TDD contracts; extract only the four basketball image
      prefixes plus `video_info.txt`, `labels.txt` and `gt.txt`, verify source CRCs and exclude every
      MP4/non-basketball entry.
- [x] Audit image-frame/annotation coverage, class and track counts, frame-count mismatches and the
      absence of shot-outcome labels. Keep the result `runtime_consumable=false` and scope any media
      training to `auxiliary_detector_and_hard_negative_only`.
- [x] Run a canonical `.venv` CPU transfer screen on 120 held-out V04 frames. Generic YOLOv8n finds
      zero of 73 ball targets (`TP/FP/FN=0/0/73`, P/R/F1 `0/0/0`, max RSS `410,615,808`); seal the
      screen SHA `85393624ef72c7d9d84dff689e0444a34ffed389975234fcfe92735001f114d1`, then remove the
      unpromoted generic weight.
- [ ] Do not promote a checkpoint or alter the official detector until an independent cross-broadcast
      game-held screen clears the existing per-game precision/recall gate; UVY remains auxiliary only.

## UVY auxiliary detector training screen (2026-08-09)

- [x] Add the fail-closed UVY training-plan/result schema, absolute-path YOLO data binding, CLI and
      four TDD contracts. Keep all plans/results `runtime_consumable=false`,
      `causal_truth_eligible=false`, `promotion_eligible=false` and outside Codex runtime answers.
- [x] Run a bounded canonical `.venv` CPU screen from `model_checkpoints/yolov8n.pt` (1 epoch,
      `imgsz=256`, `batch=2`, `workers=0`) on V01/V02/V04 train/val/test splits under the resource
      guard. Seal plan SHA `8f5d307eddc5affa3576067e3813a770b0aec57c0e1b8d8105cc16644ce2987d`,
      result SHA `2dfe3f1ee7ff089f0021be338484131726f1f2d4476c8da47d2c823962203e36`, and guard
      child-exit log SHA `247754f6980e8a308416339b9afcb9d5c45f5a6cef09ecec86d62059974103f3`.
- [x] Screen 120 independent V04 test frames (72 ball-target frames): the trained best checkpoint
      still yields `TP/FP/FN=0/0/72`, P/R/F1=`0/0/0`. The guard exits 0 with peak system memory
      `84.2%`, peak process-tree RSS `546,029,568` bytes, minimum available memory
      `2,711,584,768` bytes and peak CPU `50.0%`; delete the sealed, unpromoted best/last files and
      generated label caches after recording their hashes.
- [ ] Do not promote UVY or alter the official detector; require an independent cross-broadcast
      game-held screen and the existing per-game precision/recall gate before any detector change.

## Online candidate supplement (2026-08-09; metadata-only)

- [x] Recheck the official [NBA Games](https://choucisan.github.io/collections/nba_games/) index, the
      [BasketHAR](https://huggingface.co/datasets/Xian-Gao/BasketHAR) card and the
      [BasketLiDAR project](https://sites.google.com/keio.jp/keio-csg/projects/basket-lidar) for rights,
      continuous broadcast coverage, causal labels and access conditions.
- [x] Download no new payload: NBA Games remains metadata/video-reference only, BasketHAR is inertial
      signal data without video, and BasketLiDAR is request-only/institutional. Keep all three outside
      runtime and training truth; the five-check source gate remains at zero eligible candidates.
- [ ] Resume bounded media acquisition only when an independent source provides explicit rights,
      continuous 5×5 broadcast frames, sub-second ball–hand–rim–outcome labels and exhaustive hard negatives.

## Follow-up: high-dimensional broadcast-state fusion screen (2026-08-09)

- [x] Re-run the frozen 511-window `base+broadcast_raw` outer game-held screen with PCA36,
      preserving the label-free feature contract and the canonical `.venv`.
- [x] Seal `analysis_outputs/public_research/shot_broadcast_fusion_full511_batch4_pca36_c0p01_screen_2026-08-09.json`
      (internal SHA `824d518806344a0ea47e21564f13ec9882ddb4b0fb15e46ca4adae4d16194ca1`) with
      pooled P/R/F1 `0.787162/0.943320/0.858195` and peak RSS `729,186,304` bytes.
- [x] Reject promotion because the weakest-game precision is `0.730337`; do not modify runtime,
      detector, VLM, EBQwen or blind assets.
- [ ] Keep the rights-cleared continuous 5×5 causal/hard-negative source as the active blocker.

## Sixth-game outer-held scene/video fusion screen (2026-08-09)

- [x] Add the existing hash-bound 32-window Lakers–Magic development slice to the four-game
      scene/video embeddings under the canonical `.venv`; keep the sixth source game held out in
      turn and preserve the predeclared scene+MViT variant.
- [x] Seal `analysis_outputs/public_research/shot_validity_scene_fusion_game3_all_v1.json` with
      internal SHA `1c63957e77d67142255673291c1163ac42410d885ef1a2c5980cb81720009cac`; the run
      covers 575 windows across six source videos and remains `runtime_consumable=false`.
- [x] Reject promotion: pooled P/R/F1 is `0.586854/0.980392/0.734214`, the newly held game is
      P/R=`0.400000/0.857143`, and the weakest existing-game precision is `0.503759`.
- [ ] Do not spend another parameter-only increment on this slice; obtain licensed continuous
      5×5 causal labels and exhaustive non-shot hard negatives before resuming blind inference.

## MUVY duplicate rendered-sheet cleanup (2026-08-09)

- [x] Compare the 11 rendered sheets under the historical `muvy_ball_codex_review_v1` directory
      against the canonical `muvy_basketball_v1_review_v2` sheets; all sizes and SHA-256 values match.
- [x] Seal `analysis_outputs/public_research/muvy_duplicate_sheet_cleanup_2026-08-09.json` with
      internal SHA `aabfe7d23f89d00647de37700346d519c29d5ad9e8a1d161b8d4b945ecd70fe8`; delete only
      the exact 11 unreferenced duplicate JPGs (`14,931,003` bytes), retaining review decisions,
      manifests and the canonical replacement sheets.
- [ ] Keep the canonical MUVY review and media; no runtime, detector, VLM, EBQwen or blind asset
      changes are permitted from this cleanup.

## Online candidate supplement: MEV and VSTAT (2026-08-09)

- [x] Pin the public MEV Hub API at revision `1e9460d5909116807dd45345b174fb91e9244553` and inspect
      the full event/video metadata plus license notice without downloading any video shard. Explicit
      basketball keyword hits cover 39 UUIDs and 504.725 seconds, with a maximum candidate span of
      27.694 seconds; the public tree has no per-video source manifest.
- [x] Pin VSTAT at revision `38ef1caea89af3950fd274bf83415dcdc29c710b`; inspect its QA and YouTube
      metadata. All 192 basketball questions map to 30 clips from three YouTube source videos, so
      basketball video is not redistributed under the CC-BY annotation release.
- [x] Seal `analysis_outputs/public_research/online_candidate_supplement_2026-08-09.json` with internal
      SHA `820bd8d0f15502965bc0dd7c02ecb01bdd178095eb6dd208f1a144f996f9cff5`; record zero new payload
      downloads, `runtime_consumable=false`, `training_media_eligible=false`, and add MEV to the
      metadata-first source catalog/gate. Rebuilt catalog SHA is
      `222f7c40c64bb9080e3a4ab045627a8227b3b00b575bc636c2a826e70d7da6b4`; gate SHA is
      `ae21c18ee551b5a5e4abbfcf4e132a8b955e73a8e30642068cf69417de643a07` (64 sources, 11 candidates,
      0 eligible).
- [ ] Do not fetch MEV's multi-gigabyte video shards or VSTAT YouTube clips. Resume acquisition only
      when a source exposes explicit rights, broadcast-diverse continuous 5×5 video, sub-second causal
      ball–hand–rim–outcome labels and exhaustive non-shot hard negatives.

## ExAct basketball skill-feedback audit (2026-08-09)

- [x] Pin [ExAct](https://huggingface.co/datasets/Alexhimself/ExAct) at revision
      `1bd51bfdbd228f850f69cf81d3b4919c71608c04`; download only its README, metadata JSONL and paginated
      file tree for a bounded rights/schema audit.
- [x] Record 1,047 exact basketball metadata rows (Mikan Layup 410, Reverse Layup 388, Mid-Range Jump
      Shooting 249; GE/TIPS `165/882`) and zero MP4 downloads in
      `analysis_outputs/public_research/exact_basketball_skill_audit_2026-08-09.json`.
- [x] Register the source as `runtime_consumable=false` / `training_media_eligible=false` metadata-only
      auxiliary VLM research reference; rebuild the 65-source catalog with SHA
      `6223b0ad4995d757e774044aed18c1e8d8c46a48307e1d55cbecf23ec5a3074e`.
- [ ] Do not treat drill-form feedback as shot outcome, player-statistics or continuous causal supervision;
      the five-check gate remains at 11 candidates and zero eligible sources.

## SportVU 2015–16 trajectory tiny shard (2026-08-09)

- [x] Pin the HF SportVU card at revision `50ec5611a9128c996ac19d094145bcc4ffa57f22` and verify the
      upstream tracking/PBP repository revisions and missing-license status.
- [x] Download the first five lexicographically listed SportVU `.7z` archives (29,017,344 compressed
      bytes), filter the matching PBP rows to 2,208 records, test all archives with `bsdtar -tf`, and
      hash-bind the retained payload in `dataset/public_sources/nba_tracking_15_16_tiny_v1/manifest.json`.
- [x] Audit 2,230 events and 1,020,644 moments at a 40 ms median cadence; record 0.997771 ball-coordinate
      coverage and 1,904/2,230 PBP event-number overlap in
      `analysis_outputs/public_research/sportvu_tracking_tiny_audit_2026-08-09.json`.
- [x] Register the source as `offline_player_ball_trajectory_and_pbp_prior_research_reference` with
      `runtime_consumable=false`, `training_media_eligible=false`, and `causal_truth_eligible=false`; rebuild
      the 66-source catalog (SHA `0dc379a8a96d76643fd72d5b78ff453f5359edacec1f1d39903feb2d55ab1d9e`).
- [ ] Do not treat event-number overlap as broadcast-frame alignment or shot truth. The source has no paired
      video, explicit data license, sub-second ball–hand–rim–outcome labels or exhaustive non-shot hard negatives;
      keep the five-check gate at 11 candidates and zero eligible sources.

## SportVISTA online license audit (2026-08-09)

- [x] Pin [SportVISTA](https://huggingface.co/datasets/bouachalazhar/sportvista) at revision
      `ffb720af2a1e23ff7a8f39379a0f9606cc110e3f`; download only its API metadata, README and license.
- [x] Confirm v0.1.0 is documentation-only (no row-level manifest, annotations or media payload) and manually
      gated; record the release/terms SHA in `analysis_outputs/public_research/sportvista_online_audit_2026-08-09.json`
      (internal SHA `63f0970e111210ff82bf50343aab99319bd27bdd085fb24de6a7e7b13cfbe50a`).
- [x] Reject access for AGU: the research-only license prohibits training, fine-tuning, evaluating, benchmarking,
      aligning, distilling or improving general-purpose/foundation/generative models. Register metadata-only,
      `runtime_consumable=false`, `training_media_eligible=false`, and rebuild the 67-source catalog (SHA
      `d2808333516f6f7f22390e3e51231edf3edaf6081bc0281a0d54fb3003931d88`).
- [ ] Do not request gated access or download SportVISTA payloads for AGU base/VLM work; continue searching for
      a source whose license permits the target model route and whose continuous 5×5 causal labels clear the gate.

## Wikimedia Commons / HCTV continuous full-game seed (2026-08-09)

- [x] Audit the Commons category inventory and retain only one bounded HCTV full-game file after checking the
      per-file CC BY 4.0 metadata, original YouTube Creative Commons license, source attribution and the
      `License review needed (video)` maintenance warning.
- [x] Verify the 1,416,244,469-byte AV1/Opus WebM by exact SHA-256, `ffprobe`, and four timestamped frame decodes;
      the 59.8-minute 1280×720 file shows continuous full-court play and readable scoreboard samples.
- [x] Seal `dataset/public_sources/wikimedia_hctv_fullgame_v1/manifest.json` and
      `analysis_outputs/public_research/wikimedia_hctv_fullgame_audit_2026-08-09.json`; register the source,
      add fail-closed gate coverage, and keep runtime/training/causal-truth flags false.
- [ ] Obtain at least one independently produced game or an explicit second-broadcast source, then manually
      annotate score-linked shot/rebound/assist windows and exhaustive non-shot hard negatives. Until those labels
      exist, keep the five-check gate at 12 candidates/0 eligible and do not restore blind inference.
- [x] Sync the current readiness snapshot to
      `analysis_outputs/public_research/agu_readiness_audit_2026-08-09.json`: `status=not_ready`,
      `blind_inference=paused`, and the HCTV seed remains outside runtime/training truth.

## Wikimedia Commons / VTV independent full-game seed (2026-08-09)

- [x] Audit the independent Venezolana de Televisión (VTV) Commons file and its original YouTube
      record; use the Commons Public-domain / `PD Venezuela official` declaration as the rights basis
      and keep the YouTube record's missing license field explicit.
- [x] Download one bounded 2,228,583,591-byte VP9/Opus WebM, verify its SHA-256, `ffprobe`, and four
      timestamped frames; confirm continuous 5×5 court action and the VTV scorebug.
- [x] Seal the VTV manifest/audit and the HCTV+VTV cross-production audit; register both seeds as
      offline-only manual-annotation inputs. The combined source now proves two independent production
      sources but remains below the causal gate because labels and exhaustive hard negatives are absent.
- [x] Build a hash-bound, low-resource pilot annotation entrypoint over four five-frame windows
      (`analysis_outputs/public_research/wikimedia_hctv_vtv_pilot_annotation_v1/`): two VTV
      `not_a_shot` candidates and two conservative `uncertain` windows. The plan/review is explicitly
      pilot-only, pending human confirmation, and not runtime/training consumable.
- [ ] Manually annotate score-linked shot/rebound/assist windows and exhaustive non-shot hard negatives
      in both productions, then hold out one production for independent P/R verification before any
      runtime, training-truth or blind-inference change.

## Temporary payload cleanup (2026-08-09)

- [x] Remove only the hash-inventoried, unreferenced `/tmp` APIDIS training cache, MUVY scratch
      downloads/frames and old metadata probes; formal audit is
      `analysis_outputs/public_research/agu_tmp_cleanup_2026-08-09.json` (425,154,113 bytes).
- [x] Recheck that both Wikimedia full-game payloads, EBQwen native/MLX-4bit trees, runtime checkpoints,
      source catalog and causal gate remain present; no runtime, training truth or blind asset was deleted.

## Wikimedia Commons / HCTV Randolph third continuous seed (2026-08-09)

- [x] Audit the [HCTV Randolph Commons file](https://commons.wikimedia.org/wiki/File:Boys_Varsity_Basketball_v._Randolph_-_February_17,_2026.webm), Commons API, the US basketball video category, and the original YouTube metadata; preserve the CC BY 4.0 declaration and `License review needed (video)` warning.
- [x] Download one bounded `2,152,005,777`-byte AV1/Opus WebM under user authorization, verify SHA-256 and `ffprobe` (1920x1080, 60 FPS, 5,899.308 seconds), and extract only five timestamped audit frames; no full decode was run.
- [x] Seal the single-source manifest/audit and the HCTV-Hazen, HCTV-Randolph, VTV three-seed cross-production audit. The set remains manual-annotation-only with `runtime_consumable=false`, `training_media_eligible=false`, and `causal_truth_eligible=false`.
- [x] Use Codex offline visual review to seal one five-frame `not_a_shot` candidate (mid-court dribble with no release-to-rim chain); plan/review/manifest are hash-bound, `pilot_only=true`, pending human confirmation, and not exhaustive.
- [x] Recheck a 21-frame Randolph basket-area candidate at 4997.0–4999.0 seconds; seal the conservative `uncertain` v2 artifact because release/rim/outcome cannot be resolved, with no runtime or training use.
- [x] Delete the four exact `/tmp/agu_hctv_randolph_{meta,youtube,category,ffprobe}.json` probes after copying their provenance into the formal audit; retain the original, audit frames, manifests, audits, and pilot artifacts.
- [x] Extend the three-source offline visual pilot with 129 retained 0.1-second JPEG samples across six windows: one conservative HCTV Hazen `shot/missed` causal chain plus five HCTV/VTV/Randolph `not_a_shot` live/interstitial boundary windows. Plan/review/manifest SHA are `964a4a5170c2cd3d5ab903739483033b3fc58b1f1b60db373af761119cdd1b36`, `9c76caf5fd1dee0d9c65b409c75399bf4f3a4b5f9549cd405c41d670c8a6819a`, and `95798600013a59817f908d3a1b5cfda64dd5cc27d73c5306c691bbd0e258a164`; all remain `pilot_only=true`, pending human confirmation, non-exhaustive and unreachable by runtime/training/blind inference.
- [x] Retain the combined pilot at `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v2/`, bound to three-source audit SHA `8cb43f7ac266494ae0e09014d5151bfdca69bd83405e7376bbc2df4d4b490aac`; readiness now records 175 frames/12 windows, while the causal gate remains 16 candidates/0 eligible.
- [ ] Expand manual shot/rebound/assist, interstitial, and exhaustive non-shot hard-negative labels across all three games, then run a held-production per-game P/R gate; the causal gate remains 16 candidates/0 eligible and blind inference stays paused.

## Exact duplicate PBP cleanup (2026-08-09)

- [x] Verify that `dataset/public_sources/nba_games/games/` is an exact 567-file,
      63,474,295-byte duplicate of canonical `dataset/public_sources/nba_games_v1/games/`;
      inventory/tree SHA are `f9db849d0d63883d3768be249bcfc1389e096dde39f9b1a01c69914b8480ce2d` and
      `15050a3918448bca3f3f7859b06ba460f69bdfe4f426b8c99b4d95c7c6395ef8`.
- [x] Delete only the duplicate tree and retain canonical PBP, media, manifests, audits, three
      Wikimedia originals, EBQwen weights, training truth and blind assets; formal cleanup is
      `analysis_outputs/public_research/agu_nba_games_duplicate_cleanup_2026-08-09.json` (SHA
      `382221ea8b557930efc66915e7002df61a65eba8c33266430b15a1409e7433f2`).

## Three-source causal extension v3 and MLX-4bit screen (2026-08-09)

- [x] Scan the retained Hazen, Randolph and VTV originals with bounded contact sheets; reject sampled
      interstitial/transition/dribble windows and retain only two candidates for detailed review.
- [x] Seal `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v3/`:
      45 raw-frame/JPEG hash-bound frames, one conservative Hazen `shot/made` chain and one VTV
      `uncertain` mid-flight broadcast-cut failure. Plan/review/manifest SHA are
      `1dfb5e045c4788c42f7ae0ccca3a42be13edf5c6a483ed0bbc0534849f150d69`,
      `31e20a75347c6a3459330eb923b22c687f4df8de95462c3c7b4de6e5829171a1`, and
      `5e013f2356dc52fdd8b250d125f03f875132beaa90c47fa9a3001a95a4e2f6e4`.
- [x] Run a label-hidden EBQwen2.5-VL-3B-MLX-4bit native-video probe over the two windows with
      `sample_fps=2`, `max_pixels=151200`, and strict-release prompting; it scores precision `0.50` /
      recall `1.00` under the conservative complete-release mapping and is rejected by the `0.85` gate.
      Failure screen SHA: `f2a8d4a3cb0633ea82d9c9965f8ce2597b8bb77a43cc42b4fda62430fdf7338d`.
- [x] Delete the five exact, unreferenced visual scratch directories superseded by formal pilot retention
      (417 files, `59,775,837` bytes); cleanup SHA is
      `425e712768ca02fab2b9c42382cd508f18e7aac3af0821e568433e9c9fa8b73d`.
- [x] Extend the retained three-source pilot to v4 without downloading another payload: 89 hash-bound frames
      across four windows, including a new Hazen `shot/missed` under-basket candidate and a Randolph
      `not_a_shot` live-drive hard negative. Review/manifest/plan/retention SHA are
      `6cdc32bee69379ada7a71be23e56a5857d5fb31889b3f7b45e806d6652f5670f`,
      `ce6d7a0dfe7f4a0238db509ae638c70fffe59b5d79f88d48c0efb1b13f2f26a0`,
      `06774f5a5200118e142652f8c3a30bcbafaa19e43cae8d9cf9d7765353157c7a`, and
      `452c7d0190e4fc589e2da36058c79cfd546d6ab22cde2c11a2c6e9ec23f6c017`; all remain pilot-only,
      pending human confirmation and outside runtime/training/blind paths.
- [x] Delete the exact unreferenced `/tmp/agu_shot_scan_v4.5gB2nw` tree after SHA/reference/holder checks
      (3,068 files, `263,256,416` bytes); cleanup artifact is
      `analysis_outputs/public_research/agu_shot_scan_v4_cleanup_2026-08-09.json` with SHA
      `73ce024735b985cf4bb7016bc3033a4b3ab6a82f6adfa27a6721224f9e642796`.
- [ ] Expand all three games to exhaustive shot/rebound/assist, interstitial and non-shot hard-negative
      annotation, then run the held-production per-game P/R gate before any runtime or blind-inference change.

## Three-source causal extension v5 (2026-08-10)

- [x] Add six bounded windows from the retained HCTV Hazen, HCTV Randolph and VTV originals without a new
      payload download; inherit the four v4 windows for a 10-window/215-frame sealed package.
- [x] Seal the v5 plan/review/manifest/retention artifacts with SHA
      `972a16c19e12fd357e0a742253303efc7b513aa3eaf9af5d38e2b0ed04912a37`,
      `fa3938f2ee217fe70ba10d9633325b1616a4ed9aa19f717298cc5f06a44fd9c3`,
      `b5576dd60ee0202554aa4e4a8b0714b215cc15a30fa32517699c4a6421d55913`, and
      `73544ecf74434123a6abb6b7067aaf51cdf87367166fbc19ad5a672e06887b39`; keep all labels pilot-only,
      pending confirmation and outside runtime/training/blind paths.
- [x] Delete the exact unreferenced v5 visual scratch directories and raw-hash draft (29 files,
      `17,763,243` bytes); cleanup SHA is
      `a7047699b33873c04c9eb02c8d9d96dc1fc9017d5f0b849245ce4a7cfd4c5d49`.
- [ ] Continue exhaustive three-game temporal annotation and run the held-production per-game P/R gate;
      do not promote v5 labels or resume blind inference.

## Three-source causal extension v6 (2026-08-10)

- [x] Review online candidates under the continuous-causal source gate; no new payload was downloaded because the reviewed candidates lack the required rights, continuous 5×5 original video, or subsecond causal labels.
- [x] Add six bounded windows to the retained v5 package, sealing 16 windows / 341 raw-frame and JPEG hash bindings: 4 conservative shots (1 made, 3 missed), 8 `not_a_shot` hard negatives, and 4 `uncertain` broadcast-cut cases.
- [x] Seal v6 plan/review/retention/manifest with SHA `898d2773edb9cb94c3a485b856a17f69a7a7bc5fb9f6d9793fd29a46fb17c985`, `e8f62182a9586880ee5fdcf7df7bb7cd8210a9c2527907797936e90e8aa68655`, `3b4d1b3af6ff49f5bd8b38fa0c5af13df98062f09490f4d1f328b83ed2476b83`, and `d71f99db4a0c2ba99232300144d3adf3d66de137b21c3d30352e152a99f6d89e`; all remain pilot-only and outside runtime/training/blind paths.
- [x] Delete the exact unreferenced `/tmp/agu_v6_broad.1Q6WQl` scratch tree after inventory/reference/holder checks: 183 files / `7,745,976` bytes; cleanup artifact SHA `5221abb861d68b39eb5725e961ebdce6207f85533aafb96ee032e63ca0301920`.
- [ ] Complete exhaustive three-game temporal/interstitial/non-shot labels and run a held-production per-game P/R gate; do not promote v6 labels or resume blind inference.

## v6 independent native VLM screen and prompt ablation (2026-08-10)

- [x] Build a label-free independent native-video plan from the v6 hash-bound windows: 16 examples, raw original videos only, native temporal position encoding, 2 FPS and 151,200 max pixels. Plan SHA `faf553c8d0e678ebc9b57fb6daeb823e03f59f952043a7b39dc2bf8896cd03a5`.
- [x] Run local `EBQwen2.5-VL-3B-MLX-4bit` under the resource guard with both `strict_release_v2` and `negative_first_v3`; both variants predict every window as `live_field_goal`.
- [x] Seal the offline screen at `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v6/independent_vlm_causal_screen_v6.json` (updated SHA `7cccd79c06c9337f69f2976eafedbb48658d22a0e20cca8fb864138c7c43c98c`). Both variants score TP/FP/FN/TN=`4/12/0/0`, precision=`0.25`, recall=`1.00`; no promotion or runtime mutation.
- [x] Add the negative-first prompt variant with cache-bound provenance and regression coverage; focused native VLM tests pass.
- [x] Remove the exact unreferenced single-example probe plan/prediction/cache after the full screen: 3 files / `4,662` bytes; cleanup SHA `b9a677ea2e7b3ca0d03e4f370c2d7bdfd9e8d55494c420811b0b70d63c42f98b`.
- [ ] Obtain independent auxiliary OOF evidence for these three productions, apply the fail-closed evidence gate, and only then reassess promotion; v6 VLM results remain diagnostic and not training truth.

## v6 independent auxiliary transfer screen (2026-08-10)

- [x] Implement and test a label-hidden transfer-only contract in
  `app/analysis/independent_shot_auxiliary_transfer.py` and
  `scripts/screen_independent_shot_auxiliary_transfer.py`; `oof_predictions` is
  structurally forced empty and target labels are rejected from model input.
- [x] Run the pinned WASB-SBDT DeepBall-Large scorer over all 16 v6 windows under
  the canonical `.venv` resource guard. Screen SHA is
  `a528033a276c3bf88d7c55ab99bc5eccf36f6e445fed01c80ce8b33c28517d0c`; guard
  peak memory/CPU/RSS is `77.3%`/`44.5%`/`1,773,486,080` bytes and stop count is 0.
- [x] Seal the delayed, read-only diagnostic evaluation with
  `scripts/evaluate_independent_shot_auxiliary_transfer.py`. Its descriptive
  0.75 cutoff is pooled P/R/F1=`0.667/1.000/0.800`, while VTV precision is `0.0`;
  no threshold was selected and no OOF rows were emitted.
- [ ] Obtain real game-held auxiliary OOF rows from disjoint training
  productions, run `screen_independent_shot_vlm_evidence_gate.py`, and only then
  reassess fusion/promotion. Transfer-only scores remain diagnostic.

## Online source sweep, external calibration and media cleanup (2026-08-10)

- [x] Audit BARD, E-BARD, SportsMOT, SpaceJam, Basketball Events, NBA Games and legacy NBA PBP video dataset against the five-check continuous-causal gate; no candidate qualified and no new payload was downloaded. Sweep artifact SHA `8c9c46f42f5bd3a90c7b1905a25276b074109092d916f743670ceadcbfeb7dbd`; latest gate is 18 candidates / 0 eligible, SHA `c234381b7806c048bb20c97af4ce1d191298c5f9260b6e3c496c39656b61ac30`.
- [x] Build a label-safe external three-game DeepBall-Large calibration diagnostic. The independent calibration selected threshold `0.0` from 96 events with precision `0.177083`; frozen v6 target determinate precision is `0.333333`, recall `1.0`; artifact SHA `7d45fb7d27938aa90c8962884c8dbde851420715f243f6a0503962c1a8651b7d`; `oof_predictions=[]` and no promotion.
- [x] Delete the exact rejected cross-broadcast MP4 allowlist after holder/reference checks: 3 files / `1,909,214,774` bytes; cleanup artifact SHA `d4c1f1907ebff932d1c3de28aced37e536f82a82c80a1e271bc0f9fa4166d4c4`. Manifests, sealed screens, models, blind assets and runtime remain untouched.
- [ ] Complete exhaustive three-game event/interstitial/non-shot annotation and the held-production per-game P/R ≥ 0.85 gate; keep `not_ready` and blind inference paused.

## Three-source full-game coverage queue and v7 offline causal review (2026-08-10)

- [x] Build the label-hidden, hash-bound coverage queue at `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_annotation_queue_v1.json`: `1,922` deterministic 10-second coverage windows across Hazen, Randolph and VTV; queue SHA `8b3f45a6a895f05e8cc63e7c84b7011945681c39fbdb34ac85c1d8fdf6586022`.
- [x] Select 12 source-balanced windows and materialize 492 sparse-seek original frames at 0.2-second sampling under the canonical `.venv` resource-safe path; no AGU runtime was started and no label was written into model inputs.
- [x] Seal v7 offline review/retention/pilot artifacts under `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v7/`: one conservative `shot/missed`, seven `not_a_shot`, four `uncertain/unknown`; plan/review/retention/manifest SHA `eef5dec0ed0cb929c6493e52afb85bf2ab76a6bd87758989dbf300030c32eea9`, `842b38533d37209b48f5474513c542bf6c027404d9d04d0aa47b8cc6a2fbb460`, `45bf92fb013faee826c1aebe9b99ec2ed2abba8762b14ff48f17924d9fee81b8`, `48edb5aab03592ddfaf9beb7203acc7a043717d376c42228684382b2a420a383`.
- [ ] Review the remaining coverage queue for exhaustive event/interstitial/non-shot labels, then run a held-production per-game P/R ≥ 0.85 gate. v7 remains pilot-only, non-exhaustive and outside training/runtime/blind-inference paths.
- [x] Remove the exact unreferenced `/tmp/agu_v7_review.jo6ZgQ` visual scratch after the v7 visual pass: 18 files / `3,346,793` bytes; formal v7 evidence remains retained.

## Three-source disjoint v8 held-out causal review (2026-08-10)

- [x] Add the deterministic held-out batch contract and tests; exclude prior v2–v7 anchors by a 20-second source-local radius and sample eight windows per source. Batch SHA: `8ce44f7749e40309e87685e4180c30bcc3a325ec1e364fc09f2e09b5843793c6`.
- [x] Materialize 24 v8 windows / 504 hash-bound original frames under canonical `.venv`, with labels hidden from the plan and no AGU runtime process.
- [x] Seal the v8 offline review package: one `shot/made`, nineteen `not_a_shot`, four `uncertain`; review/manifest/plan/retention SHA are `7b1f0524f2e2d6e86d260cd4ab3616f9be1cfa5da809650c1e87cca038bad1c2`, `3a55d521a202ba07d1a019169d6030267c7be0c9f3d6c813c5329e8062167e8f`, `60ec3d0a9913032aa29cdf98fb7e8628c677c2c4043053b5d107e10af2437b08`, and `918b14ae25b1b969581a43a600f0969fc5cea4f4ee4f1312ab97cd228775262b`.
- [ ] Expand the queue to exhaustive event/interstitial/non-shot hard-negative truth and run the held-production per-game P/R ≥ 0.85 gate; v8 remains pilot-only and blind inference stays paused.

## Three-source stratified disjoint v9 held-out causal review (2026-08-10)

- [x] Extend `app/analysis/heldout_annotation_batch.py` and `scripts/build_heldout_annotation_batch.py` with a deterministic `early/middle/late` temporal-bucket quota contract; each production source receives 4 windows per bucket, with v2–v8 anchors excluded within a 20-second source-local radius.
- [x] Generate `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_annotation_batch_v9.json` (canonical SHA `78d83f204baa3c908d4bdd0d786c6c4028caf8e63cc221f9a272270c97e37857`; file SHA `51feb1f9ec6689d0def97a239b9a8886d0159d8481e13a0c4b66001f58b69d4e`) and verify 36 selected windows, 12 per source, with no overlap against prior anchors.
- [x] Materialize and seal 756 sparse-seek original/JPEG frames from 36 label-hidden windows under canonical `.venv`; review/manifest/plan/retention SHA are `1b00787d249b9da6b254d3022f4e9c1ac3ab36883789d1497a9598d5a1006890`, `b932f045cf0fa243337522f8cff5435f67a8996b3c003bf1a7f032043a094f5f`, `027e2a75b0f1b324038f27134d368480523c6f73e1941720f373a3dda1d70f10`, and `8c4df9fe74b7c93318c16152722153c55edd1a79c0d8fc13f98e2aa90fdf1386`.
- [x] Offline visual review conservatively seals 2 visible but missed shot chains, 30 `not_a_shot` hard-negative pilots and 4 `uncertain` stoppage/boundary or unresolved-outcome pilots. v8+v9 now cover 60 windows / 1,260 frames, but remain pilot-only and non-exhaustive.
- [x] After formal retention, remove the exact unreferenced v9 contact-sheet directory (36 files / `13,071,798` bytes); cleanup audit is `analysis_outputs/public_research/agu_v9_visual_sheet_cleanup_2026-08-10.json`, while raw frames and sealed artifacts remain.
- [x] Merge v8+v9 counts into `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v8_v9_summary.json` (SHA `781041510552da1a3caae00099cdb358b269e41a1a200a7b2eb5de2578745f0a`): 60 windows / 1,260 frames, 3 shot / 49 not_a_shot / 8 uncertain; per-source P/R distance is explicitly `not_computable` without runtime/OOF scores.
- [x] Re-seal readiness pointers; the updated audit SHA is `cd36e68aeaa398d206ce0e6de6f48389eab23efda359b3760366eb7f4fbed26c`, with runtime/model promotion and blind inference unchanged.
- [ ] Continue exhaustive three-source event/interstitial/non-shot annotation and run the held-production per-source P/R ≥ 0.85 gate; do not feed v9 labels to training, runtime, VLM answers, threshold selection or blind inference.

## Three-source stratified disjoint v10 held-out causal review (2026-08-10)

- [x] Exclude all v2–v9 source-local anchors within 20 seconds and generate the deterministic v10 batch with
      `early=4`, `middle=4`, `late=4` per production source. Batch canonical SHA is
      `d8e5c98305c2c9cebbeaeea5a352f9ab1f26f071fd096d6edec7b91d3cb29d9e`; file SHA is
      `4ace7ddc4ce6a73e6fc56535ea8a390ead0d67d93512886fc2f6358f75995a52`.
- [x] Materialize and visually review 36 label-hidden windows / 756 original-frame and JPEG bindings under canonical
      `.venv`; review/manifest/plan/retention SHA are `f7e85fd0b152d49f763ffded203a566904018199f71f142e4bb6d60de2f10e29`,
      `690d88bf4c36faae6cb7e22fe5d132caad6d5d1f12a7b0d328104b34446b7b59`,
      `0f9318577032f6efecc55f026b5349aa2ea86f2e8b8653ca3d2fb041c8442a91`, and
      `c5c22d613fe4ae0081ed53feb327890e62570d70ff1f2500114bb9ea4160bdc6`.
- [x] Seal 3 `shot` (1 made, 2 missed), 10 `not_a_shot`, and 23 `uncertain` windows; v8/v9/v10 merge to 96 windows /
      2,016 frames, but remains pilot-only/non-exhaustive and outside training, runtime, VLM answers and blind inference.
- [x] Delete the exact unreferenced v10 contact-sheet directory (36 files / `12,562,615` bytes); cleanup SHA
      `3509ab06f8e646f78e9143d5f7e582ae6409ca6033415f69e2b279133fffbd6c`.
- [x] Update the merge summary (SHA `c547a849b6b031d4a950093de965a20942097ef2f9afeca7a466d86aa03aa2eb`) and readiness audit
      (SHA `6c3baeda895d4c662ebb43b286ee4ee83abaadfcc722c1fc6c5857d85c062695`); status remains `not_ready` and blind inference paused.
- [ ] Continue exhaustive three-source event/interstitial/non-shot annotation and run the held-production per-source P/R ≥ 0.85 gate;
      do not promote v10 labels or change runtime/default model/blind inference.

## v10 label-free VLM resource probe (2026-08-10)

- [x] Derive a label-free independent VLM source plan from the sealed v10 review plan; canonical SHA
      `cce46b527953ac968e9d1504cdcac5e5ac645616cc40237dcefe42180ab513e9`, file SHA
      `be59a16d5109faa06294d8b5f6fb2563ddeeaac9afb545c8d6e231018a77a8e4`.
- [x] Attempt a 3-window `EBQwen2.5-VL-3B-MLX-4bit` prefix probe under the canonical `.venv` resource guard. The guard
      observed three consecutive samples below 2 GiB available memory (peak system memory `91.8%`) and stopped with exit
      `75`; no prediction/evaluation artifact exists, so this is not a model-performance result.
- [x] Retain the guard log (file SHA `93b3d44ca1c7a96f95b9b60939229084d573884cd11493869c163cfec5ec361b`) and delete the
      unreferenced 2,109-byte derived probe plan; cleanup audit SHA is
      `301d75c0822dd535ef07fc39aea4933f2fa4cd4c7eae9a0f13e525e812e8e261`.
- [ ] Retry only after the resource guard can sustain the run, then evaluate on label-hidden inputs; no v10 labels enter
      training/runtime/VLM answers and no promotion or blind-inference resume is allowed before independent OOF and the
      held-production per-source P/R ≥ 0.85 gate.

## v7/v8 visual intermediate cleanup (2026-08-10)

- [x] Delete the exact completed-review v7/v8 contact-sheet directories after confirming no manifest/review/retention references:
      36 files / `23,205,453` bytes; cleanup audit SHA
      `98becaf97be22b24f4cdf71e8928d60116b6c30681850ac308d2fb6fb8a12b3b`.
- [x] Retain v7/v8 raw frames, sealed artifacts, source videos, model weights, runtime checkpoints and blind assets; no formal
      evidence or runtime/training input was touched.

## Rebuildable cache cleanup (2026-08-10)

- [x] Remove repository-only `__pycache__`, `.pytest_cache`, `.ruff_cache` and `.mypy_cache` after the full regression run,
      excluding `.venv`, `.git` and `dataset`: 19 directories / 797 files / `10,917,903` bytes. Audit SHA
      `c0c165b9335f3ac448e6c9cb25ed07504571c1db9f7db376d5399f2a0583a8f5`.
- [x] Keep formal evidence, source videos, model weights, runtime/training assets and blind assets untouched.
- [x] Re-seal readiness after the cleanup; audit file SHA is
      `ec757e011bfff51b6223f394f7fa8098acbf1e0d35a63ae3e00f90f23bbbda3b`, with `not_ready` and blind inference paused.

## v10 DeepBall auxiliary transfer and online candidate audit (2026-08-10)

- [x] Without reading v10 review labels, run the label-hidden DeepBall-Large CPU transfer screen over all 36 windows under the canonical `.venv` resource guard.
- [x] Read the sealed labels only after inference for a descriptive diagnostic: 13 determinate windows, cutoff 0.75 P/R/F1=`0.20/0.333/0.25`; 23 `uncertain` windows excluded from cutoff metrics.
- [x] Keep `oof_predictions=[]`, `runtime_consumable=false` and `training_eligible=false`; do not select a threshold, fuse into runtime/VLM, or promote a model.
- [x] Audit Stanford bball_attention, BasketEvent and legacy NBA PBP video dataset; no candidate satisfies verifiable payload/license, continuous 5x5 and sub-second causal gates, so no new payload was downloaded. Audit: `analysis_outputs/public_research/agu_open_source_candidate_audit_2026-08-10.json`.
- [ ] Continue exhaustive event/interstitial/non-shot annotation over the three retained originals and compute held-production per-source P/R ≥ 0.85; keep blind inference and runtime defaults unchanged until then.

## Targeted regression cache cleanup (2026-08-10)

- [x] Remove the six exact cache directories regenerated by the final targeted tests: 25 files / `377,033` bytes; audit SHA
      `01f43ed88bed89d930ce825da45a902e54af1dfc69d9cdbb316e4be52ee13699`.
- [x] Confirm `.venv`, `.git`, `dataset`, source media, model weights, formal evidence and blind assets were not touched.

## v11 stratified held-out causal pilot (2026-08-10)

- [x] Exclude all v2–v10 source-local anchors within 20 seconds and retain early/middle/late quotas of four windows per source; create the v11 label-hidden batch (36 windows).
- [x] Materialize 756 sparse-seek original frames under canonical `.venv`, review all 36 sheets/raw-frame samples, and seal 34 `not_a_shot` plus 2 `uncertain/unknown` windows; no `shot` label was promoted.
- [x] Build the v11 pilot/retention manifests and the v8–v11 merge (132 windows / 2,772 frames); per-source P/R remains `not_computable` because no runtime/held-production OOF predictions exist.
- [x] Delete only the 36 unreferenced v11 contact sheets (13,044,216 bytes); retain raw frames, source originals, sealed artifacts, EBQwen weights, runtime and blind assets.
- [ ] Continue exhaustive causal/interstitial/non-shot annotation over the 1,922-window queue and calculate independent per-source P/R ≥ 0.85 before any training, runtime, VLM fusion or blind-inference resume.

## Independent native VLM probe and metadata follow-up (2026-08-10)

- [x] Build a six-window label-hidden native-video plan from sealed v8–v10 review rows under the canonical `.venv`.
- [x] Run EBQwen2.5-VL-3B-MLX-4bit with `strict_release_v2` under the resource guard; all six windows were
      predicted `live_field_goal`, yielding pooled P/R/F1=`0.50/1.00/0.6667` and per-production precision=`0.50`.
- [x] Keep the result diagnostic-only (`runtime_consumable=false`, `training_consumable=false`,
      `promotion_eligible=false`); readiness stays `not_ready`, blind inference stays paused.
- [x] Re-check Basketball Events metadata only (1,631,871 transient bytes), delete the duplicate temporary copy,
      and retain the existing canonical bounded package.
- [x] Remove the rebuildable repository caches created by the probe/checks (22 directories / 608 files /
      7,543,909 bytes); preserve `.venv`, source media, weights and formal evidence.
- [ ] Complete exhaustive three-production causal labels and independent OOF rows before any VLM fusion or runtime change.

## v12 continuation: source-disjoint manual causal annotation (2026-08-10)

### Overview

Continue the three-production causal-label gate from the existing 1,922-window
coverage queue. Select only windows outside every retained v2–v11 review anchor,
materialize hash-bound original frames, inspect them manually with Codex visual
review, and seal the decisions as another offline-only pilot increment. This
slice must not be used as AGU runtime input, VLM answers, threshold selection, or
training truth until the complete queue and held-production gate are satisfied.

### Acceptance criteria

- [x] Generate a deterministic v12 batch with six previously unreviewed windows
      per production source and a source-local 20-second exclusion radius.
- [x] Materialize and hash-bind the planned original frames under `.venv`,
      review every contact sheet/frame sequence, and seal one decision per
      planned window with release/rim/outcome evidence or an explicit
      `not_a_shot`/`uncertain` decision.
- [x] Verify all artifacts, retain raw frames and sealed manifests, delete only
      unreferenced contact-sheet scratch, and keep readiness `not_ready` with
      blind inference paused.

### Verification checkpoint

- [x] `pytest` focused causal-review contract tests pass under `.venv` (12 passed).
- [x] JSON/hash verification and `git diff --check` pass.
- [x] No AGU service/training process is running; runtime/model/default assets
      remain unchanged.
- [x] Remove the five exact rebuildable cache directories regenerated by the focused
      tests (21 files / `155,455` bytes), excluding `.venv`, `.git` and `dataset`;
      audit SHA `862ef758561b1ef1b18637897fad73114ed972ea21cf1eba38d2c8d070357868`.
- [x] Re-seal the readiness pointer after cleanup; audit file SHA is
      `1fab48cf6804ff7cff5b59700ec5048ad871a14e24cc25b10cbca26976a824a3`, with
      `not_ready` and blind inference paused.

### Dependencies

Existing queue and v2–v11 sealed review plans; no new model or external payload.

## NBA Games v2 PBP/media 配对与 HOU–SAC Q1 时钟复核（2026-08-10）

### Completed

- [x] 固定 [choucsan/NBA_Games](https://huggingface.co/datasets/choucsan/NBA_Games) 到远端 revision
      `cf59e3a42413e3ab91f6fc0b1618f280df9a7024`，只补回 5 场本地已有原片对应的 15 个结构化文件（`1,747,339` bytes）。
- [x] 生成 `analysis_outputs/public_research/nba_games_pbp_v2_paired_media_manifest.json`（SHA
      `27f0d0c1f3fa11e36ef929a7878906ca2f67d000b00801432df80d6b6039413b`）和两场离线配对计划（SHA
      `b556439e3ff77164045ef150d7fb6b19fda6c37fdd4e872b482dde8a16dee4a5`）；PBP/box-score 仅作离线参考，
      不满足 metadata gate，未修改 source gate 的 18/0 结论。
- [x] 在 HOU–SAC Q1 300–600 秒片段用 `.venv` RapidOCR 生成并清洗时钟时间线，映射 38/515 条 PBP 行，
      建立 18 个标签隐藏窗口及 Codex 离线人工复核（14 confirmed、3 partial、1 not_visible，0 个精确 release 帧）。
- [x] 删除无本地配对原片的 4 场结构化下载和可重建 HF cache（68 文件、`1,452,924` bytes），保留正式资产、原片、权重和 blind assets。
- [x] 人工复核后删除不再引用的 2 张总览 contact sheet（`1,348,948` bytes）；清理审计
      `analysis_outputs/public_research/agu_nba_games_pbp_v2_review_sheet_cleanup_2026-08-10.json`，SHA
      `296bdfa9db7e61fa5ea32c882ae9eedee4b1758b6a412f83c64767c2ec8fe1ae`。

### Open gate

- [ ] 对已配对原片扩展完整时钟/插播边界，逐帧标注 release、球—手—篮筐—结果和穷举 non-shot hard negatives。
- [ ] 仅在权利边界、完整因果真值和制作源留出 P/R ≥ 0.85 均满足后，才允许把标签用于训练/独立 VLM evidence；在此前保持
      `runtime_consumable=false`、`training_consumable=false`、`blind_inference=paused`。

## NBA Games v2 ATL–CHI 四节标签隐藏复核增量（2026-08-10）

### Completed

- [x] 新增 `app/analysis/pbp_label_hidden_review.py` 与
      `scripts/build_pbp_label_hidden_queue.py`，把官方 PBP 对齐转换为只含视频几何/opaque join key 的队列；
      focused tests `tests/test_pbp_label_hidden_review.py` 通过 2 项。
- [x] 复用 ATL–CHI 合并 OCR/PBP 对齐，覆盖四节 88 个 mapped field-goal 行；队列内部 SHA
      `25dbee3026997b9416748d4324d94a9f2928e8e184ef9e68c9302d84a627e3dd`，文件 SHA
      `73a5dbcc40c8575c799553214a7f9aea4451b47eaea533fc997a743f00ccd732`。
- [x] 24 个均匀子集完成高分辨率六帧 Codex 视觉复核：21 confirmed、3 partial、0 not_visible、0 exact release，
      视觉结果 18 made / 6 missed；manual 工件内部 SHA
      `d9149471ff18817578ef08ed39ee618bc9e81247655b4c2fa8e0646b80717fd3`。
- [x] 删除不再引用的低分辨率 strips 与临时总览图（26 文件、`6,057,647` bytes），清理审计内部 SHA
      `db5f19f7c3cd06e2ecb43026932eea7b4e67f25bb73f11ae44a56eac970dd5db`；保留高分辨率逐窗 strips、queue、truth 和 manual。
- [x] 删除本轮构建产生的 2 个精确 Python 3.11 bytecode 文件（`35,312` bytes），审计内部 SHA
      `7f58325e6a6574db68880f996490655e085e7b1b80d19138707b8fc0ca235002`；`.venv`、原片和正式工件未触碰。

### Open gate

- [ ] 扩展为配对原片完整时钟/插播边界、精确 release、球—手—篮筐—结果和穷举 non-shot hard negatives；
      该队列仍是 PBP 选样诊断，不能放宽 18 candidates / 0 eligible source gate。
- [ ] 在完整权利与 held-production P/R ≥ 0.85 前保持 `not_ready`、`blind_inference=paused`，不向训练、runtime 或 VLM
      答案提供这些标签。

## 三场源间隔离 v13 held-out 离线因果复核（2026-08-10）

- [x] 在排除 v2–v12 全部源内 20 秒锚点邻域后，按 Hazen、Randolph、VTV 各 early/middle/late
      4 个窗口生成 36 个标签隐藏批次；批次内部 SHA 为
      `72cd83c8a29cd1e2460d784f205bd4a1ece7dbf036d22b9d23c801b0520fec9`，文件 SHA 为
      `25da96817c4697ddeb1c013ffb5694355b1b90bb2dbbb3669e0028339ab3630c`。
- [x] 在 canonical `.venv` 物化并逐窗复核 756 张 hash-bound 原始/JPEG 帧；封存 1 个
      `shot/missed`（VTV 720s，release position 6、rim position 9）、25 个 `not_a_shot` 和
      10 个 `uncertain/unknown`。review/manifest/plan/retention 内部 SHA 分别为
      `985ef02f2272df85ea17ecab7fbf4c97f7fbe56aa222a4120e2c19586822da73`、
      `d94b51eb664eba17a1638fb6383440b506b0dc269578bdae9cd09fe9d0b2c8d9`、
      `78ab158f4c6b92cb38250851b65732dd214b7563addf07ad6b9b72ad93bd653c`、
      `db25caffeca7ae6da2a12c503f4ffc2c6703679fcb39190841331150b33b1c33`。
- [x] 正式保留 raw frames、review decisions/sealed、plan、manifest 和 retention 后，精确删除仅用于显示的
      36 张 contact-sheet（12,883,187 bytes）；清理审计为
      `analysis_outputs/public_research/agu_v13_visual_sheet_cleanup_2026-08-10.json`，内部 SHA
      `053b92f7906b5d64ca84b13d17ed894f5e6e953805c187dc6a091a2c5f1d2578`。
- [x] 更新 v8–v13 合并摘要
      `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v8_v9_v10_v11_v12_v13_summary.json`
      （文件 SHA `abcfd58226c28dfe7e084bc2bc32bb282ce00dd38b38db9542c04e716205c6d4`）：186 个窗口、3,906 张帧，
      7 个 shot、132 个 not_a_shot、47 个 uncertain；每制作源各 62 窗口/1,302 张帧，P/R 仍为
      `not_computable`。
- [x] 48 项 focused contract tests 通过；随后精确删除本轮重建的 4 个仓库 Python bytecode 文件
      （22,169 bytes），清理审计为
      `analysis_outputs/public_research/agu_v13_repository_cache_cleanup_2026-08-10.json`，内部 SHA
      `374bacdb6de3e9edb43621997d5a8a1ab01b4aa57fc9129f56b5ec088f390c24`。
- [x] readiness 审计已绑定 v13 工件与清理审计，文件 SHA 为
      `47e1182e7337edf00ed180cb840d91b67af072497243b121c3f740b8f7f09e4e`；状态继续 `not_ready`、
      `blind_inference=paused`、`runtime_change=none`、`model_promotion=none`。
- [ ] 继续完成三场原片穷举亚秒球—手—篮筐—结果、插播边界及 non-shot hard negatives，再按制作源留出
      计算 P/R ≥ 0.85；v13 标签不得进入训练、runtime、VLM 答案、阈值融合或盲推理。

## 三场源间隔离 v14 held-out 离线因果复核（2026-08-10）

- [x] 排除 v2–v13 源内 20 秒锚点邻域，按 Hazen、Randolph、VTV 各 early/middle/late 四窗生成 36 个标签隐藏窗口；批次内部 SHA `da2beb057a9348131576ef21429bacf49df3aae29505018e35f3db79c84296ea`，文件 SHA `7b09f10cde60ecdd583b7a958905ccd21546526a6622549c03ac021b5405f44e`。
- [x] 在 `.venv`（Python 3.11）物化 756 张 hash-bound 原始/JPEG 帧并完成离线人工复核；封存 2 `shot/unknown`、26 `not_a_shot`、8 `uncertain`，不把 Codex 结论写入 AGU runtime、训练、VLM answer 或 blind inference。
- [x] 封存 review/pilot/plan/frame/retention 工件，生成 v8–v14 合并摘要（222 窗口/4,662 帧；9/158/55；2 made/5 missed），并精确删除 36 个未引用 contact-sheet（12,618,602 bytes）；清理审计内部 SHA `ad55fb77099cfe733e490ae170d0aef676dfd488348b8392beca7759e470cbee`。
- [x] 48 项 focused contract tests 通过；最终验证短暂重建的 4 个仓库 Python 3.11 bytecode 文件（22,169 bytes）已精确删除，审计内部 SHA `5619f336a132a381a9819ff56ed3709a05eb6705a7ffed515495c761eae5ec5a`。
- [x] readiness 审计更新为文件 SHA `47a947ab86813ea88f56267a37f4304bde8523c636258b2d4b1dd752332f530d`，状态 `not_ready`、盲推理暂停，source gate 18/0 未改变。
- [ ] 继续完成三场全量穷举亚秒球—手—篮筐—结果、插播边界及 non-shot hard negatives，并按制作源留出计算 P/R ≥ 0.85；v14 标签保持 offline-only，不进入训练、runtime、VLM 答案或阈值融合。

## 三场源间隔离 v15 held-out 离线因果复核（2026-08-10）

- [x] 排除 v2–v14 源内 20 秒锚点邻域，生成 Hazen/Randolph/VTV 各 12 个 early/middle/late 标签隐藏窗口；批次文件 SHA `2777915007fafd0d1feef88bb69deff986d0ca70bfbbc0aab1445651b82d8770`。
- [x] 在 `.venv`（Python 3.11）物化并复核 756 张 hash-bound 帧，封存 3 `shot/unknown`、30 `not_a_shot`、3 `uncertain`；Codex 结论保持离线，不进入 AGU 训练、runtime、VLM 答案或盲推理。
- [x] 封存 review/pilot/plan/frame/retention 工件，合并 v8–v15 为 258 窗口/5,418 帧（12/188/58；已知 outcome 2 made/5 missed）；仅删除 36 张未引用 contact-sheet（9,709,321 bytes），清理审计内部 SHA `bf98189285052a54131f2c1d58af00ac21e0417ef34fcbfe2678850cd54129c0`。
- [x] readiness 审计已重封，文件 SHA `bf25e07be2e9043d6d6d5a49e17c4c724c7cb156759bdcc7a8f6beb583536a4a`；contact-sheet 已核空，原片、权重、runtime 和 blind assets 未触碰，状态仍 `not_ready`、source gate 18/0。
- [x] 50 项 focused contract tests 通过；精确删除 22 个可重建仓库 Python 3.11 bytecode 文件（540,051 bytes），清理审计内部 SHA `53c98d9205c6b370794142f9d178b73b539fa9ccb66515c465264cf4f65e06dd`，文件 SHA `8ff2ef8bcd613c9249231e00cfa79f5d039572bb64024a3d9edf324f218c1555`。
- [ ] 继续完成三场全量穷举亚秒因果、插播边界及 non-shot hard negatives，并按制作源留出计算 P/R ≥ 0.85；v15 标签保持 offline-only，不进入训练、runtime、VLM 答案或阈值融合。

## 三场源间隔离 v16 held-out 离线因果复核（2026-08-10）

- [x] 排除 v2–v15 源内 20 秒锚点邻域，生成 Hazen/Randolph/VTV 各 12 个 early/middle/late 标签隐藏窗口；批次文件 SHA `55236cf3ea2117469a44146384a44283c3ca430216a0057f07c10a3630def3f1`。
- [x] 在 `.venv`（Python 3.11）物化并复核 756 张 hash-bound 帧，封存 4 `shot/unknown`、30 `not_a_shot`、2 `uncertain`；Codex 结论保持离线，不进入 AGU 训练、runtime、VLM 答案或盲推理。
- [x] 封存 review/pilot/plan/frame/retention 工件，合并 v8–v16 为 294 窗口/6,174 帧（16/218/60；已知 outcome 2 made/5）；仅删除 36 张未引用 contact-sheet（15,957,173 bytes），清理审计内部 SHA `83e74205450c1dfc6f1a5ed7fbcee61d97899a6a3c04dffab39a347121a80ad8`。
- [x] readiness 审计已重封，文件 SHA `5568d665497db3d83207638a74c577d73e92bda95d3aaeb62aed6a5a4fe67d42`；contact-sheet 已核空，原片、权重、runtime 和 blind assets 未触碰，状态仍 `not_ready`、source gate 18/0。
- [x] 50 项 focused contract tests 通过；重建的 22 个仓库 Python 3.11 bytecode 文件（540,051 bytes）已删除，清理审计内部 SHA `e9b7f154b0cebbca60868659b4fb155f32cc099d0592cfdb7375e00f3a3a93a9`，文件 SHA `93bd8d6c38cf7679bad4d493f6585c54a570f1458992ad1f2d5b6be9405d838a`。
- [ ] 继续完成三场全量穷举亚秒因果、插播边界及 non-shot hard negatives，并按制作源留出计算 P/R ≥ 0.85；v16 标签保持 offline-only，不进入训练、runtime、VLM 答案或阈值融合。

## 三场源间隔离 v17 held-out 离线因果复核与源门禁复审（2026-08-11）

- [x] 排除 v2–v16 源内 20 秒锚点邻域，生成 Hazen/Randolph/VTV 各 12 个 early/middle/late 标签隐藏窗口；批次文件 SHA `1d88038e2c2378f3f90239201c5b933e017a64b88544d62f3cb265e7a22de69d`。
- [x] 在 `.venv`（Python 3.11）物化并复核 756 张 hash-bound 帧；封存 2 `shot/made`、33 `not_a_shot`、1 `uncertain`，Codex 结论不进入 AGU 训练、runtime、VLM 答案、阈值融合或盲推理。
- [x] 封存 v17 review/pilot/plan/frame/retention 工件，合并 v8–v17 为 330 窗口/6,930 帧（18/251/61；4 made/5 missed）；逐制作源 P/R 保持 `not_computable`，仍是 non-exhaustive pilot。
- [x] 只删除 36 张未引用 contact-sheet（15,537,554 bytes），并在元数据审计结束后精确删除两个临时 Git 探针 clone（1,836 文件、390,746,999 bytes）和 62 个可重建仓库/模型 cache 文件（567,368 bytes）；清理审计分别为 `agu_v17_visual_sheet_cleanup_2026-08-10.json`（内部 SHA `3cfae6535af0a016a5bf2fd36e7cf485cc30afe4d9a56f408fc3c372b4c05773`）、`agu_tal_track_probe_cleanup_2026-08-11.json`（artifact SHA `8f560e6656d000b835c38cf4726442b436c2970ffba527d17dca58686242ca26`）与 `agu_v17_repository_cache_cleanup_2026-08-11.json`（artifact SHA `ecb2e2915ff765546cd327ab6ccc53b25ad98fd2c84e1d43028d60d541a9daca`），并保留 raw frames、原片、权重及正式证据。
- [x] 将 TAL3x3 与 TrackID3x3 登记为 metadata-first annotation/tracking 辅助源；source gate 更新为 20 candidates / 0 eligible，未新增 payload/runtime/训练接入。
- [ ] 继续完成三场全量穷举亚秒球—手—篮筐—结果、插播边界和 non-shot hard negatives，再按制作源留出计算 P/R ≥ 0.85；在门禁关闭前保持 `not_ready`、盲推理暂停。

## 三场源间隔离 v18 held-out 离线因果复核（2026-08-12）

- [x] 排除 v2–v17 源内 20 秒锚点邻域，生成 Hazen/Randolph/VTV 各 12 个 early/middle/late 标签隐藏窗口；批次内部 SHA `b0d8f669819cdb774fe0e97d2e787f0e5e51549ac6c622c61cd9b7421c4651095`，文件 SHA `0fd117c46975dfc4d1b536cf2b198ab0016f66639e118203e1105481bda6cc6c`。
- [x] 在 canonical `.venv`（Python 3.11.15）物化并逐窗复核 756 张 hash-bound 帧；封存 3 个 VTV `shot/unknown`、33 个 `not_a_shot`、0 个 `uncertain`，Codex 结论保持离线，不进入 AGU 训练、runtime、VLM 答案、阈值融合或盲推理。
- [x] 封存 v18 review/pilot/plan/frame/retention 工件；合并 v8–v18 为 366 窗口/7,686 帧（21/284/61；已知 outcome 5 made/5 missed），逐制作源 P/R 仍为 `not_computable`，仍是 non-exhaustive pilot。
- [x] 仅删除封存后未引用的 v18 contact-sheet（36 文件、17,491,608 bytes）；清理审计 `analysis_outputs/public_research/agu_v18_visual_sheet_cleanup_2026-08-11.json`，内部 SHA `44665334d5f2d47bab57cc42519bd95146c764ff7af11b1ec67d9fa4de8e5cc1`；raw frames、原片、权重和正式证据保留。
- [x] readiness 审计已更新，文件 SHA `53c8ca9aec207779e4cc544d68e0f7c14544034b97e125c9f1b353d4e5b61cd6`；source gate 20 candidates / 0 eligible，状态仍 `not_ready`、`blind_inference=paused`、`runtime_change=none`、`model_promotion=none`。
- [x] 50 项 focused contract tests 通过后，按 allowlist 精确删除 26 个可重建 cache 文件（546,040 bytes），清理审计内部 SHA `475c2ec7afb0c09bca161f37ea2a97b39cfcd5b0f7e987df813d879557bbfa2e`，文件 SHA `808cee908061179f7243462d9dd1da9e5aa978c105c89fc8171cbd42865d9324`；正式资产、`.venv`、原片、权重和 blind assets 未触碰。
- [x] readiness 审计随后重封，当前文件 SHA `a1e7c1a93676dd7a97fa12454494603d7d790ae9a7a760f41b876c29a6e01f58`；source gate 20/0，状态仍 `not_ready`、盲推理暂停。
- [ ] 继续完成三场全量穷举亚秒球—手—篮筐—结果、插播边界和 non-shot hard negatives，再按制作源留出计算 P/R ≥ 0.85；在门禁关闭前保持 research-only/offline，不改 runtime、默认模型或盲推理。

## NBA_Streaming 发布核验与独立证据转移（2026-08-12）

## v19 三场源间隔离 held-out 复核（2026-08-12）

- [x] 在受限 8 秒源内排除半径下生成 36 个标签隐藏窗口（Hazen/Randolph/VTV 各 12 个），计划 SHA `48d54cecab27c5a2446727ad8d5e13cbd9819bb2572151a798121803a9236458`。
- [x] 物化 756 张 hash-bound 原始帧并完成逐帧人工复核；封存 2 `shot/unknown`、3 `uncertain/unknown`、31 `not_a_shot`，不向 runtime、训练、VLM 答案或盲推理提供标签。
- [x] 删除 36 张接触表及仓库可重建 cache/bytecode（初次 169 文件、14,473,033 bytes；回归后新增 261 文件、5,568,909 bytes）；清理审计为 `analysis_outputs/public_research/agu_v19_post_review_cache_cleanup_2026-08-12.json` 与 `analysis_outputs/public_research/agu_v19_final_cache_cleanup_2026-08-12.json`；正式原片、权重、raw frames 与封存工件保留。
- [ ] 继续三场穷举亚秒球—手—篮筐—结果、插播边界与 non-shot hard negatives；当前仍 `not_ready`、blind inference 暂停。

- [x] 论文/HTML/arXiv source 核验报告连续全场事件与 CC BY-NC 4.0，但没有官方 payload URL、hash manifest 或可复现下载 release；审计 `analysis_outputs/public_research/nba_streaming_release_audit_2026-08-12.json`（`3a571ad41cf443d2fc06ddbffedc13702e65c9b06c173f0f2411de468a79cd9d`），0 bytes 下载。
- [x] source gate 重建为 19 candidates / 0 eligible；候选保持 watchlist-only。
- [x] canonical `.venv` 守护下完成同一 compact256 冻结计划（plan SHA `ab36c2ec4eb66a0f431a49a6230e28a3f1947fbb2a7a748a6a37306009cf6b7c`）的 32 窗口 DeepBall transfer；独立 VLM/auxiliary 默认阈值一致 16/32，工件无标签、无 OOF，未进入 fusion/runtime/promotion。transfer：`analysis_outputs/public_research/independent_shot_vlm_auxiliary_transfer_compact256_same_plan_v1.json`（`b356a238888f3f7d499a87f7bfdfc675a1b6fe0590552e9f2024a20706175758`）；一致性：`analysis_outputs/public_research/independent_shot_vlm_auxiliary_consistency_compact256_same_plan_v1.json`（`e6aa8e2196a99f95f94a6e5f3b8a2bba558b81e6b2386533dc96155f1e2eae68`）。
- [x] 屏幕后精确删除项目级可重建 cache 共 71 个文件（2,043,435 bytes），审计 `analysis_outputs/public_research/agu_2026-08-12_post_screen_cache_cleanup.json`（artifact `b49ad388c4fd5330e4bdf9ec35a80ea2d2c29cd09e8121eaa1437e5fc8f3a7b6`，文件 `2b3cf453593a7a6485fba782d442745a3ff71a42cb09d4086f352a4b2df95120`）；正式资产、dataset、`.venv`、原片、权重和 blind assets 未触碰。
- [x] readiness 更新后文件 SHA `3643646dcb45520eee2a30f4430a564e02ea99b8604a4a89a4d776b3cfff8267`；source gate 文件 SHA `399a8f1f81a339064dc8862b52d77f9248dc06c963a250ea738cd92319038bf5`；状态仍 `not_ready`、blind inference 暂停。
- [ ] 继续推进三场全量亚秒因果真值和穷举 hard negatives；门禁关闭前保持 blind inference 暂停。

## 三场源间隔离 v20 held-out 离线因果复核（2026-08-12）

- [x] 以 15 秒源内排除半径生成 Hazen/Randolph/VTV 各 12 个标签隐藏窗口；批次内部 SHA `97d3e00f194fbd412d6895cb40bc16a450e59d2e91b8c606b5eb5272f333f4d3`，文件 SHA `190408944e46164ccbd93902f27a3dac5ff70d32209036ec588ae8fffdc8d76d`。
- [x] 物化并逐窗复核 756 张 hash-bound 原始帧；封存 3 `shot/unknown`、5 `uncertain/unknown`、28 `not_a_shot`，Codex 结论不进入 AGU 训练、runtime、VLM 答案、阈值融合或盲推理。
- [x] 封存 v20 review/pilot/plan/frame/retention 工件；v8–v20 合并为 438 窗口/9,198 帧（26/343/69），逐制作源 P/R 仍 `not_computable`。
- [x] 回归测试 9 项通过；随后删除 v20 contact-sheet 与项目级可重建 cache/bytecode，共 58 文件、17,928,229 bytes；raw frames、原片、权重与正式证据保留。
- [ ] 继续三场全量穷举亚秒球—手—篮筐—结果、插播边界和 non-shot hard negatives，再按制作源留出计算 P/R ≥ 0.85；门禁关闭前保持 `not_ready`、盲推理暂停。

## 2026-08-12 在线开源因果候选再审计

- [x] 在 `.venv`（Python 3.11.15）下核验 10 个在线候选及 HF API revision；磁盘剩余约 5.917 GiB，严格执行先权利/载荷/标签门禁后下载。
- [x] 0 个候选通过因果下载门禁、0 bytes 新媒体下载；GCB/Henu/BASKET 分别因 unspecified license/21.761 GB、manual-gated CC BY-NC/40.5 GB、gated/1.93 TB 与任务范围不合格而拒绝，其他候选仅辅助或 metadata/watchlist。
- [x] 产出 `analysis_outputs/public_research/agu_open_source_causal_candidate_reaudit_2026-08-12.json`（audit SHA `5a47dbb7bb7705d7348acf3c598f6cccaacf78cbf11f9787f0421c093778e8eb`，文件 SHA `d434511522eac005e3f063e2dbecae950a129ee6d2dd74a69a0b8b97b6061c06`），并更新 source gate/readiness 指针；回归后删除 19 个可重建项目/测试 cache（cleanup audit `203f7fca89f03ba569ba9623bec31ccd144e3bd6ff2773ef492ef8692e22f7f5`）。
- [ ] 继续三场原片的全量亚秒球—手—篮筐—结果、插播边界与穷举 non-shot hard negatives，再按制作源留出计算 P/R ≥ 0.85；在此之前不恢复盲推理、不改 runtime/默认模型。

## 三场源间隔离 v21 held-out 复核与清理（2026-08-12）

- [x] 以 15 秒源内排除半径完成 Hazen/Randolph/VTV 各 12 个标签隐藏窗口；批次内部 SHA `4c369b4b53346fb6bbfa5c481962aa003634c7b573e8b8b473db7f223a5c9e60`，文件 SHA `8fed2d86ec049f2348b8f9362e473b86c09707f39c9e00dd43040b882e1fc74b`。
- [x] 在 canonical `.venv`（Python 3.11.15）物化并复核 756 张 hash-bound 原始帧；封存 1 `shot/unknown`、8 `uncertain/unknown`、27 `not_a_shot`，Codex 结论不进入 AGU 训练、runtime、VLM 答案、阈值融合或盲推理。
- [x] v8–v21 合并为 474 窗口/9,954 帧（27/370/77）；逐制作源 P/R 仍为 `not_computable`，source gate 19/0，readiness 仍 `not_ready`、blind inference 暂停。
- [x] 只删除 v21 contact-sheet 与项目级可重建 cache/bytecode；初次 44 个文件、17,227,410 bytes，回归后最终新增 19 个文件、145,007 bytes，合计 63 个删除动作/17,372,417 bytes（其中重复统计的 36 contact-sheet 只计一次时为 55 个唯一文件/17,308,898 bytes）；raw frames、三场原片、权重和正式封存工件保留。清理审计：`agu_v21_visual_sheet_cleanup_2026-08-12.json`、`agu_v21_repository_cache_cleanup_2026-08-12.json` 与 `agu_v21_final_cache_cleanup_2026-08-12.json`。
- [x] 复查 Qlean、NSVA、leHarris、gsbasketball 四个在线候选，0 个通过因果下载门禁、0 bytes 新媒体下载；记录 `analysis_outputs/public_research/agu_open_source_causal_candidate_followup_2026-08-12.json`。
- [ ] 继续完成三场全量亚秒球—手—篮筐—结果、插播边界与 non-shot hard negatives，再按制作源留出计算 P/R ≥ 0.85；在门禁关闭前保持 research-only/offline，不改 runtime、默认模型或盲推理。

## v16/v18/v21 标签隐藏 native VLM + transfer-veto（2026-08-12）

- [x] 以封存 review 构造 6 个跨制作源 raw-only 窗口（plan SHA `99ef9d0ea4aa9638aafabd768cd33e53922524ed8144bdf0c70e0d1dd4045247`），模型侧无标签输入。
- [x] canonical `.venv` 资源守护下完成 EBQwen MLX-4bit native-video；6/6 live，事后 pooled/逐源 P/R `0.50/1.00`，未通过 0.85 门禁。
- [x] 完成同计划 DeepBall transfer 与 label-free consistency；无 OOF，跨模型同意 2/6，未产生融合输出。
- [x] 实现并测试固定阈值、只允许 abstain/live 的 transfer-veto 契约；真实屏幕 2/6 accepted，事后 P/R `0.50/0.3333`，保持 research-only。
- [ ] 下一步仍是三场全量亚秒因果/插播边界/non-shot hard negatives 和 game-held OOF；不恢复盲推理、不修改 runtime/default。
- [x] 回归后清理 34 个本轮可重建 cache 文件、`552,785` bytes；审计 `analysis_outputs/public_research/agu_transfer_veto_cache_cleanup_2026-08-12.json`，正式资产和 `.venv` 保留。
- [x] 2026-08-12 v2 在线候选审计完成：8 个官方候选均未通过连续 5×5、亚秒因果和 hard-negative/rights 组合门禁；0 bytes 新媒体下载，审计 `agu_open_source_causal_candidate_audit_v2_2026-08-12.json`（audit `b0318e9ff9515213457007431235067cb234e08d7dbd75f16151b9fb33a9dc91`）。
- [x] compact256 32 窗口 fixed-threshold transfer-veto 完成：16 accepted、16 abstain，post-inference P/R=`0.625/0.625`，无 OOF/阈值选择/运行时输出。
- [x] v22 增量批次：合并 v8–v21 中心并以 15 秒源内排除半径抽取三场各 12 个 label-hidden 窗口；3 窗口 smoke 已完成 63 帧 hash-bound 物化与封存，完整 36 窗口物化因长 AV1 随机 seek 资源成本暂缓。
- [x] 暂停后精确删除 v22 未完成随机 seek 帧/孤立 contact-sheet（568 个文件，196,006,492 bytes），保留 smoke raw frames、封存工件与 label-hidden plan；清理审计 `agu_v22_materialization_cleanup_2026-08-12.json`。
- [x] 将 v22 物化改为每窗口单次 seek 后顺序解码；完整 36 windows/756 frames 已在 `.venv` 下物化、人工复核并封存（30 `not_a_shot`、6 `uncertain`），仅 offline pilot；contact sheets 与临时顺序 smoke 副本已清理，正式 raw frames/manifest 保留。
- [x] v22 focused regression 16/16 通过；精确删除 25 个项目级可重建 cache/bytecode 文件（232,107 bytes），清理审计 `agu_v22_focused_regression_cache_cleanup_2026-08-12.json`，正式资产、原片、权重和 `.venv` 依赖未触碰。
- [x] v23 增量已完成：沿用 15 秒源内排除半径，从 Hazen/Randolph/VTV 各取 5 个新窗口，315 帧顺序物化并完成 label-hidden 离线复核；封存 1 `shot/missed`、14 `not_a_shot`，内部批次 SHA `8e154fb03ac7dd063ed1ceb7b9851c4c8890c609aa2d4628c3a58f61273b90c3`，sealed SHA `d57fab949a2dc6c83851d20d28d1105adefc9f2bf3613f8716876e7391a16a78`，pilot SHA `0ae9c43e0b3f2cdf8e4bab4d91ca005942994541863b6d7d0e2f90a6e3133a97`。
- [x] v23 只删除复核后可重建的 15 张 contact sheet（5,389,158 bytes）；raw frames、manifest/retention、源视频、权重与 `.venv` 依赖保留，清理审计为 `agu_v23_contact_sheet_cleanup_2026-08-12.json`（内部 SHA `b52679d64e58e3a6210675d96891eb1cc644b3a35e29fc216d7d1286f0611a12`）。
- [x] v23 focused regression 16/16 通过；清理 133 个回归期间生成的可重建 bytecode/cache 文件（3,002,360 bytes），审计 `agu_v23_regression_cache_cleanup_2026-08-12.json`（内部 SHA `893bf57c422e7d33d8f441a999017a15807a1833b4c13f96a5ff09b023fc2604`），`.venv` 依赖与正式资产保留。
- [x] v23 后 readiness/source gate 指针已重封并校验：readiness 文件 SHA `4db4c19e626e5e90d68e8dbf48a4f57cff136e37c075b4a1471039daf0d4c394`（audit `14ff7a113d0c68563dc7652cbcd9d22ab6558131a4f634cfead5148e4dae2136`），source gate 文件 SHA `409876a321af973cc2c68e6b3837c896e706d4b5d5e2b3aef5cf403ac7cf7ed6`（audit `5e2e05a6c118e46419ada1c19af0d2cd32510406ac2873ad4c996293f03d807`）。

### 2026-08-13 在线候选 v3 与基座/VLM重放

- [x] 在用户授权下完成 9 个官方候选的 metadata-first 审计；匿名 gated shot-test 无法取文件，其余候选分别因无许可、体积、任务不匹配或已有本地副本拒绝重复下载；v3 审计记录 0 个 eligible source、0 bytes 新媒体。
- [x] 在 canonical `.venv`（Python 3.11.15）重放冻结 32 行 `both_confirm` 融合；预测 artifact hash 与封存结果一致，RSS 峰值 374,964,224 bytes，P/R 0/0，未晋级且不接入 runtime。
- [x] 41 项聚焦契约测试通过；按精确 allowlist 删除本轮 pyc/pytest cache 和 12 个 `/tmp/agu_*` 探测副本，正式数据、原片、帧和权重保留。
- [ ] 下一增量：从现有源继续生成 source-disjoint label-hidden 窗口，完成人工复核和逐制作源 held-production P/R；补 game-held OOF 后才能评估任何阈值或融合规则。
- [ ] 继续覆盖三场剩余全量窗口并完成穷举事件/插播边界/non-shot hard-negative 真值；在逐制作源 held-production P/R ≥ 0.85 与 game-held OOF 门禁关闭前，保持 source gate 19/0、readiness `not_ready`、blind inference 暂停，不改 runtime/default。

### 2026-08-13 v24_rv source-disjoint 复核

- [x] 封存当前队列容量：在 15 秒源内排除半径下 Hazen `0`、Randolph `141`、VTV `482` 个可用窗口；Hazen 耗尽审计已写入 `agu_hctv_annotation_coverage_exhaustion_2026-08-13.json`。
- [x] 完成 Randolph/VTV 各 4 个 label-hidden 窗口、168 张稀疏原片帧的 Codex 离线人工复核，sealed 标签为 4 `not_a_shot`、1 `shot/made`、3 `uncertain/unknown`；sealed/pilot/retention 正式文件已绑定 SHA。
- [x] 复核后删除 16 张可重建 contact sheet，保留 raw frames、review、manifest 和 retention；13 项 v24_rv 契约回归通过，峰值 RSS `94,076,928` bytes，测试缓存已精确清理。
- [ ] v24_rv 不更新 v23 readiness 指针，不进入训练/runtime/VLM 默认答案/融合/晋级；继续全量因果事件、插播边界、穷举 hard negatives 和 game-held OOF，门禁未闭合前保持 blind inference paused。

## 2026-08-13 v25_rv source-disjoint 复核

- [x] 在 15 秒源内排除半径下从 Randolph/VTV 各抽取 early/middle/late 两个标签隐藏窗口，12 windows/252 frames；review plan 内部 SHA `eb986c52b73d88a1abff4b8c8ca25f6e9bd83210103eef15c5d563d02a86929a`，raw-frame manifest 内部 SHA `efb6d266542e2ee2ca4ece8fe685ca983720e84a878bbab39b4a3df4152adc04`。
- [x] 完成逐帧离线人工复核并封存 11 `not_a_shot`、1 `uncertain`，无新增可确认 shot；sealed 文件 SHA `4171b78327081920ac4fe5b72163b1096d604cba0178c6e935b0943e88f9744f`，pilot 文件 SHA `3e85bb628caba9081f207acd83ba40f534861973186fa6640e1937a80eca003e`，retention 内部 SHA `84375f2510b235ab78a149658ea1c591d695b5f29cea674633a1aa29b68d4bf5`。
- [x] 复核后删除 12 张已查看 contact sheet（`5,617,098` bytes），raw frames/正式 review/manifest/retention 保留；13 项 focused regression 通过，峰值 RSS `93,798,400` bytes。
- [x] 回归后精确删除 7 个项目级可重建 pyc/pytest cache 文件，`.venv` 依赖与正式资产保留；清理审计 `agu_v25_rv_regression_cache_cleanup_2026-08-13.json`。
- [ ] v25_rv 仍为 pilot-only/offline，不更新 v23 readiness 指针，不进入训练/runtime/VLM/融合/晋级；继续三场全量亚秒因果、插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF。

## 2026-08-13 通用基座迁移审计（已完成）

- [x] 新增 `scripts/audit_vru_pilot_base_model.py`，在不改 runtime、不改 checkpoint、不把 pilot 标签写入训练的前提下，用 v23/v24_rv/v25_rv 已封存帧对当前十类 `r2plus1d_v3` 动作头做 whole-frame proxy 诊断。
- [x] 在 `.venv` Python 3.11.15 与资源守护下完成 31 个确定窗口：2 positive、29 negative；固定阈值与 argmax 均 TP/FP/FN/TN=`0/0/2/29`，rank AUC=`0.465517`，因此不晋级、不调阈值、不恢复 blind inference。
- [x] 新增 3 项契约测试；Ruff、py_compile、守护运行均通过。工件为 `analysis_outputs/public_research/agu_vru_pilot_base_model_audit_2026-08-13.json`，清理审计为 `analysis_outputs/public_research/agu_vru_pilot_base_model_audit_cache_cleanup_2026-08-13.json`。
- [ ] 下一步仍是取得合法、可复现的连续因果训练 bundle，完成三场穷举事件/插播边界/non-shot hard negatives，再进行 game-held OOF 和逐制作源 P/R ≥ 0.85；在此之前保持 readiness `not_ready`、blind inference paused。

### 2026-08-13 已取代 pilot raw-frame 清理

- [x] 新增 `scripts/prune_superseded_pilot_raw_frames.py`：只允许 v2–v22 `raw_frames`，拒绝 symlink，默认
  dry-run，并执行后校验 v23/v24_rv/v25_rv 保护帧不变。
- [x] 执行删除 `12,021` 个已取代衍生文件、`3,917,617,577` bytes；源视频、review/manifest/retention、
  checkpoint、`open_models` 与 `.venv` 保留。审计 `agu_superseded_pilot_raw_frames_cleanup_2026-08-13.json`
  的内部 SHA 为 `123e8952c469be5f272576df4d7e15f84e2b668ea037fe764f2eb74ca70ab353`，文件 SHA 为
  `746e89fddc28eac3bf7e33ef7b98721fec9afaa626246ec2f0b1477289939eee`。
- [x] 新增两项临时目录安全回归，`2 passed`；Ruff/py_compile 通过。此清理可由保留原片与 review plan
  重建，不改变训练、runtime、readiness 或 blind inference 边界。
- [ ] 在三场全量亚秒因果标签、插播边界、穷举 non-shot hard negatives 与 game-held OOF 闭合前，不把
  pilot 标签转成训练真值，也不恢复盲推理。

### 2026-08-13 四场旧标注链当前基座迁移诊断

- [x] 复核旧四场 SHA 绑定原片/annotation manifest 配对为 374 windows、4 games、173 positive/201 negative。
- [x] 用当前 `r2plus1d_v3` + `agu_v3` preprocessing + `layer4` + 1 epoch 在资源守护下完成诊断 OOF；
  pooled `0/0/173/201`、P/R/F1 `0/0/0`，逐场 recall 全 0，未晋级且未接入 runtime。
- [x] 保留 metadata/guard 日志，删除可重建的 125,393,565-byte 临时 checkpoint；清理审计
  `agu_r2plus1d_currentbase_full374_agu_v3_layer4_e1_checkpoint_cleanup_2026-08-13.json`。
- [ ] 继续构建真正的跨比赛 causal training bundle；当前迁移证据仍表明不能把现有 pilot 或旧候选直接视为已训练好的全场模型。
## NBA Games local mirror audit and guarded VLM retry (2026-08-13)

- [x] Re-audit the online NBA Games metadata contract and verify the local `nba_games_v1` mirror: 189/189 game directories and source IDs match; 5,194 box-score rows, 81,355 PBP rows, 166 non-empty PBP games, 23 empty games, and zero local video payload bytes.
- [x] Add the offline-only mirror audit module/script and RED→GREEN contracts; keep `runtime_consumable=false`, `training_media_eligible=false`, and explicitly reject PBP timestamps as frame-level causal truth.
- [x] Start an 8-window raw-frame `qwen3-vl:2b` strict-release probe under the resource guard. Stop after three consecutive samples below 2 GiB available memory (exit 75); no prediction artifact was sealed and no runtime/default/model/readiness state changed.
- [ ] Do not download linked YouTube media from this metadata-only source without a separate rights review; obtain a rights-cleared continuous 5×5 source with sub-second causal labels and exhaustive hard negatives before training or blind OOF.

## 2026-08-13 冻结 shot-validity 头与 Kinetics 对照

- [x] 在四场 SHA 绑定旧链上完成当前 AGU `fc`/uniform 与训练侧 `anchor` 两个冻结 backbone 对照；374 windows 均成功配对，但 pooled OOF 分别为 `3/4/170/197` 与 `1/0/172/201`，均远低于晋级门禁。
- [x] 下载官方 Kinetics-400 R(2+1)D-18 权重做标准 `kinetics`/uniform/frozen-`fc` 对照；守护无停止，四场 OOF 为 `0/0/173/201`，未晋级。
- [x] 删除两个不合格临时 checkpoint 与 120 MiB 官方权重，保留 metadata、guard、来源 SHA 和可重建清理审计；runtime/default/readiness/blind 状态未改变。
- [x] 回归后删除 16 个项目级可重建 cache/bytecode 文件、`109,437` bytes，审计 `agu_2026-08-13_final_cache_cleanup.json`；`.venv` 与正式资产未触碰。
- [ ] 不再扩大这三个失败迁移变体；下一步必须补齐 rights + continuous causal labels + exhaustive hard negatives + game-held OOF，之后才重新评估 shot-validity 或 VLM 融合。
- [x] 追加五个在线候选的官方页面核验；无一通过连续因果/许可/可控载荷门禁，未下载新 payload，审计 v3 已更新。
## 2026-08-13 v26_rv source-disjoint 复核

- [x] 在 592 个既有 review-plan 中心锚点外按 15 秒源内排除半径，使用新增 `source_ids` 子集契约抽取 Randolph/VTV 各 early/middle/late 1 个标签隐藏窗口（6 windows/126 frames）。
- [x] 逐帧复核封存 6 个 `not_a_shot/not_applicable`，保留 SHA 绑定 raw frames、plan、review、pilot 与 retention；删除 6 张可重建 contact sheet（1,831,087 bytes），审计 `agu_v26_contact_sheet_cleanup_2026-08-13.json`。
- [x] `tests/test_heldout_annotation_batch.py` 6 passed，Ruff/py_compile/diff check 通过；源子集契约不改变旧的全源默认行为。
- [ ] v26_rv 不构成全量三源因果真值，继续插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF；readiness `not_ready`、blind inference paused。

## 2026-08-13 v27_rv source-disjoint 复核

- [x] 在既有 review-plan 中心外按 15 秒源内排除半径，从 Randolph/VTV 各抽取 early/middle/late 12 个 label-hidden 窗口，共 24 windows/504 张原片帧；held-out batch SHA `e576ec63d473544542774e1bd8f417840abe5abbc8c95cecc91c166de2da6b79`，review plan SHA `28546ef24aef9c80b9d4f938fdf5bf92ed2c20a989d5ee441522968547c0c239`。
- [x] 完成离线逐帧复核并封存 22 `not_a_shot/not_applicable`、2 `uncertain/unknown`，没有新增可确认 shot；sealed SHA `803bf293f2ced37aea4ab31101327fdd0aee3fedaba856f6f4a30808de2854df`，pilot manifest SHA `ff2a7022600d23b6657dac5c8c583534a5cd7dae6db64610e04351bd0b344785`，retention SHA `3f9e001ca3a07ef4fc682295db77a406f69e32c21afef16a7aec24591fa0e693`。
- [x] v27_rv 保持 `pilot_only`、`exhaustive=false`、`training_consumable=false`、`runtime_consumable=false`、`hard_negative_gate_eligible=false`；不进入 AGU runtime、训练、VLM 默认答案、融合、晋级或 blind inference。
- [x] canonical `.venv` focused regression `13 passed`；复核后按精确白名单删除 24 张可重建 contact sheet（`11,708,095` bytes），审计 `analysis_outputs/public_research/agu_v27_contact_sheet_cleanup_2026-08-13.json`，raw frames/正式 review/manifest/retention 保留。
- [x] 回归后再次按精确 allowlist 删除项目级 `.pytest_cache`（4 files / `1,853` bytes），审计 `analysis_outputs/public_research/agu_v27_regression_cache_cleanup_2026-08-13.json`；`.venv`、raw frames、原片与正式工件未触碰。
- [ ] v27_rv 仍不构成三场全量因果真值；继续插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF，readiness `not_ready`、blind inference paused。

## 2026-08-13 v28_rv source-disjoint 复核

- [x] 在前批全部中心外按 15 秒源内排除半径，从 Randolph/VTV 各抽取 12 个 label-hidden 窗口，共 24 windows/504 张 hash-bound 原片帧；held-out batch SHA `495c7b1e06c2178d4a37869a364f364a7af198ba3111e8b94f525b54f74a8693`，review plan SHA `b3693e4d0e81cc9dadfb269270c012d3deae1d363168cae083cc66a45a25c1e5`。
- [x] 完成离线逐帧原片复核并封存 20 `not_a_shot/not_applicable`、4 `uncertain/unknown`，没有确认 shot；sealed SHA `52a3147b0c963f05d446dc43b8da01dc75c6ec7694517d537fb9800204d35415`，pilot manifest SHA `cd9f636d800c6d8334a60a70c0ae43383142589f7c290f9a002649943f8f48c9`，retention SHA `fbcf446de61d320cc3b4ed719a6861a96fde2b9b3864201ec9d850f225d30d35`。
- [x] v28_rv 保持 `pilot_only`、`exhaustive=false`、`training_consumable=false`、`runtime_consumable=false`、`hard_negative_gate_eligible=false`；不进入 AGU 训练、runtime、VLM 默认答案、融合、晋级或 blind inference。
- [x] canonical `.venv` focused regression `13 passed`；复核后按精确白名单删除 24 张接触表（`11,613,466` bytes），审计 `analysis_outputs/public_research/agu_v28_contact_sheet_cleanup_2026-08-13.json`；raw frames、review/manifest/retention 正式工件保留。
- [x] 回归后按精确 allowlist 删除项目级 `.pytest_cache`（4 files / `1,853` bytes），审计 `analysis_outputs/public_research/agu_v28_regression_cache_cleanup_2026-08-13.json`；`.venv` 未触碰。
- [ ] v28_rv 仍不能替代三场全量因果真值；继续插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF，readiness `not_ready`、blind inference 继续暂停。

## 2026-08-13 v29_rv source-disjoint 复核

- [x] 在前批全部中心外按 15 秒源内排除半径，从 Randolph/VTV 各按 early/middle/late 配额抽取 12 个 label-hidden 窗口，共 24 windows/504 张 hash-bound 原片帧；held-out batch SHA `63ab792a00a3a90faea9e0265b750fb90a8f3f9a8ef2914b6e3371e38740038c`，review plan SHA `7734946f98ded098b53a80dd8da83b0a8040ed8b4cd64cca6a602e860473ae21`。
- [x] 完成离线逐帧原片复核并封存 19 `not_a_shot/not_applicable`、5 保守 `uncertain/unknown`，没有确认 shot；sealed SHA `e9f8db85b8846a92b77982af7ff532cff7009b651ca61ba8169e5bee11b67e90`，pilot manifest SHA `2ab607117fb6cf521210ac30a1e5cbfcc37c583d00c1c36fff0dfaedd785274f`，retention SHA `bf8e0f344ae8f0fba724c1eabf265f719ac895364d3a9c817dd198daad70be73`。
- [x] v29_rv 保持 `pilot_only`、`exhaustive=false`、`training_consumable=false`、`runtime_consumable=false`、`hard_negative_gate_eligible=false`；不进入 AGU 训练、runtime、VLM 默认答案、融合、晋级或 blind inference。raw frames 与正式 review/manifest/retention 工件保留。
- [x] canonical `.venv` focused regression 共 `13 passed`；删除 24 张可重建 contact sheet（`11,854,476` bytes），审计 `analysis_outputs/public_research/agu_v29_contact_sheet_cleanup_2026-08-13.json`；删除 9 个项目级回归缓存/bytecode 文件（`39,954` bytes），审计 `analysis_outputs/public_research/agu_v29_regression_cache_cleanup_2026-08-13.json`；`.venv` 未触碰。
- [ ] v29_rv 仍不构成三场全量因果真值；继续插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF，readiness `not_ready`、blind inference 继续暂停。
## 2026-08-13 rejected auxiliary payload cleanup

- [x] 仅按 allowlist 删除 Infactory `data/`/YOLO 派生载荷、APIDIS `raw/`、UVY `UVY/` 与 `yolo_aux_v1/`；不触碰 manifests、许可、central directory、正式审计、Wikimedia 原片、EBQwen 或 `.venv`。
- [x] 两阶段合计删除 `19,212` files / `1,667,793,510` bytes；合并审计 `analysis_outputs/public_research/agu_rejected_auxiliary_payload_cleanup_summary_2026-08-13.json`，清理后 `df -h .` 可用约 `8.7 GiB`。
- [x] 重点回归与清单/模型契约测试通过；已删除载荷不被当前 runtime、训练真值或盲推理引用，旧结果 JSON 作为 provenance-only 保留。
- [x] 回归生成的项目级 `.pyc`/pytest/Ruff cache 已按精确范围删除并审计；canonical `.venv` 未触碰。
- [ ] 训练与盲推理目标仍未闭合；必须先补齐 rights-cleared 连续因果 bundle、穷举 hard negatives 和 game-held OOF。

## 2026-08-13 v30_rv source-disjoint 复核

- [x] 在前批全部中心外按 15 秒源内排除半径，从 Randolph/VTV 各按 early/middle/late 配额抽取 12 个 label-hidden 窗口，共 24 windows/504 张 hash-bound 原片帧；held-out batch artifact SHA `3cc3c05c1e9eb00e3f16ec7b944e4485f2b709fbc83c982c6b4f142a704cc471`，review-plan SHA `fd89fc7faff48bd0bb2cc8283083b04d3c74834259f8988620259d792830b8af`。
- [x] 完成标签隐藏的原片逐帧复核并 fail-closed 封存 15 `not_a_shot/not_applicable`、9 `uncertain/unknown`，没有确认 shot；sealed SHA `84de8da98d54700f7f7bf62e8e6af2f543d780623d2441db7151e64be39e79ee`，pilot manifest SHA `7d65343c39f6647fb4ecb977674afe5c7e6904b18c141b42d6dde573ed39386b`，retention SHA `3aa99dc0e020895d6663530dd95bdf4774e6b4142fb119e081c2b487badc860c`。
- [x] v30_rv 保持 `pilot_only`、`exhaustive=false`、`training_consumable=false`、`runtime_consumable=false`、`hard_negative_gate_eligible=false`；不进入 AGU 训练、runtime、VLM 默认答案、融合、晋级或 blind inference；原片帧与正式 review/manifest/retention 工件保留。
- [x] canonical `.venv` focused regression 共 `13 passed`；按精确 allowlist 删除 24 张可重建 contact sheet（`11,640,731` bytes），以及 10 个项目级缓存/bytecode 文件（`61,430` bytes）；审计分别为 `analysis_outputs/public_research/agu_v30_contact_sheet_cleanup_2026-08-13.json` 与 `analysis_outputs/public_research/agu_v30_regression_cache_cleanup_2026-08-13.json`；`.venv` 未触碰。
- [ ] v30_rv 仍不构成三场全量因果真值；外部候选复筛新增下载 `0` bytes，继续插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF；readiness `not_ready`、blind inference 继续暂停。

## 2026-08-14 v31_rv source-disjoint 复核

- [x] 在全部既有中心外按 15 秒源内排除半径，从 Randolph/VTV 各抽取 12 个 label-hidden 窗口，共 24 windows/504 张 hash-bound 原片帧；held-out batch artifact SHA `12ca8d6c93114caa8dc2e97109a48ef7e551fb4c3a22e3a7dcf45f0852154c54`，review-plan SHA `c9b3eb5d5ac3f7f54b9da380db06dc0fd104046c461a1cd6a1f459cf463a529c`。
- [x] 完成原片逐帧人工复核并 fail-closed 封存 14 `not_a_shot/not_applicable`、9 `uncertain/unknown`、1 `shot/unknown`（VTV 872s；释放/篮筐接触链可见但稀疏终点不足以判定命中/未中）；sealed SHA `9b5da3e7e4c1f20f96d129819667b3c6ce7c708f3c55a3092ccd36abf5b68a29`，pilot SHA `da6e65211b5d4e4041094bf765c089d325bc60a46504f07a46e2dbb547f2fed0`，retention SHA `f81875637b08ec8c4c8c5254acb944a9a496cd90fb2cfa730b88db0282db0101`。
- [x] v31_rv 保持 `pilot_only`、`exhaustive=false`、`training_consumable=false`、`runtime_consumable=false`、`hard_negative_gate_eligible=false`；不进入 AGU 训练、runtime、VLM 默认答案、融合、晋级或 blind inference，原始视频/504 raw frames/正式工件保留。
- [x] canonical `.venv` causal-review regression `13 passed`；按精确 allowlist 删除 24 张已复核 contact sheet（`11,212,920` bytes）和 10 个项目缓存/bytecode 文件（`61,430` bytes），审计分别为 `agu_v31_contact_sheet_cleanup_2026-08-14.json` 与 `agu_v31_regression_cache_cleanup_2026-08-14.json`；`.venv` 未触碰。
- [ ] v31_rv 仍不构成三场全量因果真值；外部候选复筛新增下载仍为 `0` bytes，继续插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF；readiness `not_ready`、blind inference 继续暂停。

## 2026-08-14 v22 smoke 派生物清理

- [x] v22 完整批次已保留正式 raw frames/manifest/review/retention 后，按精确白名单删除被其替代的三窗口 smoke raw frames（63 files，`21,012,931` bytes）；smoke 的 metadata、review decisions、plan、sealed、pilot/retention manifests 保留，审计为 `analysis_outputs/public_research/agu_v22_smoke_derivative_cleanup_2026-08-14.json`。
- [x] 清理不触碰 `.venv`、三场原片、正式 v22 raw frames、EBQwen 或 runtime/readiness 资产；其余历史 raw evidence 仍因正式哈希绑定 provenance 保留。

## 2026-08-14 v32_rv source-disjoint 复核与诊断门禁复核

- [x] 在全部既有中心外按 15 秒源内排除半径，从 Randolph/VTV 各抽取 early/middle/late 12 个 label-hidden 窗口，共 24 windows/504 张 hash-bound 原片帧；held-out artifact SHA `2205286c5364a8507025dce334111b2fcf9c6bdea6569b12095fdc84efdd8c6a`，review-plan SHA `daad00c9a4275394372ea9eff91d502adba1e5f15fc98c7d91487df28dd16b2d`，raw-frame manifest SHA `efc32cc4712f501702d2b013b20dc24b14e4a8f105cd060272198dd9c9bc546a`。
- [x] 完成原片逐帧人工复核并 fail-closed 封存 12 `not_a_shot/not_applicable`、11 `uncertain/unknown`、1 `shot/unknown`（VTV 4842s；罚球出手到篮筐接触可见，但稀疏终点不能判定命中/未中）；sealed SHA `eb29e90a03b826d8ae45ff3de2c3a637a17db39d24e66736f7d79b6050807165`，pilot SHA `f7a73fc3800fc6f3681daeb780b43c106cc057afb17055cc97b07692c031f413`，retention SHA `7ad3f19bbd5a11d114d930eacb14eddca78a84f37100d5c4ba947f0ac9a6df0c`。
- [x] v32_rv 明确保持 `pilot_only`、`exhaustive=false`、`training_consumable=false`、`runtime_consumable=false`、`hard_negative_gate_eligible=false`；不进入 AGU 训练、runtime、VLM 默认答案、融合、晋级或 blind inference。focused causal-review regression `13 passed`。
- [x] 复核后按精确 allowlist 删除 24 张可重建 contact sheet（`12,182,043` bytes），审计 `analysis_outputs/public_research/agu_v32_contact_sheet_cleanup_2026-08-14.json`；13 项回归后再删除 4 个项目级 pytest cache 文件（`1,853` bytes），审计 `analysis_outputs/public_research/agu_v32_rv_regression_cache_cleanup_2026-08-14.json`。raw frames、review spec/plan/decisions/sealed、pilot/retention manifests 与原片保留，`.venv` 未触碰。
- [x] 重查既有训练/融合诊断门禁：当前基座 31 个确定窗口仍为 TP/FP/FN/TN=`0/0/2/29`、P/R/F1=`0/0/0`、rank AUC=`0.465517`；独立 VLM transfer post-inference 32 窗口 P/R=`0.625/0.625` 且 `promotion_eligible=false`。未启动没有新训练真值支撑的重训，也没有改 runtime/default/readiness 指针。
- [ ] v32_rv 仍不构成三场全量因果真值；继续插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF，readiness `not_ready`、blind inference 继续暂停。

> 诊断复核门禁：在没有新的可训练连续因果真值前，不启动重训或融合调参；下一步仅推进全量标注与游戏留出 OOF。

## 2026-08-14 v33_rv source-disjoint 复核与在线源复筛

- [x] 在全部既有历史锚点外按 15 秒源内排除半径，从 Randolph/VTV 各抽取 12 个 label-hidden 窗口，共 24 windows/504 张 hash-bound 原片帧；held-out artifact SHA `b578867cbbc2bd93ceb1de8e108a0919e026dfd5facb5dbbe7ca1df859aea356`，review-plan SHA `228868b5acc22c23e98ecf0306c9ffa79744b3ee57227efedbbb35a2e14ab596`，raw-frame manifest SHA `6793727d672c5743d8b7c3407ff2c154b1efcf03913547a550b7cb746747f4c5`。
- [x] 完成 labels-hidden 原片逐帧人工复核并 fail-closed 封存 10 `not_a_shot/not_applicable`、14 `uncertain/unknown`、0 `shot`；sealed SHA `2382ad54801ad75212ba94afdb1a314ce121322f1582d7e45eea127359a4a586`，pilot SHA `ca321e3629edf1ee6fbec415c3502de7cf3a306d45dd9a51962689297e46764f`，retention SHA `b787f84211fab104479ffa236b3ab2fe4f0456b789fdeffa1c5705aae66ff2b2`。
- [x] v33_rv 保持 `pilot_only`、`exhaustive=false`、`training_consumable=false`、`runtime_consumable=false`、`hard_negative_gate_eligible=false`；不进入 AGU 训练、runtime、VLM 默认答案、融合、晋级或 blind inference。canonical `.venv` focused causal-review regression `13 passed`。
- [x] 复核后按精确 allowlist 删除 24 张可重建 contact sheet（`12,523,486` bytes），审计 `analysis_outputs/public_research/agu_v33_contact_sheet_cleanup_2026-08-14.json`；测试后移除空项目 `.pytest_cache` 目录，未触碰 `.venv` 或正式工件。
- [x] 依据用户授权完成在线元数据复筛；BARD 仅保留已有本地辅助载荷，NBA Games/NSVA/NBA PBP/BASKET/NBA Streaming 均未通过连续因果、权利或可控载荷门禁，新增下载 `0` bytes；审计 `analysis_outputs/public_research/agu_online_source_sweep_2026-08-14.json`（SHA `8a35ca5a5724f94d8ac979e7746731be6829a97e60d56cdd1e10910c44fabf8d`）。
- [ ] v33_rv 仍不构成三场全量因果真值，且没有新增可训练 shot 正例；不启动无依据的重训或融合调参。继续插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF，readiness `not_ready`、blind inference 继续暂停。

## 2026-08-14 v34_rv residual source-disjoint 复核

- [x] 在 1,456 个历史源内排除锚点外继续抽取剩余合法窗口：Randolph 2 个、VTV 2 个，共 4 windows/84 张 hash-bound 原片帧；v34 held-out artifact SHA `2a12bb73ae9ed072d43684b3612a55992fb63b1664f1d7daeb24ef05eb2c2d25`，review-plan SHA `b3dbd70a6f776b5f9da1c178ab33e6672d72351b891a6c1f5cefb8c8c54fced4`，raw-frame manifest SHA `49e051773a91d84773ea78f7ef1d081b4ab65138f294715cec6c2dc5220b58b4`。
- [x] 完成原片逐帧人工复核并 fail-closed 封存 2 `not_a_shot/not_applicable`、2 `uncertain/unknown`、0 `shot`；sealed SHA `736a8f6748a1ab776af18d3a83e38bcbf5f4746e5d1ca313d03a97acc55885fa`，pilot SHA `58fb19a66a0b9930d621ca5da3f9ab6db2d8d6d624f868d5631b2ff345a991ed`，retention SHA `1014c9cd60a252e898e95ddf1f53a0a9d29369b05f772e2bb8a04cf786ff9682`。
- [x] v34_rv 保持 `pilot_only`、`exhaustive=false`、`training_consumable=false`、`runtime_consumable=false`、`hard_negative_gate_eligible=false`；canonical `.venv` focused causal-review regression `13 passed`，不进入训练、runtime、VLM 默认答案、融合、晋级或 blind inference。
- [x] 复核后按精确 allowlist 删除 4 张可重建 contact sheet（`2,111,920` bytes），审计 `analysis_outputs/public_research/agu_v34_contact_sheet_cleanup_2026-08-14.json`；raw frames、review/manifest/retention 正式工件与原片保留，项目侧未留下缓存。
- [ ] v34_rv 没有新增可训练 shot 正例，且 Hazen 在 15 秒排除半径下仍耗尽；不启动无依据的重训或融合调参，继续构建三场全量因果标签、插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF，readiness `not_ready`、blind inference 继续暂停。

## 2026-08-14 v35_rv VTV residual source-disjoint 复核

- [x] 在所有已封存 review-plan/batch 锚点之外继续使用 15 秒源内排除协议；selection anchor manifest 记录 Hazen/Randolph/VTV 共 232/331/334 个去重锚点（SHA `626bd0dd9be94f5e33543c4dc8cefd7a4eb3f671507eada89a55c48e0afbf50f`），从 VTV 选择 8 个未复核窗口，held-out batch artifact SHA `9369ffce6947d98e0c30b712b16f6fa3deae8b7499e55d27d9e19ed1587a2545`。
- [x] 生成 label-hidden review spec/plan，并物化 8 windows/168 hash-bound 原片帧；review-plan SHA `856dd0a1ff4ce49023b248e890a4c88d7b50cc89dd41c46c2dfb7ced08ef6978`，raw-frame manifest SHA `74e2961567a9a5717e7c8cf38c2a1f5c8467e59517cdde8fab6c166ad726f513`。
- [x] 完成原片逐帧人工复核并 fail-closed 封存 7 `not_a_shot/not_applicable`、1 `uncertain/unknown`（VTV 5622s；近篮动作可见但 release boundary/完整 outcome 不可见）、0 `shot`；sealed SHA `b2dc36acd9eb2e48cedd19f29c3ecb1206bfe17b77315dc1fab337784d1133f8`，pilot SHA `b920d477ad2a7f444b4b226832b031eed353df8dde1868a072d77e4daa8f8fa9`，retention SHA `e0ba7c3d81dedc42f2cf853e2df18a3938cdb4e303bf9aca14fb3ec6e7e5e406`。
- [x] canonical `.venv` causal-review contract regression 保持 13 项通过；复核后按精确 allowlist 删除 8 张可重建 contact sheet（`3,507,319` bytes），审计 `analysis_outputs/public_research/agu_v35_contact_sheet_cleanup_2026-08-14.json`（audit SHA `fddadd3dcc84fc3d5aa1d6d5d9e6c16ea7b0eb392b7c7d31978b22f226f40627`）；回归后再删除 20 个项目级 bytecode/cache 文件（`168,942` bytes），审计 `analysis_outputs/public_research/agu_v35_regression_cache_cleanup_2026-08-14.json`（audit SHA `58c39123fe2afa05784303e9b6c39819c59e4a831ec1ab63130b765ec00e03b5`），raw frames/正式 review/manifest/retention/原片保留。
- [ ] v35_rv 仍为 `pilot_only`、`exhaustive=false`、`training_consumable=false`、`runtime_consumable=false`、`hard_negative_gate_eligible=false`；没有新增可训练 shot 正例，不启动无依据的重训或融合调参，继续三场全量亚秒 causal truth、插播边界、穷举 hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF，readiness `not_ready`、blind inference 继续暂停。

## 2026-08-14 v36_rv VTV residual source-disjoint 复核

- [x] 在 v35_rv 锚点之外按 15 秒源内排除半径，从 VTV 按 early/middle/late 各 8 个 label-hidden 窗口抽取 24 个窗口，物化 504 张 ±2 秒 hash-bound 原片帧；held-out batch SHA `589908cf67f0ef6d44c8b77e206f5703bf43df0cff8b131c7a1d8dfc82680724`，anchor manifest SHA `4b302c0ce70cafd191911b8ac7261f4b1a03fdffb6d8ba1f4bc288d4470d0be9`，review-plan SHA `bb6d325f48c4eaf4832678788664fbabf73c7f5d793cc73aaa8f1c259d3a162d`。
- [x] 原片逐帧人工复核并 fail-closed 封存 23 `not_a_shot/not_applicable`、1 `uncertain/unknown`、0 `shot`；sealed SHA `f9602163a95585263a362b16c03b5fceb9c6b1bb33976155979733741a2e1a09`，pilot SHA `ef3765155ecf6f055da70ae61457feffb6eeacbe24e5c9038ba75d75d9d597ee`，retention SHA `04dfc1d5ff499405d78bf1c18f1419dd16bde83997ac1ff984064fc76d5d5215`，raw-frame manifest SHA `c5719797e739a88a43020a101cff32cb5b01bfe08f99fa6bae19804731e9ecba`。
- [x] v36_rv 保持 `pilot_only`、`exhaustive=false`、`training_consumable=false`、`runtime_consumable=false`、`hard_negative_gate_eligible=false`；canonical `.venv` causal-review regression `13 passed`，不进入 AGU 训练、runtime、VLM 默认答案、融合、晋级或 blind inference。
- [x] 记录覆盖余量：Hazen `0`、Randolph `0`、VTV 尚余 `204` 个队列窗口；coverage audit `analysis_outputs/public_research/agu_v36_rv_annotation_coverage_2026-08-14.json`（SHA `3e506cb1e2c06d7f936665146a34a17b314190908b628e80f74fa1678cc436a5`）。复核后删除 24 张 contact sheet 与 5 个 montage/manifest 派生文件，共 `29` 文件/`15,140,690` bytes，审计 `analysis_outputs/public_research/agu_v36_contact_sheet_cleanup_2026-08-14.json`（SHA `78388e5cbaa9acebc90083c6a38ee339e4452b49b61552c966c205c4bffd7db0`）；回归后另按精确 allowlist 删除 20 个项目级 cache/bytecode 文件、`171,461` bytes，审计 `analysis_outputs/public_research/agu_v36_regression_cache_cleanup_2026-08-14.json`；raw frames、正式工件、原片与 `.venv` 保留。
- [ ] v36_rv 没有新增可训练 shot 正例；不启动无依据的重训或融合调参，继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF，readiness `not_ready`、blind inference 继续暂停。

## 2026-08-14 v37_rv VTV residual source-disjoint 复核

- [x] 在 v36_rv 全部源内锚点外按 15 秒排除半径，从 VTV early/middle/late 各抽取 8 个 label-hidden 窗口，共 24 windows/504 张 ±2 秒 hash-bound 原片帧；held-out batch SHA `5237edd2383ce7bafc1f5fd6ed3ea0083b0636720fe71702ad18308a4379bd0d`，anchor manifest SHA `c2d708a9441b1e10af2e23f3d697a6c44a8eeca6eedb80720a2d4fa5192eeee3`，review-plan SHA `82866573ea39b721a60ad2c79f0a53646398006098bc8902e3a616c25c3f225f`。
- [x] 原片逐帧复核并 fail-closed 封存 22 `not_a_shot/not_applicable`、2 `uncertain/unknown`（VTV 5622s/5852s：release boundary 或镜头切换后完整因果链不可见）、0 `shot`；sealed SHA `6251db4a641c99d9ce64e52efb82ee63f376ac434774f5dd275e7e07ac155d37`，pilot SHA `6b3b814c2986fabfa80121f733b50e42523f0206f7e7bff75d39cf59daed71dc`，retention SHA `22aa075f4759d2ce72f2319f2669da84cff2eaa01ec3a3bee53692fea73ff3de`，raw-frame manifest SHA `0a809de26bdd641087c98bf5e25374a7d1359a307db748ec47e7384f43e3a210`。
- [x] v37_rv 保持 `pilot_only`、`exhaustive=false`、`training_consumable=false`、`runtime_consumable=false`、`hard_negative_gate_eligible=false`；canonical `.venv` causal-review regression `13 passed`，不接入 AGU 训练、runtime、VLM 默认答案、融合、晋级或 blind inference。
- [x] 记录覆盖余量：Hazen `0`、Randolph `0`、VTV `157` 个源内未复核队列窗口；coverage audit `analysis_outputs/public_research/agu_v37_rv_annotation_coverage_2026-08-14.json`（SHA `70504ae97479973a91d4599f45aa1e169598f7f88e1faa407644f3dc0bf3bfdb`）。复核后按精确 allowlist 删除 24 张 contact sheet 与 5 个 montage/manifest 派生文件，共 `29` 文件/`14,742,074` bytes，审计 `analysis_outputs/public_research/agu_v37_contact_sheet_cleanup_2026-08-14.json`（SHA `a7e827042c5c5bcb3a871c0ff992b2228e6dd5a010d11c402dc9232831db6406`）；回归后另删除 19 个项目级 cache/bytecode 文件、`162,651` bytes，审计 `analysis_outputs/public_research/agu_v37_regression_cache_cleanup_2026-08-14.json`（文件 SHA `1211069b3709be04d785f56cfb09d9db2208ebd2eb19eb411a8f56f2eae788b1`）；raw frames、正式工件、原片与 `.venv` 保留。
- [ ] v37_rv 没有新增可训练 shot 正例；不启动无依据的重训或融合调参，继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF，readiness `not_ready`、blind inference 继续暂停。

## 2026-08-14 v38_rv VTV residual source-disjoint 复核

- [x] 在 v37_rv 全部源内锚点外按 15 秒排除半径，从 VTV early/middle/late 各抽取 8 个 label-hidden 窗口，共 24 windows/504 张 ±2 秒 hash-bound 原片帧；held-out batch SHA `3f139af3c655635d4454d6592a8969680b558ae7db2cade5db5ef718e824dcac`，anchor manifest SHA `13c6eefeb6d50eaca318c1835610e86e6b87ac4b7f283a2b808050ee91d06f17`，review-plan SHA `8998d037d156b92a59178b2cc80b55d6e927dbfe21f92f6b5d45fd3807cc674c`。
- [x] 原片逐帧复核并 fail-closed 封存 23 `not_a_shot/not_applicable`、1 `shot/unknown`（VTV 9362s：持球—脱手—向篮筐运动—篮筐/篮网平面可见，但 made/missed 不可安全解析）；sealed SHA `e70a729df42bf052dfc628a7aeed6a08fa4fa62fd90da19ef7a1573c52a1e100`，pilot SHA `a3c15f0daa15639aaa16dcd1db56bc8da4ecc7a617a3dd68c6a2eefdbdd62783`，retention SHA `df5ba13a93a6bb9904b80c8c171f77b34607d4368618a10481e3a3895c9cd99e`，raw-frame manifest SHA `78a01c24e8435af4adb4cc848133a02a54b4d7e19b14a2bb487ec23d5e322062`。
- [x] v38_rv 保持 `pilot_only`、`exhaustive=false`、`training_consumable=false`、`runtime_consumable=false`、`hard_negative_gate_eligible=false`；canonical `.venv` causal-review regression `13 passed`，不接入 AGU 训练、runtime、VLM 默认答案、融合、晋级或 blind inference。
- [x] 记录覆盖余量：Hazen `0`、Randolph `0`、VTV `115` 个源内未复核队列窗口（early `39`/middle `29`/late `47`）；coverage audit `analysis_outputs/public_research/agu_v38_rv_annotation_coverage_2026-08-14.json`（SHA `ccb6b434fda37768b04eed3d6a281562f2ce73d5c9fc1ecbb4b3f84be7deb0de`）。复核后按精确 allowlist 删除 24 张 contact sheet 与 5 个 montage/manifest 派生文件，共 `29` 文件/`12,958,436` bytes，审计 `analysis_outputs/public_research/agu_v38_contact_sheet_cleanup_2026-08-14.json`（audit SHA `22b9b8dacad7a66f6641062ae3a8224b287470cfaaa63a4c3be3c1e180d77c76`）；回归后另删除 19 个项目级 cache/bytecode 文件、`162,651` bytes，审计 `analysis_outputs/public_research/agu_v38_regression_cache_cleanup_2026-08-14.json`（文件 SHA `53655e4e1da0d9a76ede69983dd4640f1f4ef6e1bcffa0b5f0782ae1a7559598`）；raw frames、正式工件、原片与 `.venv` 保留。
- [ ] v38_rv 没有新增可训练 made/missed shot 正例；不启动无依据的重训或融合调参，继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF，readiness `not_ready`、blind inference 继续暂停。

## 2026-08-15 v39_rv VTV residual source-disjoint 复核

- [x] 在 v38_rv 全部源内锚点外按 15 秒源内排除半径，从 VTV early/middle/late 各抽取 8 个 label-hidden 窗口，共 24 windows/504 张 ±2 秒 hash-bound 原片帧；held-out batch SHA `00aff1cf36b78d2088a652e9ba2a6eec6e82964a47c257dabe4d9a36fd5c7f30`，anchor manifest SHA `2051f01bd9fb2617735c8acb6549e87d97ce9a5f1d0e52a4c363f7cfcd331a70`，review-plan SHA `bb39767f72b81195ec616b9b29b30beffce7dcef5ec9a71131c8c6622c654c08`。
- [x] 原片逐帧人工复核并 fail-closed 封存 24 `not_a_shot/not_applicable`、0 `uncertain/unknown`、0 `shot`；sealed SHA `55a8a603947ab5bca1d15fcb3181b508923934ec96fba28c2f23f5b4894aa89d`，pilot SHA `5cd734320330a2942d2a5206b8f96fa0bb7bcfd0fb5b8ee47f369ee5076789fc`，retention SHA `4fe24cdb3131c5ba917259bd98c0f4870e0f0f738f4cfc053b2c7d4b3f6ebabf`，raw-frame manifest SHA `f3ad06812a4adb9ff414eee3b52ae467c42c9d8086fa8f2e5642465f1a01ba9`。
- [x] v39_rv 保持 `pilot_only`、`exhaustive=false`、`training_consumable=false`、`runtime_consumable=false`、`hard_negative_gate_eligible=false`；canonical `.venv` causal-review regression `13 passed`，不接入 AGU 训练、runtime、VLM 默认答案、融合、晋级或 blind inference。
- [x] 记录覆盖余量：Hazen `0`、Randolph `0`、VTV `80` 个源内未复核队列窗口（early `25`/middle `20`/late `35`）；coverage audit `analysis_outputs/public_research/agu_v39_rv_annotation_coverage_2026-08-15.json`（audit SHA `4906bc93492625de2d2ac88e2d95310ce741bb94abcb51017f54f96d6279ca36`）。复核后按精确 allowlist 删除 24 张 contact sheet 与 5 个 montage/manifest 派生文件，共 `29` 文件/`15,334,391` bytes，审计 `analysis_outputs/public_research/agu_v39_contact_sheet_cleanup_2026-08-15.json`（audit SHA `cb29db3ea5d11bb1bdb5d09a59ffb91fa57626dd658a244ad081079e8291d9bf`）；回归后另删除 19 个项目级 cache/bytecode 文件、`162,651` bytes，审计 `analysis_outputs/public_research/agu_v39_regression_cache_cleanup_2026-08-15.json`（文件 SHA `7b0d93fcde223e8d5ef5f82de517eda9d70a6cdff6937ce9616e451ca032a3ee`）；raw frames、正式工件、原片与 `.venv` 保留。
- [ ] v39_rv 没有新增可训练 made/missed shot 正例；不启动无依据的重训或融合调参，继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF，readiness `not_ready`、blind inference 继续暂停。

## 2026-08-15 v40_rv VTV residual source-disjoint 复核

- [x] 在 v39_rv 全部源内锚点外按 15 秒源内排除半径，从 VTV early/middle/late 各抽取 8 个 label-hidden 窗口，共 24 windows/504 张 ±2 秒 hash-bound 原片帧；held-out batch SHA `9b149844f4ed3848110b99c693d80e6a8ee5545285d56407b44daa32a187989b`，anchor SHA `c9d1646f142550f95bbb148162070127d14f7eed517b4673420cdc1d5c45fcd5`，review-plan SHA `19261ec72b02f8f19118a36a35b74e281439125f045baa275c9230009944bb82`。
- [x] 原片逐帧人工复核并 fail-closed 封存 22 `not_a_shot/not_applicable`、2 `uncertain/unknown`（VTV 1332s 为近篮动作但 release/rim 结果链不完整；2362s 从篮筐接触镜头开始，前置 release 不可见）、0 `shot`；sealed SHA `fdf50e0ab286eced877f3098abb0c2e994edbe4b993f0e11c59734156ecd74d5`，pilot SHA `3a5c2f90bc36934d9010240f88bd079fa140d0611c8fe39733905d062fba6ae3`，retention SHA `bbdfcddc7027a585f809c20473e072b916bf14e1f83ea307d4fa1424fcf60cea`，raw-frame manifest SHA `925772ab2d1e8fb6aa797acc2e8e3213aac8ffaa3f5b7c3a0a9a44eca58124e3`。
- [x] v40_rv 保持 `pilot_only`、`exhaustive=false`、`training_consumable=false`、`runtime_consumable=false`、`hard_negative_gate_eligible=false`；canonical `.venv` causal-review regression 为 `74 passed`（13 个聚焦文件），不接入 AGU 训练、runtime、VLM 默认答案、融合、晋级或 blind inference。
- [x] 覆盖审计 `analysis_outputs/public_research/agu_v40_rv_annotation_coverage_2026-08-15.json`（audit SHA `9406785b98e342c9db992b6f4a43413636c1bb4df9bcc10a538c26100ce8a39a`）显示 Hazen/Randolph 余量 0、VTV 余量 `47`（early 16 / middle 9 / late 22）；复核后按精确 allowlist 删除 24 张 contact sheet 与 5 个 montage/manifest 派生文件，共 29 文件/`15,682,641` bytes，审计 `analysis_outputs/public_research/agu_v40_contact_sheet_cleanup_2026-08-15.json`（audit SHA `abdb01fd33bbb45d7806a9ea54034a2134acb300fcc3838258c90f8eff9d3b42`）；回归后另删除 67 个项目级可重建 cache/bytecode 文件、`1,250,894` bytes，审计 `agu_v40_regression_cache_cleanup_2026-08-15.json`（audit SHA `3b37f486c930c4429e639cbdcfd4283c80617ab22bd5da4f70e4dd3420dd8986`）；raw frames、正式工件、原片与 `.venv` 保留。
- [ ] v40_rv 没有新增可训练 made/missed shot 正例；不启动无依据的重训或融合调参，继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF，readiness `not_ready`、blind inference 继续暂停。

## 2026-08-15 v41_rv VTV residual source-disjoint 复核

- [x] 在 v40_rv 全部源内锚点外按 15 秒源内排除半径，一次性复核 VTV 剩余 47 个合法窗口（early 16 / middle 9 / late 22），物化 987 张 ±2 秒 hash-bound 原片帧；batch SHA `55c43c871bbf622daba14dd43316f7a903bac966b53ef9d803454f31bb009862`，anchor SHA `ea67bc22ffc0585e7834e3f5fa720b546728b74aca54b6ff7a619c9a0483fd63`，review-plan SHA `845991c4c223dba5143351473816f4b4b76d6c6cee6a08a98302c90cb27a2ddd`，raw-frame manifest SHA `8c61c23947111928f1eec9bfba9170b1e5c088f4bf41ff66c522e8d2490f76cc`。
- [x] 完成原片逐帧人工复核并 fail-closed 封存 41 `not_a_shot/not_applicable`、6 `uncertain/unknown`、0 `shot`；6 个 uncertain 均为 replay/近篮动作但 release boundary 或完整 rim/outcome 链不可验证。sealed SHA `836d8b346d0edb1e05b138e639d32e8a0ecd0fd53613210b1d9d877fa7a9206b`，pilot SHA `4ff46cfc89d89301caef2a138b2aefb9afd8bf2f060a47de00ffeb2d5551f471`，retention SHA `87f5a50c9d23db6bb4ce189553a0bc4b2caed2d12c76775b8db0670dba54413f`。
- [x] v41_rv 保持 `pilot_only`、`exhaustive=false`、`training_consumable=false`、`runtime_consumable=false`、`hard_negative_gate_eligible=false`；canonical `.venv` causal-review regression 为 `74 passed`（13 个聚焦文件），不进入 AGU 训练、runtime、VLM 默认答案、融合、晋级或 blind inference。
- [x] 覆盖审计 `analysis_outputs/public_research/agu_v41_rv_annotation_coverage_2026-08-15.json`（audit SHA `3f3458cd993bc6dc93c024a2344edaf087454b5e6cc190766e6f34b1afc6fe7b`）显示 Hazen/Randolph/VTV 余量均为 0；复核后按精确 allowlist 删除 47 张 contact sheet 与 8 个 montage/manifest 派生文件，共 56 文件/`25,321,919` bytes，审计 `agu_v41_contact_sheet_cleanup_2026-08-15.json`（audit SHA `416f9e19306ecd5c16603b5e496175bd5bcadfb33d8399c143548effd0f64b5c`）；回归后再删除 67 个项目级可重建 cache/bytecode 文件、`1,250,894` bytes，审计 `agu_v41_regression_cache_cleanup_2026-08-15.json`（audit SHA `b73d5b4be37cf78c980cbfccac83e50dc345678fa2b88826e8e03176c53b9f6b`）；raw frames、正式工件、原片与 `.venv` 保留。
- [ ] v41_rv 没有新增可训练 made/missed shot 正例；不启动无依据的重训或融合调参，三场全量亚秒球—手—篮筐—结果、插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF 仍未完成，readiness `not_ready`、blind inference 继续暂停。
## 2026-08-15 nested outer-game-held 融合证据完整性 v2

### 背景与风险

当前 `shot_validity_scene_fusion` v1 虽然对特征模型生成了 outer-game-held 概率，
但每场 `best_precision_at_recall_0_85` 阈值仍使用该 held game 自身标签选择；
`fuse_vlm_with_frozen_auxiliary` 随后把该阈值应用回同一场，并声明未使用目标标签。
此外，legacy auxiliary OOF 行缺少 `candidate_bundle_sha256`，也没有把 VLM plan、
training manifest 与三元事件键完整绑定。当前 0.625/0.625 的拒绝结论不变，
但旧协议不能证明未来任何达到 0.85 的结果。scikit-learn 1.9 官方 nested-CV 指南明确要求
inner loop 负责模型/超参数选择、outer loop 只估计泛化误差；比赛必须作为不重叠 group。

### 任务与验收

- [ ] **TDD RED：泄漏与绑定回归。** 新增测试证明翻转一个 outer-held game 的标签不会改变
  该 fold 的阈值或预测；held game 不得进入 inner selection/model-fit groups；辅助 OOF 必须包含
  `(source_video_sha256, candidate_bundle_sha256, event_id)`；plan/manifest/key 不一致与 legacy v1
  threshold artifact 必须 fail-closed。验证：focused pytest 必须先因缺少 v2 实现而失败。
- [ ] **GREEN：nested outer-game-held v2。** 在 outer-train games 内用 inner leave-one-game-out OOF
  选择预声明 variant 与阈值，再只用 outer-train 拟合并预测 outer-held；工件封存每 fold 的
  selection/fit groups、三元键、training manifest、阈值、outer prediction 与逐场/pooled P/R。
  默认保持 training-only、runtime false、Codex runtime answer false。
- [ ] **冻结融合验证。** `fuse_vlm_with_frozen_auxiliary` 只接受哈希完整、plan-bound、manifest-bound
  的 v2 auxiliary artifact；legacy v1 不再可被标记为 label-free frozen gate。缺少覆盖继续 abstain，
  但正例 abstain 在事后评估中必须计 FN。
- [ ] **受守护真实诊断。** 用现有 batch4 511-window scene+Swin3D+MViT embeddings 运行 v2 nested screen，
  通过 canonical `.venv` 和资源守护记录 CPU/内存；不得改 runtime/default/checkpoint/blind assets。
- [ ] **封存与后续。** 更新 current solution、datasets/task board/readiness 说明和 LLM Wiki；旧 v1 指标标为
  provenance-only/non-promotable。若 v2 未达到 pooled 与每场 P/R >= 0.85，继续新的 plan-aligned VLM
  原片复核与 VTV score-change 正例挖掘，不恢复盲推理。

### 检查点

- [x] focused tests、Ruff、`git diff --check` 通过；真实 v2 artifact 哈希可重算。
- [x] outer-held label invariance、selection-group exclusion、plan/manifest/triple-key binding 均有测试证据。
- [x] 磁盘、AGU 进程与资源守护复核完成；仅清理已确认可重建 cache/派生产物，无 AGU 训练/服务进程遗留。

## 2026-08-15 nested fusion v2 与 24-window causal closure

- [x] 将 scene/video 辅助融合升级为真正的 nested outer-game-held v2：inner games 冻结 variant/threshold，outer held game 只做泛化评估；plan、training manifest、三元键、embedding source SHA、维度、PCA/C 与晋级策略均前置冻结并 fail-closed 校验。
- [x] 独立 VLM frozen consumer 要求外部冻结的 auxiliary artifact SHA，固定官方 observables，拒绝标签字段、不可用 VLM 行、伪造 metrics 与 plan/schema 漂移；legacy v1 路径固定不可晋级。
- [x] 通用 VLM v1 evaluator 已诚实降级为 `open_world_structural_only` legacy diagnostic：开放 schema 不宣称证明 label-free/Codex independence，所有输入永久 `promotion_eligible=false`。plan internal、prediction internal 与 prediction file 外部收据分别核验/记录；CLI prediction-first 单次字节快照、truth 单读、严格 JSON/布尔/门槛、重复 key 与路径别名防护均 fail closed。正式晋级只能另建 closed-schema consumer，不得复用 generic v1。
- [x] 冻结 v3–v41 因果闭环 selection（24 个互不重叠的 8 秒窗口/每源 8 个），生成 64 帧半开采样 plan，物化 1,536 张 hash-bound 原片帧并完成 labels-hidden 离线逐帧复核。
- [x] fail-closed 封存 14 `shot`、8 `not_a_shot`、2 `uncertain`；outcome 为 5 made / 8 missed / 3 unknown / 8 not-applicable。review internal SHA `37f85d6078a26dc5862f755de96228e18a315558afba187bdf2f51d4d6189447`。
- [x] 六个本轮工件 CLI 均在读取/训练前冻结 resolved 输入输出并拒绝别名覆盖，使用同目录随机临时文件、`fsync`、原子 replace；sealed-review verifier 也会重验精确字段、覆盖、帧映射和因果语义。
- [x] canonical `.venv` 最终联合回归 `313 passed`，focused Ruff 与 `git diff --check` 通过；v12 历史正式工件与当前 closure 均兼容。
- [ ] 当前真实 nested v4 仍为 pooled P/R=`0.5556/0.8333`，且逐场存在 P/R 低于 0.85；不恢复 blind inference，不更新 readiness/default，继续补充独立连续因果真值并做 game-held OOF。

## 2026-08-15 closure 显式训练桥接与 MViT nested 表征筛查

### 目标与边界

- [x] 审计现有训练入口：closure selection/plan/review 的
  `training_consumable=false` 禁止直连 trainer，但允许经显式、SHA-bound 转换层形成
  training-only 派生工件。旧 ExtraTrees 依赖真实 detector/candidate 特征，本批次禁止进入该路由。
- [x] 冻结数据映射：`shot -> event_present=true`、`not_a_shot -> false`；排除 2 个
  `uncertain`，得到 22 行（14 正/8 负）。`outcome`、备注、人工 release/rim 位置不得进入
  candidate geometry、sampling anchor 或特征。
- [x] 仅做 `development_diagnostic_only`：输出必须 `runtime_consumable=false`、
  `formal_evaluation_eligible=false`、`traditional_feature_training_eligible=false`；本批次不可更新 runtime/default/
  readiness，不可恢复 blind inference。

### 实现与验收

- [x] **TDD RED。** 先覆盖 selection/plan/review/raw-frame manifest/原片 SHA 任一漂移、
  frame coverage/path/hash 不一致、uncertain 误映射、outcome/release/rim 泄漏、输出覆盖输入与
  非原子写入；确认新接口未实现时测试先失败。
- [x] **最小 GREEN。** 新增 `agu.vru-causal-shot-validity-training-export.v1`，一次验证
  selection/plan/sealed review/raw-frame manifest/三个原片，导出每源 label-hidden geometry candidate bundle、
  `agu.shot-validity-labels.v1` 和 batch index；写入全链 SHA、mapping policy、excluded IDs 与路由允许列表。
- [x] **Training manifest。** 复用 `seal_training_annotation_manifest`，绑定 3 个原片、3 份 labels 与
  已封存 acceptance bundle；确认 benchmark raw SHA 集合与三个训练源不相交。
- [x] **受守护表征实验。** 优先复用本地 `model_checkpoints/mvit_v2_s-ae3be167.pth`提取
  22 个 8 秒窗口的固定 embedding；三源 outer-held、outer-train 内 inner LOGO 冻结队列和阈值，
  不在 held label 上选阈值/超参。使用 `.venv` 与 `run_guarded_training.py`，记录 CPU/内存。
- [x] **评审与收尾。** focused + broader pytest、Ruff、`git diff --check`、真实工件重验、
  fresh-context 对抗审查、llm-wiki post-hook 和 TASK-0254 文档更新；仅删除本轮可重建临时产物。

## 2026-08-15 TASK-0255 Swin3D-T 独立表征诊断（结果前冻结）

- [x] 冻结输入与两种且仅两种诊断：本地官方 Swin3D-T（checkpoint SHA
  `7615ae035996b65eb38dad437ae533d2dfcd36f9f89d28c0f0fa7bfb8e6b3130`）单骨干，以及
  MViT-V2-S + Swin3D-T 1,536 维拼接；固定 `C=0.01` 和现有 nested source-held 阈值协议。
- [x] 在 `.venv`、batch size 1、MPS 与 sustained CPU/memory/swap guard 下提取 22 行
  Swin3D-T embedding；若 exit 75，只从验签 checkpoint 续跑。
- [x] 分别封存并验证 Swin-only 与 MViT+Swin 两个 probe，记录 pooled/逐源 P/R/F1、资源峰值和 SHA。
- [x] 若任何逐源 precision 或 recall 低于 0.85，或数据仍为 prior-stratified pilot，保持
  `runtime/formal/promotion/promoted=false`、readiness `not_ready`、blind inference paused。

## 2026-08-16 TASK-0256 HCTV–Harwood 第四比赛扩展（已归档）

- [x] 下载并验签 Commons/HCTV Hazen–Harwood 2026-01-22 全场原片，记录 CC BY 4.0、原始
  YouTube CC 声明、字节、SHA、ffprobe 和抽帧证据。
- [x] TDD 实现单一 label-hidden 选择协议：30 秒候选网格、early/middle/late 各 8、SHA 排序、
  24 个互不重叠 8 秒窗口、8 FPS/64 帧；选择前不读取标签、PBP 或模型分数。
- [x] 用严格 selection/plan/raw-frame/review receipt 链完成 1,536 帧人工复核，封存 3
  `shot/missed`、20 `not_a_shot`、1 `uncertain`；不确定项不映射为负例。
- [x] 将四比赛 SHA-bound training export/manifest 与 nested MViT+Swin diagnostic 明确拆分为
  TASK-0257；不在 TASK-0256 内修改三源 v1 契约或谎称训练已完成。
- [x] focused/full regression、Ruff、资源/磁盘/进程、fresh review、docs/Wiki 后置归档；清理
  1,584 个 byte-identical/rebuildable 文件、`715,927,744` bytes，正式 v2 evidence 保留。

## 2026-08-16 TASK-0257 Harwood additive training bridge（development diagnostic 已验证）

- [x] 冻结 additive v2 规范：三源 v1 作为不可变父收据，Harwood 作为增量证据；不得改写 v1
  schema/API/工件。
- [x] RED→GREEN 实现并验签 `agu.vru-causal-source-groups.v1` 与
  `agu.vru-causal-shot-validity-training-export.v2`，解析 48 selected → 45 exported（17+/28-），
  三个 uncertain 全排除。
- [x] 新封四视频 training manifest，并在 batch size 1、canonical `.venv`、sustained resource guard
  下完整提取 45 行 MViT/Swin embeddings；不得复用不匹配的旧 resume。
- [x] 运行四 outer-game-held nested v2 probe；所有 variant/preprocessing/C/threshold 只由 outer-train
  的 inner LOGO 决定，报告四比赛与两个制作域，但不得声称 production-held。
- [x] focused/full regression、TASK-0257 scoped Ruff、工件重验、进程/磁盘检查、fresh review
  与 docs 归档完成；cross-model review 因未获用户显式授权未运行。
- [x] 执行 TASK-0257 llm-wiki post-development hook；全仓 Ruff 历史基线债务另行治理，
  不在本任务批量格式化。
- [x] 无论开发指标如何，所有 runtime/formal/promotion/promoted 标志保持 false；readiness `not_ready`、
  blind inference paused，正式 85% 门禁不变。

## 2026-08-16 TASK-0258 temporal diagnostic and independent canary（Phase-0）

- [x] 用户于 2026-08-16 批准 `docs/specs/TASK-0258-temporal-canary/capability-map.md`；
  该批准只授权 Module-A 规格起草与审查，不提前下载 Module-B 数据，也不越过 map 的当前-spec 用户批准门禁。
- [x] 冻结并 fresh-context 审阅 `existing-45-temporal-retrospective` 的 requirement/solution/gate；最终
  Critical/Required/Optional=`0/0/0`。精确 SHA tuple 为 requirement
  `5c455408bd9ba730470b66189316578dc935b39a5cdc550c67e2a9c08d3f80d3`、solution
  `87f9a447112de8d294ffedbb1adfadb478713d94e00cdcf2e4f98999ea8a857d`、gate-review
  `cc52d809f42abba25ccdfb6667325612ce2f8961842f0e375db060ea2b70fa8f`。
- [x] 用户明确批准上述 exact-SHA Module-A 规格；批准后生成的外部收据 internal/file SHA 为
  `42273c9db3e5c77da8f577ed0e5142e6210faf3caee53f572402229eac71944f` /
  `0eb1759d992b9587ad47bf9ae08793a69bac266ada81d39258919ac61e06e9fb`。更早的 v1 收据与运行
  均发生在这次明确批准之前，只保留为不可覆盖的历史诊断，不能作为已授权执行证据。
- [x] 按 RED→GREEN 实现唯一 `swin3d-t-tiled-4x2s-mean-delta-v1` 变量、受控提取器、双 evaluator、
  原子 terminal transaction 与外部 receipt verifier；focused `67 passed`，相邻链 `523 passed, 3 skipped`。
- [x] 批准前的真实 batch-1/MPS v1 运行私下完成 45/45 行和 49 个资源采样，但最终锁内磁盘复验时可用空间
  降到约 3.42 GiB，低于 `1 GiB worst-case + 3.5 GiB reserve`，因此没有发布 embeddings/final evaluator。
  已原子封存 `terminal_failure_v1`；其旧代码 `stop_reason=determinism_failure` 是已确认的分类缺陷，
  实际来源为 disk reserve，后续代码已用 `DiskWriteBudgetError -> disk_failure` 回归修正，终局工件不覆盖。
- [ ] 等待不同 fresh-context implementation review 与用户另行明确决定是否新开版本重跑；当前 v1 terminal
  outcome 不可覆盖，Module B 未授权且不得进入，readiness 与 blind inference 不变。

## 2026-08-16 TASK-0259 local YOLO checkpoint resolution

- [x] 复现本地服务因默认裸文件名 `yolov8n.pt` 触发 GitHub 下载并以 curl 35 失败。
- [x] 以 RED 测试冻结仓库内保留 checkpoint 的默认路径，并保留环境变量显式覆盖。
- [x] 将 Settings、direct tracking API 与 `.env.example` 统一为
  `model_checkpoints/yolov8n.pt`；focused `2 passed`，tracking/service 相邻回归
  `101 passed, 1 warning`。
- [x] 重跑 local curl hook：health/ready/run/status 均符合契约，同一 60-frame、VLM-off 请求约
  9.46 秒完成，未再联网下载。该修复不改变模型、VLM、readiness 或盲推理门禁。

## 2026-08-16 TASK-0260 four-game independent VLM resource probe

- [x] 机械盘点 TASK-0257 四个 bundle 的 48 个三元事件键；28 份既有 VLM 工件、667 行预测与其
  精确交集为 0，禁止换绑旧预测。
- [x] 冻结 16-window EBQwen MLX 4-bit native plan；在 90% system-memory、2 GiB available-memory、
  0.25 GiB free-swap 边界下运行，模型加载阶段连续三次越界并以 exit 75 安全停止，零预测输出。
- [x] 冻结更小的 8-window Qwen3-VL 2B frame-sampled plan；同一 guard 在完成 1/8 cache 后因
  available memory 连续三次低于 2 GiB 以 exit 75 安全停止，并显式停止 Ollama 模型。
- [x] 不把单条 cache 当作 prediction artifact；不评测、不融合、不晋级，不降低资源阈值。
- [x] 清理可重建依赖/视觉派生物与重复 checkpoint 约 2.34 GB；保留原片、模型权重、标签、封存审阅
  和 raw evidence。最终实时可用空间约 4.60 GiB，readiness `not_ready`、blind inference paused。

## 2026-08-16 TASK-0261 SmolVLM independent-video capability screen

- [x] 固定在线候选范围为 MLX SmolVLM2 256M 与 500M 8-bit，核验公开 revision、Apache-2.0、文件大小与上游权重 SHA；不降低 TASK-0260 资源边界。
- [x] 以 TDD 补齐 SmolVLM processor 方形像素预算和按实际帧数注入 image token 的兼容路径，并新增最小 `compact_json_v1` 证据契约。
- [x] 在四比赛 8-window 固定计划上完成 256M guarded screen；资源门禁通过，但 negative-first 0/8 可用，compact 4/8 可解析且 0 个 non-unknown，拒绝评测/融合/晋级。
- [x] SHA-first 删除无帮助的 256M 权重 `517,925,791` bytes；500M 首次续传文件大小/SHA 均不匹配上游，未加载即拒绝并删除损坏目录 `686,233,740` bytes。
- [x] 从全新 staging 严格 Range/Content-Range 分块重建并验签 clean 500M 权重；同一 8-window plan 下 negative-first/compact 均为 TP/FP/FN/TN=`0/0/4/4`、P/R=`0/0`，拒绝 veto/fusion 后删除 `625,339,992` bytes 模型文件。
- [x] 保留 source/download receipts、计划、prediction/cache/evaluation 与 resource logs；完成 215 项 VLM 回归、Ruff/format/diff、docs 与 llm-wiki post-hook，并累计清理 `1,849,575,509` bytes。readiness `not_ready`、blind inference paused、Module B 未授权。

## 2026-08-16 TASK-0262 lightweight VLM compatibility gate

- [x] 以固定 revision、官方许可标签、仓库总大小和本地 `mlx-vlm 0.6.7` 架构能力筛查 8 个小型候选；不升级依赖、不降低 1 GiB 下载上限或现有资源门禁。
- [x] 拒绝许可为 `other`、超过 1 GiB、仅单图或被已失败同容量视频模型支配的候选；未下载这些权重。
- [x] 下载并逐文件验签唯一通过元数据门禁的 `SmolVLM-500M-Instruct-4bit`；guarded loader 在任何预测或标签读取前因缺少原生 `video_processor` fail closed。
- [x] 保留候选审计、source manifest、README 与资源日志；删除 `291,652,074` bytes 模型载荷及本轮 nanoLLaVA 小缓存。没有 prediction/evaluation/fusion，readiness、blind inference 与 Module B 授权不变。

## 2026-08-16 TASK-0263 permissive basketball DEIM external screen

- [x] 只依据官方许可、固定 revision、体积和未查看 AGU 标签的官方报告冻结
  `ortizeg/basketball-deim-m-640`；记录官方测试比赛与训练比赛存在 clip-level 重叠，禁止把模型卡指标当作 game-held 证据。
- [x] RED→GREEN 新增最小 ONNX Runtime CPU adapter，精确复刻官方 640 方形预处理、原图尺寸输入和 raw10
  `ball`/`ball-in-basket` 合并；预先冻结 confidence=`0.10/0.25/0.50`、IoU=`0.25`。
- [x] 下载并验签 `77,722,583`-byte Apache-2.0 ONNX；在 106 条 LAL–BOS 确定人工复核行上完成 guarded screen，三档 TP/FP/FN 为
  `11/371/9`、`10/139/10`、`7/66/13`，均未达到 P/R≥`.85`。
- [x] 保留 source manifest、README、sealed screen 与 resource log；删除 `77,722,583` bytes 无帮助权重。
  不接入 runtime/training/fusion，不改变 readiness、blind inference 或 TASK-0258 Module B 授权。

## 2026-08-17 TASK-0264 DVIDS adult full-game metadata audit

- [x] 只读核验 DVIDS 2024 Armed Forces Basketball Championship 三个成人整场页面、时长、制作方、
  Public Domain 标记与分辨率/近似大小下载选项；媒体下载为 0 bytes。
- [x] 证明此前重新发现的 Commons Venezuelan SPB 比赛就是已保留的 VTV 源；同制作方其他比赛不构成
  新 production family。
- [x] 拒绝 Internet Archive 的非权威 NBA mirror license、青少年/未成年治理缺口与 Auburn 分卷/条款不一致来源。
- [x] 封存 metadata-only audit `076e69a2…8e2`：DVIDS/2D Audiovisual Squadron 是不同于
  HCTV/VTV 的强 future lead，但 Module A 尚无 mechanical pass、Module B 未授权，且 exact payload receipt、
  像素连续性、账户下载与 publicity/privacy 边界未闭合，所以不选源、不下载、不改变 readiness。

## 2026-08-17 TASK-0265 canonical .venv bytecode cache cleanup

- [x] 只读确认 VLM legacy evaluator 当前 135 项核心测试与 105 项相邻融合测试已覆盖永久 non-promotion、
  prediction-first、重复 truth、外部 receipts 和路径别名；不重复修改已完成代码。
- [x] 拒绝清理 root-owned `node-gyp` 和处于活动/打开状态的应用缓存，不提权、不干扰用户进程。
- [x] 仅删除 canonical `.venv` 内 656 个可重建 `__pycache__` 的 `.pyc/.pyo`；保留解释器、包、权重、数据和
  `.venv` 路径，旧 `venv/` 仍不存在。
- [x] 清理并完成 import/pip/Harness/Ruff 验证与最终 bytecode sweep 后，实时可用空间从
  `4,679,796` KiB 提升到 `4,784,084` KiB；cleanup artifact SHA `a0da89f0…390c`。这只解除当前磁盘前置阻点，
  不授权 TASK-0258 重跑，不构成 mechanical pass 或 readiness 变化。

## 2026-08-17 TASK-0266 DVIDS download-endpoint metadata boundary

- [x] 只读解析 TASK-0264 三场 DVIDS 官方下载弹窗，冻结 21 个衍生文件 ID、分辨率、近似显示大小和
  H.264/AAC MP4 编码信息；没有发起媒体 GET/Range，请求媒体响应体为 0 bytes。
- [x] 对每场两个低清文件执行 HEAD-only 检查；6/6 均返回 HTTP 403，且无 Content-Length 或 redirect，
  因此匿名下载访问与精确 payload bytes 仍未闭合。
- [x] 官方 Asset API 文档确认视频 `files[]` 可提供 exact size/src/bitrate，但 API key 是必需参数，缺失或
  无效 key 返回 403；本轮没有授权账户/API key，禁止把页面近似 MB/GB 当作 payload receipt。
- [x] 封存 endpoint audit `77f2b27e…8dd0`，状态 `download_receipt_gate_failed`；不选 candidate、不下载、
  不冻结 Module-B universe，不改变 readiness、blind inference 或 TASK-0258 授权状态。

## 2026-08-17 TASK-0267 closed independent-VLM evidence（Phase-0）

- [x] 重新盘点当前 v4 nested plan 的 32 个三元键与仓内 33 份 legacy prediction；0 份 prediction 绑定
  当前 plan SHA，最大键交集仅 21/32，禁止换绑历史预测。
- [x] 保持 generic plan/prediction/evaluation v1 为 `open_world_structural_only`、永久不可晋级；不原地升级。
- [x] 起草 6 模块 capability map：shared label-hidden plan -> 独立 VLM/base 两路预测 -> label-free fusion ->
  prediction-first delayed evaluation -> resource-guarded runtime adoption。显式区分 OOF 诊断与可重放 base policy。
- [ ] 等待用户审阅 `docs/specs/TASK-0267-independent-vlm-closed-evidence/capability-map.md`
  （file SHA `23580768847694fc0fc9b891c133c03eb58ef6c14b3a24ab9025be955d380c73`）。获批前不写模块规格、
  不实现、不运行模型、不揭示 truth、不融合或改变 runtime/readiness。

## 2026-08-17 TASK-0268 second-VTV adult full-game metadata audit

- [x] 核对三场 4 秒 annotation queue 的 v41 coverage ledger：Hazen/Randolph/VTV 在既有 15 秒源内排除
  协议下剩余窗口均为 0；禁止通过重叠抽样伪造新训练增量。
- [x] 只读核验 Commons 上另一场 VTV 职业整场 Spartans–Cocodrilos：2026-03-29、2:04:03、
  1920×1080 原片，Public domain / `PD Venezuela official`，且没有 license-review warning。
- [x] 证明它与已保留的 2026-04-26 Spartans–Trotamundos 是不同 actual game、同一 VTV production family；
  官方 480p derivative 为 854×480 / 1,060,216 bps，按时长估算约 0.919 GiB，暂低于 1.2 GiB cap。
- [x] 封存 metadata-only audit `eb229f90…cb54`；本机 TLS 代理未返回 derivative HEAD response，故 exact
  Content-Length/SHA、像素连续性、内容去重与 privacy/publicity 仍未闭合。媒体下载 0 bytes，Module B
  未授权，不选源、不改变 readiness/blind inference。

## 2026-08-17 TASK-0269 second-VTV exact remote endpoint receipt

- [x] 通过 Google/Cloudflare DNS-over-HTTPS 交叉得到 `upload.wikimedia.org` 公网 A record，并保持原 hostname
  SNI 执行 HEAD-only；不发送 GET/Range，不读取媒体 body。
- [x] 两次 HEAD 均得到 HTTP 200、exact Content-Length `986,415,444` bytes、对象 SHA-1
  `bf16fe7634cd6593bea05f9104ad8a43b4c69faf`、ETag `718a…690b` 与 `Accept-Ranges: bytes`。
- [x] 封存 additive endpoint receipt `e768aff4…48b0`；精确远端 payload 为 0.918671 GiB，低于 1.2 GiB cap。
  local SHA-256、像素连续性、内容重叠和 privacy/publicity 仍未闭合，Module B 未授权，禁止下载/选择。

## 2026-08-17 TASK-0270 bounded Commons long-form inventory

- [x] 通过 Commons API 受限枚举澳大利亚、加拿大、法国、立陶宛、斯洛文尼亚、西班牙、委内瑞拉 7 个
  分类，共 66 files；仅 40 条达到 ≥20 分钟。
- [x] 分类结果：3 条 CBC 纪录片不是全场；2 条 Slovenia 文件是同场分卷；Venezuela 35 条中 33 条仍为
  VTV family，另 2 条 Hidrocarburos channel 的原始制作方/公共部门权利依据未独立闭合。
- [x] 封存 inventory `957ddae1…68b8`；没有新增 production-distinct eligible source、没有下载媒体，保留
  exact-receipted second-VTV lead 但不选择、不改变 Module B/readiness/blind inference。

## 2026-08-17 TASK-0271 multilingual Commons TBF source audit

- [x] 冻结六个多语言全文查询、每词最多 100 条；243 个去重 video hit 中筛出 55 条 ≥20 分钟候选，
  补足 TASK-0270 国家分类枚举的 coverage 缺口。
- [x] 只读核验 TBF Merkezefendi–Samsunspor：成人职业整场、不同于 HCTV/VTV 的第三制作域、CC BY 3.0、
  YouTubeReviewBot reviewed，原片 6,266.781 秒 / 1,039,515,202 bytes。
- [x] 两个 DoH resolver 与两次 HEAD-only 响应冻结 480p remote receipt：818,675,123 bytes、对象 SHA-1
  `35d79e3a…5cba`、ETag `92e76128…e241`；媒体 body 仍为 0 bytes。
- [x] 封存 metadata audit `ce3d07e8…0fa0`；它只建立第三制作域 high-priority lead，不选择、不下载，
  Module B/readiness/blind inference 不变。
- [ ] 只有 Module A mechanical pass、Module B 单独授权、payload+3 GiB disk reserve、local SHA-256、连续像素、
  内容 overlap 与 privacy/publicity 全部通过后，才可冻结为 label-hidden canary source。
