# NBA-Identity primary-release audit

- [x] Audit the official paper's event/identity scope.
- [x] Audit the official repository's release links and license metadata.
- [x] Add the fail-closed source catalog entry and regression assertions.
- [x] Audit the complete MUVY basketball ZIP prefix with zero video-byte reads.
- [x] Audit NBA Rebounds availability; it is not public pending NBA permission.
- [x] Update AGU docs, task board and Wiki; retain no NBA-Identity/NBA-Rebounds files locally.
- [x] Verify `.venv` full suite, Ruff, diff check and Wiki lint.

## Nonlinear ball-candidate calibration (2026-08-02)

- [x] Bind the three RF-DETR source bundles and disjoint LAL–BOS target.
- [x] Add and test bounded nonlinear detector/track interaction features.
- [x] Run grouped ExtraTrees and Logistic screens without pixel decode.
- [x] Seal the rejected artifact and document that the hard-negative data gap is
      still the active blocker.

## E-BARD ObjectClassification crop-role source (2026-08-02)

- [x] Download and SHA-seal the official 82.5 MB CC BY 4.0 archive.
- [x] Validate 22,209 manifest-backed JPEG crops and four object-role labels.
- [x] Detect frame-level split leakage (59/56/55 game overlaps).
- [x] Retain one compressed copy only; no extracted duplicate or runtime path.
- [x] Treat the source as semantic crop pretraining only; independent
      cross-broadcast hard negatives remain required.

## Derived-data cleanup (2026-08-02)

- [x] Audit the largest derived MUVS frame caches for references.
- [x] Delete four unreferenced, rebuildable PNG cache directories (about 1.35 GiB).
- [x] Verify MUVS source/labels/compact artifacts, EBQwen weights, runtime and
      HOU–ORL blind assets remain present.

## E-BARD crop-role transfer screen (2026-08-02)

- [x] Add and test the lightweight offline role-feature baseline.
- [x] Hold out source games and select the basketball threshold without target
      labels.
- [x] Evaluate the sealed ATL–CHI candidate insets: P/R `0.326/0.255`,
      `accepted=false`.
- [x] Preserve the active blocker: independent cross-broadcast continuous
      ball/non-ball hard negatives, followed by causal ball–hand–rim evidence.

## Duplicate checkpoint cleanup (2026-08-02)

- [x] Confirm identical SHA-256 for the two local WASB checkpoint copies.
- [x] Delete the unreferenced `model_checkpoints/wasb_sbdt/` duplicate; retain
      `wasb_sbdt_official/` and all bound research evidence.

## Cross-broadcast full-frame detector increment (2026-08-02)

- [x] Materialize and hash-bind four reviewed source bundles: 417 rows, 166
      `valid_ball`, 251 hard negatives; keep the dataset offline-only.
- [x] Run the canonical `.venv` CPU YOLOv8n experiment and record the guarded
      resource envelope and incomplete 20-epoch result.
- [x] Re-screen LAL–BOS at confidence 0.10/0.25/0.50; best P/R is only
      `0.174/0.200`, so `accepted=false` at every fixed point.
- [x] Delete the unpromoted detector checkpoints after sealing their hashes;
      retain manifests, screen JSON and monitor logs for reproducibility.
- [x] Preserve the blocker: continuous cross-broadcast ball/non-ball hard
      negatives and causal ball–hand–rim evidence are still required; blind
      inference remains paused.

## Rejected-source media cleanup (2026-08-02)

- [x] Remove DeepSportRadar/APIDIS raw and derived payloads after failed gates.
- [x] Remove the rejected CC BY-NC Play-by-Play ZIP after its fixed transfer
      probe failed; keep audit metadata only.
- [x] Verify disk recovery and preserve E-BARD/BODD, EBQwen, accepted data,
      sealed evidence and HOU–ORL blind assets.

## Independent EBQwen evidence-gate screen (2026-08-03)

- [x] Implement and test the label-hidden, game-held evidence gate for EBQwen
      Q4KS plus Swin/MViT auxiliary OOF scores.
- [x] Verify exact plan/prediction/annotation/artifact hashes and freeze the
      32-window result: TP/FP/FN/TN `13/9/3/7`, pooled P/R/F1
      `0.591/0.813/0.684`.
- [x] Reject promotion (`promotion_eligible=false`, `runtime_consumable=false`)
      and leave detector, runtime and HOU–ORL blind assets unchanged.
- [x] Re-audit the 7.87 GB research-only basketball-events release; no video or
      annotation payload was downloaded because its schema/licence boundary is
      not suitable for AGU training.
- [x] Register Roboflow `ball-tracking-udopu/2` as the closest pending source
      candidate (CC BY 4.0, 700 images, `ball/goal`); defer download until an
      authorized export is available after the observed Cloudflare 403.

## Public ball-sample audit and cleanup (2026-08-03)

- [x] Audit the public `emirsahin/basketball-ball` archive in `/tmp`; confirm
      1,913 augmented JPGs, no YOLO label files and conflicting rights notices.
- [x] Add a fail-closed catalog/test entry and delete all temporary archive and
      extraction bytes after the audit.
- [x] Delete rejected Open Images generic pixels/raw CSV, retaining only the
      manifest, README and screen/resource evidence; keep the cross-broadcast
      hard-negative blocker explicit.

## Infactory auxiliary tiny-ball audit and cleanup (2026-08-04)

- [x] Materialize the metadata-only Infactory source subset with CC BY-NC 4.0
      provenance, 10 clip-disjoint splits, 1,450 frames and 924 boxes.
- [x] Complete a guarded 5-epoch `.venv` YOLOv8n auxiliary run and seal the
      result/resource/checkpoint hashes.
- [x] Reject transfer: held-source best P/R/F1 `0.662/0.377/0.480`; E-BARD
      basketball test `TP=0` at every confidence `0.05–0.50`.
- [x] Delete the unpromoted checkpoints and stale training caches; retain the
      source/YOLO data and compact offline evidence for explicitly noncommercial
      future experiments only.

## UniqueData basketball-tracking mirror audit (2026-08-04)

- [x] Audit the fixed Hugging Face revision and its CC BY-NC-ND boundary.
- [x] Confirm 70 decoded CSV images versus 106 unrelated `boxes/frame_*.PNG`
      images; reject the upstream positional pairing.
- [x] Keep only `analysis_outputs/public_research/unidata_basketball_tracking_audit_v1.json`
      and delete all temporary download/extraction bytes.
- [x] Leave `accepted=false`, `runtime_consumable=false`, and the hard-negative/
      ball-hand-rim blocker unchanged.

## CHI–UTA hard-negative review increment (2026-08-04)

- [x] Complete the 36-candidate manual review: 11 true balls and 25 false
      positives, with hash-bound plan, decisions and sealed review.
- [x] Keep the de-duplicated 441-row cross-broadcast dataset and its 354/87
      train/validation split (170 positives, 271 hard negatives).
- [x] Complete the guarded five-epoch `.venv` CPU training and external
      LAL–BOS confidence sweep; reject promotion at every fixed point.
- [x] Seal retention/checkpoint/resource hashes and delete unpromoted weights,
      stale label caches and temporary review crops.
- [x] Keep runtime, default detector, EBQwen and HOU–ORL blind assets untouched;
      the next useful input remains licensed continuous hard negatives plus
      ball–hand–rim causal evidence.

## Licensed basketball COCO ball/rim candidate (2026-08-05)

- [x] Retain the pinned CC BY 4.0 source, COCO audit, contact sheet and
      ball/rim-only YOLO materialization under `dataset/public_sources/`.
- [x] Register the source and adapter boundary in the public research catalog;
      keep runtime and external-acceptance flags fail-closed.
- [x] Complete guarded source-only and cross-broadcast combined CPU screens
      in `.venv` and preserve the independent LAL–BOS results.
- [x] Delete unpromoted checkpoints, temporary training directories and
      generated label caches after hashing; verify source/derived evidence
      remains readable.
- [x] Leave default detector, runtime, EBQwen and HOU–ORL blind assets intact;
      next useful data must add licensed game-diverse continuous ball/non-ball
      and ball–hand–rim causal evidence.

## Deduplicated 2025 basketball COCO supplement (2026-08-05)

- [x] Audit the fixed Hugging Face/Roboflow revision and treat its
      MIT/CC BY 4.0 rights conflict as attribution-required, offline-only.
- [x] Remove augmentation leakage by retaining one upstream variant per
      normalized source frame; keep the 139-frame, 99-ball/98-rim/19-empty
      materialization and contact-sheet review.
- [x] Run the combined 957-train/183-validation guarded `.venv` CPU route and
      independent LAL–BOS class-0 ball screen; record mAP50 `0.75056` and
      best P/R/F1 `0.25/0.25/0.25`.
- [x] Reject runtime/external promotion, delete temporary data, checkpoints and
      caches after hashing, and preserve the cross-broadcast hard-negative plus
      ball–hand–rim causal blocker.

## Independent EBQwen ball-candidate VLM screen (2026-08-05)

- [x] Complete the label-hidden six-band LAL–BOS VLM probe (108 candidates) with
      the retained MLX 4-bit model and sealed raw-frame/panel hashes.
- [x] Recover from the first guarded stop using the cache and finish all 108
      predictions under the retained CPU/memory limits.
- [x] Seal evaluation at precision `0.444060` and recall lower/upper
      `0.113961/0.102302`; reject promotion and preserve `runtime_consumable=false`.
- [x] Leave detector, runtime/VLM fusion, EBQwen weights and HOU–ORL blind assets
      unchanged; maintain the continuous hard-negative and causal-evidence blocker.

## ORL–CLE offline supplement and cross-game verifier (2026-08-05)

- [x] Download and integrity-audit the user-authorized NBA Games ORL–CLE
      480p media in an isolated offline-only directory; retain the media SHA
      and rights/decode audit.
- [x] Complete the guarded BODD 1 FPS full-game scan and 108-panel review;
      seal `valid_ball=49`, `uncertain=22`, `false_positive=37` and reject
      runtime/cross-game-training use.
- [x] Complete the label-hidden 12-panel EBQwen MLX 4-bit probe; reject it
      at weighted recall lower/upper `0.500617/0.459419`.
- [x] Complete LAL–BOS→ORL–CLE same-detector verification; best conservative
      P/R is `0.714836/0.750198`, below the `0.85/0.85` gate.
- [x] Keep the media, evidence, detector/VLM defaults, EBQwen weights and
      HOU–ORL blind assets separated; next input must be licensed continuous
      cross-broadcast hard negatives plus causal ball–hand–rim evidence.

## SportsGrounding metadata-only causal semantics candidate (2026-08-05)

- [x] Audit the pinned CC BY-NC 4.0 SportsGrounding revision and verify 520
      split-disjoint videos, 4,243 person tubes, 25 FPS and exact bbox spans.
- [x] Retain only the 23.8 MB train/val JSON metadata; do not download the
      approximately 14.7 GB video archives.
- [x] Seal 3,633 ball-mentioned captions as offline causal/tube semantics;
      reject ball-box training, runtime and external-acceptance use because no
      ball truth is present and the source is noncommercial.
- [x] Leave detector/VLM/runtime defaults, EBQwen and HOU–ORL blind assets
      unchanged; the active blocker remains continuous cross-broadcast
      ball/non-ball hard negatives plus independently verified causal evidence.

## Shot-causal support screen (2026-08-06)

- [x] Implement and test the offline game-held causal-support screen with
      exact source/event joins and no runtime answer path.
- [x] Evaluate the seven selected continuous reason features against the
      frozen 511-row base OOF artifact; pooled causal-support P/R/F1 is
      `0.572104/0.979757/0.722388` versus baseline
      `0.791667/0.846154/0.818004`.
- [x] Reject promotion, seal
      `analysis_outputs/public_research/shot_causal_support_screen_v1.json`,
      and leave detector/VLM/runtime defaults, EBQwen weights and HOU–ORL
      blind assets unchanged.
- [x] Run a 39-feature reason-evidence expansion as a bounded ablation;
      pooled precision/recall/F1 fell to `0.5648/0.9352/0.7043` and the
      Bdc held-game precision remained `0.6500`, so seal
      `analysis_outputs/public_research/shot_causal_support_screen_all39_v1.json`
      as rejected and keep it offline-only.
- [x] Keep the next data requirement explicit: licensed, game-/broadcast-
      diverse continuous ball/non-ball hard negatives and independent
      ball–hand–rim causal sequences.

## Broadcast false-positive diagnosis and guarded MViT probe (2026-08-06)

- [x] Test alternative offline classifiers and state gates against the frozen
      511-row outer-game-held screen; no promotion candidate passed the dual
      worst-game `P>=0.85` and `R>=0.85` gate.
- [x] Review the 23 high-confidence Bdc false positives; they are primarily
      live possession/drive/pass/setup windows without a completed release.
- [x] Seal `shot_broadcast_fp_audit_v1.json` with the observed failure mode,
      metrics and resource-guard provenance.
- [x] Stop the local MViT tail probe safely at the memory guard (`exit 75`);
      no checkpoint or runtime change was produced.
- [ ] Do not promote a threshold or classifier until a raw temporal model or
      new licensed causal/hard-negative source clears the cross-game gate.

## BARD event metadata hard-negative index (2026-08-06)

- [x] Pin BARD revision `add4109bf8b2034a32f3b6a83fa0d8c0ae638473`; retain only
      the 8.0 MB event metadata/paths plus upstream license and README.
- [x] Confirm 14,676 rows/60 games and 2,631 non-shot rows; write the compact
      `non_shot_index.jsonl` for future rights-cleared acquisition.
- [x] Seal `bard_event_metadata_audit_v1.json` with the manifest SHA
      `315f7fa1ad4c59c4cad806b2cd26170d23754eb9ec7be19178274a1257f8a75b` and
      keep all flags runtime-disabled.
- [ ] Keep NBA source-video URLs as metadata only; no raw clip download or
      training is allowed without a separate rights grant and game-disjoint
      visual review.

## BARD non-shot validation clips (2026-08-06)

- [x] Download the bounded 18-clip benchmark subset (2024/2025), covering 15
      source games and `138,530,775` bytes; labels are Turnover/Steal/Foul only.
- [x] Verify Git blob/SHA-256/decode metadata and perform label-hidden visual
      review of three quantile frames per clip; no replay/free-throw-only bias
      was observed.
- [x] Seal `bard_nonshot_validation_audit_v1.json` and retain the source
      manifest SHA `aefebdea2605b90652a85fe0668c348f8958aae05029d0f040a4271fad059e02`.
- [x] Extract MobileNetV3 phase-scene features and seal
      `bard_nonshot_scene_probe_v1.json`; the probe stays offline-only.
- [x] Reject the 18-clip augmentation because held-game worst precision/recall
      fell from `0.5862/0.9091` to `0.5455/0.8636`; no training, runtime, VLM
      default or blind promotion was made.
- [x] Extract all 18 MViT frozen embeddings under the resource guard and screen
      scene+MViT augmentation; worst precision fell to `0.5234`, so keep the
      MViT artifact and guard log offline-only with no promotion.
- [ ] Acquire independently verified ball-hand-rim causal hard negatives from
      more rights-cleared broadcasts, then require worst-game `P>=0.85` and
      `R>=0.85` before any promotion.

## VRU_Basketball continuous-scene subset (2026-08-06)

- [x] Pin revision `d256fa0b5fab474663f595abe4f386ac4d5adcc6`; extract and
      hash only eight clips (four Dongguan/four Guangzhou, 176,838,471 bytes).
- [x] Confirm 25-FPS 1920×1080 ten-second continuous scenes and record the
      source manifest plus declared CC BY 4.0/offline-only boundary.
- [x] Screen 400 sampled frames with BODD and manually review 48 selected
      candidates: 44 `valid_ball`, four `false_positive`.
- [x] Keep the subset as offline continuous-scene and hard-negative evidence
      only; no ball boxes, event labels or independently verified ball–hand–rim
      sequences were present, so no detector/VLM/runtime/training promotion.
- [x] Seal the audit artifact, remove temporary files, and keep the current
      cross-broadcast and causal-evidence blocker explicit.

## Play by Play basketball annotation slice (2026-08-06)

- [x] Download and verify the CC BY-NC 4.0 Zenodo archive (MD5
      `caac618508f17e3b08850be1bf24a090`); keep the basketball-only annotation
      tree with 381 clips and 56,578 frames.
- [x] Add and test the audit module/script; seal
      `analysis_outputs/public_research/play_by_play_basketball_audit_v1.json`
      with `runtime_consumable=false`.
- [x] Remove the redundant verified source ZIP and keep the local rights note,
      labels and audit only.
- [x] Leave detector/VLM/runtime defaults, EBQwen weights and HOU–ORL blind
      assets unchanged; no raw video or ball–hand–rim truth was added.

## VRU raw-frame ball–hand–rim causal review (2026-08-06)

- [x] Build a label-hidden plan from the pinned VRU manifest and bind every
      reviewed frame to decoded-frame SHA-256.
- [x] Manually seal 3 `shot` chains (`guangzhou_1/11/31`) and 3 `not_a_shot`
      windows (`guangzhou_21`, `dongguan_0/16`); leave outcome unknown for all
      positive windows.
- [x] Verify plan SHA `e34adda6f77698aef3b8a39f3d89483a25fa1c2e41dc911b844187b2d7088578`
      and review SHA `083f351d913c3fd8c600cc3278249b554d820ecd659245fd411a28ae29e2bde1`.
- [x] Keep all artifacts offline-only and runtime-disabled; no detector, VLM,
      EBQwen, runtime schema or blind-game asset changed.
- [ ] Obtain broader licensed cross-broadcast causal truth and exhaustive
      hard-negative coverage before resuming blind inference.

## Rejected ORL–CLE media cleanup and candidate-source audit (2026-08-06)

- [x] Re-hash the rejected ORL–CLE MP4, delete only the verified 1,097,207,098-byte
      payload, and retain its SHA/decode/measurement artifacts plus the tiny state file.
- [x] Preserve HOU–ORL blind media, EBQwen native/MLX weights and all accepted
      offline source subsets; no runtime or blind path was opened.
- [x] Reject TeamTrack (player MOT/no ball truth), the static/non-commercial
      Basketball Tracking Dataset and Dryad's non-basketball sport-ball release
      without downloading their unsuitable payloads.
- [x] Re-audit MUVY v1 and the unlicensed BasketEvent HF mirror; retain no new
      payload because MUVY's basketball slice is already screened and BasketEvent
      has no license/card or hand/rim truth.
- [ ] Continue only when a licensed, continuous cross-game/broadcast source supplies
      independent ball/non-ball and ball–hand–rim supervision.

## SPIROUDOME byte-range candidate audit (2026-08-06)

- [x] Range-audit the Kaggle mirror without downloading its 19.6 GB archive;
      confirm 192 members and no ball/event annotation files.
- [x] Review one 37.16-second, 929-frame camera-1 sample with BODD: all ten
      basketball candidates were false positives, so it is hard-negative-only.
- [x] Seal the audit and source-catalog entry under the stricter official
      non-commercial research boundary; keep `runtime_consumable=false`.
- [x] Remove all temporary media and range-audit files (396,074,466 bytes).
- [ ] Obtain independently labeled, licensed cross-broadcast ball/non-ball and
      ball–hand–rim sequences before any new promotion or blind inference.

## Targeted disk cleanup (2026-08-06)

- [x] Verify the RTMPose source archive SHA-256 against
      `analysis_outputs/open_pretraining/rtmpose_m_halpe26_source_manifest_v1.json`.
- [x] Delete only the redundant 50 MiB archive after confirming the extracted
      53 MiB `end2end.onnx` remains; retain the source manifest and all model
      provenance. Keep EBQwen native/MLX weights, runtime checkpoint and blind
      media unchanged.

## Qlean gated continuous-video candidate (2026-08-06)

- [x] Record the public card's six-game/28-MP4/26.40-GB description and gated
      academic-only access terms without bypassing the access review.
- [x] Add `qlean-video-basketball-match` to the source catalog as
      metadata-only, `runtime_consumable=false`, and training-disabled.
- [x] Seal `analysis_outputs/public_research/qlean_video_basketball_match_audit_v1.json`
      (SHA-256 `5fd4d832b7935d96f5f1284f001f153dbdcc0b9d36ca6965b4f66c4ce19793de`).
- [ ] Only pursue a gated download after an authorized academic identity is
      available and the Japanese license, file hashes, annotations and
      cross-game causal coverage have been audited.
- [ ] Do not resume blind inference: the source card exposes no ball–hand–rim
      or event truth and does not clear the active 85% gate.

## SportsAction / MultiSports gated action-tube candidate (2026-08-06)

- [x] Audit the official [MCG-NJU/SportsAction](https://huggingface.co/datasets/MCG-NJU/SportsAction)
      card: basketball action labels include shots, offensive/defensive rebounds,
      steals, screens and jump balls; the card reports 28,514 train and 10,116
      validation tubes.
- [x] Confirm the annotation boundary is person action tubes only, with no ball,
      hand, rim or made/miss truth; source recordings are described as YouTube
      competition videos.
- [x] Register `sports-action-multisports` as gated CC BY-NC-4.0 metadata-only;
      do not download the approximately 65.1 GB payload.
- [x] Seal `analysis_outputs/public_research/sports_action_multisports_audit_v1.json`
      (SHA-256 `23c19cec33442ea0c0845fb833323106322fa956bed1e16f201c55dee7371642`).
- [ ] Only pursue after an authorized noncommercial access decision and a
      source-media/ball-hand-rim audit; keep runtime and training disabled.

## MEM--OKC development PBP coverage probe (2026-08-06)

- [x] Complete the resource-guarded scoreboard/PBP alignment probe on the exposed
      development video; no blind asset was opened or modified.
- [x] Record 63/200 joins at +/-90 frames and reject the wider +/-300-frame join as
      causally too loose.
- [x] Seal `analysis_outputs/public_research/mem_okc_pbp_candidate_join_probe_v1.json`
      with `runtime_consumable=false` and `accepted_for_training=false`.
- [ ] Source denser causal proposals or an independent sub-second ball--hand--rim
      anchor before extracting training embeddings.
- [ ] Do not resume blind inference or alter runtime/VLM defaults from this probe.

## MEM--OKC candidate sweep follow-up (2026-08-06)

- [x] Sweep 15 clustering configurations from the existing BODD perception cache;
      retain only the sealed counts/config hashes, not temporary candidate bundles.
- [x] Re-run 10-second scoreboard OCR under the memory guard (229 readable samples,
      peak RSS 2,098,200,576 bytes, zero swaps) and rebuild the offline PBP alignment.
- [x] Seal `analysis_outputs/public_research/mem_okc_candidate_sweep_join_probe_v1.json`
      (SHA-256 `7644b1228e697e8826abe20cbe006ba51c4588eb0fb737e66902dc9509fdb8c5`).
- [x] Reject `d2_g3` for training: strict +/-90-frame coverage is only 88/200 under
      the permissive end-frame screen and 59--64/200 when candidate types are compatible.
- [ ] Obtain independent sub-second ball--hand--rim release/outcome anchors across
      games and broadcasts before extracting embeddings or changing runtime/VLM defaults.
- [ ] Keep blind inference paused until the cross-game 85% gate is met.

## VRU additional continuous clips and causal review (2026-08-06)

- [x] Extract eight additional CC BY 4.0 ZIP members with bounded HTTP Range
      requests; retain 179,162,740 bytes and delete no source archive because none
      was materialized.
- [x] Verify all eight clips by ZIP CRC, SHA-256 and `.venv` OpenCV metadata; keep
      the additional-only manifest hash-bound separately from the original
      eight-clip manifest and BODD audit.
- [x] Seal the label-hidden causal plan/review: one `shot` chain, one uncertain
      pass and six `not_a_shot` windows; leave the positive outcome unknown.
- [x] Seal `analysis_outputs/public_research/vru_basketball_extra_audit_v1.json`
      (SHA-256 `9e06f9817ef8a556e65114d9380b985ae1d621570c72571dc6413a307d2d3081`).
- [x] Keep the added media and labels offline-only; do not alter detector/VLM
      defaults, EBQwen weights, runtime answers or HOU–ORL blind assets.
- [x] Run a guarded BODD visibility probe on one shot, one uncertain pass and one
      not-a-shot window; seal the result with `accepted_for_training=false`.
- [x] Confirm the detector-only signal is non-causal (candidate detections occur
      in both pass and not-a-shot windows); do not promote a threshold or fusion.
- [ ] Obtain independently verifiable, cross-game/broadcast ball–hand–rim
      outcomes and exhaustive non-ball hard negatives before resuming blind inference.

## Bdc non-overlap hard-negative increment (2026-08-06)

- [x] Select 32 deterministic Bdc windows outside the existing candidate neighborhoods
      and review them from raw frames with labels hidden from selection.
- [x] Seal 8 live field-goal attempts and 24 replay/free-throw/dead-ball/no-release
      negatives with a raw-only bundle and v2 training manifest.
- [x] Extract frozen MobileNet/MViT features and run the four-game held-out screen;
      merged pooled P/R/F1 is `0.7540/0.9137/0.8262`, weakest precision `0.6827`,
      so the 85% gate is not met.
- [x] Delete temporary contact sheets after sealing the source-frame specification;
      keep source, bundle, labels, embeddings and the negative screen.
- [ ] Obtain independent licensed cross-broadcast hard negatives with sub-second
      ball–hand–rim truth before another promotion attempt or blind inference.

## Bdc broadcast-state fusion follow-up (2026-08-06)

- [x] Correct the offline broadcast-clock anchor fallback to use generic
      `candidate_event_frame` evidence and add a regression test.
- [x] OCR the 32 extra windows under the raw-only boundary: 1 `replay`, 31
      `unknown`; seal the extra clock artifact and merged 543-row state artifact.
- [x] Screen scene+MViT+`base+broadcast_raw` with outer game-held isolation;
      pooled P/R/F1 is `0.7778/0.9059/0.8370`, weakest precision `0.7037`, so
      the gate remains rejected.
- [x] Keep the clock/state/screen artifacts offline-only; do not promote a
      threshold or modify runtime/VLM/EBQwen/blind assets.
- [ ] Obtain independent licensed cross-broadcast causal truth and exhaustive
      non-ball hard negatives before another promotion attempt.

## Lakers–Magic cross-broadcast development increment (2026-08-06)

- [x] Download and hash-check the bounded 854×480 NBA Games Lakers–Magic stream;
      keep the source manifest and external-media rights boundary offline-only.
- [x] Select 32 deterministic non-overlapping windows and review raw contact sheets
      with labels hidden from selection; seal 7 positives and 25 hard negatives.
- [x] Extract frozen MobileNet/MViT/Swin3D features and run the five-game held-out
      screen; best pooled P/R/F1 is `0.6222/0.9724/0.7588`, while the new held game
      is `0.4000/0.8571/0.5455`, so the 0.85 gate is rejected.
- [x] Keep all source, labels, embeddings and screen artifacts training-only;
      do not promote a threshold or alter runtime/VLM/EBQwen/blind assets.
- [ ] Obtain independently licensed, cross-game/broadcast ball–hand–rim outcomes
      plus exhaustive non-ball hard negatives before another promotion attempt or
      resuming blind inference.

## Post-audit media cleanup (2026-08-06)

- [x] Re-hash and remove the exposed-PBP `780Byt-iJyc.mp4` development probe and
      unused `sOfCveR_usM.mp4` enrollment copy; keep their download-state SHA/size
      provenance and sealed audit JSON.
- [x] Preserve `aU5JC_9RliM.mp4` as the still-needed MEM–OKC blind source, along
      with HOU–ORL media, EBQwen native/MLX weights and accepted offline data.
- [x] Remove repository `__pycache__`, `.pytest_cache` and `.ruff_cache`; about
      1.60 GiB of media was reclaimed without touching runtime or blind assets.
- [ ] Re-check disk usage after any future bounded download and delete only
      hash-verified, unreferenced temporary payloads.

## Basketball Events pinned annotation audit (2026-08-06)

- [x] Pin revision `26d3775286f542b41daf4a94a53190440d426111` and download only the
      upstream README, four game-level annotation JSON files and four representative
      7-second clips; seal `dataset/public_sources/basketball_events_v1/manifest.json`.
- [x] Verify all downloaded hashes and decode metadata; confirm 543 clips / 897 events
      across four games and preserve the source's research-only rights boundary.
- [x] Register the source in the public-research catalog as
      `runtime_consumable=false`, `training_media_eligible=false` and audit-only.
- [x] Record the missing frame-level ball/hand/rim, possession, onset/outcome and
      offensive/defensive rebound fields; do not count this source as independent causal truth.
- [ ] If a rights-cleared, frame-level annotation release appears, re-audit it game-held;
      otherwise keep this bounded source out of runtime, blind inference and acceptance metrics.

## HOU–SAC tiled-ball cross-broadcast screen (2026-08-06)

- [x] Correct the HOU–SAC manifest to `0022400059` and keep the low-resolution external video
      offline-only.
- [x] Guard full-frame versus overlapping-tile BODD on 600 samples; full-frame ball count is
      `0`, tiled deduplicated ball count is `198` across `96` short tracks.
- [x] Seal the raw-reviewed/PBP-after-review screen with TP/FP/FN `7/6/6` and P/R/F1
      `0.5385/0.5385/0.5385`; do not treat it as training truth.
- [x] Add and test the screen-only tile geometry, box projection and same-frame overlap
      deduplication helpers in `app/analysis/perception/tiled_ball.py`.
- [x] Run the guarded YOLO11 pose follow-up and seal its unchanged `7/6/6` result in
      `analysis_outputs/public_research/hou_sac_2024_raw_review_v1/pose_tiled_screen_v1.json`.
- [ ] Build a trained tile-aware detector path and rerun on independent games/broadcasts with
      exhaustive hard negatives before any runtime or blind-inference change.
- [ ] Keep blind inference paused; the blocker remains cross-game ball recall/precision and
      ball–hand–rim causal evidence, not candidate-window count.

## Gated shot-outcome source audit (2026-08-06)

- [x] Audit `leharris3/basketball-shot-test-dataset` without downloading its gated 423 MB media;
      its empty card and unverified broadcast provenance fail the import boundary.
- [x] Keep `yerx/bb` catalog-only; its 1.72 GB free-throw clips have no declared license or
      media provenance.
- [x] Register both decisions in the source catalog (50 entries, SHA
      `86b2cac9d1e47abfe85da05995d152cedcb47ff6ff1c253e89453c15be3928ee`).
- [ ] Find a licensed, game-diverse continuous ball/hand/rim source with exhaustive hard
      negatives before any detector or blind-inference promotion.

## E-BARD/MUVY tile-aware ball training screen (2026-08-06)

- [x] Generate and hash-bind 3,458 overlapping tiles from the existing CC BY 4.0 source;
      keep the materialization offline-only.
- [x] Guard a canonical `.venv` CPU 3-epoch BODD fine-tune; final val P/R/mAP50/mAP50-95 is
      `0.7280/0.5648/0.589/0.267`, with zero guard stops.
- [x] Seal HOU–SAC `7/3/6` (P/R/F1 `0.700/0.5385/0.6087`) and independent Lakers–Magic
      `7/19/0/6` (P/R/F1 `0.2692/1.000/0.4242`) screens.
- [x] Keep `accepted=false`; remove the rejected tile directory, checkpoint and external run
      after retaining `screen_v1.json` and compact evidence.
- [ ] Continue only after acquiring licensed cross-game/cross-broadcast ball–hand–rim causal
      supervision and hard negatives; blind inference remains paused.

## Official DeepBall-Large basketball baseline screen (2026-08-06)

- [x] Pin the official WASB-SBDT revision, model-zoo URL, MIT code boundary and
      DeepBall-Large basketball checkpoint hash; record the no-media-rights boundary.
- [x] Add the hash-checked offline CPU adapter and fixed-window screen with TDD;
      keep it outside runtime detector/VLM defaults.
- [x] Select threshold `0.50` from HOU–SAC development windows (`6/1/1/2`,
      P/R/F1 `0.8571/0.8571/0.8571`) under the resource guard.
- [x] Lock that threshold on independent Lakers–Magic windows (`6/14/1/11`,
      P/R/F1 `0.3000/0.8571/0.4444`) and seal the screen artifact.
- [x] Remove temporary smoke and duplicate download files; retain only the small
      official baseline, evidence, scripts and tests.
- [ ] Acquire licensed, broadcast-diverse causal ball–hand–rim outcomes and exhaustive
      non-ball hard negatives before another promotion attempt or blind-inference recovery.

## Lakers–Magic independent VLM and DeepBall fusion screen (2026-08-06)

- [x] Build a label-hidden 32-window raw-frame plan from the fixed candidate bundle; the model
      saw no `event_present`, review notes, PBP or Codex runtime answer.
- [x] Complete the guarded `qwen3-vl:2b` strict-release run at 4 frames/512 px; all 32 outputs
      are `live_field_goal` and remain offline-only.
- [x] Delay truth opening and seal VLM-only metrics `7/25/0/0` (P/R/F1 `0.2188/1.0000/0.3590`).
- [x] Screen fixed VLM-only, DeepBall-only, AND and OR rules; AND remains `6/14/1/11`
      (P/R/F1 `0.3000/0.8571/0.4444`), below the gate.
- [x] Keep runtime, EBQwen native/MLX weights and blind assets unchanged; retain only compact
      plans/predictions/evaluation/fusion/resource evidence.
- [ ] Return to licensed cross-broadcast causal supervision and hard negatives before promotion.

## BasketEvent player-grounded trajectory audit (2026-08-06)

- [x] Pin revision `85aaa3ce62bc096e3995c39ecfa6773fcc9fe5e1` and add the fail-closed catalog
      entry for the player/ball trajectory release.
- [x] Retain the 557-file valid slice and 120-file 37-game test slice after API-size/SHA checks;
      seal both manifests and the trajectory/event audit artifacts.
- [x] Confirm that the test slice contains event labels and ball/player boxes, while the valid
      slice has 14 missing-ball clips; no raw media, rim/hand boxes or explicit license is present.
- [x] Keep the source offline auxiliary-only (`runtime_consumable=false`,
      `training_media_eligible=false`); transiently hash-check and probe `playnet.pt` only, then
      delete the unlicensed 1,813,273,995-byte checkpoint and retain compact provenance.
- [ ] Find a rights-cleared raw-video release or independent sub-second ball–hand–rim labels
      before using this source to change detector/VLM defaults or resume blind inference.

## LAL–ORL Game 3 cross-broadcast screen (2026-08-06)

- [x] Download and SHA-seal the bounded `Agffi33pz8w` Game 3 stream and record its
      external-platform/no-redistribution boundary.
- [x] Correct the seeded uniform-window sampler so slack is spread across the full upload;
      add tests for deterministic coverage and spacing.
- [x] Seal 32 raw-reviewed windows (1 visible field-goal attempt, 31 hard negatives) and keep
      the candidate spec, raw-only bundle, labels and embeddings offline-only.
- [x] Screen scene+MViT (`1/9/0/22`, P/R/F1 `0.1000/1.0000/0.1818`) and DeepBall (`1/14/0/17`,
      P/R/F1 `0.0667/1.0000/0.1250`); both are rejected.
- [x] Check the augmentation against Lakers–Magic; its held P/R/F1 remains `0.4000/0.8571/0.5455`.
- [x] Complete the all-window label-hidden Qwen3-VL Game 3 probe: all 32 outputs are positive;
      delayed truth is `1/31/0/0` (P/R/F1 `0.03125/1.0000/0.0606`), and the fixed AND with
      DeepBall is `1/14/0/17` (P/R/F1 `0.0667/1.0000/0.1250`); no promotion.
- [ ] Keep searching for rights-cleared, frame-level ball–hand–rim outcomes plus exhaustive
      non-ball hard negatives; do not resume blind inference until the 0.85 gate clears.

## LAC–DAL 2024 independent cross-broadcast screen (2026-08-06)

- [x] Download and SHA/decode-seal the bounded `waBDFY4ihS0` stream under the external-media
      no-redistribution boundary; preserve NBA metadata/PBP only for delayed consistency checks.
- [x] Seal 32 label-hidden raw windows (9 visible releases, 23 hard negatives) with the generic
      candidate/bundle/labels/training-manifest builder and its hash/boolean-label tests.
- [x] Screen locked DeepBall (`7/15/2/8`, P/R/F1 `0.3182/0.7778/0.4516`) and independent
      Qwen3-VL (`9/23/0/0`, P/R/F1 `0.2813/1.0000/0.4390`); fixed AND/OR rules are rejected.
- [x] Keep all media, labels, plans, predictions, resource logs and fusion output offline-only;
      clean only the sealed contact/detail sheets, not source media or compact provenance.
- [ ] Acquire rights-cleared continuous ball–hand–rim outcomes and exhaustive non-ball hard
      negatives before altering runtime/defaults or resuming blind inference.

## Basketball Events shot-positive VLM audit increment (2026-08-06)

- [x] Add the bounded 16-clip four-game make/miss subset (226,664,456 bytes) with SHA/size/decode
      checks; keep the upstream research-only terms and no-runtime/no-training boundary.
- [x] Run label-hidden strict-release Qwen3-VL at 4 frames/512px under `.venv` guard; positive
      event-centered recall is 16/16, but this is not a precision benchmark.
- [x] Combine with LAC–DAL hard negatives and seal the 48-row screen at
      `25/23/0/0`, P/R/F1 `0.5208/1.0000/0.6849`; reject promotion and leave runtime/weights/
      blind inference unchanged.
- [x] Keep compact plan/labels/predictions/evaluation/resource artifacts and update catalog/docs;
      remove no source clips because all 16 are part of the bounded audit evidence.
- [ ] Continue the search for rights-cleared continuous cross-broadcast causal labels and
      exhaustive hard negatives before resuming blind inference.

## Post-screen scratch cleanup (2026-08-06)

- [x] Remove only hash-sealed/reproducible audit scratch under `/tmp` (VRU/MUVY extraction,
      old candidate sheets, BasketEvent API cache and stale probe media).
- [x] Remove repository bytecode/test/lint caches; preserve `.venv`, Ollama/EBQwen weights,
      source clips, compact evidence, blind media and all new Basketball Events artifacts.

## Closed CHI–UTA media handoff cleanup (2026-08-06)

- [x] Re-hash the three CHI–UTA/1997–98 external media files against the sealed pair handoff
      before deletion (`3,350,552,015` bytes total).
- [x] Delete only those exact files after the old identity/development handoff was closed;
      retain `download-state.json` and `chi_uta_archive_pair_handoff_v1.json` as provenance.
- [x] Confirm HOU–ORL and MEM–OKC blind media, current cross-broadcast screens, EBQwen weights,
      `.venv` and runtime inputs remain present; no blind answer was opened.
- [ ] Continue only with a rights-cleared continuous ball–hand–rim source and exhaustive
      non-ball hard negatives; media cleanup does not clear the 0.85 gate.

## LAC–DAL EBQwen native-video resource probe (2026-08-06)

- [x] Run a label-hidden four-window prefix with local EBQwen MLX 4-bit, pinned revision
      `c6a93cbb325f9d20236f85bcaba7827a2808443e`, native temporal encoding and 2 FPS.
- [x] Seal TP/FP/FN/TN `1/3/0/0` (P/R=`0.25/1.00`) after delayed labels; reject the probe and
      leave runtime, weights, fusion, `.venv` and blind media unchanged.
- [x] Remove the first-run artifacts whose model revision provenance was malformed; retain only
      the corrected hash-bound prediction/evaluation/resource evidence.
- [ ] Continue only with rights-cleared cross-broadcast ball–hand–rim outcomes plus exhaustive
      non-shot hard negatives.

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

## BQwen2.5-VL-3B BARD probe and cleanup (2026-08-06)

- [x] Hash-verify the pinned two-shard BQwen download and record the provenance manifest.
- [x] Convert to MLX 4-bit under `.venv` resource guard; no AGU runtime or default model change.
- [x] Run the label-hidden four-window native-video probe; delayed result is `1/3/0/0`,
      P/R=`0.25/1.00`, equal to EBQwen and below the independent promotion gate.
- [x] Delete BF16/MLX weight payloads and keep compact provenance plus offline evidence.
- [ ] Obtain licensed cross-broadcast causal ball–hand–rim labels and exhaustive hard negatives
      before another model swap, fusion attempt or blind-inference recovery.

## Basketball-51 bounded outcome sample (2026-08-06)

- [x] Extract and hash only 32 Apache-2.0-uploader-labeled clips (8 classes × 4 source groups);
      retain the underlying broadcast-rights fail-closed boundary.
- [x] Run MViT source-group embedding screen: free-throw separation passes a narrow source gate,
      but made/missed and eight-class outcome screens fail.
- [x] Run guarded tail fine-tune; held source group is P/R/F1=`0/0/0`, no checkpoint emitted.
- [ ] Do not promote this sample; continue toward licensed cross-broadcast causal outcomes and
      exhaustive non-ball hard negatives.

## F-16-NBA shot-test online candidate gate (2026-08-06)

- [x] Audit the pinned Apache-2.0 card and shot-test metadata: 1,970 `Yes/No` clips across
      16 source games; no labels were sent to a model.
- [x] Range-audit only 37,830,656 bytes of the first tar shard and verify eight H.264 1280×720,
      60-fps clips plus JSON/shard/sample hashes.
- [x] Reject and clean the candidate because clip-level outcome labels do not supply causal
      ball–hand–rim timing or exhaustive hard negatives, and underlying NBA redistribution rights
      are not separately granted; retain manifest/audit only.
- [ ] Do not promote or download the remaining archive; continue toward licensed continuous
      cross-broadcast causal outcomes.

## Basketball-51 expanded source-group screen (2026-08-06)

- [x] Extract and hash 128 clips (16 per class, 46 source groups, `80,328,593` bytes) with
      ZIP64 Range and ffprobe verification; keep labels withheld during screening.
- [x] Run the guarded canonical `.venv` MViT source-group screen; free-throw separation reaches
      worst-fold P/R/F1=`0.9565/0.9565/0.9565`, but made/missed remains `0.5455/0.375/0.48` and
      eight-class macro-F1=`0.1667`.
- [x] Retain only the bounded sample, embeddings, audit and rejected metadata for the auxiliary
      free-throw diagnostic; no second fine-tune or checkpoint was produced.
- [ ] Do not promote this sample; continue toward licensed cross-broadcast causal outcomes and
      exhaustive non-ball hard negatives.

## Basketball_Detection static candidate gate (2026-08-07)

- [x] Audit the pinned GitHub tree (`c6d4583974f412f8b40215298234e6c13a0da30a`) and
      `dataset.yaml`; it contains 9,739 image/label pairs across `ball/made/person/rim/shoot`.
- [x] Confirm the repository has no declared license or `LICENSE` file, and no frame-level
      ball–hand–rim ordering, shot-outcome sequence or exhaustive non-shot hard negatives.
- [x] Download no image bytes; retain only
      `dataset/public_sources/basketball_detection_github/source_manifest.json` and
      `analysis_outputs/public_research/basketball_detection_github_audit_v1.json` with the
      fail-closed catalog entry.
- [ ] Continue toward an explicitly licensed, cross-broadcast temporal causal release before
      any detector/VLM promotion or blind-inference recovery.

## Qwen3-VL-4B resource-fit screen and cleanup (2026-08-07)

- [x] Pull and hash-bind the official Ollama `qwen3-vl:4b` asset for the frozen LAC–DAL
      label-hidden plan; keep it offline-only.
- [x] Run the strict-release 4-frame/512-pixel probe under the canonical `.venv` guard; it
      stopped before the first prediction at three consecutive samples below 2.5 GiB available
      memory (exit code 75), so no accuracy result was produced.
- [x] Retain `independent_shot_vlm_qwen3vl4b_4f512_screen_v1.json` and the raw guard trace, then
      delete the unhelpful 3.3 GB local payload. Qwen3-VL-2B, EBQwen native/MLX, runtime and blind
      assets remain present.
- [ ] Continue only after a rights-cleared cross-broadcast causal source with exhaustive hard
      negatives is available; do not reopen blind inference from this resource-only attempt.

## SportsShot online candidate gate (2026-08-07)

- [x] Pin and inspect MCG-NJU/SportsShot revision `2fca05a9c49366d2c52b4d90e322a7a83e634262`;
      card declares CC BY-NC 4.0, 1,200 videos and about 183 GB of shot-segmentation media.
- [x] Confirm anonymous file access returns `401 GatedRepo`; download no gated archive, labels or
      video bytes.
- [x] Keep only `analysis_outputs/public_research/sportsshot_online_gate_v1.json`, add its
      fail-closed catalog entry and regenerate the 55-source catalog.
- [ ] Do not promote shot-boundary labels; continue toward authorized causal ball–hand–rim and
      exhaustive non-ball hard-negative supervision.

## strict-release-v3 evidence consistency screen (2026-08-07)

- [x] Add and test the offline `strict_release_v3` chronology/contradiction prompt; leave AGU
      runtime prompts and answer boundaries unchanged.
- [x] Complete all 32 label-hidden LAC–DAL windows with the canonical `.venv` resource guard;
      guard exit 0, minimum available memory about 1.96 GiB and no stop.
- [x] Delay truth opening, seal raw `9/23/0/0` and normalized `9/20/0/3` evidence; both fail
      the independent promotion gate, so no model or parser promotion is allowed.
- [x] Retain compact prediction/evaluation/audit/resource artifacts and remove temporary cache
      after verification.
- [ ] Obtain rights-cleared cross-broadcast causal ball–hand–rim outcomes and exhaustive
      non-ball hard negatives before another VLM/prompt screen or blind recovery.

## NUS Basketball Detection online candidate gate (2026-08-07)

- [x] Audit pinned revision `3ee1a0decfde199117b3a99d79bcf136c902ebc5`; inventory 719 clips,
      11 shot/background classes and `2,967,530,545` total bytes.
- [x] Download and verify a deterministic 12-clip sample (`37,378,249` bytes) with Hub LFS SHA,
      size and ffprobe checks; review make/miss/background visual coverage.
- [x] Reject and clean the sample because license, provenance, frame-level release/outcome labels
      and exhaustive hard negatives are absent; move media to
      `/Users/ppt/.Trash/agu-nus-basketball-20260807` and keep compact audit only.
- [x] Register `nus-basketball-detection-cs5260` as `runtime_consumable=false` and
      `training_media_eligible=false`; regenerate the 56-source catalog with internal SHA
      `f947fb5c702aa36de6ec631ddd5fca32efb63f6435e893860f85885e0cbc94ca`.
- [ ] Continue toward an explicitly licensed, cross-broadcast temporal causal release before any
      detector/VLM promotion or blind-inference recovery.

## PL-NBA possession-level temporal annotation gate (2026-08-07)

- [x] Pin `holhouse/PL-NBA-Dataset` commit `0b3d5013b5828004062ed27b10c9eddfca313711` and retain
      only the 4,461,082-byte annotation archive; no original NBA or separately hosted video was
      downloaded.
- [x] Verify archive SHA/CRC and parse 6,408 possession JSON files across 32 games (31,964 events,
      3,607 shot events), including timestamps, outcomes, four reversed intervals, four `nan` rows
      and path/possession-id label leakage.
- [x] Reject paired-media/training/runtime import: no local video, no independently verified NBA
      broadcast rights, no pixel ball–hand–rim causality and no exhaustive full-game hard negatives.
      Keep the compact archive, manifest and audit as offline taxonomy reference only.
- [x] Register the fail-closed catalog entry and regenerate 57 sources with internal SHA
      `93542b43e62169f7c25946f3dc910fbb46f9cfe8f76f643247566f88d03efb9f`.
- [ ] Continue toward a rights-cleared paired continuous-video source with sub-second causal
      ball–hand–rim outcomes and exhaustive non-shot hard negatives; do not use PL-NBA labels for
      runtime answers, model promotion or blind inference.

## TAL3x3 temporal action annotation gate (2026-08-07)

- [x] Pin `open-starlab/TAL3x3` commit `f1093d62818c3e8930561fda5939db19c0981d95` and inspect its
      public Drive inventory; do not download the separate 271.6 MB skeleton archive or any video.
- [x] Download and verify only `dataset.zip` (`5,094,624` bytes, SHA-256
      `0dbcf92e866fd8b8428f297b48c80e4aae603a1845d432d27bc5e669ec3db28f`): 318 clips, 10 source
      videos, 1,881 frame-bounded events, 387 shot/free-throw outcome events and 105,589 bbox frames.
- [x] Reject paired-media/training/runtime import because the repository/archive have no dataset
      rights statement or paired video, and there are no ball boxes, causal ball–hand–rim order or
      exhaustive full-game hard negatives; paper CC BY 4.0 is not treated as a data license.
- [x] Register the fail-closed catalog entry and regenerate 58 sources with internal SHA
      `460aca7b4a3f72d33666776dcc371854a2741d21290f176dfcf27b072836c5f4`.
- [ ] Continue toward a rights-cleared paired continuous 5x5 video source with sub-second causal
      ball–hand–rim outcomes and exhaustive non-shot hard negatives; keep TAL3x3 offline-only.

## MUVY multi-view basketball auxiliary sample gate (2026-08-07)

- [x] Pin Zenodo `10.5281/zenodo.13883315` and verify its CC BY 4.0 record metadata and ZIP64
      central directory without downloading the 7.76 GB archive.
- [x] Extract the 26 basketball metadata files and two `basketball_event_01` camera videos via
      bounded Range requests; verify the 34,211,695 retained media bytes with SHA/ffprobe.
- [x] Record the 232/48 ball-box rows, title leakage and diagnostic 32.7-second audio offset in
      `analysis_outputs/public_research/muvy_basketball_audit_v1.json`; keep it offline-only.
- [x] Add the fail-closed catalog entry/regression test and regenerate the 59-source catalog with
      internal SHA `d9b5b85feedf1309e7b5aa0d2c9b2bc978b3e93fee88ff63eb29133b61ba18c2`.
- [ ] Do not use MUVY as shot-outcome truth; continue toward licensed continuous 5x5 causal
      ball–hand–rim supervision and exhaustive non-shot hard negatives.

## Basketball Events annotation re-audit (2026-08-07)

- [x] Re-pin the four-game Basketball Events tree at revision
      `26d3775286f542b41daf4a94a53190440d426111`; keep only existing bounded annotations and the shot
      subset rather than downloading the 7.86 GB upstream media collection.
- [x] Confirm 543 clips/897 events (601 shots, 153 assists, 110 rebounds, 33 blocks), 58 duplicate
      event tuples, 776/897 integer clock matches, and event-centered rather than continuous coverage.
- [x] Confirm train/val has zero path overlap but shares all four games; no frame-level causal labels,
      explicit rebound polarity or exhaustive non-shot hard negatives; keep `runtime_consumable=false`
      and `training_media_eligible=false`.
- [x] Hash/ffprobe the eight temporary sample clips, then permanently clean duplicate metadata/media and
      stale rejected candidate copies. Re-audit artifact:
      `analysis_outputs/public_research/basketball_events_reaudit_v1.json` (SHA
      `f4e19cbf3958fb5ee9543943ecee97abc75165c2affc52df4ee0d1c93943b776`).
- [ ] Continue only with a rights-cleared continuous 5x5 source containing sub-second causal
      ball–hand–rim outcomes and exhaustive non-shot hard negatives; do not use this source as runtime,
      training or blind truth.

## NBA Games full-game video/PBP index gate (2026-08-07)

- [x] Pin `choucsan/NBA_Games` revision `cf59e3a42413e3ab91f6fc0b1618f280df9a7024`; download only
      README, index, box-score and play-by-play metadata (63,619,934 bytes), not linked YouTube media.
- [x] Verify 189 unique game/video references, 5,194 box rows and 81,355 PBP rows; 166 PBP files are
      non-empty, 23 are empty; count 12,659 made-shot and 15,030 missed-shot actions.
- [x] Keep fail-closed: MIT covers the released metadata only; underlying NBA/YouTube video rights and
      frame-causal labels are not verified. `videoAvailable` is an action-feed flag, not local media.
      Audit: `analysis_outputs/public_research/nba_games_fullgame_audit_v1.json` (SHA
      `7fc8df8aeb002962743f08bed240c27890f31e714183d699f57ae954bee9091a`).
- [ ] Pair only a separately rights-cleared continuous 5x5 video subset to this PBP index, then verify
      sub-second release/contact/outcome truth and exhaustive non-shot hard negatives before any promotion.

## Independent VLM evidence-gate integrity hardening (2026-08-07)

- [x] Require auxiliary OOF coverage for every VLM event and reject duplicate required observables.
- [x] Validate finite `[0, 1]` auxiliary probabilities/thresholds; abstain invalid live VLM confidence.
- [x] Verify with 913 full tests, targeted Ruff, Harness, compileall and `git diff --check`.
- [x] Keep the prior screen metrics and rejection status unchanged; no runtime/VLM/EBQwen/blind asset
      changed.
- [ ] Obtain and validate a rights-cleared continuous 5x5 cross-broadcast causal source with exhaustive
      non-shot hard negatives.

## Shot-causal support threshold protocol correction (2026-08-07)

- [x] Replace the zero-filled causal training-score threshold selection with nested game-held OOF scores.
- [x] Add regression coverage; canonical `.venv` full suite passes `914 passed, 15 warnings`.
- [x] Keep corrected artifact `analysis_outputs/public_research/shot_causal_support_screen_v2.json`
      rejected (`P/R/F1 0.557562/1.000000/0.715942`, weakest precision `0.414634`).
- [ ] Gather a rights-cleared, broadcast-diverse continuous 5×5 source with causal ball–hand–rim
      outcomes and exhaustive non-shot hard negatives; keep runtime and blind boundaries unchanged.

## HOU–SAC tiled ball candidate VLM probe (2026-08-07)

- [x] Review 48 label-hidden raw-frame candidates (`11 valid_ball / 2 uncertain / 35 false_positive`).
- [x] Run the 12-row EBQwen MLX-4bit probe under `.venv` resource guard; all states were `uncertain`,
      with lower/upper evaluation P/R `0/0`, `accepted=false`.
- [x] Preserve plan/review/prediction/evaluation/resource evidence under the HOU–SAC development
      output; no detector/runtime/weight/blind asset changed.
- [ ] Obtain a rights-cleared cross-broadcast continuous 5×5 source with causal ball–hand–rim truth and
      exhaustive non-shot hard negatives before attempting another promotion path.

## E-BARD candidate verifier transfer to HOU–SAC (2026-08-07)

- [x] Reuse the sealed 3,675-row E-BARD manifest and HOU–SAC 48-candidate plan with no new payload.
- [x] Run confidence-augmented and visual-only MobileNetV3-small screens under `.venv` resource guard;
      source thresholds were selected from game-held OOF scores only.
- [x] Seal both rejected results: confidence lower/upper P/R `0.003423/0.245232`, `0.004259/0.267303`;
      visual-only `0.303030/0.408719`, `0.347475/0.410501`; no checkpoint or runtime change.
- [ ] Move on from threshold-only/E-BARD-only variants; obtain rights-cleared, broadcast-diverse
      continuous 5×5 ball/non-ball plus ball–hand–rim causal supervision and exhaustive hard negatives.

## MUVY full standardized subset cross-event ball review (2026-08-07)

- [x] Compare the bounded ZIP64 extraction with the existing standardized 39-file MUVY subset;
      delete the exact 41-file duplicate extraction and all unreferenced temporary central-directory
      files.
- [x] Decode and inventory all 13 videos/13 annotations across five events: 513 source ball rows,
      zero ffprobe/frame-count mismatches, and no hand/rim/shot-outcome or continuous 5×5 labels.
- [x] Manually review the 373 hash-bound candidates and seal 74 valid balls, 5 uncertain rows and
      294 hard negatives; materialize `muvy_ball_yolo_v2` with 74 images/boxes and a 57/17
      event-held split, all `runtime_consumable=false`.
- [x] Record audit `analysis_outputs/public_research/muvy_basketball_cross_event_review_v1.json`
      (artifact SHA `430ea5e5951b1eef6312423b1cbe1fa98dcaadf2e9770424653d2fc2ddf1c49d`).
- [ ] Keep the result as offline auxiliary evidence only; the next useful increment must be a
      rights-cleared broadcast-diverse continuous 5×5 source with sub-second ball–hand–rim outcomes
      and exhaustive non-shot hard negatives.

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

- [x] Verify and download the 640×360/29.97-fps OKC–CLE stream for offline-only review; seal SHA
      `e9955b698de9894276de90a0f79ff2dba21322ae04476287103bc55accb5414a` and decode checks.
- [x] Manually review 32 label-hidden windows with the strict live-release protocol: `0` qualifying
      positives, `32` negatives (free throws/replays/stoppages/no-release/postgame).
- [x] Seal the compact audit `analysis_outputs/public_research/okc_cle_2025_cross_broadcast_screen_v1.json`
      (file SHA `5de8f68f965a46e8b11f8b2a03360262d60ff694f2081c6e`) and delete the video plus temporary
      inspection crops; retain only the 2.4 MB audit package.
- [ ] Keep this source rejected and offline-only. The next promotion attempt still requires licensed,
      broadcast-diverse continuous 5×5 ball–hand–rim outcome labels and exhaustive non-shot negatives.

## Current readiness audit (2026-08-08)

- [x] Seal `analysis_outputs/public_research/agu_readiness_audit_2026-08-08.json` with the latest
      causal-support, cross-broadcast detector, OKC–CLE and official-count upper-bound evidence.
- [x] Verify canonical `.venv` Python 3.11.15, no legacy `venv`, and no active AGU training/runtime
      process; retain EBQwen native/MLX weights and blind assets unchanged.
- [x] Seal the OCR-clock aligned training-only review audit
      `analysis_outputs/public_research/pbp_clock_training_review_audit_v1.json` (48 windows per game;
      conservative positives `20/48` LAL–BOS and `17/48` ATL–CHI), remove superseded interpolation
      reviews and temporary montages, and keep all candidate artifacts out of runtime/independent eval.
- [ ] Keep the goal paused until the rights-cleared continuous 5×5 causal/hard-negative source is
      available and independent per-game P/R >= 0.85 is demonstrated.

## SVI-Bench gate and E-BARD team-color auxiliary (2026-08-08)

- [x] Seal the SVI-Bench metadata-only gate artifact and record HTTP 401 on representative basketball
      payloads; no gated bytes were downloaded.
- [x] Download/verify `dataset/public_sources/e_bard_team_attribution_v1/all.zip` (64,715,118 bytes,
      SHA `f25c2bb8d8527992e68fc952d840f6c379b4ce24412d0dd764bdcc9c95134be7`) and write its source
      manifest; keep the archive compressed as one copy.
- [x] Run the group-held color baseline (accuracy `0.783279`, macro-F1 `0.732430`) and reject it for
      promotion; it has no ball-hand-rim or outcome supervision.
- [ ] Keep SVI-Bench and the color archive out of runtime, blind inference and acceptance truth until
      rights/terms and independent causal evidence satisfy the existing gate.

## APIDIS multi-view/event alignment restoration and detector screen (2026-08-08)

- [x] Restore the 52 predeclared APIDIS files through the public Kaggle file endpoint without downloading
      the 28.56 GB archive; verify all 52 SHA-256 values and seven AVI ffprobe specs.
- [x] Add tested local-time/UTC frame mapping and nested-event parsing; seal
      `analysis_outputs/public_research/apidis_alignment_v1.json` with 4,219 mapped ball frames and five
      Q2 event timestamps.
- [x] Materialize the camera-disjoint YOLO screen (train cameras 1–5, validation 6, test 7) and run a
      guarded five-epoch CPU experiment in canonical `.venv`; no resource stop occurred.
- [x] Screen the held camera and sealed LAL–BOS/ATL–CHI reviews; retain rejection evidence and checkpoint
      hashes in `analysis_outputs/public_research/apidis_alignment_train_v1/retention.json`.
- [x] Delete generated frames, caches and unpromoted checkpoints after sealing hashes; keep source media,
      alignment and compact results offline-only.
- [ ] Do not resume blind inference or promote this detector. Obtain a rights-cleared, broadcast-diverse
      continuous 5×5 source with sub-second ball–hand–rim–outcome labels and exhaustive hard negatives.

## Independent Qwen3-VL negative-first prompt screen (2026-08-08)

- [x] Add and regression-test the label-hidden `negative_first_v4` prompt; bind prompt/sampling/model
      hashes in the independent runner.
- [x] Complete the guarded 32-window qwen3-vl:2b run at 4 frames/256 pixels/context 3072. Three
      available-memory guard stops were resumed safely; final resource log and prediction SHA are
      retained.
- [x] Evaluate only after the run: TP/FP/FN/TN `16/16/0/0`, pooled/per-game P/R `0.500000/1.000000`,
      `promotion_eligible=false`; do not promote the prompt or VLM/base fusion.
- [x] Remove exact resumable caches and the failed context-2048/partial context-3072 outputs after
      sealing hashes; keep compact predictions, delayed evaluation, plan and resource evidence.
- [ ] Wait for rights-cleared, broadcast-diverse continuous 5×5 ball–hand–rim–outcome supervision and
      exhaustive non-shot hard negatives before another prompt-only VLM iteration or blind inference.

## Online continuous-causal source re-audit (2026-08-08)

- [x] Re-audit BASKET, NBA Games, MUVY and BASKET-Multiview official metadata and task scope.
- [x] Download no new payload: each candidate fails at least one of rights, continuous 5×5 coverage,
      sub-second causal outcome labels or exhaustive hard negatives; retain existing compact evidence.
- [ ] Keep the goal paused until a source clears every gate and the independent per-game P/R threshold.

## Independent VLM + frozen scene/video OOF gate (2026-08-08)

- [x] Implement the label-free frozen-threshold gate and regression tests; target labels are not read by
      the fusion command, and missing auxiliary evidence fails closed.
- [x] Seal the 32-event prediction/evaluation pair. Coverage is 31/32; pooled P/R is `0.625/0.625`,
      worst-game P/R is `0.400/0.500`, and promotion is rejected.
- [ ] Keep this screen and the base/VLM chain offline-only; prioritize a rights-cleared continuous
      cross-broadcast causal source before another prompt or threshold variant.

## Continuous causal source gate (2026-08-08)

- [x] Implement and test the metadata-first five-check gate; missing values and non-boolean truthy values
      fail closed.
- [x] Seal the nine-candidate audit at
      `analysis_outputs/public_research/continuous_causal_source_gate_2026-08-08.json`: zero eligible
      sources and zero payload downloads (latest internal SHA
      `29b3b9190f4ad9bdd56a73dde926190d9cfdc4f0c40b01e2e50d8543b81af607`).
- [ ] Do not download or train from a candidate until all five checks and immutable metadata are present;
      after that, require independent per-game P/R >= 0.85 before blind inference.

## NSVA event-text metadata increment (2026-08-08)

- [x] Download the pinned 23,167-byte NSVA metadata subset (1,316 rows) with atomic, byte-capped fetches
      and self-verifying source manifest.
- [x] Normalize intents into the offline event ontology and seal the audit: 953 field-goal attempts,
      7 free throws, 495 rebounds, 220 fouls, 135 turnovers, 293 assisted rows, unknown segments 0.
- [x] Retain only bounded upstream source code for license/download audit; no raw clips, runtime import or
      training-media promotion.
- [ ] Obtain a separately licensed continuous cross-broadcast 5×5 causal source before any media training
      or blind-inference resume.

## GCB/GameCommBench event metadata increment (2026-08-08)

- [x] Download only the pinned basketball metadata JSONL and summary (6,521,665 bytes total); do not fetch
      the declared 21.761 GB MP4 payload because the card says `license=other` and the source is clip-level.
- [x] Normalize and audit 2,981 metadata rows across 132 games with the tested GCB event parser; source
      unknown segments are zero and the audit/manifest hashes are sealed.
- [x] Rebuild the source catalog after registration: 62 sources, internal SHA
      `4921a2cdc65d4b30b2c3471c1dd6064f30fb8cde83d5435c8ac17b3122ec270f`.
- [x] Recheck unresolved visual-state IDs `visual-state-0029` and `visual-state-0094`; retain both as
      `stoppage_other`, medium confidence, unresolved in the manual recheck artifact.
- [x] Deterministically replay the existing base/VLM `both_confirm` fusion under `.venv` with maximum RSS
      339,755,008 bytes; the sealed 32-row artifact matches and pooled/per-game P/R remains `0/0`.
- [ ] Keep GCB metadata offline-only and wait for a rights-cleared continuous 5×5 causal/hard-negative
      source before training, runtime import or blind-inference resume.

## Obsolete rendered-review cleanup (2026-08-08)

- [x] Confirm the four completed `ball_release_hard32*_review_sheets_v1/` directories have zero
      repository references and that compact review/decision JSON remains available.
- [x] Seal `analysis_outputs/public_research/obsolete_review_sheet_cleanup_2026-08-08.json` and delete
      260 rendered JPG files (`158,086,309` bytes); do not delete source videos, labels or model weights.
- [x] Remove only `.ruff_cache` and stale EBQwen download lock files; preserve both complete EBQwen
      weight trees and their metadata caches for future continuation.

## Hugging Face basketball directory sweep (2026-08-08)

- [x] Pull the bounded, latest-updated `basketball` directory index (72 results) without payload
      downloads.
- [x] Reconcile 9 catalog matches and triage 11 high-relevance entries; no new source clears the
      rights/continuous-5×5/sub-second-causal/exhaustive-hard-negative gate.
- [x] Retain the zero-payload audit at
      `analysis_outputs/public_research/hf_basketball_directory_sweep_2026-08-08.json` (internal
      SHA `8deef5dcbfaffcb8cc0c97a965d97b725adb27ff94262595009c8cad72b0f587`).
- [ ] Leave the remaining 61 directory-only results metadata-only until their cards provide the
      five required checks; do not download by name alone.

## PBP/OCR causal coverage diagnostic (2026-08-08)

- [x] Read the two sealed LAL–BOS/ATL–CHI PBP-clock alignments and bind source/alignment hashes.
- [x] Seal the pooled coverage audit: 873 events → 341 clock anchors, 300 field-goal attempts →
      163 anchors; mapped made shots `0.823129` versus misses `0.274510`.
- [x] Keep the artifact offline-only and rejected; it provides temporal-neighborhood evidence, not
      sub-second ball–hand–rim outcomes or exhaustive hard negatives.
- [ ] Wait for a separately licensed continuous 5×5 causal source before media training or blind
      inference.

## UVY basketball detector auxiliary subset (2026-08-08)

- [x] Pin the [UVY Zenodo record](https://zenodo.org/records/21303900), verify CC-BY-4.0 metadata and
      parse the 3.27 GB ZIP central directory without fetching the full archive.
- [x] Use bounded HTTP Range requests to extract only basketball images and MOT annotations, verify
      ZIP CRCs, and exclude MP4/non-basketball payloads. Preserve the source archive MD5 and per-file
      hashes in `dataset/public_sources/uvy_v1/manifest.json`.
- [x] Seal the image/annotation audit, including the V03 frame-count mismatch and the explicit absence
      of shot-outcome labels. Keep `runtime_consumable=false`, with training scope limited to detector
      auxiliary and hard-negative use.
- [x] Run the canonical `.venv` CPU YOLOv8n transfer screen on 120 V04 test frames: 73 ball targets,
      `TP/FP/FN=0/0/73`, P/R/F1=`0/0/0`, max RSS `410,615,808`; seal screen SHA
      `85393624ef72c7d9d84dff689e0444a34ffed389975234fcfe92735001f114d1` and delete the unpromoted
      generic weight.
- [ ] Run an independent cross-broadcast game-held check before considering any auxiliary detector
      checkpoint; do not alter runtime or official-statistics promotion from UVY.

## UVY auxiliary detector training screen (2026-08-09)

- [x] Add and test the fail-closed training schema/plan, absolute-path YOLO binding and guarded
      offline training CLI. The plan/result remain `runtime_consumable=false`,
      `causal_truth_eligible=false` and `promotion_eligible=false`.
- [x] Run a 1-epoch CPU screen from the retained `model_checkpoints/yolov8n.pt` with V01/V02/V04
      train/val/test splits (`imgsz=256`, `batch=2`, `workers=0`). Plan SHA is
      `8f5d307eddc5affa3576067e3813a770b0aec57c0e1b8d8105cc16644ce2987d`; result internal SHA is
      `2dfe3f1ee7ff089f0021be338484131726f1f2d4476c8da47d2c823962203e36`.
- [x] On 120 independent V04 test frames (72 ball-target frames), best remains
      `TP/FP/FN=0/0/72`, P/R/F1=`0/0/0`. Guard exit 0: peak system memory `84.2%`, process-tree
      RSS `546,029,568` bytes, minimum available memory `2,711,584,768` bytes, CPU `50.0%`.
      Best/last hashes and the child-exit guard SHA are recorded in the plan/docs; both unpromoted
      checkpoints and generated label caches are removed after sealing.
- [ ] Do not promote UVY or change runtime/default detector; require an independent cross-broadcast
      game-held screen that clears the existing per-game gate.

## Online candidate supplement (2026-08-09; metadata-only)

- [x] Recheck [NBA Games](https://choucisan.github.io/collections/nba_games/),
      [BasketHAR](https://huggingface.co/datasets/Xian-Gao/BasketHAR) and
      [BasketLiDAR](https://sites.google.com/keio.jp/keio-csg/projects/basket-lidar) against the
      active rights/continuous-5×5/sub-second-causal/exhaustive-hard-negative gate.
- [x] Keep `payload_downloads_performed=0`: NBA Games has references/PBP but no redistributed video,
      BasketHAR has inertial signals rather than broadcast frames, and BasketLiDAR is request-only.
- [ ] Wait for a rights-cleared continuous 5×5 causal source before downloading another media payload,
      training a promoted detector or resuming blind inference.

## 广播状态融合高维筛选（2026-08-09）

- [x] 在同一 511 窗口、四比赛外层留出协议下，将 scene/video PCA 从 16 扩展到 36，保留
      label-free broadcast-clock 特征；封存工件内部 SHA
      `824d518806344a0ea47e21564f13ec9882ddb4b0fb15e46ca4adae4d16194ca1`。
- [x] PCA36 最佳 `base+broadcast_raw` 达到 pooled P/R/F1 `0.787162/0.943320/0.858195`，
      但最差比赛 precision `0.730337`，未达每场 `0.85/0.85`，`runtime_consumable=false`。
- [ ] 不再把 PCA/正则化微调当作主解；等待权利清晰的连续 5×5 因果数据和穷举 hard negatives。

## 第六场外层留出场景/视频融合屏幕（2026-08-09）

- [x] 将既有哈希绑定的 Lakers–Magic 32 窗口开发切片加入四场基线，生成 575 个窗口并按六个
      source-video 分组做外层留出；全程使用 canonical `.venv`，不把新片写入 runtime。
- [x] 封存工件 `analysis_outputs/public_research/shot_validity_scene_fusion_game3_all_v1.json`，
      内部 SHA 为 `1c63957e77d67142255673291c1163ac42410d885ef1a2c5980cb81720009cac`。
- [x] 最佳 scene+MViT 结果 pooled P/R/F1=`0.586854/0.980392/0.734214`；新增比赛 P/R=`0.400000/0.857143`，
      最差既有比赛 precision=`0.503759`，因此拒绝晋级。
- [ ] 不再把该开发片的参数搜索当作主解；继续等待权利清晰的连续 5×5 亚秒因果标签和穷举
      non-shot hard negatives。

## MUVY 重复渲染表清理（2026-08-09）

- [x] 逐文件比较历史 `muvy_ball_codex_review_v1/sheets` 与 canonical
      `muvy_basketball_v1_review_v2/sheets`，11 张 JPG 的大小和 SHA-256 全部一致。
- [x] 封存 `analysis_outputs/public_research/muvy_duplicate_sheet_cleanup_2026-08-09.json`，
      内部 SHA 为 `aabfe7d23f89d00647de37700346d519c29d5ad9e8a1d161b8d4b945ecd70fe8`，删除仅
      这 11 个无引用副本（`14,931,003` bytes），保留决定、manifest 和 canonical 图。
- [ ] 保留 canonical MUVY 复核与原片；本清理不改变 runtime、detector、VLM、EBQwen 或盲推理。

## 在线候选补充：MEV 与 VSTAT（2026-08-09）

- [x] 固定 MEV Hub revision `1e9460d5909116807dd45345b174fb91e9244553`，检查完整事件/视频元数据和
      mixed-third-party-licenses 说明；篮球关键词仅命中 39 个 UUID、504.725 秒，最长候选片段跨度
      27.694 秒，公开树没有逐视频 source manifest。
- [x] 固定 VSTAT revision `38ef1caea89af3950fd274bf83415dcdc29c710b`，检查 QA 与 YouTube metadata；
      192 条篮球问题全部来自 3 个 YouTube 源视频的 30 个 clip，篮球视频未随 CC-BY 标注再分发。
- [x] 封存 `analysis_outputs/public_research/online_candidate_supplement_2026-08-09.json`（内部 SHA
      `820bd8d0f15502965bc0dd7c02ecb01bdd178095eb6dd208f1a144f996f9cff5`），本轮没有下载新视频，
      MEV 已加入 metadata-first catalog/gate，均保持 `runtime_consumable=false`。重建后的 catalog
      SHA 为 `222f7c40c64bb9080e3a4ab045627a8227b3b00b575bc636c2a826e70d7da6b4`，gate SHA 为
      `ae21c18ee551b5a5e4abbfcf4e132a8b955e73a8e30642068cf69417de643a07`（64 个来源、11 个候选、0 个
      eligible）。
- [ ] 不下载 MEV 多 GB video shards 或 VSTAT YouTube 原片；继续等待权利清晰、跨转播连续 5×5、亚秒
      球–手–篮筐–结果标签和穷举 non-shot hard negatives 的来源。

## ExAct 篮球技能反馈审计（2026-08-09）

- [x] 固定 [ExAct](https://huggingface.co/datasets/Alexhimself/ExAct) revision
      `1bd51bfdbd228f850f69cf81d3b4919c71608c04`，仅下载 README、metadata JSONL 与文件树，未下载 MP4。
- [x] 审计精确篮球子集 1,047 行（Mikan Layup 410、Reverse Layup 388、Mid-Range Jump Shooting 249；
      GE/TIPS `165/882`），并封存
      `analysis_outputs/public_research/exact_basketball_skill_audit_2026-08-09.json`（内部 SHA
      `a8b2ba782a2f9bc697b6b1e449d56a7474dd1f3d7d769993ea55dfdcaad2fb7f`）。
- [x] 以 `runtime_consumable=false`、`training_media_eligible=false` 注册为元数据辅助参考；来源目录重建为
      65 条，内部 SHA `6223b0ad4995d757e774044aed18c1e8d8c46a48307e1d55cbecf23ec5a3074e`。
- [ ] 不把操练姿态反馈当作投篮结果、球员统计或连续因果监督；当前 11 个候选仍无 eligible，盲推理继续暂停。

## SportVU 2015–16 轨迹 tiny 分片（2026-08-09）

- [x] 下载并保留五个 hash-bound SportVU `.7z` 档案（压缩后 `29,017,344` bytes）及匹配的 `2,208` 行
      PBP；审计后删除完整 PBP CSV 与展开 JSON。
- [x] 封存 `dataset/public_sources/nba_tracking_15_16_tiny_v1/manifest.json` 与
      `analysis_outputs/public_research/sportvu_tracking_tiny_audit_2026-08-09.json`，记录来源 revision、
      未声明 license、档案/字段 SHA、25 Hz moment、球覆盖率和 PBP 事件交集。
- [x] 以 `runtime_consumable=false`、`training_media_eligible=false`、`causal_truth_eligible=false` 登记为
      轨迹/PBP 先验参考；来源目录重建为 66 条，SHA
      `0dc379a8a96d76643fd72d5b78ff453f5359edacec1f1d39903feb2d55ab1d9e`。
- [ ] 不把事件编号交集当作广播帧对齐或投篮结果真值；继续寻找权利清晰、跨转播连续 5×5、含亚秒球–手–
      篮筐–结果标签和穷举 non-shot hard negatives 的来源。

## SportVISTA 许可门禁审计（2026-08-09）

- [x] 固定 SportVISTA revision `ffb720af2a1e23ff7a8f39379a0f9606cc110e3f`，只下载 API/README/LICENSE
      元数据；v0.1.0 只有文档，人工 gated 且无行级 manifest、标注或媒体 payload。
- [x] 封存 `analysis_outputs/public_research/sportvista_online_audit_2026-08-09.json`（内部 SHA
      `63f0970e111210ff82bf50343aab99319bd27bdd085fb24de6a7e7b13cfbe50a`），确认其许可禁止 AGU 基座/VLM
      的通用模型训练、评估或改进；访问请求和媒体下载均为 0。
- [x] 登记为许可拒绝的 metadata-only 参考，catalog 更新为 67 条（SHA
      `d2808333516f6f7f22390e3e51231edf3edaf6081bc0281a0d54fb3003931d88`），不进入 runtime、训练媒体或
      连续因果 gate。
- [ ] 不申请 SportVISTA gated payload；继续寻找可合法用于 AGU 基座+VLM 且具备连续 5×5 因果标签的源。

## Wikimedia Commons / HCTV 连续全场种子（2026-08-09）

- [x] 审计 Wikimedia Commons 篮球视频目录：23 个 HCTV 完整比赛条目、约 35.44 小时、逐文件 CC BY 4.0；
      原 YouTube 元数据也报告 Creative Commons Attribution，记录 Commons 的 `License review needed (video)` 警示。
- [x] 只下载一场 `Hazen Boys Basketball playing Lyndon`，断点续传后以声明长度 `1,416,244,469` bytes、SHA-256、
      AV1/Opus `ffprobe` 和 30/900/1800/3000 秒帧样本完成完整性/连续性审计。
- [x] 保留原片作为人工标注/硬负样本底座，写入 manifest、审计工件、source catalog 与 2026-08-09 gate；
      `runtime_consumable=false`、`training_media_eligible=false`、`causal_truth_eligible=false`。
- [ ] 需要第二个独立转播/制作源，并人工复核亚秒出手—篮筐—结果、篮板/助攻及穷举 non-shot hard negatives；
      当前 12 个 gate 候选仍为 0 个 eligible，盲推理保持暂停。

## Wikimedia Commons / VTV 独立制作全场种子（2026-08-09）

- [x] 审计 VTV 的 Commons Public domain / `PD Venezuela official` 声明，并记录原 YouTube 元数据未声明
      license 的边界；未把 YouTube 页面当作独立权利依据。
- [x] 下载并校验一场 Spartans–Trotamundos 连续全场：`2,228,583,591` bytes、VP9/Opus、1920×1080、
      30 FPS、约 2 小时 42 分；四个时间点均可解码并看到 VTV 比分牌/全场画面。
- [x] 封存 VTV manifest、审计与 HCTV+VTV 跨制作配对审计；两个独立制作源已具备，但仍无亚秒因果标签
      或穷举 non-shot hard negatives，不能进入 runtime、训练真值或盲推理。
- [x] 建立低资源试标注入口并保留 20 个哈希绑定 JPEG 帧：HCTV 1 个 `uncertain` 窗口、VTV 2 个
      `not_a_shot` 候选和 1 个近篮筐 `uncertain` 窗口；计划/复核工件标明 `pilot_only`、待人工确认、
      `hard_negative_gate_eligible=false`，不把试标注冒充穷举真值。
- [ ] 在 HCTV 与 VTV 两场上完成人工出手—篮筐—结果、篮板/助攻和穷举 hard-negative 标注，再以一场
      完整留出做逐场 P/R 门禁；当前 gate 为 14 个候选、0 个 eligible。

## 临时数据清理（2026-08-09）

- [x] 按用户授权删除已被正式审计工件替代、且无项目/Wiki 引用的 `/tmp` APIDIS 训练缓存、MUVY
      临时下载/图片和旧元数据探针，共 `425,154,113` bytes；清单与内部 SHA 见
      `analysis_outputs/public_research/agu_tmp_cleanup_2026-08-09.json`。
- [x] 复核 HCTV/VTV 原片、EBQwen native/MLX-4bit、运行时检查点、正式 catalog/gate 均保留；不删除
      任何正式标注、原片、模型或 blind asset。

## HCTV Randolph 第三连续原片种子（2026-08-09）

- [x] 以 Commons CC BY 4.0 / HCTV provenance 审计并下载 `2,152,005,777` bytes 原片；SHA-256
      `f234d5bfadd0b182c9a15a6d7b206335fcb371e9971cd2bf67313f544dc6a102`，AV1/Opus、1920×1080、60 FPS、
      `5,899.308s`。
- [x] 记录赛前/捐赠插播与现场全场样本，建立单源 audit、三源 cross-source audit，并把 gate 更新为
      16 候选/0 eligible；不进入 runtime、训练真值或盲推理。
- [x] 建立 Randolph 五帧离线标注 pilot：中场运球、无脱手—篮筐链，保守 `not_a_shot` candidate；
      `pilot_only=true`、待人工确认、非穷举 hard negative。
- [x] 逐帧复核 Randolph 4997.0–4999.0 秒篮下候选窗口，保留 21 帧并封存为 `uncertain` v2；
      出手分离、向篮圈推进和结果不可可靠闭合，故不声称 shot-positive、不进入训练或 runtime。
      v2 plan/review/manifest SHA 分别为 `25abad0c2c6268161cb916b7af703cd7d160eff7bad8c643e5396397b597639e`、
      `a3eff9b24ed9abdb1f8f36ba4eaaeea5bfcc28c6dc5ce64c473dcfa4de9f158d`、
      `4da2b9d59259a66a4296eb6d0a167aaf8124a11bb85f4af2951524ad6e231959`。
- [x] v2 哈希封存后精确删除 36 张不再引用的候选中间 JPEG（`6,443,465` bytes）；删除审计为
      `analysis_outputs/public_research/randolph_pilot_intermediate_cleanup_2026-08-09.json`，内部 SHA
      `5ef8e77482bd66bd8806d20537a633dafc7d53d06ae13fca48efcf4641007401`。
- [x] 删除已入正式审计的四个精确 Randolph `/tmp` 元数据探针；原片、抽帧、manifest/audit、pilot 均保留。
- [x] 扩展三源低资源人工 pilot：`wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v2/` 保留 129 张 JPEG、6 个窗口，
      包含一个 HCTV Hazen `shot/missed` 因果窗口和五个跨制作源 `not_a_shot` 边界窗口；plan/review/manifest SHA 为
      `964a4a5170c2cd3d5ab903739483033b3fc58b1f1b60db373af761119cdd1b36`、
      `9c76caf5fd1dee0d9c65b409c75399bf4f3a4b5f9549cd405c41d670c8a6819a`、
      `95798600013a59817f908d3a1b5cfda64dd5cc27d73c5306c691bbd0e258a164`；仍为 pilot-only、待人工确认、非穷举。
- [x] readiness 已记录 175 帧/12 窗口；runtime、训练真值、默认模型和盲推理均未改变，causal gate 仍为 16 候选/0 eligible。
- [ ] 三场原片继续完成人工事件、插播区段及穷举 hard-negative 标注，再做制作源留出 P/R 门禁；未通过前保持盲推理暂停。

## NBA Games PBP 重复清理（2026-08-09）

- [x] 精确核对并删除 `dataset/public_sources/nba_games/games/` 的 567 个重复 PBP 文件（`63,474,295`
      bytes）；canonical `dataset/public_sources/nba_games_v1/games/` 逐文件同 SHA，未触碰媒体、正式审计、
      三场 Wikimedia 原片、EBQwen、训练真值或 blind assets。
- [x] 保留删除前 inventory/tree SHA 与删除记录：
      `analysis_outputs/public_research/agu_nba_games_duplicate_cleanup_2026-08-09.json`
      （SHA `382221ea8b557930efc66915e7002df61a65eba8c33266430b15a1409e7433f2`).

## Three-source causal extension v3 and MLX-4bit screen (2026-08-09)

- [x] Scan the three retained full-game originals and keep only two detailed windows after rejecting
      transition, interstitial and dribble samples.
- [x] Retain 45 hash-bound frames in
      `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v3/`:
      one Hazen `shot/made` causal chain and one VTV `uncertain` mid-flight cut. All remain pilot-only,
      pending human confirmation and outside runtime/training/blind paths.
- [x] Run the independent EBQwen MLX-4bit native-video screen; the VTV causal-support failure is a
      false positive, giving precision `0.50` / recall `1.00` on this two-window pilot. Screen SHA is
      `f2a8d4a3cb0633ea82d9c9965f8ce2597b8bb77a43cc42b4fda62430fdf7338d`; no model promotion occurred.
- [x] Delete five exact unreferenced visual scratch directories after verifying no project/Wiki references;
      reclaimed 417 files / `59,775,837` bytes, with cleanup SHA
      `425e712768ca02fab2b9c42382cd508f18e7aac3af0821e568433e9c9fa8b73d`.
- [x] Extend the same three-source pilot to v4 without a new download: retain 89 hash-bound frames across
      four windows, including Hazen `shot/missed` and Randolph `not_a_shot` live-drive hard-negative candidates;
      v4 review/manifest/plan/retention SHA are
      `6cdc32bee69379ada7a71be23e56a5857d5fb31889b3f7b45e806d6652f5670f`,
      `ce6d7a0dfe7f4a0238db509ae638c70fffe59b5d79f88d48c0efb1b13f2f26a0`,
      `06774f5a5200118e142652f8c3a30bcbafaa19e43cae8d9cf9d7765353157c7a`, and
      `452c7d0190e4fc589e2da36058c79cfd546d6ab22cde2c11a2c6e9ec23f6c017`.
- [x] Delete the exact unreferenced v4 scan tree `/tmp/agu_shot_scan_v4.5gB2nw` after inventory, reference and
      holder checks: 3,068 files / `263,256,416` bytes; cleanup audit SHA
      `73ce024735b985cf4bb7016bc3033a4b3ab6a82f6adfa27a6721224f9e642796`.
- [ ] Keep the v4 package pilot-only and complete exhaustive three-game event, interstitial and hard-negative
      labels before any runtime, training-truth or blind-inference change; the 0.85 held-production gate is open.
- [ ] Complete exhaustive three-game temporal labels and a held-production P/R gate; blind inference
      remains paused until the 0.85 per-game contract is met.

## 三场原片 v5 因果扩展（2026-08-10）

- [x] 在既有三场原片上新增 6 个有界窗口，继承 v4 的 4 个窗口；v5 封存 10 个窗口、215 张原始/JPEG
      哈希绑定帧，无新增下载。
- [x] 完成 v5 review/manifest/plan/retention 封存，SHA 为
      `fa3938f2ee217fe70ba10d9633325b1616a4ed9aa19f717298cc5f06a44fd9c3`、
      `b5576dd60ee0202554aa4e4a8b0714b215cc15a30fa32517699c4a6421d55913`、
      `972a16c19e12fd357e0a742253303efc7b513aa3eaf9af5d38e2b0ed04912a37`、
      `73544ecf74434123a6abb6b7067aaf51cdf87367166fbc19ad5a672e06887b39`。
- [x] 精确删除 v5 视觉复核草稿和原始哈希草稿：29 文件、`17,763,243` bytes；清理审计 SHA
      `a7047699b33873c04c9eb02c8d9d96dc1fc9017d5f0b849245ce4a7cfd4c5d49`。
- [ ] 继续把 pilot 扩展为三场穷举事件/插播/hard-negative 真值，并在通过逐场 P/R ≥ 0.85 前保持
      runtime、默认模型和盲推理不变。

## 三场原片 v6 因果/硬负样本扩展（2026-08-10）

- [x] 联网复核候选数据集；没有新的源同时满足连续 5×5 原片、亚秒球—手—篮筐—结果标签和可训练权利，因此没有新增 payload 下载。
- [x] 从已保留的 Hazen、Randolph、VTV 原片新增 6 个有界窗口，和 v5 合计 16 个窗口、341 张哈希绑定帧：4 个保守出手（1 中/3 失）、8 个 `not_a_shot`、4 个 `uncertain`。
- [x] 封存 v6 plan/review/retention/manifest，SHA 为 `898d2773edb9cb94c3a485b856a17f69a7a7bc5fb9f6d9793fd29a46fb17c985`、`e8f62182a9586880ee5fdcf7df7bb7cd8210a9c2527907797936e90e8aa68655`、`3b4d1b3af6ff49f5bd8b38fa0c5af13df98062f09490f4d1f328b83ed2476b83`、`d71f99db4a0c2ba99232300144d3adf3d66de137b21c3d30352e152a99f6d89e`；仍为 pilot-only、待人工确认、不可供 runtime/训练/盲推理使用。
- [x] 精确删除 `/tmp/agu_v6_broad.1Q6WQl`：183 文件、`7,745,976` bytes；清理审计 SHA 为 `5221abb861d68b39eb5725e961ebdce6207f85533aafb96ee032e63ca0301920`。
- [ ] 继续完成三场穷举事件、插播边界和 non-shot hard-negative 真值，并通过逐场 P/R ≥ 0.85 门禁；在此之前保持 `not_ready`、盲推理暂停和默认模型不变。

## v6 独立原片 VLM 屏幕与提示对照（2026-08-10）

- [x] 构建不泄露 v6 标签的 16 窗口 native-video 计划，使用 2 FPS、151,200 max pixels 和本地 EBQwen MLX 4bit。
- [x] `strict_release_v2` 与新增 `negative_first_v3` 均对 16/16 窗口输出 `live_field_goal`；TP/FP/FN/TN=`4/12/0/0`，precision=`0.25`、recall=`1.00`，未达到 0.85 门禁。
- [x] 封存 `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v6/independent_vlm_causal_screen_v6.json`，更新 SHA `f709a0a4ba6c3fe70277e4f304ab2e91a28cfefdca7dfb788a5704261b91bbb3`；两轮资源守护均未停止。
- [x] 整批屏幕后精确删除不再引用的单例 probe 计划/预测/cache：3 文件、`4,662` bytes；清理 SHA `b9a677ea2e7b3ca0d03e4f370c2d7bdfd9e8d55494c420811b0b70d63c42f98b`。
- [ ] 为三场制作源补齐独立 auxiliary OOF 证据并通过 fail-closed evidence gate；在此之前不晋级 VLM、不改 runtime、不恢复盲推理。

## v6 独立 auxiliary transfer 屏幕（2026-08-10）

- [x] 新增标签隐藏的 transfer-only auxiliary 契约与 focused tests；模型输入不接受
  review/label 字段，artifact 结构强制 `oof_predictions=[]`。
- [x] 用固定 WASB-SBDT DeepBall-Large 权重在 `.venv` + 资源守护下跑完三场原片的 16 个窗口；
  screen SHA `a528033a276c3bf88d7c55ab99bc5eccf36f6e445fed01c80ce8b33c28517d0c`，内存/CPU/RSS 峰值
  `77.3%`/`44.5%`/`1,773,486,080` bytes，未触发停止。
- [x] 延迟读取 v6 review 做只读诊断；评估 SHA
  `2c19eaa7c4b6f5b908bd18ec65b53dce8c8d09168700eda0117ad87ff0e48a25`，描述性 0.75 cutoff 的 pooled
  P/R/F1=`0.667/1.000/0.800`，但 VTV precision=`0.0`，没有选择阈值或形成 OOF。
- [ ] 取得真正 game-held auxiliary OOF rows，运行 fail-closed evidence gate；在逐制作源门禁通过前保持
  runtime、默认模型和盲推理暂停。

## NBA_Streaming 发布核验与独立证据转移（2026-08-12）

- [x] 论文/HTML/arXiv source 核验无官方 payload URL、hash manifest 或 release；审计 `analysis_outputs/public_research/nba_streaming_release_audit_2026-08-12.json`（`3a571ad41cf443d2fc06ddbffedc13702e65c9b06c173f0f2411de468a79cd9d`），0 bytes 下载，source gate 19/0。
- [x] `.venv` 资源守护下完成同一 compact256 冻结计划（plan SHA `ab36c2ec4eb66a0f431a49a6230e28a3f1947fbb2a7a748a6a37306009cf6b7c`）的 32 窗口 DeepBall transfer；独立 VLM/auxiliary 默认 0.5 一致 16/32，无标签、无 OOF、无 fusion/promotion 输出。transfer SHA `b356a238888f3f7d499a87f7bfdfc675a1b6fe0590552e9f2024a20706175758`；同计划一致性 SHA `e6aa8e2196a99f95f94a6e5f3b8a2bba558b81e6b2386533dc96155f1e2eae68`。
- [x] 屏幕后精确删除项目级可重建 cache 共 71 个文件（2,043,435 bytes），审计 `analysis_outputs/public_research/agu_2026-08-12_post_screen_cache_cleanup.json`（artifact `b49ad388c4fd5330e4bdf9ec35a80ea2d2c29cd09e8121eaa1437e5fc8f3a7b6`）。
- [x] readiness 更新后文件 SHA `3643646dcb45520eee2a30f4430a564e02ea99b8604a4a89a4d776b3cfff8267`；source gate 文件 SHA `399a8f1f81a339064dc8862b52d77f9248dc06c963a250ea738cd92319038bf5`；状态仍 `not_ready`、blind inference 暂停。
- [ ] 继续三场全量亚秒因果真值、插播边界和穷举 hard negatives；blind inference 保持暂停。

## 在线源扫查、外部校准与跨转播媒体清理（2026-08-10）

- [x] 审计 BARD、E-BARD、SportsMOT、SpaceJam、Basketball Events、NBA Games 与 legacy NBA PBP video dataset；没有候选同时满足连续 5×5 原片、亚秒因果标签、穷举 hard negatives、权利和可复现实物五项 gate，未下载新的 payload。扫查 SHA `8c9c46f42f5bd3a90c7b1905a25276b074109092d916f743670ceadcbfeb7dbd`，gate 为 18 候选/0 eligible，SHA `c234381b7806c048bb20c97af4ce1d191298c5f9260b6e3c496c39656b61ac30`。
- [x] 完成三份外部 DeepBall-Large 屏幕的 game-held 校准诊断：96 事件、独立阈值 `0.0`、校准 precision `0.177083`；冻结后目标 determinate precision `0.333333`、recall `1.0`；工件 SHA `7d45fb7d27938aa90c8962884c8dbde851420715f243f6a0503962c1a8651b7d`，`oof_predictions=[]`。
- [x] 在无活动占用且 manifest/screen/evaluation 已留存后，删除三份精确 allowlist 跨转播 MP4，共 `1,909,214,774` bytes；清理审计 SHA `d4c1f1907ebff932d1c3de28aced37e536f82a82c80a1e271bc0f9fa4166d4c4`。
- [ ] 继续完成三场穷举事件、插播边界、non-shot hard-negative 人工标注，并通过制作源留出逐场 P/R ≥ 0.85 门禁；在此之前不改 runtime、默认模型或盲推理。

## v16/v18/v21 标签隐藏 transfer-veto 探针（2026-08-12）

- [x] 6 个窗口的 native VLM probe 完成，6/6 live；三制作源 P/R 均 `0.50/1.00`，不晋级。
- [x] DeepBall transfer 无 OOF，label-free consistency 2/6；固定阈值 veto 仅保留 2/6，事后 P/R `0.50/0.3333`，不进入训练/runtime。
- [x] 新契约和 3 个测试已加入，强制 plan/key 对齐、无标签、无 OOF、fail-closed abstain。
- [ ] 仍需三场全量亚秒真值、插播边界、穷举 non-shot hard negatives 与 game-held OOF；完成前保持 `not_ready`、blind inference 暂停。
- [x] 回归后只删除本轮 34 个可重建 cache 文件（`552,785` bytes）；模型权重、原片、raw frames、正式工件和 `.venv` 保留。
- [x] 最终回归后再次精确清理 34 个新生成 cache 文件（`552,785` bytes），最终清理审计 `agu_transfer_veto_final_cache_cleanup_2026-08-12.json`。

## 三场源间隔离 v21 held-out 复核（2026-08-12）

- [x] 完成 36 个源间隔离窗口、756 张原始帧复核：1 `shot/unknown`、8 `uncertain/unknown`、27 `not_a_shot`；v8–v21 合计 474 窗口/9,954 帧。
- [x] 更新 readiness/source gate；仍为 19 candidates / 0 eligible、`not_ready`、blind inference 暂停，P/R 不可计算。
- [x] 删除 v21 接触表和项目级可重建 cache/bytecode；初次 44 文件、17,227,410 bytes，回归后最终新增 19 文件、145,007 bytes；唯一文件合计 55 个、17,308,898 bytes。原片、raw frames、权重和正式工件保留。
- [x] 复查 4 个新增在线候选，0 个通过因果下载门禁、0 bytes 新媒体下载；follow-up 审计已写入 gate/readiness。
- [ ] 继续完成三场全量亚秒球—手—篮筐—结果、插播边界与 non-shot hard negatives，并通过逐制作源 P/R ≥ 0.85 门禁；v21 仍不进入训练、runtime、VLM 答案或盲推理。

## 三场源间隔离 v18 held-out 原片复核（2026-08-12）

## v19 三场源间隔离 held-out 离线复核（2026-08-12）

- [x] 生成并 hash-bind v19 批次/计划（受限 8 秒源内排除半径），三源共 36 个窗口、756 张帧。
- [x] 封存 2 `shot/unknown`、3 `uncertain/unknown`、31 `not_a_shot`；保持 offline-only、pilot-only、非穷举。
- [x] 清理未引用 contact-sheet 与可重建 cache/bytecode：初次 169 文件、14,473,033 bytes，回归后新增 261 文件、5,568,909 bytes；正式原片、权重、raw frames 与审计工件未触碰。
- [ ] 继续全场穷举亚秒因果、插播边界和 non-shot hard negatives，再按制作源做 held-production OOF P/R ≥ 0.85；之前不得恢复盲推理或改变 runtime 默认。

- [x] 排除 v2–v17 源内 20 秒锚点邻域，按 Hazen/Randolph/VTV 各 12 个 early/middle/late 窗口生成 v18 批次；内部 SHA `b0d8f669819cdb774fe0e97d2e787f0e5e51549ac6c622c61cd9b7421c4651095`，文件 SHA `0fd117c46975dfc4d1b536cf2b198ab0016f66639e118203e1105481bda6cc6c`。
- [x] 使用 canonical `.venv`（Python 3.11.15）物化并离线复核 756 张 hash-bound 帧；封存 3 个 VTV `shot/unknown`、33 个 `not_a_shot`、0 个 `uncertain`，标签不进入 AGU 训练、runtime、VLM 答案、阈值融合或盲推理。
- [x] 封存 review/pilot/plan/frame/retention 工件，v8–v18 合并为 366 窗口/7,686 帧（21/284/61；5 made/5 missed），逐制作源 P/R 仍 `not_computable`，不构成穷举监督。
- [x] 精确删除未引用的 v18 contact-sheet（36 文件、17,491,608 bytes），审计为 `analysis_outputs/public_research/agu_v18_visual_sheet_cleanup_2026-08-11.json`，内部 SHA `44665334d5f2d47bab57cc42519bd95146c764ff7af11b1ec67d9fa4de8e5cc1`；raw frames、原片、权重和正式证据保留。
- [x] readiness 审计文件 SHA `53c8ca9aec207779e4cc544d68e0f7c14544034b97e125c9f1b353d4e5b61cd6`；source gate 20/0，`not_ready`、盲推理暂停、runtime/model promotion 未改变。
- [x] 50 项 focused contract tests 通过后，按 allowlist 精确删除 26 个可重建 cache 文件（546,040 bytes），清理审计为 `analysis_outputs/public_research/agu_v18_repository_cache_cleanup_2026-08-12.json`（内部 SHA `475c2ec7afb0c09bca161f37ea2a97b39cfcd5b0f7e987df813d879557bbfa2e`，文件 SHA `808cee908061179f7243462d9dd1da9e5aa978c105c89fc8171cbd42865d9324`）；`.venv`、dataset、原片、权重和正式证据未触碰。
- [x] readiness 审计随后重封，当前文件 SHA `a1e7c1a93676dd7a97fa12454494603d7d790ae9a7a760f41b876c29a6e01f58`；source gate 20/0，状态仍 `not_ready`、盲推理暂停。
- [ ] v18 仍为 research-only/offline pilot；继续完成三场全量亚秒球—手—篮筐—结果、插播边界和 non-shot hard-negative 真值，并按制作源留出计算 P/R ≥ 0.85。

## 三场源间隔离 v16 held-out 原片复核（2026-08-10）

- [x] 排除 v2–v15 源内 20 秒锚点邻域，生成 Hazen/Randolph/VTV 各 12 个 early/middle/late 标签隐藏窗口；批次文件 SHA `55236cf3ea2117469a44146384a44283c3ca430216a0057f07c10a3630def3f1`。
- [x] 在 `.venv`（Python 3.11）物化并逐窗复核 756 张 hash-bound 帧，封存 4 `shot/unknown`、30 `not_a_shot`、2 `uncertain`；Codex 结论保持离线，不进入 AGU 训练、runtime、VLM 答案或盲推理。
- [x] 封存 review/pilot/plan/frame/retention 工件，合并 v8–v16 为 294 窗口/6,174 帧（16/218/60；已知 outcome 2 made/5）；仅删除 36 张未引用 contact-sheet（15,957,173 bytes），清理审计内部 SHA `83e74205450c1dfc6f1a5ed7fbcee61d97899a6a3c04dffab39a347121a80ad8`。
- [x] readiness 审计已重封，文件 SHA `5568d665497db3d83207638a74c577d73e92bda95d3aaeb62aed6a5a4fe67d42`；contact-sheet 已核空，原片、权重、runtime 和 blind assets 未触碰，状态仍 `not_ready`、source gate 18/0。
- [x] 50 项 focused contract tests 通过；重建的 22 个仓库 Python 3.11 bytecode 文件（540,051 bytes）已删除，清理审计内部 SHA `e9b7f154b0cebbca60868659b4fb155f32cc099d0592cfdb7375e00f3a3a93a9`，文件 SHA `93bd8d6c38cf7679bad4d493f6585c54a570f1458992ad1f2d5b6be9405d838a`。
- [ ] v16 仍是 research-only/offline pilot；继续完成三场穷举事件、插播边界、non-shot hard-negative 人工标注，并通过制作源留出逐场 P/R ≥ 0.85 门禁；在此之前不改 runtime、默认模型或盲推理。

## 三场源间隔离 v17 held-out 原片复核与源门禁复审（2026-08-11）

- [x] 排除 v2–v16 源内 20 秒锚点邻域，生成 Hazen/Randolph/VTV 各 12 个 early/middle/late 标签隐藏窗口；批次文件 SHA `1d88038e2c2378f3f90239201c5b933e017a64b88544d62f3cb265e7a22de69d`。
- [x] 在 `.venv`（Python 3.11）物化并逐窗复核 756 张 hash-bound 帧，封存 2 `shot/made`、33 `not_a_shot`、1 `uncertain`；Codex 结论保持离线，不进入 AGU 训练、runtime、VLM 答案、阈值融合或盲推理。
- [x] 封存 v17 review/pilot/plan/frame/retention 工件，合并 v8–v17 为 330 窗口/6,930 帧（18/251/61；4 made/5 missed），逐制作源 P/R 仍为 `not_computable`，不构成穷举监督。
- [x] 仅删除 36 张未引用 contact-sheet（15,537,554 bytes），清理审计内部 SHA `3cfae6535af0a016a5bf2fd36e7cf485cc30afe4d9a56f408fc3c372b4c05773`；raw frames、原片、权重和正式证据保留。
- [x] 将 TAL3x3 与 TrackID3x3 登记为 metadata-first annotation/tracking 辅助源；source gate 为 20 candidates / 0 eligible，未新增 payload、runtime 或训练接入；随后精确删除两个临时 Git clone（1,836 文件、`390,746,999` bytes）及 62 个可重建 cache 文件（`567,368` bytes），清理审计分别为 `analysis_outputs/public_research/agu_tal_track_probe_cleanup_2026-08-11.json` 与 `analysis_outputs/public_research/agu_v17_repository_cache_cleanup_2026-08-11.json`。
- [ ] v17 仍是 research-only/offline pilot；继续完成三场全量亚秒球—手—篮筐—结果、插播边界及 non-shot hard-negative 真值，并按制作源留出计算 P/R ≥ 0.85；在门禁关闭前不改 runtime、默认模型或盲推理。

## 三场源间隔离 v15 held-out 离线因果复核（2026-08-10）

- [x] 排除 v2–v14 锚点邻域，生成 36 个标签隐藏窗口（Hazen/Randolph/VTV 各 12，early/middle/late 各 4），批次文件 SHA `2777915007fafd0d1feef88bb69deff986d0ca70bfbbc0aab1445651b82d8770`。
- [x] 使用 `.venv`（Python 3.11）物化并完成 756 张帧的离线人工复核；封存 3 `shot/unknown`、30 `not_a_shot`、3 `uncertain`，不向训练、runtime、VLM answer 或 blind inference 提供标签。
- [x] v8–v15 合并为 258 窗口/5,418 帧（12 shot、188 not_a_shot、58 uncertain；已知 outcome 2 made、5 missed）；删除 36 张未引用 contact-sheet（9,709,321 bytes），清理审计内部 SHA `bf98189285052a54131f2c1d58af00ac21e0417ef34fcbfe2678850cd54129c0`。
- [x] readiness 审计文件 SHA 更新为 `bf25e07be2e9043d6d6d5a49e17c4c724c7cb156759bdcc7a8f6beb583536a4a`；contact-sheet 已核空，原片、权重、runtime、blind assets 未触碰，source gate 18/0、`not_ready`、盲推理暂停。
- [x] 50 项 focused contract tests 通过；精确删除 22 个可重建仓库 Python 3.11 bytecode 文件（540,051 bytes），清理审计内部 SHA `53c98d9205c6b370794142f9d178b73b539fa9ccb66515c465264cf4f65e06dd`，文件 SHA `8ff2ef8bcd613c9249231e00cfa79f5d039572bb64024a3d9edf324f218c1555`。
- [ ] 继续全量穷举因果、插播边界与 hard negatives；在三场逐制作源 held-production P/R ≥ 0.85 前不恢复盲推理，不改变 runtime 默认，不把 v15 标签用于训练或阈值融合。

## 三场源间隔离 v14 held-out 离线因果复核（2026-08-10）

- [x] 生成排除 v2–v13 锚点邻域的 36 个标签隐藏窗口（Hazen/Randolph/VTV 各 12，early/middle/late 各 4），批次文件 SHA `7b09f10cde60ecdd583b7a958905ccd21546526a6622549c03ac021b5405f44e`。
- [x] 使用 canonical `.venv`（Python 3.11）物化并复核 756 张 hash-bound 帧；封存 2 `shot/unknown`、26 `not_a_shot`、8 `uncertain`，Codex 结论不进入 AGU runtime、训练、VLM answer 或 blind inference。
- [x] 保留正式 raw frames/封存工件、三场原片和 EBQwen 权重，精确删除 36 个未引用 contact-sheet（12,618,602 bytes）；审计内部 SHA `ad55fb77099cfe733e490ae170d0aef676dfd488348b8392beca7759e470cbee`。
- [x] 48 项 focused contract tests 通过；验证导入生成的 4 个仓库 Python 3.11 bytecode 文件（22,169 bytes）已删除，审计内部 SHA `5619f336a132a381a9819ff56ed3709a05eb6705a7ffed515495c761eae5ec5a`。
- [x] v8–v14 合并为 222 窗口/4,662 帧（9 shot、158 not_a_shot、55 uncertain；2 made、5 missed），readiness 文件 SHA `47a947ab86813ea88f56267a37f4304bde8523c636258b2d4b1dd752332f530d`。
- [ ] 继续全量穷举因果、插播边界与 hard negatives；source gate 18/0、`not_ready`、盲推理暂停，直至制作源留出 P/R ≥ 0.85。

## NBA Games v2 ATL–CHI 四节标签隐藏复核（2026-08-10）

- [x] 新增 PBP 标签隐藏队列构建器与 2 项 focused contract tests；队列只保留窗口几何和 opaque join key，官方真值单独 post-freeze 保存。
- [x] 复用 ATL–CHI 合并 OCR/PBP 对齐覆盖四节 88 个 mapped field-goal 行；队列内部 SHA
  `25dbee3026997b9416748d4324d94a9f2928e8e184ef9e68c9302d84a627e3dd`，manual review 内部 SHA
  `d9149471ff18817578ef08ed39ee618bc9e81247655b4c2fa8e0646b80717fd3`。
- [x] Codex 视觉复核 24 个窗口：21 confirmed、3 partial、0 not_visible、0 exact release；18 made、6 missed；保持
  `runtime_consumable=false`、`training_consumable=false`、`independent_evaluation_eligible=false`。
- [x] 清理 24 个低分辨率 strips 与 2 张临时总览图：26 文件、`6,057,647` bytes；清理审计
  `analysis_outputs/public_research/nba_games_pbp_v2_atl_chi_review_queue/cleanup_audit.json` 内部 SHA
  `db5f19f7c3cd06e2ecb43026932eea7b4e67f25bb73f11ae44a56eac970dd5db`；高分辨率 strips 保留。
- [x] 删除本轮构建产生的 2 个精确 Python 3.11 bytecode 文件（`35,312` bytes），审计
  `analysis_outputs/public_research/nba_games_pbp_v2_atl_chi_review_queue/targeted_cache_cleanup.json` 内部 SHA
  `7f58325e6a6574db68880f996490655e085e7b1b80d19138707b8fc0ca235002`。
- [ ] 继续完整时钟/插播边界、精确 release、球—手—篮筐—结果和穷举 non-shot hard negatives；不恢复盲推理、不改变 runtime 默认。

## NBA Games v2 PBP/media 配对复核（2026-08-10）

- [x] 固定远端 `choucsan/NBA_Games` revision `cf59e3a42413e3ab91f6fc0b1618f280df9a7024`，为 5 场已有本地原片补回 15 个结构化文件（`1,747,339` bytes）；manifest SHA `27f0d0c1f3fa11e36ef929a7878906ca2f67d000b00801432df80d6b6039413b`。
- [x] HOU–SAC Q1 300–600 秒片段完成 `.venv` RapidOCR 时钟清洗、PBP 近邻映射和 18 个标签隐藏窗口的 Codex 离线复核；14 个 confirmed、3 个 partial、1 个 not_visible，精确 release 帧留空；队列和人工 SHA 分别为 `2408bd165a3ed26bface189f454bf005d3dab7a65943299e5167a40ce139225a`、`b6d8ea329facb22eb2fa4b7eb8ed07bb75a9a9fcbbe3994f5b6f222c71e1d4f2`。
- [x] 删除没有本地配对原片的 4 场及可重建 HF cache，68 文件、`1,452,924` bytes；清理审计 SHA `61cd28d0c107631357dfc06b54e665874eb32d2e4fac348b257213f435467e54`。
- [x] 删除人工复核后不再引用的 2 张总览 contact sheet（`1,348,948` bytes）；清理审计 SHA `296bdfa9db7e61fa5ea32c882ae9eedee4b1758b6a412f83c64767c2ec8fe1ae`。
- [ ] 扩展到完整时钟/插播边界、精确 release、球—手—篮筐—结果和穷举 non-shot hard negatives；在完整因果真值与逐制作源 P/R ≥ 0.85 前，保持 `not_ready`、盲推理暂停，禁止进入训练/runtime/VLM 答案。

## v8–v10 native VLM probe and metadata follow-up (2026-08-10)

- [x] Build a label-hidden six-window native-video plan from sealed v8–v10 review rows under `.venv`.
- [x] Run EBQwen2.5-VL-3B-MLX-4bit with `strict_release_v2` under the resource guard; all six windows were
      predicted positive, yielding pooled P/R/F1=`0.50/1.00/0.6667` and per-production precision=`0.50`.
- [x] Keep the probe diagnostic-only (`runtime_consumable=false`, `training_consumable=false`,
      `promotion_eligible=false`); readiness remains `not_ready` and blind inference remains paused.
- [x] Re-check Basketball Events metadata only (1,631,871 transient bytes), delete the duplicate temporary copy,
      and retain the existing bounded canonical package.
- [x] Delete rebuildable repository caches after the probe (22 directories, 608 files, 7,543,909 bytes); formal
      assets and `.venv` remain untouched.
- [ ] Acquire exhaustive three-production causal labels and independent OOF rows before any VLM fusion or runtime change.

## v12 source-disjoint manual causal annotation (2026-08-10)

- [x] Build a deterministic six-window-per-source batch from the 1,922-window
      label-hidden queue, excluding every v2–v11 source-local anchor by 20 s.
- [x] Materialize and hash-bind the v12 original-frame sequences with the
      canonical `.venv`; keep all labels hidden from the plan and model inputs.
- [x] Visually inspect every v12 sequence and seal release/rim/outcome,
      `not_a_shot`, or `uncertain` decisions with evidence and confidence.
- [x] Verify and retain the sealed offline artifacts; delete only scratch sheets
      after retention; do not feed v12 into runtime, training, VLM answers or
      blind inference.
- [x] Run the 12 focused contract tests and delete their five rebuildable cache
      directories (21 files / `155,455` bytes); retain formal evidence and weights.
- [ ] Continue v13+ until the complete three-production queue is exhaustively
      labeled, then compute held-production P/R ≥ 0.85 before any promotion.

## v13 source-disjoint manual causal annotation (2026-08-10)

- [x] Build a deterministic 36-window batch from the label-hidden coverage queue,
      excluding every v2–v12 source-local anchor by 20 seconds and keeping
      early/middle/late quotas balanced across Hazen, Randolph and VTV.
- [x] Materialize 756 hash-bound original/JPEG frames under the canonical
      `.venv` and complete the Codex offline visual pass with labels hidden from
      the queue and from AGU runtime inputs.
- [x] Seal one VTV missed-shot chain plus 25 `not_a_shot` and 10
      `uncertain/unknown` decisions; build the v13 pilot and retention manifests.
- [x] Delete only the 36 unreferenced contact-sheet display intermediates
      (12,883,187 bytes) after retention; keep raw frames, source videos,
      review artifacts, weights, runtime and blind assets.
- [x] Merge v8–v13 to 186 windows / 3,906 retained frames and re-seal readiness;
      per-source P/R remains `not_computable`, `not_ready` and blind inference
      remains paused.
- [x] Pass the 48 focused contract tests and delete the four exact repository
      bytecode files regenerated by the checks (22,169 bytes); cleanup audit is
      `analysis_outputs/public_research/agu_v13_repository_cache_cleanup_2026-08-10.json`.
- [ ] Continue v14+ / exhaustive three-production causal and non-shot annotation,
      then produce held-production OOF predictions and enforce P/R ≥ 0.85 before
      any training, VLM fusion, runtime or blind-inference change.

## 三场分层不重叠 v11 held-out 离线复核（2026-08-10）

- [x] 排除 v2–v10 源内 20 秒锚点邻域，按每场 early/middle/late 各 4 窗生成 v11 标签隐藏批次；批次内部 SHA 为
  `1080b0f6038a733a8950b9e64108806950dcfa26e294e695886b69ea63c07673`。
- [x] 在 canonical `.venv` 物化 36 个窗口/756 张原始帧并逐帧复核；封存 34 个 `not_a_shot`、2 个
  `uncertain/unknown`、0 个 `shot`，review/manifest/plan/retention SHA 分别为
  `c048dca489b9b5edf22c2e47e66d2acc6ab6d5348e78eccbc8ac00929873b8ce`、
  `b3ddd6ed1f982b427cd87b27cf5957171d086546786c8e9078c147be786ed913`、
  `1e50f5f27a5bef62fd8a51256df5ac2ec0d35334d1d01810afcada9801d42496`、
  `209f3e6b8b75c593ae1c59526f9e0d72fe41630372130ecf1ce800e35b5193e3`。
- [x] v8–v11 合并为 132 窗口/2,772 帧；逐制作源 P/R 仍为 `not_computable`，不向训练、runtime、VLM 答案或盲推理提供标签。
- [x] 精确删除已完成复核且未被正式工件引用的 v11 contact-sheet：36 文件、13,044,216 bytes；raw frames、原片、权重和 blind assets 保留。
- [x] v11 契约测试 9 项通过；随后删除 6 个可重建缓存目录（18 文件、109,770 bytes），正式资产未触碰。
- [ ] 继续完成 1,922 个队列窗口的穷举亚秒因果、插播边界与 non-shot hard negatives，并按制作源留出通过 P/R ≥ 0.85 门禁。

## v10 DeepBall 辅助 transfer 与在线候选复核（2026-08-10）

- [x] 在 `.venv` 资源守护下对 v10 的 36 个标签隐藏窗口运行 DeepBall-Large transfer；资源守护未触发停止，输出保留 `oof_predictions=[]`。
- [x] 延迟读取 review 后做描述性评估：13 个 determinate 窗口在 0.75 cutoff 下 P/R/F1=`0.20/0.333/0.25`，23 个 `uncertain`。
- [x] 保持 `runtime_consumable=false`、`training_eligible=false`，不选阈值、不做融合、不晋级模型。
- [x] 审计 Stanford bball_attention、BasketEvent 和 legacy NBA PBP video dataset；3 个候选均不满足可核验 payload/license 与连续 5×5 亚秒因果 gate，未下载新 payload。记录：`analysis_outputs/public_research/agu_open_source_candidate_audit_2026-08-10.json`。
- [ ] 继续把三场原片队列推进到穷举事件、插播边界和 non-shot hard negatives，并按制作源留出做 P/R ≥ 0.85；在此之前不恢复盲推理或修改 runtime。

## 最终 targeted regression 缓存清理（2026-08-10）

- [x] 删除目标测试生成的 6 个可重建缓存目录：25 个文件、`377,033` bytes；清理审计 SHA
      `01f43ed88bed89d930ce825da45a902e54af1dfc69d9cdbb316e4be52ee13699`。
- [x] `.venv`、`.git`、`dataset`、原片、权重、正式证据与 blind assets 均未触碰。

## 三场分层不重叠 v10 held-out 离线因果复核（2026-08-10）

- [x] 以 v2–v9 锚点和 20 秒源内排除半径生成 v10 批次，三场各按 early/middle/late 4 个窗口配额；批次规范化 SHA
      `d8e5c98305c2c9cebbeaeea5a352f9ab1f26f071fd096d6edec7b91d3cb29d9e`，文件 SHA
      `4ace7ddc4ce6a73e6fc56535ea8a390ead0d67d93512886fc2f6358f75995a52`。
- [x] 在 `.venv` 下物化、逐帧复核并封存 36 个窗口/756 张哈希绑定帧；review/manifest/plan/retention SHA 为
      `f7e85fd0b152d49f763ffded203a566904018199f71f142e4bb6d60de2f10e29`、
      `690d88bf4c36faae6cb7e22fe5d132caad6d5d1f12a7b0d328104b34446b7b59`、
      `0f9318577032f6efecc55f026b5349aa2ea86f2e8b8653ca3d2fb041c8442a91`、
      `c5c22d613fe4ae0081ed53feb327890e62570d70ff1f2500114bb9ea4160bdc6`。
- [x] 保守封存 3 `shot`（1 made、2 missed）、10 `not_a_shot`、23 `uncertain`；与 v8/v9 合并 96 窗口/2,016 帧，
      仍是 pilot-only/non-exhaustive，不进入训练、runtime、VLM 答案或盲推理。
- [x] 删除封存后未引用的 v10 contact-sheet：36 文件、`12,562,615` bytes；清理 SHA
      `3509ab06f8e646f78e9143d5f7e582ae6409ca6033415f69e2b279133fffbd6c`。
- [x] 更新离线合并摘要（SHA `c547a849b6b031d4a950093de965a20942097ef2f9afeca7a466d86aa03aa2eb`）和 readiness 审计
      （SHA `6c3baeda895d4c662ebb43b286ee4ee83abaadfcc722c1fc6c5857d85c062695`）；`not_ready`、盲推理暂停。
- [ ] 继续覆盖队列剩余窗口，完成三场穷举亚秒事件、插播边界与 non-shot hard negatives，并通过制作源留出 P/R ≥ 0.85；
      在此之前不晋级 v10 标签、不改 runtime/默认模型/盲推理。

## v10 label-free VLM 资源探针（2026-08-10）

- [x] 从 v10 封存 review plan 派生不含人工标签的独立 VLM source plan；规范化 SHA
      `cce46b527953ac968e9d1504cdcac5e5ac645616cc40237dcefe42180ab513e9`，文件 SHA
      `be59a16d5109faa06294d8b5f6fb2563ddeeaac9afb545c8d6e231018a77a8e4`。
- [x] 在 `.venv` 资源守护下尝试 3 窗口 `EBQwen2.5-VL-3B-MLX-4bit` 前缀探针；连续三次可用内存 `<2 GiB`，峰值系统内存 `91.8%`，exit `75`，没有 prediction/evaluation 输出，不能作为模型性能结论。
- [x] 保留守护日志（文件 SHA `93b3d44ca1c7a96f95b9b60939229084d573884cd11493869c163cfec5ec361b`），删除未引用的 2,109-byte probe plan；清理审计 SHA `301d75c0822dd535ef07fc39aea4933f2fa4cd4c7eae9a0f13e525e812e8e261`。
- [ ] 待资源恢复后再重试 label-free VLM screen；在独立 OOF 与逐制作源 P/R ≥ 0.85 通过前，不修改 runtime/默认模型、不恢复盲推理。

## v7/v8 视觉中间物清理（2026-08-10）

- [x] 确认 review/manifest/retention 不引用后，精确删除 v7/v8 contact-sheet：36 个文件、`23,205,453` bytes；清理审计 SHA
      `98becaf97be22b24f4cdf71e8928d60116b6c30681850ac308d2fb6fb8a12b3b`。
- [x] v7/v8 raw frames、封存工件、原片、权重、runtime checkpoint 与 blind assets 均保留，未触碰正式证据。

## 可重建缓存清理（2026-08-10）

- [x] 回归测试后清理仓库内可重建 `__pycache__`、`.pytest_cache`、`.ruff_cache`、`.mypy_cache`（不含 `.venv`、`.git`、`dataset`）：19 个目录、797 个文件、`10,917,903` bytes；审计 SHA
      `c0c165b9335f3ac448e6c9cb25ed07504571c1db9f7db376d5399f2a0583a8f5`。
- [x] 正式证据、原片、权重、runtime、训练资产与 blind assets 均未触碰。
- [x] 清理后 readiness 审计 SHA 为 `ec757e011bfff51b6223f394f7fa8098acbf1e0d35a63ae3e00f90f23bbbda3b`，状态继续 `not_ready`、盲推理暂停。

## 三场分层不重叠 v9 held-out 离线因果复核（2026-08-10）

- [x] 扩展 held-out 批次契约，按每个制作源 early/middle/late 各 4 个窗口抽取，并以 20 秒源内半径排除 v2–v8 锚点；v9 批次规范化工件 SHA `78d83f204baa3c908d4bdd0d786c6c4028caf8e63cc221f9a272270c97e37857`，文件字节 SHA `51feb1f9ec6689d0def97a239b9a8886d0159d8481e13a0c4b66001f58b69d4e`。
- [x] 物化并封存 36 个窗口、756 张原始/JPEG 哈希绑定帧；review/manifest/plan/retention SHA 为 `1b00787d249b9da6b254d3022f4e9c1ac3ab36883789d1497a9598d5a1006890`、`b932f045cf0fa243337522f8cff5435f67a8996b3c003bf1a7f032043a094f5f`、`027e2a75b0f1b324038f27134d368480523c6f73e1941720f373a3dda1d70f10`、`8c4df9fe74b7c93318c16152722153c55edd1a79c0d8fc13f98e2aa90fdf1386`。
- [x] 逐帧保守标注 2 个可见但未命中的投篮链、30 个 `not_a_shot` 和 4 个 `uncertain`；v8+v9 累计 60 窗口/1,260 帧，仍为离线 pilot，不进入训练、runtime、VLM 答案或盲推理。
- [x] 正式封存后精确删除未引用的 v9 contact-sheet 目录：36 文件、`13,071,798` bytes；清理审计 `analysis_outputs/public_research/agu_v9_visual_sheet_cleanup_2026-08-10.json`，raw frames 与封存工件保留。
- [x] 合并 v8/v9 离线证据摘要（60 窗口/1,260 帧；3 shot、49 not_a_shot、8 uncertain）；摘要 SHA `781041510552da1a3caae00099cdb358b269e41a1a200a7b2eb5de2578745f0a`，逐制作源 P/R 距离因缺少 runtime/OOF 预测记为 `not_computable`。
- [x] readiness 指针已更新并复核，审计文件 SHA `cd36e68aeaa398d206ce0e6de6f48389eab23efda359b3760366eb7f4fbed26c`；状态仍 `not_ready`、盲推理暂停。
- [ ] 继续覆盖全队列 1,922 个窗口，完成穷举亚秒因果、插播边界和 non-shot hard negatives，并通过逐制作源 P/R ≥ 0.85 门禁。

## 三场源间隔离 v20 held-out 复核（2026-08-12）

- [x] 完成 36 个源间隔离窗口、756 张原始帧复核：3 `shot/unknown`、5 `uncertain/unknown`、28 `not_a_shot`；v8–v20 合计 438 窗口/9,198 帧。
- [x] 更新 readiness/source gate；仍为 19 candidates / 0 eligible、`not_ready`、blind inference 暂停，P/R 不可计算。
- [x] 删除接触表和项目级可重建 cache/bytecode（58 文件、17,928,229 bytes）；原片、raw frames、权重和正式工件保留。
- [ ] 继续完成三场全量亚秒球—手—篮筐—结果、插播边界与 non-shot hard negatives，并通过逐制作源 P/R ≥ 0.85 门禁。

## 2026-08-12 在线开源因果候选再审计

- [x] 完成 10 个候选的在线 metadata/API/revision/许可证与任务适配复核；剩余磁盘约 5.917 GiB。
- [x] 0 个候选具备可安全下载的连续因果训练载荷，0 bytes 新媒体下载；GCB 本地仍仅 6.2 MB 元数据，Henu/BASKET 等大载荷未拉取。
- [x] 审计工件 `analysis_outputs/public_research/agu_open_source_causal_candidate_reaudit_2026-08-12.json` 已写入 source gate/readiness；状态保持 `not_ready`、blind inference 暂停。
- [x] 回归后精确删除 19 个可重建项目/测试 cache；`.venv`、原片、正式工件、权重均未触碰；cleanup audit 内部 SHA `203f7fca89f03ba569ba9623bec31ccd144e3bd6ff2773ef492ef8692e22f7f5`。
- [ ] 继续三场全量亚秒球—手—篮筐—结果与 non-shot hard-negative 人工真值，完成后再按制作源留出计算 P/R ≥ 0.85。

## 三场不重叠 v8 held-out 离线因果复核（2026-08-10）

- [x] 新增 held-out 批次契约与测试，以 v2–v7 锚点和 20 秒排除半径生成三场各 8 个窗口；批次 `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_annotation_batch_v8.json`，SHA `8ce44f7749e40309e87685e4180c30bcc3a325ec1e364fc09f2e09b5843793c6`。
- [x] 以 canonical `.venv` 物化 24 个窗口、504 张原始/JPEG 哈希绑定帧，并封存 v8 review/manifest/plan/retention；标签为 1 `shot/made`、19 `not_a_shot`、4 `uncertain`，SHA 分别为 `7b1f0524f2e2d6e86d260cd4ab3616f9be1cfa5da809650c1e87cca038bad1c2`、`3a55d521a202ba07d1a019169d6030267c7be0c9f3d6c813c5329e8062167e8f`、`60ec3d0a9913032aa29cdf98fb7e8628c677c2c4043053b5d107e10af2437b08`、`918b14ae25b1b969581a43a600f0969fc5cea4f4ee4f1312ab97cd228775262b`。
- [ ] 继续覆盖剩余 1,922 个队列窗，完成三场穷举事件/插播边界/non-shot hard-negative 真值并按制作源留出做 P/R ≥ 0.85；v8 仍不进入训练、runtime、VLM 答案或盲推理。

## 三场全场覆盖队列与 v7 离线因果复核（2026-08-10）

- [x] 建立标签隐藏、原片 SHA 绑定的全场队列：`analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_annotation_queue_v1.json`，三场共 `1,922` 个 10 秒覆盖窗，SHA `8b3f45a6a895f05e8cc63e7c84b7011945681c39fbdb34ac85c1d8fdf6586022`。
- [x] 以 canonical `.venv` 稀疏 seek 抽取 12 个均衡窗口、492 张原始帧/JPEG 哈希绑定帧；未启动 AGU runtime，未把标签写入任何模型输入。
- [x] 封存 v7 pilot：1 个 `shot/missed`、7 个 `not_a_shot`、4 个 `uncertain/unknown`；目录 `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v7/`，plan/review/retention/manifest SHA 为 `eef5dec0ed0cb929c6493e52afb85bf2ab76a6bd87758989dbf300030c32eea9`、`842b38533d37209b48f5474513c542bf6c027404d9d04d0aa47b8cc6a2fbb460`、`45bf92fb013faee826c1aebe9b99ec2ed2abba8762b14ff48f17924d9fee81b8`、`48edb5aab03592ddfaf9beb7203acc7a043717d376c42228684382b2a420a383`。
- [ ] 继续完成队列中的穷举事件/插播边界/non-shot hard-negative 标注，再按制作源留出做逐场 P/R ≥ 0.85 门禁；v7 不改变 `not_ready`、盲推理暂停或 runtime 默认。
- [x] 复核后精确删除未被引用的 `/tmp/agu_v7_review.jo6ZgQ` 视觉草稿：18 个文件、`3,346,793` bytes；正式 v7 证据仍保留。

## 在线源扫查、外部校准与跨转播媒体清理（2026-08-10）

- [x] 审计 BARD、E-BARD、SportsMOT、SpaceJam、Basketball Events、NBA Games 与 legacy NBA PBP video dataset；没有候选同时满足连续 5×5 原片、亚秒因果标签、穷举 hard negatives、权利和可复现实物五项 gate，未下载新的 payload。扫查 SHA `8c9c46f42f5bd3a90c7b1905a25276b074109092d916f743670ceadcbfeb7dbd`，gate 为 18 候选/0 eligible，SHA `c234381b7806c048bb20c97af4ce1d191298c5f9260b6e3c496c39656b61ac30`。
- [x] 完成三份外部 DeepBall-Large 屏幕的 game-held 校准诊断：96 事件、独立阈值 `0.0`、校准 precision `0.177083`；冻结后目标 determinate precision `0.333333`、recall `1.0`；工件 SHA `7d45fb7d27938aa90c8962884c8dbde851420715f243f6a0503962c1a8651b7d`，`oof_predictions=[]`。
- [x] 在无活动占用且 manifest/screen/evaluation 已留存后，删除三份精确 allowlist 跨转播 MP4，共 `1,909,214,774` bytes；清理审计 SHA `d4c1f1907ebff932d1c3de28aced37e536f82a82c80a1e271bc0f9fa4166d4c4`。
- [ ] 继续完成三场穷举事件、插播边界、non-shot hard-negative 人工标注，并通过制作源留出逐场 P/R ≥ 0.85 门禁；在此之前不改 runtime、默认模型或盲推理。
- [x] 2026-08-12 v2 在线候选审计完成，0 bytes 新媒体下载；source gate/readiness 指针已更新，仍 `not_ready`、blind inference 暂停。
- [x] compact256 transfer-veto 32 窗口完成，16/32 保留、16/32 abstain，post-inference P/R=`0.625/0.625`，研究工件不进入 AGU runtime/训练。
- [x] v22 source-disjoint label-hidden 增量批次已建立（36 windows）；受控 smoke 完成（63 帧、1 not_a_shot、2 uncertain），完整批次待受控分段物化和人工复核。
- [x] v22 长 AV1 物化暂停后已删除 568 个可重建中间文件（196,006,492 bytes）；正式 plan、smoke raw frames 和封存工件保留。
- [x] v22 已改为每窗口单次 seek 后顺序解码并完成全量 36 windows/756 frames 人工复核；30 `not_a_shot`、6 `uncertain`，仍不进入训练/runtime/晋级；contact sheets/临时 smoke 副本已清理，正式 raw frames/manifest 保留。
- [x] v22 focused regression 16/16 通过；随后删除 25 个项目级可重建 cache/bytecode 文件（232,107 bytes），正式 raw frames、原片、权重、`.venv` 依赖与封存工件未触碰。
- [x] v23 已完成三源 source-disjoint 增量（各 5 个、共 15 windows/315 frames）：1 `shot/missed`、14 `not_a_shot`；封存 review/pilot/retention 与原始帧，仍不进入训练、runtime、VLM 默认答案或盲推理。
- [x] v23 contact sheet 已按精确路径删除 15 个文件、5,389,158 bytes；清理审计 `analysis_outputs/public_research/agu_v23_contact_sheet_cleanup_2026-08-12.json`，正式 raw frames、manifest、源视频、权重和 `.venv` 依赖保留。
- [x] v23 focused regression 16/16 通过；随后删除 133 个可重建 bytecode/cache 文件、3,002,360 bytes，清理审计 `analysis_outputs/public_research/agu_v23_regression_cache_cleanup_2026-08-12.json`，清理后 `.venv` bytecode 与项目测试 cache 均核实为空。
- [x] v23 后 readiness/source gate 指针已重封并校验：readiness 文件 SHA `4db4c19e626e5e90d68e8dbf48a4f57cff136e37c075b4a1471039daf0d4c394`（audit `14ff7a113d0c68563dc7652cbcd9d22ab6558131a4f634cfead5148e4dae2136`），source gate 文件 SHA `409876a321af973cc2c68e6b3837c896e706d4b5d5e2b3aef5cf403ac7cf7ed6`（audit `5e2e05a6c118e46419ada1c19af0d2cd32510406ac2873ad4c996293f03d807`）。
- [ ] 继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 non-shot hard negatives 与 game-held OOF；当前 v8–v23 为 525 windows/11,025 frames，P/R 仍 `not_computable`，source gate 19/0，readiness `not_ready`，盲推理暂停。

### 2026-08-13 在线候选 v3 / 融合重放

- [x] 完成 9 个官方在线候选的 metadata-first 筛选；无候选通过权利、连续因果标签、穷举 hard-negative 与磁盘门禁，未下载新媒体。工件：`analysis_outputs/public_research/agu_open_source_causal_candidate_audit_v3_2026-08-13.json`。
- [x] `.venv` Python 3.11.15 重放 32 行 `both_confirm`；预测 SHA 一致，RSS 峰值 `374,964,224` bytes，P/R=`0/0`，不晋级、不修改 runtime/default/blind。
- [x] 41 项聚焦契约回归通过；清理本轮可重建 pyc/pytest cache 和 `/tmp/agu_*` 探测副本，正式资产未触碰，清理审计已封存。
- [ ] 继续三场剩余全量窗口与 game-held OOF；在门禁闭合前保持 blind inference paused。

### 2026-08-13 v24_rv 离线人工复核

- [x] 记录 15 秒 source-disjoint 队列容量：Hazen `0`、Randolph `141`、VTV `482`；不放宽排除半径，不复用旧锚点邻域。
- [x] 完成 v24_rv 8 windows/168 frames（Randolph 4、VTV 4）标签隐藏复核：4 `not_a_shot`、1 `shot/made`、3 `uncertain/unknown`；sealed review、pilot、retention 与 raw-frame manifest 均保留并绑定哈希。
- [x] 复核后删除 16 张 contact sheet、`15,395,408` bytes；13 项回归通过，清理本轮 102 个 pyc、19 个空 cache 目录和 4 个 pytest cache 文件、`2,304,638` bytes。
- [ ] v24_rv 仍是 offline pilot，不进入 AGU 训练/runtime/VLM 答案/融合/晋级；继续全量因果真值、non-shot hard negatives、逐制作源 P/R 与 game-held OOF，盲推理保持暂停。

### 2026-08-13 v25_rv 离线标注增量

- [x] 在 15 秒 source-disjoint 半径下从 Randolph/VTV 各取 6 个标签隐藏窗口，完成 12 windows/252 frames 原片人工复核；封存 11 `not_a_shot`、1 `uncertain`，无新增可确认 shot。
- [x] v25_rv pilot-only/offline，不进入训练、runtime、VLM 默认答案、融合、晋级或 blind inference；保留 raw frames 与正式工件，删除已完整查看的 12 张 contact sheet（`5,617,098` bytes）。
- [x] v25_rv focused regression `13 passed`，峰值 RSS `93,798,400` bytes；无服务代码、模型、runtime 输入或 readiness 指针变更。
- [x] 回归后按精确清单删除 7 个项目级可重建 pyc/pytest cache 文件；`.venv` 依赖、raw frames、正式工件、原片和权重保留，清理审计 `agu_v25_rv_regression_cache_cleanup_2026-08-13.json`。
- [ ] 继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF；readiness 继续 `not_ready`、blind inference 继续暂停。

### 2026-08-13 基座—因果任务迁移证据

- [x] 对 v23/v24_rv/v25_rv 的 31 个确定窗口完成当前 `r2plus1d_v3` whole-frame proxy 审计；2 个 shot 全漏检，TP/FP/FN/TN=`0/0/2/29`，rank AUC=`0.465517`。
- [x] 审计工件、资源守护日志和 3 项测试已封存；诊断明确标记为 `runtime_consumable=false`、`training_consumable=false`，没有改变模型、runtime 或 readiness。
- [x] 删除本轮生成的 35 次项目级可重建 bytecode/pytest-cache 文件（370,712 bytes；32 个 distinct paths），保留 `.venv` 依赖、模型、原片、raw frames 与正式证据。
- [ ] 继续寻找/构建满足 rights + continuous causal + exhaustive hard-negative + game-held OOF 的训练输入；在证据闭合前不把 pilot 当训练真值。

### 2026-08-13 已取代 pilot raw-frame 清理

- [x] allowlist-only 删除 v2–v22 `raw_frames`：`12,021` files / `3,917,617,577` bytes；v23/v24_rv/v25_rv
  保护帧 `315/168/252` 未变化，源视频、正式证据、权重和 `.venv` 未触碰。
- [x] 审计 `analysis_outputs/public_research/agu_superseded_pilot_raw_frames_cleanup_2026-08-13.json`
  （内部 SHA `123e8952c469be5f272576df4d7e15f84e2b668ea037fe764f2eb74ca70ab353`，文件 SHA
  `746e89fddc28eac3bf7e33ef7b98721fec9afaa626246ec2f0b1477289939eee`）；安全回归 `2 passed`。
- [ ] 清理后继续建立真正 source-disjoint 的全量因果训练 bundle；当前 pilot 仍不可训练，readiness
  仍 `not_ready`、盲推理暂停。

### 2026-08-13 四场旧标注链当前基座迁移诊断

- [x] 374 windows / 4 games 的旧 SHA 绑定 bundle/labels 配对成功；使用当前 `r2plus1d_v3` 做 1-epoch
  `agu_v3`/layer4 game-held OOF，pooled P/R/F1=`0/0/0`，TP/FP/FN/TN=`0/0/173/201`，未晋级。
- [x] 资源守护最低可用内存约 `2.438 GiB`、CPU 峰值约 `80.9%`；metadata/guard 审计保留，未晋级临时
  checkpoint 删除 `125,393,565` bytes，清理审计已写入。
- [ ] 不能据此宣称“模型训练好了”；继续 rights + causal labels + exhaustive negatives + game-held OOF，
  readiness 仍 `not_ready`、blind inference 仍暂停。
## 2026-08-13 NBA Games 镜像审计与资源守护 VLM 试跑

- [x] 新增本地 NBA Games 镜像审计：189 场、81,355 条 PBP、0 bytes 视频，确认索引与 game 目录完全一致；工件 `agu_nba_games_local_mirror_audit_2026-08-13.json`。
- [x] 对 8 个 HOU–SAC 均匀原片窗口启动 qwen3-vl:2b raw-frame 试跑；可用内存连续三次低于 2 GiB 后安全停止（exit 75），删除 `/tmp/agu_hou_sac_vlm_resource8_v2_cache.json`，并封存守护/清理审计。
- [ ] 继续寻找有权利和完整 frame-causal truth 的连续 5×5 源；在 game-held OOF/P/R ≥ 0.85 前不恢复盲推理、不改 runtime/default VLM。

## 2026-08-13 冻结 shot-validity 头与官方 Kinetics 对照

- [x] 完成当前 AGU `r2plus1d_v3` 冻结 `fc`/uniform 两 epoch 四场 OOF：`3/4/170/197`，P/R=`0.4286/0.0173`；未晋级，临时权重删除并审计。
- [x] 完成训练侧 `anchor` 采样对照：374/374 条有证据锚点，但 OOF=`1/0/172/201`，P/R=`1.0/0.0058`；无跨场收益，临时权重删除并审计。
- [x] 完成官方 torchvision Kinetics-400 R(2+1)D-18 冻结 `fc` 对照：OOF=`0/0/173/201`，P/R/F1=`0/0/0`，守护最低可用内存约 2.46 GB、CPU 峰值 72.4%；不保留无用 120 MiB 权重。
- [x] 保留三份 metadata/guard/cleanup 审计和官方来源 SHA；没有修改 runtime、默认 VLM、readiness 或 blind inference。
- [x] 18 项 focused regression、Ruff、py_compile 通过；按精确路径删除 16 个项目级 `.pyc`/pytest/Ruff cache 文件（`109,437` bytes），正式工件与 `.venv` 保留。
- [ ] 当前阻塞仍是可复现的 rights-cleared 连续因果训练 bundle、穷举 non-shot hard negatives 和 game-held OOF；不能把这些迁移诊断当作“模型训练好了”。
- [x] 追加 BasketEvent/Qlean/Play by Play/MUVY/FineAction 官方核验；五者不满足组合门禁，下载 `0` bytes，继续转向本地 source-disjoint 因果标注 bundle。

### 2026-08-13 v26_rv 离线标注增量

- [x] 读取既有 review plan 的 592 个中心锚点，在 15 秒源内排除半径下从 Randolph/VTV 各抽取 early/middle/late 1 个 label-hidden 窗口，完成 6 windows/126 frames 的原片复核。
- [x] 封存 6 `not_a_shot/not_applicable`（1 个直播运动态势窗口保守归为非投篮），无新增可确认 shot；保留 raw frames、review/manifest/retention，删除 6 张 contact sheet（1,831,087 bytes），清理审计已封存。
- [x] held-out batch 新增显式 source subset 支持与 6 项契约测试；focused regression、Ruff、py_compile 通过。
- [x] 回归后删除本轮项目级 `.pyc`、pytest/Ruff cache，项目侧核验为 0；审计 `agu_v26_rv_regression_cache_cleanup-2026-08-13.json`，`.venv` 未触碰。
- [ ] v26_rv 仍不进入训练/runtime/VLM/融合/晋级或 readiness；继续全量亚秒因果、插播边界、穷举 non-shot hard negatives 与 game-held OOF。

### 2026-08-13 v27_rv source-disjoint 离线标注增量

- [x] 在已有 review-plan 中心外按 15 秒源内排除半径，从 Randolph/VTV 各 early/middle/late 抽取 12 个窗口，共 24 windows/504 张 hash-bound 原片帧；held-out batch SHA `e576ec63d473544542774e1bd8f417840abe5abbc8c95cecc91c166de2da6b79`，plan SHA `28546ef24aef9c80b9d4f938fdf5bf92ed2c20a989d5ee441522968547c0c239`。
- [x] 逐帧复核封存 22 `not_a_shot/not_applicable` 与 2 `uncertain/unknown`，没有新增可确认 shot；sealed SHA `803bf293f2ced37aea4ab31101327fdd0aee3fedaba856f6f4a30808de2854df`，pilot SHA `ff2a7022600d23b6657dac5c8c583534a5cd7dae6db64610e04351bd0b344785`。
- [x] v27_rv 明确为 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；不更新 readiness、不接入 VLM 默认答案、融合或 blind inference。raw frames 与正式工件保留。
- [x] `tests/test_vru_causal_review.py`、`tests/test_vru_causal_review_scripts.py`、`tests/test_vru_causal_pilot_manifest.py`、`tests/test_heldout_annotation_batch.py` 共 13 项通过；删除 24 张可重建 contact sheet（`11,708,095` bytes），清理审计 `analysis_outputs/public_research/agu_v27_contact_sheet_cleanup_2026-08-13.json`。
- [x] 回归后按精确 allowlist 删除项目级 `.pytest_cache`（4 files / `1,853` bytes），清理审计 `analysis_outputs/public_research/agu_v27_regression_cache_cleanup_2026-08-13.json`；`.venv`、raw frames 与正式工件保留。
- [ ] 继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF；readiness 仍 `not_ready`，blind inference 仍暂停。

### 2026-08-14 v33_rv source-disjoint 离线标注与在线源复筛

- [x] 在全部既有历史锚点外按 15 秒源内排除半径，从 Randolph/VTV 各抽取 12 个 label-hidden 窗口，共 24 windows/504 张 hash-bound 原片帧；held-out SHA `b578867cbbc2bd93ceb1de8e108a0919e026dfd5facb5dbbe7ca1df859aea356`，review plan SHA `228868b5acc22c23e98ecf0306c9ffa79744b3ee57227efedbbb35a2e14ab596`，raw-frame manifest SHA `6793727d672c5743d8b7c3407ff2c154b1efcf03913547a550b7cb746747f4c5`。
- [x] 原片逐帧复核封存 10 `not_a_shot/not_applicable`、14 `uncertain/unknown`、0 `shot`；sealed SHA `2382ad54801ad75212ba94afdb1a314ce121322f1582d7e45eea127359a4a586`，pilot SHA `ca321e3629edf1ee6fbec415c3502de7cf3a306d45dd9a51962689297e46764f`，retention SHA `b787f84211fab104479ffa236b3ab2fe4f0456b789fdeffa1c5705aae66ff2b2`。
- [x] v33_rv 保持 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；不接入 AGU 训练、runtime、VLM 默认答案、融合或 blind inference。canonical `.venv` causal-review regression `13 passed`。
- [x] 删除已查看 24 张 contact sheet（`12,523,486` bytes），审计 `analysis_outputs/public_research/agu_v33_contact_sheet_cleanup_2026-08-14.json`；测试后仅有的空 `.pytest_cache` 目录已移除，`.venv` 与正式工件未触碰。
- [x] 依据用户授权完成在线元数据复筛，BARD 保留既有本地辅助载荷，其余候选均未通过连续因果/权利/可控载荷门禁，新增下载 `0` bytes；审计 `analysis_outputs/public_research/agu_online_source_sweep_2026-08-14.json`，SHA `8a35ca5a5724f94d8ac979e7746731be6829a97e60d56cdd1e10910c44fabf8d`。
- [ ] v33_rv 仍不构成三场全量因果真值且没有新增可训练 shot 正例；继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF，readiness 仍 `not_ready`，blind inference 仍暂停。

### 2026-08-14 v34_rv residual source-disjoint 离线标注

- [x] 在 1,456 个历史源内排除锚点外抽取 Randolph 2 个、VTV 2 个窗口，共 4 windows/84 张 hash-bound 原片帧；held-out SHA `2a12bb73ae9ed072d43684b3612a55992fb63b1664f1d7daeb24ef05eb2c2d25`，review plan SHA `b3dbd70a6f776b5f9da1c178ab33e6672d72351b891a6c1f5cefb8c8c54fced4`，raw-frame manifest SHA `49e051773a91d84773ea78f7ef1d081b4ab65138f294715cec6c2dc5220b58b4`。
- [x] 原片逐帧复核封存 2 `not_a_shot/not_applicable`、2 `uncertain/unknown`、0 `shot`；sealed SHA `736a8f6748a1ab776af18d3a83e38bcbf5f4746e5d1ca313d03a97acc55885fa`，pilot SHA `58fb19a66a0b9930d621ca5da3f9ab6db2d8d6d624f868d5631b2ff345a991ed`，retention SHA `1014c9cd60a252e898e95ddf1f53a0a9d29369b05f772e2bb8a04cf786ff9682`。
- [x] v34_rv 保持 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；focused causal-review regression `13 passed`，不接入 VLM 默认答案、融合或 blind inference。
- [x] 删除 4 张已复核 contact sheet（`2,111,920` bytes），审计 `analysis_outputs/public_research/agu_v34_contact_sheet_cleanup_2026-08-14.json`；raw frames、正式工件、原片与 `.venv` 保留，项目侧缓存核验为 0。
- [ ] v34_rv 无新增可训练 shot 正例；Hazen 在 15 秒排除半径下耗尽，继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF，readiness 仍 `not_ready`，blind inference 仍暂停。

### 2026-08-14 v35_rv VTV residual source-disjoint 离线标注

- [x] 在所有已封存 review-plan/batch 锚点之外继续按 15 秒源内排除协议，从 VTV 选择 8 个 label-hidden 窗口；anchor manifest SHA `626bd0dd9be94f5e33543c4dc8cefd7a4eb3f671507eada89a55c48e0afbf50f`，held-out batch SHA `9369ffce6947d98e0c30b712b16f6fa3deae8b7499e55d27d9e19ed1587a2545`。
- [x] 物化并人工复核 168 张 hash-bound 原片帧，封存 7 `not_a_shot/not_applicable`、1 `uncertain/unknown`、0 `shot`；sealed SHA `b2dc36acd9eb2e48cedd19f29c3ecb1206bfe17b77315dc1fab337784d1133f8`，pilot SHA `b920d477ad2a7f444b4b226832b031eed353df8dde1868a072d77e4daa8f8fa9`，retention SHA `e0ba7c3d81dedc42f2cf853e2df18a3938cdb4e303bf9aca14fb3ec6e7e5e406`。
- [x] v35_rv 保持离线 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；focused causal-review regression 13 passed，不接入 AGU runtime、VLM 默认答案、融合或 blind inference。
- [x] 删除 8 张已查看 contact sheet（`3,507,319` bytes），审计 `analysis_outputs/public_research/agu_v35_contact_sheet_cleanup_2026-08-14.json`；回归后删除 20 个项目级 bytecode/cache 文件（`168,942` bytes），审计 `analysis_outputs/public_research/agu_v35_regression_cache_cleanup_2026-08-14.json`；raw frames、正式工件、原片与 `.venv` 保留。
- [ ] v35_rv 没有新增可训练 shot 正例；继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF，readiness 仍 `not_ready`，blind inference 仍暂停。

### 2026-08-14 v36_rv VTV residual source-disjoint 离线标注

- [x] 在 v35_rv 的全部源内锚点外按 15 秒排除半径，按 early/middle/late 各 8 个窗口抽取 VTV 共 24 个 label-hidden 窗口；物化并复核 504 张 ±2 秒原片帧。held-out SHA `589908cf67f0ef6d44c8b77e206f5703bf43df0cff8b131c7a1d8dfc82680724`，anchor SHA `4b302c0ce70cafd191911b8ac7261f4b1a03fdffb6d8ba1f4bc288d4470d0be9`。
- [x] fail-closed 封存 23 `not_a_shot/not_applicable`、1 `uncertain/unknown`、0 `shot`；plan `bb6d325f48c4eaf4832678788664fbabf73c7f5d793cc73aaa8f1c259d3a162d`，sealed `f9602163a95585263a362b16c03b5fceb9c6b1bb33976155979733741a2e1a09`，pilot `ef3765155ecf6f055da70ae61457feffb6eeacbe24e5c9038ba75d75d9d597ee`，retention `04dfc1d5ff499405d78bf1c18f1419dd16bde83997ac1ff984064fc76d5d5215`。
- [x] 保持离线 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；focused causal-review regression `13 passed`，不接入 AGU runtime、VLM 默认答案、融合或 blind inference。Hazen/Randolph 余量为 0，VTV 仍有 204 个队列窗口。
- [x] 删除已复核的 24 张 contact sheet 和 5 个 montage 派生文件，共 29 文件/15,140,690 bytes；审计 `analysis_outputs/public_research/agu_v36_contact_sheet_cleanup_2026-08-14.json`，raw frames/正式工件、原片与 `.venv` 保留；覆盖审计 `agu_v36_rv_annotation_coverage_2026-08-14.json`。回归后另删除 20 个项目级 cache/bytecode 文件（171,461 bytes），审计 `agu_v36_regression_cache_cleanup_2026-08-14.json`。
- [ ] v36_rv 没有新增可训练 shot 正例；继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF，readiness 仍 `not_ready`，blind inference 仍暂停。

### 2026-08-14 v37_rv VTV residual source-disjoint 离线标注

- [x] 在 v36_rv 全部源内锚点外按 15 秒排除半径，从 VTV early/middle/late 各抽取 8 个 label-hidden 窗口，共 24 windows/504 张 ±2 秒 hash-bound 原片帧；held-out batch SHA `5237edd2383ce7bafc1f5fd6ed3ea0083b0636720fe71702ad18308a4379bd0d`，anchor SHA `c2d708a9441b1e10af2e23f3d697a6c44a8eeca6eedb80720a2d4fa5192eeee3`，review-plan SHA `82866573ea39b721a60ad2c79f0a53646398006098bc8902e3a616c25c3f225f`。
- [x] 原片逐帧 fail-closed 封存 22 `not_a_shot/not_applicable`、2 `uncertain/unknown`、0 `shot`；sealed SHA `6251db4a641c99d9ce64e52efb82ee63f376ac434774f5dd275e7e07ac155d37`，pilot SHA `6b3b814c2986fabfa80121f733b50e42523f0206f7e7bff75d39cf59daed71dc`，retention SHA `22aa075f4759d2ce72f2319f2669da84cff2eaa01ec3a3bee53692fea73ff3de`，raw-frame SHA `0a809de26bdd641087c98bf5e25374a7d1359a307db748ec47e7384f43e3a210`。
- [x] v37_rv 保持 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；不接入 AGU 训练、runtime、VLM 默认答案、融合或 blind inference；canonical `.venv` causal-review regression `13 passed`。
- [x] 覆盖审计 `agu_v37_rv_annotation_coverage_2026-08-14.json`（SHA `70504ae97479973a91d4599f45aa1e169598f7f88e1faa407644f3dc0bf3bfdb`）显示 Hazen/Randolph 余量 0、VTV 余量 `157`；删除 24 张 contact sheet 与 5 个 montage 派生文件，共 29 文件/`14,742,074` bytes，审计 `agu_v37_contact_sheet_cleanup_2026-08-14.json`（audit SHA `a7e827042c5c5bcb3a871c0ff992b2228e6dd5a010d11c402dc9232831db6406`）；回归后另删除 19 个项目级 cache/bytecode 文件（`162,651` bytes），审计 `agu_v37_regression_cache_cleanup_2026-08-14.json`（文件 SHA `1211069b3709be04d785f56cfb09d9db2208ebd2eb19eb411a8f56f2eae788b1`）。
- [ ] v37_rv 没有新增可训练 shot 正例；继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF，readiness 仍 `not_ready`，blind inference 仍暂停。

### 2026-08-14 v38_rv VTV residual source-disjoint 离线标注

- [x] 在 v37_rv 全部源内锚点外按 15 秒排除半径，从 VTV early/middle/late 各抽取 8 个 label-hidden 窗口，共 24 windows/504 张 ±2 秒 hash-bound 原片帧；held-out batch SHA `3f139af3c655635d4454d6592a8969680b558ae7db2cade5db5ef718e824dcac`，anchor SHA `13c6eefeb6d50eaca318c1835610e86e6b87ac4b7f283a2b808050ee91d06f17`，review-plan SHA `8998d037d156b92a59178b2cc80b55d6e927dbfe21f92f6b5d45fd3807cc674c`。
- [x] 原片逐帧人工复核并 fail-closed 封存 23 `not_a_shot/not_applicable`、1 `shot/unknown`（VTV 9362s：持球—脱手—向篮筐运动—篮筐/篮网平面可见，但 made/missed 不可安全解析）；sealed SHA `e70a729df42bf052dfc628a7aeed6a08fa4fa62fd90da19ef7a1573c52a1e100`，pilot SHA `a3c15f0daa15639aaa16dcd1db56bc8da4ecc7a617a3dd68c6a2eefdbdd62783`，retention SHA `df5ba13a93a6bb9904b80c8c171f77b34607d4368618a10481e3a3895c9cd99e`，raw-frame SHA `78a01c24e8435af4adb4cc848133a02a54b4d7e19b14a2bb487ec23d5e322062`。
- [x] v38_rv 保持 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；focused causal-review regression `13 passed`，不接入 AGU 训练、runtime、VLM 默认答案、融合或 blind inference。
- [x] 覆盖审计 `agu_v38_rv_annotation_coverage_2026-08-14.json`（SHA `ccb6b434fda37768b04eed3d6a281562f2ce73d5c9fc1ecbb4b3f84be7deb0de`）显示 Hazen/Randolph 余量 0、VTV 余量 `115`（early 39 / middle 29 / late 47）；删除 24 张 contact sheet 与 5 个 montage 派生文件，共 29 文件/`12,958,436` bytes，审计 `agu_v38_contact_sheet_cleanup_2026-08-14.json`（audit SHA `22b9b8dacad7a66f6641062ae3a8224b287470cfaaa63a4c3be3c1e180d77c76`）；回归后另删除 19 个项目级 cache/bytecode 文件（`162,651` bytes），审计 `agu_v38_regression_cache_cleanup_2026-08-14.json`（文件 SHA `53655e4e1da0d9a76ede69983dd4640f1f4ef6e1bcffa0b5f0782ae1a7559598`）。
- [ ] v38_rv 没有新增可训练 made/missed shot 正例；继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF，readiness `not_ready`、blind inference 继续暂停。

### 2026-08-15 v39_rv VTV residual source-disjoint 离线标注

- [x] 在 v38_rv 全部源内锚点外按 15 秒排除半径，从 VTV early/middle/late 各抽取 8 个 label-hidden 窗口，共 24 windows/504 张 ±2 秒 hash-bound 原片帧；held-out batch SHA `00aff1cf36b78d2088a652e9ba2a6eec6e82964a47c257dabe4d9a36fd5c7f30`，anchor SHA `2051f01bd9fb2617735c8acb6549e87d97ce9a5f1d0e52a4c363f7cfcd331a70`，review-plan SHA `bb39767f72b81195ec616b9b29b30beffce7dcef5ec9a71131c8c6622c654c08`。
- [x] 原片逐帧 fail-closed 封存 24 `not_a_shot/not_applicable`、0 `uncertain/unknown`、0 `shot`；sealed SHA `55a8a603947ab5bca1d15fcb3181b508923934ec96fba28c2f23f5b4894aa89d`，pilot SHA `5cd734320330a2942d2a5206b8f96fa0bb7bcfd0fb5b8ee47f369ee5076789fc`，retention SHA `4fe24cdb3131c5ba917259bd98c0f4870e0f0f738f4cfc053b2c7d4b3f6ebabf`。
- [x] v39_rv 保持 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；focused causal-review regression `13 passed`，不接入 AGU 训练、runtime、VLM 默认答案、融合或 blind inference。
- [x] 覆盖审计 `agu_v39_rv_annotation_coverage_2026-08-15.json`（audit SHA `4906bc93492625de2d2ac88e2d95310ce741bb94abcb51017f54f96d6279ca36`）显示 Hazen/Randolph 余量 0、VTV 余量 `80`（early 25 / middle 20 / late 35）；删除 24 张 contact sheet 与 5 个 montage 派生文件，共 29 文件/`15,334,391` bytes，审计 `agu_v39_contact_sheet_cleanup_2026-08-15.json`（audit SHA `cb29db3ea5d11bb1bdb5d09a59ffb91fa57626dd658a244ad081079e8291d9bf`）；回归后另删除 19 个项目级 cache/bytecode 文件（`162,651` bytes），审计 `agu_v39_regression_cache_cleanup_2026-08-15.json`（文件 SHA `7b0d93fcde223e8d5ef5f82de517eda9d70a6cdff6937ce9616e451ca032a3ee`）。
- [ ] v39_rv 没有新增可训练 made/missed shot 正例；继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF，readiness `not_ready`、blind inference 继续暂停。

### 2026-08-15 v40_rv VTV residual source-disjoint 离线标注

- [x] 在 v39_rv 的全部源内锚点外按 15 秒排除半径，按 early/middle/late 各 8 个窗口抽取 VTV 共 24 个 label-hidden 窗口；物化并复核 504 张 ±2 秒原片帧。held-out SHA `9b149844f4ed3848110b99c693d80e6a8ee5545285d56407b44daa32a187989b`，anchor SHA `c9d1646f142550f95bbb148162070127d14f7eed517b4673420cdc1d5c45fcd5`，review-plan SHA `19261ec72b02f8f19118a36a35b74e281439125f045baa275c9230009944bb82`。
- [x] fail-closed 封存 22 `not_a_shot/not_applicable`、2 `uncertain/unknown`、0 `shot`；1332s 为近篮动作但 release/rim 结果链不完整，2362s 从篮筐接触镜头开始且前置 release 不可见。sealed `fdf50e0…`，pilot `3a5c2f9…`，retention `bbdfcddc…`，raw-frame `925772ab…`。
- [x] 保持离线 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；canonical `.venv` causal-review regression 为 `74 passed`（13 个聚焦文件），不接入 AGU runtime、VLM 默认答案、融合或 blind inference。覆盖审计显示 Hazen/Randolph 余量 0、VTV 余量 `47`（early 16 / middle 9 / late 22），记录于 `analysis_outputs/public_research/agu_v40_rv_annotation_coverage_2026-08-15.json`（audit SHA `9406785b…`）。
- [x] 删除已复核的 24 张 contact sheet 和 5 个 montage/manifest 派生文件，共 29 文件/`15,682,641` bytes；审计 `agu_v40_contact_sheet_cleanup_2026-08-15.json`（audit SHA `abdb01fd…`）。回归后另删除 67 个项目级可重建 cache/bytecode 文件、`1,250,894` bytes，审计 `agu_v40_regression_cache_cleanup_2026-08-15.json`（audit SHA `3b37f486…`）；raw frames/正式工件、原片与 `.venv` 保留。
- [ ] v40_rv 没有新增可训练 made/missed shot 正例；继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF，readiness 仍 `not_ready`，blind inference 仍暂停。

### 2026-08-15 v41_rv VTV residual source-disjoint 离线标注

- [x] 在 v40_rv 的全部源内锚点外按 15 秒排除半径，一次性抽取 VTV 剩余 47 个 label-hidden 窗口（early 16 / middle 9 / late 22）；物化并复核 987 张 ±2 秒原片帧。batch SHA `55c43c8…`，anchor `ea67bc2…`，review-plan `845991c…`，raw-frame `8c61c23…`。
- [x] fail-closed 封存 41 `not_a_shot/not_applicable`、6 `uncertain/unknown`、0 `shot`；6 个 uncertain 均为 replay/近篮动作但 release boundary 或完整 rim/outcome 链不可验证。sealed `836d8b3…`，pilot `4ff46cf…`，retention `87f5a50…`。
- [x] 保持离线 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；canonical `.venv` causal-review regression 为 `74 passed`（13 个聚焦文件），不接入 AGU runtime、VLM 默认答案、融合或 blind inference。coverage audit `agu_v41_rv_annotation_coverage_2026-08-15.json` 显示 Hazen/Randolph/VTV 余量均为 0。
- [x] 删除已复核的 47 张 contact sheet 和 8 个 montage/manifest 派生文件，共 56 文件/`25,321,919` bytes；回归后另删除 67 个项目级可重建 cache/bytecode 文件、`1,250,894` bytes；审计分别为 `agu_v41_contact_sheet_cleanup_2026-08-15.json`、`agu_v41_regression_cache_cleanup_2026-08-15.json`，raw frames/正式工件、原片与 `.venv` 保留。
- [ ] v41_rv 没有新增可训练 made/missed shot 正例；继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF，readiness 仍 `not_ready`，blind inference 仍暂停。

### 2026-08-14 v22 smoke 派生物清理

- [x] 完整 v22 批次已替代三窗口 smoke 原始帧后，按精确 allowlist 删除 63 个 smoke raw-frame 派生文件（`21,012,931` bytes）；保留 smoke 的所有 metadata/plan/review/sealed/pilot/retention provenance，审计 `analysis_outputs/public_research/agu_v22_smoke_derivative_cleanup_2026-08-14.json`。
- [x] `.venv`、三场原片、正式 v22 raw frames、EBQwen、runtime/readiness 均未触碰。

### 2026-08-14 v32_rv source-disjoint 离线标注与门禁复核

- [x] 在全部既有中心外按 15 秒源内排除半径，从 Randolph/VTV 各 early/middle/late 抽取 12 个 label-hidden 窗口，共 24 windows/504 张 hash-bound 原片帧；held-out artifact SHA `2205286c5364a8507025dce334111b2fcf9c6bdea6569b12095fdc84efdd8c6a`，review plan SHA `daad00c9a4275394372ea9eff91d502adba1e5f15fc98c7d91487df28dd16b2d`，raw-frame manifest SHA `efc32cc4712f501702d2b013b20dc24b14e4a8f105cd060272198dd9c9bc546a`。
- [x] 原片逐帧复核封存 12 `not_a_shot/not_applicable`、11 `uncertain/unknown`、1 `shot/unknown`（VTV 4842s，罚球出手—篮筐接触链可见但不能判定命中/未中）；sealed SHA `eb29e90a03b826d8ae45ff3de2c3a637a17db39d24e66736f7d79b6050807165`，pilot SHA `f7a73fc3800fc6f3681daeb780b43c106cc057afb17055cc97b07692c031f413`，retention SHA `7ad3f19bbd5a11d114d930eacb14eddca78a84f37100d5c4ba947f0ac9a6df0c`。
- [x] v32_rv 保持 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；不接入 VLM 默认答案、融合或 blind inference。canonical `.venv` focused causal-review regression `13 passed`。
- [x] 删除已查看 24 张 contact sheet（`12,182,043` bytes），审计 `analysis_outputs/public_research/agu_v32_contact_sheet_cleanup_2026-08-14.json`；13 项回归后再删除 4 个项目级 pytest cache 文件（`1,853` bytes），审计 `analysis_outputs/public_research/agu_v32_rv_regression_cache_cleanup_2026-08-14.json`；504 张 raw frames、review spec/plan/decisions/sealed、pilot/retention manifests 与原片保留，`.venv` 未触碰。
- [x] 既有诊断门禁复核仍未通过：当前基座 31 个确定窗口 TP/FP/FN/TN=`0/0/2/29`、P/R/F1=`0/0/0`、rank AUC=`0.465517`；独立 VLM transfer post-inference P/R=`0.625/0.625`、不可晋级。未启动无新增真值支撑的重训。
- [ ] 继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF；readiness 仍 `not_ready`，blind inference 仍暂停。

### 2026-08-13 v28_rv source-disjoint 离线标注增量

- [x] 在前批全部中心外按 15 秒源内排除半径，从 Randolph/VTV 各抽取 12 个 label-hidden 窗口，共 24 windows/504 张 hash-bound 原片帧；held-out batch SHA `495c7b1e06c2178d4a37869a364f364a7af198ba3111e8b94f525b54f74a8693`，plan SHA `b3693e4d0e81cc9dadfb269270c012d3deae1d363168cae083cc66a45a25c1e5`。
- [x] 逐帧复核封存 20 `not_a_shot/not_applicable` 与 4 `uncertain/unknown`，没有确认 shot；sealed SHA `52a3147b0c963f05d446dc43b8da01dc75c6ec7694517d537fb9800204d35415`，pilot SHA `cd9f636d800c6d8334a60a70c0ae43383142589f7c290f9a002649943f8f48c9`，retention SHA `fbcf446de61d320cc3b4ed719a6861a96fde2b9b3864201ec9d850f225d30d35`。
- [x] v28_rv 保持 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；不接入 VLM 默认答案、融合或 blind inference，raw frames 与正式工件保留。
- [x] 四组 causal-review contract tests 共 `13 passed`；删除 24 张可重建接触表（`11,613,466` bytes），审计 `analysis_outputs/public_research/agu_v28_contact_sheet_cleanup_2026-08-13.json`。
- [x] 回归后按精确 allowlist 删除项目级 `.pytest_cache`（4 files / `1,853` bytes），审计 `analysis_outputs/public_research/agu_v28_regression_cache_cleanup_2026-08-13.json`；`.venv` 未触碰。
- [ ] 继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF；readiness 仍 `not_ready`，blind inference 仍暂停。
### 2026-08-13 v29_rv source-disjoint 离线标注增量

- [x] 在前批全部中心外按 15 秒源内排除半径，从 Randolph/VTV 各 early/middle/late 抽取 12 个 label-hidden 窗口，共 24 windows/504 张 hash-bound 原片帧；held-out batch SHA `63ab792a00a3a90faea9e0265b750fb90a8f3f9a8ef2914b6e3371e38740038c`，review plan SHA `7734946f98ded098b53a80dd8da83b0a8040ed8b4cd64cca6a602e860473ae21`。
- [x] 离线逐帧复核封存 19 `not_a_shot/not_applicable`、5 `uncertain/unknown`，没有确认 shot；sealed SHA `e9f8db85b8846a92b77982af7ff532cff7009b651ca61ba8169e5bee11b67e90`，pilot SHA `2ab607117fb6cf521210ac30a1e5cbfcc37c583d00c1c36fff0dfaedd785274f`，retention SHA `bf8e0f344ae8f0fba724c1eabf265f719ac895364d3a9c817dd198daad70be73`。
- [x] v29_rv 保持 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；不接入 VLM 默认答案、融合或 blind inference，raw frames 与正式工件保留。
- [x] canonical `.venv` focused regression 共 `13 passed`；删除 24 张可重建 contact sheet（`11,854,476` bytes），审计 `analysis_outputs/public_research/agu_v29_contact_sheet_cleanup_2026-08-13.json`；删除 9 个项目级回归缓存/bytecode 文件（`39,954` bytes），审计 `analysis_outputs/public_research/agu_v29_regression_cache_cleanup_2026-08-13.json`；`.venv` 未触碰。
- [ ] v29_rv 仍不构成三场全量因果真值；继续插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF，readiness `not_ready`、blind inference 继续暂停。

## 2026-08-13 rejected auxiliary payload cleanup

- [x] 删除已拒绝且不被 AGU 使用的 Infactory/APIDIS/UVY 像素与派生载荷，共 `19,212` files / `1,667,793,510` bytes。
- [x] 保留 manifest、许可证、central directory、正式审计、代码/测试、Wikimedia 原片、两套 EBQwen 和 canonical `.venv`；合并审计 `analysis_outputs/public_research/agu_rejected_auxiliary_payload_cleanup_summary_2026-08-13.json`。
- [x] 回归生成的项目缓存已清理并封存 `analysis_outputs/public_research/agu_post_cleanup_regression_cache_cleanup_2026-08-13.json`；`.venv` 未触碰。
- [ ] 不将此次磁盘清理解释为模型已训练好；readiness 仍 `not_ready`，blind inference 仍暂停，继续寻找/构建完整跨比赛因果训练 bundle。

### 2026-08-13 v30_rv source-disjoint 离线标注增量

- [x] 在前批全部中心外按 15 秒源内排除半径，从 Randolph/VTV 各 early/middle/late 抽取 12 个 label-hidden 窗口，共 24 windows/504 张 hash-bound 原片帧；held-out batch artifact SHA `3cc3c05c1e9eb00e3f16ec7b944e4485f2b709fbc83c982c6b4f142a704cc471`，plan SHA `fd89fc7faff48bd0bb2cc8283083b04d3c74834259f8988620259d792830b8af`。
- [x] 逐帧原片复核封存 15 `not_a_shot/not_applicable`、9 `uncertain/unknown`，没有确认 shot；sealed SHA `84de8da98d54700f7f7bf62e8e6af2f543d780623d2441db7151e64be39e79ee`，pilot SHA `7d65343c39f6647fb4ecb977674afe5c7e6904b18c141b42d6dde573ed39386b`，retention SHA `3aa99dc0e020895d6663530dd95bdf4774e6b4142fb119e081c2b487badc860c`。
- [x] v30_rv 保持 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；不接入 VLM 默认答案、融合或 blind inference；504 张 raw frames 与正式工件保留。
- [x] canonical `.venv` causal-review regression `13 passed`；精确 allowlist 删除 24 张可重建 contact sheet（`11,640,731` bytes）和 10 个项目缓存/bytecode 文件（`61,430` bytes），审计分别为 `agu_v30_contact_sheet_cleanup_2026-08-13.json` 与 `agu_v30_regression_cache_cleanup_2026-08-13.json`；`.venv` 未触碰。
- [ ] v30_rv 仍不构成三场全量因果真值；外部候选复筛新增下载 `0` bytes，继续插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF；readiness `not_ready`、blind inference 继续暂停。

### 2026-08-14 v31_rv source-disjoint 离线标注增量

- [x] 在前批全部中心外按 15 秒源内排除半径，从 Randolph/VTV 各抽取 12 个 label-hidden 窗口，共 24 windows/504 张 hash-bound 原片帧；held-out batch SHA `12ca8d6c93114caa8dc2e97109a48ef7e551fb4c3a22e3a7dcf45f0852154c54`，plan SHA `c9b3eb5d5ac3f7f54b9da380db06dc0fd104046c461a1cd6a1f459cf463a529c`。
- [x] 原片逐帧复核封存 14 `not_a_shot/not_applicable`、9 `uncertain/unknown`、1 `shot/unknown`（VTV 872s，释放—篮筐接触链可见但不能判定命中/未中）；sealed SHA `9b5da3e7e4c1f20f96d129819667b3c6ce7c708f3c55a3092ccd36abf5b68a29`，pilot SHA `da6e65211b5d4e4041094bf765c089d325bc60a46504f07a46e2dbb547f2fed0`，retention SHA `f81875637b08ec8c4c8c5254acb944a9a496cd90fb2cfa730b88db0282db0101`。
- [x] v31_rv 明确为 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；不接入 VLM 默认答案、融合或 blind inference。focused causal-review regression 13 项通过。
- [x] 删除已查看的 24 张 contact sheet（`11,212,920` bytes）和回归生成的 10 个项目缓存/bytecode 文件（`61,430` bytes）；审计 `analysis_outputs/public_research/agu_v31_contact_sheet_cleanup_2026-08-14.json`、`analysis_outputs/public_research/agu_v31_regression_cache_cleanup_2026-08-14.json`，`.venv` 未触碰。
- [ ] 继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF；readiness 仍 `not_ready`，blind inference 仍暂停。
### 2026-08-15 nested outer-game-held 融合证据完整性 v2

- [x] RED：新增 outer-held label invariance、inner/outer group exclusion、三元键、plan/manifest 绑定和
  legacy-v1 fail-closed 回归；先确认测试因 v2 未实现而失败。
- [x] GREEN：实现 nested outer-game-held v2 screen/verifier；inner game-held 只在 outer-train 中选择
  variant/threshold，outer-held 标签只用于最终评估。
- [x] 让 frozen VLM auxiliary fusion 只接受 plan-bound、manifest-bound、hash-bound v2 工件；缺证据 abstain。
- [x] 在资源守护下对现有 511-window batch4 embeddings 生成并评估真实 v2 artifact；不晋级未过 0.85 的结果。
- [x] focused regression、Ruff、哈希、磁盘/进程核验通过后同步 docs、task board 与 LLM Wiki，清理可重建缓存。

## 2026-08-15 causal closure 后续

- [x] 完成 24 个 8 秒 causal-closure 窗口的 labels-hidden 原片复核与 SHA 封存。
- [x] 完成 nested outer-game-held v2、外部 receipt、legacy nonpromotable 与 destructive-output alias 防护。
- [x] 运行融合/评估/closure/CLI 联合回归：`313 passed`，focused Ruff 通过；verifier 语义重验与六个工件 CLI 的 alias/atomic 回归均覆盖。
- [x] 将新封存的 14 shot / 8 non-shot / 2 uncertain 只通过显式、SHA-bound 的 training manifest 纳入下一轮离线实验；unknown 不作为 made/missed 真值。
- [ ] 扩大 rights-cleared、连续比赛、穷举 negatives 的独立来源；在至少三场 source-disjoint/game-held 上逐场 precision/recall 均达到 0.85 前，保持 readiness `not_ready` 和 blind inference paused。
- [ ] 后续正式评估必须同时冻结 prediction receipt、plan、training manifest 与 auxiliary receipt；不得用运行后自封 SHA 替代外部冻结凭据。
- [x] 将 generic independent-shot VLM v1 固定为 `open_world_structural_only`、`legacy_provenance_only` 且永久不可晋级；开放 schema 的已知字段卫生检查不得再表述为 label-free/Codex-independent 证明。正式晋级须使用另行定义的 closed-schema contract。

## 2026-08-15 TASK-0254 closure training bridge

- [x] RED：为 closure 训练导出器增加端到端契约测试，覆盖全链 SHA、原片、帧清单、
  标签映射、不确定排除、禁止 outcome/release/rim 泄漏和安全原子写入。
- [x] GREEN：导出 3 份 label-hidden candidate geometry bundle + 3 份 shot-validity labels + 1 份
  `agu.vru-causal-shot-validity-training-export.v1`，严格得到 22 行（14+/8-）、排除 2 uncertain。
- [x] 封存新 training manifest，确认与 acceptance raw video SHA 无交集；显式禁止 ExtraTrees，
  仅允许 scene/video representation screen。
- [x] 用本地 MViT checkpoint 在资源守护下提取 22 行 embedding，运行 nested 三源
  game-held 诊断；记录 pooled/逐源 P/R 与守护峰值，始终 formal false。
- [x] 完成 focused/broader regression、Ruff/diff check、真实工件验证、fresh review、docs/task board/
  llm-wiki post-hook 和精确缓存清理。
- [x] 若逐源 P/R 任一低于 0.85，保持 readiness `not_ready` 与 blind inference paused；即使诊断达标，
  仍需独立、非 prior-stratified 的连续全场真值才能正式晋级。

## 2026-08-15 TASK-0255 Swin3D-T diagnostic

- [x] 在查看新结果前冻结 Swin-only 与 MViT+Swin 两个诊断及全部超参/非晋级边界。
- [x] 用本地验签 Swin3D-T checkpoint 在资源守护下生成 22 行 embedding。
- [x] 运行并验签两个 nested source-held probe，记录指标、资源日志与 artifact/file SHA。
- [x] 完成 focused/full regression、Ruff、进程/磁盘检查和文档/Wiki 归档；未更新 runtime/readiness。

## 2026-08-16 TASK-0256 HCTV–Harwood fourth-game development expansion

- [x] 验证官方许可/大小/时长后，以可续传方式下载并 SHA 封存 2026-01-22 HCTV–Harwood 全场。
- [x] RED→GREEN 实现并重放 24 窗、三时间桶、无标签/无模型分数的 deterministic selection。
- [x] 物化 1,536 张 hash-bound 原片帧并完成 labels-hidden 人工复核与严格封存：3 shot、20
  not-a-shot、1 uncertain。
- [x] 保持三源 v1 不变，将 23 条确定标签的四比赛 export/probe 迁入 TASK-0257；本任务未训练、未晋级。
- [x] 完成严格工件重验、fresh review、回归/Ruff、docs/Wiki 与 715,927,744-byte 派生物清理。

## 2026-08-16 TASK-0257 Harwood additive training bridge

- [x] 完成 v2 additive contract、TDD 矩阵、资源边界与非晋级 spec。
- [x] 实现 source-group manifest 与 v2 export/verifier，保持 TASK-0254 v1 全部字节不变。
- [x] 生成四视频 training manifest 和 45 行 MViT/Swin embeddings，batch size 1，资源守护。
- [x] 运行四比赛 nested outer-held probe，验证 outer-held label invariance 与两制作域诚实声明。
- [x] 完成真实工件验签、fresh review、guarded full suite、scoped Ruff、docs 和安全磁盘清理；
  全仓 Ruff 历史基线债务未跨任务批量修改。
- [x] 完成 TASK-0257 llm-wiki post-development hook；cross-model interactive review 未获显式授权，
  依约未运行。

## 2026-08-16 TASK-0258 Phase-0 capability map

- [x] 用户于 2026-08-16 批准 `docs/specs/TASK-0258-temporal-canary/capability-map.md`。
- [x] 完成 Module-A requirement/solution/gate 与 fresh-context 规格复核，最终
  Critical/Required/Optional=`0/0/0`；精确 SHA tuple 记录在 `tasks/plan.md` 和 gate review。
- [x] 用户明确批准当前 exact-SHA Module-A 规格；批准后外部收据 internal/file SHA 为
  `42273c9db3e5c77da8f577ed0e5142e6210faf3caee53f572402229eac71944f` /
  `0eb1759d992b9587ad47bf9ae08793a69bac266ada81d39258919ac61e06e9fb`。更早的 v1 收据与执行
  不具备这次批准的时间顺序，只能作为历史诊断。
- [x] 批准前 v1 运行完成 45/45 私有行，但最终发布前磁盘保底复验失败；仅封存不可晋级
  `terminal_failure_v1`，未留下 embeddings、retrospective、双 evaluator 或 mechanical pass。
- [ ] 先完成独立 fresh-context implementation review，并由用户另行明确决定是否为磁盘失败新开版本；禁止删除/
  覆盖 v1 terminal evidence，禁止进入 Module B，readiness `not_ready`、blind inference paused。

## 2026-08-16 TASK-0259 local YOLO checkpoint resolution

- [x] 用 RED 测试证明默认裸文件名会忽略仓库内已保留的 YOLO checkpoint。
- [x] 统一本地默认路径并保留环境覆盖；focused 与相邻回归、Ruff、formatter 均通过。
- [x] 真实 local curl hook 完成，未发生权重下载；继续保持当前模型与 readiness 门禁不变。

## 2026-08-16 TASK-0260 four-game independent VLM resource probe

- [x] 证明四比赛 48 个候选键与仓内 667 行既有 VLM 预测零重叠，旧结果不可复用。
- [x] 运行 guarded EBQwen 16-window 与 Qwen3-VL 2B 8-window 探针；两者均以 exit 75 安全停止。
- [x] 保留计划、resource logs 与 Qwen3 单行断点 cache；不生成完整 prediction/evaluation/fusion 工件。
- [x] 停止 Ollama 模型并确认无残留推理进程；不降低 90%/2 GiB/0.25 GiB 资源边界。
- [x] 删除约 2.34 GB 可重建依赖、视觉派生物、pytest scratch 与重复权重副本；保留所有正式证据。
- [x] 归档资源阻点；不改变 readiness、blind inference 或 Module B 授权状态。

## 2026-08-16 TASK-0261 SmolVLM independent-video capability screen

- [x] 固定并验签公开 Apache-2.0 MLX SmolVLM2 256M/500M 候选，不降低现有 memory/swap guard。
- [x] RED→GREEN 支持 SmolVLM processor 像素预算和实际帧数 image token 映射；新增 compact JSON prompt。
- [x] 完成 256M 四比赛 8-window guarded probe：资源可承载，但 0 个 non-unknown 决策，不能评测或融合。
- [x] 删除无帮助的 256M 权重；500M 首次下载大小与上游 SHA 不匹配，未执行即删除损坏 payload。
- [x] clean-room 分块重下并验签 500M；guarded negative-first/compact 都是 P/R=`0/0`，不得作为 veto/fusion，验签后删除 `625,339,992` bytes 权重与配置。
- [x] 保留精确 receipts/evidence，完成 215 项相邻回归并累计清理 `1,849,575,509` bytes；不改变 runtime、readiness、blind inference 或 Module B 授权。

## 2026-08-16 TASK-0262 lightweight VLM compatibility gate

- [x] 核验 SmolVLM image-instruct、nanoLLaVA、InternVL、LLaVA-Interleave 与 Moondream 候选的固定 revision、许可、体积和本地后端能力。
- [x] 仅下载唯一通过元数据门禁的 `SmolVLM-500M-Instruct-4bit`，逐文件核对大小、权重 LFS SHA 和 safetensors 头。
- [x] 在冻结 8-window plan 与原资源 guard 下执行；模型在预测/标签前因无原生 video processor 以 exit 1 fail closed，不改变输入契约。
- [x] 删除 `291,652,074` bytes 无帮助模型载荷，只保留 README、source manifest、候选审计与 resource log；不评测、不融合、不晋级。

## 2026-08-16 TASK-0263 permissive basketball DEIM external screen

- [x] 从 8 个 `ortizeg` Apache-2.0 ONNX detector 中按预先固定的许可、体积和官方报告选择 DEIM-M；未以 AGU target 结果挑模型。
- [x] 以 TDD 固定官方预处理、raw10 球类映射、严格有限数/输出 shape 和三档阈值；聚焦测试 `9 passed`。
- [x] 在封存 LAL–BOS 108-row review（106 determinate）上运行 CPU guarded screen；资源峰值 memory `84.5%`、CPU `76.4%`、RSS `0.531 GiB`，无 guard breach。
- [x] 最佳 F1 仅 `.150538`，最佳 recall `.55` 对应 precision `.028796`；拒绝 runtime/fusion 后删除 `77,722,583`-byte 权重，只保留可重放证据。

## 2026-08-17 TASK-0264 DVIDS adult full-game metadata audit

- [x] 只读核验三场 DVIDS 2024 Armed Forces Basketball 成人整场官方页面和 Public Domain 标记；不下载媒体。
- [x] 将 VTV 重叠、Internet Archive 非权威镜像、青少年治理与 Auburn 分卷来源明确 fail closed。
- [x] 封存 `analysis_outputs/public_research/agu_dvids_armed_forces_basketball_metadata_audit_2026-08-17/audit.json`
  （artifact SHA `076e69a291117609ad71ef3d45f8b04b56030da90f6a4bb28632de0efd3e38e2`）。
- [ ] 只有 Module A 产生有效 mechanical pass 且用户另行授权 Module B 后，才可冻结有限候选 universe；
  仍需逐 item rights、下载访问、精确 payload receipt、磁盘与连续性核验。当前不选源、不下载，readiness 不变。

## 2026-08-17 TASK-0265 canonical .venv bytecode cache cleanup

- [x] 验证 VLM legacy hardening 已在当前树中通过 240 项核心/相邻测试，不重复枚举开放 schema 别名。
- [x] 删除 `.venv` 内 123,796 KiB 可重建 Python bytecode；保留全部包、解释器、权重和数据。
- [x] `.venv` Python 3.11.15、Torch 2.13.0、scikit-learn 1.9.0 导入通过，`pip check` 无坏依赖，
  Harness `2 passed`、VLM focused Ruff 通过，旧 `venv/` 不存在。
- [x] 封存 `analysis_outputs/public_research/agu_venv_bytecode_cache_cleanup_2026-08-17.json`
  （artifact SHA `a0da89f0d2d86d138bf3f1f213407568dc0e0411c370898876c4bc6b5496390c`）。
- [ ] 磁盘当前达到写入前门槛，但仍需不同 fresh-context implementation review 和用户单独明确指令，
  才可开新的 TASK-0258 versioned rerun；Module B 仍未授权。

## 2026-08-17 TASK-0266 DVIDS download-endpoint metadata boundary

- [x] 解析三场官方下载弹窗并记录 21 个 derivative file ID、分辨率、近似大小与编码；媒体 body 0 bytes。
- [x] HEAD-only 抽查每场 512p/256p 端点；6/6 为 403，未获得 exact bytes 或 direct payload receipt。
- [x] 核对官方 Search/Asset API 的 API-key 与 403 契约；无授权 key 时 fail closed。
- [x] 封存 `analysis_outputs/public_research/agu_dvids_download_endpoint_audit_2026-08-17/audit.json`
  （artifact SHA `77f2b27ef6d5ff831143e26293be83eca3196ec09ade60c8d2295f5f8e058dd0`）。
- [ ] 只有未来 Module A mechanical pass、Module B 单独授权、授权账户/API key 与 item-specific rights/disk
  preflight 都成立时，才可冻结 exact payload receipt；当前不下载、不选择、不改变 readiness。

## 2026-08-17 TASK-0267 closed independent-VLM evidence（Phase-0）

- [x] 证明当前 32-key nested v4 plan 没有任何 plan-SHA-aligned VLM prediction；历史预测不可重绑。
- [x] 冻结拟议模块边界：label-hidden shared plan、receipt-bound VLM/base predictions、label-free fusion、
  prediction-first evaluation、resource-guarded runtime adoption。
- [x] 保留 v1 legacy/non-promotion 与当前 32/45-row development-only 边界；不声称正式/运行/85% 达标。
- [ ] 用户审阅 capability map file SHA
  `23580768847694fc0fc9b891c133c03eb58ef6c14b3a24ab9025be955d380c73`；获批只允许起草第一个模块规格。

## 2026-08-17 TASK-0268 second-VTV adult full-game metadata audit

- [x] 证明 v41 后既有三源队列在 15 秒排除协议下没有剩余非重叠窗口，不继续制造 source-local leakage。
- [x] 核对 Commons Spartans–Cocodrilos 成人职业整场的官方元数据、Public-domain 类别、时长、原片 receipt
  和 480p derivative URL/带宽；只读元数据，下载媒体 0 bytes。
- [x] 封存 `analysis_outputs/public_research/agu_wikimedia_vtv_cocodrilos_metadata_audit_2026-08-17/audit.json`
  （artifact SHA `eb229f90718f1072fb02c6f7dd712be74eac0fa656b0daeae44ea52cc1e9cb54`）。
- [ ] 只有未来 Module A mechanical pass、Module B 单独授权、exact derivative file receipt、连续单场像素、
  与现有 VTV 内容去重及 privacy/publicity governance 全部通过后，才可冻结为第二场 VTV source；当前不下载。

## 2026-08-17 TASK-0269 second-VTV exact remote endpoint receipt

- [x] 用两个公共 DoH resolver 交叉确认真实 upload endpoint，并用 hostname SNI 做两次 HEAD-only 复核。
- [x] 冻结 480p remote receipt：`986,415,444` bytes、SHA-1 `bf16fe76…9faf`、ETag `718a…690b`；
  media body 仍为 0 bytes。
- [x] 封存 `analysis_outputs/public_research/agu_wikimedia_vtv_cocodrilos_endpoint_receipt_2026-08-17/audit.json`
  （artifact SHA `e768aff432f27d44fd826887ddae759c595894485732cc3246f917bc6bdf48b0`）。
- [ ] 保留远端收据；只有未来 Module B 单独授权后，才可 resource-guarded 下载、验 size/SHA-1、计算本地
  SHA-256，并继续连续性/内容去重/privacy-publicity 复核。

## 2026-08-17 TASK-0270 bounded Commons long-form inventory

- [x] 枚举 7 个 Commons 国家篮球视频分类的 66 个文件，并按 ≥20 分钟筛出 40 条 long-form rows。
- [x] 拒绝纪录片、分卷比赛、同 VTV production family 和制作/权利依据未独立验证的 Hidrocarburos rows。
- [x] 封存 `analysis_outputs/public_research/agu_commons_longform_basketball_inventory_2026-08-17/audit.json`
  （artifact SHA `957ddae1960a4d11e79dcd8364ef94af30463ea62f06e9597ce3c63c454868b8`）。
- [x] 该七分类限定结论已由 TASK-0271 的多语言全文检索补充；历史 inventory 保持不变，但不再代表 Commons
  全站没有第三制作域 lead。

## 2026-08-17 TASK-0271 multilingual Commons TBF source audit

- [x] 冻结 `basketball/baloncesto/basquete/basket-ball/basketbal/баскетбол` 六个全文查询与 100-result cap，
  记录 243 个去重视频和 55 个 ≥20-minute rows。
- [x] 核验 TBF Merkezefendi–Samsunspor 的成人职业整场、制作方、CC BY 3.0、reviewed license、原片 receipt
  与不同 HCTV/VTV 制作域边界。
- [x] 两次 HEAD-only 绑定 480p derivative 为 818,675,123 bytes、SHA-1 `35d79e3a…5cba`；未下载媒体。
- [x] 封存 `analysis_outputs/public_research/agu_wikimedia_tbf_merkezefendi_samsunspor_metadata_audit_2026-08-17/audit.json`
  （artifact SHA `ce3d07e8d534116223ffe6cecd835509dc3121de7a4e7d02fa3e2990713a0fa0`）。
- [ ] 等待 Module A mechanical pass 与用户单独授权 Module B；同时补足磁盘 reserve、本地 payload SHA-256、
  单场连续像素、内容 overlap 和 privacy/publicity。当前不下载、不选择、不改变 readiness。
