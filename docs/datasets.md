# Dataset Guide

AGU does not include datasets in the repository. This keeps the project lightweight and avoids redistributing videos or annotations whose licenses may differ from AGU's MIT-licensed source code.

## Supported Training Layout

The current training scripts expect a local dataset layout similar to:

```text
dataset/
├── annotation_dict.json
├── augmented_annotation_dict.json
├── examples/
│   ├── clip_0001.mp4
│   └── ...
└── augmented-examples/
    ├── clip_aug_0001.mp4
    └── ...
```

`dataset/` is ignored as local runtime data and should not be committed.

## SpaceJam Notes

AGU training utilities were designed around a SpaceJam-style basketball action dataset.

Before using SpaceJam or any other third-party dataset:

- Read the original dataset license and terms.
- Download data from the original source or official mirror.
- Keep attribution in experiment notes.
- Do not redistribute raw videos through this repository.
- Do not publish trained weights unless the dataset terms allow derived model distribution.

Suggested local preparation flow:

```bash
mkdir -p dataset
# Place annotation_dict.json under dataset/
# Place video clips under dataset/examples/
python scripts/gen_splits.py --annotation-path dataset/annotation_dict.json
```

## Smoke Data

For CI and public examples, prefer tiny synthetic or explicitly licensed smoke data.

Recommended rules:

- Keep files small.
- Avoid private footage.
- Avoid copyrighted broadcast clips unless redistribution is allowed.
- Document the source of every sample video.

`examples/benchmark/` is AGU's checked-in public contract fixture. It contains
only authored JSON/CSV labels and predictions. A deterministic license-free MP4
can be generated locally with `scripts/make_public_benchmark_fixture.py`; the
generated video remains under ignored `analysis_outputs/`. This validates
evaluation plumbing and must not be reported as model accuracy.

## Publishing Dataset References

If you add a dataset reference, include:

- Dataset name.
- Official URL.
- License or usage terms URL.
- Expected local directory layout.
- Required annotation format.
- Any preprocessing steps.

## Annotation Format

AGU's training utilities expect annotations that can be mapped to the 10 action labels:

```text
0 block
1 pass
2 run
3 dribble
4 shoot
5 ball in hand
6 defense
7 pick
8 no_action
9 walk
```

If you adapt a dataset with different labels, document the mapping and keep the mapping script outside generated output directories.

## Non-Commercial Public Research Ladder

AGU separates reusable pretraining data, identity enrollment, and post-freeze
acceptance truth. Public data never becomes a runtime answer channel.

| Source | AGU role | License boundary | Runtime use |
| --- | --- | --- | --- |
| TrackID3x3 | player detection, tracking, pose, identity pretraining | CC BY 4.0 data; Apache 2.0 code with upstream exceptions | trained model only |
| TeamTrack | detection and MOT pretraining | MIT as declared by the distribution; verify the downloaded copy | trained model only |
| BARD | event-semantics pretraining | CC BY 4.0 repository/annotations; 201 validation MP4 blobs are embedded, but referenced NBA media rights remain separate | trained model only after an external-domain gate |
| E-BARD Detection | ball, hoop, player and referee detection pretraining | CC BY 4.0 dataset; verify broadcast-frame redistribution before release | trained model only |
| [E-BARD ObjectClassification](https://huggingface.co/datasets/GabrieleGiudici/E-BARD-ObjectClassification) | object-role crop/VLM semantic pretraining | CC BY 4.0 crop archive; frame-level split shares game IDs | offline training only; no cross-game benchmark |
| WASB-SBDT | basketball ball-coordinate model screening | MIT code/checkpoint; underlying Basketball dataset media rights are separate | trained model only after AGU validation |
| SpaceJam | coarse basketball action pretraining | repository declares MIT; separately review broadcast clips | trained model only |
| Basketball-51 | game-grouped shot type/outcome pretraining | Apache 2.0 as declared by uploader; verify broadcast-clip rights | trained model only after source-group gate |
| F-16-NBA shot test | clip-level shot make/miss outcome screen | Apache 2.0 as declared by Hugging Face card; underlying NBA media rights remain unverified | offline screen only; no import |
| BasketHAR | wearable-signal research reference only | Apache 2.0 Hub release; current release has no accessible video | no visual import or AGU pretraining |
| Wikimedia Commons portraits | optional named face enrollment | record every file's individual license and attribution | sealed face embeddings/gallery |
| NBA_Games | two independent full-game acceptance candidates | MIT metadata; underlying NBA/YouTube media and statistics are external, private non-commercial research only | forbidden; post-freeze evaluator only |

The bounded 256-clip Basketball-51 subset was deleted after its balanced MViT
checkpoint failed target-domain transfer (ROC AUC 0.496). Its downloader,
source manifest, compact embedding/evaluation artifact and rejection evidence
remain; the 157,440 KiB decoded clip directory is not an active local dataset.

### Open Images V7 ball-box slice (2026-08-01)

The official [Open Images V7 download page](https://storage.googleapis.com/openimages/web/download_v7.html)
provides validation bounding boxes and a downloader for selected image IDs. AGU
downloaded only a deterministic, hash-bound local slice under
`dataset/public_sources/open_images_sports_ball_v7/`: 240 validation images,
358 boxes from the generic/sports-ball classes and about 79 MiB of pixels.
The source annotation CSV and validation image index were retained during the
screen; `manifest.json` records the original URL, landing page, per-image
license and pixel SHA-256 for every image. The annotation terms are CC BY 4.0,
while image rights remain per-file and must be verified before redistribution.

Fourteen selected images carry verified basketball/basketball-player image
labels and visibly include broadcast basketball, but most of the selected
boxes are generic or other sports balls. A MobileNetV3-small exact-crop
verifier trained on these boxes (one deterministic non-overlapping background
crop per image, GroupKFold by image ID) reaches source OOF AP about `0.993` yet
transfers to the disjoint LAL–BOS RF-DETR review at only lower-bound P/R
`0.559/0.356` (best regularization), so the screen is rejected. Screen SHA-256
is `53a6d52248c63ff04c30220c0b8b7e2f3ca2320d0263051153fafb9801446639`;
resource guard recorded zero stops with peak process-tree RSS
`834,289,664` bytes and peak system memory `84.6%`. No checkpoint or runtime
path is promoted. The generic pixel and raw CSV payloads were removed after
the failed gate; the manifest, README and screen/resource evidence remain, so
a future re-materialization can be bounded by the recorded source URLs and
hashes. It is not a replacement for game-diverse broadcast annotations.

### YouTube-BoundingBoxes audit (2026-08-02)

The official [YouTube-BoundingBoxes page](https://research.google.com/youtube-bb/)
declares a CC BY 4.0 dataset of 23 object classes, while its download page
lists only URL/frame-level classification and detection CSVs
([download details](https://research.google.com/youtube-bb/download.html)). A
bounded inspection of the official detection training CSV confirmed the class
list contains no `ball`, `sports ball`, or `basketball` class and has no
basketball event/state labels. No CSV, video URL media, or derived weights were
retained; the source is cataloged as research reference only and cannot fill
AGU's continuous small-ball supervision gap.

### Temporal basketball ball-tracking screen (2026-07-24)

The [official WASB-SBDT repository](https://github.com/nttcom/WASB-SBDT)
provides MIT-licensed code and a basketball checkpoint trained for three-frame
heatmap prediction plus temporal candidate selection. AGU downloaded only the
5.8 MiB basketball checkpoint and pinned source revision
`923462cacdeb3353b84ddebdedb3f4b7a8553b0f`; the checkpoint SHA-256 is
`8d1ba9870d0a6ab37b06ab82bed593c6c09133e713810bac475d0c000bb7e948`.
It remains an offline research asset and is not an AGU runtime answer source.

Other official sources did not satisfy the immediate reusable-data gate:
[APIDIS](https://ispgroup.gitlab.io/code/apidis/) has the desired
multi-view ball/player annotations but prohibits commercial use;
[VRU_Basketball](https://huggingface.co/datasets/BestWJH/VRU_Basketball)
is CC BY 4.0 but targets dynamic Gaussian reconstruction and provides no ball
coordinates; a later eight-clip Range-extracted subset is retained only for
offline scene screening. BASKET is gated player-highlight/skill data rather
than ball trajectories. TrackNetV3 and TOTNet are permissively licensed temporal
architectures, but their public checkpoints are racket-sport checkpoints. None
was downloaded as a basketball runtime dependency.

The expanded target-domain screen freezes 30 independently reviewed visible
ball frames across five broadcast events and camera geometries. WASB's raw
local peaks cover 22/30 at Top-1, 27/30 at Top-2 and 30/30 at Top-10, while the
official 0.5 threshold covers only 9/30. Current BODD candidates cover 26/30 at
Top-2, and the offline E-BARD-derived visual ordering also covers 26/30 at
Top-2. These are candidate-recall results, not deployable tracking results:
frozen greedy WASB association reaches 22/30 and the simple WASB+visual fusion
reaches 21/30. The 85% association gate therefore remains closed. Codex labels
in this experiment contain only offline ball visibility/location, are marked
`runtime_consumable=false`, and may not supply any runtime shot or statistic.

A follow-up used consecutive frames, matching the temporal assumption of the
official tracker. A global Top-10 peak path developed on five events reached
29/30, then hit 10/10 visible ball regions on two newly labeled event intervals
whose selections were sealed before annotation. This is enough only for an
opt-in experimental selector: all seven events come from one broadcast and no
complete-game statistic was evaluated. WASB remains an offline model research
asset rather than a default AGU runtime dependency.

Cross-game validation then froze the same rule on the disjoint 2008 LAL-BOS
broadcast before any new ball annotations. The first three-event plan reached
10/11 visible balls (.909) but correctly failed its predeclared sample floor:
only 11 visible frames remained and one event contributed two. That failure is
preserved. A separate supplemental plan selected three unused events and all
unique BODD hit frames without exact ball labels; dense inference and path
choices were sealed before review. It reached 12/13 (.923), with 5/4/4 visible
frames per event, and passed both the .85 accuracy and sample-size gates. This
authorizes only default-off downstream research. It is not a production
activation or complete-game statistic claim, and Codex remains prohibited as a
runtime answer source.

Generate the catalog and two-game acceptance plan from a local NBA_Games clone:

```bash
python scripts/build_public_research_plan.py \
  --dataset-root dataset/public_sources/nba_games \
  --benchmark 2011-05-10-atl-vs-chi=2011-05-06-chi-vs-atl \
  --benchmark 2008-06-05-lal-vs-bos=2008-06-08-lal-vs-bos \
  --verify-public-video --probe-media --yt-dlp-executable yt-dlp \
  --output analysis_outputs/public_research/nba_two_game_plan.json \
  --catalog-output analysis_outputs/public_research/source_catalog.json
```

The target games use different teams and seasons. Their enrollment games are
disjoint by YouTube ID and cover every active box-score player. The plan reports
separate gates for structured truth/public metadata, real playback extraction,
and complete local SHA-256 media sealing. oEmbed metadata is never proof that
video bytes are retrievable.

Use `--media-root` only after complete files named `<youtube-id>.<ext>` exist;
partial files are ignored. The result remains `runtime_consumable=false`: AGU
inference receives video, frozen models, configuration, and a benchmark-disjoint
face gallery, while box score/play-by-play truth is read only after predictions
are frozen.

Run the resumable downloader in a terminal or process supervisor. It retries
network and fragment failures indefinitely by default, uses capped exponential
backoff between fatal attempts, resumes `.part` files, and records SHA-256 only
after a complete file exists:

```bash
python scripts/download_public_research_media.py \
  --plan analysis_outputs/public_research/nba_two_game_plan.json \
  --output-dir dataset/public_sources/nba_games/media \
  --yt-dlp-executable yt-dlp
```

TrackID3x3 and TeamTrack both expose MOT-style player annotations. Import either
downloaded dataset without installing its upstream training stack:

```bash
python scripts/import_open_tracking_annotations.py \
  --dataset-root dataset/public_sources/trackid3x3 \
  --dataset-id trackid3x3 \
  --source-revision <immutable-revision> \
  --output analysis_outputs/open_pretraining/trackid3x3-mot.json
```

The importer recursively validates `gt.txt` rows, frame/track IDs and bounding
boxes, hashes annotations and colocated sequence videos, and emits an
`agu.open-tracking-dataset.v1` catalog. It is training-only, cannot overlap
acceptance media, and is not read by the FastAPI inference path. TeamTrack uses
the same command with `--dataset-id teamtrack`. TrackID3x3's optional jersey
number pipeline is isolated because its third-party license is CC BY-NC 3.0;
AGU does not require that subproject for MOT/ReID pretraining.

### BasketHAR bounded source audit (2026-07-27)

The official BasketHAR paper and Hugging Face card declare Apache 2.0 and
describe a synchronized 90-minute basketball training video with doubly
reviewed action boundaries. The paper reports 14,044 samples across 14 actions,
including 621 shooting samples. The pinned Hub revision is
`4b9d4b7c46bde1b61a4a02ce253e04c692056d43`.

Only 792 KiB of audit material was downloaded under
`dataset/public_sources/baskethar/audit/`: the label vector, README,
supplementary PDF and five official notebooks. The label SHA-256 is
`1a1aa2ada8b21a928c3408faf9bfd8490dcebab2c5a259dff2ae19a3fd7ecece`.
The 270 MiB signal matrix and externally hosted video were deliberately not
downloaded while disk free space was about 15 GiB.

The first label-map diagnosis was incorrect. The downloaded vector contains
counts `[621, 2706, 728, 1164, 81, 288, 2352, 3875, 783, 776, 118, 103,
294, 155]`, which follow the paper's class-number order and are approximately
1.25 times its per-class table values. The exact reason for the 1.25 ratio is
not stated by the release and remains an inference, while the official
notebooks still demonstrate leakage-prone random sample splits.

The decisive visual-data result is stronger: upstream commit
`1d44b79560c34926f5329536f0bba356c11e76e9` removed the only Google Drive video
link, and that former link now returns 404. The current accessible Hub release
contains wearable-signal arrays and supplementary material but no synchronized
raw video. BasketHAR is therefore rejected for AGU visual pretraining rather
than left as a pending candidate. Its catalog boundary is
`research_reference_only_no_visual_import`; it cannot support runtime
inference, target-domain evaluation or an accuracy claim.

### TrackID3x3 Indoor bounded acquisition (2026-07-27)

The official repository revision
`9822a3dde59fc80fbb5eaacd443e79e8b059d94e` declares the dataset CC BY 4.0 and
provides video, all-frame six-player boxes and partial ten-keypoint pose
annotations. AGU parsed the public Drive folder before downloading. Indoor is
42 MP4 files totaling 69,690,612 bytes; Outdoor is about 18.19 GB and the six
Drone videos total about 11.63 GB. Only Indoor was selected under the roughly
15 GiB free-space constraint.

`dataset/public_sources/trackid3x3/Indoor/source-manifest.json` binds every
Drive file ID, declared size, local SHA-256, repository revision and license.
Its final manifest SHA-256 is
`cb944d2cb78455fcb3ce6edc2991c908f2cbfefcb4645239f33bc06289ab95fb`.
All 42 downloads decode at 1280×720 and about 19.98 FPS. The fixed revision has
7,534 video/annotation frames, 45,204 boxes and exactly six boxes per frame;
this is three frames and 18 boxes above the paper table, so the hash-bound
release is authoritative for local experiments.

The TrackID3x3 importer now accepts the upstream named-MOT layout instead of
requiring only `gt/gt.txt`. The sealed training catalog contains 42 sequences,
45,204 annotations and media hashes for every sequence, with catalog SHA-256
`f96792dc18d8187d339f80f86cac6dc2a69d9a26c866eb12847fdefa5abe5dc2`.
Track IDs 1–6 repeat across all 42 clips, so the reported 252 sequence-track
observations must not be interpreted as 252 independent people.

Codex reviewed a hash-bound, raw-frame-only five-sample sheet for each clip.
The source contains short fixed-camera mini-game possessions and is useful for
player detection, tracking, pose and short drive/occlusion research. It has no
event labels, broadcast cuts, replay, scoreboard or free-throw examples visible
in the audit, and all clips use the same gym/camera and six players. It is not
accepted as supervision for replay, stoppage, free-throw or complete-game event
statistics. The optional upstream jersey-number pipeline remains excluded
because it is CC BY-NC 3.0.

The official E-BARD Detection archive is stored locally at
`dataset/public_sources/e_bard_detection/all.zip`; it is excluded from source
control. `source-manifest.json` binds Hugging Face revision
`00563215490c9a9642797b1495ea178535b3f59c`, archive SHA-256
`4b0a5ef8fd25565714e6b36a7020bc68b1cc2765afdc82a3b3d7a099e5c2ab81`,
the CC BY 4.0 declaration, and the extracted YOLO layout. The verified archive
contains 1,800 images and 22,210 boxes: 1,496 basketball, 1,565 hoop, 15,296
player and 3,853 referee annotations. It is training-only and must not be read
as an answer source during official inference.

The official E-BARD model repository also supplies an RF-DETR Nano
`checkpoint_best_total.pth` (historical SHA-256
`759968ecf8f83663c85de3d881713e072f4a9dd0ea2f136b082878b1e4fe2400`). It was
never a runtime dependency or promoted AGU model. The provider reports higher
precision but lower recall than its YOLOv8n detector; AGU evaluated it through
the optional Apache-2.0 `rfdetr==1.8.3` adapter, and a frozen ATL probe added no
causal hit beyond YOLO. The 120,811,190-byte local screening copy was removed
in the 2026-08-01 disk cleanup; the hash and historical evaluation remain here
for audit, while the checkpoint can be re-downloaded from the pinned source if
needed.

For the 2026-07-24 seed-verifier screen, AGU did not use E-BARD's official
train/validation/test split as an independence boundary: the three splits share
game IDs. Instead, the existing E-BARD YOLOv8n detector generated 3,675
candidate crops over 60 games at confidence 0.01. The best prediction for each
ground-truth ball at IoU >= 0.25 became a positive; predictions at maximum ball
IoU <= 0.05 became hard negatives; ambiguous candidates were omitted. The
sealed training-only manifest contains 1,348 positives and 2,327 negatives and
has file SHA-256
`887b3b5c1e4c1fa50e29ba76130d79eebf9ce97eb7c2403d71214a312d9d0efe`.
Evaluation groups by source game, not archive split. Neither this manifest nor
its frozen MobileNet screen is runtime-consumable.

Convert a sealed MOT catalog into benchmark-disjoint ReID crops without assigning
roster names:

```bash
python scripts/prepare_mot_reid_dataset.py \
  --catalog analysis_outputs/open_pretraining/teamtrack-mot.json \
  --dataset-root dataset/public_sources/teamtrack-mot \
  --output-root dataset/public_sources/teamtrack-reid-crops \
  --manifest analysis_outputs/open_pretraining/teamtrack-reid-crops.json \
  --identity-scope dataset_track
```

The crop manifest binds every JPEG to the source catalog and SHA-256, preserves
the upstream MOT track ID only as a training identity, and remains
`training_only=true`, `runtime_consumable=false`, and
`acceptance_media_allowed=false`. A five-clip basketball seed download currently
contains 49,403 annotations over 55 source tracks. The selected sequences are
contiguous segments of one upstream recording whose player track IDs are stable,
so the explicit `dataset_track` scope retains 999 crops across 10 identities
instead of incorrectly treating the same ten players as fifty classes. Do not use
that scope for datasets whose IDs reset per sequence.

Train and gate the hash-bound MobileNetV3 checkpoint with a temporally held-out
sequence:

```bash
python scripts/train_reid_model.py \
  --manifest analysis_outputs/open_pretraining/teamtrack-reid-crops.json \
  --crop-root dataset/public_sources/teamtrack-reid-crops \
  --output model_checkpoints/reid/teamtrack_mobilenet_v3_small.pt \
  --validation-sequence Q4_side_60-90 \
  --minimum-validation-top1 0.85
```

The current seed improves held-out retrieval Top-1 from 44.0% to 86.0%. This is a
training-domain promotion gate, not evidence for either target game's final event
accuracy.

When one benchmark/enrollment pair has completed, generate a truth-free handoff
before any perception or face work:

```bash
python scripts/prepare_public_game_pair.py \
  --plan analysis_outputs/public_research/nba_two_game_plan.json \
  --download-state dataset/public_sources/nba_games/media/download-state.json \
  --benchmark-slug 2008-06-05-lal-vs-bos \
  --output analysis_outputs/public_research/lal-bos-pair-handoff.json
```

This command fails until every file in the selected pair is complete and sealed.
Its output contains only media identity and hashes: no roster, box score,
play-by-play, active-player list, or statistics path. Enrollment face extraction
uses only `face_enrollment_media`; AGU inference later receives only
`benchmark_inference_media` after the gallery and models are frozen.

After Codex has exhaustively accepted/rejected enrollment-only face clusters,
build the sealed YuNet/SFace gallery and check it against an independently
declared expected roster:

```bash
python scripts/build_face_gallery.py \
  --manifest analysis_outputs/public_research/enrollment/annotated_face_list.json \
  --output analysis_outputs/public_research/enrollment/face_gallery.json \
  --detector-model model_checkpoints/opencv_face/face_detection_yunet_2023mar.onnx \
  --recognizer-model model_checkpoints/opencv_face/face_recognition_sface_2021dec.onnx

python scripts/check_face_gallery_coverage.py \
  --gallery analysis_outputs/public_research/enrollment/face_gallery.json \
  --roster analysis_outputs/public_research/enrollment/active_roster.json \
  --minimum-coverage 0.95 \
  --output analysis_outputs/public_research/enrollment/face_gallery_coverage.json
```

The roster uses schema `agu.face-roster.v1` with `players[].person_id`. The
coverage command exits with status 2 and emits `ready=false` when eligible
gallery entries cover less than 95% of the declared roster. Missing or
low-quality identities must remain unresolved; neither Codex nor an acceptance
box score may fill them at runtime.

For missing identities, collect license-recorded Wikimedia Commons candidates
without approving them automatically:

```bash
python scripts/download_wikimedia_face_candidates.py \
  --roster analysis_outputs/public_research/enrollment/portrait_roster.json \
  --gallery analysis_outputs/public_research/enrollment/face_gallery.json \
  --output-dir analysis_outputs/public_research/enrollment/wikimedia_candidates \
  --manifest analysis_outputs/public_research/enrollment/wikimedia_candidates.json
```

The adapter uses the official Commons MediaWiki API, records the file page,
author/credit, license name and URL, verifies image MIME signatures, and SHA
seals downloaded bytes. The resulting
`agu.wikimedia-face-candidates.v1` remains `runtime_consumable=false` and
`identity_verified=false`; Codex must still inspect the depicted face and AGU
must still create the SFace embedding. Commons requires checking each file's
individual reuse and attribution terms; catalog inclusion is not legal advice.

## Google DeepMind TAPNet / Online BootsTAPIR screening

- Official source: `https://github.com/google-deepmind/tapnet`
- License: Apache-2.0 for TAPNet software and the official checkpoint used
  here. TAPVid-3D has separate terms and was not downloaded.
- Screened source revision:
  `c2cbab81cc06092b5f05bfe2da7bfec54e2079c9`
- Screened checkpoint path:
  `model_checkpoints/google_deepmind/tapnet/causal_bootstapir_checkpoint.pt`
- Size / SHA-256: 218,887,028 bytes /
  `87c1e752cf5ce56e3e2f7da460aeb4d40fc826d04ef2939bade86a5c7495377f`
- Scope: optional offline point-tracking research only. It is not an AGU
  runtime dependency or an answer source. Apple MPS emits NaN tracks; CPU is
  valid but the frozen 12-event probe takes 417.862 seconds and does not meet
  the shot-validity precision gate.
- Cleanup: the rejected checkpoint and its Hugging Face download metadata were
  deleted on 2026-07-27. The sealed probe and source/SHA record are retained,
  so the experiment remains reproducible without treating the weight as an
  active local dependency.

## RTMPose-m Halpe26 screening

- Official implementation: `https://github.com/open-mmlab/mmpose`
- Lightweight inference repository:
  `https://github.com/Tau-J/rtmlib` at revision
  `2f36e3b62d73ed27a0b492a0d0fee1bba5fe3ace`
- License: Apache-2.0 for the inspected `rtmlib` source.
- Official model:
  `https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip`
- Archive / ONNX SHA-256:
  `55b81170e236040b59fc792ad0a8315301ac4c079a3bdb1095d838aad3088d18`
  /
  `26f3a19e61304a600dfb82d1001d41d24343b89fc70a33ffc84657e0b0bf2ecf`
- Provenance manifest:
  `analysis_outputs/open_pretraining/rtmpose_m_halpe26_source_manifest_v1.json`
- Scope: optional offline research only. The frozen 12-event screen reused AGU
  player boxes and OpenCV DNN CPU; it did not download another detector or add a
  project dependency. Three-frame wrist+ball sequence coverage improved only
  from 10/12 to 11/12, so the model was not expanded or promoted.

## Rejected release-chain datasets

- BasketHAR (`Xian-Gao/BasketHAR`, Apache-2.0) was inspected at revision
  `4b9d4b7c46bde1b61a4a02ce253e04c692056d43`. Its public release contains
  inertial-sensor arrays, spreadsheets and notebooks. Upstream commit
  `1d44b79560c34926f5329536f0bba356c11e76e9` removed the only video link and
  the former Drive URL returns 404, so the current release cannot train AGU's
  frame-level release chain.
- DeepSport publishes calibrated basketball images and ball annotations under
  CC BY-NC-ND 4.0. It is excluded from the open/commercially usable training
  path.
- SportsHHI is distributed under CC BY-NC 4.0 and is likewise excluded.

## SmolVLM2-2.2B MLX screening

- Source checkpoint:
  `https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- Apple Silicon conversion:
  `https://huggingface.co/mlx-community/SmolVLM2-2.2B-Instruct-mlx`
  at revision `844516024a1c4400d34489b89ee067d794e432ed`
- License: Apache-2.0.
- Weight size / SHA-256: 4,493,651,795 bytes /
  `ed6c59250704f09f921dce1a25e0d4eff611b6c9c53e382a7eb04ce9113f2773`
- Provenance manifest:
  `analysis_outputs/open_pretraining/smolvlm2_2b_mlx_source_manifest_v1.json`
- Scope: offline independent-VLM research only. The model failed the frozen
  four-event schema/recall risk gate and was not expanded or promoted.

Molmo2-8B was researched but not downloaded. Its official card describes an
Apache-2.0 checkpoint and strong video evaluation, while also warning that
third-party training datasets can be academic/non-commercial. The available
full-precision artifact also exceeds this 16 GB machine's remaining local
memory/disk budget. It is therefore not currently an AGU dependency or a
commercially cleared candidate.

## MEM-OKC 2011-05-11 reserved broadcast

- Source: publicly accessible full-game YouTube broadcast retained under the
  existing non-commercial research caveat; it is not a redistributable project
  dependency.
- Local media size: 749,106,815 bytes; 854x480, 30 FPS, 5,038.4 seconds.
- SHA-256:
  `1caf38b7b9c1997805cf5e66a37a05af974fdb5d99dfaa8816f328710523bbb8`.
- Scope: the raw video was isolated from the four-game development manifests.
  A three-window component-score prediction was frozen before event truth was
  opened. Its single emitted decision was correct, but live-FGA outcome
  coverage was zero and the game is now development-only after evaluation.
- Contamination control: 2011-05-13 MEM-OKC is explicitly excluded from blind
  eligibility because event-level play-by-play was exposed during diagnosis.
  Neither game may now be presented as a future untouched blind game.

## TransNetV2 replay-transition research assets

- Official source: `soCzech/TransNetV2`, MIT license, revision
  `85cef72af9a916bdfd7cc94a670c9cdfbf12d1ed`.
- PyTorch checkpoint source: `transnetv2-pytorch` 1.0.5 sdist, MIT license,
  PyPI archive SHA-256
  `72af739d55481e68782b096349cbf8e3ee76b09f1b676ebacd06aebd11eeda90`;
  checkpoint SHA-256
  `a313d0b3bebfa9a71914b375bfdf918a30b5c3b1e6be51972d35dd8078b442de`.
- Scope: offline raw-video shot-boundary geometry only. The four-video replay
  proxy failed promotion at precision 0.571 and recall 0.160, so transition
  semantics remain forbidden. The boundary detector may still delimit
  event-centered shots because that use consumes only timestamps, never a
  replay/highlight label. Neither weights nor outputs are runtime-consumable or
  AGU answer sources.
- The label-hidden event-plan run expanded boundary coverage from 11 resolved
  events to all 110 planned events (108 resolved, two unknown). The four
  boundary artifact SHA-256 values are
  `b937380bfb1a3ac374c8b7c9eebc41cb4cde64747ab65df872d40b754374f51a`,
  `259bdfc6e9a1429d99b8454c8a69db9585834bf5e79f5524fb83213025fa90a2`,
  `bdb33152b3aa3aadb8ee568d6f21b2ba574df56da1b00a9268c464a75d1de6b4`
  and
  `591b94382dba6d480a96ff0575b9192347613eb689633f84dcfed4ade1802f5b`.
  The official paper's fixed 0.5 threshold is retained; its open-world
  false-hit/miss warning is why boundaries are auxiliary geometry rather than
  event truth.

## Broadcast-clock replay evidence

- OCR engine: the existing optional RapidOCR ONNX Runtime 1.4.4 dependency;
  no new model or dataset was added.
- Development scope: 63 explicit windows from four distinct ATL-CHI/LAL-BOS
  broadcasts, bound to the existing raw-video, candidate and offline-label
  hashes.
- Frozen evaluation SHA-256:
  `c382346d5f00b431532a367f03ad350af73e980452e1f4681408b99632f0385c`.
  It records TP/FP/FN/TN `2/0/23/38`, precision `1.000`, recall `0.080`,
  a minimum prediction-support floor of five, and
  `promotion_eligible=false`.
- Post-freeze MEM-OKC evidence SHA-256:
  `78588fb0b509734b386f0b0c2feb190d02e5a40342449b619a5114de203dec67`.
  All three windows abstain, including the known replay, so this is a
  game-disjoint sparsity check rather than a successful validation.
- Scope: offline raw-video replay screening only. Clock absence never proves
  replay, post-shot frozen clocks are deliberately ignored, and no artifact is
  runtime-consumable or an answer source.

## Pending GSW-DAL / LAL-HOU two-game blind pair

- Benchmarks: 2007-04-25 GSW-DAL and 2009-05-08 LAL-HOU.
- Enrollment: two same-series games per benchmark, selected only to obtain 100%
  active-roster coverage without reusing benchmark video.
- Plan SHA-256:
  `6616a16ec1d3aa53cb375789036f8820a1de2d8e9c8a545ca5fa3b63495b0ad3`.
- Metadata and playback probes pass. Media download and SHA sealing do not:
  the current network fails TLS negotiation with the selected Googlevideo CDN
  under native yt-dlp, IPv4 and external curl. The pair is therefore reserved
  but not yet an admissible blind media evaluation set.

The sixth plan-bound external attempt produced the same zero-byte
`SSL_ERROR_SYSCALL` failure. This is transport evidence only; no benchmark
answers were opened and the media gate remains false.

## Archive-backed CHI-UTA benchmark pair

- Benchmark: 1998-06-14 CHI-UTA. Enrollment media: 1998-06-12 UTA-CHI and
  1997-12-17 LAL-CHI. The external broadcasts are non-redistributable local
  research inputs with no declared Archive license; the NBA_Games metadata
  remains MIT.
- All three files completed and passed media probing before cleanup. Their sizes/SHA-256
  (preserved as provenance) are:
  `1,866,259,876` /
  `bf6341a81ee55491c737e00fb8b62a636089f15223d28a5ed97971bdbe17610f`,
  `823,276,729` /
  `6c6e2a6b9b8466e2d169b383b3ebdbdb6791722a87ac6b03717a1f7708b5c746`,
  and `661,015,410` /
  `dae2cd2cfd471d18124aa702560ed838fe14e25cb1631f43326c10d07310ee16`.
- The truth-free pair handoff has canonical hash
  `ab8a723cd8672a81fe353c5e7d2d51b124a2b32992df04d92212a3e63004f`.
  It contains only media identities, sizes and hashes; it excludes answer
  assets and remains `runtime_consumable=false`.
- The three external media files were deleted on 2026-08-06 after the
  identity/development handoff was closed; the exact-file state and hashes remain
  in `dataset/public_sources/nba_games/media/chi_uta_gsw_dal_archive/download-state.json`
  and the handoff artifact. The source is now metadata-only and cannot be used for
  a rerun without a fresh download. It never satisfied the required two-game
  acceptance gate, and no benchmark inference result or 85% complete-statistics
  claim exists.

## EBQwen2.5-VL-3B independent raw-frame screening

- Official model: `GabrieleGiudici/EBQwen2.5-VL-3B`, revision
  `c6a93cbb325f9d20236f85bcaba7827a2808443e`, CC BY 4.0. Its model card
  reports exact-action precision/recall/F1 of 0.6709/0.5579/0.6092 and uses
  native video at 3 FPS and 420x784.
- Prior local artifact: Q4_K_S GGUF plus Q8 vision projector from
  `mradermacher/EBQwen2.5-VL-3B-GGUF`, revision
  `3942c6f942da55aedfd701bb29f1a37578aee942`. Exact sizes and hashes are
  recorded in `THIRD_PARTY_NOTICES.md`; the rejected local copies were removed
  after the screening result was sealed.
- Frozen development plan: 32 raw-video windows, exactly eight from each of
  four ATL-CHI/LAL-BOS broadcasts. Selection was SHA-stable and stratified at
  four positive/four negative windows per video, while the runner received no
  labels, review notes, play-by-play or statistics. The input contract is six
  raw frames at width 704.
- Result: TP/FP/FN/TN `16/12/0/4`, precision `0.5714`, recall `1.0000`.
  Per-video precision ranges from `0.5000` to `0.6667`; every video fails the
  fixed precision `>=0.95` gate. `promotion_eligible=false`.
- Boundary: this tests Ollama multi-image compatibility, not the official
  native video-temporal path. All plan, prediction and evaluation artifacts
  declare `runtime_consumable=false` and
  `codex_runtime_answer_used=false`. No result is connected to runtime fusion.

### Native-video MLX follow-up

- The complete official BF16 checkpoint at revision
  `c6a93cbb325f9d20236f85bcaba7827a2808443e` was hash-verified and converted
  locally with `mlx-vlm==0.6.7` to affine 4-bit, group-size 64. The conversion
  ran under the CPU/memory guard and produced a readable 1,330-tensor
  safetensors artifact.
- The native runner passes a raw temporal frame tensor through Qwen2.5-VL's
  video processor with a frozen 2 FPS / 151,200-pixel budget. Labels,
  statistics, review notes and Codex answers are not supplied to inference.
- On the same sealed 32-window development set the native path predicted all
  32 windows as live field goals: TP/FP/FN/TN `16/16/0/0`, precision `0.5000`,
  recall `1.0000`. All four videos independently scored `0.5000/1.0000`.
  `promotion_eligible=false`; this result is development evidence, not a new
  blind acceptance claim.
- The evaluator verifies the derived native plan against the exact frozen
  annotation-source plan before opening annotations. Predictions and
  evaluation remain `runtime_consumable=false` and
  `codex_runtime_answer_used=false`.

BARD and E-BARD code/annotations remain useful CC BY 4.0 research assets, but
their referenced basketball broadcasts require separate media authorization.
MultiSports was rejected for commercial/open AGU training because its dataset
license is CC BY-NC 4.0. FineSports was not treated as immediately open because
access requires a signed release agreement.

## 2026-07-26 event and commentary source screening

This pass looked specifically for player-grounded, temporally localized
basketball events that could train the missing player-ball-global interaction
stage. No large duplicate download was started:

| Source | License/access finding | AGU decision |
| --- | --- | --- |
| [BasketEvent / PlayNet](https://github.com/zhangyu2003/BasketEvent) | Code is public at revision `8a313f3ad4476735ddac38543578e19c1bccebd5`, and the [Hub repository](https://huggingface.co/datasets/zaywas/BasketEvent) exposes trajectory JSON files plus a 1,813,273,995-byte `playnet.pt` at revision `85aaa3ce62bc096e3995c39ecfa6773fcc9fe5e1`. Neither repository declares a license, the Hub release contains no matching source videos, and the code assumes Python 3.12/CUDA plus machine-specific paths. The paper reports only 0.682 macro F1 for PlayNet. A bounded `.venv` load/forward probe matched all state keys, but the unlicensed checkpoint was deleted after hashing. | Research evidence only. Keep the compact provenance/probe artifacts; do not import into runtime or train until upstream supplies compatible terms, raw media and frame-level labels. |
| [GameCommBench](https://huggingface.co/datasets/anonymousForBlind/GameCommBench) | 2,981 aligned basketball clips, about 21.8 GB, with the Hub license recorded as `other`. | Reject for AGU open/commercial training until explicit compatible terms exist. |
| [BASKET](https://huggingface.co/datasets/yulupan/BASKET) | Apache-2.0 metadata, gated access, about 1.93 TB, and primarily basketball skill estimation rather than broadcast event localization. | Reject for this event-fusion increment: wrong target, gated, and disproportionate size. |
| [saveerjain/basketball-events](https://huggingface.co/datasets/saveerjain/basketball-events) | Small event-clip collection, but no declared license was found and the viewer/schema was not consistently loadable. | Reject until provenance, license, and schema are repaired upstream. |
| [choucsan/NBA_Games](https://huggingface.co/datasets/choucsan/NBA_Games) | Public MIT dataset already present locally at revision `3a20f2b9f60a025c8c641f4c763d4ac467ff059e`; 189 game metadata rows and 166 play-by-play assets are available. | Reuse the existing local clone for development manifests and offline labels; do not duplicate-download it. |
| [TOTNet](https://github.com/AugustRushG/TOTNet) | MIT code at revision `8a757f63391b262c14d18b4095486336852dbeef`; the screened 94,145,122-byte tennis checkpoint had SHA-256 `36caadd2453cf1a37f26afb0024b861fd3285ca6e9a91e9e4a20d35e90a0a24a`. | Reject for AGU integration: the weight is not basketball-trained, MPS lacks a required 3D pooling operator, one CPU window took 17.88 seconds, and a three-window batch was stopped after more than four minutes under the resource guard. The downloaded clone and temporary dependency were removed. |

BasketEvent's global-player, player-player, player-ball and gated cross-clip
architecture matches the current technical gap, but the present release is not
an open-source training dependency. A deterministic 70-file audit found all
files readable, but 36 clips contained duplicate player-name tracks, action
labels use both title-case and abbreviated lower-case forms (`ast`, `block`,
`steal`), and trajectory rows averaged about 9.6% missing boxes. Those findings
do not substitute for the missing license or videos. AGU may use the published
paper as research evidence, but must independently implement the interaction
contract from already licensed local development assets and hash-sealed Codex
offline annotations.

## 2026-07-27 targeted free-throw/event source rescreen

This pass rechecked primary project pages and immutable repository revisions
before any new download. The existing BARD import was also counted directly:
it already contains all 14,676 annotation rows from 60 games at revision
`add4109bf8b2034a32f3b6a83fa0d8c0ae638473`, with 23,523 normalized actions.
That includes 2,589 free-throw actions in 2,547 clip records and 2,305 foul
actions in 2,283 records. BARD is therefore not annotation-starved. Its clip
URLs point to external NBA media, however, and it does not label replay or
dead-ball negatives, so no additional BARD media was downloaded.

| Source | Primary-source finding | AGU decision |
| --- | --- | --- |
| [NSVA](https://github.com/jackwu502/NSVA) | Revision `cf93ad95b4f1be7e13c2db8802243acb4f73f37c`; the project explicitly applies CC BY-NC to most code and data and separately asks users to ensure fair use for NBA media. | Taxonomy/paper reference only; reject as an open/commercial AGU training dependency. |
| [NCAA Basketball Attention](https://research.google/pubs/detecting-events-and-key-actors-in-multi-person-videos/) | The paper reports 257 games, 14K annotations and made/missed free throws, but the official App Engine dataset host and browser both return HTTP 404 and no dataset license is declared. The published preprocessing also removes instant replays. | Paper reference only; no annotation or media import. It cannot supply the missing replay/stoppage negatives. |
| [PoseShot](https://doi.org/10.1038/s41598-026-41025-0) | The paper uses 75 continuous free-throw videos with five phase labels, but its data-availability statement says the dataset is available only from the authors on reasonable request. The paper's CC BY 4.0 license does not independently publish the dataset. | Free-throw phase research reference only; no public dataset to import. |
| [nba_pbp_video_dataset](https://github.com/alijkhalil/nba_pbp_video_dataset) | Revision `385a7bbab71743072ee783612e9bb054acd2e1c1`; a legacy downloader for several-terabyte external NBA clips with no declared code or data license. | Do not execute or download. |
| [FineAction](https://github.com/Richard-61/FineAction) | Revision `b0bd8b10f39bbffa03be988cdfdf1106598653f4`; 106-class generic temporal-action data is available through OpenDataLab, but the official repository does not declare dataset/code terms and the taxonomy does not target AGU's free-throw/replay/full-game event gap. | Reject for this increment; no large generic-video download. |

These decisions are encoded in the public research catalog so future runs fail
closed instead of rediscovering or accidentally downloading the same sources.

## 2026-08-01 full-game PBP alignment source gate

The next development increment reuses the existing local `choucsan/NBA_Games`
clone rather than duplicating media. Its MIT metadata/PBP assets are sufficient
to build a hash-bound, training-only event timeline; the linked external game
videos remain private research media and are not redistributed. Two visible
LAL–BOS games now have scoreboard-anchored alignment artifacts with 460/467 and
439/442 mapped PBP rows respectively. The artifact contract records the raw-video,
PBP and scoreboard hashes and rejects sealed blind hashes before any labels can
be materialized.

Two tempting alternatives were closed before download:

| Source | Primary-source finding | AGU decision |
| --- | --- | --- |
| [SportsMOT](https://github.com/MCG-NJU/SportsMOT) | Official repository provides 240 720p/25fps sports clips, including basketball, but states CC BY-NC 4.0 and no redistribution without prior written permission. | Research reference only; do not download or use for an open/commercial AGU checkpoint. |
| [nba_pbp_video_dataset](https://github.com/alijkhalil/nba_pbp_video_dataset) | README describes a several-terabyte external NBA clip downloader with `event.xml` play labels, but no compatible code/data/media license is declared. | Fail closed; do not execute or download external NBA media. |

The alignment implementation therefore stays inside the AGU Python stack and
does not add either source as a runtime or training dependency.

## 2026-07-27 scoreboard OCR and ball-detector rescreen

| Source | Primary-source finding | AGU decision |
| --- | --- | --- |
| [ScoreboardOCR](https://github.com/RenarsKokins/ScoreboardOCR) | MIT code, but its supplied SVM is trained for physical seven-segment displays and requires manually selected regions. | Do not import for broadcast overlays; retain RapidOCR and temporal validation. |
| [ScoreSight](https://github.com/royshil/scoresight) | Cross-platform scoreboard GUI built around OpenCV/Tesseract; upstream marks development stalled. | Research reference only; no new GUI/runtime dependency. |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Apache-2.0 pipeline with small numeric-capable models, but published metrics use general OCR sets rather than AGU's NBA score bugs. | No download in this increment; reconsider only with a frozen broadcast-domain OCR benchmark. |
| [446f6e6e79/YOLO-basketball-fineTuned](https://huggingface.co/446f6e6e79/YOLO-basketball-fineTuned) | MIT-tagged YOLO11m weight trained on one university game; its card explicitly warns of little or no cross-game generalization. Local SHA-256 was `cc7fcb512d7c4d6a12b3d4ad828e5ca4ef6dff895d9f3e75e0f3900e75ed9313`. | Rejected after E-BARD test P/R/mAP50 `0.121/0.039/0.015`; deleted the ~39 MB local weight. |
| [Lumos-88 basketball YOLO11](https://huggingface.co/Lumos-88/YOLO11-fine-tuned-for-basketball-detection) | MIT-tagged, but trained mainly on 224 drone-view images plus augmentation and explicitly not validated for broadcast sideline views. | Do not download; domain and scale are weaker than already-tested E-BARD. |
| [Basketball Hoop, Ball and Player](https://universe.roboflow.com/basketball-stat-tracker/basketball-hoop-ball-and-player/dataset/2) | CC BY 4.0, but version 2 contains only 199 images (10 test) and no evidence of continuous release trajectories. | Catalog as possible annotation supplement, not sufficient current supervision; no download. |

The retained E-BARD detector source remains the stronger licensed baseline:
its official held-out basketball precision/recall are `0.811/0.566` for
YOLOv8n and `0.845/0.322` for RF-DETR Nano. AGU's earlier raw-window probe also
showed RF-DETR causal hits were a strict subset of BODD, so neither another
threshold sweep nor the rejected single-game/drone weights justify a runtime
change.

## 2026-07-27 targeted ball-trajectory source rescreen

Three newly indexed sources fail before download:

- UniqueData `basketball_tracking` exposes only a 70-image sample and declares
  CC BY-NC-ND 4.0; the card directs commercial users to a paid custom dataset.
  It is not eligible for AGU open training or derivative model work.
- UNC BASKET declares Apache 2.0 and spans 4,477 video hours across 21 leagues,
  but its released labels are 20 whole-highlight player skill ratings. It has
  no ball boxes, point trajectories, release timing or event boundaries, so a
  multi-terabyte download cannot address the current causal-release blocker.
- SVI-Bench is a 15.2 TB gated benchmark under CC BY-NC 4.0, requires an
  institutional `.edu` account, forbids redistribution and limits access to
  approved non-commercial research. Its tracker assets are not an open
  commercial-training source for AGU.

All three are recorded as fail-closed research references in the public source
catalog. No media, code or weights were downloaded.

## 2026-07-28 event-state source and local supervision rescreen

- BasketEvent/PlayNet is the closest newly published architecture to the
  current player/event/time grounding gap, but the paper still says that data,
  code and models will be made public. The already-indexed repositories do not
  provide compatible dataset terms or source broadcasts, so AGU keeps it as
  research evidence rather than a training dependency.
- BASKET is Apache-2.0 and large, but labels whole-player highlight skill
  levels rather than free-throw, dead-ball, replay or release/outcome
  intervals. SpaceJam is MIT code for short single-player actions, not
  broadcast event state, and its public dataset link is unavailable. Neither
  justifies a large download for this blocker.
- The local reviewed development subset is now materialized as 81 hash-bound
  five-frame rows: 41 `FREE_THROW` and 40 `LIVE_FIELD_GOAL`, distributed over
  four source videos as 27/5/6/43. It is training-only,
  `runtime_consumable=false`, excludes 29 unresolved rows, and rejects sealed
  blind hashes at both build and load boundaries.
- The dataset manifest SHA-256 is
  `5a00875f6611558cc9309623ff84be99b03538d7331107e915da6d227f25f1e8`.
  No new external media or model weights were downloaded in this rescreen.

## 2026-07-29 reviewed-state expansion and source audit

- A wider, label-hidden raw-frame follow-up resolved 27 of the 29 previously
  uncertain development rows. The combined training-only supervision now has
  108 examples: 55 free throws and 53 live field goals across four source
  videos. Two cutaway/graphic/replay-fragment rows remain explicitly
  unresolved. The follow-up correction artifact is parent-hash-bound and has
  SHA-256
  `74cdb7d0dc2a49aa1f04eaac5a21e96a3c95a5eb16981b4d81ab8cac2c81cae1`.
- The retained compact Basketball-51 artifact contains 256 MViT embeddings
  from 51 source groups. It and the 511-row target artifact use the exact same
  official Kinetics-400 MViT weight SHA-256
  `ae3be16733081f6d1cd40e4ab980ca23d6df6dc6486d15ada05a5e8ab8c9b975`.
  Strict nested leave-one-target-video-out assistance still reaches only
  `0.695` pooled balanced accuracy, `0.692` F1 and `0.629` weakest-video
  balanced accuracy. The source artifact remains training evidence only.
- The existing 39-feature pose/ball/rim artifact covers all 108 reviewed rows,
  but a direct visual-state screen falls to `0.601` pooled balanced accuracy
  and `0.575` weakest-video performance. A new read-only 262-feature
  seven-timepoint pose-layout diagnostic also fails at `0.612` pooled and
  `0.333` weakest-video balanced accuracy. No diagnostic weight was retained.
- Upstream BasketEvent remained at code revision
  `8a313f3ad4476735ddac38543578e19c1bccebd5` and Hub revision
  `ca4af13e7754e0f2fe9653a604b5182080b6a62a`; neither release now supplies a
  compatible license plus source video. `leharris3/basketball-shot-test-dataset`
  is MIT-tagged and about 423 MB but gated, has an empty card and does not
establish broadcast-media provenance. `BestWJH/VRU_Basketball` is CC BY 4.0
but its full approximately 1.51 GB multiview reconstruction archive has no
event-state labels; only eight clips were later extracted for offline screening.
`saveerjain/basketball-events` and its mirror restrict use to research and do
not provide compatible open terms. None of these sources is a runtime dependency.
- No external media, dataset archive or new model weight was added. EBQwen
  native BF16 and MLX 4-bit inference assets remain retained; sealed HOU–ORL
  G1/G2 media and truth remain outside every training and review boundary.

## 2026-07-29 court-topology source screen

- `koppolusameer/yolo11n-basketball-court-keypoints` was pinned to model
  revision `6cb899251439982067a81baa225b33f45f335981`. Its 7,934,664-byte
  weight had SHA-256
  `68e5faf5fb5388cf83477238abe22e9824a69fb33a4cf34f79ab61858f410064`.
  The model is AGPL-3.0 and predicts 48 unnamed pixel keypoints; no semantic
  index-to-court/world-coordinate mapping is released.
- The associated 851-row Hugging Face dataset card says MIT while the
  underlying Roboflow project says CC BY 4.0. Its sampled filenames and
  images are NBA broadcasts, but the uploader does not document underlying
  broadcast-media rights. This conflicts with an open/commercial runtime
  dependency even before accuracy is considered.
- A label-hidden 12-frame risk probe found high-confidence false court poses
  on player/coach closeups and missed some clear court views. A strict
  108-row, four-game nested screen accepted only 57 rows under the
  `>=12/48` keypoints-at-`>=0.25` evidence gate and reached pooled balanced
  accuracy/F1 `0.602/0.598`; weakest-game balanced accuracy was `0.562`.
- The candidate is cataloged as research-probe-only and rejected. Its weight
  and temporary montage were deleted after the result; no dataset archive,
  classifier or runtime artifact was retained.
- KaliCalib code is CeCILL-2.1, but its basketball calibration data and
  trained weights derive from DeepSport under CC BY-NC-ND 4.0. It remains a
  paper/code reference and cannot supply AGU open/commercial training weights.

## 2026-07-29 DINOv2 centered-wide temporal evidence

- Existing pre/anchor/post DINOv2-small artifacts cover the same 110
  development candidates and use model SHA-256
  `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`.
- The official `facebook/dinov2-small` release is Apache-2.0. Revision
  `ed25f3a31f01632728cabb09d1542f84ab7b0056` was downloaded with only
  `config.json`, `preprocessor_config.json` and `model.safetensors`; the
  safetensors hash exactly matched the existing artifacts.
- A new centered nine-frame protocol at `-8..+8s` extracted 110 rows and 990
  embeddings. Its sealed artifact SHA-256 is
  `ceedfbc19c8ca7faf2a7b4001b91680bfd30f6ac2873baa6f7056552ae11a3fc`.
- Strict nested held-game screening rejected the centered-wide representation
  at pooled balanced accuracy/F1 `0.674/0.715` and weakest-game balanced
  accuracy `0.645`. The narrow temporal screen was stronger at
  `0.711/0.744`, but also failed the gate.
- The 84 MB downloaded checkpoint and 1.1 MB temporary error-review package
  were deleted after rejection. Compact embeddings, resource logs and screen
  JSONs remain training-only; no blind media or external dataset was added.

## 2026-07-29 dense cut-aligned MViT and source rescreen

- Reused the existing torchvision MViT-v2-S Kinetics-400 checkpoint, SHA-256
  `ae3be16733081f6d1cd40e4ab980ca23d6df6dc6486d15ada05a5e8ab8c9b975`.
  Torchvision code is BSD-3-Clause, while its repository cautions that
  pretrained weights may carry additional terms from their training data.
  This experiment is retained as local research evidence and does not
  redistribute the checkpoint.
- Extracted 16-frame dense clips separately inside the previous, anchor and
  next TransNet-bounded shots. Four artifacts contain 110 events and 256
  available clips; absent neighboring shots remain explicit masks.
- Strict nested leave-one-game-out screening reaches pooled balanced
  accuracy/precision/recall/F1/AUC
  `0.677/0.717/0.600/0.653/0.730`, with weakest-game balanced accuracy
  `0.645`. Artifact SHA-256 is
  `69c692acec1871b98903f771966d6307b721e64521ee6a186648e1ce2ec05abd`;
  `accepted=false`.
- A label-hidden in-memory fusion of cut-aligned MViT with existing DINOv2
  representations is weaker at pooled balanced accuracy/F1 `0.648/0.661` and
  weakest-game `0.625`, so no fusion artifact or model was retained.
- All guarded runs completed with zero breaches. Extraction peaked at
  736,428,032 bytes process-tree RSS, 85.6% system memory and 57.2% CPU;
  minimum available memory was 2,481,979,392 bytes and minimum free swap was
  1,579,614,208 bytes.

The same pass audited three primary-source candidates before any media
download:

| Source | Primary-source finding | AGU decision |
| --- | --- | --- |
| [MultiSports](https://github.com/MCG-NJU/MultiSports) | 66 multi-sport localized action classes; dataset and annotations are CC BY-NC 4.0, while only the repository software is MIT. | Paper/taxonomy reference only; no media or annotation import into open/commercial AGU training. |
| [FineSports](https://github.com/PKU-ICST-MIPL/FineSports_CVPR2024) | 10,000 NBA clips, 12 action types and 52 sub-actions, including free throw and result labels. Its signed agreement limits annotations to scientific research and forbids unauthorized commercial use or distribution; the repository declares no code license. | Do not sign, download, redistribute or train; paper taxonomy only. |
| [SHOT7M2](https://huggingface.co/datasets/amathislab/SHOT7M2) | Public, ungated synthetic release at revision `d92456bb29e574159f0bf728bd562d145ec5b827`; Hub card declares ECL-2.0. Five arrays total 4,670,992,142 bytes and contain one-agent 3D poses/actions, not broadcast frames, multi-player state, replay, free-throw formation or dead-ball labels. | Do not download the 4.67 GB assets for the current blocker; synthetic pose/action-segmentation reference only. |

No new external media, annotations or pretrained weights were downloaded.
HOU–ORL G1/G2 remain sealed.

## 2026-07-30 causal shot-phase review and source audit

- Built a raw-frame-only causal review package for all 110 development events.
  Each event exposes 24 time-ordered positions from `-4.0s`, through a dense
  `-3.5s..+1.5s` interval at `0.25s`, to `+2.0s` and `+4.0s`. Codex reviewed
  only those contact sheets, without labels, PBP, predictions or sealed blind
  media. The sealed review SHA-256 is
  `8431c47b74c3312e5b02d8f7b46925c9381c83bfd629bbb5356d1e1884174130`.
- The review records formation, broadcast context, sequence completeness,
  release visibility, rim-arrival visibility and outcome. It contains 110
  exact, unique event bindings; the existing main field-goal/free-throw
  correction chain resolves 108 of them. Two cutaway/stoppage rows remain
  explicitly excluded from the main target rather than receiving fabricated
  labels.
- A strict double-isolated screen used eight predicted phase features as an
  auxiliary input to the existing DINOv2 score. The outer fold holds out one
  source video, inner selection uses only the remaining videos, and both phase
  and meta-classifier training use other-game/OOF predictions. Baseline pooled
  balanced accuracy/F1 and weakest-game balanced accuracy are
  `0.674/0.711/0.656`; the phase auxiliary reaches `0.684/0.702/0.563`.
  Artifact SHA-256 is
  `ce25623081bc43add404f475abb667f8f68aa4d52c1f87fe33dfcb750d36528b`;
  `accepted=false`, so no runtime model or checkpoint was promoted.
- The guarded screen completed with zero resource breaches. Peak process-tree
  RSS was 560,447,488 bytes, peak system memory 79.5%, peak CPU 22.9%, and
  minimum available memory was about 3.53 GB.

The same pass audited four primary sources before any download:

| Source | Primary-source finding | AGU decision |
| --- | --- | --- |
| [CVBallTracking](https://github.com/GeekAlexis/CVBallTracking) | The repository has no root license. Its detector path uses generic YOLOv3 COCO weights rather than a released basketball-specific checkpoint or event dataset. | Code/pipeline reference only; do not import weights or data. |
| [AI Basketball Analysis on Google Colab](https://github.com/hardik0/AI-basketball-analysis-on-google-colab) | The repository relies on an OpenPose distribution restricted to noncommercial research, while its basketball videos/data are private. | Reject for AGU open/commercial training and runtime. |
| [Basketball Players v17](https://universe.roboflow.com/workspace-5ujvu/basketball-players-fy4c2/dataset/17) | The 320-image release is CC BY 4.0 and labels Ball, Player, Clock, Hoop, Overlay, Ref and Scoreboard. Official export requires a Roboflow API key, which is not configured locally. | Candidate only for future bounded detector self-training; no credential bypass and no download in this pass. |
| [Hana AI Basketball](https://github.com/hanac-ai/hanac-ai-basketball) | The repository and linked Google Drive weight do not declare compatible code, data or weight terms. | Reject the repository and weight; any separately licensed Roboflow source must be audited independently. |

No external media, annotation archive or pretrained weight was added. Existing
EBQwen native BF16 and MLX 4-bit inference assets remain retained, and HOU–ORL
G1/G2 remain sealed.

## 2026-07-30 dense causal MobileNet temporal screen

- Reused the 57 SHA-256-bound causal review sheets described above. Their
  manifest declares raw frames only and no labels or predictions. After
  removing rendered time/frame headers, they provide 2,640 ordered panels:
  110 events × 24 positions. No source video, label or blind asset was added.
- Reused the existing official torchvision MobileNetV3-small ImageNet-1K
  checkpoint, SHA-256
  `8686af4db642744b117ea3082d70de7dcf4e19dc73761d85d960ba8f15a6b2ac`.
  The 110×24×576 embedding artifact SHA-256 is
  `497086f838bb20f9fba3eec68bc290e9dccd1c57a2ce2f4babcb438908e15c81`.
- A trainable projection and temporal convolution consumed all 24 frame
  embeddings and jointly learned the main field-goal/free-throw target plus
  eight causal phase targets. Hyperparameters were selected only by
  leave-one-game-out folds inside the three outer training games. Feature
  standardization used training examples only.
- The convolution screen reaches pooled balanced
  accuracy/precision/recall/F1/AUC
  `0.732/0.741/0.727/0.734/0.765`, with weakest-game balanced accuracy
  `0.694`. Artifact SHA-256 is
  `447699cb2573846425b137d74a9d5e32d2ccdb709c787f48988755820d26f37b`.
  Two held games reach `0.900` and `1.000`, but the other two remain at
  `0.719` and `0.694`.
- A second inner-selected convolution/bi-GRU family is worse at pooled
  balanced accuracy/F1 `0.695/0.692` and weakest-game `0.575`; its artifact
  SHA-256 is
  `b6bb60d511d5454e6f244a49305ba24c8e826fb2edd3fd34405560e842227894`.
  Both screens record `accepted=false`; no checkpoint or runtime model was
  created.
- Both extraction and training ran under the local resource guard with zero
  breaches. Extraction peaked at 679,870,464 bytes process-tree RSS, 85.4%
  system memory and 23.6% CPU. The wider training screen peaked at
  723,615,744 bytes RSS and 77.6% system memory; maximum CPU across the two
  training screens was 27.7%.

This improves the strongest weakest-game score for the reviewed
free-throw/live-field-goal state from `0.656` to `0.694`, but remains far below
the `0.85` gate. Compact embeddings and rejected screen evidence are retained;
no external download, promoted weight or HOU–ORL blind access occurred.

## 2026-07-30 dense geometry and motion fusion

- Reused the four existing, source-video-bound perception stacks: 5 FPS
  BoT-SORT player boxes, 5 FPS rim boxes and candidate-window 10 FPS basketball
  boxes. No detector or source video was rerun. The builder loads one game at a
  time, verifies the four video SHA-256 values against the causal review plan,
  then releases the expanded JSON before loading the next game.
- Compressed 110 events × 24 positions into 44 normalized features covering
  player counts/centroids/spread/team separation, rim and ball
  visibility/coordinates/size, ball-rim and player-ball distances,
  rim-relative formation and adjacent-frame motion. The 2.6 MB artifact
  SHA-256 is
  `36589012299fd314e89b34abb0276fba77ed0db349b1cf8742e7d8c67899a457`.
- Coverage across 2,640 positions is 2,589 player-detection frames, 1,618
  visible-rim frames, 1,728 frames inside the candidate-window ball sampling
  grid, 975 visible-ball frames and 790 simultaneous ball/rim frames. At least
  one ball is visible in 105/110 events and at least one rim in 109/110.
- Repeating the existing 39 aggregate reason features at every temporal
  position is harmful: pooled balanced accuracy/F1 is `0.676/0.673` and the
  weakest game is `0.531`. Artifact SHA-256 is
  `abc0dfe6a0f4d94e78c95f81377c2eb183b07371463a1daa5a747d5461eadac7`.
- True per-position geometry improves pooled balanced
  accuracy/precision/recall/F1/AUC to
  `0.748/0.719/0.836/0.773/0.794`, but weakest-game balanced accuracy is
  `0.688`, slightly below the visual-only `0.694`. Artifact SHA-256 is
  `1143681b1aaad3ac5195ce3bacac594c970745543006e6bea562bd4837e83785`.
  A fixed 50/50 average of the two independently OOF probability streams also
  fails, with weakest-game balanced accuracy `0.656`.
- All guarded builds/screens recorded zero breaches. Geometry construction
  peaked at 498,466,816 bytes RSS, 82.6% system memory and 23.3% CPU; geometry
  screening peaked at 667,222,016 bytes RSS, 78.7% memory and 25.3% CPU.

All three fusion paths are rejected and no runtime checkpoint is created.
The next data requirement is denser, more continuous basketball visibility:
only 36.9% of the 2,640 reviewed positions currently have a detected ball, so
coordinate features cannot yet express a stable release-to-rim trajectory
across every broadcast.

## 2026-07-30 full causal-window basketball detection

- Reused the retained CC BY 4.0 E-BARD/BODD YOLOv8n weight
  `dfe3534d51bb21024d1a400c37f0c1fbf0c8b96ea9a56a5f3cb5454813bfd641`
  only on the four development videos and exact sealed-plan `-4s..+4s`
  windows. The new verifier rejects unplanned/blind sources and mismatched
  video, model, event, window, sampling, completeness or artifact hashes.
- Sampled 8,760 frames and produced 4,579 ball boxes plus 1,236 tracks. The
  four perception SHA-256 values are
  `61e9fadcb4b237e8490bbaf4c1ffde6455772c94fb70b93e4d1cd9519000ba29`,
  `6fcbbc32fd7c699ceadc7926ae30f5c3a5450b3b39dd9c30a279ca898af1bfe8`,
  `a5c27f225fad9951e07939743a3845c883d9030607ba20c006e8b3491d97052f`
  and `df325f3b2f7a11720be8313a6f1a1894b7504ea101236f92d073565a133f35cb`.
- Ball sampling availability rises from 1,728/2,640 to 2,640/2,640
  positions, but actual visible-ball coverage rises only from 975 to 1,047;
  simultaneous ball/rim coverage rises from 790 to 862. The new geometry
  artifact is
  `51a3f7f8a2e383ef96d797811b61f0b5c3455748fe099e26cdf73cf12f91c35e`.
  Its strict held-game screen reaches only `0.701/0.742` pooled balanced
  accuracy/F1 and `0.625` on the weakest game.
- A supported-track variant keeps tracks with at least two real observations
  and uses their bounded interpolated centers. It covers 1,068 ball positions
  and 890 ball/rim positions; artifact SHA-256 is
  `9779f5215962b4db3d21825b993d0652055cc3ff4e0193bdacfc6a48ae654cab`.
  Screening recovers to `0.739/0.767` pooled and `0.656` weakest-game, still
  below the previous geometry result of `0.688`.
- Both routes are rejected and create no checkpoint. Eight guarded
  build/screen runs exited zero with no breach; combined peaks were
  765,100,032 bytes process-tree RSS, 83.5% system memory and 40.0% CPU.

The experiment rejects insufficient window coverage as the primary
explanation. The remaining data bottleneck is BODD small-ball recall and
false positives across broadcast domains, followed by rim/court-constrained
normalization. HOU–ORL G1/G2 remain sealed.

## 2026-07-30 MUVY small-ball audit and transfer screen

- Added the official [MUVY Zenodo release](https://zenodo.org/records/13883315)
  as a CC BY 4.0, offline-only broadcast small-ball candidate. Instead of
  downloading the 7.76 GB ZIP, the bounded ZIP64 range importer verified the
  7,758,643,869-byte archive declaration and downloaded only 13 basketball
  videos, 13 detection files and 13 metadata files. The 39-file, 77,193,749-byte
  subset manifest has artifact SHA-256
  `efeb99853caea236e2b725714d34ae4f949c55a3c2664c26557f4b0a71ad154c`.
- MUVY's proposed `sports ball` labels were not accepted as ground truth.
  The geometry screen produced 373 candidates; Codex inspected all 11
  hash-bound raw-frame contact sheets and sealed 177 `valid_ball`, 190
  `false_positive` and six `uncertain` decisions. The review artifact SHA-256
  is `60b2cc4dfef88ff578fcb3e8b959787e382208a5422c02a8a005898d16e3b5c6`.
  False and uncertain boxes were excluded.
- The cleaned YOLO set contains 169 frames and 177 basketball boxes. It splits
  by complete source event, not frame: 138 training frames and 31 validation
  frames. Dataset artifact SHA-256 is
  `2faf3ad15f54b8969f78628dc1ce3d6b214285a87fd2016a172bfad334d43e06`.
- Direct BODD fine-tuning is invalid with the current training stack because
  the retained inference checkpoint is fused: it exposes 127 state-dict items
  while the current YOLOv8n training graph requires 355, so only 70 transfer.
  A standard COCO YOLOv8n route transfers 319/355 and learned, but early-stopped
  at epoch 17. Its best validation P/R/mAP50/mAP50-95 was
  `0.424/0.182/0.217/0.074`, worse than frozen BODD on the exact same held
  event at `0.452/0.364/0.232/0.109`.
- The new detector is rejected before E-BARD or four-game causal screening.
  All eight failed `.pt` files were moved to macOS Trash; compact arguments,
  metrics, resource logs, source subset and reviewed labels remain
  reproducible. The original BODD and both EBQwen formats remain retained.
- A second CC BY 4.0 candidate,
  [Basketball Broadcast v2](https://universe.roboflow.com/peppes-project/basketball-broadcast-fwdpq/dataset/2),
  declares 1,853 images and 11 classes, but export requires a user-provided
  Roboflow login/API key. AGU did not bypass authentication or download it.

Across 144 resource samples, the complete import/review/materialization/train
screen had zero breach samples; peak process-tree RSS was 898,318,336 bytes,
system memory 86.4% and CPU 73.6%. This negative result narrows the next data
need to higher-quality, game-diverse small-ball labels with explicit hard
negatives. It does not train the full AGU model or start HOU–ORL blind
inference.

## 2026-07-30 whole-game E-BARD + MUVY detector screen

- E-BARD's published frame-level split leaks game identity: 55 of its 60
  canonical NBA game IDs occur in train, validation and test. AGU rebuilt the
  split deterministically by whole game as 44/8/8 games and retained all
  no-ball frames as hard negatives. The source remains the official
  [CC BY 4.0 E-BARD release](https://huggingface.co/datasets/GabrieleGiudici/E-BARD-detection).
- The combined hard-linked dataset contains 1,969 images and 1,673 basketball
  boxes. Train/validation/test contain 1,458/271/240 images; MUVY contributes
  only to train and validation, while test is eight unseen E-BARD game IDs.
  Its sealed artifact SHA-256 is
  `894dc1c4f5d23314680eb00daf2581a3cf9839bc2c16c186bd6a8fa296140f17`.
- COCO YOLOv8n training stopped at epoch 30 with epoch 21 selected. Validation
  P/R/mAP50/mAP50-95 is `0.845/0.586/0.682/0.326`; whole-game unseen E-BARD
  test is `0.801/0.729/0.771/0.375`; held MUVY is
  `0.605/0.303/0.268/0.129`. The frozen BODD reference is stronger on the
  same E-BARD test at `0.854/0.733/0.814/0.425`, but that checkpoint's
  historical training included all E-BARD games and is not a fair unseen-game
  comparator.
- The detector fails the first real-video causal gate decisively: zero ball
  boxes and zero tracks across 405 LAL-BOS enrollment samples at the frozen
  10 FPS, 704 and confidence 0.1 protocol. The remaining three games were
  skipped because worst-game recall could no longer reach 0.85.
- The rejected best/last and smoke best/last checkpoints were moved to macOS
  Trash, while data, labels, manifests, CSV metrics and resource logs remain.
  Across 1,417 samples the resource guard recorded no breach; peak memory was
  85.7%, peak CPU 66.7% and peak process-tree RSS 1,034,665,984 bytes.

## 2026-07-30 Roboflow/Hugging Face RF-DETR candidate audit

- Roboflow Basketball Player Detection 3 revision 18 is CC BY 4.0 and contains
  654 images (464/96/94 train/validation/test). The mirrored COCO metadata
  contains 608 ball boxes, but all three splits draw from the same three NBA
  games, so it is not a game-independent acceptance set.
- The older 1,196-image revision expands about 1,140 training images through
  augmentation and also repeats the same two loose source groups across
  train/validation/test. It is unsuitable as independent evidence.
- The Apache-2.0 Transformers RF-DETR Nano model at immutable revision
  `46c33088c790670a7e81e21e753fa368b2d77a70` has a 120,659,816-byte
  `model.safetensors` with SHA-256
  `c141ac1f05ffce2ac678406df34143efb03d3470f32a6817505de3fdb8b76c35`.
  Its model card reports overall mAP `0.4681`, mAP50 `0.7416`, but ball mAP
  only `0.1805`, and notes limited team diversity.
- On 405 allowed LAL–BOS causal samples it proposes 900 ball boxes, but a
  deterministic 72-candidate stratified review finds 25 valid, 45 false and
  two uncertain. Weighted precision is about `0.331..0.351` at threshold
  `0.1` and only `0.667` at `0.6`; confidence thresholding cannot meet the
  0.85 gate.
- The 121 MB weight was removed during the 2026-07-31 disk cleanup after the
  screen was rejected; its immutable revision, SHA and small metadata remain as
  audit evidence. No dataset images were downloaded. Runtime remains unchanged
  and blind games were not accessed.

## 2026-07-30 E-BARD-to-RF-DETR candidate verifier transfer

- Reused the existing hash-bound 3,675-row E-BARD candidate manifest:
  1,348 positive boxes, 2,327 hard negatives and 60 game groups. No new
  dataset or model weight was downloaded.
- Exact-box MobileNetV3-small embeddings plus detector confidence reach
  game-held OOF precision/recall `0.874/0.851`, but the fixed model/threshold
  transfers to the sealed 72-item LAL–BOS RF-DETR review at only weighted
  precision `0.559..0.615` and recall `0.614..0.636`.
- Visual-only transfer is worse: weighted precision `0.576..0.657`, recall
  `0.441..0.474`. This confirms candidate-generation domain shift rather than
  a confidence-calibration-only defect.
- Neither classifier was saved. The next data requirement is same-detector,
  broadcast-diverse hard-negative supervision from a separate development
  game, while LAL–BOS remains external evaluation.

## 2026-07-30 same-detector ATL–CHI hard-negative review

- Generated 2,014 RF-DETR candidates from 729 frames in nine allowed
  ATL–CHI enrollment causal windows. The source artifact SHA-256 is
  `2440ea967dd37d4c7bd9042adc9ee6d78c5489014e7d95b5dcc34c7917e9693d`.
- Deterministically sampled 18 candidates from each of six confidence bands.
  Codex reviewed only raw-frame red boxes and enlarged local insets, sealing
  29 visible balls, 68 false positives and 11 uncertain rows. Event outcome,
  identity and statistic labels were forbidden.
- The 97 determinate rows are too sparse across nine windows. Exact-crop
  MobileNet transfer reaches only `0.646..0.676` precision and
  `0.677..0.688` recall on the game-disjoint LAL–BOS review. A 2× crop and
  geometry/confidence diagnostics collapse to retaining every target row.
- The reusable data need is now broader same-RF-DETR supervision across
  multiple games/windows, not more E-BARD examples or threshold tuning.

## 2026-07-30 second ATL–CHI RF-DETR review and multi-source transfer

- A second opened ATL–CHI benchmark contributes 4,424 RF-DETR candidates and
  1,135 tracks across 2,610 decoded frames in 28 causal windows. Its sealed
  perception artifact SHA-256 is
  `b524c224b19d622d710467a0a357b58e3b3903d86e6068e280af300a534f3069`.
- A deterministic six-band, 108-row Codex review contains 42 visible balls,
  52 false positives and 14 uncertain rows. Review was limited to box validity;
  event outcome, identity and statistics were forbidden. The sealed review
  SHA-256 is
  `11e6e51141ac2861085cf9b9978572966f360f37f434aa6ffa8fa6592dc8be3f`.
- Combining both ATL–CHI sources yields 191 determinate examples
  (71 positive, 120 negative) in 36 causal-window groups. Only those reviewed
  source detections are decoded for training; LAL–BOS remains a disjoint
  900-candidate/72-review external set.
- Exact-crop visual transfer reaches only lower-bound external P/R
  `0.547/0.770`; visual plus detector confidence reaches `0.542/0.735`.
  Track geometry alone reaches `0.382/0.965`, while visual plus track geometry
  reaches `0.572/0.766`.
- The final screen artifact SHA-256 is
  `6acf45243fc0e4f4856935736d81cbb53a94faa182b53b9d3323ad65155035d8`;
  `accepted=false`. No checkpoint or runtime path is promoted. The retained
  RF-DETR weight remains an offline proposal source only.

## 2026-07-30 EBQwen candidate-panel development probe

- Added a label-hidden, hash-bound panel protocol over the existing
  ATL–CHI benchmark review plan. Two candidates are selected evenly from each
  of six confidence bands without reading review decisions.
- Each 960×540 panel contains the raw broadcast frame, one red candidate box
  and a yellow-bordered enlarged inset. Prediction provenance binds the panel,
  prompt, immutable EBQwen revision
  `c6a93cbb325f9d20236f85bcaba7827a2808443e` and retained MLX 4-bit weight
  SHA-256
  `f2b3cf74e088a06b7bf1c6462664ca13f95e1c1c8be4c6c8a0e469f68dd1d72c`.
- Both the conservative prompt and one development-only forced-binary
  revision return `uncertain` for all 12 examples. The final weighted
  precision and recall are both zero; evaluation artifact SHA-256 is
  `7fc6aecd75486a74a5773a9681e53d7f508f3d25dafcf92bcbe8f6bec2bdab80`.
- The development gate fails before LAL–BOS inference. No new weight,
  checkpoint, external dataset or runtime dependency is created. EBQwen is
  retained for other independent-video research, but this candidate-panel
  prompt is rejected.

## 2026-07-30 frozen RF-DETR query verifier

- The current official Transformers object-detection task guide describes
  full RF-DETR fine-tuning with COCO-format frame annotations and notes that
  full training requires a GPU:
  <https://huggingface.co/docs/transformers/en/tasks/object_detection>.
  The model output contract is documented at
  <https://huggingface.co/docs/transformers/en/model_doc/rf_detr>.
- AGU's 191 reviewed rows are candidate-validity labels, not exhaustive
  per-frame COCO annotations. Full detector fine-tuning would silently treat
  unreviewed balls as background, so this screen freezes the pinned detector
  and binds each reviewed export back to its original decoder query.
- Features contain the 256-dimensional final query state, three raw logits
  and four normalized box coordinates. LAL–BOS review labels are opened only
  after all source OOF fits, thresholds and target scores are fixed.
- The best external lower-bound P/R is only `0.659/0.859`; query context plus
  track geometry reaches `0.599/0.937`. The canonical result hash is
  `3624e90b7be00ee1546ce76c8f313b018c7cdf4ec7c722d2e1d1d27b81b657a5`.
  No classifier checkpoint is saved and the RF-DETR weight remains an
  offline-only candidate proposer.

## 2026-07-30 CHI–UTA same-detector supervision

- Reused the already opened CHI–UTA development video and its hash-bound
  anonymous candidate windows; no new media or model was downloaded.
- The local Transformers RF-DETR processes 8,004 samples in 180 shot-candidate
  windows and emits 13,108 ball proposals. Perception canonical SHA-256 is
  `41bfd2d89e165d8eed00c6ed2ef12844052f5dcca1cdb7138be7f541b5fe2200`.
- A deterministic six-band, 108-row offline Codex review contains 50 visible
  balls, 56 false positives and two uncertain rows. Only box validity was
  labeled; event outcome, identity and statistics were excluded. Review
  canonical SHA-256 is
  `1477efbfeda8e2007482d878c19d09f8debee20d71d41db23f5004cfcc3db59c`.
- Three-source training now contains 297 determinate rows, but the best
  LAL–BOS lower-bound P/R remains `0.660/0.937`. More single-frame labels
  improve recall without solving precision; next data extraction must retain
  multi-frame query trajectories and a fourth game for external evaluation.

## 2026-07-30 LAL–BOS Game 1 hard-negative review and temporal query closure

- The additional opened source is 2008-06-05 LAL–BOS Game 1,
  `BdcP-XwUCk8.mp4`, raw SHA-256
  `8c37eba7483874743b848f062ce74956d8a65d012efd7b11fdc1a455b39a219a`.
  It is a game-disjoint holdout from Game 2, but not a new broadcast domain.
- Local Transformers RF-DETR processed 5,016 10 FPS samples in 58 merged
  causal windows and emitted 10,566 proposals plus 2,441 tracks. Perception
  canonical/file SHA-256 values are
  `647e7a49581fd907b0fc9e857d2028c8d116acd721c602b13eaae36011582894`
  and
  `f5d8a12479957822b66b68a5f465a6c2413c76f03b0de9165437612d9e61b815`.
- A deterministic 108-row offline Codex review contains 20 visible physical
  basketballs, 86 false positives and two uncertain rows. Broadcast graphics,
  player heads, rim hardware and crowd objects are negatives. Event outcome,
  player identity and statistics were excluded. Review canonical/file
  SHA-256 values are
  `c4bd0291237837373e4d52a06879d5ae395c275a8312e07640e76812f33a0c4d`
  and
  `4c50fdd598bba375ad8b8bdf1898e77cc113acb6d0ad0906079615f1c4cbe6cc`.
- Track-sequence features do not improve the three-source Game 2 result:
  temporal-only reaches lower-bound P/R `0.611/0.937`, temporal full-query
  fusion `0.637/0.902`, versus `0.660/0.937` without temporal statistics.
- On Game 1, all frozen three-source variants fail; the best lower-bound P/R
  is only `0.372/0.629`. Adding Game 1's 106 determinate rows and diagnosing
  on the already opened Game 2 still reaches only `0.665/0.937`.
- The frozen-query verifier route is closed. The review remains reusable
  training data, but no classifier checkpoint or runtime path is promoted.
  Future data must provide exhaustive, game-diverse small-ball boxes suitable
  for detector adaptation, or support rim/court-constrained trajectory
  modeling.

## 2026-07-30 entity-interaction follow-up

The BasketEvent/PlayNet repository and Hub revisions remain unchanged and
license-free, so AGU did not download its 1.81 GB checkpoint, trajectory set or
source code. The paper's interaction decomposition was used only as research
guidance for an independent screen over already local, licensed development
assets.

AGU compressed existing four-game player/rim/ball detections into a 3.08 MB,
110-event entity-relation artifact. The resulting visual+relation model failed
the strict game-held gate at worst-game balanced accuracy `0.656`; a fixed
fusion diagnostic reached only `0.688`. These compact generated artifacts are
retained for reproducibility, but no external dataset, media or model was added
and no runtime dependency was promoted.

## 2026-07-30 BARD embedded-validation event-state screen

The official [BARD project page](https://bodai.unibs.it/basketball-action-recognition/bard/)
and [repository](https://github.com/GabrieleGiudic/BARD) were fixed at revision
`add4109bf8b2034a32f3b6a83fa0d8c0ae638473`. The repository contains 201
actual validation MP4 blobs under `validation/2024/multi` and
`validation/2025/multi`; this corrects the older catalog statement that BARD
contains no media. It does not change the rights boundary: CC BY 4.0 covers
the released repository and annotations, while redistribution of the
underlying NBA broadcast clips still requires separate review.

AGU selected 38 exact Git blobs, balanced as 19 live field goals and 19 free
throws across 33 games. The sealed subset is 252,967,937 bytes, every blob has
a unique SHA-256, and every file passes H.264/1280x720/60 FPS probing. The
selection plan canonical SHA-256 is
`38205c205f1aba5c013b10023791bba746f5477b5cbe687fe558d09732da8f25`;
the materialized subset SHA-256 is
`8a9f869a264e9caff999c51edf666b6d04043ae4418a6ec588432f0f89dfdeec`.
Only the 241 MiB bounded subset and a 108 KiB blobless metadata repository are
retained.

The first NBA event-page downloader was rejected before training: 48 distinct
resolved URLs returned the same 31,580,089-byte MP4 and identical ETag/hash.
The duplicate-media guard now fails closed, and the invalid 1.41 GiB download
directory was deleted. Its URL-resolution and invalid-download manifests
remain as audit evidence.

MobileNetV3-small sampled 24 normalized full-clip positions. A source-only,
game-grouped five-fold screen selected quarter pooling with `C=0.01` and
reached BARD OOF balanced accuracy/F1 `0.974/0.974`. Target labels were not
loaded before the 108 AGU development predictions were sealed. Delayed
evaluation then failed decisively: all 108 rows were predicted as free throws,
giving balanced accuracy `0.500`, F1 `0.675`, AUC `0.644`, and worst-game
balanced accuracy `0.500`. No checkpoint or runtime adapter was promoted.
Because these four development games are now open, they may support diagnosis
but not a new promotional tuning loop. HOU-ORL G1/G2 remain sealed.

## 2026-07-30 SpaceJam cleanup and exact-candidate fusion audit

The local `spacejam-action-recognition.zip` was audited before deletion:

- exact size 715,601,629 bytes and SHA-256
  `96ce912eb478cf81af6d13de7e86ad54ebb56f32f01edb1a3440d3d6b8b32b8e`;
- pinned repository revision
  `441cdded139b905b815e722927fe330b0f236d08`;
- 37,085 annotations, 37,201 MP4 entries, 4,528 flipped clips and 11,126
  test keys;
- ten single-player action labels, without made/missed result, free throw,
  replay, event timing or actor grounding.

AGU v3 was already trained on this source, so the 682 MiB archive could not
resolve the current broadcast event-state gap. It was irreversibly deleted;
the compact
`analysis_outputs/public_research/spacejam_archive_audit_v1.json` remains, and
the source can be downloaded again from the official
[SpaceJam repository](https://github.com/simonefrancia/SpaceJam). Media rights
still require separate review from the repository's MIT code license.

An exact 32-row, four-game development plan paired AGU v3 with frozen EBQwen
MLX 4-bit predictions. AGU v3 produced 65 player clips over 22 available rows,
but none had `shoot` as argmax; ten rows remained unknown. Maximum shoot
probability has AUC `0.511719` and AP `0.564260`. EBQwen alone reaches
precision/recall `0.571429/1.000000`; fixed base-only, VLM-only, AND and OR
rules are all ineligible for promotion. The base artifact canonical/file
SHA-256 values are
`ae4231450f3686773536d0fd372ac1aa30fbe4fb13e180b1e638e02648d31a70`
and
`3abab80a8d1507bedbc3e297c129aecc87b60eb4ca76d017e9b06c6f91af99c4`.
No model, dataset, checkpoint or runtime dependency was added.

## 2026-07-30 sparse-gated temporal rescreen

Three current primary sources were checked before any download:

| Source | Current official finding | Decision |
| --- | --- | --- |
| [yerx/bb](https://huggingface.co/datasets/yerx/bb) | 200 free-throw made/missed videos, 1.72 GB, but no dataset card, license or media provenance. | Catalog only; reject media and annotation import. |
| [MUVS](https://zenodo.org/records/20708683) | Zenodo record metadata declares CC-BY-4.0 and exposes 5,039 three-second fragments plus 258 period videos; the packaged README still contains `License [TO BE ADDED]`. Camera selection is human annotated, but basketball event semantics are not supplied. | Use only bounded, hash-bound source frames with manual training-only event-state labels; preserve attribution and recheck the README/record discrepancy before redistribution. |
| [BasketEvent](https://github.com/zhangyu2003/BasketEvent) | The official repository now documents SAM3/Qwen cleaning and TimeSformer event training, while its Hub asset is 6.73 GB. Neither official repository declares a license and the Hub has no matching source videos. | Continue paper-level architecture research only; no code, annotation, checkpoint or media import. |

The sealed source catalog now contains 31 entries and canonical SHA-256
`d5c4ad0701e35f403da7d1cf1ff689e9a82d94dd537963a49477cc1a93c82fe3`.
No external bytes were downloaded.

AGU independently added a sparse gated temporal pool to the existing
24-position MobileNet plus entity-relation screen. Under unchanged nested
whole-game holdout, expanded conv/GRU/gated selection reaches balanced
accuracy/F1 `0.703/0.719` and worst-game balanced accuracy `0.583`.
The rejected artifact canonical SHA-256 is
`a6c7b41a1a09ca047217b07573999c1da4bc1dde977e6cc9cf0b072c0717f10b`.
No checkpoint or runtime data dependency was created.

## 2026-07-30 independent formation VLM screen

- No new dataset or pretrained checkpoint was downloaded. The screen reuses
  the existing four opened development broadcasts and their sealed 110-row
  causal phase plan; both HOU–ORL blind-game hashes remain excluded.
- The derived label-hidden plan fixes two raw frames at `-1.0s` and `0.0s`,
  384px image width and exact source video/bundle/event identities. Its
  canonical SHA-256 is
  `c06703753d3ca58d268fffbbeb724dd212dbd9f39c17691c961b5e75959934e2`.
- Local Ollama assets were pre-existing `qwen3-vl:4b` and `qwen3-vl:2b`
  packages. The 4B model is resource-infeasible under the AGU guard; the 2B
  model completed the frozen screen but failed the delayed-label gate.
- These model packages and all generated predictions remain local,
  development-only and non-runtime. The opened 110 labels cannot be reused for
  prompt/offset selection in a promotional experiment.

## 2026-07-30 MUVS bounded source-state audit

- The official Zenodo record declares `CC-BY-4.0`; the downloaded package
  README still says `License [TO BE ADDED]`, so the catalog records both facts
  and requires a release-time rights recheck.
- The 9,742,259-byte metadata archive has MD5
  `8fd1d426e0b4308655ea71884d1b0c8e` and SHA-256
  `df25517793df22f9f57eef1f8f0ffba131d98b5958313110a5d9a7592820912b`.
  Twelve basketball events contribute 195 period videos; full video download
  was rejected at roughly 174.4 GB.
- AGU fetched only two 1280-pixel frames per planned sample by remote seek.
  The base plan covers four balanced timestamps per event. A second bounded
  plan adds twelve timestamps for the NBA and WNBA events only.
- Codex reviewed public-source images offline, never AGU runtime games. The
  two sealed reviews contain 48 and 24 decisions. Cross-plan frame hashes
  found eight duplicates and exposed one first-pass label error; the source
  decision was corrected before embeddings were resealed.
- After exact duplicate removal, the combined source screen contains 60
  determinate examples over 12 events: 11 free-throw and 49 non-free-throw
  rows. Leave-one-event-out MobileNet features reach balanced accuracy
  `0.641`, ROC AUC `0.796`, free-throw recall `0.364`, and specificity `0.918`.
  The route is rejected; no model or source artifact is runtime-consumable.
- The same frozen rows were subsequently screened with E-BARD/BODD
  player/rim detections and fixed local DINOv2 views. Geometry reaches only
  `0.712` balanced accuracy and local DINOv2 reaches `0.686`; both have
  worst-positive-event recall `0`. These are rejected training-only audits,
  not new MUVS labels or runtime dependencies.
- The official
  [DINOv2 repository](https://github.com/facebookresearch/dinov2) and pinned
  [facebook/dinov2-small revision](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056)
  declare Apache-2.0. Its 84 MiB temporary local checkpoint was removed after
  the local-view screen failed; only compact embeddings and provenance hashes
  remain.

## 2026-07-30 MUVS professional extension

- A source-coordinate-disjoint extension adds 118 NBA/WNBA samples. Two
  out-of-range source positions were repaired before sealing. A stale
  pre-repair frame cache for `muvs-state-0077` was detected by modification
  time and independent re-extraction; the corrected frame hash is
  `5ad6c30ec6ce8ebfa2046c5d77e0a4b2b20a9a7fa47a83630f30263b42c04ca9`.
- The final plan, 236-frame manifest and public-source-only review hashes are
  `b952fcd7b3effec3e30614519e36d9f5d69aea30bcd4f4156def2444aac132f8`,
  `3a5772ae1d852382e516a2eb18ae38aa0dcf0ac294d9d1ba38f87cefc5d6a1d0`,
  and `df6ed2f32a583d1ea61418b11af73e9a35b894dc57aaac295e65cb1466b94b2d`.
  The review contains 75 live-play, 27 dead-ball and 16 free-throw rows.
- BODD detections and DINOv2 embeddings have canonical SHA-256
  `cb48a77e878f0b2a10183bb6294b51293e7aab6e330d409c521ba2d19025ff50`
  and `c4f40c205da2f831197db3538a1172e38cde798c0174ee76fd65ca14dbffa6a2`.
  Combining all compatible MUVS artifacts yields 466 determinate examples,
  54 positives and 412 negatives.
- Static DINOv2 context falls to `0.752` balanced accuracy and BODD geometry
  reaches only `0.646`; both have worst-positive-event recall `0`. The larger
  professional sample therefore rejects, rather than promotes, both static
  routes.
- The pinned DINOv2 checkpoint was downloaded again for this extension and is
  currently retained for a bounded short-temporal follow-up. No MUVS artifact
  is runtime-consumable, and the HOU–ORL blind games remain mechanically
  excluded.

## 2026-07-31 MUVS pair-motion screen

- The current official Zenodo record still exposes the 5,039 fragment/258
  period-video dataset and download assets. Local provenance continues to
  preserve both the record-level `CC-BY-4.0` declaration and the packaged
  README's unresolved `License [TO BE ADDED]`; redistribution remains blocked
  pending a release-time rights recheck.
- A fixed directed DINOv2 candidate adds signed temporal change to the existing
  two-frame local views. It falls to `0.534` balanced accuracy and `0.536`
  ROC AUC, below the prior static-context result, and is rejected.
- A separate OpenCV-only artifact uses ORB/Hamming correspondences, RANSAC
  homography compensation, and grid-level residual/Farneback flow summaries.
  Homography passes for 459/478 source pairs. The four feature artifact hashes
  are `70099046d06933cb39ef451be28c46910a43c9a5940ee52452b932b6a61fc4d5`,
  `3f6ae00cda5ef351b23d3fb8c6392139fd52ecd47a06734feb6e765f91d60560`,
  `b134a63610b54553d52a75058f897efccbb3f61c4b919ff4faca8cbc197c6f53`,
  and `cda2be51ff7034c450d51c42bb191bd908b67956815c67b2029e56fa1ec8b00b`.
- Exact pair deduplication leaves the same 466 determinate rows. Fixed
  leave-one-event-out motion screening reaches only `0.539` balanced
  accuracy, `0.558` AUC, `0.407` free-throw recall and `0.670`
  specificity; worst positive-event recall remains `0`. Its canonical
  SHA-256 is
  `d070c0501c144054fd9d5311f86bcfe5295a791dda5f9107a2b593d09e9383ca`.
- No new media or weights were downloaded for this screen. Compact features
  and resource logs are retained; no checkpoint or runtime path is promoted.
 Further MUVS work requires a pre-frozen denser sequence protocol rather than
 additional selection on the opened two-frame labels.

## 2026-07-31 MUVS dense-sequence contract

- The official Zenodo API was rechecked after the pair-motion screen. The
  record is open and declares CC-BY-4.0, while the packaged README still says
  `License [TO BE ADDED]`; AGU therefore keeps the raw-video redistribution
  boundary unchanged. The downloaded metadata package is only 9.7 MiB and its
  SHA-256 is `df25517793df22f9f57eef1f8f0ffba131d98b5958313110a5d9a7592820912b`.
- AGU added a separate label-hidden five-frame contract with fixed offsets
  `0.25, 0.75, 1.25, 1.75, 2.25` seconds inside each three-second fragment.
  It excludes every source coordinate used by the prior MUVS plans. One
  event's 95 coordinates were already exhausted, so the new diagnostic plan
  explicitly covers the remaining 11 events and 22 unused coordinates. Plan
  SHA-256:
  `080315811b0bed4628606800d15955899f90d747cd24eb5d5e8a1aa5a4123e3d`.
- A resumable materializer now performs one remote seek per sample and emits
  five hash-bound 1280-pixel PNGs. One public-source smoke sequence was
  visually reviewed as `live_play`; its plan/frame/review hashes are
  `889999edafd0d5dd3d2a9581b608e45d9d9ed7088094385ada59e48fae7e7ef2`,
  `d97ab00125b2749045637c7d04edb76dca32fbd3df3017d4ac003b87e54e6a6d`, and
  `372b3271b466d70d3786e719d09bbecea16b569d71d74099b91ddd93842afe6a`.
- The full 110-frame materialization is not yet a model screen: remote random
  seeks are too slow (the one-sample smoke took about 19 minutes), and the
  contract remains offline-only. No AGU runtime, VLM fusion, checkpoint or
  blind-game asset changed.

### 2026-07-31 MUVS sequential-download development slice

- To avoid the remote random-seek bottleneck, two complete public period videos
  were downloaded temporarily, decoded locally, and removed after extraction.
  The slice contains two previously unused coordinates (Chicago Sky–Seattle
  Storm and Naperville women's youth match), five fixed frames per coordinate,
  and 10 hash-bound 1280-pixel PNGs.
- Plan/frame/review SHA-256 values are
  `255a8b4285d7a112d84ec8c8304ecc97a48215110f9cc0ab4a61a4cd611f87a8`,
  `0c0ce01d12627faf64db2484e8601f8f23f42fea24959d48c954a88a9d3c3386`, and
  `55295352a02bc6b7ad29df0b85d90c780c25194c8635555875169989698d2eac`.
  Offline review states are one `live_play` and one `dead_ball_timeout`.
- The raw MP4s are not retained or redistributed. This is a protocol/decode
  validation slice, not a model screen or runtime input; the sample size is
  insufficient for an accuracy claim.

### 2026-07-31 disk cleanup boundary

- Deleted the two temporary MUVS period videos (about 3.5 GiB), a 115 MiB
  rejected RF-DETR Hugging Face cache, and three unreferenced legacy weights
  (2021 S3D plus two old YOLO files, about 92 MiB total).
- Retained the canonical `.venv`, active `r2plus1d_v3`, BODD/E-BARD,
  Yunet/SFace, YOLO11 pose, both EBQwen weight formats, and all source datasets
  required by the current offline plans. No runtime or blind-game asset was
  removed.

## 2026-08-01 PBP parser repair and new-source gate

The official NBA Games PBP uses `scoreHome/scoreAway = 0,0` on many non-scoring
rows. The training-only PBP alignment parser now forward-fills the last monotonic
cumulative state instead of treating those placeholders as score resets. A
regression test covers a made-shot followed by a missed-shot placeholder. The
existing LAL–BOS artifacts remain at `460/467` and `439/442` mapped rows after the
repair.

Two visible ATL–CHI development games were then aligned without opening any blind
asset:

| Game | PBP rows | Mapped | Score anchors | Artifact |
| --- | ---: | ---: | ---: | --- |
| 2011-05-06 CHI–ATL | 431 | 428 | 100 | `analysis_outputs/public_research/pbp_event_alignment/atl_chi_g0_v1.json` |
| 2011-05-10 ATL–CHI | 405 | 404 | 102 | `analysis_outputs/public_research/pbp_event_alignment/atl_chi_g1_v1.json` |

Read-only joining with the existing 2 FPS BODD perception shows the same structural
limitation: exact-anchor basketball visibility is only `1.2%/1.7%` for ATL–CHI and
`1.7%/2.3%` for LAL–BOS; a ±12-second window reaches only `37.9%/37.4%` and
`37.8%/33.7%`, respectively. This is a visibility gate, not an accuracy claim, so
no detector training, checkpoint, runtime, or blind-game input changed.

The current source catalog now contains 36 entries and records four newly
screened candidates. Zenodo
[Play by Play](https://zenodo.org/records/12698090) has useful ball/player
trajectories but is CC BY-NC and cannot support the open-commercial AGU target.
[VRU_Basketball](https://huggingface.co/datasets/BestWJH/VRU_Basketball) is CC BY-4.0
but supplies dynamic-scene sequences without event/ball labels; the full archive
is not retained and only eight clips were Range-extracted for offline screening.
[VSTAT](https://huggingface.co/datasets/VSTAT-NeurIPS2026/VSTAT)
is a general visual-state benchmark without basketball event/ball-track labels.
[basketball-events](https://huggingface.co/datasets/saveerjain/basketball-events)
is marked research-only, 7.87 GB, and its dataset viewer currently fails schema
post-processing. None was downloaded or added as an AGU dependency.

## 2026-08-01 same-detector broadcast hard-negative review

The opened 2011-05-10 ATL–CHI development broadcast was sampled with the same
Transformers RF-DETR proposer in six confidence bands, 30 candidates per band.
The resulting 180-candidate plan is hash-bound to the perception artifact and
raw video; its internal plan SHA-256 is
`93a9062caf41b0ef3f6aa89540562c4eb00fdf17ea750c492686b605c02468d3`. Codex
reviewed only whether the red box contains a visible physical basketball. The
sealed review contains 55 `valid_ball`, 98 `false_positive` and 27 `uncertain`
rows; review SHA-256 is
`a46062ade1a9b883f2c136dda4c968e836f3f715144a8cc70797c4956f2991aa`.

The offline verifier trained on three disjoint opened games (ATL–CHI G3/G5 and
CHI–UTA G5) and evaluated on LAL–BOS G2. Best lower-bound precision/recall was
`0.542/0.677` for the visual-plus-track-geometry variant; all four variants were
rejected (`accepted=false`). The guard recorded zero stops with peak system
memory/CPU/process-tree RSS `85.1%/62.1%/494,698,496` bytes. No detector
checkpoint, runtime dependency or blind-game asset was changed.

## 2026-08-01 external soccer-ball checkpoint transfer screen

- The MIT-licensed [`hugosanchezgallego/futbol-detector-v1`](https://huggingface.co/hugosanchezgallego/futbol-detector-v1)
  YOLOv8s checkpoint was downloaded as a bounded candidate. Its exact
  `best.pt` SHA-256 was
  `0f80ebd39466d26a883d3e2826a62addf2597006f246d5bbe81e947863431459`.
- On the opened ATL–CHI causal plan, fH (`995` samples) produced `415` boxes
  and `194` tracks at confidence `0.05`, with median confidence `0.083791`.
  The same run at confidence `0.20` produced only `27` boxes and `20` tracks.
  Independent aIgj at confidence `0.20` produced `60/279` boxes and `23` tracks.
- All three runs were CPU/resource-guarded with zero stops and did not access
  blind videos. The candidate is rejected (`accepted=false`), never enters the
  runtime, and the 21.5 MiB weight was deleted after the evidence manifest was
  sealed. Reproducible JSON and guard logs remain under
  `analysis_outputs/public_research/futbol_detector_v1/`.

## 2026-08-01 basketball-specific YOLO11 smoke gate

- A single 5.55 MiB MIT checkpoint from
  [`Lumos-88/YOLO11-fine-tuned-for-basketball-detection`](https://huggingface.co/Lumos-88/YOLO11-fine-tuned-for-basketball-detection)
  was downloaded for a bounded smoke test. The fixed source revision is
  `1f59c9c31abe7f268de300f1f2176324fa130f79` and the `best.pt` SHA-256 is
  `37edc5288522e410a482e2043873e39e14b358efb5212585b7243f01f0edb354`.
- The loaded checkpoint exposes COCO 80 names; AGU maps only `sports ball`
  (class 32) to `basketball`. On 24 fH causal samples at `conf=0.05`, it emits
  `165` mapped boxes and `31` tracks, up to `14` boxes per frame, including
  large edge/scene boxes. This fails the first precision gate, so no full-source
  run was attempted.
- The CPU guard stopped zero times but recorded one low-swap observation
  (`526,450,688` free bytes); no threshold was relaxed. The candidate is
  rejected and the weight was deleted after sealing the JSON/guard evidence in
  `analysis_outputs/public_research/lumos_basketball_v1/`. Runtime, checkpoint
  and blind assets are unchanged.

## 2026-08-01 Play by Play standardized-space trajectory source

- Downloaded the [Zenodo Play by Play dataset](https://zenodo.org/records/12698090),
  which declares CC BY-NC 4.0. The archive is 162,983,009 bytes, MD5
  `caac618508f17e3b08850be1bf24a090`, SHA-256
  `3e9d9cd2179e0a7d3e86bcd0dfe257bd0f0fe00082cf03a9a5d85960cc79c07e`, and passed
  `unzip -t`.
- The actual archive has 381 basketball JSON clips and 56,578 standardized
  coordinate frames, with visible-ball annotations on 11,851 frames. It has no
  raw broadcast images or pixel boxes. `basketball/info.json` names a `match-2`
  test set, but no `match-2` files exist; only `match-0` and `match-1` are present.
  This anomaly is fail-closed for benchmark claims.
- A 66-dimensional offline trajectory summary (ball visibility/position/motion
  plus player position/speed) classified penalty/timeout versus attack/counter
  with two-match hold-out balanced accuracy `0.924/0.929` and ROC-AUC
  `0.961/0.961`. The sealed probe is
  `analysis_outputs/public_research/play_by_play_trajectory_probe_v1.json`,
  canonical SHA-256
  `d2d27af44094f093d99a3ecc4bbaacf7d0f5cbde3cdabb8a59e7bbd513d4da7d`.
- The source is retained only as an offline, noncommercial trajectory
  pretraining candidate (`runtime_consumable=false`, `accepted=false`). It does
  not provide AGU ground truth, cannot replace cross-broadcast ball-box
  supervision, and is not used for blind-game truth or runtime promotion.

For a fixed transfer check, the source summary was mapped to a common 38-dimensional
interface and evaluated directly on the existing 110 AGU causal examples, without
target threshold tuning. `formation_free_throw` versus `live_play` reached pooled
balanced accuracy/F1/ROC-AUC `0.638/0.698/0.670`, with weakest-game balanced
accuracy `0.583`; the route is rejected. The transfer artifact is
`analysis_outputs/public_research/play_by_play_transfer_probe_v1.json`, canonical
SHA-256 `b5c358b08d76ead21298e818bca0adc5a834c74db6fb67b3539ec44074b9f167`.

## 2026-08-01 ATL–CHI full-frame hard-negative detector screen

An offline structural experiment materialized 53 reviewed `valid_ball` frames and 89
reviewed `false_positive` empty-label frames from the opened ATL–CHI broadcast. The
temporal split was 99/21/22 train/val/test; the manifest records artifact SHA-256
`1717eb4a3122bcec68b2b45b2f58028ac172f19e797b50ad69344c8409908031`. A YOLOv8n
full-frame fine-tune ran in the canonical `.venv` (Python 3.11), CPU-only, for 20 epochs.
The guard stopped zero times; peak system memory/CPU/process-tree RSS were
`83.5%/74.5%/1,354,891,264` bytes, with minimum available memory/swap
`2,842,394,624/716,177,408` bytes. Internal validation was only P/R/mAP50
`0.455/0.346/0.320`.

The frozen external lower-bound screen used opened LAL–BOS review frames at `conf=0.25`
and IoU match `0.25`: 2/20 valid-ball frames were recovered, while 13/85 false-positive
frames emitted a detection, giving P/R/F1 `0.133/0.100/0.114`. The evidence artifact is
`analysis_outputs/public_research/atl_chi_hard_negative_detector_external_screen_v1.json`,
SHA-256 `9f9fbcd74e4b843d5512071d44d0f1dae60fbc49a812a2355e324eea1697c4ff`; it is
`accepted=false`. The generated checkpoint was deleted after sealing evidence, and the
route did not change runtime, detector defaults, or blind-game inputs. Dataset images,
labels, args/results, and resource logs remain for audit and future label expansion.

## 2026-08-01 DeepSportRadar Basketball Instants bounded subset

The [DeepSport project](https://ispgroup.gitlab.io/code/deepsport/) and
[Kaggle dataset metadata](https://www.kaggle.com/datasets/deepsportradar/basketball-instants-dataset)
declare CC BY-NC-SA 4.0 for non-commercial research. The full listing is 2,509 files
(about 6.60 GB); AGU retains a bounded eight-arena subset of 107 records (53 positive
and 54 no-ball) with 267 raw image/mask/annotation files (about 454 MB). The source
mask class `3` is the basketball mask; positive selection requires at least 50 pixels,
and the local subset deliberately excludes `_40.png` companions.

The offline materialization is arena-disjoint: 80/14/13 train/val/test, with
`KS-FR-LIMOGES` as validation and `KS-FR-MONACO` as test. Per-record image/mask hashes
are in `analysis_outputs/public_research/deepsport_detector_v1/manifest.json` (SHA-256
`9618598a192303ca44823b99883b397d80ffb2b4eb900d9ef287820e9d9e1338`). A guarded CPU
YOLOv8n run in the canonical `.venv` completed 15 epochs, but internal validation
P/R/mAP50/mAP50-95 stayed `0/0/0/0`; the source-only detector is rejected
(`accepted=false`, `runtime_consumable=false`). Its checkpoints were deleted after
hash capture in `deepsport_detector_train_v1/retention.json`; the source subset and
audit evidence remain for future label expansion and are not a runtime dependency.

A follow-up high-resolution structure screen used deterministic 2x2 edge tiles (65%
of each source dimension, minimum tile dimension 512 px, 50% label-intersection
threshold), producing 428 derived images with the same arena-disjoint boundaries
(320/56/52). The guarded 20-epoch CPU YOLOv8n run in `.venv` reached best internal
validation P/R/mAP50/mAP50-95 of `0.03125/0.38889/0.02115/0.01081`; this is still
well below the 0.85 gate, so the route was not taken to an external frozen screen.
The derived manifest SHA-256 is
`1430324edff6aba1629c5d9ecd42140132314fe4003c956efdb616e42d99d2f4`, and the
checkpoint/resource evidence is sealed in
`analysis_outputs/public_research/deepsport_detector_tiles_train_v1/retention.json`.
The approximately 188 MiB tile cache and both checkpoints were deleted after hash
capture; the source subset remains offline-only and `accepted=false`.

## 2026-08-01 APIDIS bounded multi-view ball source

The [APIDIS project page](https://ispgroup.gitlab.io/code/apidis/) describes a
seven-view professional basketball acquisition with pseudo-synchronised video,
whole-game event annotations, and manual ball annotations. The linked
[APIDIS metadata package](https://www.kaggle.com/datasets/gabrielvanzandycke/apidis-metadata)
is restricted to non-commercial video-signal-processing research and requires
mentioning APIDIS. Its ZIP is 28,559,743,766 bytes, so AGU did not download the
archive wholesale. A central-directory tail audit selected only eight ball
ground-truth files, four quarter-event XML files, seven object XML files,
calibration/trajectory metadata, and seven 60-second 800×600 25-FPS
pseudo-synchronised AVI/index pairs. The exact source ETag, central-directory
coordinates, range audit and per-file SHA-256 values are in
`dataset/public_sources/apidis_ball_metadata_v1/manifest.json` (about 526 MB
on disk).

The camera1 offline detector slice is materialized at
`analysis_outputs/public_research/apidis_detector_camera1_v1/` with a temporal
875/325/300 train/validation/test split over 1,500 frames. Manual centers map to
the nearest 25-FPS frame; unlisted frames remain empty labels. This is a
single-game/single-camera research slice, therefore it is explicitly
`runtime_consumable=false` and `accepted=false`, cannot supply HOU–ORL truth, and
must pass a frozen cross-broadcast screen before any model artifact could be
considered. The source media remain non-commercial/offline-only.

The guarded `.venv` CPU YOLOv8n screen used 800px inputs, batch 1 and a temporal
split. All six completed epochs had internal val P/R/mAP50/mAP50-95 `0/0/0/0`;
epoch 7 was interrupted at about 25% because continuing a zero-fitness route was
not a useful use of local resources. The guard recorded zero stops, with peak
system memory/CPU/process-tree RSS `82.6%/69.5%/831,537,152` bytes and minimum
available memory/free swap `2,988,294,144/560,726,016` bytes. The two identical
checkpoint hashes were sealed and the weights deleted; retention, results and
resource evidence remain in
`analysis_outputs/public_research/apidis_detector_camera1_train_v1/`. No external
screen, runtime promotion or blind-game access followed.

### 2026-08-01 Qwen3-VL 2B strict-release prompt screen

The already-local Apache-2.0 `qwen3-vl:2b` was rerun on the same hash-bound,
label-free 32-window compact plan with an offline-only `strict_release_v2`
prompt (SHA-256 `153193b4b3bf8bec5b8d7f379ba7028831c195e4e678a3c1c57a4f392cd924c6`).
The model saw only four chronological 512-pixel frames per window at context
3072; the prompt and cache fingerprint were sealed independently from the
evaluation labels.

All 32 predictions were `live_field_goal`: TP/FP/FN/TN `16/16/0/0`, precision
`0.500` and recall `1.000`, with every held game at precision `0.500`. This
fails the `0.95/0.85` promotion gate and is worse as a usable gate than the
previous compact result `0.455/0.312`; no runtime or VLM-fusion path changed.
The prediction/evaluation artifacts are
`analysis_outputs/public_research/independent_shot_vlm_qwen3vl2b_compact_ctx3072_strict_release_predictions_v1.json`
and the corresponding `..._evaluation_v1.json`.

A first resource-guarded attempt with a 2 GiB available-memory floor stopped
after model load. The completed evidence run retained the 90% system-memory
cap and used a temporary 1.5 GiB available-memory floor; 82 samples recorded
zero stops, peak system memory/CPU/process-tree RSS `89.6%/80.0%/97,157,120`
bytes, and minimum available memory/swap `1,785,643,008/271,777,792` bytes.
This is a rejected offline prompt experiment (`accepted=false`,
`runtime_consumable=false`), not a complete-game accuracy claim.

### 2026-08-02 EBQwen native strict-release prompt probe

The native MLX/Transformers runner now exposes an offline-only `strict_release_v2`
prompt variant. The prompt text and variant are bound into the cache fingerprint and
sealed prediction artifact (strict prompt SHA-256
`9b10feb8470ec760a2dcf698d9a8066838e665e1f086569418c03a2c0529a392`); the historical baseline remains the default, and no AGU
runtime path consumes the result. The annotation verifier also accepts a derived
resource-probe subset when its label-bearing source plan covers the selected rows.

A four-window prefix was derived directly from the 32-row development plan and ran
with raw video only, native temporal position encoding, 2 FPS and a 151,200-pixel
budget. The local `EBQwen2.5-VL-3B` MLX 4-bit weights are pinned to revision
`c6a93cbb325f9d20236f85bcaba7827a2808443e` and SHA-256
`f2b3cf74e088a06b7bf1c6462664ca13f95e1c1c8be4c6c8a0e469f68dd1d72c`.

All four strict-prompt predictions were `live_field_goal`; delayed evaluation gives
TP/FP/FN/TN `2/2/0/0`, precision/recall `0.500/1.000`, identical to the existing
native baseline on this prefix. The route is `accepted=false`, so no 32-window
expansion was justified. The guarded run had zero stops, peak system memory/CPU/
process-tree RSS `89.7%/48.0%/863,813,632` bytes, and minimum available memory/free
swap `1,776,844,800/387,842,048` bytes. Plan, prediction, evaluation and resource
artifacts remain offline-only under
`analysis_outputs/public_research/independent_shot_vlm_native_strict_release_probe4_*`.

## 2026-08-02 NBA-Identity primary-release audit

The official [NBA-Identity paper](https://arxiv.org/abs/2507.20163) describes 40
full games and 9,726 variable-length clips with player boxes, temporal boundaries,
captions and fine-grained basketball events (including shots, rebounds, assists,
blocks, fouls and turnovers). It is therefore relevant to AGU's event/identity
gap at the paper level.

The authors' [official code repository](https://github.com/Zeyu1226-mt/LLM-IAVC),
however, only documents a TimesFormer feature archive and supplementary files on
an external file host; it does not publish raw game videos in the repository, and
the repository declares no code or dataset license. The public release therefore
does not satisfy AGU's license-and-media gate. No feature archive, annotation
bundle, raw video or weight was downloaded; the source is cataloged as
`accepted=false`, `runtime_consumable=false`, and `paper_reference_only` until
the authors publish explicit terms and a reproducible file manifest.

## 2026-08-02 NBA Rebounds anticipation audit

The [NBA Rebounds paper](https://arxiv.org/abs/2512.15386) is another relevant
lead: it describes 100,000 basketball clips and more than 2,000 manually
annotated rebound events. The paper explicitly says the dataset is not yet
public while NBA permission is pending, so there is no lawful, reproducible
download to add to AGU. It is cataloged as `paper_reference_only` and
`accepted=false`; no clips, annotations or weights were downloaded.

## 2026-08-02 MUVY full-basketball-prefix inventory

Before considering another MUVY download, AGU performed a bounded HTTP-Range audit
of the official [Zenodo archive](https://zenodo.org/records/13883315). The ZIP
central directory reports 333,858 entries; selecting only the basketball prefix's
`detections_info.txt` and `video_info.txt` entries yields 26 text files across five
events and 13 cameras. Parsing those files finds 513 `sports ball` rows across
489 unique ball-frame/camera positions; no additional basketball annotation files
exist in the archive prefix beyond the 13 videos already retained locally.

Only 586,767 bytes of text were range-read and no video bytes were downloaded.
The inventory is sealed at
`analysis_outputs/public_research/muvy_basketball_annotation_inventory_v1.json`
(canonical SHA-256
`483848623315d98b5c55736fdd2782216e743d39aa6ad49ce3f3771ae3684840`). The
source remains `accepted=false` for the current cross-broadcast gate: expanding
MUVY would require unrelated sports or the full archive, not more basketball
supervision. No new MUVY data or weight was added.

## 2026-08-02 Nonlinear ball-candidate calibration screen

To test whether the remaining cross-game ball-candidate gap was only a linear
decision-boundary problem, AGU reused three sealed RF-DETR review bundles (356
determinate candidates: 134 valid balls and 222 false positives over 111 causal
window groups). The screen added bounded logit, ratio and interaction features
over detector confidence, box geometry and track metadata. It used grouped
cross-validation to select the recall-floor threshold and opened the disjoint
LAL–BOS review only after source-only fitting. No video was decoded and no
checkpoint was saved.

The ExtraTrees variant reached source OOF P/R `0.461/0.888` and external
lower-bound P/R `0.456/0.965`; nonlinear Logistic reached external lower-bound
P/R `0.460/0.822`. Both fail the `0.85/0.85` gate and are below the existing
RF-DETR query-context lower-bound precision (`0.660`). The canonical artifact is
`analysis_outputs/public_research/nonlinear_ball_calibration_screen_v2.json`
with SHA-256
`a74564bcf14e830b814d46f08c3086651ee5b95dcb6809608e0b71fffced57e5`. It is
`accepted=false`, `runtime_consumable=false`, and `pixel_decode=false`; no
detector, runtime or blind-game asset changed.

### 2026-08-02 E-BARD ObjectClassification crop-role audit

The official [E-BARD ObjectClassification release](https://huggingface.co/datasets/GabrieleGiudici/E-BARD-ObjectClassification)
is CC BY 4.0 and exposes one 82,522,831-byte `all.zip` at revision
`a0919d0bfbf0ff57502b1f2ec515899cc9c5e818`. Its SHA-256 is
`515b2c5bc1f9f00f4d30eb58a7bcd844045116b9bbe885674bd78303ef3e75b2`.
The archive contains 22,209 manifest-backed JPEG crops with four labels:

| label | count |
| --- | ---: |
| basketball | 1,496 |
| hoop | 1,565 |
| player | 15,295 |
| referee | 3,853 |

Every manifest reference is present and decodes successfully. There are 766
additional image members not referenced by the three manifests; they are not
used by the offline adapter. The archive is retained once at
`dataset/public_sources/e_bard_object_classification_v1/all.zip`; no extracted
duplicate is retained.

The source is accepted only for offline object-role/VLM crop training. The
split manifests are frame-level rather than game-level: train/valid/test cover
60/59/56 games, with 59/56/55 game overlaps. Consequently the audit artifact
`analysis_outputs/public_research/e_bard_object_classification_screen_v1.json`
(SHA-256
`77b3370f095a645336ef354c4b309b3808c3d69bb4f30b3e2867e411c14b4a13`) records
`accepted_for_offline_object_role_training=true` but
`accepted_for_cross_game_benchmark=false`. It cannot supply temporal ball
tracks, causal event labels, runtime answers, or the final blind-game gate.

### 2026-08-02 Derived-frame cleanup

After confirming no code, test or documentation still referenced the completed
MUVS visual review PNG caches, AGU deleted the four rebuildable directories
`analysis_outputs/public_research/muvs_event_state_frames_v1/`,
`muvs_event_state_frames_v2/`, `muvs_event_state_frames_v3/` and
`muvs_professional_extension_frames_v1/` (about 1,410,256 KiB). MUVS source
media, labels, compact features/evaluations and all blind-game/model assets were
retained. This cleanup does not change any runtime or acceptance artifact.

The storage audit also removed one redundant WASB checkpoint copy. The deleted
`model_checkpoints/wasb_sbdt/wasb_basketball_best.pth.tar` had the same SHA-256
(`8d1ba9870d0a6ab37b06ab82bed593c6c09133e713810bac475d0c000bb7e948`) as the
retained `wasb_sbdt_official` file; no dataset, runtime path or research
evidence was removed.

### 2026-08-02 E-BARD crop-role transfer screen

AGU ran a CPU-guarded, offline-only baseline over the retained E-BARD archive:
16x16 RGB pixels, HSV histograms, channel moments and grayscale gradients feed
a balanced multiclass logistic classifier. All 22,209 manifest rows were pooled,
but the sanity split held out 20% of game IDs so the frame-level source split
leakage could not inflate the source check. The basketball threshold was chosen
from that source holdout only at the highest recall meeting precision `>=0.85`.

The target contains no newly decoded video. It uses the border-free insets from
the hash-bound ATL--CHI review sheets: 180 candidates total, with 55
`valid_ball`, 98 `false_positive`, and 27 `uncertain` excluded from the metric.
Source holdout P/R was `0.857/0.728`; target transfer P/R was only
`0.326/0.255` (F1 `0.286`). The result is rejected at the `0.85/0.85` gate and
sealed at
`analysis_outputs/public_research/atl_chi_continuous_review_v1/e_bard_role_transfer_screen_v1.json`
with canonical SHA-256
`a1115b2ce11b4b010be5a2cb4f7ecfa9ff3aeac09d87d3a601af312314927eb1`.
The resource guard recorded zero stops; no checkpoint, runtime dependency or
blind-game asset was changed. E-BARD remains an offline semantic pretraining
candidate, while cross-broadcast continuous ball supervision and hard-negative
precision remain the active data gap.

## 2026-08-02 Cross-broadcast full-frame ball-detector increment

AGU added a truth-free, offline-only materializer that verifies each source
plan/review, raw-video SHA-256 and decoded-frame pixel SHA-256 before writing a
full-frame YOLO dataset. `valid_ball` rows become one ball box, `false_positive`
rows become empty labels, and `uncertain` rows are excluded. Four sealed bundles
(ATL--CHI benchmark, ATL--CHI continuous, ATL--CHI enrollment as validation,
and CHI--UTA benchmark) produce 417 rows: 330 train and 87 validation, with
166 positive and 251 hard-negative frames. The manifest is
`analysis_outputs/public_research/broadcast_ball_detector_cross_domain_v1/manifest.json`
(SHA-256
`8d47e0798c027f661d3182e921309ed74941d7eafbd2da1d8522071b6212b178`); the
approximately 70 MiB image set is retained for reproducibility. It is marked
`runtime_consumable=false` and `codex_runtime_answer_used=false`.

An online audit of the Hugging Face
[basketball_tracking card](https://huggingface.co/datasets/TrainingDataPro/basketball_tracking/blob/main/basketball_tracking.py)
found a CC BY-NC-ND release, so no copy was downloaded or admitted as an AGU
training source.

The canonical `.venv` (Python 3.11) ran a CPU YOLOv8n screen (30 requested
epochs, batch 4, 640px, workers 0); the output reached epoch 20 with best
internal mAP50 about 0.205 and was rejected. The guard never stopped on a
resource threshold: peak system memory/CPU/process-tree RSS were
`83.8%/86.5%/1,512,701,952` bytes and minimum available memory was
`2,788,966,400` bytes. Checkpoint hashes were sealed and the unpromoted
`best.pt`/`last.pt` files were removed after the screen.

The disjoint LAL--BOS external lower-bound screen has 20 valid and 86 negative
frames. With the final best checkpoint, confidence 0.25 gives
TP/FP/FN `4/19/16`, P/R/F1 `0.174/0.200/0.186`; confidence 0.10 gives
`7/58/13`, `0.108/0.350/0.165`; confidence 0.50 gives `0/6/20`, `0/0/0`.
The three `broadcast_ball_detector_external_screen_v2*.json` artifacts are
all `accepted=false`, offline-only, and do not alter AGU runtime or blind-game
assets. The data gap is still continuous cross-broadcast ball/non-ball hard
negatives plus causal ball--hand--rim evidence.

## 2026-08-02 rejected-source media cleanup

After the cross-broadcast detector increment failed its external gate, AGU
removed only source/derived media that had already been sealed as
`accepted=false` and had no remaining code or test consumer. The following
payloads were deleted while their README/manifest, result, resource and hash
records were retained:

- DeepSportRadar bounded raw images/masks/annotations (about 454 MiB) and its
  452 MiB derived YOLO image/label cache; the CC BY-NC-SA 4.0 source had
  internal P/R/mAP50/mAP50-95 `0/0/0/0`, and the tile route remained far below
  the promotion gate.
- APIDIS bounded `raw/` and HTTP-Range chunks (about 523 MiB) and the 228 MiB
  camera-1 derived detector cache; its single-game CPU run had zero fitness
  through six epochs and was never eligible for an external screen.
- The 155 MiB CC BY-NC 4.0 Play-by-Play ZIP; its fixed transfer probe reached
  only pooled/worst-game balanced accuracy `0.638/0.583` and the archive had a
  missing `match-2` referenced by `info.json`.

The deletions free about 1.8 GiB. Small audit manifests and documentation remain
for reproducibility, but none of these sources is a runtime dependency or a
candidate for blind-game truth. The retained CC BY 4.0 E-BARD/BODD weights,
EBQwen local models, accepted datasets, sealed evidence and HOU--ORL blind
assets were not touched.

## 2026-08-03 Independent EBQwen evidence-gate screen

The first conservative post-VLM gate is implemented in
`app/analysis/independent_shot_vlm_evidence_gate.py` and exercised by
`scripts/screen_independent_shot_vlm_evidence_gate.py`. For each held game, the
auxiliary threshold is selected only from the other games' OOF rows; the held
windows are then scored without using their labels for threshold selection.
The VLM contract requires continuous live play, controlled ball before release
and ball separation, and vetoes explicit replay/highlight or free-throw
observations. Missing auxiliary evidence abstains as `unknown`.

The EBQwen Q4KS 32-window screen bound to the frozen plan and the
Swin+MViT auxiliary OOF artifact produced TP/FP/FN/TN `13/9/3/7`, pooled
precision/recall/F1 `0.590909/0.812500/0.684211`; no held game satisfied both
promotion floors. The sealed result is
`analysis_outputs/public_research/independent_shot_vlm_ebqwen_swin_mvit_evidence_gate_screen_v1.json`
(SHA-256
`76aa0e716d9324544b301cce48e7261912952bb0189c773cba3b4216edc9064e`) with
`promotion_eligible=false`, `runtime_consumable=false`, and
`codex_runtime_answer_used=false`. This is a rejected offline screen, not a
runtime change or a blind-game claim.

The [basketball-events research release](https://huggingface.co/datasets/saveerjain/basketball-events)
was re-audited as a possible VLM event source. Its card says research-only,
the repository is about 7.87 GB, and the Hub viewer currently fails because
the train files do not share one schema. Since it has no compatible open
license for AGU training and no frame-level ball boxes, no video or annotation
payload was downloaded; the existing catalog entry remains a rejected
research reference.

### Online continuous-ball source re-audit (2026-08-03)

The follow-up search found no new materialization candidate for the active
cross-broadcast ball/non-ball gap. [YouTube-BoundingBoxes](https://research.google.com/youtube-bb/)
is CC BY 4.0 but supplies generic object boxes and video URLs rather than a
basketball-ball event stream. [SportsMOT](https://github.com/MCG-NJU/SportsMOT)
contains basketball clips but only player MOT boxes, a roughly 35 GB package,
and competition/non-redistribution terms. The [Dryad sport-ball annotation
release](https://datadryad.org/dataset/doi%3A10.5061/dryad.3bk3j9m13) contains
football, tennis and table-tennis archives only and is CC BY-NC-ND 4.0. None
was downloaded; no new source or runtime path was added.

The Roboflow [ball-tracking-udopu/2](https://universe.roboflow.com/basketball-lez3k/ball-tracking-udopu)
page is a better-fit candidate: CC BY 4.0, 700 images and `ball`/`goal`
classes. Its version/export endpoint returned a Cloudflare 403 challenge, so
the catalog records it as `candidate_only_authenticated_export_not_available`;
no auth bypass or download was attempted.

AGU also temporarily audited the Hugging Face
[emirsahin/basketball-ball](https://huggingface.co/datasets/emirsahin/basketball-ball)
sample. Its Hub card says MIT, while the embedded Roboflow README says CC BY
4.0, so the rights boundary is contradictory. The 214 MiB archive contains
1,913 JPGs from one `emir-shoots` source prefix with augmentation, but no YOLO
label files (only two README text files). It cannot provide hash-verifiable
continuous ball/non-ball supervision; the temporary download was removed and
the catalog keeps it as `research_reference_only_no_training_import`.

### Infactory soccer-ball auxiliary screen (2026-08-04)

The [Infactory ball-tracking dataset](https://huggingface.co/datasets/infactory-ai/ball-tracking)
is a CC BY-NC 4.0 soccer-ball release. Its card describes 1,450 metadata-bound
frames from 10 broadcast clips: 884 visible rows, 566 `not_visible` hard
negatives and 924 normalized boxes. The repository contains additional
unreferenced image payloads; AGU downloaded only the 1,450 rows named by
`metadata.csv` and sealed them at source revision
`75854a5837dd3302cbb1843355ecc1a6f124f818`. The source manifest SHA-256 is
`414277673bae3b14137d76bbf05fb0e7771563c1cac51932157d40f9cdd1108b`; its
derived YOLO manifest is
`18146cc0c9af4811778996401556c2570c824efcca3e9c057d6dbf6fa651ab72`.

A five-epoch, CPU-only YOLOv8n auxiliary run in the canonical `.venv`
reached best internal validation mAP50 `0.48375` (P/R `0.60616/0.46092`).
On the held Infactory test clips the best fixed confidence was only
P/R/F1 `0.662162/0.376923/0.480392`; no confidence reached the `0.85/0.85`
gate. On the independent 240-image E-BARD basketball test, the candidate had
`TP=0` at every screened confidence from `0.05` through `0.50`. The results
are sealed in `analysis_outputs/public_research/infactory_yolov8n_aux_v1/`;
the candidate checkpoints were hash-sealed and deleted, while the licensed
source subset, YOLO materialization and offline evidence remain explicitly
non-commercial and runtime-inert. This source is not promoted to AGU runtime
or blind inference; the active blocker remains cross-broadcast basketball
small-ball recall/precision plus causal ball-hand-rim evidence.

### UniqueData basketball-tracking mirror binding audit (2026-08-04)

The public [UniqueData `basketball_tracking` mirror](https://huggingface.co/datasets/UniqueData/basketball_tracking)
was temporarily downloaded at revision
`2e646ed7624adaca55b282e8692f4c9d2c3cb88f` to verify whether its sample could
be used as a small-ball supplement. It declares CC BY-NC-ND 4.0 and contains
only 70 CSV rows (67 positive boxes and 3 explicit no-box rows). All 70 image
entries decode, but the archive's `boxes.tar.gz` contains 106 unrelated
`1280x720` `frame_*.PNG` images rather than files keyed by the 70 CSV image
names; the upstream loader zips the two archives by position. This is not a
reproducible image/box binding and cannot supply the required continuous
cross-broadcast hard negatives. The commercial-use note also points to a paid
dataset.

The compact audit record is
`analysis_outputs/public_research/unidata_basketball_tracking_audit_v1.json`
(SHA-256
`b1c84428c16f31d39dbb64bd974a70ca44e3e210980e9a79ae9900f2a453b5b3`). The
temporary payload was deleted; no source data, checkpoint or runtime path is
retained. The catalog remains `accepted=false` and
`runtime_consumable=false`.

### CHI–UTA hard-negative review increment (2026-08-04)

AGU manually reviewed 36 additional candidate boxes from the non-blind CHI–UTA
development broadcast: 11 `valid_ball` and 25 `false_positive` (candidate
precision `0.3056`). After de-duplicating the existing reviewed bundles, the
offline full-frame dataset contains 441 rows, split 354 train / 87 validation,
with 170 positive and 271 empty hard-negative labels. The dataset manifest
artifact SHA-256 is
`e8b9439fcc9a303d5246a14d7d3767a954ef668dc7bb6b68e4d190b7d52adbe1`.

A guarded five-epoch CPU YOLOv8n run in the canonical Python 3.11 `.venv`
(batch 4, 640px, workers 0) ended at validation P/R/mAP50/mAP50-95
`0.26508/0.19231/0.11972/0.04087`; the 44-sample resource trace peaked at
83.9% system memory and 76.0% CPU. The disjoint LAL–BOS screen produced
TP/FP/FN `3/24/17` (P/R/F1 `0.111/0.150/0.128`) at confidence 0.10,
`1/3/19` (`0.250/0.050/0.083`) at 0.25, and `0/0/20` at 0.50. All three
are `accepted=false`, far below the `0.85/0.85` promotion gate.

Retention evidence is sealed in
`analysis_outputs/public_research/broadcast_ball_detector_cross_domain_v2_increment_v1/retention.json`
(SHA-256
`bb1d595845da61c7cbf6df5a7b89727dd5536311961ebb078d102cbac11ac4dc`). The
unpromoted best/last checkpoints and generated label caches were deleted after
hash capture. The materialized dataset, review decisions, training arguments/
results, resource log and external screens remain offline-only; no detector,
runtime, EBQwen or HOU–ORL blind asset changed.

### Licensed basketball COCO ball/rim candidate and external screens (2026-08-05)

AGU audited and retained the fixed Hugging Face mirror
`[koppolusameer/basketball-coco-20260416](https://huggingface.co/datasets/koppolusameer/basketball-coco-20260416)`,
whose upstream Roboflow export is
`[Basketball Player Detection 3](https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-3-ycjdo)`.
The pinned revision is `92dea3042e1ec707be5b0d4f7ac474a2b314293`; both source
README files declare CC BY 4.0. The three COCO splits contain 654 decoded
1920x1080 broadcast frames from 15/3/3 Celtics–Knicks and Celtics–Magic clip
groups, with disjoint file names and clip groups. The retained source payload
is `dataset/public_sources/basketball_coco_2026_v1/` (659 files,
172,253,022 bytes) and the audit artifact is
`analysis_outputs/public_research/basketball_coco_2026_audit_v1.json` with
artifact SHA-256
`4f1cc5157013f8d07a26431156d3bc790ad7accb95ab1af85259b9bb85df5f27`.

The COCO schema includes `ball`, `rim`, `ball-in-basket`,
`player-in-possession`, `player-jump-shot`, `player-layup-dunk`, and
`player-shot-block` among other classes. A 12-panel contact-sheet review
confirmed continuous broadcast frames and visible ball/rim boxes, but every
export has the same bottom black Roboflow watermark; action-subtype boxes are
not consistently nested in player boxes, so they remain offline frame-level
evidence candidates rather than player identity or game-truth labels.

AGU materialized a ball/rim-only YOLO view with 654 frames and
train/validation/test boxes `887/184/181`. A guarded five-epoch CPU YOLOv8n
run in the canonical Python 3.11 `.venv` reached source-only validation
mAP50 `0.79665`, but the disjoint LAL–BOS screen peaked at
P/R/F1 `0.12/0.15/0.1333` (confidence 0.20). Combining the 441 reviewed
ATL–CHI/CHI–UTA cross-broadcast rows improved the best external point only to
P/R/F1 `0.13636/0.15/0.14286`; neither route reaches the `0.85/0.85` gate.
Both guarded runs stopped normally (no resource stop). The summary artifact
is `analysis_outputs/public_research/basketball_coco_2026_training_v1.json`
(artifact field SHA-256
`88cc57b9b52f709d7769146bd9fad42d4c5d3890182073f8deae7a69a3452dda`);
the unpromoted checkpoints, temporary training directories and label caches
were deleted after hash capture.

The source is therefore an offline training candidate only:
`accepted_for_runtime=false` and
`accepted_for_external_acceptance_benchmark=false`. The default detector,
VLM, runtime schema, EBQwen and HOU–ORL blind assets are unchanged. The
active blocker remains game-/broadcast-diverse continuous ball/non-ball hard
negatives plus independently verifiable ball–hand–rim causal sequences.

### Deduplicated 2025 basketball COCO supplement and external screen (2026-08-05)

AGU audited the fixed
[basketball-coco-2025-04-02](https://huggingface.co/datasets/koppolusameer/basketball-coco-2025-04-02)
mirror at revision `d0b72a61490311431c00774db5277cdca737dbe5`, from the upstream
[Roboflow basketball players project](https://universe.roboflow.com/roboflow-universe-projects/basketball-players-fy4c2).
The Hub card and embedded README disagree (MIT versus CC BY 4.0), so AGU
uses the stricter CC BY 4.0 attribution-required boundary and keeps the source
offline-only. The export contains repeated augmentation variants of the same
source frames; grouping by normalized source identity and retaining the first
upstream variant (train preferred, then valid/test) produced 139 unique frames.
Manual review found 99 ball boxes, 98 rim boxes and 19 empty frames. The
deduplicated source and its audit/contact sheet are retained under
`dataset/public_sources/basketball_coco_2025_v1/` and
`analysis_outputs/public_research/basketball_coco_2025_audit_v1/`. The audit
artifact SHA-256 is
`85d51b016a800961c04256a1a712a5d190f3e619503dc3ed7c6fe97b35b52a0e`.

For a guarded supplemental route, AGU combined those 139 frames with the
2026 COCO ball/rim view (464 train / 96 validation) and the reviewed
cross-broadcast hard-negative set (354 train / 87 validation), yielding
957 train and 183 validation rows. The five-epoch CPU run in the canonical
`.venv` reached internal best mAP50 `0.75056`; the hash-bound independent
LAL–BOS screen peaked at confidence `0.20` with TP/FP/FN `5/15/15` and
P/R/F1 `0.25/0.25/0.25`. The resource guard sampled 118 times, peaked at
84.4% system memory and 69.3% CPU, and stopped zero times. Training evidence
is retained in
`analysis_outputs/public_research/basketball_coco_2025_training_v1.json`
(artifact SHA-256
`c801540066bce78078a7305960e6cab3c5d52b69399668ef6668025de882102f`), and
the independent screen is
`analysis_outputs/public_research/basketball_coco_2025_external_screen_lal_bos_v1.json`
(SHA-256
`2fb3b658c9d76a4c80537c76f32a5bf033cf4b89e3a55b97b3eecdd6424e7833`).
The sealed training summary records Python 3.9.6 in its
`training_environment.python` field, while the current canonical
`.venv/bin/python` is verified as Python 3.11.15. The historical artifact is
left immutable and the rejected route is not promoted; future training must
re-capture environment metadata from `.venv`.

The candidate remains `accepted=false`,
`accepted_for_runtime=false` and
`accepted_for_external_acceptance=false`. Temporary combined data, training
directories, checkpoints and label caches were deleted after hash capture.
The source, deduplication audit, resource trace and external screen remain
offline-only; the blocker is still game-/broadcast-diverse continuous
ball/non-ball hard negatives plus causal ball–hand–rim evidence.

### Independent EBQwen ball-candidate VLM screen (2026-08-05)

To test a conservative “detector candidate plus independent raw-frame VLM” gate,
AGU used the hash-bound LAL–BOS RF-DETR perception artifact and its sealed review
plan. Six confidence bands were sampled at 18 candidates each (108 total). The
VLM saw only the raw full frame with a red candidate box and a yellow local inset;
the plan, `valid_ball`/`false_positive` decisions, statistics and labels were not
exposed. The model was the retained local
`EBQwen2.5-VL-3B-MLX-4bit` from
`GabrieleGiudici/EBQwen2.5-VL-3B`, revision
`c6a93cbb325f9d20236f85bcaba7827a2808443e`, with weight SHA-256
`f2b3cf74e088a06b7bf1c6462664ca13f95e1c1c8be4c6c8a0e469f68dd1d72c`.
The native BF16 copy remains a fidelity backup and was not promoted.

All 108 responses were parseable, but the strict states were `uncertain=103`,
`ball=5` and `not_ball=0`. Against the sealed 108-row review, the stratified
weighted `ball`-keep policy scored precision `0.444060` and recall
`0.113961` (lower bound) / `0.102302` (upper bound), below the `0.85/0.85`
promotion gate; the route is `accepted=false`. Some `uncertain` responses contain
decisive-sounding text in their reason field, but AGU does not rewrite that text
into a binary label, so this remains a strict independent-VLM diagnostic rather
than parser tuning on external labels.

The first guarded run stopped safely at 46/108 after three consecutive samples
below 2 GiB available memory. A cache-bound continuation used a 1.5 GiB minimum
while retaining 90% system-memory, 95% CPU and three-sample stop limits; it
completed normally. The continuation peaked at 89.7% system memory and 46.7% CPU,
with about 1.64 GiB minimum available memory. Prediction evidence is
`analysis_outputs/public_research/ebqwen_ball_candidate_probe/lal_bos_benchmark/predictions_v1.json`
(artifact SHA-256
`fc1e5b77d95ea05627501b9086af7c47e928f9f3784eef86c0ecde18f5b31ef9`), and the
evaluation is
`analysis_outputs/public_research/ebqwen_ball_candidate_probe/lal_bos_benchmark/evaluation_v1.json`
(artifact SHA-256
`f727229e02954e73494f1516db0ceb74bcc9f01579f26f9e82ffdb5ba6916648`).

The screen remains offline-only: no detector default, runtime/VLM fusion,
runtime schema, EBQwen weight or HOU–ORL blind asset changed. The active blocker
remains licensed game-/broadcast-diverse continuous ball/non-ball hard negatives
plus independently verifiable ball–hand–rim causal sequences.

### ORL–CLE NBA Games offline media supplement and cross-game screen (2026-08-05)

The official NBA Games collection and metadata expose a 2009-05-28 ORL–CLE game
record (`0040800305`, YouTube ID `GxQV8JhJCMs`). Under the user-authorized
offline-research boundary, AGU downloaded a 480p local copy to
`dataset/public_sources/nba_games/media_orl_probe_v1/GxQV8JhJCMs.mp4` and bound
it to SHA-256
`d62254aeae222cd601e440942a78ab3a1fab985b4e4a82ffb2bbe45a35cec508`. OpenCV
decoded all 230,597 frames and 11 fraction seeks; the duration delta was about
0.253 seconds. `analysis_outputs/public_research/nba_orl_cle_media_audit_v1.json`
records the media as `runtime_consumable=false` and
`codex_runtime_answer_used=false`: the metadata licence does not grant
redistribution or runtime use of the underlying broadcast, so the payload is
an offline candidate-audit input only.

The BODD YOLOv8n scan used 1 FPS (7,687 sampled points, 230,597 decoded frames),
producing 3,330 basketball detections and 1,789 candidate tracks. A stratified
108-panel object-only review yielded 49 `valid_ball`, 22 `uncertain` and 37
`false_positive`; the lower-bound P/R at confidence `>=0.45` was
`0.712076/0.311670`, and no threshold met `0.85/0.85`. The sealed summary is
`analysis_outputs/public_research/nba_orl_cle_bodd_review_summary_v1.json`
(artifact SHA-256
`7270b5d05dbc93292734b7e81969f7a8a390a4dd2210834fe8385f2bc2952e32`).

An independent label-hidden EBQwen MLX 4-bit probe on 12 panels returned
`ball=2`, `uncertain=10`, `not_ball=0`; weighted P/R was
`1.0/0.500617` (lower) and `1.0/0.459419` (upper), so it remains rejected.
Prediction/evaluation artifacts are
`analysis_outputs/public_research/nba_orl_cle_bodd_vlm_probe_v1.json` (SHA-256
`41efe7163e8284a30abdfbfedac2c4f66ef9e5a7b12c3d2fe74812061e88bada`) and
`analysis_outputs/public_research/nba_orl_cle_bodd_vlm_evaluation_v1.json`
(SHA-256 `d7654fe529bc7eef708e80c553eeee4ddbc66bbd18521642bed445f23a059614`).

The same-detector LAL–BOS to ORL–CLE verifier tested visual, confidence and
track-geometry variants. The best conservative variant,
`visual_plus_track_geometry`, reached only P/R=`0.714836/0.750198` (upper
bound P/R=`0.855281/0.555607`); all variants failed the dual gate. The sealed
screen is `analysis_outputs/public_research/
lal_bos_orl_cle_bodd_verifier_screen_v3.json` (artifact SHA-256
`7558002341d4c0c6ea75e9d8941f17ff5cc2b0b67c2eca7689e21b2ff99dd06c`).
This supplement is therefore `accepted_for_runtime=false`,
`accepted_for_cross_game_training=false`, and does not alter detector/VLM
defaults, runtime schema, EBQwen weights or HOU–ORL blind assets. The remaining
data gap is licensed, game-/broadcast-diverse continuous ball/non-ball hard
negatives plus independently verifiable ball–hand–rim causal sequences.

The rejected MP4 was deleted on 2026-08-06 after the audit completed. Its
SHA-256, size, decode audit, predictions and review artifacts remain sealed;
only the 564-byte `download-state.json` provenance record remains in the source
directory. This reclaimed approximately 1.02 GiB without touching the retained
HOU–ORL blind media or the EBQwen native/MLX weights.

### SportsGrounding metadata-only causal semantics candidate (2026-08-05)

AGU audited the fixed revision
`1fb03e7786228ee6d1d7909eb7011e7e3fb7eb2c` of
[MCG-NJU/SportsGrounding](https://huggingface.co/datasets/MCG-NJU/SportsGrounding).
The dataset card declares CC BY-NC 4.0 and describes a basketball subset of
MultiSports with 520 split-disjoint videos, 4,243 continuous person tubes and
25 FPS metadata. The train/validation JSON contains 3,633 captions mentioning
`ball`, including control, pass, shoot, defend, block and steal language; every
`bbox` list matches its inclusive start/end frame span and no invalid boxes were
found. It does not provide basketball-ball boxes or official game statistics.

Only the 17,360,122-byte train JSON and 6,431,983-byte validation JSON were
downloaded and retained under
`dataset/public_sources/sports_grounding_metadata_v1/`; the approximately
14.7 GB video archives were not downloaded. Their SHA-256 values and the full
schema/term-count audit are sealed in
`analysis_outputs/public_research/sports_grounding_metadata_audit_v1.json`.
The source is `accepted_for_offline_causal_semantics=true`, but
`accepted_for_offline_ball_training=false`, `accepted_for_runtime=false` and
`accepted_for_external_acceptance=false`. It may inform offline caption/tube
semantics or VLM pretraining experiments only; it cannot replace licensed
continuous ball/non-ball hard negatives or ball–hand–rim causal truth.

### 2026-08-06 Offline shot-causal support screen (no new dataset)

No new source payload was downloaded for this experiment. The existing sealed
`shot_reason_evidence_full511_v3.json` was joined to the frozen
`shot_broadcast_fusion_full511_batch4_pca16_c0p01_v1.json` OOF probabilities by
`(source_video_sha256,event_id)` and evaluated with an outer game-held logistic
screen. The selected continuous features increased recall but reduced pooled
precision to `0.572104` and the weakest-game precision to `0.455172`; the baseline
was `0.791667` pooled precision and `0.732558` weakest-game precision under the
same held-out threshold protocol. The result is
`analysis_outputs/public_research/shot_causal_support_screen_v1.json`,
`accepted=false`, `runtime_consumable=false`; it is evidence only and does not
change any dataset, detector, VLM or runtime decision.

### VRU_Basketball continuous-scene subset (2026-08-06)

AGU fetched a small, hash-bound subset from the pinned revision
`d256fa0b5fab474663f595abe4f386ac4d5adcc6` of
[BestWJH/VRU_Basketball](https://huggingface.co/datasets/BestWJH/VRU_Basketball),
whose card declares CC BY 4.0. Only eight individual ZIP members were extracted
with HTTP Range requests: four Dongguan clips and four Guangzhou clips, totaling
`176,838,471` bytes. Each is a ten-second, 25-FPS, 1920×1080 continuous
basketball scene. The full approximately 1.51 GB archives were not retained,
and the source supplies no ball boxes, event labels, or ball–hand–rim truth.

The existing E-BARD BODD detector was run offline on 400 sampled frames (every
five frames), producing 6,572 detections. A detector-candidate-stratified review
of 48 basketball detections found 44 `valid_ball` and four false positives. This
is a candidate-biased precision sanity check, not an exhaustive recall estimate
or an independent causal gate. The source manifest and result are sealed in
`dataset/public_sources/vru_basketball_v1/source_manifest.json` and
`analysis_outputs/public_research/vru_basketball_audit_v1.json` (audit SHA-256
`92662dc9455256e0a38dc5a87c67d7239a5fd3d67ef0db19ae75b9c9411a194e`).

The subset is retained only as an offline continuous-scene and hard-negative
candidate (`runtime_consumable=false`). It is not imported into the default
detector/VLM/runtime, training weights, or blind evaluation. Timeline inspection
shows dribble, pass, possession and hoop views but no independently verified
release–rim–outcome sequence, so the cross-game/broadcast hard-negative and
ball–hand–rim causal-evidence blocker remains open.

### Play by Play basketball annotation slice (2026-08-06)

AGU downloaded and hash-verified the [Zenodo Play by Play](https://zenodo.org/records/12698090)
archive under its CC BY-NC 4.0 terms. The source archive is 162,983,009 bytes with
MD5 `caac618508f17e3b08850be1bf24a090`. The retained basketball-only tree contains
381 JSON clips and 56,578 standardized-space frames, including 460,803 player
points, 11,851 frames with visible-ball observations and 120 stopped-ball points.
The coordinates are court-normalized (x `0..2`, y `0..1`); no raw broadcast video,
pixel boxes, hand annotations or rim annotations are included. The metadata names
`match-2` as a test match, but that directory is absent from the archive; the
audit records it as a missing split rather than treating it as an independent
test set.

The sealed audit is
`analysis_outputs/public_research/play_by_play_basketball_audit_v1.json` (artifact
SHA-256 `4e19d06d7998a8cda7b7194ae522c94d7a1d1d780a4118daf24d76475f5b9e30`). The
annotations and rights note are retained under
`dataset/public_sources/play_by_play_v1/`; the verified source ZIP was removed as
redundant after extraction. This source is offline auxiliary trajectory calibration
only (`runtime_consumable=false`) and cannot provide AGU runtime answers, blind-game
truth, or the missing ball–hand–rim causal supervision.

### VRU raw-frame ball–hand–rim causal review (2026-08-06)

The eight retained VRU clips were used to build a label-hidden, frame-hash-bound
manual review plan. The `.venv` OpenCV builder binds the source manifest and every
reviewed decoded frame; the plan covers six windows: three visible release-to-rim
chains from `guangzhou_1`, `guangzhou_11` and `guangzhou_31`, plus three
cross-scene non-shot windows from `guangzhou_21`, `dongguan_0` and `dongguan_16`.
The positive windows resolve release frames near `180/180/177` and rim-proximity
frames near `216/212/228`; outcome and rim contact remain `unknown`/unconfirmed.
The negative windows contain no release-to-rim chain.

The source manifest SHA-256 is
`09c0aeac0b6fd61c4b74222c61c90a3563482ad58ffe43c7ae55f15606214685`; the plan is
`analysis_outputs/public_research/vru_causal_review_plan_v1.json` (SHA-256
`e34adda6f77698aef3b8a39f3d89483a25fa1c2e41dc911b844187b2d7088578`), and the
sealed review is `analysis_outputs/public_research/vru_causal_review_v1.json`
(SHA-256 `083f351d913c3fd8c600cc3278249b554d820ecd659245fd411a28ae29e2bde1`).
The plan, decisions and review are training/calibration-only with
`runtime_consumable=false` and `codex_runtime_answer_used=false`; they do not
alter detector/VLM defaults, EBQwen weights, runtime answers or blind evaluation.
This small manual batch demonstrates a usable annotation contract but does not
provide enough game-/broadcast-diverse or independently sourced causal truth to
clear the active gate.

### VRU additional range subset and causal review (2026-08-06)

To expand the offline annotation pool without materializing either full archive,
AGU extracted eight additional members from the same pinned CC BY 4.0 revision
using bounded HTTP Range requests: four Dongguan and four Guangzhou clips,
`179,162,740` bytes in total. Each member is CRC/SHA-256 checked and decodes as
250 frames at 25 FPS, 1920×1080, 10 seconds. The additional-only manifest is
`dataset/public_sources/vru_basketball_v1/source_manifest_extra_v1.json` (SHA-256
`fee356d865c072879b2b07b4e5b3a1fe581dd41fae8960e8c337beafaa9cb866`); the original
eight-clip manifest and its detector audit remain unchanged and separately bound.

The new label-hidden, frame-hash-bound Codex review covers eight windows:
`guangzhou_33` contains one visible release-to-rim chain (release frame 180,
rim-proximity frame 216, outcome unknown), `guangzhou_15` is an uncertain
cross-lane pass, and six windows are `not_a_shot`. The plan and sealed review are
`analysis_outputs/public_research/vru_causal_review_extra_plan_v1.json` (SHA-256
`71425bf2feef69735778886d3d0d5d3eb502ef54989a4c71692f09a5f3f90137`) and
`analysis_outputs/public_research/vru_causal_review_extra_v1.json` (SHA-256
`a631a4f1852af3e830cca7b02147c8f5956f86a6226c2bde633aaa5441f9c1fd`). The
extraction/review audit is
`analysis_outputs/public_research/vru_basketball_extra_audit_v1.json` (SHA-256
`9e06f9817ef8a556e65114d9380b985ae1d621570c72571dc6413a307d2d3081`). This is
Codex annotation for offline research only (`training_media_eligible=false`,
`runtime_consumable=false`); the single positive remains outcome-unresolved and
does not clear the cross-game/broadcast causal or hard-negative gate.

As a bounded model feedback probe, the frozen E-BARD/BODD checkpoint was run only
on `guangzhou_33` (shot), `guangzhou_15` (uncertain pass) and `dongguan_28`
(`not_a_shot`) at 5 FPS on CPU under the resource guard. Candidate visibility was
`14/15` sampled frames in the shot release-to-rim window, `9/10` in the pass
window, and `6/19` in the non-shot window. Because ball candidates appear in both
the pass and non-shot windows, this probe rejects detector-only causal fusion; it
does not estimate shot accuracy or recall. The sealed result is
`analysis_outputs/public_research/vru_bodd_extra_causal_probe_v1.json` (SHA-256
`33d40254b71cca475be3ddc2034d10872e69b68057cfd6f4d61e812c3ddae25e`), with
`accepted_for_training=false` and no runtime/VLM/default-weight change.

### 2026-08-06 candidate source audit and rejected downloads

AGU checked three additional public candidates before spending more disk:
[TeamTrack](https://atomscott.github.io/TeamTrack/) is a large MIT player-MOT
corpus with basketball-side/top views but no documented ball, hand or rim truth;
the full Kaggle distribution is approximately 17.7 GB and was not downloaded.
The [Basketball Tracking Dataset](https://www.kaggle.com/datasets/trainingdatapro/basketball-tracking-dataset)
contains static/object-style samples and non-commercial terms, not a licensed
continuous game sequence with exhaustive negatives or causal labels. The
[Dryad sport-ball release](https://datadryad.org/dataset/doi%3A10.5061/dryad.3bk3j9m13)
contains football, tennis and table-tennis material but no basketball. None
meets the current acceptance contract, so no new payload or checkpoint was
created. The active requirement remains licensed, game-/broadcast-diverse
continuous ball/non-ball hard negatives plus independently verifiable
ball–hand–rim causal sequences.

Two additional records were checked before closing the search. The
[MUVY v1 Zenodo record](https://zenodo.org/records/13883315) is a 7.8 GB
multi-view archive; its basketball portion is only 13 recordings/14,105 frames,
and that bounded portion is already present locally and has been screened. Its
ball boxes are detector-generated and manually filtered, not an independent
ball–hand–rim truth source, so the full archive was not downloaded. The
[BasketEvent HF mirror](https://huggingface.co/datasets/zaywas/BasketEvent) has
a 6.73 GB unlicensed repository of JSON player/ball trajectories and play-by-play
event fields. A four-file transient probe confirmed useful ball trajectories,
but the fixed revision has no README, LICENSE or dataset card and no hand/rim
annotations; the probe was deleted and the source is rejected for training,
runtime and external acceptance. The bounded source decisions are sealed in
`analysis_outputs/public_research/basketball_source_candidate_audit_v1.json`;
no unlicensed payload is retained.

### Qlean Basketball Match Videos: gated continuous-video candidate (2026-08-06)

The public [Qlean Basketball Match Videos card](https://huggingface.co/datasets/qleandataset/video-basketball-match)
describes six games and 28 MP4 files (about 26.40 GB) from multiple camera
positions. Qlean/Amanaimages describes the material as rights-cleared training
data for academic research, but access is gated: the card asks for a name,
affiliation, academic status, intended use and contact email, and the Japanese
Qlean Academic Research License is the governing text. The public card does not
list ball, hand, rim or event annotations, so the source cannot yet supply the
causal labels AGU's acceptance gate requires.

No gated payload was downloaded. The source is registered in the catalog as
`qlean-video-basketball-match` and retained only as metadata with
`runtime_consumable=false` and `training_media_eligible=false`. The audit
(`analysis_outputs/public_research/qlean_video_basketball_match_audit_v1.json`,
SHA-256 `5fd4d832b7935d96f5f1284f001f153dbdcc0b9d36ca6965b4f66c4ce19793de`)
records the card facts and the fail-closed decision. Pursuing it requires an
authorized academic access identity followed by a separate file-level license,
annotation and cross-game audit; it is not a reason to resume blind inference.

### SportsAction / MultiSports: gated action-tube candidate (2026-08-06)

The public [MCG-NJU/SportsAction card](https://huggingface.co/datasets/MCG-NJU/SportsAction)
describes MultiSports spatial-temporal action localization with `rawframes.tar` and
`multisports_GT.pkl`. Its basketball taxonomy includes pass, drive, dribble, 2-/3-point
shot, free throw, block, offensive/defensive rebound, steals, screen, save and jump ball;
the card reports 28,514 train tubes and 10,116 validation tubes. The annotations are
person action tubes (`frame,x1,y1,x2,y2`), not ball, hand, rim or made/miss truth, and the
source recordings were collected from YouTube competition videos.

The card declares CC BY-NC 4.0 and requires acknowledging the license before file access.
No 65.1 GB payload was downloaded. The catalog entry is therefore
`metadata_only_until_gate_and_ball_hand_rim_audit`, with
`runtime_consumable=false` and `training_media_eligible=false`. The sealed audit is
`analysis_outputs/public_research/sports_action_multisports_audit_v1.json` (SHA-256
`23c19cec33442ea0c0845fb833323106322fa956bed1e16f201c55dee7371642`). It is retained
only as a future noncommercial action/rebound hard-negative candidate; it cannot clear
AGU's independent ball--hand--rim acceptance gate.

### MEM--OKC candidate/PBP coverage probe (development-only, 2026-08-06)

The exposed MEM--OKC 2011-05-11 video (SHA-256
`1caf38b7b9c1997805cf5e66a37a05af974fdb5d99dfaa8816f328710523bbb8`) was used only for
an offline coverage probe. A resource-guarded 10-second scoreboard OCR aligned 470 of 477
local PBP events and produced 200 mapped shot labels (156 field goals, 44 free throws).
The existing 621 candidate anchors join only 63/200 labels at a strict +/-90-frame
tolerance (31.5%); +/-300 frames reaches 143/200 (71.5%) but is too loose for causal
supervision.

No embeddings or training labels were created. The sealed probe is
`analysis_outputs/public_research/mem_okc_pbp_candidate_join_probe_v1.json` (SHA-256
`a7a6b103cd5c84c87250ecc33d65059b1b2aec2c57db1a7d325e71a310b36624`), and remains
`runtime_consumable=false` and `accepted_for_training=false`. The result does not
change the blind-media boundary or the active 85% gate.

### MEM--OKC candidate sweep follow-up (development-only, 2026-08-06)

A parameter sweep over the existing perception cache tested 15 candidate-clustering
variants without decoding or modifying any blind media. The selected `d2_g3` variant
(`max_normalized_distance=2`, `cluster_gap_sec=3`, `merge_gap_sec=3`) increases the
raw candidate bundle from 621 to 668 events (334 field-goal candidates and 334
rebound candidates). A fresh resource-guarded 10-second scoreboard pass produced
229 readable samples, with peak RSS 2,098,200,576 bytes and zero swaps; the PBP
alignment remains 470/477 events and 198/200 shot labels have frame mappings.

The exact one-to-one matching protocol is sealed in
`analysis_outputs/public_research/mem_okc_candidate_sweep_join_probe_v1.json`
(SHA-256 `7644b1228e697e8826abe20cbe006ba51c4588eb0fb737e66902dc9509fdb8c5`).
At +/-90 frames, the selected bundle covers 88/200 under the end-frame/all-candidate
screen (44.0%), versus 62/200 for the same rerun baseline; the canonical event-frame
screen is 95/200 (47.5%). Type-compatible matching is lower at 64/200 and 59/200.
These are useful proposal-count diagnostics, not causal labels: all remain far below
the 85% supervision gate, so no embeddings, training labels, runtime change or blind
inference was created. The temporary sweep bundles and OCR cache are disposable.

### Bdc non-overlap hard-negative increment (2026-08-06)

An offline-only increment was reviewed from the retained development video
`BdcP-XwUCk8.mp4`. Thirty-two deterministic windows were selected outside existing
candidate neighborhoods; raw-frame review marked 8 live field-goal attempts and 24
replay/free-throw/dead-ball/no-release negatives. The sealed bundle, labels and
training manifest are under `analysis_outputs/public_research/bdc_extra_hard_negative_v1/`
with manifest SHA-256 `785a72b156d0fdcc5aaa59e892dde6c1d993855e3884968f09f1e66cf43dfc44`.

Frozen MobileNet/MViT embeddings were appended to the 511-row development screen with
game-held exclusion. Scene+MViT pooled P/R/F1 changed from `0.7379/0.9231/0.8201` to
`0.7540/0.9137/0.8262`; weakest-game precision changed from `0.6737` to `0.6827`, still
below the `0.85` gate. This is retained as a reproducible training/calibration negative
result only (`runtime_consumable=false`, `accepted=false`); temporary contact sheets
were deleted after sealing. No runtime, detector, VLM, EBQwen weight or blind asset changed.

### Bdc non-overlap broadcast-state fusion follow-up (2026-08-06)

The 32-window Bdc increment was re-screened with label-free broadcast-clock OCR. The
anchor reader was corrected to honor the generic `candidate_event_frame` evidence field,
and the new path is covered by `tests/test_broadcast_clock_script.py`. Only one of the
32 windows was classified as `replay`; it is a manually reviewed non-live window, while
the remaining 31 are `unknown`, so this proxy is not causal ground truth.

The extra clock artifact SHA-256 is
`d84e32cbc8a97106f4fdf7831323f0b8e6012345297d489f862ce08f13ad0d41`; the merged
543-row broadcast-state artifact SHA-256 is
`3dbc6409af2fce60fff0c438a88d22b1e622bcbc08a3133a57de0e217a61729b`. The merged
scene+MViT+`base+broadcast_raw` screen gives pooled P/R/F1
`0.7778/0.9059/0.8370`, with weakest-game precision `0.7037`, versus the original
511-row `0.7897/0.9271/0.8529` and `0.7326`. The screen SHA-256 is
`6830243648c9e087b8645320c75e0ecdb9dce1c11432f2e72a7f52010f5e9e4d`; it remains
`runtime_consumable=false`, `codex_runtime_answer_used=false`, and unpromoted. No
checkpoint, runtime/VLM default, EBQwen weight or blind asset changed.

### NBA Games Lakers–Magic cross-broadcast development increment (2026-08-06)

The fixed NBA Games revision `3a20f2b9f60a025c8c641f4c763d4ac467ff059e` supplied a
new 2009-06-04 Lakers–Magic YouTube source (`uIxqFnBCOyk`). Only the 854×480 video
stream was retained locally: 938,827,586 bytes, SHA-256
`fb5ae34fd80793fff49689af93de4b11f505c79068efb1afbcecdc7f2baa31da`, 198,217 frames
at 29.97 fps. The media/source manifest is
`dataset/public_sources/nba_games/media_new_dev_v1/source_manifest_v1.json`; the
underlying external media is not redistributed and remains `runtime_consumable=false`.

Thirty-two deterministic, non-overlapping raw windows were selected with labels hidden
from selection. Independent raw-frame review sealed 7 complete live field-goal attempts
and 25 intro/replay/free-throw/dead-ball/no-release negatives. Bundle SHA is
`caa07eb127a4aca8d925476d80d9331a0c2c8deb7716dcef133015a6ca00d2a9`; training-manifest
SHA is `1415fb8fd2a17062c833c434fe4d18c611a3c26a42f1dfc9c3db33359c966c34`.

Frozen MobileNet/MViT/Swin3D embeddings were merged with the 511-row development
artifact and screened with game-held OOF isolation. The best scene+MViT variant gives
pooled P/R/F1 `0.6222/0.9724/0.7588`; the newly held Lakers–Magic game gives
`0.4000/0.8571/0.5455`. The increment is therefore an audited cross-broadcast hard-negative
negative result, not a promotion: no runtime, detector, VLM, EBQwen weight or blind asset
changed. Full provenance and hashes are in
`analysis_outputs/public_research/lakers_magic_extra_v1/audit_v1.json`.

### Post-audit media cleanup (2026-08-06)

After hashing, AGU removed two no-longer-useful MEM–OKC development files while keeping
their download-state provenance and all sealed evidence: `780Byt-iJyc.mp4` (SHA-256
`1caf38b7b9c1997805cf5e66a37a05af974fdb5d99dfaa8816f328710523bbb8`) and
`sOfCveR_usM.mp4` (SHA-256
`69bdd21fcc33f804403328180386819f0bd869bc0e31accd03bffcc673eab137`). The two verified
deletions reclaimed about 1.60 GiB; the still-needed `aU5JC_9RliM.mp4` blind source,
HOU–ORL media and both EBQwen weight variants were preserved.

### Basketball Events pinned annotation audit (2026-08-06)

AGU downloaded a bounded, hash-pinned slice of the public
[Basketball Events](https://huggingface.co/datasets/saveerjain/basketball-events) repository
at revision `26d3775286f542b41daf4a94a53190440d426111`: the upstream README, four game-level
annotation JSON files, and four representative 7-second MP4 clips. The source describes 543
clips and 897 events across four 2015–16 NBA games; the local files and their decode checks are
sealed in `dataset/public_sources/basketball_events_v1/manifest.json`. The approximately
7.87 GB full media collection was not downloaded.

The upstream terms say “For research purposes only” and provide no OSI or Creative Commons
license. The bounded slice is therefore retained for internal audit only
(`runtime_consumable=false`, `training_media_eligible=false`, not redistributed). It provides
short-window event semantics—shot type/make-miss, assist, block, rebound, jersey identity and
integer game clock—but no frame-level onset/outcome, ball/hand/rim boxes, possession labels or
camera-cut truth; `rebound` has no explicit offensive/defensive field. It is useful for event
vocabulary and parser checks, but is not counted as independent causal evidence and cannot
close the continuous-game 85% gate. No runtime, detector/VLM default, EBQwen weight or blind
asset changed.

### NBA Games HOU–SAC tiled-ball cross-broadcast screen (2026-08-06)

The bounded development copy is `q4cNj7ix58Q` (2024-12-03 HOU–SAC), 640×360 at 29.97 fps,
SHA-256 `f5961225d53ee113dbc94c3d635dbaffb07c0e5f13841f71f62559927297d189`. The source
manifest originally carried the wrong game ID; it now agrees with the local NBA metadata/PBP
(`0022400059`) and remains external-media, offline-only, non-redistributable, and
`runtime_consumable=false`.

On 600 samples from 300–600 seconds, the unmodified BODD full-frame pass produced 312 rim
detections and zero basketball detections. A label-free two-overlapping-horizontal-tile ball
pass at 960 input size recovered 198 deduplicated ball boxes and 96 short tracks, yielding 23
shot candidates. After raw contact-sheet review, a development-only first-quarter PBP alignment
(frames 8991–14300, before replay) gives TP/FP/FN `7/6/6`, P/R/F1 `0.5385/0.5385/0.5385`; the
full-frame baseline emitted no candidates. This is an input-scale diagnostic, not a training
label or acceptance benchmark. The sealed screen is
`analysis_outputs/public_research/hou_sac_2024_raw_review_v1/tiled_ball_screen_v1.json` (SHA
`68a1c4aeb1d8cd41d23dd35b8f96609447e05fa416eed0f4b4112996d7963c7e`). No detector default,
checkpoint, VLM, blind asset, or runtime path changed.

An independent pose follow-up used `model_checkpoints/yolo11n-pose.pt` only inside the same
candidate windows. It sampled 1,200 frames across 15 merged windows and attached 6,303 pose
detections; the guarded CPU run had exit code 0 and zero guard stops (peak system memory 79.6%,
peak CPU 69.7%, peak process-tree RSS 756,596,736 bytes). The pose-enriched bundle still has
23 shot candidates and the first-quarter lower-bound screen remains TP/FP/FN `7/6/6`,
P/R/F1 `0.5385/0.5385/0.5385`. The result is sealed in
`analysis_outputs/public_research/hou_sac_2024_raw_review_v1/pose_tiled_screen_v1.json`
(artifact SHA
`f5bda09a086702869433b81fb24e3a92e6b43be1b6e0139613ee2fc53e5ca5cb`) and is rejected for
training, runtime, VLM and blind use. The tested pure tile helpers in
`app/analysis/perception/tiled_ball.py` make crop projection and same-frame deduplication
reproducible without changing the runtime detector.

### Gated shot-outcome source audit (2026-08-06)

The [leharris3 basketball-shot-test-dataset](https://huggingface.co/datasets/leharris3/basketball-shot-test-dataset)
is labelled MIT on the Hub and is about 423 MB, but file access requires login/contact sharing,
the README is empty, and the page does not establish broadcast-media provenance or redistribution
rights. It therefore remains `research_reference_only_no_media_or_annotation_import`; no bytes
were downloaded. The already-audited [yerx/bb](https://huggingface.co/datasets/yerx/bb) has 200
free-throw clips but no dataset card, license or media provenance and remains catalog-only. Both
sources lack the continuous ball/hand/rim supervision and hard negatives required by AGU's gate.
The regenerated 50-source catalog has SHA-256
`86b2cac9d1e47abfe85da05995d152cedcb47ff6ff1c253e89453c15be3928ee`.

### E-BARD/MUVY tile-aware ball-head training screen (2026-08-06)

The already audited E-BARD/MUVY source manifest is CC BY 4.0 and is pinned by SHA-256
`ac9b1f72c573c1d13769dfb3c3a212803f67266e5d965daa3e9f63fc8dd761b7`. A deterministic two-tile
materialization produced 3,458 images (`train=2,916`, `val=542`) for an offline BODD four-class
head fine-tune. Three guarded CPU epochs in the canonical `.venv` reached final validation
P/R/mAP50/mAP50-95 `0.7280/0.5648/0.589/0.267`; peak system memory was 84.5% and no guard stop
occurred.

The trained tile specialist recovered 189 HOU–SAC ball boxes. With the frozen player/rim/referee
perception and the predeclared first-quarter interval, the candidate screen was TP/FP/FN
`7/3/6` (P/R/F1 `0.700/0.5385/0.6087`), better than the prior untuned tile screen's F1 but
below the gate. On the independent 32-window Lakers–Magic hard-negative set, any in-window ball
box yielded TP/FP/FN/TN `7/19/0/6` (P/R/F1 `0.2692/1.000/0.4242`). The route is therefore
`accepted=false`, `runtime_consumable=false`, and not a detector/VLM/runtime promotion.

The sealed evidence is `analysis_outputs/public_research/tiled_ebard_ball_train_v1/screen_v1.json`
(artifact SHA `3e10bcdeb02f1777973f8eed8d5861a32ec99d97dc212ee4b99cb658dba4d51f`). The generated
tile directory, checkpoint and external Ultralytics run were deleted after hash sealing; the
small perception/candidate/window evidence and screen-only scripts/tests remain. No blind media,
EBQwen native/MLX weights or runtime defaults were touched.

### Official WASB-SBDT DeepBall-Large basketball baseline (2026-08-06)

The official [WASB-SBDT repository](https://github.com/nttcom/WASB-SBDT) revision
`923462cacdeb3353b84ddebdedb3f4b7a8553b0f` publishes a DeepBall-Large basketball checkpoint
through its model-zoo Google Drive link. The downloaded file is `4,131,456` bytes with
SHA-256 `387878a2f45e6d9e1e3c2867c87f454b7d9676e6e546bd979301b52ec2832dfd`; the pinned model
source and configuration hashes plus the MIT code boundary are recorded in
`model_checkpoints/wasb_sbdt_official/deepball_large_basketball_manifest.json`. The upstream
code license does not grant rights to the source training media, so this is an offline model
comparison only (`runtime_consumable=false`, `training_eligible=false`).

The new hash-checked CPU adapter is covered by `tests/test_deepball_large.py` and the fixed-window
screen by `tests/test_deepball_large_windows.py`. At the predeclared threshold `0.50`, the HOU–SAC
development windows scored TP/FP/FN/TN `6/1/1/2`, P/R/F1 `0.8571/0.8571/0.8571`; a development-only
threshold `0.70` sweep fell to `5/1/2/2`, so `0.50` was frozen before the held screen. The
independent Lakers–Magic windows scored TP/FP/FN/TN `6/14/1/11`, P/R/F1 `0.3000/0.8571/0.4444`.
The route improves visible-ball recall but does not reject ordinary live possession/non-shot
windows and fails the cross-broadcast 0.85 gate.

The sealed screen is
`analysis_outputs/public_research/deepball_large_v1/screen_v1.json` (canonical artifact SHA
`9ccb50cef3d0762a89d42fc620218d1a3de3a1768151ca6a0f3215b575c67674`). The 3.9 MiB checkpoint is
retained as a future offline fusion baseline; it is not a runtime default, VLM evidence source or
blind-game truth, and `accepted=false`.

### Lakers–Magic independent VLM and fixed DeepBall fusion screen (2026-08-06)

The 32-window Lakers–Magic candidate bundle was converted directly into a label-hidden raw-frame
plan; `event_present` and review notes were not exposed to the model. The sealed 4-frame/512-pixel
plan file SHA-256 is `dc173f2e03232b727f9dd7301d838fa1c6b11513503e399f9fb762df3a11859`.
The already-local Apache-2.0 `qwen3-vl:2b` Ollama asset was used only as an offline independent
reviewer (digest `0635d9d857d497aeadba3d7d27485746c50554446f9f6ec01ef39788221adbe8`, weights
source SHA `ebabfa59b71a5b96e0281ec2994977e785284e0939807a99fc340dec3c6f10de`).

The strict-release run completed all 32 windows with no resource-guard stop. Peak system memory,
CPU and process-tree RSS were `88.3%`, `100.0%` (one non-consecutive startup sample) and
`255,344,640` bytes; minimum available memory and swap were `2,015,395,840` and `328,138,752`
bytes. Every VLM decision was `live_field_goal`; delayed labels give TP/FP/FN/TN `7/25/0/0`,
P/R/F1 `0.2188/1.0000/0.3590`. A fixed, predeclared `VLM AND DeepBall-Large` rule is identical
to DeepBall-only (`6/14/1/11`, P/R/F1 `0.3000/0.8571/0.4444`), while OR is identical to VLM-only.
The sealed offline fusion artifact is
`analysis_outputs/public_research/lakers_magic_extra_v1/independent_shot_vlm_qwen3vl2b_deepball_fusion_screen_v1.json`;
it is not training truth, runtime input or a blind-game answer (`accepted=false`,
`runtime_consumable=false`). No EBQwen native/MLX asset or runtime default changed.

### BasketEvent player-grounded trajectory slice (2026-08-06)

The public [zaywas/BasketEvent](https://huggingface.co/datasets/zaywas/BasketEvent) release was
fixed to revision `85aaa3ce62bc096e3995c39ecfa6773fcc9fe5e1`. Its paper describes player-grounded
NBA broadcast event understanding with player and ball trajectories; the current Hub page has no
dataset card or explicit license. The repository contains JSON trajectory files and a `playnet.pt`
checkpoint, but no raw video was imported and the checkpoint was not downloaded.

We retained only a bounded annotation slice: all 557 `valid` JSON files (`85,794,901` bytes) and a
deterministic round-robin sample of 120 `test` JSON files across 37 games (`18,911,750` bytes).
Each file was checked against the Hub API size and SHA-256 before retention. The manifests are
`dataset/public_sources/basketevent_v1/manifest.json` and `test_manifest.json`; the sealed audits
are `analysis_outputs/public_research/basketevent_v1/audit.json` and `test_audit.json`. The valid
slice contains 5,588 player tracks, 1,143,488 player boxes and 543 ball tracks (14 clips omit the
`ball` key). The test slice contains 1,135 player tracks, 25,648 visible ball boxes and 120 event
clips spanning made/missed shots, rebounds, turnovers, fouls, free throws, assists, blocks and
steals.

This is useful only as offline structured auxiliary supervision for player/ball relations and event
taxonomy. The JSON does not provide redistributable raw video, rim/hand boxes or verifiable frame-
level event intervals, and no explicit dataset license is declared. The catalog therefore sets
`runtime_consumable=false` and `training_media_eligible=false`; no runtime detector/VLM/EBQwen
default or blind asset changed, and the cross-game causal ball–hand–rim gate remains unresolved.

### LAL–ORL Game 3 cross-broadcast raw-window screen (2026-08-06)

The fixed NBA Games metadata revision `3a20f2b9f60a025c8c641f4c763d4ac467ff059e` points to
2009-06-09 Game 3 (`0040800403`, YouTube `Agffi33pz8w`). The bounded local media is
`dataset/public_sources/nba_games/media_new_dev_v2/Agffi33pz8w.mp4`: 640×350, 25 FPS,
8546.04 seconds, 579,857,484 bytes, SHA-256
`ecbc705c03044ec33dfa16aa50151ebcb5436b213106239ffba7d28c8ff9d04c`. The source manifest
records the NBA PBP path and the external-platform/no-redistribution boundary; it remains
`runtime_consumable=false` and `training_eligible=false`.

`select_uniform_shot_windows` now spreads seeded slack over every gap, preventing a 32-window
sample from drifting to the end of a long pregame/game/postgame upload. The label-hidden spec
(`817b3e7a1de68336e862c57e0782b968201a970b87f0ac0c99683786f2574f95`) was reviewed from raw
contact sheets before any PBP access. It contains 1 complete live field-goal attempt and 31
hard negatives (pregame, replay, free throw, close-up, dead ball or no-release possession).

The held-game scene+MViT screen is TP/FP/FN/TN `1/9/0/22` (P/R/F1
`0.1000/1.0000/0.1818`). The locked official DeepBall-Large detector is `1/14/0/17`
(P/R/F1 `0.0667/1.0000/0.1250`). Adding the 32 windows as an offline augmentation to the
existing Lakers–Magic corpus leaves the Lakers–Magic held result at `0.4000/0.8571/0.5455`.
These outputs are diagnostics only: no checkpoint, detector/VLM default or runtime path was
promoted, and blind inference remains paused. The persistent data blocker is still licensed,
broadcast-diverse, sub-second ball–hand–rim outcome supervision with exhaustive non-shot hard
negatives.

The same 32 windows were then screened by the independent local `qwen3-vl:2b` reviewer using
the strict release contract (four chronological frames at 512 px). The model input was label-free
and the prediction artifact was sealed before it was used for metric computation; the plan contains
all one positive plus 31 negatives. All 32 model outputs were `live_field_goal`. Evaluation is
TP/FP/FN/TN `1/31/0/0` (P/R/F1
`0.03125/1.0000/0.0606`). The fixed `VLM AND DeepBall-Large` rule is `1/14/0/17`
(`0.0667/1.0000/0.1250`), so this independent fusion is rejected and remains offline-only.
The plan, prediction, evaluation and fusion artifacts are under
`analysis_outputs/public_research/lal_orl_game3_v2/`; no runtime/VLM default, checkpoint,
EBQwen asset or blind asset changed.

### LAC–DAL 2024 independent cross-broadcast raw-window screen (2026-08-06)

The pinned [NBA Games](https://huggingface.co/datasets/choucsan/NBA_Games) revision
`3a20f2b9f60a025c8c641f4c763d4ac467ff059e` supplies metadata/PBP for the 2024-12-21
[LAC–DAL game](https://www.nba.com/game/lac-vs-dal-0022400385). Its external
[YouTube video](https://www.youtube.com/watch?v=waBDFY4ihS0) was downloaded at H.264 640×360,
29.97 FPS, 175,259 frames and 390,529,704 bytes (SHA-256
`9c75972c7b04fef318619a9f0e43a582b793b912bed7ccb56188456aa4e2277e`). The source manifest is
`dataset/public_sources/nba_games/media_new_dev_v3/source_manifest_v1.json`; metadata follows
the dataset's MIT documentation, while the external media is retained locally only and is not
redistributed. It is explicitly `runtime_consumable=false` and `training_eligible=false`.

A seeded, label-hidden sampler produced 32 non-overlapping 120-frame windows. Delayed raw-frame
review found 9 complete visible releases and 23 context/replay/setup/no-release hard negatives.
The sealed candidate spec, raw-only bundle, labels and training manifest are in
`analysis_outputs/public_research/lac_dal_2024_v3/` (spec internal SHA
`1df38be93ccc40fbfad4ae716954a9169f9b8b3c8d0e917c765628c72018ea51`, bundle internal SHA
`b00d47bc5e24bb9c01e7c4a2b71a93ac28e09cda85f7d8a651521cf60c886eff`, labels file SHA
`2a96d3013e690f898fc8a05b6eef007f99e8dae0e1bb803d546e6417331f6cae`, manifest internal SHA
`7b388575f164a9ff4d40c9a77b807e208bf882d15db7e461151923c20f307395`).

The locked DeepBall-Large screen is TP/FP/FN/TN `7/15/2/8` (P/R/F1
`0.3182/0.7778/0.4516`). Independent local `qwen3-vl:2b` received only four chronological
512-pixel frames and returned `live_field_goal` for all 32 windows; its delayed evaluation is
`9/23/0/0` (P/R/F1 `0.2813/1.0000/0.4390`). Predeclared AND/OR fusion is respectively identical
to DeepBall-only/VLM-only and is rejected. The screen and resource logs remain offline evidence;
no runtime, EBQwen native/MLX asset, detector/VLM default or blind asset changed. The source does
not resolve the licensed sub-second ball–hand–rim outcome and exhaustive hard-negative gate.

### Basketball Events shot-positive VLM audit increment (2026-08-06)

The bounded research-only source was extended with 16 single-shot clips (two makes and two
misses per game across four pinned 2015–16 games). The media subset is
`dataset/public_sources/basketball_events_v1/shot_subset_v1/` (226,664,456 bytes; manifest
SHA-256 `1bf49905d97fd3ce7e2411da69eb92d44d9dc3b7efb400f569b24b38bd8cb451`). The source terms
remain “For research purposes only”; no clip is redistributable, runtime-consumable or training-
eligible. The 16 clips have one upstream shot event each but no frame-level release annotation.

A label-free 4-frame/512px strict-release Qwen3-VL run recalled all 16 event-centered positives.
That positive-only probe is not an acceptance metric. Combining it with the fixed LAC–DAL screen
(9 positives, 23 hard negatives) gives 48 windows with TP/FP/FN/TN `25/23/0/0`, P/R/F1
`0.5208/1.0000/0.6849`; the continuous LAC–DAL subset remains `9/23` precision. The summary
artifact is `analysis_outputs/public_research/basketball_events_shot_vlm_v1/screen_summary_v1.json`
(internal SHA-256 `93417db262438bd3a421da11aa9e86fe8ff58e2e6e2a2fd59fdb22e52ec0ee67`). This is an
offline model-diagnostic only: `accepted=false`, `runtime_consumable=false`, and the causal
ball–hand–rim plus exhaustive hard-negative gate is unchanged.

### LAC–DAL EBQwen native-video resource probe (2026-08-06)

To separate model choice from frame-sampling effects, a four-window prefix was derived from the
same label-free LAC–DAL plan. The local `EBQwen2.5-VL-3B-MLX-4bit` model was pinned to upstream
revision `c6a93cbb325f9d20236f85bcaba7827a2808443e` and weight SHA-256
`f2b3cf74e088a06b7bf1c6462664ca13f95e1c1c8be4c6c8a0e469f68dd1d72c`; input used native temporal
encoding, 2 FPS, a 151,200-pixel budget and the frozen `strict_release_v2` prompt. Labels and
review notes were withheld until prediction sealing.

All four outputs were `live_field_goal`. Delayed evaluation is TP/FP/FN/TN `1/3/0/0`,
precision/recall `0.25/1.00`, `promotion_eligible=false`; the prefix therefore was not expanded
to 32 windows and cannot be used as runtime or training truth. The plan, predictions, evaluation
and resource log are retained under
`analysis_outputs/public_research/lac_dal_2024_v3/independent_shot_vlm_native_ebqwen_probe4_*`;
all artifacts are offline-only. The guard exited 0 with zero stops (peak system memory 86.5%,
minimum available memory about 2.16 GiB). No runtime, default VLM, checkpoint, EBQwen asset or
blind media changed; the cross-broadcast causal/hard-negative gate remains closed.

### BasketEvent PlayNet bounded compatibility probe and cleanup (2026-08-06)

The official code revision `8a313f3ad4476735ddac38543578e19c1bccebd5` and Hub revision
`85aaa3ce62bc096e3995c39ecfa6773fcc9fe5e1` were pinned. The 1,813,273,995-byte `playnet.pt`
checkpoint was downloaded once, SHA-256 verified as
`179adc633166e435f19b75fa60625f3ff57c1ae45c7160d3b2c08a6cc96ea287`, and tested against a
temporary offline wrapper of the published `PlayerEventModel` in canonical `.venv`. The eight-
player/eight-frame example loaded with zero missing and zero unexpected state keys and completed
a CPU forward at checkpoint epoch 14. Its ball trajectory has zero valid frames and no matching
event gold label, so this is interface evidence only, not an accuracy or training result.

The upstream code, dataset and checkpoint expose no explicit license, and the Hub contains no
matching raw videos or frame-level event truth. The checkpoint was therefore deleted after the
probe; only `dataset/public_sources/open_models/playnet_basketevent/source_manifest.json` and
`analysis_outputs/public_research/basketevent_playnet_probe/` remain. The source stays
`runtime_consumable=false` and `training_media_eligible=false`; no runtime, detector/VLM default,
EBQwen asset or blind inference changed, and the licensed continuous causal/hard-negative gate
remains closed.

### 2026-08-06 online candidate gate: VC_NBA_2022, FineSports and NSVA

The paper and project pages were rechecked before any new payload download. The [VC_NBA_2022
paper](https://arxiv.org/abs/2401.13888) describes 25 NBA full games collected from the
commercial Fishker NBA platform and a 3,977-clip shot/rebound caption subset, but says the
dataset and code would be made public soon; no public data archive or compatible dataset license
was found. It remains a paper-only reference despite its nine shot/make/miss/rebound labels.

The [FineSports repository](https://github.com/PKU-ICST-MIPL/FineSports_CVPR2024) confirms that
its 10,000 clips and `FineSports-GT.pkl` are released only after signing and emailing a release
agreement. The repository has no separate code license, so the useful make/miss/free-throw
taxonomy cannot be imported or downloaded under the AGU source gate.

The [NSVA repository](https://github.com/jackwu502/NSVA) does publish action lists and metadata,
but its README assigns CC BY-NC to most code/data, asks users to ensure fair use for NBA media,
and points raw-video retrieval at NBA.com. It is therefore taxonomy/reference material only,
not an open AGU training dependency. These findings are encoded as the `vc-nba-2022` catalog
entry; catalog SHA-256 is `a1a2e072524bd908b22dc0883544d41c97a27b2a31d0b129f61386d0c56bbcc5`.
No media, checkpoint or runtime default changed in this audit.

### BQwen2.5-VL-3B BARD bounded probe and deletion (2026-08-06)

The public BARD `BQwen2.5-VL-3B` checkpoint was pinned to revision
`50f97bce9bf1709d68fcc7e131d26ad3c4b7d82c` under its CC BY 4.0 model card. Both BF16 shards
were downloaded and hash-verified (`7,509,337,944` bytes total; SHA-256
`53b17ef859254f2914788478ae84a7d4abcd60f743d163ac4570521c82445fbd` and
`0d880f6695d68bb36478865ef2c4c0c185ad16a729a11a137aadef88a890b085`). Existing EBQwen base
processor/tokenizer files were reused only to complete the identical Qwen2.5-VL format. A
canonical `.venv` resource-guarded conversion produced an MLX 4-bit file of `3,073,719,979`
bytes (SHA-256 `a66f8b3779a39fc89787f3d6f6a1ce6adf5de27fba89a45022a7bfd8f68384aa`).

The model then received only four raw LAC–DAL windows from the sealed native-video plan (2 FPS,
151,200-pixel budget, `strict_release_v2`); labels and review notes were withheld until sealing.
All four predictions were `live_field_goal`, giving delayed TP/FP/FN/TN `1/3/0/0`, precision
`0.25`, recall `1.00`, exactly matching the EBQwen prefix and failing the `0.95/0.85` gate.
The guard exited 0 with zero stops (peak memory 88.1%, minimum available memory 2,050,588,672
bytes, peak CPU 34.6%). The BF16 and MLX weight payloads were deleted after the failed probe;
only compact provenance/configuration and offline artifacts remain under
`dataset/public_sources/open_models/bqwen25_vl_3b_bard/` and
`analysis_outputs/public_research/bqwen25_vl_3b_bard_v1/`. The candidate is not a training or
runtime dependency, and the rights-cleared continuous causal/hard-negative gate remains open.

### Basketball-51 bounded make/miss sample and MViT screen (2026-08-06)

The [Basketball-51 Kaggle card](https://www.kaggle.com/datasets/sarbagyashakya/basketball-51-dataset)
declares 10,311 roughly six-second 320×240 broadcast clips in eight labels (`2p0`, `2p1`,
`3p0`, `3p1`, `ft0`, `ft1`, `mp0`, `mp1`) and an Apache 2.0 uploader license. The uploader
license does not establish redistribution rights for the underlying NBA broadcasts, so AGU
used ZIP64 HTTP ranges to extract only 32 clips (four source-game tokens × eight labels,
19,867,852 bytes total). Every member passed ZIP CRC/size/SHA checks and ffprobe (H.264,
320×240, approximately six seconds). The sample manifest is training-only and explicitly
`runtime_consumable=false`, `training_eligible=false`.

With the existing MViT-V2-S backbone, source-game-disjoint embeddings separate field goals from
free throws at worst-fold precision/recall `0.8571/1.00`, but made-vs-missed falls to worst-fold
precision/recall/F1 `0.50/0.25/0.3333` and eight-class macro-F1 to `0.0625`. A guarded three-epoch
one-block tail fine-tune on the same source sample has held `v112` precision/recall/F1 `0/0/0`;
the training script emitted no checkpoint and exited 2. Compact clips, embeddings, screen,
resource traces and rejected metadata remain under
`dataset/public_sources/basketball_51_v1/` and
`analysis_outputs/public_research/basketball_51_v1/`; no runtime or blind asset changed.

### Basketball-51 expanded source-group screen (2026-08-06)

To rule out a four-game sampling accident, AGU repeated the same ZIP64 Range procedure with a
fixed seed and 16 clips per label. The bounded v2 sample contains 128 H.264 clips from 46 source
game groups (`80,328,593` bytes); labels stayed withheld from the embedding and screen commands,
and the uploader's Apache 2.0 statement remains separate from underlying broadcast rights.

The source-group MViT screen improves field-goal vs free-throw separation to worst-fold
precision/recall/F1 `0.9565/0.9565/0.9565`, but made-vs-missed remains only
`0.5455/0.375/0.48`, with eight-class worst-fold macro-F1 `0.1667`. This is retained only as a
narrow free-throw auxiliary diagnostic; no second tail fine-tune or checkpoint was emitted, and
the sample remains `runtime_consumable=false` and `training_eligible=false`. Evidence and the
source manifest are under `dataset/public_sources/basketball_51_v2/` and
`analysis_outputs/public_research/basketball_51_v2_*`.

### F-16-NBA shot-test online candidate gate (2026-08-06)

The [F-16-NBA Hugging Face card](https://huggingface.co/datasets/tsinghua-ee/F-16-NBA) declares
Apache-2.0 and describes a shot-test split with 1,970 `Yes`/`No` clips from 16 NBA games. A
bounded HTTP Range audit read only the first `37,830,656` bytes of the first tar shard and
verified eight H.264 1280×720, 60-fps clips of approximately 8–9 seconds. The compact audit binds
the card revision, JSON SHA, shard object IDs, and sample media SHA/ffprobe results.

This is a direct outcome-only candidate, but it provides no sub-second ball–hand–rim timestamps,
no exhaustive non-shot hard negatives, and no separate grant of underlying NBA broadcast
redistribution rights. AGU therefore deleted the temporary prefix, did not download the remaining
14.7 GB shot archive, and did not run training, VLM or fusion. The candidate remains
`runtime_consumable=false`, `training_eligible=false`; provenance is retained at
`dataset/public_sources/f16_nba_shot_test/source_manifest.json` and
`analysis_outputs/public_research/f16_nba_shot_test_online_gate_v1.json`.

### Basketball_Detection static ball/rim candidate (2026-08-07)

The pinned [Basketball_Detection repository](https://github.com/tranvietcuong03/Basketball_Detection)
(`c6d4583974f412f8b40215298234e6c13a0da30a`) contains 8,521 train, 812 validation and 406 test
images with YOLO classes `ball`, `made`, `person`, `rim` and `shoot`. AGU inspected only the
GitHub tree and `dataset.yaml`; no image bytes were downloaded. The API reports no repository
license and the tree has no `LICENSE` file, while the paper does not grant underlying-image
redistribution rights.

The release is static detection only: it has no frame timestamps, ball–hand–rim ordering,
shot-outcome sequence or exhaustive continuous non-shot hard negatives. It is therefore cataloged
as `research_reference_only_no_media_or_annotation_import`, with both
`runtime_consumable=false` and `training_media_eligible=false`. Retained provenance is limited to
`dataset/public_sources/basketball_detection_github/source_manifest.json` and
`analysis_outputs/public_research/basketball_detection_github_audit_v1.json`; the catalog SHA is
`bd18e4fbd6cd344994b726f94b39562b6326fed9c81e203c5788fdb7cfa92339`.

### Qwen3-VL-4B resource-fit screen (2026-08-07)

AGU pulled the official [Ollama qwen3-vl:4b](https://ollama.com/library/qwen3-vl:4b) asset
(`1343d82ebee3`, weight SHA-256
`9c60bdd691c1897bbfe5ddbc67336848e18c346b7ee2ab8541b135f208e5bb38`) only for a label-hidden
LAC–DAL independent VLM screen. The canonical `.venv` guard stopped before the first prediction
after three consecutive samples below the 2.5 GiB available-memory floor (16 GiB host; observed
available memory 3.83, 2.20, 1.81 and 1.61 GiB; exit code 75). The run has no prediction or
accuracy claim: `completed_windows=0`, `promotion_eligible=false`.

The 3.3 GB local Ollama payload was removed after the resource-fit failure. The compact screen and
hash-bound guard log remain at
`analysis_outputs/public_research/lac_dal_2024_v3/independent_shot_vlm_qwen3vl4b_4f512_screen_v1.json`
and `analysis_outputs/public_research/lac_dal_2024_v3/resource_independent_shot_vlm_qwen3vl4b_4f512_v1.jsonl`.
The model was never made runtime-consumable; the remaining local VLM research asset is Qwen3-VL-2B,
while the EBQwen native/MLX weights and AGU runtime checkpoint remain untouched.

### SportsShot online candidate gate (2026-08-07)

The [MCG-NJU/SportsShot card](https://huggingface.co/datasets/MCG-NJU/SportsShot) is pinned to
revision `2fca05a9c49366d2c52b4d90e322a7a83e634262` and declares CC BY-NC 4.0. It describes 1,200
high-definition clips (400 each for basketball, football and volleyball) with frame-level shot
segmentation and shot-boundary labels, totalling about 183 GB. The files are gated behind contact
information; an anonymous metadata endpoint returned `401 GatedRepo`, so AGU downloaded no media or
annotation archive.

Shot segmentation does not provide ball–hand–rim ordering, made/missed outcomes or exhaustive
non-shot hard negatives. The source is therefore cataloged as `metadata_only_until_access_and_causal_label_audit`
with `runtime_consumable=false` and `training_media_eligible=false`; the compact audit is
`analysis_outputs/public_research/sportsshot_online_gate_v1.json`. The source catalog now contains
55 entries with SHA-256 `0dd4e24c7502a22775d3110e1fcf3d0d83849064a586db2698c005a16fbd4eb6`.

### strict-release-v3 consistency screen (2026-08-07)

The offline prompt variant `strict_release_v3` adds a chronology and contradiction check to the
existing four-frame/512-pixel Qwen3-VL-2B screen. It requires visible control, release and
toward-rim motion in order and forbids a positive state when the textual reason says that release
is not visible or the ball is already in flight. The prompt is label-free and development-only;
AGU runtime does not consume it.

The frozen LAC–DAL plan completed all 32 windows under the canonical `.venv` guard (exit 0,
minimum available memory about 1.96 GiB, peak system memory 87.8%, peak CPU 39.8%). Delayed
evaluation is unchanged at TP/FP/FN/TN `9/23/0/0`, P/R/F1 `0.2813/1.0000/0.4390`. Three rows
contain a positive state with a contradicted required observable; conservative offline
normalization changes the diagnostic to `9/20/0/3`, P/R=`0.3103/1.0000`, still below the
`0.95/0.85` gate. Predictions, evaluation, resource log and consistency audit are retained under
`analysis_outputs/public_research/lac_dal_2024_v3/independent_shot_vlm_qwen3vl2b_4f512_v3_*`;
all remain `runtime_consumable=false` and are not training truth. No model weight, runtime default,
checkpoint or blind asset changed.

### NUS Basketball Detection online candidate gate (2026-08-07)

The pinned [NUS Basketball Detection Hub release](https://huggingface.co/datasets/linhuaian3/nus-basketball-detection-cs5260)
(`3ee1a0decfde199117b3a99d79bcf136c902ebc5`) contains 719 MP4 clips (`2,967,530,545` bytes)
organized by shot type and make/miss/background. A fixed 12-clip sample (`37,378,249` bytes)
was downloaded, SHA/size checked against the Hub LFS metadata, and decoded as 640×360 H.264
broadcast footage at about 30 fps. The sample was moved to
`/Users/ppt/.Trash/agu-nus-basketball-20260807` after review.

The source has no README, license tag, card metadata, LICENSE file, source-game provenance or
underlying broadcast redistribution grant. Folder names expose labels, and the clips contain no
frame-level release/impact/outcome timing, ball–hand–rim annotations or exhaustive non-shot hard
negatives. It is therefore catalog-only (`runtime_consumable=false`, `training_media_eligible=false`)
with audit `analysis_outputs/public_research/nus_basketball_detection_audit_v1.json` (artifact SHA
`b10767fc5dee4d7122640705247cb50760e7db6dff3066ce74691df4f0f3d6b9`). The regenerated catalog has
56 sources and internal SHA `f947fb5c702aa36de6ec631ddd5fca32efb63f6435e893860f85885e0cbc94ca`;
no runtime, checkpoint, VLM default, EBQwen asset or blind media changed.

### PL-NBA possession-level temporal annotation gate (2026-08-07)

The pinned [PL-NBA repository](https://github.com/holhouse/PL-NBA-Dataset) is fixed to main commit
`0b3d5013b5828004062ed27b10c9eddfca313711`. AGU downloaded only the 4,461,082-byte
`PL-NBA-JSON.zip` annotation archive (SHA-256
`391948eef9fd8b4cecfc58a6088edf1253b0ce4bc5531edd7147d15a8f94ee48`), not the original NBA games
or the separately linked trimmed-video host. The archive parses into 32 game directories, 6,408
possession JSON files and 31,964 events, including 3,607 `2ptShot`/`3ptShot`/`FreeThrow` events
with outcome labels and temporal spans. It is retained as a compact offline event-taxonomy reference.

The upstream README declares CC BY-NC 4.0 for academic research, but the archive contains no paired
video and no independently verified grant for redistribution of the underlying NBA broadcasts. It
also lacks pixel boxes, sub-second ball–hand–rim causal order and exhaustive full-game non-shot hard
negatives; four intervals are non-monotonic, four rows contain unknown `nan` labels/results, and
filenames/possession IDs leak outcome tokens. The candidate is therefore registered as
`runtime_consumable=false` and `training_media_eligible=false`; keep only
`dataset/public_sources/pl_nba_v1/PL-NBA-JSON.zip`, its manifest and
`analysis_outputs/public_research/pl_nba_annotation_audit_v1.json`. The source catalog now contains
57 entries with internal SHA-256
`93542b43e62169f7c25946f3dc910fbb46f9cfe8f76f643247566f88d03efb9f`.

### TAL3x3 temporal action annotation gate (2026-08-07)

The pinned [TAL3x3 repository](https://github.com/open-starlab/TAL3x3) is fixed to commit
`f1093d62818c3e8930561fda5939db19c0981d95`. AGU inspected the repository and its public Google Drive
folder, then downloaded only the 5,094,624-byte `dataset.zip` annotation archive (SHA-256
`0dbcf92e866fd8b8428f297b48c80e4aae603a1845d432d27bc5e669ec3db28f`); the separate 271.6 MB
`skeletons.zip` and all video files were not downloaded. The archive contains 318 event clips from
10 3x3 source videos, 1,881 frame-bounded events, 306 clips containing shot events, 387 2P/3P/free-
throw outcome events and normalized person boxes for 105,589 frames.

The GitHub tree has no LICENSE and the Drive archive has no dataset rights statement. The paper is
CC BY 4.0, but that publication license is not treated as a grant for the underlying source video or
archive. The bounded archive has no paired video, ball boxes, ball–hand–rim causal order or exhaustive
full-game non-shot hard negatives, and its 3x3 action-localization scope is not a 5x5 broadcast gate.
Keep only `dataset/public_sources/tal3x3_v1/dataset.zip`, its manifest and
`analysis_outputs/public_research/tal3x3_annotation_audit_v1.json`; the source is
`runtime_consumable=false` and `training_media_eligible=false`. The catalog now contains 58 entries
with internal SHA-256 `460aca7b4a3f72d33666776dcc371854a2741d21290f176dfcf27b072836c5f4`.

### MUVY bounded multi-view basketball sample (2026-08-07)

The [MUVY Zenodo record](https://zenodo.org/records/13883315) is a CC BY 4.0 release
(DOI `10.5281/zenodo.13883315`, 2024-10-03). Its 7,758,643,869-byte ZIP was not downloaded in full:
AGU extracted 26 basketball metadata files by ZIP64 Range and retained only two cameras from
`basketball_event_01` (34,211,695 bytes total). The two videos contain 2,033 and 1,776 frames at
about 68 and 59 seconds; their object annotations include 232 and 48 ball rows. Audio-energy
cross-correlation provides a diagnostic 32.7-second relative offset, not a shot timestamp.

MUVY supplies object boxes and multi-view grouping, but no shot start/end, release/rim/result timing,
made/missed labels, player identity, continuous 5x5 coverage or exhaustive non-shot hard negatives.
The event/video titles also expose “winning shot”. Keep it as a bounded offline multi-view ball-box
and synchronization reference only: `runtime_consumable=false`, `training_media_eligible=false`,
and no AGU runtime, model, training truth or blind asset may consume it. The audit is
`analysis_outputs/public_research/muvy_basketball_audit_v1.json` (artifact SHA-256
`cf6ac265f3cf401de4884b7b26c8ff0b375b3eb47a85e303ae43dcab54f72cdf`); metadata and media manifests
are `dataset/public_sources/muvy_v1/metadata_manifest.json` and
`dataset/public_sources/muvy_v1/media_manifest.json`. The catalog now contains 59 entries with
internal SHA-256 `d9b5b85feedf1309e7b5aa0d2c9b2bc978b3e93fee88ff63eb29133b61ba18c2`.

### Basketball Events annotation re-audit (2026-08-07)

The pinned [Basketball Events dataset](https://huggingface.co/datasets/saveerjain/basketball-events)
revision is `26d3775286f542b41daf4a94a53190440d426111`. Its tree has 564 entries (558 files, 543 MP4s)
across four NBA games; the upstream media inventory is `7,864,277,848` bytes. The bounded local
annotation set contains 543 clips and 897 events: 601 shots (259 makes/342 misses), 153 assists,
110 rebounds and 33 blocks. All clips are eventful; 58 duplicate event tuples arise from overlapping
clips. The train/val paths are disjoint, but both splits contain all four games.

The integer clock index matches 776/897 events and places every matched index inside its 7-second clip
(median offset 5 seconds, maximum 7 seconds). It is not frame-level causal evidence: there are no
ball/hand/rim/player boxes, release/contact/result timing, possession or exhaustive non-event hard
negatives. Eight bounded sample clips decoded successfully, were hashed and then permanently cleaned;
the duplicate metadata copy and other rejected candidate duplicates were also removed from the project.
Keep the existing bounded annotation and shot-subset manifests only for offline audit;
the source remains `runtime_consumable=false`, `training_media_eligible=false`, and is not training or
blind truth. The complete re-audit is
`analysis_outputs/public_research/basketball_events_reaudit_v1.json` (artifact SHA-256
`f4e19cbf3958fb5ee9543943ecee97abc75165c2affc52df4ee0d1c93943b776`). The regenerated catalog has
59 sources with internal SHA-256
`872093931962865a9ea9b10ea947e19e94787cb4116cf5e1873a4aafcfe6e00`.

### NBA Games full-game video index gate (2026-08-07)

The pinned [NBA Games dataset](https://huggingface.co/datasets/choucsan/NBA_Games) revision is
`cf59e3a42413e3ab91f6fc0b1618f280df9a7024`. Under its declared MIT metadata license, AGU retained the
189-game index plus all 189 games' box-score and play-by-play JSONL (63,619,934 bytes); no YouTube video
was downloaded. The release links full-game YouTube references to NBA.com official game data and reports
189 verified games, about 347 hours of linked video, 5,194 box-score rows and 81,355 PBP rows, with 166
non-empty PBP games and 23 empty ones.

The PBP contains 12,659 made-shot and 15,030 missed-shot rows, but these are official action metadata,
not visual labels. No local frames, independently verified underlying-video rights, release/contact/result
timing, exhaustive non-shot negatives or cross-broadcast split are present; `videoAvailable` flags do not
prove local media. Keep the source as an offline full-game/PBP alignment index only, with
`runtime_consumable=false` and `training_media_eligible=false`. Manifest:
`dataset/public_sources/nba_games_v1/manifest.json`; audit:
`analysis_outputs/public_research/nba_games_fullgame_audit_v1.json` (artifact SHA-256
`7fc8df8aeb002962743f08bed240c27890f31e714183d699f57ae954bee9091a`). The catalog now has 60 entries
with internal SHA-256
`df1c33d609b6690f74e22b9abd89c3e07ae24924d9421bbda09b2f794bc82976`.

### Independent VLM evidence-gate integrity hardening (2026-08-07)

The offline evidence gate now fails closed on incomplete or malformed evidence: every evaluated VLM
event must have an auxiliary out-of-fold row, required observable names must be unique, and auxiliary
probabilities/thresholds must be finite values in `[0, 1]`. A non-finite or out-of-range live VLM
confidence is abstained as `unknown`. Extra auxiliary rows remain permitted only for held-group
threshold selection; they do not silently create evaluation coverage.

This is a contract/test hardening change, not a dataset or model promotion. The retained EBQwen +
Swin/MViT screen remains TP/FP/FN/TN `13/9/3/7` with P/R/F1 `0.5909/0.8125/0.6842` and is still
rejected. No runtime, training, VLM default, weight or blind asset changed.

### Shot-causal support threshold protocol correction (2026-08-07)

The first causal-support screen had a threshold-selection bug: the outer-fold training score buffer
was still all zeros when the causal threshold was selected. The offline implementation now computes
leave-one-training-game-out scores inside each outer fold, and regression coverage rejects a zero-filled
selection buffer. The corrected artifact is
`analysis_outputs/public_research/shot_causal_support_screen_v2.json` (artifact SHA-256
`f16f34a1049b606a51cd873ffe683d09f1b1fa49ba834aa339d91ada907a7043`; file SHA-256
`15af30f016ccf821d611eee1ef1f021089f14102d14ce968c3e375a399f7c427`). Its frozen 511-row baseline is
P/R/F1 `0.791667/0.846154/0.818004`; corrected causal-support is `0.557562/1.000000/0.715942`, with
weakest-game precision `0.414634`, so it remains rejected and offline-only. The v1 artifact remains only
as historical audit evidence and is not used for promotion or runtime answers.

### HOU–SAC tiled ball candidate VLM probe (2026-08-07)

The retained HOU–SAC Q1 development media was used for a label-hidden, offline candidate-ball audit.
Forty-eight confidence-stratified raw-frame candidates were reviewed as `11 valid_ball`, `2 uncertain`
and `35 false_positive`. A local EBQwen2.5-VL-3B MLX 4-bit probe then selected two candidates per
confidence band (12 predictions); every prediction was `uncertain`. The sealed evaluation at
`analysis_outputs/public_research/hou_sac_2024_raw_review_v1/q1_300_600/ball_candidate_review_v1/`
has lower/upper precision and recall `0/0`, `accepted=false`. The run used the canonical `.venv`
resource guard without a stop/retry (process-tree RSS peak about 584 MiB; system memory peak 86.6%;
CPU peak 33.4%; minimum available memory about 2.15 GiB). This is a rejected detector/VLM evidence
probe only; no detector, runtime, weight, training truth or blind asset changed.

### E-BARD candidate verifier transfer to HOU–SAC (2026-08-07)

AGU reused the hash-bound E-BARD candidate manifest (`3,675` examples: `1,348` positive,
`2,327` negative, `60` game groups) and the sealed HOU–SAC tiled perception/review plan. No new
model or dataset bytes were downloaded. MobileNetV3-small exact-crop features were evaluated with
5-fold game-held OOF threshold selection on the source, then applied without opening target labels
until the sealed review join.
This is a candidate-level verifier screen rather than an exhaustive detector-recall benchmark; the
external numbers use the predeclared confidence-band population weighting.

- The confidence-augmented artifact
  `analysis_outputs/public_research/hou_sac_2024_raw_review_v1/q1_300_600/ball_candidate_review_v1/verifier_screen_ebard_v1.json`
  has canonical SHA `ce35a22ed7d05abc355143324cbec02a9b248bed70513800746be38aa520c1e5` and file SHA
  `cee8e6f8329dbcd2ff59e3c7d4724d35c159e11d1c578c3638a6dd9b5656ab5a`. Source OOF AP is `0.940367`;
  source threshold P/R is `0.87357197/0.85089021`; HOU–SAC external lower/upper P/R is
  `0.003423/0.245232` and `0.004259/0.267303`.
- The visual-only artifact
  `analysis_outputs/public_research/hou_sac_2024_raw_review_v1/q1_300_600/ball_candidate_review_v1/verifier_screen_ebard_visual_v1.json`
  has canonical SHA `6188f2e236e57c0aa678dc19c835b2b708c888f2b37cab552c3dce5235fd4f2b` and file SHA
  `c7450a2fba9934217352ddd54d66c2967a5a4d50e0abc8d15c099a893d32aada`. Source OOF AP is `0.889434`;
  source threshold P/R is `0.75016351/0.85089021`; HOU–SAC external lower/upper P/R is
  `0.303030/0.408719` and `0.347475/0.410501`.

Both artifacts are `accepted=false`, `runtime_consumable=false`, and `checkpoint_saved=false`.
The confidence run recorded 37 resource samples (peak system memory/CPU/RSS
`84.0%/71.7%/1,284,063,232` bytes; minimum available memory `2,752,921,600` bytes) and the visual
run recorded 39 samples (`84.0%/68.2%/891,944,960` bytes; minimum available memory
`2,756,411,392` bytes); neither run stopped or retried. These paired screens reject a threshold-only
or E-BARD-only promotion path and leave the rights-cleared, broadcast-diverse continuous 5x5
ball/non-ball plus ball–hand–rim causal data gate as the active blocker.

### MUVY full standardized subset cross-event ball review (2026-08-07)

To check whether MUVY adds useful contexts beyond the earlier bounded `basketball_event_01` sample, AGU
performed a ZIP64-range extraction from the [official Zenodo record](https://zenodo.org/records/13883315)
and compared every extracted MP4 with the existing standardized
`dataset/public_sources/muvy_basketball_v1`. All 13 video SHA-256 values matched. The newly extracted
41-file `muvy_basketball_v2` duplicate was deleted, as were the temporary ZIP central-directory files and
unreferenced review zoom sheets; the canonical 39-file subset remains (13 videos, 13 annotations, 13 metadata
files, five events, 77,193,749 bytes). No duplicate raw media or EBQwen/blind payload was retained.

All 13 videos were decoded and checked against metadata. There were 513 source `sports ball` rows and zero
frame-count mismatches. The release has no hand/rim boxes, release/contact/rim/result timing, shot-outcome
labels, player/team linkage or continuous 5x5 coverage. The five items are short highlight/fan-cam clips and
their titles/frames can reveal game-winner context, so they are not shot-outcome truth.

All 373 hash-bound geometry candidates were manually reviewed: 74 `valid_ball`, 5 `uncertain` and 294
`false_positive`. The false positives are dominated by scoreboards, LED ribbons, jersey numbers, court logos,
phones/hats and spectators. The 74 confirmed physical-ball boxes were materialized as the offline-only
`dataset/public_sources/muvy_ball_yolo_v2` (74 images/boxes; 57 train and 17 validation, with
`curry_game_winner_vs_okc_2016` held out for validation). Its manifest and the review plan are explicitly
`runtime_consumable=false`.

The sealed audit is `analysis_outputs/public_research/muvy_basketball_cross_event_review_v1.json` (internal
artifact SHA-256 `430ea5e5951b1eef6312423b1cbe1fa98dcaadf2e9770424653d2fc2ddf1c49d`); the plan/review SHA-256
values are `b03a73f993ee040f3bd5243cc2150520bfa938bac8c9427ae6db264ac80d1d47` and
`0f2ae67cf7c127af6432dab6fd612605201babe8134e9fc2dbbe80db6c151174`. This increment is useful only for an
offline auxiliary detector/hard-negative screen. It does not change the active gate: AGU still needs a
rights-cleared, broadcast-diverse continuous 5x5 source with sub-second ball–hand–rim outcomes and exhaustive
non-shot hard negatives before any runtime or blind promotion.

### E-BARD+MUVY v2 detector short fine-tune and independent screen (2026-08-07)

The offline-only `dataset/public_sources/e_bard_muvy_ball_v2` combines 1,800 E-BARD frames with 74
hash-bound, manually reviewed MUVY ball boxes. Its manifest artifact SHA-256 is
`cf51edd127d7ed9f8aea22e9bac05af18e8ffa458e715755b79d2a0a0366d0aa`; it remains
`runtime_consumable=false` and is not training truth for AGU runtime.

A guarded five-epoch YOLOv8n MPS run under `.venv` reached internal validation P/R/mAP50
`0.747/0.591/0.649`. The external screen was corrected to distinguish legacy detector-record
`frame_sha256` from explicit decoded-pixel hashes, then evaluated on two sealed disjoint broadcasts:

| Source | TP/FP/FN | Precision | Recall | F1 | Accepted |
| --- | ---: | ---: | ---: | ---: | --- |
| LAL–BOS enrollment | 6/11/19 | 0.352941 | 0.240000 | 0.285714 | false |
| ATL–CHI enrollment | 5/50/24 | 0.090909 | 0.172414 | 0.119048 | false |

The compact audit is `analysis_outputs/public_research/e_bard_muvy_ball_v2_detector_screen_v1.json`
(internal SHA-256 `813b0f5ecbcda30513fa15a9d01137da977f15e9b8f95ae76804e8ae005dd655`). Resource logs,
sealed predictions and external artifacts remain available for audit. After sealing the `best.pt` and
`last.pt` hashes, both unpromoted checkpoints were deleted; no detector/runtime/VLM/EBQwen/blind asset
was changed. The active requirement remains a rights-cleared, broadcast-diverse continuous 5x5 source
with sub-second ball–hand–rim outcomes and exhaustive non-shot hard negatives.

### OKC–CLE cross-broadcast development screen (2026-08-07)

The NBA Games index supplied a new 2025-01-08 OKC–CLE reference (official game ID `0022400509`,
YouTube `RkoCwXJXqnc`). AGU downloaded a 640×360/29.97-fps video stream only for an offline
label-hidden screen. The video SHA-256 was
`e9955b698de9894276de90a0f79ff2dba21322ae04476287103bc55accb5414a` and the size was `417,867,807`
bytes. After 11 decode spot checks and manual review of 32 hash-bound windows, the predeclared
protocol (accept only a visible live release toward the rim; exclude free throws, replays and
stoppages) produced `0/32` qualifying positives. The video was deleted after the screen; only the
2.4 MB manifest/contact-sheet/sealed-label audit package remains.

The compact audit is `analysis_outputs/public_research/okc_cle_2025_cross_broadcast_screen_v1.json`
(file SHA-256 `5de8f68f965a46e8b11f8b2a03360262d60ff694f2081c6e`). Its candidate spec SHA is
`d3818d3db1474a46136692b665c8f8c51da1b0c2ddca5cd35fa41fe4b0c1ca41` and its training-manifest SHA is
`397e2bdd943c02327207bec89ae69445665f4bb6b8efcfd13f60122f4d2cbdb8`. This source is rejected
(`accepted=false`) and is not training truth, runtime media, or a model-promotion input. No runtime,
VLM default, EBQwen weight, detector, training truth, or blind asset changed. The active gate remains
a rights-cleared, broadcast-diverse continuous 5x5 source with sub-second ball–hand–rim outcomes and
exhaustive non-shot hard negatives.

### Current readiness audit (2026-08-08)

The sealed audit `analysis_outputs/public_research/agu_readiness_audit_2026-08-08.json` (file SHA-256
`5c535b1fa1cccc762bea34b5dee91ddeffd31b9769c2ea712dad812c368f90e6`) records the current promotion boundary. AGU is `not_ready` and `accepted=false`:
blind inference remains paused, no training/runtime process is active, and no runtime default or model
weight was promoted.

The corrected causal-support screen is pooled P/R/F1 `0.557562/1.000000/0.715942`; the E-BARD+MUVY
detector reaches only F1 `0.285714` on LAL–BOS and `0.119048` on ATL–CHI; the independent OKC–CLE
screen produced `0/32` qualifying live releases; and the CHI–UTA official-count candidate-geometry
upper bound is F1 `0.632212`, below the `0.85` gate. The active blocker is therefore data and evidence,
not another detector-only epoch: AGU still needs rights-cleared, broadcast-diverse continuous 5x5
ball/non-ball frames with sub-second ball–hand–rim–outcome labels and exhaustive non-shot hard negatives.

The canonical environment remains `.venv` (Python 3.11.15); the legacy `venv` directory is absent. Both
EBQwen native and MLX-4bit assets are retained for offline probes and are not runtime defaults.

### OCR-clock aligned training review (2026-08-08; training-only)

The superseded score-interpolation review directories and dense montages were removed. Using the sealed
OCR game-clock alignments, AGU retained 48 hash-bound windows for each of LAL–BOS and ATL–CHI. A
conservative raw-frame contact-sheet/dense-strip review marked `20/48` and `17/48` positives respectively;
official PBP was used only to select candidate windows and was withheld from the visual labels.

The complete path/hash ledger is
`analysis_outputs/public_research/pbp_clock_training_review_audit_v1.json` (SHA-256
`6ef2a1f6775b70fd4635d3d9d9b15ac49b2be21dd9f3545496ac580484f5ff7c`). The candidate specs, sealed labels,
bundles and training manifests are `runtime_consumable=false` and `independent_evaluation_eligible=false`.
Traditional feature preflight found one identical all-zero vector per source and no continuous
ball–hand–rim supervision, so no model training or promotion was started. This is an offline training
increment only and does not change the readiness gate.

### SVI-Bench online gate and E-BARD team-attribution auxiliary (2026-08-08)

The public Hub API for [SVI-Bench](https://huggingface.co/datasets/MVP-Group/SVI-Bench) was pinned to
revision `aee244344c6ad5bcf0caee298ed53daecc4cff4a`. Its card declares CC BY-NC 4.0 and manual gating:
approved research/education use requires an institutional `.edu` account and accepts no-redistribution,
non-commercial and no-source-video-mirroring terms. Payload probes for basketball T1/T2/T3/T4 files
returned HTTP 401, so no SVI-Bench bytes were downloaded. The release shape is recorded as metadata only
(`analysis_outputs/public_research/svi_bench_online_gate_v1.json`, artifact SHA
`3c1d46d72b29cdd138bfcab85939ab45932d922370d47906ef3ec1de466c722b`); it remains
`runtime_consumable=false` and `training_media_eligible=false` until access and a basketball causal-label
audit are available.

The separately licensed [E-BARD Team Attribution](https://huggingface.co/datasets/GabrieleGiudici/E-BARD-TeamAttribution)
archive was downloaded at revision `6e1436694da204da7afdb26e6b9fa591c8efba65` and verified with SHA-256
`f25c2bb8d8527992e68fc952d840f6c379b4ce24412d0dd764bdcc9c95134be7` (64,715,118 bytes; ZIP integrity
passed). It contains 15,295 crop labels across 1,798 game groups for eight dominant jersey-color classes,
not team/player or event truth. A 69-dimensional RGB/HSV histogram baseline reached group-held accuracy
`0.783279` and macro-F1 `0.732430`; it is retained as an offline auxiliary only, with no runtime or blind
promotion. Manifest: `dataset/public_sources/e_bard_team_attribution_v1/source-manifest.json`. The regenerated
61-source catalog is `analysis_outputs/public_research/source_catalog.json` with SHA-256
`af17c79b83e9af73074d3657e41b0f1b92dcf5098b17d2287dffe53dbf809bd6`.

## 2026-08-08：APIDIS 多视角/事件时轴恢复与 detector 增量（离线拒绝）

经用户授权，按 APIDIS 中央目录已有的 52 条清单从 Kaggle 公共文件端点逐条恢复，未下载
28,559,743,766-byte 原始压缩包。`raw/` 现有 52 个文件、281,635,731 bytes，全部按原始
SHA-256 校验；七个 AVI 均为 800×600、25 FPS、60 秒、1,500 帧。恢复方法与校验信息补充在
`dataset/public_sources/apidis_ball_metadata_v1/manifest.json` 的 `local_restoration` 字段。

新增 `analysis_outputs/public_research/apidis_alignment_v1.json`（文件 SHA-256
`74a1a4a15bab158ff129a3aa5a59faaec63c43c20b7b125b1a699d0606c1d329`，artifact SHA
`00fd98ed18321e1e3e009f7a04d5a8f9ba6467d7b1b9e5b9c53ab48cfe0d5060`）显式对齐本地
`184700+02` 球心时间与 `164700Z` 伪同步视频帧。七机位共得到 4,219 个唯一映射球帧；
16:47–16:48 UTC 片段落在 Q2 事件 XML 内，包含 5 个事件时刻（Throw/Rebound/Violation/
Ball-back-to-court）。这使 APIDIS 可作为离线多视角/事件时轴研究源，但仍不是跨比赛真值。

在 canonical `.venv`（Python 3.11.15）中按机位切分物化 2,100 张训练帧（train 1–5、val 6、
test 7，stride 5，未标注帧保留为空标签 hard negatives），运行 CPU YOLOv8n 五 epoch 资源守护
训练。机位 6 验证达到 P/R/mAP50 `0.66186/0.52618/0.54523`，机位 7 独立视频 screen
仅 F1 `0.358974`（`TP/FP/FN=14/37/13`）；同一 checkpoint 在已封存的 LAL–BOS 与
ATL–CHI 复核集均为 `TP=0`，P/R/F1 均为 `0/0/0`。训练、资源与两场外部 screen 的完整
保留清单在 `analysis_outputs/public_research/apidis_alignment_train_v1/retention.json`。

结论：APIDIS 增量显著减少了同源视频误检，但没有跨转播迁移能力，候选明确
`accepted=false`。生成帧、缓存和未晋级 `best.pt/last.pt` 已在封存 SHA 后删除；源媒体和
对齐/结果证据保持 `runtime_consumable=false`，不修改默认 detector、VLM、EBQwen 或盲推理。

## 2026-08-08：独立 Qwen3-VL negative-first 提示复测（离线拒绝）

为验证独立 VLM 的假阳性抑制能力，AGU 在冻结的四场/32 窗口 raw-frame 计划上加入
`negative_first_v4`：只有同一现场 possession 中“live play → 控球 → 球离手 → 朝篮筐运动”四个
可见环节全部成立才允许 `live_field_goal`，否则否决或 abstain。运行器仍只接收图像和提示，真值
在运行完成后才由独立 evaluator 读取。

本次使用本地 Apache-2.0 `qwen3-vl:2b`，4 帧/256 像素/context 3072；计划 SHA
`ab36c2ec4eb66a0f431a49a6230e28a3f1947fbb2a7a748a6a37306009cf6b7c`，预测内部 artifact SHA
`fd8230aae39bb6b92b3b67eeeb985d08e3a609612cd5271068f02b04b80d7c4a`。在 `.venv` 资源守护下可用内存
下限触发三次安全续跑，最终 32/32 完成；资源日志 SHA 为
`40dbac71baccf43e4cd5d378cfd2663dcd9c01f15427a4c52aa6ab7c511f9343`。

延迟真值为 `TP/FP/FN/TN=16/16/0/0`，P/R/F1=`0.500000/1.000000/0.666667`；四场逐场 P/R
相同，低于 `0.95/0.85` 门禁。评估内部 artifact SHA 为
`9dbd7094fd5743727f2c420fd52805629009f876630de95533a586b6cb648546`，故
`accepted=false`、`runtime_consumable=false`。该提示变体不进入 VLM evidence gate 或 base/VLM
融合，且不改变 detector、EBQwen、runtime、训练真值或盲推理；仅保留预测、评估、计划和资源日志。

### 2026-08-08 在线候选复核（无新 payload）

在用户授权下载的范围内重新核验了四个容易被误判为“完整因果源”的候选：

- [BASKET](https://arxiv.org/abs/2503.20781) 的任务是每名球员 8–10 分钟高光上的 20 类技能等级，
  不是连续比赛或逐帧出手/命中结果标签；不下载其约 TB 级 gated payload。
- [NBA Games](https://choucisan.github.io/collections/nba_games/) 提供 189 场 YouTube 引用、
  box score/PBP 元数据，但明确不再分发视频；PBP 也不是亚秒球–手–篮筐因果标签，继续只保留
  元数据索引。
- [MUVY](https://pmc.ncbi.nlm.nih.gov/articles/PMC13333307/) 的篮球部分是 CC 视频的用户拍摄
  多视角与逐帧空间检测；本地已审计的篮球事件仍无连续 5×5、命中/投丢结果或穷举非投篮负例，
  不重复下载。
- [BASKET-Multiview](https://humansensinglab.github.io/basket-multiview/data.html) 是合成的 7 个
  full plays/约 9K 帧、需要机构邮箱申请的重建基准，不是真实跨转播整场源；不提交申请、不下载。

这轮只更新了门禁记录，没有新媒体落盘；当前仍缺少可核验权利、连续 5×5 亚秒球–手–篮筐–结果
监督和穷举 hard negatives，故不改变 `not_ready`、盲推理暂停或任何 runtime/model 默认。

## 2026-08-08：独立 VLM + 冻结 scene/video OOF 门控（离线拒绝）

为验证冻结的 scene/video shot-validity 概率能否纠正 negative-first Qwen3-VL 的全正例偏差，新增
无标注融合器 `scripts/fuse_independent_shot_vlm_auxiliary.py`。它使用训练屏幕已封存的逐比赛
`best_precision_at_recall_0_85` 阈值；不重新拟合、不读取目标标注，且辅助 OOF 缺失时输出
`unknown`。融合输出和代码均标记 `runtime_consumable=false`、`target_labels_used=false`。

输入 scene/video 工件的 artifact SHA 是
`4022199d7b6d884fd92c5b622f88f07651fa40a84632b36fb98a5614cdd0b7b5`；输出预测工件
`analysis_outputs/public_research/independent_shot_vlm_qwen3vl2b_compact256_negative_first_scene_oof_gate_predictions_v1.json`
的 artifact SHA 是 `1d2b9f6aa89418f149f548e988f18891dd4542e0a82252a92ec5c7310aa9b371`，32 个事件覆盖
31 个，缺失 1 个。延迟评估
`analysis_outputs/public_research/independent_shot_vlm_qwen3vl2b_compact256_negative_first_scene_oof_gate_evaluation_v1.json`
为 `TP/FP/FN/TN=10/6/6/10`、unknown `16`，P/R=`0.625000/0.625000`；逐比赛最差
P/R=`0.400000/0.500000`，没有达到 `0.95/0.85` 门禁。该结果只证明当前辅助概率不足以修复
跨比赛独立 VLM，未改变训练、runtime、EBQwen 或盲推理资产。

## 2026-08-08 continuous causal source gate (metadata-only)

AGU now applies a fail-closed five-field gate before any new basketball payload is downloaded:
`rights_cleared`, `broadcast_diverse`, `continuous_five_by_five_video`,
`subsecond_ball_hand_rim_outcome_labels`, and `exhaustive_non_shot_hard_negatives`. The source must
also have an immutable URL, revision and declared size. Missing fields are false; a truthy string is not
accepted. The gate is implemented in `app/analysis/public_research_datasets.py` and exercised by
`tests/test_public_research_datasets.py`.

The initial metadata-only audit recorded seven candidates (BasketEvent, NBA Rebounds Anticipation, NBA
Games, Basketball Events, MUVY, VRU Basketball and BASKET). NSVA metadata became the eighth candidate,
and the metadata-only GCB/GameCommBench basketball increment is now the ninth. The current artifact still
has `eligible_source_count=0` and no payload download. The initial internal audit SHA was
`0fa1b64c8ee64922b5b1ab07097d29e7604b7ee370737191fb78cc64ab14e59e`; the current SHA is recorded in the
GCB section below. All records remain
`runtime_consumable=false` and `training_media_eligible=false`; this does not alter the readiness or
blind-inference boundary.

## 2026-08-08 NSVA event-text metadata increment (offline only)

The [NSVA subset](https://huggingface.co/datasets/sportsvision/nsva_subset) was fetched at immutable
revision `97c211f85beec54209b44faea2bff73544a16125` through the two known Parquet paths. The bounded
download is 23,167 bytes and 1,316 rows (1,051 train / 265 validation) under the
`fair-noncommercial-research-license`; it contains no raw video. The upstream
[NSVA repository](https://github.com/jackwu502/NSVA) was retained only as a bounded source-code archive
for license/downloader inspection. Its unbounded clip collector was not run.

`app/analysis/nsva_event_text.py` and `scripts/download_nsva_subset.py` normalize compact intents into an
offline AGU ontology. The sealed audit counts 953 field-goal attempts, 7 free throws, 495 rebounds,
220 fouls, 135 turnovers, 8 jump balls, 12 period boundaries, 1 violation and 1 ejection; 293 rows carry
an assist marker and no row has an unknown segment. The source manifest is
`dataset/public_sources/nsva_subset_v1/source-manifest.json` (internal SHA
`759991c4b82ff1a58f505dda53712af4ee47da7670861b0ff5110af1a505e0c4`), and the event audit is
`analysis_outputs/public_research/nsva_event_text_audit_v1.json` (internal SHA
`c66bb53c2487b4743cb7a0f4f6b9b262514a6adee81eec8d035689db7272b96e`). Both explicitly remain
`runtime_consumable=false`, `training_media_eligible=false`, and `media_downloaded=false`.

After adding NSVA metadata, the continuous causal source gate had eight candidates, zero eligible sources,
and zero payload downloads. The latest GCB increment makes nine candidates while the gate remains at zero
eligible sources and zero payload downloads. The latest gate internal SHA is recorded below; these increments
change ontology coverage only and do not promote any detector, VLM, EBQwen weight, runtime or blind-inference
asset.

## 2026-08-08：GCB/GameCommBench 篮球事件元数据增量（仅离线）

联网核验了 [GameCommBench/GCB](https://huggingface.co/datasets/A4Blind/GCB) 的固定 revision
`728f3a67839c10c54a2aba792d8659f94b8fa6ad`。上游数据卡将许可证标为 `other`，篮球子集是 2,981 个
短视频事件记录、约 2,981 个 MP4/21.761 GB 的媒体索引，并非连续整场转播；因此按用户授权的
“先元数据、过门禁再媒体”策略，仅下载 `human_commentary/basketball/metadata.jsonl`（6,519,620 bytes）
和 `dataset_summary.json`（2,045 bytes），没有下载视频本体。

`app/analysis/gcb_event_audit.py` 与 `scripts/download_gcb_metadata.py` 通过 TDD RED/GREEN 校验了
复合标签、period/timeout/technical 文本归一化、来源文本结果与比赛范围封存，以及路径穿越防护和
自校验 manifest。审计工件
`analysis_outputs/public_research/gcb_basketball_event_audit_v1.json`（内部 SHA
`dae4b926ae034c885cf497569a16066512c0dbeb82d1deb56565701a1a11d0c4`）包含 2,981 行、132 场比赛；
显式标签统计为 field-goal attempt 2,495、rebound 855、turnover 219、foul 253、free-throw 22，
来源文本统计为 field-goal attempt 2,602、rebound 898、turnover 220、foul 254，来源结果为 make
1,609/miss 1,015，且 unknown segment 为 0。

来源清单 `dataset/public_sources/gcb_basketball_v1/source-manifest.json`（内部 SHA
`8cc03775d9396cdab7f9c1252f7fd506deff62d4db380d1b2df814f3bda975e1`）明确记录
`media_downloaded=false`、`runtime_consumable=false`、`training_media_eligible=false`，并将声明的
21,761,000,000-byte 视频负载纳入五项门禁但拒绝下载。门禁更新为 9 个候选、0 个 eligible、0 个
payload download，最新内部 SHA 为
`29b3b9190f4ad9bdd56a73dde926190d9cfdc4f0c40b01e2e50d8543b81af607`；没有改变 runtime、默认 VLM、
EBQwen、训练真值或盲推理边界。source catalog 已重建为 62 个来源，内部 SHA
`4921a2cdc65d4b30b2c3471c1dd6064f30fb8cde83d5435c8ac17b3122ec270f`。

## 2026-08-08：既有 PBP 视觉状态人工复核再检查（仅标注质量审计）

对既有 label-hidden contact sheet 中的 `visual-state-0029` 与 `visual-state-0094` 重新逐帧检查，
二者均保留 `stoppage_other` / medium confidence / unresolved：前者是现场宽景与教练/回放切换，
后者是活动画面后接篮筐、回放和 cutaway，均不足以证明罚球准备或现场出手。复核结果封存在
`analysis_outputs/public_research/pbp_visual_state_codex_followup_v2/manual_recheck_v1.json`（内部
SHA `b4fe543ea6e1849b17d72a483df91c4ebbf09e9835e0adb12f4df4d66a2d80af`），明确
`codex_runtime_answer_used=false`、`runtime_consumable=false`、`promoted_to_training_truth=false`；
没有把这两条不确定样本写入 runtime 或训练真值。

## 2026-08-08：AGU 基座 + 独立 VLM 融合确定性复核（离线拒绝）

在 canonical `.venv` 下对既有 32 个事件的 AGU v3 基座与独立 VLM 预测做固定 `both_confirm` 重放，
不重新训练、不读取目标标签来调参。重放与封存预测 artifact SHA 一致
`8ee0dfc577641e30ef67442ed1cc820eeb1fde025b6365aaf0138bb4772d2491`，最大 RSS
`339,755,008` bytes；跨四场比赛 pooled P/R 仍为 `0.000000/0.000000`，没有满足晋级门禁。
紧凑复核 `analysis_outputs/public_research/independent_base_vlm_v3_cross_game_recheck_v1.json` 的
内部 SHA 为 `13a09c158b65efe22df2eaf6bb66d9002b5f9181efbe7b780925529bcbdc6af2`，仅保留离线证据，
不进入 runtime、训练真值或盲推理。

## 2026-08-08：拒绝 RF-DETR 特征缓存清理

RF-DETR query verifier 的 `cache_v1`/`cache_v2` 只包含已拒绝屏幕的中间 `.npz` 特征，结果与资源
日志已经封存且没有当前代码引用它们，故删除约 18 MB；保留所有 screen/resource JSON、源视频和模型
权重，未触碰 EBQwen native/MLX-4bit 或 AGU v3 基座检查点。

## 2026-08-08：UVY 篮球检测辅助子集（CC-BY，非因果）

联网核验并固定了 [UVY: Sport User-Generated Videos for Multi-Object Tracking](https://zenodo.org/records/21303900)
记录 `10.5281/zenodo.21303900`，许可证为 CC-BY-4.0。源 `UVY.zip` 声明大小为
3,274,165,269 bytes、MD5 `f99594a1bd9f627ebe21219db86317a6`；没有下载整包，而是读取中央目录后用
39 个 HTTP Range 请求提取四个篮球序列的图像和 `video_info.txt`/`gt/labels.txt`/`gt/gt.txt`。
最终保留 6,183 张 JPG 和 12 个标注/元数据文件（约 438 MiB），所有提取条目均通过源 ZIP CRC；所有
MP4 与非篮球序列均未写入本地。下载清单为
`dataset/public_sources/uvy_v1/manifest.json`（manifest SHA `a55059d165c2010e05600544732fadd55d9fba44d7a19e7049bf637898ca1f0c`）。

原始 MOT 审计工件 `analysis_outputs/public_research/uvy_basketball_audit_2026-08-08.json`（内部 SHA
`e4dce068d14dd87c0845363e162057370e14ba7fa143acc20a97cad9f753641c`）包含 4 个序列、6,183 帧和
33,639 个框：player 24,167、referee 5,550、goal 2,235、sports ball 1,687。V02/V03 的 `gt.txt`
哈希完全相同，V03 的 `video_info` 自报 2,818 帧但实际只有 2,320 张图；V04 声明 `img` 但实际
目录是 `img1`。这些差异已封存，防止重复标注进入验证集。

`scripts/materialize_uvy_yolo_auxiliary.py` 生成了不复制原图的硬链接 YOLO 辅助集
`dataset/public_sources/uvy_v1/yolo_aux_v1/`（manifest SHA
`69a7d10fcc2baa1e4ed0e860ec2a190e0afacecfd02d5d86656ce249d7b9871c`）：V01 为 train 193 帧，V02
为 val 2,320 帧，V04 为 test 1,350 帧；重复的 V03 被排除。四类框仅用于 sports ball/goal/player/referee
检测和 hard-negative，不含命中/投丢、出手时刻、球–手–篮筐因果关系，故
`training_media_eligible=true` 仅限 `auxiliary_detector_and_hard_negative_only`，而
`runtime_consumable=false`、`causal_truth_eligible=false`。

为确认辅助源是否能直接改善小球检测，固定 `yolov8n.pt` 做了 120 帧 V04 held-sequence transfer screen。
屏幕工件 `analysis_outputs/public_research/uvy_yolo_transfer_screen_2026-08-08.json` 的内部 SHA 为
`85393624ef72c7d9d84dff689e0444a34ffed389975234fcfe92735001f114d1`，73 个带球目标全部漏检，
`TP/FP/FN=0/0/73`、P/R/F1=`0/0/0`，最大 RSS `410,615,808` bytes；通用权重在哈希封存后删除，
没有写入 runtime 或默认 detector。UVY 本身也进入连续因果 gate，但五项中仅权利通过，当前 10 个候选
仍为 `eligible_source_count=0`，gate SHA `89f59c4412f6e200ea8fc520cfd75f2ce92523eeab21b39dc93d751cd7eda4f0`。
source catalog 现为 63 个来源，SHA `2ac0fe84af076cb6d8959ca57c821a94ddb867191c4c5023b25fe06a4c01b5ff`。

### UVY 辅助检测训练屏幕（2026-08-09；拒绝晋级）

新增 `app/analysis/uvy_detector_training.py`、`scripts/train_uvy_yolo_auxiliary.py` 和
`tests/test_uvy_detector_training.py`，把 UVY 训练约束固定为离线、不可晋级的
`auxiliary_detector_and_hard_negative_only`。在 canonical `.venv`（Python 3.11.15）下以
`model_checkpoints/yolov8n.pt` 做 1 epoch CPU smoke/transfer screen：V01/V02/V04 为
train/val/test，`imgsz=256`、`batch=2`、`workers=0`，并用绝对路径数据清单避免工作目录漂移。
训练计划 `analysis_outputs/public_research/uvy_yolo_training_screen_2026-08-09_compact_v4/run/plan.json`
的 SHA 为 `8f5d307eddc5affa3576067e3813a770b0aec57c0e1b8d8105cc16644ce2987d`；结果工件的内部
SHA 为 `2dfe3f1ee7ff089f0021be338484131726f1f2d4476c8da47d2c823962203e36`。

在独立 V04 test 的 120 帧中，72 帧含 sports-ball 目标；best checkpoint 的球检测为
`TP/FP/FN=0/0/72`、P/R/F1=`0/0/0`，没有生成球预测。资源守护 exit 0，峰值系统内存
`84.2%`、进程树 RSS `546,029,568` bytes、CPU `50.0%`，最低可用内存 `2,711,584,768` bytes，
无停止/重试。best/last（各 6,194,986 bytes；SHA 分别为
`11c385be59af6feb6f7bc478978b6aef9a338ea0ea2d3f4fc7cf80c7fc56c2c7` 与
`80b5e63438cd298dd09356c83c238d223c5e42c6cd427e573a043eadeef48590`）在封存后删除，未修改
runtime、默认 detector、VLM、EBQwen 或盲推理。该屏幕只说明 UVY 不能直接改善跨转播小球检测，
也不能解除当前连续 5×5 因果数据门禁。

### 第六场外层留出场景/视频融合屏幕（2026-08-09；离线拒绝）

在不重新下载媒体的情况下，使用已封存的 `dataset/public_sources/nba_games/media_new_dev_v1/uIxqFnBCOyk.mp4`
对应 32 个哈希绑定窗口，把四场 scene/video 基线扩展到六个 source-video 分组。工件
`analysis_outputs/public_research/shot_validity_scene_fusion_game3_all_v1.json` 的内部 SHA-256 为
`1c63957e77d67142255673291c1163ac42410d885ef1a2c5980cb81720009cac`，覆盖 575 个窗口；其
`runtime_consumable=false`，仅用于离线外层留出审计。

最佳 scene+MViT 屏幕 pooled P/R/F1=`0.586854/0.980392/0.734214`，新增 Lakers–Magic held game
P/R=`0.400000/0.857143`，2008 Lakers–Boston held precision=`0.503759`。该新增片不是权利清晰的
训练真值，且未改善每场 `0.85/0.85` 门禁；不晋级任何 checkpoint 或运行时输入，保留原片与紧凑工件仅供
开发复盘。

### MUVY 重复渲染表清理（2026-08-09）

历史 `muvy_ball_codex_review_v1/sheets/` 与 canonical
`muvy_basketball_v1_review_v2/sheets/` 的 11 张 JPG 逐文件 SHA-256/大小完全相同，且历史图目录
没有当前文档或代码引用。清理证据 `analysis_outputs/public_research/muvy_duplicate_sheet_cleanup_2026-08-09.json`
（内部 SHA `aabfe7d23f89d00647de37700346d519c29d5ad9e8a1d161b8d4b945ecd70fe8`）绑定了所有文件；
已删除精确的 `14,931,003` bytes，保留 canonical 图、MUVY 原片与紧凑 review/manifest。

### 在线候选补充核验（2026-08-09；无新 payload）

本轮对三个可能补足数据缺口的公开入口做了官方页面/卡片核验：

- [NBA Games](https://choucisan.github.io/collections/nba_games/) 仅提供整场 YouTube 引用和
  PBP/box-score 元数据，视频不随 release 重新分发，不能作为权利已核验的像素级亚秒因果训练源；
- [BasketHAR](https://huggingface.co/datasets/Xian-Gao/BasketHAR) 的 Apache-2.0 内容是惯性信号、标签和
  notebook，不是连续广播视频；
- [BasketLiDAR](https://sites.google.com/keio.jp/keio-csg/projects/basket-lidar) 按项目说明需合理学术
  申请，当前没有公开可下载的跨转播 5×5 因果 payload。

三者都未通过 `rights_cleared`、连续 5×5、亚秒球–手–篮筐–结果和穷举 non-shot hard negatives
门禁组合；本轮 `payload_downloads_performed=0`，不改变任何 runtime、默认模型、训练真值或盲推理资产。

### NBA Games PBP 重复树清理（2026-08-09）

磁盘审计确认 `dataset/public_sources/nba_games/games/` 与 canonical
`dataset/public_sources/nba_games_v1/games/` 的 567 个 PBP 文件逐文件同名、同大小、同 SHA-256；代码和正式
审计只使用 v1 树。按用户授权精确删除旧树 `63,474,295` bytes，保留 v1、媒体、source manifest、审计与
盲测资产。删除前 inventory/tree SHA 为 `f9db849d0d63883d3768be249bcfc1389e096dde39f9b1a01c69914b8480ce2d`
和 `15050a3918448bca3f3f7859b06ba460f69bdfe4f426b8c99b4d95c7c6395ef8`；正式记录为
`analysis_outputs/public_research/agu_nba_games_duplicate_cleanup_2026-08-09.json`（SHA
`382221ea8b557930efc66915e7002df61a65eba8c33266430b15a1409e7433f2`）。

### 广播状态融合高维筛选（2026-08-09；离线未晋级）

在已有 511 个 hash-bound 窗口上按四比赛外层留出重新筛选 `base+broadcast_raw`，仅改变
scene/video PCA 维度，未读取新的媒体或标签。PCA36 工件
`analysis_outputs/public_research/shot_broadcast_fusion_full511_batch4_pca36_c0p01_screen_2026-08-09.json`
的内部 SHA-256 为 `824d518806344a0ea47e21564f13ec9882ddb4b0fb15e46ca4adae4d16194ca1`；
Pooled TP/FP/FN/TN=`233/63/14/201`，P/R/F1=`0.787162/0.943320/0.858195`，优于 PCA16
的 F1=`0.852886`，但最差比赛 precision 只有 `0.730337`，仍未通过每场 `P/R≥0.85`。
峰值 RSS=`729,186,304` bytes。结果保持 `runtime_consumable=false`，不改变任何默认模型、
detector、VLM、EBQwen 或盲推理资产；它只排除了“调 PCA/正则化即可解除阻塞”的假设。

### 在线候选补充：MEV 与 VSTAT（2026-08-09；元数据拒绝）

本轮在不拉取视频 shard 或 YouTube payload 的前提下，按固定 revision 审计了
[MultiEventVideo/MEV](https://huggingface.co/datasets/MultiEventVideo/MEV) 与
[VSTAT](https://huggingface.co/datasets/VSTAT-NeurIPS2026/VSTAT)。审计证据为
`analysis_outputs/public_research/online_candidate_supplement_2026-08-09.json`，内部 SHA-256 为
`820bd8d0f15502965bc0dd7c02ecb01bdd178095eb6dd208f1a144f996f9cff5`。

- MEV 的公开元数据中，保守的 `basketball` 精确关键词只命中 39 个 UUID、128 行事件；这些 UUID
  的事件总时长约 `504.725` 秒，最长连续 span 仅 `27.694` 秒。数据卡声明标注/元数据为 CC BY 4.0，
  但视频为混合第三方许可，公开树没有逐视频权利 manifest，也没有连续全场和亚秒球–手–篮筐–结果
  标签；约 `76,009,107,602` bytes 的视频 shard 未下载。
- VSTAT 的 `source_task=basketball` 仅有 192 个问题、30 个片段文件、3 个 YouTube 来源，视频不随
  release 再分发，只有 URL/时间戳；它不是连续全场因果真值，未下载 YouTube 内容。

两者均 `runtime_consumable=false`、`training_media_eligible=false`，五项连续因果来源门禁工件
`analysis_outputs/public_research/continuous_causal_source_gate_2026-08-08.json` 内部 SHA-256 为
`ae21c18ee551b5a5e4abbfcf4e132a8b955e73a8e30642068cf69417de643a07`，当前为 `11` 个候选、`0` 个
eligible、`payload_downloads_performed=0`；来源目录 64 个来源，内部 SHA-256 为
`222f7c40c64bb9080e3a4ab045627a8227b3b00b575bc636c2a826e70d7da6b4`。保留紧凑元数据证据，不把短片或
第三方视频写入 AGU 训练/运行时。

### ExAct 篮球技能反馈补充（2026-08-09；元数据辅助参考）

本轮固定 [ExAct](https://huggingface.co/datasets/Alexhimself/ExAct) revision
`1bd51bfdbd228f850f69cf81d3b4919c71608c04`，只下载 README、`data/metadata.jsonl` 与分页文件树，未下载
任何 MP4。审计证据为
`analysis_outputs/public_research/exact_basketball_skill_audit_2026-08-09.json`，内部 SHA-256 为
`a8b2ba782a2f9bc697b6b1e449d56a7474dd1f3d7d769993ea55dfdcaad2fb7f`；README、metadata 和文件树 SHA-256
分别为 `d23586bac7a290a3917f18344b31ed8163669dbe936e4a763e99a179044cfcb2`、
`0115845727d2bfb178fcca1a583700171f6336b701125806c759365b0ed719c5` 和
`91a4653cb47fb0401b3b8cf02f229bb9201e9dcb1e0bc2361683c9c55474db4f`。

- 公开树含 3,521 个短视频条目；精确 `domain=basketball` 子集为 1,047 条、183,263,004 bytes，覆盖
  Mikan Layup 410、Reverse Layup 388、Mid-Range Jump Shooting 249；GE/TIPS 为 165/882。
- 标签是专家动作姿态反馈及四个文本干扰项，不是投篮命中/未中、球–手–篮筐亚秒时序、球员身份或全场穷举
  hard negatives；底层逐片媒体来源也未另行核验。
- 因此只登记为 `metadata_only_auxiliary_research_reference`，`runtime_consumable=false`、
  `training_media_eligible=false`，媒体下载量为 0，不加入连续因果来源 gate，也不把其标签导入 AGU
  投篮结果或球员统计。

重建后的来源目录为 65 个来源，catalog 内部 SHA-256 为
`6223b0ad4995d757e774044aed18c1e8d8c46a48307e1d55cbecf23ec5a3074e`；连续因果 gate 仍为 11 个候选、0 个
eligible、0 次 payload 下载。

### SportVU 2015–16 轨迹 tiny 分片（2026-08-09；离线辅助）

按固定 [Hugging Face 数据卡](https://huggingface.co/datasets/dcayton/nba_tracking_data_15_16)
revision `50ec5611a9128c996ac19d094145bcc4ffa57f22`，从其公开的
[NBA-Player-Movements](https://github.com/linouk23/NBA-Player-Movements) 与
[nba-alt-awards](https://github.com/sumitrodatta/nba-alt-awards) 上游下载了五个按目录排序的
`.7z` 比赛档案（压缩后 `29,017,344` bytes），以及用于过滤的完整 PBP CSV；最终只保留五场压缩档案和
`2,208` 行 `pbp_selected_5_games.csv`（`367,366` bytes）。完整 PBP 和展开后的约 `500,679,703`
bytes JSON 已在审计后删除。每个档案均通过 `bsdtar -tf`，逐档 SHA、内层 JSON 大小和指标见
`dataset/public_sources/nba_tracking_15_16_tiny_v1/manifest.json`；紧凑屏幕工件为
`analysis_outputs/public_research/sportvu_tracking_tiny_audit_2026-08-09.json`（SHA
`82614f153373e8db7d8e0e67259d63b40dc28a334b87673ed0c865f8254b60ea`）。

五场合计 `2,230` 个 SportVU 事件、`1,020,644` 个 25 Hz moment，球坐标行覆盖率
`0.997771`；PBP 事件编号交集为 `1,904/2,230`，另有 `408` 个 shot-like、`216` 个 made-like、
`192` 个 missed-like、`217` 个助攻和 `468` 个篮板文本行。这个交集只是事件索引/空间先验审计，不是
广播帧对齐分数，也不产生投篮结果、球员统计或球–手–篮筐亚秒真值。

两个上游 GitHub API 均未声明仓库 license，HF card 也没有 data license 字段，因此该分片标记为
`unverified-no-license-declared`、`runtime_consumable=false`、`training_media_eligible=false`、
`causal_truth_eligible=false`。它只允许离线几何、时钟/事件索引和轨迹先验实验；不进入连续因果来源
gate、不接入默认 runtime/VLM/EBQwen，也不解除盲推理暂停。

### SportVISTA 多运动视听源（2026-08-09；许可拒绝）

对新发现的 [SportVISTA](https://huggingface.co/datasets/bouachalazhar/sportvista) 固定 revision
`ffb720af2a1e23ff7a8f39379a0f9606cc110e3f`，只下载 README 和 LICENSE 做元数据/许可审计。v0.1.0
当前只发布文档（声明快照约 8,144 个录制、4,285 小时、3.04 TB），没有行级 manifest、标注或媒体
payload；HF 仓库是人工 gated。README/LICENSE/API 的 SHA 与完整判定见
`analysis_outputs/public_research/sportvista_online_audit_2026-08-09.json`（SHA
`63f0970e111210ff82bf50343aab99319bd27bdd085fb24de6a7e7b13cfbe50a`）。

其 `sportvista-research-only-v1.0` 许可要求个人逐项审批、禁止再分发，并明确禁止用 Dataset Materials
训练、微调、评估、benchmark、对齐、蒸馏或改进通用/基础/生成模型；这与 AGU 自有基座+VLM 的目标路线
直接冲突。因此本轮访问请求和媒体下载均为 `0`，登记为
`metadata_only_license_audit_no_agu_base_vlm_training_or_evaluation`，保持
`runtime_consumable=false`、`training_media_eligible=false`，不进入连续因果 gate。

### Wikimedia Commons / HCTV 连续全场种子（2026-08-09；人工标注候选）

目录：[Videos of basketball in the United States](https://commons.wikimedia.org/wiki/Category:Videos_of_basketball_in_the_United_States)。目录审计到 23 个由 Hardwick Community Television (HCTV) 上传的完整高中 5×5 比赛，合计约 35.44 小时、52.34 GB；逐文件 Commons 元数据为 CC BY 4.0，两个抽查的原 YouTube 元数据也明确为 Creative Commons Attribution。Commons 文件页的 `License review needed (video)` 维护标签已写入审计，不把它忽略。

AGU 只保留一场有界原片：`dataset/public_sources/wikimedia_hctv_fullgame_v1/raw/hctv_hazen_lyndon_2023.webm`，`1,416,244,469` bytes，SHA-256 `54dedaee0b04a0bab210b6fe730f857edaf193ec5c28a03a84bf005aa64f5bf9`，AV1/Opus、1280×720、29.97 FPS、约 59.8 分钟。四个时间点帧样本能解码并显示全场/比分牌，但原片不带事件金标准、球/手/篮筐亚秒标签或穷举 hard negatives。

因此该源的边界是 `manual_annotation_seed_only`：`runtime_consumable=false`、`training_media_eligible=false`、`causal_truth_eligible=false`，不用于盲推理或默认模型。完整来源/许可/下载/删除记录见 `dataset/public_sources/wikimedia_hctv_fullgame_v1/manifest.json` 与 `analysis_outputs/public_research/wikimedia_hctv_fullgame_audit_2026-08-09.json`；当前连续因果门禁为 12 个候选、0 个 eligible。

### Wikimedia Commons / VTV 独立制作全场种子（2026-08-09；人工标注候选）

新增 [VTV Commons 全场文件](https://commons.wikimedia.org/wiki/File:Superliga_Profesional_de_Baloncesto_-_Spartans_de_Distrito_Capital_Vs._Trotamundos_de_Carabobo.webm)，制作方为 Venezolana de Televisión。Commons 元数据声明 Public domain，并有 `PD Venezuela official`/VTV 来源类别；原 YouTube 元数据没有 license 字段，因此只把 Commons 声明作为权利依据，不从 YouTube 页面推导许可。

按用户授权只保留一场 `Spartans de Distrito Capital vs Trotamundos de Carabobo`：`dataset/public_sources/wikimedia_vtv_fullgame_v1/raw/vtv_spartans_trotamundos_2026.webm`，`2,228,583,591` bytes，SHA-256 `e5cdcdfd1e49f5a71765d4cece56a4810a563938fca4d407836ea6c12409fbec`，VP9/Opus、1920×1080、30 FPS、约 2 小时 42 分。四个时间点均成功解码，多个样本显示完整球场和 VTV 比分牌。

VTV 与 HCTV 已组成两个独立制作源的人工标注配对，配对审计为 `analysis_outputs/public_research/wikimedia_hctv_vtv_cross_source_audit_2026-08-09.json`。两场均无亚秒球—手—篮筐—结果标签和穷举 non-shot hard negatives，因此 VTV 同样保持 `runtime_consumable=false`、`training_media_eligible=false`、`causal_truth_eligible=false`；只可用于离线人工标注和跨制作留出验证。VTV manifest/audit 为 `dataset/public_sources/wikimedia_vtv_fullgame_v1/manifest.json` 与 `analysis_outputs/public_research/wikimedia_vtv_fullgame_audit_2026-08-09.json`。

已先建立低资源的人工复核入口 `analysis_outputs/public_research/wikimedia_hctv_vtv_pilot_annotation_v1/`：20 个保留 JPEG 帧、4 个连续五帧窗口（HCTV 1 个 `uncertain`，VTV 2 个 `not_a_shot` 候选和 1 个近篮筐 `uncertain`）。`pilot_review_plan.json`、`pilot_review.json` 与 `pilot_manifest.json` 均绑定原片 SHA、来源 manifest/配对审计 SHA 和帧 SHA；这是待人工确认的 pilot，不是穷举 hard-negative 集，也不进入 runtime、训练真值或盲推理。

更新后 source catalog 为 70 条，连续因果 gate 为 14 个候选、0 个 eligible（`analysis_outputs/public_research/continuous_causal_source_gate_2026-08-09.json`）。

### 临时缓存清理（2026-08-09）

按用户授权清除了只存在于 `/tmp`、且已被正式审计结果替代的 APIDIS/MUVY 临时缓存与元数据探针，
共 `425,154,113` bytes；逐目标路径/大小、删除前 SHA/树 SHA 和保留项见
`analysis_outputs/public_research/agu_tmp_cleanup_2026-08-09.json`。正式全场原片、开源权重、运行时检查点、
标注/审计工件和 blind 资产均未删除。

### HCTV Randolph 连续全场原片（2026-08-09）

新增 [Wikimedia Commons HCTV Randolph 文件](https://commons.wikimedia.org/wiki/File:Boys_Varsity_Basketball_v._Randolph_-_February_17,_2026.webm)，来源目录为 [Videos of basketball in the United States](https://commons.wikimedia.org/wiki/Category:Videos_of_basketball_in_the_United_States)，原 YouTube provenance 为 `-VeEs0aVr5M`。Commons API 声明 `2,152,005,777` bytes、1920×1080、`5,899.308s`，CC BY 4.0；文件仍带 `License review needed (video)`。本地 payload SHA-256 为 `f234d5bfadd0b182c9a15a6d7b206335fcb371e9971cd2bf67313f544dc6a102`，`ffprobe` 为 AV1/Opus、60 FPS。

正式资产：

- `dataset/public_sources/wikimedia_hctv_randolph_v1/manifest.json`，内部 SHA `4c9d5f1e8a6040faea2e6f686f412cc709a63678ce6eca47a5bb36d10e304c6e`；
- `analysis_outputs/public_research/wikimedia_hctv_randolph_audit_2026-08-09.json`，内部 SHA `2fb3d82f25be4061b0ea392bc1e13df831a4c35a4129e1d4eae78e8667de3018`；
- `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_cross_source_audit_2026-08-09.json`，内部 SHA `8cb43f7ac266494ae0e09014d5151bfdca69bd83405e7376bbc2df4d4b490aac`。

五个审计抽帧中有赛前画面、HCTV 捐赠插播、暂停/团队聚集和现场全场画面；因此连续长录制须先标记/排除插播区段。Randolph pilot v1 保留一个中场运球 `not_a_shot` 候选五帧窗口，计划/复核/manifest SHA 为 `1ea8f7897815c938403fce6e4018d163ae887bfef02492416d3976f046eba5d3`、`742d118de6789ddeaeeda486af1da7c9041aac4a1bbc46388de524c54788a132`、`5350395cb0bd89b0e0a95b842ee42bde44475f6000e63f580168cc53f4316fce`。随后新增 `wikimedia_hctv_randolph_pilot_annotation_v2/` 的 21 帧篮下攻防窗口；逐帧无法闭合出手—篮圈—结果链，因此只封存 `uncertain`，plan/review/manifest SHA 为 `25abad0c2c6268161cb916b7af703cd7d160eff7bad8c643e5396397b597639e`、`a3eff9b24ed9abdb1f8f36ba4eaaeea5bfcc28c6dc5ce64c473dcfa4de9f158d`、`4da2b9d59259a66a4296eb6d0a167aaf8124a11bb85f4af2951524ad6e231959`。两个 pilot 均保持 `runtime_consumable=false`、`training_media_eligible=false`、`causal_truth_eligible=false`，不进入 runtime、训练真值或盲推理。
v2 封存后，精确删除了不再引用的 36 张粗抽帧/联系表/候选中间 JPEG（`6,443,465` bytes）；删除审计为 `analysis_outputs/public_research/randolph_pilot_intermediate_cleanup_2026-08-09.json`，内部 SHA `5ef8e77482bd66bd8806d20537a633dafc7d53d06ae13fca48efcf4641007401`。正式 v1/v2 pilot 帧、原片、权重与盲资产均保留。

随后在三场原片上完成一轮跨制作源的低资源扩展试标，工件位于
`analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v2/`：129 张保留 JPEG、6 个
连续窗口，其中 HCTV Hazen 约 600 秒窗口可见控球—脱手—向篮圈飞行—篮圈接触/未命中链，封存为一个保守
`shot/missed`；另外五个 HCTV 运球、VTV 现场转场/插播、Randolph 运球/罚球前边界窗口均封存为
`not_a_shot`。plan/review/manifest SHA 分别为
`964a4a5170c2cd3d5ab903739483033b3fc58b1f1b60db373af761119cdd1b36`、
`9c76caf5fd1dee0d9c65b409c75399bf4f3a4b5f9549cd405c41d670c8a6819a`、
`95798600013a59817f908d3a1b5cfda64dd5cc27d73c5306c691bbd0e258a164`，源绑定三源 cross-source audit SHA 为
`8cb43f7ac266494ae0e09014d5151bfdca69bd83405e7376bbc2df4d4b490aac`。该扩展仍是
`pilot_only`、待人工确认、非穷举、`runtime_consumable=false`、`training_media_eligible=false`，不进入训练真值、
VLM 答案或盲推理；就绪审计仅更新为 175 帧/12 窗口，因果 gate 仍为 16 候选/0 eligible。

来源目录现在为 72 条，连续因果 gate 为 16 个候选/0 个 eligible；详情见 `analysis_outputs/public_research/source_catalog.json`（SHA `8e673f9105b6ca77e77f01a1652b8f112cef78c9e2b3bd888e01f162dda1c66c`）和 `analysis_outputs/public_research/continuous_causal_source_gate_2026-08-09.json`（SHA `f820833f81caa0a12a4a6efb3ef6e5e3202c85964dae11cada7ae3617933fb4f`）。

### HCTV Hazen + VTV v3 因果窗口与 MLX-4bit 屏幕（2026-08-09）

- [x] 在既有连续原片上扫描并保留两个新窗口（Hazen `2101.6–2103.8s` 命中链、VTV
      `6598.5–6600.6s` 中途入镜失败案例），共 45 张哈希绑定复核帧。
- [x] 封存 v3 plan/review/manifest，SHA 分别为
      `1dfb5e045c4788c42f7ae0ccca3a42be13edf5c6a483ed0bbc0534849f150d69`、
      `31e20a75347c6a3459330eb923b22c687f4df8de95462c3c7b4de6e5829171a1`、
      `5e013f2356dc52fdd8b250d125f03f875132beaa90c47fa9a3001a95a4e2f6e4`；均保持
      `pilot_only=true`、`runtime_consumable=false`、`training_consumable=false`。
- [x] 运行 EBQwen2.5-VL-3B-MLX-4bit 原片屏幕；保守完整释放链 precision=`0.50`、recall=`1.00`，
      因 VTV 中途入镜窗被误报为 `live_field_goal` 未达 `0.85` 门禁。失败屏幕 SHA 为
      `f2a8d4a3cb0633ea82d9c9965f8ce2597b8bb77a43cc42b4fda62430fdf7338d`。
- [ ] 继续补齐三场的亚秒球—手—篮筐—结果、插播边界和穷举 non-shot hard negatives，再做制作源留出
      的逐场 P/R 门禁；v3 试标和 VLM 输出不得作为训练真值或恢复盲推理。
- [x] v3 候选扫描临时目录已精确删除 52 个文件、`6,746,469` bytes；删除前 tree SHA 为
      `fccba325b30ed56174664f8054975d340f7295d5866c5cc63c55fc1e6e587f3b`，清理审计为
      `analysis_outputs/public_research/agu_causal_scan_v2_cleanup_2026-08-09.json`
      （SHA `dd446573c2efd7fc3535eaebccd2d8a917f809104db3fffc2919706b53d90a33`）。
- [x] 进一步精确清理 5 个不再引用的临时视觉目录（417 个文件、`59,775,837` bytes）；删除前 inventory SHA
      `769bd4fd353f4e3e3589e1e271f250b839f90529dd438f7b1b152a2a106fc0d5`，清理审计为
      `analysis_outputs/public_research/agu_superseded_tmp_visual_cleanup_2026-08-09.json`
      （SHA `425e712768ca02fab2b9c42382cd508f18e7aac3af0821e568433e9c9fa8b73d`）。

### 三场原片 v4 因果扩展与扫描清理（2026-08-09）

- [x] 在已保留的三场原片上新增两个边界窗口，封存
      `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v4/`：共 89 张
      哈希绑定帧、4 个窗口，包含 Hazen `shot/made`、Hazen 新增 `shot/missed`、Randolph 新增
      `not_a_shot` 运球/传球 hard-negative 和 VTV `uncertain`。Randolph 窗口经逐帧复核未见脱手—篮圈链，
      因而不把初始候选误作 shot。
- [x] v4 review/manifest/plan/retention SHA 分别为
      `6cdc32bee69379ada7a71be23e56a5857d5fb31889b3f7b45e806d6652f5670f`、
      `ce6d7a0dfe7f4a0238db509ae638c70fffe59b5d79f88d48c0efb1b13f2f26a0`、
      `06774f5a5200118e142652f8c3a30bcbafaa19e43cae8d9cf9d7765353157c7a`、
      `452c7d0190e4fc589e2da36058c79cfd546d6ab22cde2c11a2c6e9ec23f6c017`。仍为
      `pilot_only=true`、待人工确认、非穷举、不可供 runtime/训练/盲推理使用。
- [x] 删除无项目/Wiki 引用、且已被 v4 正式帧工件替代的 `/tmp/agu_shot_scan_v4.5gB2nw`：3,068 个文件、
      `263,256,416` bytes；删除前 inventory/tree SHA 为
      `f5b3464ea4e8e3e51cadb9a055a4ad43566ed044855077f8c0f42bb272bd3bfa` /
      `ecb8daf61aaf12eb6a603513b2a4074dce255ba38e9a0b3ce78b34f3827e9678`，审计工件为
      `analysis_outputs/public_research/agu_shot_scan_v4_cleanup_2026-08-09.json`
      （SHA `73ce024735b985cf4bb7016bc3033a4b3ab6a82f6adfa27a6721224f9e642796`）。
- [ ] 继续补齐三场的亚秒球—手—篮筐—结果、插播边界和穷举 non-shot hard negatives，再做制作源留出
      的逐场 P/R 门禁；v4 仍是 pilot，不改变 `not_ready`、盲推理暂停或 16 候选/0 eligible gate。

### 三场原片 v5 扩展与视觉草稿清理（2026-08-10）

- [x] 在三场已保留原片上新增 6 个有界窗口，并继承 v4 的 4 个窗口；v5 共 10 个窗口、215 张哈希绑定帧，
      新增 Hazen `shot/missed`、Hazen live-pass `not_a_shot`、Randolph `uncertain` 与 free-throw-boundary
      `not_a_shot`、VTV cut-occluded `uncertain` 与 live-halfcourt `not_a_shot`。没有新增下载。
- [x] 封存 v5 plan/review/manifest/retention，SHA 分别为
      `972a16c19e12fd357e0a742253303efc7b513aa3eaf9af5d38e2b0ed04912a37`、
      `fa3938f2ee217fe70ba10d9633325b1616a4ed9aa19f717298cc5f06a44fd9c3`、
      `b5576dd60ee0202554aa4e4a8b0714b215cc15a30fa32517699c4a6421d55913`、
      `73544ecf74434123a6abb6b7067aaf51cdf87367166fbc19ad5a672e06887b39`；仍为
      `pilot_only=true`、待人工确认、非穷举、不可供 runtime/训练/盲推理使用。
- [x] 精确删除 v5 复核产生、且无项目/Wiki 引用的 6 个 `/tmp` 视觉草稿目录及 1 个原始哈希草稿：29 个文件、
      `17,763,243` bytes；inventory/tree SHA 为
      `c9bbc932ed03192d2ac5caf76e5eabec8455cd9134b272352d13292bc6644288` /
      `eb171fa496e2b396c03bfc4d1c2952cea017cfc20dda062dec71489e9e6f6f8b`，清理审计为
      `analysis_outputs/public_research/agu_v5_visual_scratch_cleanup_2026-08-10.json`
      （SHA `a7047699b33873c04c9eb02c8d9d96dc1fc9017d5f0b849245ce4a7cfd4c5d49`）。
- [ ] 将三场窗口继续扩展到穷举事件/插播/非投篮 hard negatives，并按制作源留出做逐场 P/R ≥ 0.85 门禁；
      v5 不改变当前 `not_ready`、盲推理暂停或 16 候选/0 eligible gate。

### 三场原片 v6 因果/硬负样本扩展与视觉清理（2026-08-10）

- [x] 在线候选核验后没有下载新的 payload：受限的 SVI-Bench 不满足权利/训练条件，BASKET、TrackID3x3 和 Play-by-Play 不提供三场连续 5×5 亚秒因果真值；继续复用已保留的 Hazen、Randolph、VTV 原片。
- [x] 新增 6 个有界窗口并继承 v5 的 10 个窗口，v6 封存 16 个窗口、341 张原始 CV2/JPEG 哈希绑定帧：4 个 `shot`（1 made/3 missed）、8 个 `not_a_shot`、4 个 `uncertain`。plan/review/retention/manifest SHA 分别为 `898d2773edb9cb94c3a485b856a17f69a7a7bc5fb9f6d9793fd29a46fb17c985`、`e8f62182a9586880ee5fdcf7df7bb7cd8210a9c2527907797936e90e8aa68655`、`3b4d1b3af6ff49f5bd8b38fa0c5af13df98062f09490f4d1f328b83ed2476b83`、`d71f99db4a0c2ba99232300144d3adf3d66de137b21c3d30352e152a99f6d89e`。
- [x] v6 仍为 `pilot_only=true`、待人工确认、非穷举、不可供 runtime/训练/盲推理使用；正式目录为 `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v6/`。
- [x] 精确删除无引用的 `/tmp/agu_v6_broad.1Q6WQl`：183 个文件、`7,745,976` bytes；inventory/tree SHA 为 `5e825ac003822571bd95df6d3f4d0b534f682403ef271eb916883cfa7bd3be8f` / `c06611b1f5b70e921c321433bdf173928b8a3a4a08674aa014d733e0a92f9004`，清理审计为 `analysis_outputs/public_research/agu_v6_visual_scratch_cleanup_2026-08-10.json`（SHA `5221abb861d68b39eb5725e961ebdce6207f85533aafb96ee032e63ca0301920`）。
- [ ] 继续把三场扩展为穷举事件、插播边界和 non-shot hard-negative 真值，并按制作源留出做逐场 P/R ≥ 0.85 门禁；v6 不改变 `not_ready`、盲推理暂停或 16 候选/0 eligible gate。

### v6 独立原片 VLM 屏幕（2026-08-10）

- [x] 在不向模型暴露 v6 人工标签或复核说明的条件下，用本地 `EBQwen2.5-VL-3B-MLX-4bit` 对 16 个原片窗口执行 `strict_release_v2` 和 `negative_first_v3` 两个提示变体；两者均输出 16/16 `live_field_goal`。
- [x] 离线评估为总体 TP/FP/FN/TN=`4/12/0/0`、precision=`0.25`、recall=`1.00`；Hazen precision=`0.667`，Randolph/VTV precision=`0.0`，两个变体均不晋级。结果工件为 `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v6/independent_vlm_causal_screen_v6.json`（更新后 SHA `f709a0a4ba6c3fe70277e4f304ab2e91a28cfefdca7dfb788a5704261b91bbb3`）。
- [x] 两次运行均由资源守护完成且未触发停止：系统内存峰值 `86.6%/86.5%`、最低可用内存 `2.147/2.163 GiB`、CPU 峰值 `86.9%/80.0%`；没有 auxiliary OOF 证据，因此没有执行 runtime 融合或模型晋级。
- [x] 整批屏幕后精确删除不再引用的单例探针计划/预测/cache：3 文件、`4,662` bytes；清理审计 `analysis_outputs/public_research/agu_v6_vlm_probe_cleanup_2026-08-10.json`（SHA `b9a677ea2e7b3ca0d03e4f370c2d7bdfd9e8d55494c420811b0b70d63c42f98b`）。
- [ ] 为三场生产源取得独立 auxiliary OOF 证据并通过 fail-closed evidence gate，再重新评估逐场 P/R；在此之前保持默认模型、runtime 和 blind inference 不变。

### v6 DeepBall-Large auxiliary transfer（2026-08-10）

- [x] 在不读取 v6 review/标签的条件下，用已保留的 WASB-SBDT DeepBall-Large 权重对 16 个窗口按 2 FPS 生成事件级 transfer 分数；结果为 `runtime_consumable=false`、`training_eligible=false`、`auxiliary_oof_evidence_available=false`，`oof_predictions=[]`。
- [x] 只读诊断工件为 `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v6/independent_auxiliary_transfer_evaluation.json`（SHA `2c19eaa7c4b6f5b908bd18ec65b53dce8c8d09168700eda0117ad87ff0e48a25`）。描述性 0.75 cutoff 的 determinate pooled P/R/F1=`0.667/1.000/0.800`，但最弱 VTV 制作源 precision=`0.000`；未选择阈值、未形成 OOF、未进入 evidence gate。
- [ ] 继续取得真正 game-held auxiliary OOF rows，并在三场逐制作源 P/R 均达到 0.85 前保持 runtime、默认模型和盲推理不变。

### 三场分层不重叠 v10 held-out 离线因果复核（2026-08-10）

- [x] 以 v2–v9 源内锚点和 20 秒排除半径生成 v10 批次；每个制作源保持 `early=4`、`middle=4`、`late=4`，批次
      `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_annotation_batch_v10.json` 的规范化 SHA 为
      `d8e5c98305c2c9cebbeaeea5a352f9ab1f26f071fd096d6edec7b91d3cb29d9e`，文件字节 SHA 为
      `4ace7ddc4ce6a73e6fc56535ea8a390ead0d67d93512886fc2f6358f75995a52`。
- [x] 在 canonical `.venv` 物化并复核 36 个窗口、756 张原始/JPEG 哈希绑定帧；v10 review/manifest/plan/retention SHA 为
      `f7e85fd0b152d49f763ffded203a566904018199f71f142e4bb6d60de2f10e29`、
      `690d88bf4c36faae6cb7e22fe5d132caad6d5d1f12a7b0d328104b34446b7b59`、
      `0f9318577032f6efecc55f026b5349aa2ea86f2e8b8653ca3d2fb041c8442a91`、
      `c5c22d613fe4ae0081ed53feb327890e62570d70ff1f2500114bb9ea4160bdc6`。
- [x] 逐帧保守封存 3 个 `shot`（1 made、2 missed）、10 个 `not_a_shot` 和 23 个 `uncertain`；与 v8/v9 合并为
      96 个窗口/2,016 张帧。所有标签仍 `pilot_only=true`、`exhaustive=false`，不可供训练、runtime、VLM 答案或盲推理。
- [x] 正式 raw frames、review、manifest 和 retention 封存后，精确删除未引用的 v10 contact-sheet 目录：36 文件、
      `12,562,615` bytes；审计 `analysis_outputs/public_research/agu_v10_visual_sheet_cleanup_2026-08-10.json`，SHA
      `3509ab06f8e646f78e9143d5f7e582ae6409ca6033415f69e2b279133fffbd6c`。
- [x] 合并摘要 `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v8_v9_v10_summary.json`
      （SHA `c547a849b6b031d4a950093de965a20942097ef2f9afeca7a466d86aa03aa2eb`）；没有 runtime/held-production OOF，
      逐制作源 P/R 仍为 `not_computable`，readiness 继续 `not_ready`、盲推理暂停。
- [ ] 继续覆盖 1,922 个队列窗口，完成三场穷举亚秒事件、插播边界和 non-shot hard negatives，再按制作源留出做
      P/R ≥ 0.85 门禁；在此之前不改 runtime、默认模型或盲推理。

### v10 label-free VLM 资源探针（2026-08-10）

- [x] 从 v10 封存 review plan 派生只含原片/窗口 provenance 的独立 source plan；规范化 SHA
      `cce46b527953ac968e9d1504cdcac5e5ac645616cc40237dcefe42180ab513e9`，文件 SHA
      `be59a16d5109faa06294d8b5f6fb2563ddeeaac9afb545c8d6e231018a77a8e4`，没有向模型暴露人工标签或 review notes。
- [x] 在 canonical `.venv` 下用 `EBQwen2.5-VL-3B-MLX-4bit` 运行 3 窗口资源前缀探针；守护连续三次触发可用内存 `<2 GiB`（峰值系统内存 `91.8%`，exit `75`），无 prediction/evaluation 工件生成，因此不形成模型性能证据或晋级输入。守护日志文件 SHA
      `93b3d44ca1c7a96f95b9b60939229084d573884cd11493869c163cfec5ec361b`。
- [x] 删除可由 source plan 重建且未生成输出的 2,109-byte probe plan；清理审计
      `analysis_outputs/public_research/agu_v10_vlm_resource_probe_cleanup_2026-08-10.json`，SHA
      `301d75c0822dd535ef07fc39aea4933f2fa4cd4c7eae9a0f13e525e812e8e261`；正式 review/raw frames、权重、runtime 和 blind assets 均保留。
- [ ] 待资源守护满足门禁后再重试 label-free VLM screen；在形成独立 OOF/逐制作源 P/R ≥ 0.85 之前，不改 runtime、默认模型或盲推理。

### v7/v8 视觉中间物清理（2026-08-10）

- [x] 精确删除已完成复核且未被 review/manifest/retention 引用的 v7/v8 contact-sheet 目录：36 文件、`23,205,453` bytes；清理审计
      `analysis_outputs/public_research/agu_v7_v8_visual_sheet_cleanup_2026-08-10.json`，SHA
      `98becaf97be22b24f4cdf71e8928d60116b6c30681850ac308d2fb6fb8a12b3b`。
- [x] 保留 v7/v8 raw frames、三场原片、sealed review/manifest/retention、EBQwen native/MLX-4bit、runtime 和 blind assets；`formal_assets_touched=false`。

### 可重建缓存清理（2026-08-10）

- [x] 回归测试后精确删除仓库内 `__pycache__`、`.pytest_cache`、`.ruff_cache`、`.mypy_cache`（排除 `.venv`、`.git`、`dataset`）：19 个目录、797 个文件、`10,917,903` bytes；审计
      `analysis_outputs/public_research/agu_rebuildable_cache_cleanup_2026-08-10.json`，SHA
      `c0c165b9335f3ac448e6c9cb25ed07504571c1db9f7db376d5399f2a0583a8f5`。
- [x] 正式证据、原片、权重、runtime、training assets 与 blind assets 均未触碰。
- [x] 清理后 readiness 审计文件 SHA 为 `ec757e011bfff51b6223f394f7fa8098acbf1e0d35a63ae3e00f90f23bbbda3b`；状态仍 `not_ready`、盲推理暂停。

### 2026-08-10 在线源扫查、外部 game-held 校准与媒体清理

本轮对 BARD、E-BARD、SportsMOT、SpaceJam、Basketball Events、NBA Games 和 legacy NBA PBP video dataset 做了官方来源/仓库/HF card 审计。它们分别只能补充事件/物体标注、球员 MOT、16 帧单人动作、少量研究 clips、YouTube 访问线索或未声明兼容许可的外部 NBA 片段，均不满足 AGU 连续 5×5 原片的亚秒球—手—篮筐—结果真值与穷举 hard negatives 五项 gate。没有新的 payload 下载。扫查工件为 `analysis_outputs/public_research/online_source_sweep_2026-08-10.json`（SHA `8c9c46f42f5bd3a90c7b1905a25276b074109092d916f743670ceadcbfeb7dbd`）；source catalog 为 74 条（SHA `629566b7b6350d63327055347eb5bd0f3e9abf889e8b010b9c6bd8c05cdad5b6`），continuous causal gate 为 18 候选/0 eligible（SHA `c234381b7806c048bb20c97af4ce1d191298c5f9260b6e3c496c39656b61ac30`）。

另以三份已标注 DeepBall-Large 外部屏幕做 game-held 校准，96 事件独立选出阈值 `0.0`；校准 precision=`0.177083`，冻结后 v6 目标 determinate precision=`0.333333`、recall=`1.0`。工件 `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v6/independent_auxiliary_external_calibration_v1.json`（SHA `7d45fb7d27938aa90c8962884c8dbde851420715f243f6a0503962c1a8651b7d`）保留 `oof_predictions=[]`，所以没有证据融合或模型晋级。

### 三场全场覆盖队列与 v7 离线因果复核（2026-08-10）

- [x] 新增标签隐藏、原片 SHA 绑定的全场覆盖队列 `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_annotation_queue_v1.json`；三场按 10 秒起点网格生成 `1,922` 个窗口，内部 SHA 为 `8b3f45a6a895f05e8cc63e7c84b7011945681c39fbdb34ac85c1d8fdf6586022`。队列仅供离线人工抽样，不进入 AGU runtime、训练或 Codex 运行时答案。
- [x] 选取三场均衡的 12 个 v7 候选窗口，以 0.2 秒原片稀疏 seek 物化 `492` 张原始 CV2/JPEG 哈希绑定帧；v7 plan/review/retention/manifest SHA 分别为 `eef5dec0ed0cb929c6493e52afb85bf2ab76a6bd87758989dbf300030c32eea9`、`842b38533d37209b48f5474513c542bf6c027404d9d04d0aa47b8cc6a2fbb460`、`45bf92fb013faee826c1aebe9b99ec2ed2abba8762b14ff48f17924d9fee81b8`、`48edb5aab03592ddfaf9beb7203acc7a043717d376c42228684382b2a420a383`。
- [x] Codex 离线视觉复核保守封存 `1` 个 `shot/missed`、`7` 个 `not_a_shot`、`4` 个 `uncertain/unknown`；v7 目录为 `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v7/`，`pilot_only=true`、`exhaustive=false`、`hard_negative_gate_eligible=false`、`training_consumable=false`、`runtime_consumable=false`。
- [ ] 将队列扩展为三场穷举事件/插播边界/non-shot hard-negative 人工真值，再按制作源留出做逐场 P/R ≥ 0.85 门禁；v7 仅是离线 pilot，不改变 `not_ready`、盲推理暂停或 runtime 默认。
- [x] 复核后精确删除不再引用的 `/tmp/agu_v7_review.jo6ZgQ` 视觉草稿：18 个文件、`3,346,793` bytes；正式 v7 raw frames 和封存工件保留。

按用户授权，精确删除三份已被封存屏幕/评估替代的跨转播 MP4，共 `1,909,214,774` bytes：`media_new_dev_v1/uIxqFnBCOyk.mp4`、`media_new_dev_v2/Agffi33pz8w.mp4`、`media_new_dev_v3/waBDFY4ihS0.mp4`。source manifests、screen/evaluation 工件、三场 Wikimedia 原片、EBQwen native/MLX-4bit 与 blind assets 均保留。删除审计为 `analysis_outputs/public_research/agu_rejected_cross_broadcast_media_cleanup_2026-08-10.json`（SHA `d4c1f1907ebff932d1c3de28aced37e536f82a82c80a1e271bc0f9fa4166d4c4`）。状态继续 `not_ready`、盲推理暂停；下一步仍是三场穷举人工因果标注和制作源留出 P/R ≥ 0.85 门禁。

### 三场不重叠 v8 held-out 离线因果复核（2026-08-10）

- [x] 新增 `app/analysis/heldout_annotation_batch.py`、`scripts/build_heldout_annotation_batch.py` 及契约测试；以 v2–v7 锚点和 20 秒排除半径生成不重叠、三场各 8 个窗口的 v8 批次。批次为 `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_annotation_batch_v8.json`，SHA `8ce44f7749e40309e87685e4180c30bcc3a325ec1e364fc09f2e09b5843793c6`。
- [x] 在 canonical `.venv` 下按 0.2 秒稀疏 seek 物化 24 个窗口、504 张原始/JPEG 哈希绑定帧，并用 `--review-tag v8` 保持 plan 与目录隔离；未启动 AGU runtime，未向模型输入写入标签。
- [x] 封存 `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v8/`：1 个 `shot/made`、19 个 `not_a_shot`、4 个 `uncertain/unknown`。review/manifest/plan/retention SHA 为 `7b1f0524f2e2d6e86d260cd4ab3616f9be1cfa5da809650c1e87cca038bad1c2`、`3a55d521a202ba07d1a019169d6030267c7be0c9f3d6c813c5329e8062167e8f`、`60ec3d0a9913032aa29cdf98fb7e8628c677c2c4043053b5d107e10af2437b08`、`918b14ae25b1b969581a43a600f0969fc5cea4f4ee4f1312ab97cd228775262b`。
- [ ] v8 仍为 `pilot_only=true`、`exhaustive=false`、`hard_negative_gate_eligible=false`、不可供训练/runtime/盲推理使用；继续完成全队列三场穷举事件、插播边界和 non-shot hard negatives，再按制作源留出做 P/R ≥ 0.85 门禁。

### 三场分层不重叠 v9 held-out 离线复核（2026-08-10）

- [x] 扩展 held-out 批次契约，按每个制作源早/中/晚各 4 个窗口抽样，并以源内 20 秒半径排除 v2–v8 锚点；v9 批次为 `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_annotation_batch_v9.json`，规范化工件 SHA `78d83f204baa3c908d4bdd0d786c6c4028caf8e63cc221f9a272270c97e37857`（文件字节 SHA `51feb1f9ec6689d0def97a239b9a8886d0159d8481e13a0c4b66001f58b69d4e`）。
- [x] 从已保留的三场原片物化 36 个窗口、756 张原始/JPEG 哈希绑定帧并封存 v9 review/manifest/plan/retention；SHA 分别为 `1b00787d249b9da6b254d3022f4e9c1ac3ab36883789d1497a9598d5a1006890`、`b932f045cf0fa243337522f8cff5435f67a8996b3c003bf1a7f032043a094f5f`、`027e2a75b0f1b324038f27134d368480523c6f73e1941720f373a3dda1d70f10`、`8c4df9fe74b7c93318c16152722153c55edd1a79c0d8fc13f98e2aa90fdf1386`。
- [x] 复核封存 2 个可见但未命中的投篮链、30 个 `not_a_shot` 和 4 个 `uncertain`；v8+v9 累计 60 个窗口/1,260 张帧。所有标签仍仅用于离线 pilot，三场原片与 EBQwen native/MLX-4bit 权重未改变。
- [x] 正式封存后精确删除无引用的 v9 contact-sheet 目录：36 个文件、`13,071,798` bytes；清理审计为 `analysis_outputs/public_research/agu_v9_visual_sheet_cleanup_2026-08-10.json`，v9 raw frames 与封存工件均保留。
- [x] 将 v8/v9 离线证据合并为 `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v8_v9_summary.json`（SHA `781041510552da1a3caae00099cdb358b269e41a1a200a7b2eb5de2578745f0a`）；三场各 20 窗口/420 帧，逐制作源 P/R gate 因无 runtime/OOF 预测仍记为 `not_computable`。
- [x] readiness 审计仍为 `not_ready`、盲推理暂停；更新后文件 SHA `cd36e68aeaa398d206ce0e6de6f48389eab23efda359b3760366eb7f4fbed26c`。
- [ ] 继续完成 1,922 个全场覆盖窗中的穷举因果/插播/非投篮真值，并按制作源做独立 P/R ≥ 0.85 门禁；在此之前不得进入训练、runtime、VLM 答案或盲推理。

### v10 独立辅助筛选与在线候选复核（2026-08-10）

- [x] 对 v10 的 36 个标签隐藏窗口运行保留的 DeepBall-Large CPU transfer screen；预测工件
  `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v10/independent_auxiliary_transfer_predictions.json`
  内部 SHA 为 `ef8bd88b9161a1a16183b26a64e9a4bce08e00a3eb59f2f1c514909abfc60340`，文件 SHA 为
  `eefe51a7a3d394730acd55739922ea4d572799d03706a617e3c44d055c6cdcd3`；resource guard 文件 SHA 为
  `ebc362cfd3976d34300593cb56d2dcef238fa1ec1bb86c21f277b66bc1a1e2c7`。
- [x] 推理结束后才读取封存标签做描述性评估：13 个 determinate 窗口在 0.75 cutoff 下 pooled P/R/F1=`0.20/0.333/0.25`，
  23 个窗口为 `uncertain`；评估工件内部 SHA 为 `db3b94a97a209dee93e163701fbf9b94ee3125091f8997b8f7826ff568d5e746`，
  文件 SHA 为 `c03285e7c3d67605b95d64bfde90f5e18c1b46298e3564997d279c246d8ad489`。
- [x] transfer 工件保持 `oof_predictions=[]`、`runtime_consumable=false`、`training_eligible=false`；不选阈值、不做 runtime/VLM
  融合、不晋级模型。
- [x] 新增公开源审计 `analysis_outputs/public_research/agu_open_source_candidate_audit_2026-08-10.json`（文件 SHA
  `0c29ee4c08b0da6671898e6377a1cd64297b4994280b722fb1b17883fbc61fbd`）：Stanford bball_attention 数据入口 404，BasketEvent
  尚未发布可核验 payload，legacy NBA PBP 仅提供单个 play clips；3 个候选均未下载，source gate 仍为 0 eligible。
- [ ] 继续完成三场原片的穷举亚秒球—手—篮筐—结果、插播边界与 non-shot hard negatives，再按制作源留出计算 P/R ≥ 0.85。

### 最终 targeted regression 缓存清理（2026-08-10）

- [x] 删除目标测试重新生成的 6 个可重建缓存目录，共 25 个文件、`377,033` bytes；审计为
  `analysis_outputs/public_research/agu_targeted_test_cache_cleanup_2026-08-10.json`，SHA
  `01f43ed88bed89d930ce825da45a902e54af1dfc69d9cdbb316e4be52ee13699`。
- [x] `.venv`、`.git`、`dataset`、原片、权重、正式证据和 blind assets 均未触碰。

### v11 held-out 原片复核与保留边界（2026-08-10）

- [x] v11 从三场已保留 Wikimedia/HCTV/VTV 原片抽取 36 个源内去锚窗口、756 张原始/JPEG
  哈希绑定帧；批次内部 SHA 为
  `1080b0f6038a733a8950b9e64108806950dcfa26e294e695886b69ea63c07673`。
- [x] Codex 仅做离线逐帧人工复核，封存 34 个 `not_a_shot`、2 个
  `uncertain/unknown`，没有 `shot`；review/manifest/plan/retention 内部 SHA 为
  `c048dca489b9b5edf22c2e47e66d2acc6ab6d5348e78eccbc8ac00929873b8ce`、
  `b3ddd6ed1f982b427cd87b27cf5957171d086546786c8e9078c147be786ed913`、
  `1e50f5f27a5bef62fd8a51256df5ac2ec0d35334d1d01810afcada9801d42496`、
  `209f3e6b8b75c593ae1c59526f9e0d72fe41630372130ecf1ce800e35b5193e3`。
- [x] 仅删除已完成复核且无正式引用的 v11 contact-sheet（36 文件、13,044,216 bytes），
  保留 raw frames、原片、sealed review/manifest/retention、EBQwen 权重、runtime 和
  blind assets；清理审计为
  `analysis_outputs/public_research/agu_v11_visual_sheet_cleanup_2026-08-10.json`，
  内部 SHA 为 `dc1768b9522b6720e0d531c6f08ad8f8375ee5f8573d3d7068cbc10de3428d7a`。
- [x] v8–v11 合并摘要
  `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v8_v9_v10_v11_summary.json`
  为 132 窗口/2,772 帧（6 shot、93 not_a_shot、33 uncertain），仍为
  `pilot_only=true`、`exhaustive=false`、不可训练/runtime；逐制作源 P/R 为
  `not_computable`，readiness 继续 `not_ready`、盲推理暂停。
- [x] v11 契约测试 9 项通过；测试后精确清理重新生成的 6 个可重建缓存目录（18 文件、
  109,770 bytes），审计为
  `analysis_outputs/public_research/agu_v11_targeted_test_cache_cleanup_2026-08-10.json`。

### v12 source-disjoint held-out 原片复核与保留边界（2026-08-10）

- [x] v12 排除 v2–v11 的源内 20 秒锚点邻域，按 Hazen、Randolph、VTV 各 early/middle/late
  2 个窗口生成 18 个源间隔离窗口；批次内部 SHA 为
  `9ca56a7958f345e0fdd270d9e7a36d78e00600d39faefb3a855f6ca69e594d41`，文件 SHA 为
  `fb30dad34c6eda47532004a968d82a77a675b4eaf3177fefe7178a346bf34e6b`。
- [x] 在 canonical `.venv` 物化并哈希绑定 378 张原始/JPEG 帧，离线逐帧复核后封存
  14 个 `not_a_shot` 与 4 个 `uncertain`（含一个 release 在窗口外但可见 made 结果的边界样本），
  0 个 `shot`；review/manifest/plan/retention 内部 SHA 为
  `ae350605d6b473d22add1186715001c29d5a887857d2162ac7963e602b51891e`、
  `38b88df7b08abee8aa142801501b78884062420c7714b6f1006e45f83e4e8d8c`、
  `10d4b9986beaa2d4bc9ccd6bc8f1cb6002967d28a02491c23a2f187b09b0bd62`、
  `bee7cfa58f6b260f95cb0fd72094f951603164475d89061dab73068da979d9e1`。
- [x] 仅删除已完成复核且无正式引用的 v12 contact-sheet（18 文件、8,878,214 bytes），保留
  raw frames、source originals、review decisions/sealed、manifest、retention 和 EBQwen 权重；
  清理审计为 `analysis_outputs/public_research/agu_v12_visual_sheet_cleanup_2026-08-10.json`，
  文件 SHA 为 `6bb155eb785cf3f3d640199c9686f0a24bb115179ba3b573b22f150fd95bce28`。
- [x] v8–v12 合并摘要
  `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v8_v9_v10_v11_v12_summary.json`
  为 150 窗口/3,150 帧（6 shot、107 not_a_shot、37 uncertain），仍为
  `pilot_only=true`、`exhaustive=false`、不可训练/runtime；逐制作源 P/R 为
  `not_computable`，readiness 继续 `not_ready`、盲推理暂停。摘要文件 SHA 为
  `2e577124e0e27411cfcb2f51eeae3d6e5c03512a9676e6a0de8d2b3d175fe6f5`。
- [x] v12 focused contract tests 通过 12 项；测试重建的 5 个仓库缓存目录已精确删除
  （21 文件、`155,455` bytes），审计为
  `analysis_outputs/public_research/agu_v12_targeted_test_cache_cleanup_2026-08-10.json`，
  文件 SHA 为 `862ef758561b1ef1b18637897fad73114ed972ea21cf1eba38d2c8d070357868`；
  `.venv`、`.git`、`dataset`、原片、权重、正式证据与 blind assets 未触碰。
- [ ] 继续完成三场原片穷举亚秒球—手—篮筐—结果、插播边界与 non-shot hard negatives，再按制作源留出计算 P/R ≥ 0.85。

### v16/v18/v21 标签隐藏 native VLM 与 transfer-veto 探针（2026-08-12）

- [x] 从已封存的 v16/v18/v21 源间隔离 review 中抽取三制作源各一正一负，共 6 个 raw-video-only 窗口；plan SHA `99ef9d0ea4aa9638aafabd768cd33e53922524ed8144bdf0c70e0d1dd4045247`，输入不含标签/review notes，2 FPS、`max_pixels=151200`。
- [x] EBQwen MLX-4bit native-video 在资源守护下将 6/6 判为 live；事后评测 artifact SHA `3a900b968e664d35cc2445453b5680b3c1df0524658250376a149f6beb46f2`，pooled/逐源 P/R 均 `0.50/1.00`，不满足 0.85 门禁，不改变模型或 runtime。
- [x] DeepBall transfer 工件 artifact SHA `6619135170bead7c2d89116bc91f66a92fb177bcff383f27416aaa4771a93fb3` 无 OOF；label-free consistency 仅 2/6 同意（artifact SHA `ae1a1e318d71c5632e69ea9bff516bc9608470d7fa21955e4ebbf349bbb3abdd`）。
- [x] 新增 fail-closed `independent_shot_vlm_transfer_veto.py` 及测试；真实 veto artifact SHA `9313e1ea4f9b8b26fd4499e846a3b923505169c7e6da60d6be65fa3092beb399` 只保留 2/6，事后 P/R `0.50/0.3333`（evaluation SHA `4bd079a5c883529ab80b56c5b11a85e4f0de884ead68606efb85977e9f073b9d`）。固定阈值未选择，无标签/OOF，`runtime_consumable=false`、`training_eligible=false`。
- [ ] 继续三场全量亚秒因果、插播边界和穷举 non-shot hard negatives，并取得 game-held OOF 后再计算逐制作源 P/R ≥ 0.85；当前 source gate 19/0、readiness `not_ready`、blind inference 暂停。
- [x] 回归后精确删除本轮 34 个可重建 cache 文件（`552,785` bytes）；审计 `analysis_outputs/public_research/agu_transfer_veto_cache_cleanup_2026-08-12.json`，`.venv`、权重、原片、raw frames 和正式证据未触碰。

### v17 三场源间隔离 held-out 原片复核与新增元数据源门禁（2026-08-11）

- [x] 排除 v2–v16 源内 20 秒锚点邻域，按三场各 early/middle/late 四窗生成 v17 批次；批次内部 SHA `058f1503f67ec572981cbb12e98534eebeb1eeca9ff0e89fa5853b57087444a`，文件 SHA `1d88038e2c2378f3f90239201c5b933e017a64b88544d62f3cb265e7a22de69d`。
- [x] 在 canonical `.venv`（Python 3.11）物化并复核 756 张 hash-bound 帧；v17 sealed review 为 2 `shot/made`、33 `not_a_shot`、1 `uncertain`，其中 Hazen 2/10、Randolph 0/12、VTV 0/11+1 uncertain。sealed review 文件 SHA `45a74786eb0a5243102823901e78c4bef0e52c12748f66179c3f4a07ed37efd0`。
- [x] 封存 v17 review/pilot/plan/frame/retention 工件，合并 v8–v17 为 330 窗口/6,930 帧（18/251/61；4 made/5 missed）；逐制作源 P/R 仍为 `not_computable`，不进入训练、runtime、VLM 答案、阈值融合或盲推理。
- [x] 只删除 36 个未引用 contact-sheet（15,537,554 bytes），清理审计 `analysis_outputs/public_research/agu_v17_visual_sheet_cleanup_2026-08-10.json`；raw frames、原片、权重与正式 review 工件保留。
- [x] 将 TAL3x3（annotation-only、无配对原片/可核验 license）与 TrackID3x3（CC-BY-4.0、3×3 固定室内机位 tracking/pose 辅助）纳入 metadata-first source gate；20 candidates / 0 eligible，未下载新的 payload，未接入 runtime/训练。
- [x] 元数据探针完成后精确删除两个临时 Git clone（1,836 文件、`390,746,999` bytes）；清理审计 `analysis_outputs/public_research/agu_tal_track_probe_cleanup_2026-08-11.json`，项目 dataset、`.venv`、原片、权重与正式证据未触碰。
- [x] 50 项 focused contract tests 通过后删除 62 个仓库/模型下载可重建 cache 文件（`567,368` bytes），清理审计 `analysis_outputs/public_research/agu_v17_repository_cache_cleanup_2026-08-11.json`；EBQwen 权重本体、`.venv`、原片与正式证据未触碰。
- [ ] 继续完成三场全量穷举亚秒因果、插播边界及 non-shot hard negatives，并按制作源留出计算 P/R ≥ 0.85；v17 标签保持 offline-only，readiness 仍 `not_ready`、blind inference 仍暂停。

### v18 三场源间隔离 held-out 原片复核（2026-08-12）

- [x] 排除 v2–v17 源内 20 秒锚点邻域，按三场各 early/middle/late 四窗生成 v18 批次；批次内部 SHA `b0d8f669819cdb774fe0e97d2e7870f5e51549ac6c622c61cd9b7421c4651095`，文件 SHA `0fd117c46975dfc4d1b536cf2b198ab0016f66639e118203e1105481bda6cc6c`。
- [x] 在 canonical `.venv`（Python 3.11.15）物化并复核 756 张 hash-bound 帧；sealed review 为 3 `shot/unknown`（均 VTV；释放与篮筐链可见但 outcome 未解析）、33 `not_a_shot`、0 `uncertain`，不进入训练、runtime、VLM 答案或 blind inference。sealed review 内部 SHA `2be89b670333851626b0b187b71e681b9113d6f81558fdb0192ed7aba46a1d53`，文件 SHA `211cb37ae7ae4c40da9e23e2e8b4643ffa22394438b1363c5f157010b068861f`。
- [x] 封存 v18 review/pilot/plan/frame/retention 工件；pilot manifest、retention manifest 内部 SHA 分别为 `b24e949c9d427ac39eb07056c42a3da9765f45df56933c492ff84ef0df57acc2`、`3239facf60ab7f8a03ff35da89bab133ebc0be9f04e465c56aeefc67a848dd1e`。v8–v18 合并为 366 窗口/7,686 帧（21/284/61；5 made/5 missed），逐制作源 P/R 仍 `not_computable`。
- [x] 保留 raw frames、原片、权重和正式 review 工件，仅删除 36 张未引用 contact-sheet（17,491,608 bytes）；清理审计 `analysis_outputs/public_research/agu_v18_visual_sheet_cleanup_2026-08-11.json`，artifact SHA `44665334d5f2d47bab57cc42519bd95146c764ff7af11b1ec67d9fa4de8e5cc1`，文件 SHA `6d173fb0862c4dc1654dd5feff743d83c747d6b6647172435e12055390b4f706`。
- [x] 回归测试后精确删除 26 个可重建仓库/测试/模型下载 cache 文件（546,040 bytes）；清理审计 `analysis_outputs/public_research/agu_v18_repository_cache_cleanup_2026-08-12.json`，artifact SHA `475c2ec7afb0c09bca161f37ea2a97b39cfcd5b0f7e987df813d879557bbfa2e`，文件 SHA `808cee908061179f7243462d9dd1da9e5aa978c105c89fc8171cbd42865d9324`；`.venv`、dataset、原片、权重和正式证据未触碰。
- [x] readiness 审计当前文件 SHA `a1e7c1a93676dd7a97fa12454494603d7d790ae9a7a760f41b876c29a6e01f58`；source gate 20/0，状态 `not_ready`，blind inference 继续暂停。
- [ ] 继续完成三场全量穷举亚秒因果、插播边界及 non-shot hard negatives，并按制作源留出计算 P/R ≥ 0.85；v18 标签保持 offline-only。

### NBA_Streaming 发布核验与 label-hidden transfer screen（2026-08-12）

### v20 三场源间隔离 held-out 原片复核（2026-08-12）

- [x] 以 15 秒源内间隔离半径生成 Hazen/Randolph/VTV 各 12 个标签隐藏窗口，批次内部 SHA `97d3e00f194fbd412d6895cb40bc16a450e59d2e91b8c606b5eb5272f333f4d3`，文件 SHA `190408944e46164ccbd93902f27a3dac5ff70d32209036ec588ae8fffdc8d76d`；review plan 内部 SHA `9458866ea4d717af488384a3cd15d6288828d9bd68923574723dd99cb7eff718`。
- [x] 在 canonical `.venv`（Python 3.11）物化并复核 756 张 hash-bound 帧；封存 3 `shot/unknown`、5 `uncertain/unknown`、28 `not_a_shot`，不向 runtime、训练、VLM 答案或盲推理提供标签。sealed review、pilot manifest、retention manifest 内部 SHA 分别为 `20f9af9430056b87f8a2242231fbb5d16c0eb807d37d4a0e81f2f661cd201417`、`2e632bf20fe012c20f5e4be8969bc8ffdd718c80b440825667d1eaff904b16b3`、`766231d847aa5ba029b3ed7a3732fcd6d45a9814edc889c4f051799cfd61b061`。
- [x] v8–v20 合并为 438 窗口/9,198 帧（26/343/69），逐制作源 P/R 仍 `not_computable`，pilot-only、非穷举且无 held-production OOF。
- [x] 保留 raw frames、原片、权重与正式封存工件，仅删除 v20 36 张 contact-sheet 及 22 个项目级可重建 cache/bytecode 文件，共 58 文件、17,928,229 bytes；清理审计 `analysis_outputs/public_research/agu_v20_final_cache_cleanup_2026-08-12.json`，audit SHA `eb1d05ae220ab651b52727a96d6f3e3444721e165576ee058731c79b5f7c2554`，文件 SHA `aaa20edccac28d6a552a4911e647c3c53573d7568442359b05b98414927d91dc`。

## 2026-08-12 在线开源因果候选再审计

- [x] 以 metadata-first 门禁核验 10 个在线候选；磁盘剩余约 `5.917 GiB`，没有候选同时满足权利明确、连续 5×5 原片、亚秒球—手—篮筐—结果标签、穷举 non-shot hard negatives 与可承受载荷，故新载荷下载为 `0 bytes`。
- [x] GCB 仅保留本地 6.2 MB 元数据（2,981 条篮球条目；外部视频约 21.761 GB，license `other` 未说明）；Henu-MultiSubjects 为 manual-gated、CC BY-NC 4.0、约 40.5 GB；BASKET 为 gated、约 4,477 小时/1.93 TB；均拒绝下载。
- [x] SpaceJam/MUVY/BasketHAR 记为 auxiliary，NBA Games 记为 metadata-only，BasketEvent 记为 watchlist，SportsShot 记为 causal-mismatch；未改变 AGU runtime、训练或 blind-inference 门禁。
- [x] 审计工件 `analysis_outputs/public_research/agu_open_source_causal_candidate_reaudit_2026-08-12.json`：内部 SHA `5a47dbb7bb7705d7348acf3c598f6cccaacf78cbf11f9787f0421c093778e8eb`，文件 SHA `d434511522eac005e3f063e2dbecae950a129ee6d2dd74a69a0b8b97b6061c06`；source gate/readiness 已更新，仍 `19/0`、`not_ready`。回归后仅清理 19 个可重建项目/测试 cache，清理审计内部/文件 SHA 为 `203f7fca89f03ba569ba9623bec31ccd144e3bd6ff2773ef492ef8692e22f7f5` / `66f81531fc71e325eb3bcf7bc9f32bb0ad32abd1b654e4eda3329c9a489d35ee`。
- [ ] 继续三场全量穷举亚秒球—手—篮筐—结果、插播边界和 non-shot hard negatives，再按制作源留出计算 P/R ≥ 0.85；v20 标签保持 offline-only。

### v19 三场源间隔离原片复核（2026-08-12）

- [x] 生成 v19 批次（36 个窗口、每源 12 个、756 张 raw/JPEG 帧）；因 Hazen 队列在 20 秒排除半径下无足够候选，本批次采用 8 秒源内间隔离半径并在审计中保留该限制。批次文件 SHA `ee3c8275c4b4b3ef7163de445ac7fe32e36a9cf09c966e23951a2941f02613b5`，review plan 内部 SHA `48d54cecab27c5a2446727ad8d5e13cbd9819bb2572151a798121803a9236458`。
- [x] 完成标签隐藏的逐帧 Codex 离线复核并封存：2 `shot/unknown`、3 `uncertain/unknown`、31 `not_a_shot`；sealed review 内部 SHA `03c41aeb006cdcef7ed915bca544a339785e8d1a0ab08161d507eaa87d51ccfb`，pilot manifest 内部 SHA `c13f55d773efcd6c052c9c4e140cc1d27bd2783c101f89f7c651b3fd13e758b6`。
- [x] 只删除 contact-sheet 和可重建 cache/bytecode（初次 169 文件、14,473,033 bytes；回归后新增清理 261 文件、5,568,909 bytes）；审计分别为 `analysis_outputs/public_research/agu_v19_post_review_cache_cleanup_2026-08-12.json` 与 `analysis_outputs/public_research/agu_v19_final_cache_cleanup_2026-08-12.json`，正式原片、权重、raw frames、review 与 manifest 保留。
- [ ] v19 仍是 pilot-only、非穷举且无 held-production OOF；继续完成三场全量亚秒因果、插播边界和 non-shot hard negatives，再按制作源验证 P/R ≥ 0.85。

- [x] 论文与 HTML 页面确认 NBA_Streaming 的连续全场事件 taxonomy 和 CC BY-NC 4.0 声明；未发现官方 payload URL、哈希清单或可复现 release，故仅登记观察候选，未下载视频/标注。审计：`analysis_outputs/public_research/nba_streaming_release_audit_2026-08-12.json`，artifact SHA `3a571ad41cf443d2fc06ddbffedc13702e65c9b06c173f0f2411de468a79cd9d`。
- [x] source gate 重建为 19 candidates / 0 eligible；`nba-streaming-2026` 明确 `watchlist_no_download`，不进入训练、runtime 或 blind inference。
- [x] 在四份保留本地原片上，用 canonical `.venv` 资源守护完成同一 compact256 冻结计划（plan SHA `ab36c2ec4eb66a0f431a49a6230e28a3f1947fbb2a7a748a6a37306009cf6b7c`）的 32 个 label-hidden DeepBall transfer；独立 VLM/auxiliary 默认 0.5 一致 16/32（0.50），无标签、无 OOF、未选阈值、未输出融合/晋级工件。同计划 transfer 工件 `analysis_outputs/public_research/independent_shot_vlm_auxiliary_transfer_compact256_same_plan_v1.json`（内部 SHA `b356a238888f3f7d499a87f7bfdfc675a1b6fe0590552e9f2024a20706175758`，文件 SHA `61a00701b86a36b8bae25ddf0365a9e3c8d8059dabdac4c4a752d5f4f3a4e964`）；一致性工件 `analysis_outputs/public_research/independent_shot_vlm_auxiliary_consistency_compact256_same_plan_v1.json`（内部 SHA `e6aa8e2196a99f95f94a6e5f3b8a2bba558b81e6b2386533dc96155f1e2eae68`，文件 SHA `cdec8fe29c477c868ee0399d539b969e86081eaf5ee8489b6188ee5a2a9f78ce`）。
- [x] 屏幕后精确删除项目级可重建 cache 共 71 个文件（2,043,435 bytes），清理审计 `analysis_outputs/public_research/agu_2026-08-12_post_screen_cache_cleanup.json`，artifact SHA `b49ad388c4fd5330e4bdf9ec35a80ea2d2c29cd09e8121eaa1437e5fc8f3a7b6`，文件 SHA `2b3cf453593a7a6485fba782d442745a3ff71a42cb09d4086f352a4b2df95120`；`.venv`、dataset、原片、权重和正式证据未触碰。
- [x] readiness 更新后文件 SHA `3643646dcb45520eee2a30f4430a564e02ea99b8604a4a89a4d776b3cfff8267`；source gate 文件 SHA `399a8f1f81a339064dc8862b52d77f9248dc06c963a250ea738cd92319038bf5`；状态仍 `not_ready`、blind inference 暂停。

### 三场源间隔离 v16 held-out 原片复核（2026-08-10）

- [x] 排除 v2–v15 源内 20 秒锚点邻域，按 Hazen、Randolph、VTV 各 early/middle/late 四窗生成 36 个标签隐藏窗口；批次内部 SHA `0f02864f0925d3645c5527f21789271e3f1be0e2098c62e1b9b1234c7bb0bdba`，文件 SHA `55236cf3ea2117469a44146384a44283c3ca430216a0057f07c10a3630def3f1`。
- [x] 在 canonical `.venv`（Python 3.11）物化并逐窗复核 756 张 hash-bound 帧，封存 4 `shot/unknown`、30 `not_a_shot`、2 `uncertain`；各源为 Hazen 3/9/0、Randolph 1/10/1、VTV 0/11/1。结果不进入训练、runtime、VLM answer 或 blind inference。
- [x] 保留 raw frames、封存 review/manifest/plan/retention、原片和 EBQwen 权重，仅删除 36 张未引用 contact-sheet（15,957,173 bytes）；清理审计 `analysis_outputs/public_research/agu_v16_visual_sheet_cleanup_2026-08-10.json`，内部 SHA `83e74205450c1dfc6f1a5ed7fbcee61d97899a6a3c04dffab39a347121a80ad8`，文件 SHA `3b7ba10c6be166f8ac01c6468d66d46fce8cf8ad670e2100cfd34fb576214906`。
- [x] v8–v16 合并摘要 `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v8_v9_v10_v11_v12_v13_v14_v15_v16_summary.json`（文件 SHA `b58a7ee123dddd9edf8d880f77526f41fbd5535b06f913403259e50aeb610003`）覆盖 294 窗口/6,174 帧（16 shot、218 not_a_shot、60 uncertain；2 made、5 missed），逐制作源 P/R 仍 `not_computable`。
- [x] readiness 审计已重封，文件 SHA `5568d665497db3d83207638a74c577d73e92bda95d3aaeb62aed6a5a4fe67d42`；contact-sheet 已核空，原片、权重、runtime 和 blind assets 未触碰，状态仍 `not_ready`、source gate 18/0。
- [x] 50 项 focused contract tests 通过；测试重建的 22 个仓库 Python 3.11 bytecode 文件（540,051 bytes）已精确删除，清理审计 `analysis_outputs/public_research/agu_v16_repository_cache_cleanup_2026-08-10.json`，内部 SHA `e9b7f154b0cebbca60868659b4fb155f32cc099d0592cfdb7375e00f3a3a93a9`，文件 SHA `93bd8d6c38cf7679bad4d493f6585c54a570f1458992ad1f2d5b6be9405d838a`。
- [ ] v16 仍是 research-only/offline pilot；继续完成三场原片穷举亚秒球—手—篮筐—结果、插播边界与 non-shot hard negatives，再按制作源留出计算 P/R ≥ 0.85。

### 三场源间隔离 v15 held-out 原片复核（2026-08-10）

- [x] 排除 v2–v14 源内 20 秒锚点邻域，按 Hazen、Randolph、VTV 各 early/middle/late 四窗生成 36 个标签隐藏窗口；批次内部 SHA `f2dd7cf2d72cb34f482eb0a2b4021c8f71ecbbb97c56e47f3320f9385b14a4ed`，文件 SHA `2777915007fafd0d1feef88bb69deff986d0ca70bfbbc0aab1445651b82d8770`。
- [x] 在 canonical `.venv`（Python 3.11）物化并逐窗复核 756 张 hash-bound 帧；封存 3 `shot/unknown`、30 `not_a_shot`、3 `uncertain`，各源为 Hazen 1/10/1、Randolph 2/8/2、VTV 0/12/0。结果不进入训练、runtime、VLM answer 或 blind inference。
- [x] 保留 raw frames、封存 review/manifest/plan/retention、原片和 EBQwen 权重，仅删除 36 张未引用 contact-sheet（9,709,321 bytes）；清理审计 `analysis_outputs/public_research/agu_v15_visual_sheet_cleanup_2026-08-10.json`，内部 SHA `bf98189285052a54131f2c1d58af00ac21e0417ef34fcbfe2678850cd54129c0`，文件 SHA `799cd86c856a299e01778efb49b705b47c1ecc102896777c34a0265912b66799`。
- [x] v8–v15 合并摘要 `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v8_v9_v10_v11_v12_v13_v14_v15_summary.json`（文件 SHA `daf06875f12e800c71883faabfed101b57b5076bb40102298ddc227c814baf9d`）覆盖 258 窗口/5,418 帧（12 shot、188 not_a_shot、58 uncertain；2 made、5 missed），逐制作源 P/R 仍 `not_computable`。
- [x] readiness 审计以文件 SHA `bf25e07be2e9043d6d6d5a49e17c4c724c7cb156759bdcc7a8f6beb583536a4a` 重封；contact-sheet 已核空，runtime/训练/盲资产未触碰，状态仍 `not_ready`、盲推理暂停、source gate 18/0。
- [x] 50 项 focused contract tests 通过；随后精确删除 22 个仓库 Python 3.11 bytecode 文件（540,051 bytes），清理审计 `analysis_outputs/public_research/agu_v15_repository_cache_cleanup_2026-08-10.json`，内部 SHA `53c98d9205c6b370794142f9d178b73b539fa9ccb66515c465264cf4f65e06dd`，文件 SHA `8ff2ef8bcd613c9249231e00cfa79f5d039572bb64024a3d9edf324f218c1555`。
- [ ] v15 仍是 research-only/offline pilot；继续完成三场原片穷举亚秒球—手—篮筐—结果、插播边界与 non-shot hard negatives，再按制作源留出计算 P/R ≥ 0.85。

### 三场源间隔离 v14 held-out 原片复核（2026-08-10）

- [x] 在排除 v2–v13 源内 20 秒锚点邻域后，按 Hazen、Randolph、VTV 各 early/middle/late 四窗生成 36 个标签隐藏窗口；批次内部 SHA `da2beb057a9348131576ef21429bacf49df3aae29505018e35f3db79c84296ea`，文件 SHA `7b09f10cde60ecdd583b7a958905ccd21546526a6622549c03ac021b5405f44e`。
- [x] 在 canonical `.venv`（Python 3.11）物化 756 张 hash-bound 原始/JPEG 帧并完成 Codex 离线复核；封存 2 `shot/unknown`、26 `not_a_shot`、8 `uncertain`，各源计数为 Hazen 1/8/3、Randolph 0/8/4、VTV 1/10/1。结果不进入训练、runtime、VLM answer 或 blind inference。
- [x] 保留 raw frames、封存 review/manifest/plan/retention、原片和 EBQwen 权重，仅删除 36 张未引用 contact-sheet（12,618,602 bytes）；清理审计 `analysis_outputs/public_research/agu_v14_visual_sheet_cleanup_2026-08-10.json`，内部 SHA `ad55fb77099cfe733e490ae170d0aef676dfd488348b8392beca7759e470cbee`。
- [x] v8–v14 合并摘要 `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v8_v9_v10_v11_v12_v13_v14_summary.json`（文件 SHA `8e64033f64cee0521a4c6835016e872d55d7e7f913f55849b9b280566513ada9`）覆盖 222 窗口/4,662 帧（9 shot、158 not_a_shot、55 uncertain；2 made、5 missed），P/R 仍 `not_computable`。
- [x] 最终 48 项 focused contract tests 通过；验证过程重建的 4 个仓库 Python 3.11 bytecode 文件（22,169 bytes）已按精确清单删除，审计 `analysis_outputs/public_research/agu_v14_repository_cache_cleanup_2026-08-10.json`，内部 SHA `5619f336a132a381a9819ff56ed3709a05eb6705a7ffed515495c761eae5ec5a`。
- [ ] v14 仍是 research-only/offline pilot，不改变 18 candidates / 0 eligible source gate；继续完成三场穷举亚秒因果、插播边界和 non-shot hard negatives，再按制作源留出计算 P/R ≥ 0.85。

### v13 source-disjoint held-out 原片复核与保留边界（2026-08-10）

- [x] v13 排除 v2–v12 的源内 20 秒锚点邻域，按 Hazen、Randolph、VTV 各 early/middle/late 四窗生成
  36 个标签隐藏窗口；批次内部 SHA 为
  `72cd83c8a29cd1e2460d784f205bd4a1ece7dbf036d22b9d23c801b0520fec9`，文件 SHA 为
  `25da96817c4697ddeb1c013ffb5694355b1b90bb2dbbb3669e0028339ab3630c`。
- [x] 在 canonical `.venv` 物化并复核 756 张 hash-bound 原始/JPEG 帧；封存 1 个 VTV
  `shot/missed`（release position 6、rim position 9）、25 个 `not_a_shot` 和 10 个
  `uncertain/unknown`。review/manifest/plan/retention 内部 SHA 为
  `985ef02f2272df85ea17ecab7fbf4c97f7fbe56aa222a4120e2c19586822da73`、
  `d94b51eb664eba17a1638fb6383440b506b0dc269578bdae9cd09fe9d0b2c8d9`、
  `78ab158f4c6b92cb38250851b65732dd214b7563addf07ad6b9b72ad93bd653c`、
  `db25caffeca7ae6da2a12c503f4ffc2c6703679fcb39190841331150b33b1c33`。
- [x] 仅删除复核后不再被正式工件引用的 36 张 contact-sheet（12,883,187 bytes），保留 raw frames、
  原片、review decisions/sealed、manifest/retention、权重和运行/盲推理资产。清理审计为
  `analysis_outputs/public_research/agu_v13_visual_sheet_cleanup_2026-08-10.json`，内部 SHA
  `053b92f7906b5d64ca84b13d17ed894f5e6e953805c187dc6a091a2c5f1d2578`。
- [x] v8–v13 合并摘要
  `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v8_v9_v10_v11_v12_v13_summary.json`
  为 186 窗口/3,906 帧（7 shot、132 not_a_shot、47 uncertain），每制作源 62 窗口/1,302 帧；
  逐制作源 P/R 仍为 `not_computable`，readiness 继续 `not_ready`、盲推理暂停。摘要文件 SHA 为
  `abcfd58226c28dfe7e084bc2bc32bb282ce00dd38b38db9542c04e716205c6d4`。
- [x] v13 focused contract tests 通过 48 项；随后精确删除 4 个重建的仓库 Python bytecode 文件
  （22,169 bytes），清理审计为
  `analysis_outputs/public_research/agu_v13_repository_cache_cleanup_2026-08-10.json`，内部 SHA
  `374bacdb6de3e9edb43621997d5a8a1ab01b4aa57fc9129f56b5ec088f390c24`；readiness 文件 SHA 为
  `47e1182e7337edf00ed180cb840d91b67af072497243b121c3f740b8f7f09e4e`。
- [ ] 继续完成三场原片穷举亚秒球—手—篮筐—结果、插播边界与 non-shot hard negatives，再按制作源留出计算
  P/R ≥ 0.85；v13 标签不进入训练、runtime、VLM 答案、阈值融合或盲推理。

### NBA Games v2 配对 PBP/原片复核增量（2026-08-10）

远端 [choucsan/NBA_Games](https://huggingface.co/datasets/choucsan/NBA_Games) 已固定到 revision
`cf59e3a42413e3ab91f6fc0b1618f280df9a7024`。该仓库的 MIT 许可覆盖已发布的结构化元数据；其
NBA/YouTube 原片权利没有随元数据转移，原片仍按外部非商业研究边界处理。

- [x] 只为本地已有 5 场原片补回 15 个小型 `metadata.json`、`box-score.jsonl`、`play-by-play.jsonl` 文件，
  共 `1,747,339` bytes；配对清单为
  `analysis_outputs/public_research/nba_games_pbp_v2_paired_media_manifest.json`，SHA
  `27f0d0c1f3fa11e36ef929a7878906ca2f67d000b00801432df80d6b6039413b`。
- [x] 5 场合计 2,260 条 PBP 行、368 条 made-shot、412 条 missed-shot；两场媒体/PBP 计划的
  `two_game_media_gate_ready=true`，但 `two_game_metadata_gate_ready=false`（交叉比赛 roster coverage 为 0），
  因此不满足独立 acceptance gate。
- [x] HOU–SAC Q1 原片 300–600 秒采用 `.venv` 的 RapidOCR 1 Hz 时钟读数，经同节时钟单调性清洗后
  接受 261/300 个读数，映射 38/515 条 PBP 行，生成 18 个标签隐藏窗口；复核覆盖为 14 个可见、3 个部分可见、
  1 个不可见，未标注精确 release 帧。结果在
  `analysis_outputs/public_research/nba_games_pbp_v2_hou_sac_q1_review_queue/`，总览 SHA
  `cb2229dd60faf38e5dd62e9f22245bd61e00799c189189d6d0b9c23ee03e73de`，人工复核 SHA
  `b6d8ea329facb22eb2fa4b7eb8ed07bb75a9a9fcbbe3994f5b6f222c71e1d4f2`。
- [x] 复核队列由官方 PBP 选样，PBP 时钟不等价于帧级因果真值；所有队列、时钟对齐和人工结果均为
  `runtime_consumable=false`、`training_consumable=false`、`independent_evaluation_eligible=false`，不能进入
  runtime、VLM 答案、训练或 blind inference。
- [x] 先下载的 4 场没有本地配对原片，已连同可重建 Hugging Face cache 精确删除 68 个文件、`1,452,924` bytes；
  清理审计为 `analysis_outputs/public_research/agu_nba_games_pbp_v2_unpaired_cleanup_2026-08-10.json`，SHA
  `61cd28d0c107631357dfc06b54e665874eb32d2e4fac348b257213f435467e54`。正式资产、原片、权重和 blind assets 未触碰。
- [x] 人工复核后删除不再被工件引用的 2 张总览 contact sheet（`1,348,948` bytes），保留逐窗 raw strips、队列、时钟、
  truth join 和人工结果；清理审计为 `analysis_outputs/public_research/agu_nba_games_pbp_v2_review_sheet_cleanup_2026-08-10.json`，
  SHA `296bdfa9db7e61fa5ea32c882ae9eedee4b1758b6a412f83c64767c2ec8fe1ae`。
- [ ] 继续为已配对原片完成完整时钟/插播边界、精确 release、球—手—篮筐—结果标签以及穷举 non-shot hard negatives；
  source gate 仍为 18 candidates / 0 eligible，readiness 仍为 `not_ready`、盲推理继续暂停。

#### ATL–CHI 四节标签隐藏复核增量（2026-08-10）

- [x] 复用已哈希绑定的 ATL–CHI（`aIgjLjACcgM.mp4`）合并 OCR/PBP 对齐工件，覆盖四节 88 个
  mapped field-goal 行；新建不泄露 `period`、`clock`、球员或结果的标签隐藏队列，队列内部 SHA
  `25dbee3026997b9416748d4324d94a9f2928e8e184ef9e68c9302d84a627e3dd`，文件 SHA
  `73a5dbcc40c8575c799553214a7f9aea4451b47eaea533fc997a743f00ccd732`。
- [x] 从四节队列均匀选取 24 个窗口，使用高分辨率六帧 raw strips 做 Codex 离线视觉复核：21 个
  `confirmed`、3 个 `partial`、0 个 `not_visible`、0 个精确 release 帧；视觉结果为 18 made、6 missed，
  但仍为 PBP 选样诊断，不是独立 held-out 评测。人工工件内部 SHA 为
  `d9149471ff18817578ef08ed39ee618bc9e81247655b4c2fa8e0646b80717fd3`，coverage 文件 SHA 为
  `01b9d0ff1809372a50a13bb527270bffc19dd931a99c1d73fbbfaab02ca07acc`。
- [x] 官方 PBP 真值单独写入 `source_truth_post_freeze.json`（内部 SHA
  `608ab69ce391c0ead9d5eaf7e10f518167d1d80cba09c79cb73052c234b1653b`），所有队列/人工/真值工件保持
  `runtime_consumable=false`、`training_consumable=false`、`independent_evaluation_eligible=false`。
- [x] 复核后精确删除 24 个低分辨率 strips 与 2 张临时总览图，共 26 个文件、`6,057,647` bytes；清理审计
  `analysis_outputs/public_research/nba_games_pbp_v2_atl_chi_review_queue/cleanup_audit.json`，内部 SHA
  `db5f19f7c3cd06e2ecb43026932eea7b4e67f25bb73f11ae44a56eac970dd5db`，高分辨率逐窗 strips 保留。
- [x] 复核脚本生成的两份精确 Python 3.11 bytecode 也已删除，共 2 个文件、`35,312` bytes；审计
  `analysis_outputs/public_research/nba_games_pbp_v2_atl_chi_review_queue/targeted_cache_cleanup.json`，内部 SHA
  `7f58325e6a6574db68880f996490655e085e7b1b80d19138707b8fc0ca235002`。
- [ ] 继续为配对原片完成完整插播边界、精确 release、球—手—篮筐—结果标签和穷举 non-shot hard negatives；
  本增量不改变 source gate（18 candidates / 0 eligible），readiness 仍为 `not_ready`、盲推理继续暂停。

### v21 三场源间隔离离线因果复核（2026-08-12）

- [x] 在 canonical `.venv`（Python 3.11.15）以 15 秒源内排除半径完成 Hazen、Randolph、VTV 各 12 个窗口，共 36 窗口/756 张 hash-bound 原始帧；封存 1 `shot/unknown`、8 `uncertain/unknown`、27 `not_a_shot`，唯一完整闭合窗口为 Hazen 罚球且结果仍 unknown。
- [x] v8–v21 合并为 474 窗口/9,954 帧（27 shot、370 not_a_shot、77 uncertain）；逐制作源 held-production P/R 仍 `not_computable`，批次保持 pilot-only、非穷举、不可训练/不可运行时消费。
- [x] 保留 raw frames、原片、review/manifest/retention 正式工件和 EBQwen native/MLX-4bit 权重；删除 v21 contact-sheet 36 文件/17,163,891 bytes，项目级可重建 cache/bytecode 初次 8 文件/63,519 bytes、回归后最终 19 文件/145,007 bytes，合计 55 文件/17,308,898 bytes。清理审计为 `analysis_outputs/public_research/agu_v21_visual_sheet_cleanup_2026-08-12.json`、`agu_v21_repository_cache_cleanup_2026-08-12.json` 与 `agu_v21_final_cache_cleanup_2026-08-12.json`。
- [x] source gate/readiness 已更新，仍为 19 candidates / 0 eligible、`not_ready`、blind inference 暂停；新在线候选 follow-up（Qlean、NSVA、leHarris、gsbasketball）为 4 candidates / 0 eligible / 0 media downloads。
- [ ] 继续完成三场原片全量亚秒球—手—篮筐—结果、插播边界及穷举 non-shot hard negatives，再按制作源留出计算 P/R ≥ 0.85；v21 标签不进入训练、runtime、VLM 答案或盲推理。

### v8–v10 独立原片 VLM 小屏与 Basketball Events 元数据复核（2026-08-10）

- [x] 从已封存的 v8–v10 人工复核计划中构造 6 个标签隐藏的原片窗口（Hazen、Randolph、VTV
  各 1 个 `shot` 与 1 个 `not_a_shot`），只读取保留的三场原片；source plan 内部 SHA 为
  `177a6fc9d438e665132d9068ab4f5f68cbd59fc994242229318ad46275b5a66a`。
- [x] 使用本地 `EBQwen2.5-VL-3B-MLX-4bit`、`strict_release_v2`、2 FPS 原生时序输入，在
  canonical `.venv` 的资源守护下完成 6 个窗口。模型把 6 个窗口全部判为
  `live_field_goal`；评估为 pooled P/R/F1=`0.50/1.00/0.6667`，三场逐场 precision 均为
  `0.50`。评估内部 SHA 为 `97a50d479227ccbc88825ff273b584edb2df879aab29c166b53f0c65fcf3720b`，
  文件 SHA 为 `307c32e118eda259d7a0d4afa4a0609828622521cd998da2874b44173c2b5577`。
- [x] 资源守护 exit=0；日志文件 SHA 为
  `99060f90f362a3b252c459616a58d7fb94af1573c57dd652794747d01275933e`，期间有短暂可用内存低于
  2 GiB 但没有连续触发停止。该试验仅为独立 VLM 诊断，`runtime_consumable=false`、
  `training_consumable=false`、`promotion_eligible=false`，不进入 evidence gate、runtime 或盲推理。
- [x] 在线再次检查 `saveerjain/basketball-events` 时只短暂下载 10 个 README/annotation metadata
  文件（1,631,871 bytes），未下载视频；由于内容与现有
  `dataset/public_sources/basketball_events_v1/` 重复，临时副本已删除。复核工件为
  `analysis_outputs/public_research/agu_basketball_events_metadata_followup_2026-08-10.json`，
  内部 SHA `a1f295a10a1b77d6e9344ce6ba0a800fb41e3780a38ddbe741eb7c07fb0f308f`；源仍是
  research-only，不能补足连续亚秒因果真值。
- [x] 推理和 focused checks 完成后清理仓库可重建缓存（22 个目录、608 个文件、`7,543,909` bytes）；
  清理审计为 `analysis_outputs/public_research/agu_native_vlm_probe_cache_cleanup_2026-08-10.json`
  （SHA `48341ed81bc1bbb77fa49f3cd8942fe531ae5e22000adc49830672da910c8c30`），`.venv`、`.git`、
  `dataset`、原片、权重和正式证据未触碰。
- [ ] 继续完成三场原片穷举亚秒球—手—篮筐—结果、插播边界与 non-shot hard negatives，再按制作源留出计算 P/R ≥ 0.85。
- [x] 2026-08-12 v2 在线候选审计：以 `.venv`（Python 3.11.15）复核 `nba_pbp_video_dataset`、SportsAction、SHOT7M2、NSVA subset、NBA_Games、SportsMOT、Henu-MultiSubjects、SVI-Bench 的官方 API/树/README；8 个候选均未同时满足权利、可复现载荷、连续 5×5、亚秒因果标签和 exhaustive hard-negative 门禁，0 bytes 新媒体下载。审计为 `analysis_outputs/public_research/agu_open_source_causal_candidate_audit_v2_2026-08-12.json`（audit `b0318e9ff9515213457007431235067cb234e08d7dbd75f16151b9fb33a9dc91`，文件 `be23577c52a494b1342b01dbcf92b781ca9af0539b2cfc27947c4c3a83f7fbd9`）。
- [x] v2 同计划 compact256 的 32 窗口 label-hidden VLM/DeepBall transfer-veto 已封存：固定阈值保留 16/32、abstain 16/32；post-inference P/R=`0.625/0.625`，无 OOF/阈值选择，工件不进入 runtime、训练或默认答案。
- [x] v22 重新合并 v8–v21 中心后按 15 秒源内排除半径抽取三场各 12 个 label-hidden 窗口（36 windows，batch SHA `0d19ae26246dfc2f015cd03564abdce99cb60aa23d4dc1d80d150062a9eb0848`）；完成 3 窗口/63 帧受控 smoke，封存为 1 `not_a_shot`、2 `uncertain`，不进入训练/runtime/晋级。
- [x] v22 资源暂停后的 568 个未完成 raw/contact-sheet 中间文件已精确删除，smoke raw frames、封存工件和 v22 label-hidden plan 保留。
- [x] v22 随后改为每窗口单次 seek 后顺序解码，完整物化并复核 36 windows/756 frames；封存为 30 `not_a_shot`、6 `uncertain`，不进入训练/runtime/晋级。contact sheets 与临时顺序 smoke 副本已清理，正式 raw frames/manifest 保留。
- [x] v23 在 v8–v22 锚点外按 15 秒源内排除半径抽取 Hazen/Randolph/VTV 各 5 个 label-hidden 窗口，完成 15 windows/315 frames 的顺序物化与离线复核；封存为 1 `shot/missed`、14 `not_a_shot`，仍为 pilot-only，不进入训练、runtime、默认答案或晋级。批次文件 `analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_annotation_batch_v23.json`（内部 SHA `8e154fb03ac7dd063ed1ceb7b9851c4c8890c609aa2d4628c3a58f61273b90c3`，文件 SHA `e7fc2f712fb76ab4bacab921e7bf8e1e9a2ece085023088675feaa0f1624ca08`），sealed review 内部 SHA `d57fab949a2dc6c83851d20d28d1105adefc9f2bf3613f8716876e7391a16a78`，pilot manifest 内部 SHA `0ae9c43e0b3f2cdf8e4bab4d91ca005942994541863b6d7d0e2f90a6e3133a97`。
- [x] v23 复核后删除 15 张可重建 contact sheet（`5,389,158` bytes），raw frames、formal manifests、源视频和 EBQwen native/MLX-4bit 保留；清理审计 `analysis_outputs/public_research/agu_v23_contact_sheet_cleanup_2026-08-12.json`（内部 SHA `b52679d64e58e3a6210675d96891eb1cc644b3a35e29fc216d7d1286f0611a12`）。
- [x] v23 focused regression 16/16 通过；删除测试重建的 133 个 bytecode/cache 文件（`3,002,360` bytes），`.venv` 依赖、正式 raw frames、原片和权重未触碰。清理审计 `analysis_outputs/public_research/agu_v23_regression_cache_cleanup_2026-08-12.json`（内部 SHA `893bf57c422e7d33d8f441a999017a15807a1833b4c13f96a5ff09b023fc2604`）。
- [x] v23 后 readiness/source gate 指针已重封并校验：readiness 文件 SHA `4db4c19e626e5e90d68e8dbf48a4f57cff136e37c075b4a1471039daf0d4c394`（audit `14ff7a113d0c68563dc7652cbcd9d22ab6558131a4f634cfead5148e4dae2136`），source gate 文件 SHA `409876a321af973cc2c68e6b3837c896e706d4b5d5e2b3aef5cf403ac7cf7ed6`（audit `5e2e05a6c118e46419ada1c19af0d2cd32510406ac2873ad4c996293f03d807`）。
- [ ] v8–v23 仍不构成三场全量因果真值：累计 525 windows/11,025 frames，逐制作源 held-production P/R 仍 `not_computable`，source gate 19/0，readiness `not_ready`、盲推理暂停；继续完成全量亚秒球—手—篮筐—结果、插播边界、穷举 non-shot hard negatives 与 game-held OOF。

### 2026-08-13 在线候选 v3 与融合重放

- [x] 在已获得用户下载授权后，按官方来源固定 9 个候选并先做 metadata-first 门禁。匿名读取 `leharris3/basketball-shot-test-dataset` 的文件返回 `GatedRepo/HTTP 401`；`zaywas/BasketEvent` 无声明许可证；VRU、Qlean 和 SportsMOT 检查分别受体积/门禁/任务或许可约束；BARD 与 Basketball Events 的本地元数据/受限包已存在。没有候选满足五项连续因果门禁，未下载新媒体（0 bytes）。审计文件 `analysis_outputs/public_research/agu_open_source_causal_candidate_audit_v3_2026-08-13.json`，audit SHA `0f1b3c46d843ff01d572a7fc26cdf3722a5e761e0e1d0faff17a7f24cc2d4655`，文件 SHA `1557c5994c101532273657a3570ac56e8b52919eb3e5f54788ee98e746b54f61`。
- [x] 以 `.venv` Python 3.11.15 重放冻结 32 行 `both_confirm` 基座/VLM 规则；封存预测 hash 一致，峰值 RSS `374,964,224` bytes，TP/FP/FN/TN=`0/0/16/16`、unknown=`8`、P/R=`0/0`，`promotion_eligible=false`。记录 `analysis_outputs/public_research/agu_independent_base_vlm_cross_game_replay_2026-08-13.json`，内部 SHA `cc31fbbc0d76d2d9adb61e3c13ea95ca35bc86bde3963f228188378c8f0b022d`，文件 SHA `60cb0f7c8119e143ec1fb2c891583e622c0f226ebee558ab6e9767eed1649628`。
- [x] 41 项聚焦契约测试通过；删除本轮可重建 pyc/pytest cache 与 12 个 `/tmp/agu_*` 探测副本，清理审计 `analysis_outputs/public_research/agu_2026-08-13_regression_cache_cleanup.json`（audit `34b8ffa8be83593428d6b9f4faab3d5355ee0ea97a49f37f6dbeead5a586f6d1`，文件 `a951bbe37db10c74e84031b6ad178ee5fb834136ecd254fc02fd9560e2c15afd`）。
- [ ] 继续 source-disjoint 全量标注、插播边界、穷举 non-shot hard negatives 和 game-held OOF；在源 gate 19/0、readiness `not_ready`、blind inference paused 状态改变前，不将 Codex 离线标注或研究融合结果接入 AGU runtime。

### 2026-08-13 v24_rv 离线标注增量与队列容量

- [x] 在 15 秒源内排除门禁下审计 1,922 个队列窗：Hazen 可用留白 `0`、Randolph `141`、VTV `482`；Hazen 在当前队列/策略下耗尽，未放宽门禁或复用邻域。审计：`analysis_outputs/public_research/agu_hctv_annotation_coverage_exhaustion_2026-08-13.json`（audit `90f9d0325fdd94ba4c69e4c464df4ed5e88a3c3e042272c6c8db057a6523673f`）。
- [x] 从 Randolph/VTV 各取 4 个 label-hidden 窗口，168 张 hash-bound 原片帧完成人工复核，封存为 4 `not_a_shot`、1 `shot/made`、3 `uncertain/unknown`；sealed 文件 SHA `747efdd005be26c87d2ad0b27728a4cf04666e23b5e5b46e1d5f1736460e574c`，pilot 文件 SHA `7af820161ba0616c7e263d74d345c7826402a85eabce1a15ae20cc2bc9bdf1b7`，retention 文件 SHA `9621356a8143168f891627bb4d7e667b89b22e3c0383cec9591f3328104a3b7a`。
- [x] v24_rv 明确为 pilot-only/offline，不进入训练、runtime、VLM 答案、融合、晋级或盲推理；复核后精确删除 16 张 contact sheet（`15,395,408` bytes），原片帧和正式 manifest 保留。
- [x] v24_rv focused regression `13 passed`，峰值 RSS `94,076,928` bytes；清理 102 个 pyc、19 个空 cache 目录和 4 个 pytest cache 文件（`2,304,638` bytes），清理审计已封存。
- [ ] 不能把 v24_rv 当作全量因果真值；继续三场全量亚秒事件/插播边界/non-shot hard negatives 与 game-held OOF，官方 readiness 仍 `not_ready`、blind inference 仍暂停。

### 2026-08-13 v25_rv Randolph/VTV source-disjoint 人工标注增量

- [x] 沿用 15 秒源内排除半径，从 Randolph/VTV 各抽取 early/middle/late 两个标签隐藏窗口，共 12 windows、252 张原片帧；review plan 内部 SHA `eb986c52b73d88a1abff4b8c8ca25f6e9bd83210103eef15c5d563d02a86929a`，raw-frame manifest 内部 SHA `efb6d266542e2ee2ca4ece8fe685ca983720e84a878bbab39b4a3df4152adc04`。
- [x] 完成离线逐帧复核并封存 11 `not_a_shot/not_applicable`、1 `uncertain/unknown`，没有新增可确认 shot；sealed 文件 SHA `4171b78327081920ac4fe5b72163b1096d604cba0178c6e935b0943e88f9744f`，pilot 文件 SHA `3e85bb628caba9081f207acd83ba40f534861973186fa6640e1937a80eca003e`，retention 内部 SHA `84375f2510b235ab78a149658ea1c591d695b5f29cea674633a1aa29b68d4bf5`。
- [x] 批次保持 `pilot_only`、`runtime_consumable=false`、`training_consumable=false`、`hard_negative_gate_eligible=false`，不接入 AGU runtime、训练、VLM 默认答案、融合或 blind inference；删除已复核的 12 张 contact sheet（`5,617,098` bytes），保留 raw frames/正式工件。清理审计：`analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v25_rv/cleanup_audit.json`。
- [x] 针对性回归 `13 passed`，峰值 RSS `93,798,400` bytes；无服务代码、模型或 readiness 指针变更。
- [ ] v25_rv 仍不能替代全量三场因果真值；继续插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF，readiness 继续 `not_ready`、blind inference 继续暂停。

### 2026-08-13 基座迁移诊断与数据保留边界

- [x] 对现有公开候选继续执行 metadata-first 审计；没有同时满足可用权利、连续全场原片、亚秒球—手—篮筐—结果标签、穷举 non-shot hard negatives 和受控载荷的来源，v3 审计记录 `eligible_source_count=0`、`downloaded_bytes=0`，因此没有为了“有数据”而新增大体积下载。
- [x] 对已封存 v23/v24_rv/v25_rv 的 31 个确定窗口运行通用 `r2plus1d_v3` 动作头 whole-frame proxy 迁移审计：`shoot` TP/FP/FN/TN=`0/0/2/29`，P/R/F1=`0/0/0`，rank AUC=`0.465517`；这不是训练集或部署基准，结果仅用于证明当前基座与因果 shot 任务不匹配。
- [x] 保留正式原片、raw frames、review/manifest/retention、checkpoint、`dataset/public_sources/open_models` 和 canonical `.venv`；删除审计/回归生成的 35 次项目级 `.pyc`/pytest-cache 文件（370,712 bytes；32 个 distinct paths），不清理 `.venv` 依赖。
- [ ] 在得到可复现的 SHA 绑定原片候选 bundle、逐事件因果真值和至少三场 game-held OOF 之前，不启动新的 shot-validity 训练，不修改 runtime/default/VLM，不恢复 blind inference。

### 2026-08-13 已取代 pilot 原始帧清理

- [x] 对已被正式 review/manifest/摘要/retention 工件替代、且从未进入 runtime 或训练的 v2–v22
  `raw_frames` 执行 allowlist-only 清理：删除 `12,021` 个文件、`3,917,617,577` bytes。
- [x] v23、v24_rv、v25_rv 的保护帧分别为 `315/168/252`，执行前后未变化；三场原片、正式
  review/manifest/retention、checkpoint、`open_models` 与 `.venv` 均保留。审计为
  `analysis_outputs/public_research/agu_superseded_pilot_raw_frames_cleanup_2026-08-13.json`，
  内部 audit SHA `123e8952c469be5f272576df4d7e15f84e2b668ea037fe764f2eb74ca70ab353`，文件 SHA
  `746e89fddc28eac3bf7e33ef7b98721fec9afaa626246ec2f0b1477289939eee`。
- [x] `tests/test_prune_superseded_pilot_raw_frames.py` 覆盖 dry-run、allowlist 删除与保护批次，
  `2 passed`；脚本通过 Ruff/py_compile。清理不改变 `not_ready`、`blind_inference=paused` 或
  三场全量因果与 game-held OOF 门禁。

### 2026-08-13 四场旧标注链当前基座迁移诊断

- [x] 只读配对核验旧四场 SHA 绑定原片与 annotation manifest：`374` windows、`4` games、
  `173/201` positive/negative；不混入 v23–v25_rv pilot 标签。
- [x] 以当前 `r2plus1d_v3`、`agu_v3` preprocessing、`layer4`、1 epoch 做资源守护 game-held OOF；
  pooled TP/FP/FN/TN=`0/0/173/201`，P/R/F1=`0/0/0`，四个 held-game recall 全为 `0`，未晋级。
  metadata/guard 文件与 SHA 记录于 `agu_r2plus1d_currentbase_full374_agu_v3_layer4_e1_2026-08-13.*`。
- [x] 诊断 checkpoint 未晋级且可重建，删除 `125,393,565` bytes；清理审计为
  `agu_r2plus1d_currentbase_full374_agu_v3_layer4_e1_checkpoint_cleanup_2026-08-13.json`。
- [ ] 该结果不能证明训练完成；继续补齐 rights/causal/exhaustive hard-negative 数据和跨比赛 OOF，
  在 readiness 门禁闭合前不改 runtime/default、不恢复 blind inference。
## 2026-08-13：NBA Games 本地镜像一致性与受控 VLM 试跑

- [x] 对在线 NBA Games 页面和本地 `nba_games_v1` 做索引/目录/PBP 一致性核对：189 场、5,194 条 box-score、81,355 条 PBP，166 场非空；视频载荷 0 bytes。
- [x] 新增 `scripts/audit_nba_games_local_mirror.py` 与 `app/analysis/nba_games_local_mirror.py`，契约测试覆盖索引不匹配和零媒体边界；审计保持 metadata/PBP-only、不可训练/不可运行时晋级。
- [x] 在资源守护下启动 8 窗口本地 qwen3-vl:2b 原帧试跑；连续三次可用内存低于 2 GiB 后按安全规则 exit 75，未产出半成品预测；仅保留守护日志并删除 `/tmp` 可重建缓存。
- [ ] 仍需取得权利清晰、连续 5×5、亚秒球—手—篮筐—结果和穷举 non-shot hard negatives，才能进入训练与 game-held OOF。

## 2026-08-13：冻结 shot-validity 头与 Kinetics-400 对照

- [x] 四场旧标注链（374 windows、173/201）完成当前 AGU `fc`/uniform 两 epoch 对照：OOF TP/FP/FN/TN=`3/4/170/197`，P/R=`0.4286/0.0173`；未晋级，临时 checkpoint 已删除，审计为 `agu_r2plus1d_currentbase_full374_agu_v3_fc_e2_checkpoint_cleanup_2026-08-13.json`。
- [x] 训练侧 `anchor` 采样对照解析全部 374 个证据锚点，但 OOF 仅 TP/FP/FN/TN=`1/0/172/201`，P/R=`1.0/0.0058`；没有跨场收益，临时 checkpoint 与审计 `agu_r2plus1d_currentbase_full374_agu_v3_fc_anchor_e2_checkpoint_cleanup_2026-08-13.json` 已清理。
- [x] 下载官方 torchvision R(2+1)D-18 Kinetics-400 权重做标准预处理冻结 `fc` 对照；OOF TP/FP/FN/TN=`0/0/173/201`、P/R/F1=`0/0/0`，守护最低可用内存约 2.46 GB、CPU 峰值 72.4%，没有模型/运行时晋级。
- [x] 无用官方权重已删除 `126,162,996` bytes；来源 URL、文件 SHA、实验 metadata/guard 与恢复说明保留在 `agu_r2plus1d_kinetics_weight_cleanup_2026-08-13.json`。该实验只证明当前迁移契约仍不匹配，不把旧候选或 pilot 标签转成训练真值。
- [ ] 继续寻找/构建权利清晰、连续全场、亚秒因果且包含穷举 hard negatives 的 source-disjoint training bundle；readiness 继续 `not_ready`，blind inference 继续暂停。

### 2026-08-13 v26_rv Randolph/VTV source-disjoint 人工标注增量

- [x] 在既有 592 个 review-plan 中心锚点外按 15 秒源内排除半径选择 Randolph/VTV 各 early/middle/late 1 个标签隐藏窗口，共 6 windows/126 frames；批次 SHA `de545261fd884c332f6f95d12162b43ed7f5996271f1c90a623fa8e524c19f`，review plan SHA `01fc47d1f59d86bf2ecbcfe053ca744862d9a23c4147de7b679a65be6258551c`。
- [x] 完成离线原片复核并封存 6 `not_a_shot/not_applicable`（1 个直播运动态势窗口保守记为非投篮），无新增可确认 shot；sealed SHA `57ea79c3e8307981eda48684e30619c8cba1953bcdd795841638d7ea01936ec4`，pilot SHA `61ca013b26f4430d8e13f7af15b1e3403ac49ac3b9381e1423fde04a7799d60d`，retention SHA `d00a4af893582bef4ab766e0768f87589f57997548fb91d5fc26195fa1d29f89`。
- [x] 扩展 held-out batch 的显式源子集契约（`source_ids`/`--source-id`），加入未知/重复源与子集配额测试；6 项 focused tests、Ruff、py_compile 通过。
- [x] 删除已查看的 6 张 contact sheet（`1,831,087` bytes），保留 raw frames 和正式 review/manifest/retention；清理审计 `analysis_outputs/public_research/agu_v26_contact_sheet_cleanup_2026-08-13.json`，audit SHA `67fa2bac9c4c8ead242fbdabf54483b4bff1348bc45d40aa113267fdd573f15d`。
- [x] 回归后删除本轮项目级 `.pyc`/pytest/Ruff cache，并核验项目侧均为 0（`.venv` 未触碰）；清理审计 `analysis_outputs/public_research/agu_v26_rv_regression_cache_cleanup-2026-08-13.json`，audit SHA `7770905d5fb13704258531312504006d8a21ad1bb67b7b7270827ea19dabcb0d`。
- [ ] v26_rv 仍是 pilot-only/offline，不更新 readiness 指针，不进入训练/runtime/VLM/融合/晋级；继续三场全量亚秒因果、插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF。

### 2026-08-13 v27_rv Randolph/VTV source-disjoint 人工标注增量

- [x] 在已有 review-plan 中心外按 15 秒源内排除半径，从 Randolph/VTV 各 early/middle/late 抽取 12 个 label-hidden 窗口，共 24 windows/504 张 hash-bound 原片帧；held-out batch SHA `e576ec63d473544542774e1bd8f417840abe5abbc8c95cecc91c166de2da6b79`，review plan SHA `28546ef24aef9c80b9d4f938fdf5bf92ed2c20a989d5ee441522968547c0c239`。
- [x] 离线逐帧复核封存 22 `not_a_shot/not_applicable`、2 `uncertain/unknown`，没有新增可确认 shot；sealed SHA `803bf293f2ced37aea4ab31101327fdd0aee3fedaba856f6f4a30808de2854df`，pilot SHA `ff2a7022600d23b6657dac5c8c583534a5cd7dae6db64610e04351bd0b344785`，retention SHA `3f9e001ca3a07ef4fc682295db77a406f69e32c21afef16a7aec24591fa0e693`。
- [x] 批次保持 `pilot_only`、`exhaustive=false`、`training_consumable=false`、`runtime_consumable=false`、`hard_negative_gate_eligible=false`，不接入训练、runtime、VLM 默认答案、融合、晋级或 blind inference；raw frames 与正式工件保留。
- [x] focused regression 13 项通过；复核后按精确白名单删除 24 张可重建 contact sheet（`11,708,095` bytes），审计 `analysis_outputs/public_research/agu_v27_contact_sheet_cleanup_2026-08-13.json`。
- [x] 回归后按精确 allowlist 删除项目级 `.pytest_cache`（4 files / `1,853` bytes），清理审计 `analysis_outputs/public_research/agu_v27_regression_cache_cleanup_2026-08-13.json`；`.venv`、raw frames 与正式工件保留。
- [ ] v27_rv 仍不能替代三场全量因果真值；继续插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF，readiness 继续 `not_ready`、blind inference 继续暂停。

### 2026-08-13 v28_rv Randolph/VTV source-disjoint 人工标注增量

- [x] 在前批全部 review-plan 中心外按 15 秒源内排除半径，从 Randolph/VTV 各 early/middle/late 抽取 12 个 label-hidden 窗口，共 24 windows/504 张 hash-bound 原片帧；held-out batch SHA `495c7b1e06c2178d4a37869a364f364a7af198ba3111e8b94f525b54f74a8693`，review plan SHA `b3693e4d0e81cc9dadfb269270c012d3deae1d363168cae083cc66a45a25c1e5`。
- [x] 离线逐帧复核封存 20 `not_a_shot/not_applicable`、4 `uncertain/unknown`，没有确认 shot；sealed SHA `52a3147b0c963f05d446dc43b8da01dc75c6ec7694517d537fb9800204d35415`，pilot SHA `cd9f636d800c6d8334a60a70c0ae43383142589f7c290f9a002649943f8f48c9`，retention SHA `fbcf446de61d320cc3b4ed719a6861a96fde2b9b3864201ec9d850f225d30d35`。
- [x] v28_rv 保持 `pilot_only`、`exhaustive=false`、`training_consumable=false`、`runtime_consumable=false`、`hard_negative_gate_eligible=false`，不进入训练/runtime/VLM 默认答案/融合/晋级/blind inference；raw frames 与正式工件保留。
- [x] focused regression 共 `13 passed`；删除 24 张可重建 contact sheet（`11,613,466` bytes），审计 `analysis_outputs/public_research/agu_v28_contact_sheet_cleanup_2026-08-13.json`。
- [x] 回归后按精确 allowlist 删除项目级 `.pytest_cache`（4 files / `1,853` bytes），审计 `analysis_outputs/public_research/agu_v28_regression_cache_cleanup_2026-08-13.json`；`.venv` 未触碰。
- [ ] v28_rv 仍不构成三场全量因果真值；继续插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF，readiness `not_ready`、blind inference 继续暂停。

### 2026-08-13 v29_rv Randolph/VTV source-disjoint 人工标注增量

- [x] 在前批全部中心外按 15 秒源内排除半径，从 Randolph/VTV 各按 early/middle/late 配额抽取 12 个 label-hidden 窗口，共 24 windows/504 张 hash-bound 原片帧；held-out batch SHA `63ab792a00a3a90faea9e0265b750fb90a8f3f9a8ef2914b6e3371e38740038c`，review plan SHA `7734946f98ded098b53a80dd8da83b0a8040ed8b4cd64cca6a602e860473ae21`。
- [x] 离线逐帧复核封存 19 `not_a_shot/not_applicable` 与 5 `uncertain/unknown`，没有确认 shot；sealed SHA `e9f8db85b8846a92b77982af7ff532cff7009b651ca61ba8169e5bee11b67e90`，pilot manifest SHA `2ab607117fb6cf521210ac30a1e5cbfcc37c583d00c1c36fff0dfaedd785274f`，retention SHA `bf8e0f344ae8f0fba724c1eabf265f719ac895364d3a9c817dd198daad70be73`。
- [x] v29_rv 保持 `pilot_only`、`exhaustive=false`、`training_consumable=false`、`runtime_consumable=false`、`hard_negative_gate_eligible=false`，不进入训练/runtime/VLM 默认答案/融合/晋级/blind inference；raw frames 与正式工件保留。
- [x] focused regression 共 `13 passed`；删除 24 张可重建 contact sheet（`11,854,476` bytes），审计 `analysis_outputs/public_research/agu_v29_contact_sheet_cleanup_2026-08-13.json`；删除 9 个项目级回归缓存/bytecode 文件（`39,954` bytes），审计 `analysis_outputs/public_research/agu_v29_regression_cache_cleanup_2026-08-13.json`；`.venv` 未触碰。
- [ ] v29_rv 仍不构成三场全量因果真值；继续插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF；readiness `not_ready`、blind inference 继续暂停。

- [x] 聚焦回归闭合后删除 16 个项目级可重建 `.pyc`/pytest/Ruff cache 文件（`109,437` bytes），清理审计 `agu_2026-08-13_final_cache_cleanup.json`；`.venv`、原片、正式帧/工件、权重和 runtime 资产保留。
- [x] 追加核验 BasketEvent、Qlean、Play by Play、MUVY、FineAction：分别缺可验证公开许可/载荷、缺 causal outcome labels、仅合成坐标/空间检测或为通用 TAL；五者均未通过 AGU 五项 source gate，新增下载 `0` bytes。候选审计 v3 已更新（audit `f165b78dac8f49ad6948ce42e691d3359d93b788fd15ce9aed1b97dea5b44aee`）。
### 2026-08-13：拒绝辅助数据像素载荷清理

- [x] 依据用户授权和 allowlist，删除已审计不满足 AGU 因果/跨源门禁、且不被 runtime、训练真值或盲推理引用的像素载荷：Infactory `data/`、`yolo/images/`、`yolo/labels/`，APIDIS `raw/`，UVY `UVY/` 与 `yolo_aux_v1/`。
- [x] 两阶段合计删除 `19,212` files / `1,667,793,510` bytes；合并审计为 `analysis_outputs/public_research/agu_rejected_auxiliary_payload_cleanup_summary_2026-08-13.json`（summary SHA `97e5e1f8ce3281156ccb1ec8a7c2c0757c75006d0c9c05b580cf2182023d48dd`），第二阶段执行审计为 `agu_rejected_auxiliary_payload_cleanup_2026-08-13.json`（audit SHA `3c95d5596af62c00439694e123142479b115dc8002092d85b7db35503a9af99d`）。
- [x] 保留 manifests、许可证/README、UVY central directory、正式审计/代码/测试、Wikimedia 原片、native 与 MLX-4bit EBQwen、canonical `.venv`；载荷清理后即时可用空间约 `8.7 GiB`，回归缓存清理后的最终核验约 `7.8 GiB`。旧结果 JSON 中的已删除像素路径仅作 provenance，不代表本地载荷仍在。
- [x] 53 项 focused regression 后重新生成的项目 `.pyc`、`.pytest_cache`、`.ruff_cache` 已清理；`.venv` 受保护。收尾审计为 `analysis_outputs/public_research/agu_post_cleanup_regression_cache_cleanup_2026-08-13.json`。
- [ ] 清理不构成训练完成或 readiness 晋级；继续 rights-cleared 连续 5×5 亚秒因果标注、插播边界、穷举 non-shot hard negatives 与 game-held OOF，保持 `not_ready`/`blind_inference=paused`。

### 2026-08-13 v30_rv Randolph/VTV source-disjoint 人工标注增量

- [x] 在前批全部中心外按 15 秒源内排除半径，从 Randolph/VTV 各 early/middle/late 配额抽取 12 个 label-hidden 窗口，共 24 windows/504 张 hash-bound 原片帧；held-out batch artifact SHA `3cc3c05c1e9eb00e3f16ec7b944e4485f2b709fbc83c982c6b4f142a704cc471`，review-plan SHA `fd89fc7faff48bd0bb2cc8283083b04d3c74834259f8988620259d792830b8af`。
- [x] 离线逐帧原片复核封存 15 `not_a_shot/not_applicable`、9 `uncertain/unknown`，没有确认 shot；sealed SHA `84de8da98d54700f7f7bf62e8e6af2f543d780623d2441db7151e64be39e79ee`，pilot manifest SHA `7d65343c39f6647fb4ecb977674afe5c7e6904b18c141b42d6dde573ed39386b`，retention SHA `3aa99dc0e020895d6663530dd95bdf4774e6b4142fb119e081c2b487badc860c`。
- [x] v30_rv 为 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；不进入 AGU 训练/runtime/VLM 默认答案/融合/blind inference；504 张 raw frames 与正式工件保留。
- [x] canonical `.venv` focused regression 共 `13 passed`；删除 24 张可重建 contact sheet（`11,640,731` bytes）和 10 个项目缓存/bytecode 文件（`61,430` bytes），清理审计分别为 `analysis_outputs/public_research/agu_v30_contact_sheet_cleanup_2026-08-13.json` 与 `analysis_outputs/public_research/agu_v30_regression_cache_cleanup_2026-08-13.json`；`.venv` 未触碰。
- [ ] v30_rv 仍不构成三场全量因果真值；外部候选复筛新增下载 `0` bytes，继续插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF；readiness `not_ready`、blind inference 继续暂停。

### 2026-08-14 v31_rv Randolph/VTV source-disjoint 人工标注增量

- [x] 在前批全部中心外按 15 秒源内排除半径，从 Randolph/VTV 各 early/middle/late 抽取 12 个 label-hidden 窗口，共 24 windows/504 张 hash-bound 原片帧；held-out batch SHA `12ca8d6c93114caa8dc2e97109a48ef7e551fb4c3a22e3a7dcf45f0852154c54`，review plan SHA `c9b3eb5d5ac3f7f54b9da380db06dc0fd104046c461a1cd6a1f459cf463a529c`。
- [x] 原片逐帧复核封存 14 `not_a_shot/not_applicable`、9 `uncertain/unknown`、1 `shot/unknown`（VTV 872s；释放—篮筐接触链可见但无法判定命中/未中）；sealed SHA `9b5da3e7e4c1f20f96d129819667b3c6ce7c708f3c55a3092ccd36abf5b68a29`，pilot SHA `da6e65211b5d4e4041094bf765c089d325bc60a46504f07a46e2dbb547f2fed0`，retention SHA `f81875637b08ec8c4c8c5254acb944a9a496cd90fb2cfa730b88db0282db0101`。
- [x] v31_rv 保持 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；不进入 VLM 默认答案、融合或 blind inference。canonical `.venv` causal-review regression `13 passed`。
- [x] 删除已查看 24 张 contact sheet（`11,212,920` bytes）和 10 个项目缓存/bytecode 文件（`61,430` bytes），审计分别为 `agu_v31_contact_sheet_cleanup_2026-08-14.json`、`agu_v31_regression_cache_cleanup_2026-08-14.json`；`.venv` 未触碰。
- [ ] v31_rv 仍不构成三场全量因果真值；外部候选复筛新增下载 `0` bytes，继续插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF；readiness `not_ready`、blind inference 继续暂停。

### 2026-08-14 v22 smoke 派生物清理

- [x] 完整 v22 批次已保留正式 raw frames/manifest/review/retention 后，按精确 allowlist 删除被替代的三窗口 smoke raw frames（63 files，`21,012,931` bytes）；smoke metadata、review decisions、plan、sealed、pilot/retention manifests 保留，审计 `analysis_outputs/public_research/agu_v22_smoke_derivative_cleanup_2026-08-14.json`。
- [x] 清理未触碰 `.venv`、三场原片、正式 v22 raw frames、EBQwen 或 runtime/readiness 资产；其它历史 raw evidence 仍因正式哈希绑定 provenance 保留。

### 2026-08-14 v32_rv Randolph/VTV source-disjoint 人工标注增量

- [x] 在前批全部中心外按 15 秒源内排除半径，从 Randolph/VTV 各 early/middle/late 抽取 12 个 label-hidden 窗口，共 24 windows/504 张 hash-bound 原片帧；held-out batch artifact SHA `2205286c5364a8507025dce334111b2fcf9c6bdea6569b12095fdc84efdd8c6a`，review plan SHA `daad00c9a4275394372ea9eff91d502adba1e5f15fc98c7d91487df28dd16b2d`，raw-frame manifest SHA `efc32cc4712f501702d2b013b20dc24b14e4a8f105cd060272198dd9c9bc546a`。
- [x] 原片逐帧复核封存 12 `not_a_shot/not_applicable`、11 `uncertain/unknown`、1 `shot/unknown`（VTV 4842s；罚球出手至篮筐接触可见但不能判定命中/未中）；sealed SHA `eb29e90a03b826d8ae45ff3de2c3a637a17db39d24e66736f7d79b6050807165`，pilot SHA `f7a73fc3800fc6f3681daeb780b43c106cc057afb17055cc97b07692c031f413`，retention SHA `7ad3f19bbd5a11d114d930eacb14eddca78a84f37100d5c4ba947f0ac9a6df0c`。
- [x] v32_rv 保持 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；不进入 AGU 训练/runtime/VLM 默认答案/融合/blind inference。canonical `.venv` causal-review regression `13 passed`。
- [x] 复核后按精确 allowlist 删除 24 张可重建 contact sheet（`12,182,043` bytes），审计 `analysis_outputs/public_research/agu_v32_contact_sheet_cleanup_2026-08-14.json`；13 项回归后再删除 4 个项目级 pytest cache 文件（`1,853` bytes），审计 `analysis_outputs/public_research/agu_v32_rv_regression_cache_cleanup_2026-08-14.json`；504 张 raw frames、review spec/plan/decisions/sealed、pilot/retention manifests 与原片保留，`.venv` 未触碰。
- [x] 当前基座与独立 VLM 诊断门禁重查仍不通过：基座 31 个确定窗口 TP/FP/FN/TN=`0/0/2/29`、P/R/F1=`0/0/0`、rank AUC=`0.465517`；VLM transfer post-inference 32 窗口 P/R=`0.625/0.625`、`promotion_eligible=false`。v32 不作为训练真值，未启动无新增真值支撑的重训。
- [ ] v32_rv 仍不构成三场全量因果真值；继续插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF；readiness `not_ready`、blind inference 继续暂停。

### 2026-08-14 v33_rv Randolph/VTV source-disjoint 人工标注增量与在线源复筛

- [x] 在全部既有历史锚点外按 15 秒源内排除半径抽取 Randolph/VTV 各 12 个 label-hidden 窗口，共 24 windows/504 张 hash-bound 原片帧；held-out SHA `b578867cbbc2bd93ceb1de8e108a0919e026dfd5facb5dbbe7ca1df859aea356`，review plan SHA `228868b5acc22c23e98ecf0306c9ffa79744b3ee57227efedbbb35a2e14ab596`，raw-frame manifest SHA `6793727d672c5743d8b7c3407ff2c154b1efcf03913547a550b7cb746747f4c5`。
- [x] 原片逐帧复核封存 10 `not_a_shot/not_applicable`、14 `uncertain/unknown`、0 `shot`；sealed SHA `2382ad54801ad75212ba94afdb1a314ce121322f1582d7e45eea127359a4a586`，pilot SHA `ca321e3629edf1ee6fbec415c3502de7cf3a306d45dd9a51962689297e46764f`，retention SHA `b787f84211fab104479ffa236b3ab2fe4f0456b789fdeffa1c5705aae66ff2b2`。
- [x] v33_rv 仍为 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；不接入 VLM 默认答案、融合或 blind inference。canonical `.venv` causal-review regression `13 passed`。
- [x] 删除 24 张已复核 contact sheet（`12,523,486` bytes），审计 `analysis_outputs/public_research/agu_v33_contact_sheet_cleanup_2026-08-14.json`；测试后空 `.pytest_cache` 目录已移除，`.venv` 与正式工件保留。
- [x] 在线源复筛审计 `analysis_outputs/public_research/agu_online_source_sweep_2026-08-14.json`（SHA `8a35ca5a5724f94d8ac979e7746731be6829a97e60d56cdd1e10910c44fabf8d`）：BARD 仅复用已有本地辅助载荷，NBA Games、NSVA、NBA PBP、BASKET、NBA Streaming 未通过组合门禁，新增下载 `0` bytes。
- [ ] v33_rv 仍不构成三场全量因果真值且没有新增可训练 shot 正例；继续全量亚秒球—手—篮筐—结果、插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF；readiness `not_ready`、blind inference 继续暂停。

### 2026-08-14 v34_rv residual source-disjoint 人工标注增量

- [x] 在 1,456 个历史源内排除锚点外抽取 Randolph 2 个、VTV 2 个窗口，共 4 windows/84 张 hash-bound 原片帧；held-out SHA `2a12bb73ae9ed072d43684b3612a55992fb63b1664f1d7daeb24ef05eb2c2d25`，review plan SHA `b3dbd70a6f776b5f9da1c178ab33e6672d72351b891a6c1f5cefb8c8c54fced4`，raw-frame manifest SHA `49e051773a91d84773ea78f7ef1d081b4ab65138f294715cec6c2dc5220b58b4`。
- [x] 原片逐帧复核封存 2 `not_a_shot/not_applicable`、2 `uncertain/unknown`、0 `shot`；sealed SHA `736a8f6748a1ab776af18d3a83e38bcbf5f4746e5d1ca313d03a97acc55885fa`，pilot SHA `58fb19a66a0b9930d621ca5da3f9ab6db2d8d6d624f868d5631b2ff345a991ed`，retention SHA `1014c9cd60a252e898e95ddf1f53a0a9d29369b05f772e2bb8a04cf786ff9682`。
- [x] v34_rv 保持 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；focused causal-review regression `13 passed`，不接入 VLM 默认答案、融合或 blind inference。
- [x] 删除 4 张已复核 contact sheet（`2,111,920` bytes），审计 `analysis_outputs/public_research/agu_v34_contact_sheet_cleanup_2026-08-14.json`；raw frames、正式工件、原片与 `.venv` 保留，项目侧缓存为 0。
- [ ] v34_rv 没有新增可训练 shot 正例；Hazen 在 15 秒排除半径下耗尽，继续三场全量亚秒因果标签、插播边界、穷举 hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF；readiness `not_ready`、blind inference 继续暂停。

### 2026-08-14 v35_rv VTV residual source-disjoint 人工标注增量

- [x] 在所有已封存 review-plan/batch 锚点之外按 15 秒源内排除协议从 VTV 抽取 8 个 label-hidden 窗口；anchor manifest SHA `626bd0dd9be94f5e33543c4dc8cefd7a4eb3f671507eada89a55c48e0afbf50f`，held-out batch SHA `9369ffce6947d98e0c30b712b16f6fa3deae8b7499e55d27d9e19ed1587a2545`。
- [x] 物化并完成人工复核 168 张 hash-bound 原片帧，封存 7 `not_a_shot/not_applicable`、1 `uncertain/unknown`、0 `shot`；sealed SHA `b2dc36acd9eb2e48cedd19f29c3ecb1206bfe17b77315dc1fab337784d1133f8`，pilot SHA `b920d477ad2a7f444b4b226832b031eed353df8dde1868a072d77e4daa8f8fa9`，retention SHA `e0ba7c3d81dedc42f2cf853e2df18a3938cdb4e303bf9aca14fb3ec6e7e5e406`。
- [x] v35_rv 保持 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；focused causal-review regression `13 passed`，不接入 AGU 训练、runtime、VLM 默认答案、融合或 blind inference。
- [x] 删除 8 张已查看可重建 contact sheet（`3,507,319` bytes），审计 `analysis_outputs/public_research/agu_v35_contact_sheet_cleanup_2026-08-14.json`（audit SHA `fddadd3dcc84fc3d5aa1d6d5d9e6c16ea7b0eb392b7c7d31978b22f226f40627`）；回归后删除 20 个项目级 bytecode/cache 文件（`168,942` bytes），审计 `analysis_outputs/public_research/agu_v35_regression_cache_cleanup_2026-08-14.json`（audit SHA `58c39123fe2afa05784303e9b6c39819c59e4a831ec1ab63130b765ec00e03b5`）；raw frames、正式 review/manifest/retention、原片与 `.venv` 保留。
- [ ] v35_rv 没有新增可训练 shot 正例；继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF；readiness `not_ready`、blind inference 继续暂停。

### 2026-08-14 v36_rv VTV residual source-disjoint 人工标注增量

- [x] 在 v35_rv 全部源内锚点外按 15 秒排除半径，从 VTV early/middle/late 各抽取 8 个 label-hidden 窗口，共 24 windows/504 张 ±2 秒 hash-bound 原片帧；held-out batch SHA `589908cf67f0ef6d44c8b77e206f5703bf43df0cff8b131c7a1d8dfc82680724`，anchor manifest SHA `4b302c0ce70cafd191911b8ac7261f4b1a03fdffb6d8ba1f4bc288d4470d0be9`，review-plan SHA `bb6d325f48c4eaf4832678788664fbabf73c7f5d793cc73aaa8f1c259d3a162d`。
- [x] 原片逐帧复核并 fail-closed 封存 23 `not_a_shot/not_applicable`、1 `uncertain/unknown`、0 `shot`；sealed SHA `f9602163a95585263a362b16c03b5fceb9c6b1bb33976155979733741a2e1a09`，pilot SHA `ef3765155ecf6f055da70ae61457feffb6eeacbe24e5c9038ba75d75d9d597ee`，retention SHA `04dfc1d5ff499405d78bf1c18f1419dd16bde83997ac1ff984064fc76d5d5215`，raw-frame manifest SHA `c5719797e739a88a43020a101cff32cb5b01bfe08f99fa6bae19804731e9ecba`。
- [x] v36_rv 保持 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；focused causal-review regression `13 passed`，不接入 AGU 训练、runtime、VLM 默认答案、融合或 blind inference。覆盖审计显示 Hazen `0`、Randolph `0`、VTV 仍有 `204` 个未复核队列窗口，记录于 `analysis_outputs/public_research/agu_v36_rv_annotation_coverage_2026-08-14.json`（SHA `3e506cb1e2c06d7f936665146a34a17b314190908b628e80f74fa1678cc436a5`）。
- [x] 复核后按精确 allowlist 删除 24 张 contact sheet 与 5 个 montage 派生文件，共 `29` 文件/`15,140,690` bytes；审计 `analysis_outputs/public_research/agu_v36_contact_sheet_cleanup_2026-08-14.json`（SHA `78388e5cbaa9acebc90083c6a38ee339e4452b49b61552c966c205c4bffd7db0`）；回归后另删除 20 个项目级 cache/bytecode 文件（`171,461` bytes），审计 `analysis_outputs/public_research/agu_v36_regression_cache_cleanup_2026-08-14.json`；raw frames、正式 review/manifest/retention、原片与 `.venv` 保留。
- [ ] v36_rv 没有新增可训练 shot 正例；继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF；readiness `not_ready`、blind inference 继续暂停。

### 2026-08-14 v37_rv VTV residual source-disjoint 人工标注增量

- [x] 在 v36_rv 全部源内锚点外按 15 秒排除半径，从 VTV early/middle/late 各抽取 8 个 label-hidden 窗口，共 24 windows/504 张 ±2 秒 hash-bound 原片帧；held-out batch SHA `5237edd2383ce7bafc1f5fd6ed3ea0083b0636720fe71702ad18308a4379bd0d`，anchor SHA `c2d708a9441b1e10af2e23f3d697a6c44a8eeca6eedb80720a2d4fa5192eeee3`，review-plan SHA `82866573ea39b721a60ad2c79f0a53646398006098bc8902e3a616c25c3f225f`。
- [x] 原片逐帧人工复核并 fail-closed 封存 22 `not_a_shot/not_applicable`、2 `uncertain/unknown`、0 `shot`（VTV 5622s/5852s 的 release boundary 或镜头切换后完整因果链不可见）；sealed SHA `6251db4a641c99d9ce64e52efb82ee63f376ac434774f5dd275e7e07ac155d37`，pilot SHA `6b3b814c2986fabfa80121f733b50e42523f0206f7e7bff75d39cf59daed71dc`，retention SHA `22aa075f4759d2ce72f2319f2669da84cff2eaa01ec3a3bee53692fea73ff3de`，raw-frame SHA `0a809de26bdd641087c98bf5e25374a7d1359a307db748ec47e7384f43e3a210`。
- [x] v37_rv 保持 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；focused causal-review regression `13 passed`，不接入 AGU 训练、runtime、VLM 默认答案、融合或 blind inference。覆盖审计显示 Hazen/Randolph 余量 0、VTV 余量 `157`，记录于 `analysis_outputs/public_research/agu_v37_rv_annotation_coverage_2026-08-14.json`（SHA `70504ae97479973a91d4599f45aa1e169598f7f88e1faa407644f3dc0bf3bfdb`）。
- [x] 复核后按精确 allowlist 删除 24 张 contact sheet 与 5 个 montage 派生文件，共 `29` 文件/`14,742,074` bytes；审计 `analysis_outputs/public_research/agu_v37_contact_sheet_cleanup_2026-08-14.json`（SHA `a7e827042c5c5bcb3a871c0ff992b2228e6dd5a010d11c402dc9232831db6406`）；回归后另删除 19 个项目级 cache/bytecode 文件（`162,651` bytes），审计 `analysis_outputs/public_research/agu_v37_regression_cache_cleanup_2026-08-14.json`（文件 SHA `1211069b3709be04d785f56cfb09d9db2208ebd2eb19eb411a8f56f2eae788b1`）。raw frames、正式 review/manifest/retention、原片与 `.venv` 保留。
- [ ] v37_rv 没有新增可训练 shot 正例；继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF；readiness `not_ready`、blind inference 继续暂停。

### 2026-08-14 v38_rv VTV residual source-disjoint 人工标注增量

- [x] v38_rv 在 v37_rv 全部源内锚点外按 15 秒排除半径，从 VTV early/middle/late 各抽取 8 个 label-hidden 窗口，共 24 windows/504 张 ±2 秒 hash-bound 原片帧；held-out batch SHA `3f139af3c655635d4454d6592a8969680b558ae7db2cade5db5ef718e824dcac`，anchor SHA `13c6eefeb6d50eaca318c1835610e86e6b87ac4b7f283a2b808050ee91d06f17`，review-plan SHA `8998d037d156b92a59178b2cc80b55d6e927dbfe21f92f6b5d45fd3807cc674c`。
- [x] 原片逐帧复核并 fail-closed 封存 23 `not_a_shot/not_applicable`、1 `shot/unknown`（VTV 9362s：持球—脱手—向篮筐运动—篮筐/篮网平面可见，但 made/missed 不可安全解析）；sealed SHA `e70a729df42bf052dfc628a7aeed6a08fa4fa62fd90da19ef7a1573c52a1e100`，pilot SHA `a3c15f0daa15639aaa16dcd1db56bc8da4ecc7a617a3dd68c6a2eefdbdd62783`，retention SHA `df5ba13a93a6bb9904b80c8c171f77b34607d4368618a10481e3a3895c9cd99e`，raw-frame SHA `78a01c24e8435af4adb4cc848133a02a54b4d7e19b14a2bb487ec23d5e322062`。
- [x] v38_rv 保持 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；focused causal-review regression `13 passed`，不接入 AGU 训练、runtime、VLM 默认答案、融合或 blind inference。覆盖审计显示 Hazen/Randolph 余量 0、VTV 余量 `115`（early 39 / middle 29 / late 47），记录于 `analysis_outputs/public_research/agu_v38_rv_annotation_coverage_2026-08-14.json`（SHA `ccb6b434fda37768b04eed3d6a281562f2ce73d5c9fc1ecbb4b3f84be7deb0de`）。
- [x] 复核后按精确 allowlist 删除 24 张 contact sheet 与 5 个 montage 派生文件，共 `29` 文件/`12,958,436` bytes；审计 `analysis_outputs/public_research/agu_v38_contact_sheet_cleanup_2026-08-14.json`（audit SHA `22b9b8dacad7a66f6641062ae3a8224b287470cfaaa63a4c3be3c1e180d77c76`）；回归后另删除 19 个项目级 cache/bytecode 文件（`162,651` bytes），审计 `analysis_outputs/public_research/agu_v38_regression_cache_cleanup_2026-08-14.json`（文件 SHA `53655e4e1da0d9a76ede69983dd4640f1f4ef6e1bcffa0b5f0782ae1a7559598`）。raw frames、正式 review/manifest/retention、原片与 `.venv` 保留。
- [ ] v38_rv 没有新增可训练 made/missed shot 正例；继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF；readiness `not_ready`、blind inference 继续暂停。

### 2026-08-15 v39_rv VTV residual source-disjoint 人工标注增量

- [x] v39_rv 在 v38_rv 全部源内锚点外按 15 秒排除半径，从 VTV early/middle/late 各抽取 8 个 label-hidden 窗口，共 24 windows/504 张 ±2 秒 hash-bound 原片帧；held-out batch SHA `00aff1cf36b78d2088a652e9ba2a6eec6e82964a47c257dabe4d9a36fd5c7f30`，anchor SHA `2051f01bd9fb2617735c8acb6549e87d97ce9a5f1d0e52a4c363f7cfcd331a70`，review-plan SHA `bb39767f72b81195ec616b9b29b30beffce7dcef5ec9a71131c8c6622c654c08`。
- [x] v39_rv 原片逐帧人工复核并 fail-closed 封存 24 `not_a_shot/not_applicable`、0 `uncertain/unknown`、0 `shot`；sealed SHA `55a8a603947ab5bca1d15fcb3181b508923934ec96fba28c2f23f5b4894aa89d`，pilot SHA `5cd734320330a2942d2a5206b8f96fa0bb7bcfd0fb5b8ee47f369ee5076789fc`，retention SHA `4fe24cdb3131c5ba917259bd98c0f4870e0f0f738f4cfc053b2c7d4b3f6ebabf`，raw-frame manifest SHA `f3ad06812a4adb9ff414eee3b52ae467c42c9d8086fa8f2e5642465f1a01ba9`。
- [x] v39_rv 保持 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；focused causal-review regression `13 passed`，不接入 AGU 训练、runtime、VLM 默认答案、融合或 blind inference。覆盖审计显示 Hazen/Randolph 余量 0、VTV 余量 `80`（early 25 / middle 20 / late 35），记录于 `analysis_outputs/public_research/agu_v39_rv_annotation_coverage_2026-08-15.json`（audit SHA `4906bc93492625de2d2ac88e2d95310ce741bb94abcb51017f54f96d6279ca36`）。
- [x] 复核后按精确 allowlist 删除 24 张 contact sheet 与 5 个 montage/manifest 派生文件，共 `29` 文件/`15,334,391` bytes；审计 `analysis_outputs/public_research/agu_v39_contact_sheet_cleanup_2026-08-15.json`（audit SHA `cb29db3ea5d11bb1bdb5d09a59ffb91fa57626dd658a244ad081079e8291d9bf`）；回归后另删除 19 个项目级 cache/bytecode 文件（`162,651` bytes），审计 `analysis_outputs/public_research/agu_v39_regression_cache_cleanup_2026-08-15.json`（文件 SHA `7b0d93fcde223e8d5ef5f82de517eda9d70a6cdff6937ce9616e451ca032a3ee`）。raw frames、正式 review/manifest/retention、原片与 `.venv` 保留。
- [ ] v39_rv 没有新增可训练 made/missed shot 正例；继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF；readiness `not_ready`、blind inference 继续暂停。

### 2026-08-15 v40_rv VTV residual source-disjoint 人工标注增量

- [x] v40_rv 在 v39_rv 全部源内锚点之外按 15 秒排除半径，从 VTV early/middle/late 各抽取 8 个 label-hidden 窗口，共 24 windows/504 张 ±2 秒 hash-bound 原片帧；batch SHA `9b149844…`，anchor SHA `c9d1646…`，review-plan SHA `19261ec…`。
- [x] 原片逐帧人工复核并 fail-closed 封存 22 `not_a_shot/not_applicable`、2 `uncertain/unknown`、0 `shot`；1332s 的近篮动作与 2362s 的篮筐接触镜头均缺少可安全封存的完整 release-to-rim-outcome 链。sealed `fdf50e0…`，pilot `3a5c2f9…`，retention `bbdfcddc…`，raw-frame `925772ab…`。
- [x] 批次保持 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；canonical `.venv` causal-review regression 为 `74 passed`（13 个聚焦文件）。覆盖审计显示 Hazen/Randolph 余量 0、VTV 余量 `47`（early 16 / middle 9 / late 22），审计 `agu_v40_rv_annotation_coverage_2026-08-15.json`。
- [x] 复核后删除 24 张 contact sheet 与 5 个 montage/manifest 派生文件（29 文件、`15,682,641` bytes），并删除 67 个项目级可重建 cache/bytecode 文件（`1,250,894` bytes）；两项清理均保留 raw frames、正式 review/manifest/retention、原片与 `.venv`，详见 `agu_v40_contact_sheet_cleanup_2026-08-15.json`、`agu_v40_regression_cache_cleanup_2026-08-15.json`。
- [ ] v40_rv 没有新增可训练 made/missed shot 正例；继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF；readiness `not_ready`，blind inference 继续暂停。

### 2026-08-15 v41_rv VTV residual source-disjoint 人工标注增量

- [x] v41_rv 在 v40_rv 全部源内锚点之外按 15 秒排除半径，一次性复核 VTV 剩余 47 个 label-hidden 窗口（early 16 / middle 9 / late 22），共 987 张 ±2 秒 hash-bound 原片帧；batch SHA `55c43c8…`，anchor `ea67bc2…`，review-plan `845991c…`，raw-frame manifest `8c61c23…`。
- [x] 原片逐帧人工复核并 fail-closed 封存 41 `not_a_shot/not_applicable`、6 `uncertain/unknown`、0 `shot`；6 个 uncertain 均为 replay/近篮动作但 release boundary 或完整 rim/outcome 链不可验证。sealed `836d8b3…`，pilot `4ff46cf…`，retention `87f5a50…`。
- [x] 批次保持 `pilot_only`、非穷举、不可训练/不可 runtime/不可晋级；canonical `.venv` causal-review regression 为 `74 passed`（13 个聚焦文件），不接入 AGU 训练、runtime、VLM 默认答案、融合或 blind inference。覆盖审计 `agu_v41_rv_annotation_coverage_2026-08-15.json` 显示 Hazen/Randolph/VTV 余量均为 0。
- [x] 复核后删除 47 张 contact sheet 与 8 个 montage/manifest 派生文件（56 文件、`25,321,919` bytes），并删除 67 个项目级可重建 cache/bytecode 文件（`1,250,894` bytes）；两项清理均保留 raw frames、正式 review/manifest/retention、原片与 `.venv`，详见 `agu_v41_contact_sheet_cleanup_2026-08-15.json`、`agu_v41_regression_cache_cleanup_2026-08-15.json`。
- [ ] v41_rv 没有新增可训练 made/missed shot 正例；继续三场全量亚秒球—手—篮筐—结果、插播边界、穷举 non-shot hard negatives、逐制作源 P/R ≥ 0.85 与 game-held OOF；readiness `not_ready`，blind inference 继续暂停。

## 2026-08-15 causal-closure offline annotation set

- 来源：现有三场完整比赛原片（Hazen、Randolph、VTV），不新增第三方载荷；选择输入仅来自 v3–v41 已封存复核工件，明确排除已被正式 v22 替代的 `v22_smoke`。
- selection：942 条历史观察按 `(source_video_sha256, anchor_frame)` 物理去重为 918 个窗口；18 个 collision groups、3 个 conflict groups；最终 182 个候选中冻结每源 8 个、共 24 个互不重叠的 8 秒窗口。prior 四秒判断只用于预声明分层，不作为新八秒标签，也不逐窗暴露给 reviewer。
- sampling：严格 `[center-4s, center+4s)`，8 FPS，固定 64 帧/窗口，共 1,536 张原始解码帧；raw-frame manifest internal SHA `cf645285d1a4e20beed8deb6e01f921cfc749191038484c37941aa47ddddc742`。
- annotation：labels-hidden 离线逐帧 causal review 得到 14 shot、8 not-a-shot、2 uncertain；5 made、8 missed、3 unknown、8 not-applicable；20 high-confidence、4 medium-confidence。sealed review internal SHA `37f85d6078a26dc5862f755de96228e18a315558afba187bdf2f51d4d6189447`。
- 使用边界：`runtime_consumable=false`、`codex_runtime_answer_used=false`；只能经后续显式 SHA-bound training manifest 进入离线训练/诊断。`unknown` 不得转成 made/missed；该小批量不代表连续全场穷举 truth，也不支撑 85% readiness。
- 在线复筛：NBA_Streaming v2 仍未公开官方 payload；BasketEvent 缺视频与明确许可证且为 event-centered clips；SVI-Bench 约 15.2 TB 且受人工/机构门禁；APIDIS、NSVA、NBA-Identity、WASB 只能作为辅助。当前仍无候选同时满足 rights、连续视频、穷举 negatives、subsecond hand/ball/rim/outcome 与可控复现成本。

## 2026-08-15 closure training derivative and representation diagnostics

- 显式 training export 验证完整 selection/plan/review/raw-frame/source 链，并由外部 expected SHA 冻结
  selection、plan、review、raw-frame manifest。输出 22 行（14+/8-），排除
  `closure-randolph-0002` 与 `closure-vtv-0003` 两个 uncertain；candidate geometry 不含 outcome、
  notes、reviewed release/rim 或 label-derived anchor。
- Training export internal SHA `1b0bcef2c67f0d0e1fa35b79042584c43d938f05e893e89a4a64266ab9d0e464`；
  manifest internal SHA `56cbcda7ab0b1586e5284d40d0010c02f4c26042ca95aa78187e3462e44f71e0`。
- 固定 MViT 与 Swin3D-T 均为 torchvision Kinetics-400 representation screen；不修改权重，不输出可部署
  checkpoint。MViT/Swin/拼接 pooled P/R 分别为 `.6667/.1429`、`.875/.5`、`.8889/.5714`；
  Randolph 在拼接后仍 `0/0`。全部工件是 prior-stratified、非 runtime、非 formal、不可晋级。

## 2026-08-15 online source rescreen

- [NBA_Streaming v2](https://arxiv.org/abs/2608.09200) 描述 152 场 2025–26 NBA 完整转播、
  307.5 小时与约 35K 个 PBP 对齐事件，并声明 CC BY-NC 4.0；但论文同时写明 code/data 将在论文接收后公开，
  截至本次复核没有官方 payload。因此它是最接近的未来候选，但当前不可下载、不可复现，也没有公开的
  ball-hand-rim 坐标真值或 exhaustive non-shot 声明。
- [MUVS](https://zenodo.org/records/20708683) 是 17 个事件、4h05m、5,039 个三秒片段、258 个
  overlap-period 视频的多视角手机数据，完整视频约 164.9 GB；Zenodo metadata 标记 CC BY 4.0，但包内
  README 的 License 仍为 `TO BE ADDED`。视频每段仅约 1–5 分钟，标注是机位选择/传感器，不含
  shot/release/outcome/PBP 或穷举负例，故没有下载 164.9 GB 载荷。
- BasketEvent 仍只有事件 JSON 与模型权重，没有视频、清晰数据许可证或连续全场负例；其他已知来源也
  至少缺 rights、连续性、因果真值、穷举 negatives 或可控规模之一。本轮在线数据下载为 0 bytes。

## 2026-08-16 HCTV–Harwood continuous-game causal review

- 来源：Wikimedia Commons `Boys Varsity Basketball v. Harwood - January 22, 2026.webm`，HCTV
  attribution；Commons per-file metadata 为 CC BY 4.0，并保留 license-review warning。原片为
  `1,376,342,882` bytes、`4,125.088` seconds，SHA-256
  `bf7f3135774027aec3c2827cfa282c95ac4f958dd2bccd93e42f370d4b1f8b81`。
- 角色：新的比赛 source，但不是新的制作域；Hazen、Randolph、Harwood 均归 HCTV，VTV 才是另一制作域。
- 选择：冻结 24 个 label/model/PBP-free 窗口，early/middle/late 各 8；每窗严格 8 秒、8 FPS、64
  个 half-open 帧，共 1,536 帧。selection artifact `5a7eaa1e…a80f`，plan artifact
  `87c9b92c…4911`，raw manifest `0757e339…e776`。
- 标注：20 `not_a_shot`、3 `shot`（均 `missed`）、1 `uncertain`；sealed review
  `5f2219a4…5cb6`。只有 23 条确定标签可在后续显式导出中使用，`uncertain` 必须排除。
- 使用边界：offline/development-only，`codex_runtime_answer_used=false`；不得作为 AGU runtime 答案、
  formal evaluation 或 85% promotion 证据。四比赛训练派生物由 TASK-0257 新 v2 契约生成，不修改既有三源 v1。
- 清理：严格验证后删除 byte-identical 的 v1 raw-frame 副本和 v1/v2 contact sheets，共 1,584 文件、
  `715,927,744` bytes；保留 v2 1,536-frame evidence、原片和全部 receipt/sealed artifacts。

## 2026-08-16 Harwood additive v2 training derivative

- 数据视图：冻结三源/22-row v1 作为不可变父边，加入 Harwood 23 条确定标签；共
  45 rows（17 positive / 28 negative），三个 uncertain 显式排除。v2 export internal/file
  SHA 为 `9dbbffc8…7b35` / `6fd76806…9aa`。
- 分组/清单：Hazen/Randolph/Harwood -> HCTV，VTV -> VTV；source-group internal/file
  `029cb1b4…74b0` / `d48f9907…1c05e`。四视频 training manifest internal/file
  `ddf1af3b…7aba` / `e8261c55…ac3b`，`benchmark_overlap=false`。
- 视频表征：MViT-V2-S 与 Swin3D-T 各 45 rows、768 维、16 帧，batch size 1；MViT
  internal/file `d6aa1d9b…b201f` / `01dbd913…9a51`，Swin `3dc7caac…a40b` /
  `1c849096…31b5`。两个 staging 在完整工件发布后已删除。
- 诊断：probe plan internal/file `de1444c7…0333` / `78cc7dc6…9c2`；nested probe
  internal/file `92d5d804…ae21` / `7d1c7636…d720`。pooled P/R/F1=`.777778/.823529/.8`；
  Harwood P/R 仅 `.4/.666667`。HCTV/VTV aggregate 是 descriptive only，不是 production-held gate。
- 用途边界：22 条父集为 prior-stratified，23 条 Harwood 虽为 geometry-only/label-hidden，但整体
  并非连续穷举全场真值。所有 runtime/formal/promotion/promoted=false，readiness
  `not_ready`，blind inference paused，不支持 85% 正式验收或 checkpoint 晋级。

## 2026-08-16 ortizeg basketball DEIM-M diagnostic

- 模型：`ortizeg/basketball-deim-m-640`，固定 revision
  `1b4f378bc1fa8d3a7980768d3b08486b6f3fe357`，Apache-2.0；ONNX 原始大小 `77,722,583`
  bytes、SHA-256 `29f575c8127e5eadde6da60cd66c3d0a5873adc0cbfd4e2af9ee35fde339fac7`。
- 上游训练数据：`basketball-player-detection-3`，model card 标为 CC BY 4.0；官方 test 的三场比赛也出现在
  train 的其他 clips，因此官方 AP 不能作为 AGU game-held 证据。
- AGU 评测：复用 SHA-bound LAL–BOS detector candidate review，108 条中 20 valid、86 false-positive、
  2 uncertain；它不是完整帧的 exhaustive truth，只能作一致的保守 transfer diagnostic。
- 三档固定阈值均失败：confidence `.10/.25/.50` 的 P/R/F1 为 `.0288/.55/.0547`、
  `.0671/.50/.1183`、`.0959/.35/.1505`。所有 runtime/formal/promotion 标志为 false。
- 清理：验签、guarded inference 和结果封存后删除 ONNX `77,722,583` bytes；保留
  `dataset/public_sources/open_models/basketball_deim_m_640/{README.md,source_manifest.json}` 与
  `analysis_outputs/public_research/basketball_deim_external_lal_bos_v1/`。

## 2026-08-17 DVIDS Armed Forces Basketball metadata audit

- 审计范围：DVIDS 2024 Armed Forces Basketball Championship 官方页面；本轮只读取 HTML/官方条款，
  下载媒体 `0` bytes，不创建 source manifest，不冻结 Module-B candidate universe。
- 三个精查条目：DOD `DOD_110604818` Army–Navy、`DOD_110604091` Navy–Air Force、
  `DOD_110608501` Army–USMC。页面时长均约 112–116 分钟、标为 Public Domain，并提供低于 1 GiB 的
  低分辨率近似下载选项；制作域可与 HCTV/VTV 区分。
- 权利边界：DVIDS 的政府作品规则与具体页面 Public Domain 标记是强证据，但官方条款仍保留第三方、
  publicity/privacy 与禁止暗示背书边界；账户下载、精确 bytes/SHA、实际连续性和逐 item 限制尚未闭合。
- 排除项：VTV/SPB 是现有源/同制作域；IA NBA mirrors 的许可由非权威私人镜像上传者声明；社区高中/
  青少年赛事缺少未成年人治理许可；Auburn 历史影片为多卷且条款不一致。
- 封存证据：`analysis_outputs/public_research/agu_dvids_armed_forces_basketball_metadata_audit_2026-08-17/audit.json`
  artifact SHA `076e69a291117609ad71ef3d45f8b04b56030da90f6a4bb28632de0efd3e38e2`。
  状态是 `strong_metadata_lead_only`，不是 accepted dataset；training/runtime/formal/promotion 均为 false。
- 下载端点增量：三场下载弹窗共提供 21 个 derivative file ID；页面只给近似大小。每场抽查 512p/256p
  两个端点，HEAD-only 共 6/6 返回 403，未返回 Content-Length/redirect，且媒体 body 0 bytes。
  官方 Asset API 可返回 `files[].src/size/bitrate`，但要求 API key；无授权 key 时不能冻结 exact payload receipt。
  增量 audit SHA `77f2b27ef6d5ff831143e26293be83eca3196ec09ade60c8d2295f5f8e058dd0`，状态为
  `download_receipt_gate_failed`，不改变该数据源的未选择边界。

## 2026-08-17 second VTV professional full-game metadata audit

- 候选：[Spartans Distrito Capital vs Cocodrilos de Caracas](https://commons.wikimedia.org/wiki/File:Superliga_Profesional_de_Baloncesto_-_Spartans_Distrito_Capital_vs_Cocodrilos_de_Caracas.webm)，
  2026-03-29、2:04:03、Venezolana de Televisión；成年职业 5v5。它与保留的 2026-04-26
  Spartans–Trotamundos 是不同实际比赛，但同属 VTV production family。
- Commons 原片元数据：2,077,440,504 bytes、SHA-1 `3703133d57666ff976b6c7b8be54403c2c855b5c`、
  1920×1080 AV1/Opus；页面为 `Public domain` / `PD Venezuela official`，没有 license-review warning。
- 官方 480p transcode 为 854×480 VP9/Opus、1,060,216 bps，按时长估算 986,412,774 bytes（0.919 GiB），
  暂低于 1.2 GiB cap。该估算不是 exact file receipt；本轮 HEAD 未获得 HTTP response，未下载媒体。
- 封存 metadata-only 工件 artifact SHA
  `eb229f90718f1072fb02c6f7dd712be74eac0fa656b0daeae44ea52cc1e9cb54`。状态为
  `high_priority_metadata_candidate_only`；exact bytes/SHA、连续性、内容去重和 privacy/publicity 未验证，
  Module B 未授权，training/runtime/formal/promotion 全部 false。
- HEAD-only 增量已把官方 480p endpoint 精确绑定到 `986,415,444` bytes、对象 SHA-1
  `bf16fe7634cd6593bea05f9104ad8a43b4c69faf`、ETag `718a0369346496ebce656df8575b690b`；官方对象大小
  为 0.918671 GiB，确定低于 1.2 GiB cap。additive receipt artifact SHA 为
  `e768aff432f27d44fd826887ddae759c595894485732cc3246f917bc6bdf48b0`。
- 增量仍读取 0 媒体 bytes；local SHA-256、连续性、内容 overlap 与 privacy/publicity 没有验证，故不构成
  source manifest、candidate selection 或 Module-B 授权。
- Commons long-form inventory 又受限枚举 7 个国家分类、66 files / 40 条 ≥20-minute rows。3 条 CBC 是
  纪录片、2 条 Slovenia 是同场分卷、33 条是 VTV family，2 条 Hidrocarburos uploader 缺独立 producer/
  public-sector-rights 证明；没有新增第三制作域 eligible source。inventory SHA 为
  `957ddae1960a4d11e79dcd8364ef94af30463ea62f06e9597ce3c63c454868b8`，下载仍为 0 bytes。

## 2026-08-17 TBF third-production-family full-game metadata audit

- 分类枚举没有覆盖所有文件。本轮改用六个冻结的 Commons 多语言全文查询，每个最多 100 条；243 个去重
  video hit 中有 55 条达到 20 分钟。这个 bounded search 新发现 TBF 制作的 2020-01-24
  Merkezefendi Belediyesi Denizli Basket–Samsunspor 成人职业整场。
- 官方 metadata：6,266.781 秒、1280×720、1,039,515,202-byte VP9/Opus 原片，SHA-1
  `c68ca3e4f848487f2202ec2eeca9d778a68a0077`；逐文件 CC BY 3.0，页面含
  `License reviewed by YouTubeReviewBot`，不含 `License review needed (video)`。
- 两个公共 DoH resolver 对 upload endpoint 的 A record 一致；两次 HEAD-only 响应把官方 480p derivative
  精确绑定为 `818,675,123` bytes（0.762451 GiB）、对象 SHA-1
  `35d79e3a94eb3045e904f0fbbdaeeb56a5af5cba`、ETag `92e76128d3b752293b6608b0c099e241`。
  没有媒体 GET/Range，媒体 response body 为 0 bytes。
- TBF 是不同于已保留 HCTV/VTV 的第三制作域，且比赛本身不与保留源相同；但 content/pixel overlap 尚未
  验证。审计工件 canonical SHA 为 `ce3d07e8d534116223ffe6cecd835509dc3121de7a4e7d02fa3e2990713a0fa0`，
  状态仅 `high_priority_third_production_family_metadata_candidate_only`。
- 当前不下载：local SHA-256、单场连续像素、privacy/publicity、Module-A mechanical pass 与 Module-B 单独
  授权都未闭合；审计时 3,598,568 KiB 可用空间也低于 payload + 3 GiB reserve。training/runtime/formal/
  promotion 全为 false，不改变 readiness 或 blind inference。
