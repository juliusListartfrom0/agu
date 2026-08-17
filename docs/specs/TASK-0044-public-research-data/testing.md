# Testing

Completed gates:

- focused public-source and existing BARD adapter pytest suites;
- complete project pytest suite;
- Ruff on changed Python files;
- `git diff --check`;
- AGU harness verification.

Results: focused public-source/BARD tests passed (`6 passed` after adding the
local SHA-seal case); the complete suite passed (`326 passed`); Ruff and
`git diff --check` passed; harness verification passed. Real oEmbed and yt-dlp
probes passed for all four selected videos. The first full download was stopped
after the public host throttled it, so the generated real plan truthfully keeps
`two_game_media_gate_ready=false`.

Additional focused tests cover job deduplication without truth leakage, partial
download rejection, retry backoff, MOT geometry validation, media hashing,
training-only enforcement, and catalog tamper rejection. A new full-suite result
passes with `330 passed`; touched-file Ruff, `git diff --check`, and the harness
also pass. The background downloader recovered from a real HTTP read timeout and
continued the first video from its partial file.

The background retry completed all four videos (`download-state.json` status
`completed`) with byte sizes and SHA-256 seals. Truth-free pair handoffs were
generated for LAL-BOS and ATL-CHI. The two enrollment galleries pass internal
SFace consistency checks but intentionally fail the 95% roster gate at 9/19
(47.4%) and 11/23 (47.8%). The focused public-data and face-gallery suite passes
15 tests; touched-file Ruff passes. Full regression passes 333 tests with 15
existing warnings, the harness passes, and `git diff --check` passes. The local
service hook on port 8878 returned `ok`/`ready`; task
`6e2ccc451d4d4fc0b87b8629ffc2173f` completed at progress 100 with no error.
README and `docs/api.md` were checked; the additive offline coverage CLI does
not change their public service contract.

The Commons parser tests cover license rejection, non-runtime/unverified
provenance, manifest sealing and HTML/error-body rejection by MIME signature.
The live adapter retained 42 attributed image candidates across the two missing
rosters after a robot-policy-compliant retry. Identity approval and rebuilt
gallery coverage remain pending and no candidate is runtime-consumable. The
latest focused suite passes 18 tests; the full repository passes 336 tests with
15 existing warnings, touched Ruff passes, `git diff --check` passes, and the
harness passes.

The 720p LAL-BOS supplement produced a sealed 11-cluster targeted review subset
and a separate three-cluster P.J. Brown window. Both passed exhaustive approval;
the three reviewed face lists merged to 19 unique roster identities. Gallery
build succeeded and `check_face_gallery_coverage.py` returned `19/19`, coverage
`1.0`, `ready=true`, no missing or low-quality IDs, and gallery SHA-256
`18d2b60b343c1cae735184aa80fad492cab45186e44fbe7a06ad8300e44dcc18`.
The selection/approval/merge/gallery focused suite passes 13 tests. Full
regression, Ruff, harness and service-hook results will be refreshed after the
ongoing benchmark perception run.

After the benchmark was opened, the project `venv` full suite passed 346 tests.
The secondary `.venv` suite had three environment-only failures because its OpenCV
5 build omits `CascadeClassifier` and `cv2.legacy.MultiTracker_create`; the project
OpenCV 4.13 environment remains the acceptance runtime. Event-scoped identity
filtering and the raw-team/gallery-team namespace regression are covered by focused
tests. The first LAL-BOS raw-only pass produced 170 shot and 168 rebound review
candidates; event-scoped direct face evidence matched six tracklets across four
registered people, so named-player acceptance remains open rather than being filled
with inferred answers.

The 5 FPS player-only pass produced 164818 player detections. Event-scoped
gallery matches increased from 6 tracklets/4 people to 19 tracklets/8 people,
and events containing at least one registered candidate increased from 7 to 23.
The first 23 semantic VLM cache entries exposed an all-made degeneration. A new
observable-outcome gate reduced the nine-event probe to four evidence-complete
makes, four unknown outcomes, and one rejected candidate without consulting
benchmark truth.

A same-video 5 FPS BoT-SORT comparison ran to completion in 543 seconds. It was
more fragmented by generic tracking metrics (10975 IDs versus ByteTrack's 10698),
but camera-motion compensation yielded 23 direct gallery matches across nine
people and 29 events with registered candidates, compared with ByteTrack's
19/eight and 23 events. BoT-SORT is retained as an identity-evidence option, not
accepted as a complete-game identity solution.

The retrying TeamTrack seed download produced five independently stored basketball
MOT sequences. Catalog import sealed 49,403 annotations over 55 source tracks;
the declared dataset-stable track scope retained 999 SHA-bound crops across ten
anonymous identities. With `Q4_side_60-90` held out, ImageNet retrieval Top-1 was
44.0% and the domain checkpoint reached 86.0% on 200 validation crops. LAL–BOS
then dropped from 1662 to 1115 identity components while direct registered-face
coverage remained unchanged at 23 tracks/nine people/29 candidate events, proving
that body ReID did not leak names. The focused crop/import/training/checkpoint
suite passes locally.

Final verification passed 353 tests with 15 warnings, touched-path Ruff,
`git diff --check`, and `verify_harness.py --run-tests`. The local service hook
returned healthy/ready, accepted task `c8b6bb76b1c041b0837abf6f48e725ce`, and
completed it at progress 100 with no error in about seven seconds. README startup/request fields remain
consistent with the current API and `docs/api.md`; the new controls affect only
offline official-perception/identity/VLM scripts.

The E-BARD archive passed `unzip -tq`. Extracted YOLO image/label pairs are
1,440/1,440 train, 180/180 validation and 180/180 test. Label aggregation
matches the official card: basketball 1,496, hoop 1,565, player 15,296 and
referee 3,853. The source catalog regression asserts the CC BY 4.0 and offline
adapter boundary.

The repository-wide refresh passes 458 tests with 15 existing warnings,
touched-path Ruff, `git diff --check`, and the complete harness. Local service
task `f6ed00d58e0741d2af9c0ef323212c9e` completed at progress 100 with no error.

The 2026-07-24 detector-candidate audit verifies 1,800 unique source-image
hashes, 60 game groups and 3,675 labeled candidates. The manifest self-hash
recomputes exactly. Five-fold validation groups by game rather than E-BARD's
overlapping archive split. No dataset bytes, trained classifier or review label
were connected to AGU runtime.

The 2026-07-25 Archive pair download state records all three selected files as
completed and independently recomputed SHA-256 values match. `ffprobe` confirms
H.264/AAC for the benchmark and Theora/Vorbis for both enrollment files. The
truth-free handoff binds the original plan SHA, one benchmark, two enrollment
files and no answer assets. Public-research/native-VLM focused tests pass
29 tests; the full repository passes 527 tests with 15 existing warnings.

The 2026-07-27 TrackID3x3/BasketHAR correction increment was verified with:

- RED-to-GREEN regressions for public-Drive metadata parsing, exact path
  resolution, unsafe/ambiguous names, completed-manifest preservation,
  self-hash verification, bounded response writes, completed partial promotion,
  local-video hash mismatch rejection, official named-MOT import and missing
  matching-video rejection.
- `.venv/bin/python -m pytest tests/test_download_trackid3x3_subset.py
  tests/test_open_tracking_datasets.py tests/test_public_research_datasets.py
  -q`: 18 passed.
- `.venv/bin/python -m pytest -q`: 630 passed with 15 existing warnings.
- Touched-path Ruff, `git diff --check`, `.venv/bin/python -m pip check` and
  `.venv/bin/python scripts/verify_harness.py`: passed.
- The final TrackID3x3 source manifest self-hash recomputes to
  `cb944d2cb78455fcb3ce6edc2991c908f2cbfefcb4645239f33bc06289ab95fb`;
  42/42 files have declared-size and SHA seals, totaling 69,690,612 bytes.
- The imported MOT catalog self-verifies at
  `f96792dc18d8187d339f80f86cac6dc2a69d9a26c866eb12847fdefa5abe5dc2`,
  with 42 media-bound sequences, 7,534 frames and 45,204 boxes.
- Disk cleanup removed 389,394,770 fixed-revision-recoverable clone bytes.
  The retained TrackID3x3 source is 93 MiB, and free disk increased to about
  16 GiB. No model training or runtime/API behavior changed, so the local API
  smoke hook was not required.
