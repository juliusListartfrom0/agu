# Development

Added `app/analysis/public_research_datasets.py` for the source catalog, sealed
NBA_Games plan, integrity verification, oEmbed metadata check, and optional
yt-dlp playback probe. Added `scripts/build_public_research_plan.py` as the
offline entry point and tests for licensing boundaries, roster coverage,
benchmark/enrollment separation, incomplete truth, playback failure, and plan
tampering.

The implementation uses project memory from the AGU autonomous two-pass VLM and
training-annotation wiki query. It preserves the existing BARD adapter and adds
no new mandatory dependency.

Added a resumable public-media supervisor with infinite yt-dlp network retries,
capped outer exponential backoff, atomic progress state, complete-file detection,
and post-download SHA sealing. Added a shared MOT adapter for TrackID3x3 and
TeamTrack so their annotations can be validated and cataloged without importing
either upstream runtime stack. The optional TrackID3x3 jersey-number subproject
remains isolated because it is CC BY-NC 3.0.

Added a pair-readiness handoff that requires both benchmark and enrollment media
to be fully sealed against the same plan hash. The handoff deliberately omits
truth paths and active rosters and keeps enrollment media separate from benchmark
inference media.

The retry supervisor completed all four selected videos and SHA-sealed them.
Enrollment-only YuNet/SFace extraction retained 65 LAL-BOS clusters and 74
ATL-CHI clusters. Exhaustive Codex face-only review produced sealed galleries
for 9 and 11 players respectively; the SFace consistency filter removed
inconsistent samples. Added `assess_face_gallery_coverage` and
`scripts/check_face_gallery_coverage.py` so an incomplete roster cannot be
mistaken for complete-game identity readiness.

Added a Wikimedia Commons adapter for the explicit missing-roster path. It
queries file-namespace image metadata, accepts only reusable CC BY/CC BY-SA/CC0
or public-domain candidates, records attribution fields, validates MIME bytes,
and leaves every downloaded face `identity_verified=false`. A live run found
license-recorded candidates for 10/10 missing LAL-BOS people (19 files) and
12/12 missing ATL-CHI people (23 files); these are pending face/bbox review and
do not yet change gallery coverage. Commons returned HTTP 429 until the client
identified the project URL and throttled requests, after which the retry passed.

The 720p LAL-BOS enrollment supplement then completed at 2,021,564,330 bytes
with SHA-256
`eabe0b31a1c17ea50b8481d046ba8d289b9b3a6ca26b3f9100e70a9926cf9e7f`.
YuNet/SFace detected 8,157 faces and retained 1,164 multi-sample clusters.
Added `scripts/select_face_enrollment_candidates.py` so a targeted review set
is hash-bound to the complete candidate manifest without claiming that Codex
reviewed all 1,164 clusters. Codex face-only review identified the ten missing
roster identities; P.J. Brown was recovered from a narrow independent-game
window with visible `BROWN 93` evidence, and Jordan Farmar was confirmed by the
number 5 uniform context. The existing exhaustive approval and merge gates
produced a 19-person gallery with 100% roster coverage. Runtime recognition
still uses YuNet/SFace; Codex supplied enrollment labels only.

2026-07-23 E-BARD detection increment:

- Registered the official E-BARD Detection source separately from BARD event
  semantics. The dataset is CC BY 4.0 and remains offline training input.
- Downloaded and integrity-tested `all.zip` at immutable Hugging Face revision
  `00563215490c9a9642797b1495ea178535b3f59c`. The 662,567,020-byte archive has
  SHA-256 `4b0a5ef8fd25565714e6b36a7020bc68b1cc2765afdc82a3b3d7a099e5c2ab81`.
- Extracted the official YOLO layout and sealed a local source manifest. It
  contains 1,440 train, 180 validation and 180 test images with 22,210 boxes.
  Dataset bytes remain ignored and are not redistributed with AGU.

2026-07-24 E-BARD crop-screening increment:

- Audited game IDs and found that the official archive partitions overlap by
  game, so they are not used as independent model-validation folds.
- Reused the sealed detector to produce a separate 3,675-row,
  `runtime_consumable=false` ball-versus-hard-negative crop manifest. The
  evaluation boundary is source game, and the reserved MEM-OKC acceptance
  truth remains unopened.
- A frozen MobileNetV3-small screen and a 12-seed Codex crop-only review remain
  offline research artifacts. The result was rejected from runtime because
  external basketball-seed recall is .778 and shot-event precision remains
  .286.

2026-07-25 Archive-backed game-pair increment:

- Added verified Archive media overrides for a 1998-06-14 CHI-UTA benchmark
  plus 1998-06-12 UTA-CHI and 1997-12-17 LAL-CHI enrollment games.
- Downloaded and SHA-sealed all three files (3,350,552,015 bytes total). Media
  probing confirms readable video and audio streams. No benchmark event answer
  was opened by the downloader or handoff builder.
- Generated a truth-free `agu.public-game-pair-handoff.v1` artifact with
  canonical hash
  `ab8a723cd8672a81fe353c5e7d2d51b124a2b32992df04d92212a3e63004f`.
  The handoff remains non-runtime and separates enrollment from benchmark
  inference media.
- Removed the unused 2,021,564,330-byte 720p LAL-BOS duplicate after confirming
  that current plans/runners reference the retained smaller source filename and
  the already-produced downstream gallery is independently sealed. The deleted
  source is redownloadable; the completed gallery evidence remains unchanged.

2026-07-27 TrackID3x3 Indoor and BasketHAR correction increment:

- Corrected BasketHAR governance after checking current upstream state. Its
  label counts follow the paper class order at roughly 1.25 times the table,
  but upstream removed the only video link and the former Drive URL now
  returns 404. The source is a wearable-signal reference with no visual import.
- Added a standard-library TrackID3x3 public-Drive manifest/downloader with
  exact folder resolution, drift checks, disk reserve, resumable partial files,
  declared-size validation and per-file SHA sealing.
- Selected only the 69,690,612-byte Indoor subset. Outdoor and Drone were not
  downloaded because their public videos total about 18.19 GB and 11.63 GB.
- Extended the shared MOT importer to accept TrackID3x3's official named
  `ground_truth/<subset>/MOT/*.txt` layout and bind each sequence to its
  `Indoor/raw` video.
- Sealed 42 videos, 7,534 frames and 45,204 six-player boxes. Offline Codex
  contact-sheet review found short fixed-camera mini-game possessions but no
  event labels or broadcast/free-throw/replay supervision, so the source is
  limited to bbox/track/pose research.
- Preserved the Indoor annotations, README and license, then deleted
  389,394,770 bytes of recoverable Git history, third-party stacks and
  Outdoor/Drone annotation content without corresponding local video.
