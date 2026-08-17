# TASK-0256 Solution

## Source acquisition

Use the canonical Wikimedia Commons original for
`Boys Varsity Basketball v. Harwood - January 22, 2026.webm`. Record the
Commons file page, original URL, attribution, per-file license, source YouTube
metadata, declared size/duration, local SHA-256, and `ffprobe` facts. Download to
a same-directory partial path, verify it, then atomically rename it.

## Label-hidden selection

Build a fixed 30-second candidate grid over all valid eight-second centers.
Partition candidates into chronological thirds. Within each third, rank by
SHA-256 over the frozen namespace, source-video SHA, and center frame; take the
first eight. Because the grid spacing is greater than the eight-second window,
selected windows cannot overlap. Reviewer IDs are neutral ordinals.

The selection artifact is training/development-only and contains no labels,
PBP, probabilities, outcome fields, or prior review answers. Its verifier uses
exact field schemas, recomputes the candidate/ranking result, and rejects any
re-sealed drift.

## Review and training boundary

Reuse the existing strict 8-second/8-FPS review plan, sequential materializer,
and sealed-review verifier. Only `shot` and `not_a_shot` may later cross a new
SHA-bound training-export boundary; `uncertain` remains excluded. The fourth
game is evaluated separately in nested game-held diagnostics, but all results
remain `formal_evaluation_eligible=false` and `promotion_eligible=false`.

The completed v2 review contains 24 rows: 3 `shot/missed`, 20
`not_a_shot/not_applicable`, and 1 `uncertain/unknown`. The uncertain row is not
a negative example. Training export and four-game screening are deliberately
split into TASK-0257 so this task can archive the frozen annotation evidence
without changing the existing three-game v1 export contract.

## Frozen result

- Source video SHA-256:
  `bf7f3135774027aec3c2827cfa282c95ac4f958dd2bccd93e42f370d4b1f8b81`.
- Selection artifact SHA-256:
  `5a7eaa1ecefa295c5d97632e05e7aea2187eafac0082ba6ea6341fc6b372a80f`.
- Review-plan artifact SHA-256:
  `87c9b92c548000527f253082c6190ec82022934d9c3db244be32e50505244911`.
- Raw-frame manifest artifact SHA-256:
  `0757e339ffd5acb97d5b8f42790cfced2de825e44deea20095abd7bedfeae776`.
- Sealed-review artifact SHA-256:
  `5f2219a4cdcc7e8d4e40dde060c8ba82b5070eb6db471c1ce2855cea9d3e5cb6`.

The v2 chain keeps reviewer-visible identifiers opaque and carries external
selection and plan receipts through to the sealed plan. `ffprobe` counted
247,504 packets while OpenCV/the manifest report a nominal 247,505 frames. The
strict verifier permits this one-frame metadata difference; the latest sampled
frame is 237,832, safely before either endpoint.

After the seal and fresh review, byte-identical v1 raw frames plus v1/v2 contact
sheets were removed under a recorded allowlist. The cleanup deleted 1,584 files
and 715,927,744 bytes while retaining the v2 1,536-frame evidence set, video,
receipts, decisions, sealed review, `.venv`, weights, and `open_models`.

## Open-source capability assessment

Media acquisition, decoding, logistic screening, and verification reuse
Wikimedia Commons, ffmpeg/OpenCV, scikit-learn, and existing AGU contracts.
AGU only adds deterministic selection and provenance glue; it does not create a
new decoder, model framework, or annotation UI.
