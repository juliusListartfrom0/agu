# TASK-0256 Development

## Acquisition and frozen selection

- Downloaded the canonical Wikimedia Commons WebM with resume support and
  retained the exact 1,376,342,882-byte payload. Local SHA-256 is
  `bf7f3135774027aec3c2827cfa282c95ac4f958dd2bccd93e42f370d4b1f8b81`.
- Recorded Commons CC BY 4.0 metadata, the original HCTV YouTube attribution,
  and the Commons license-review warning. This is a development-use source,
  not an unrestricted redistribution or identity-training claim.
- Froze 24 geometry-only windows before review: early/middle/late 8 each,
  non-overlapping 8-second half-open intervals, 8 FPS and 64 frames per window.

## Provenance and file safety

TDD hardening made the source-selection receipt part of the final plan contract
instead of trusting a mutable intermediate spec. The path from source manifest
through selection, plan, frame materialization, decisions, and sealed review is
externally receipt-bound. Reviewer-visible names are opaque. CLI writers reject
direct, symlink, hardlink, and case-fold aliases to protected inputs and use
same-directory temporary files with flush, fsync, and atomic replacement.

## Offline review

Codex reviewed only the frozen raw-frame evidence, never AGU runtime output.
The sealed result is 3 `shot/missed`, 20 `not_a_shot/not_applicable`, and 1
`uncertain/unknown`. Release/rim positions are ordinals into each 64-frame plan,
not inferred timestamps. A second independent blind review resolved the only
material disagreement (window 0015 rim position 27 rather than 30).

## Cleanup

After strict verification and fresh review, the byte-identical superseded v1
raw-frame copy and both rebuildable contact-sheet sets were removed. Audit
`agu_harwood_v2_superseded_visual_cleanup_2026-08-16.json` records 1,584 files
and 715,927,744 bytes deleted. The v2 1,536-frame evidence, original video,
receipts, decisions, sealed review, `.venv`, checkpoints, `open_models`, and
EBQwen assets remain protected. A root-owned 64 MiB `node-gyp` cache was left
untouched after normal user permissions rejected deletion; no escalation was
used.
