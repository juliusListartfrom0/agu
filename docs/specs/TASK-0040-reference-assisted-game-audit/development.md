# Development

Complete.

Durable outputs:

- normalized detail-event ledger and player aggregation module;
- CLI audit package generator;
- reference-to-raw fingerprint manifest and Codex review queue;
- regression tests for relation expansion and summary reconciliation;
- game-specific report and workbook under ignored `analysis_outputs/`.

Implemented `app/analysis/reference_audit.py`,
`scripts/audit_reference_game.py`,
`scripts/build_reference_codex_review.py`, and focused regression tests. The
game-specific package remains ignored; the reusable method is documented in
`docs/reference-assisted-game-audit.md`.
