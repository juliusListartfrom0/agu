# Gate Review

Status: PASS for the offline ledger and evidence-package implementation.

- The source of truth remains raw video; supplied edits and CSVs are reference
  evidence and reconciliation targets.
- Statistical agreement, raw-video localization coverage, and Codex-reviewed
  event accuracy are reported separately.
- A 95% claim is allowed only for a metric whose denominator and reviewed sample
  are emitted in the machine-readable report.
- Paths are CLI inputs; no user-specific path or environment value enters AGU
  service code.
- The v3 preprocessing contract, FastAPI schemas, and public API are unchanged.
