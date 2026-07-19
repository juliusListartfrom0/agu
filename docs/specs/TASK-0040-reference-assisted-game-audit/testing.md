# Testing

- Focused pytest: `tests/test_reference_audit.py`.
- Full-game ledger: 605 normalized events, 19 players, 304/304 counting fields.
- Reference fingerprint gate: 477/481 scenes localized (99.17%).
- Codex fixed-seed stratified review: 60/60 confirmed; one-sided exact 95%
  lower bound 95.13%, passing the requested 95% localization-accuracy gate.
- Spreadsheet formula scan: no formula errors; all five sheets rendered and
  visually checked after a column-width repair.
- Ruff passed; full pytest completed with `161 passed`; structural harness
  verification passed.
- Local service curl hook: not applicable because FastAPI routes, service task
  behavior, configuration, public schemas, README, and API contracts did not
  change; `README.md` and `docs/api.md` were checked and left unchanged.
- Remaining evidence queue: four long scenes are intentionally unresolved.
