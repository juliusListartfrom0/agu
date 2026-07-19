# Requirement

Audit the six-period `20260128_原片` basketball game against the supplied
`20260128_早鸟jq3` player highlight, missed-shot, defense, foul/violation,
detail-event, image, and team-summary references. Produce per-player technical
statistics, retain raw-video evidence provenance, and define accuracy with an
executable reconciliation gate rather than a conversational claim.

The reusable solution must remain an offline Python capability inside AGU. It
must not change the v3 inference preprocessing contract or turn Codex into an
online FastAPI dependency.
