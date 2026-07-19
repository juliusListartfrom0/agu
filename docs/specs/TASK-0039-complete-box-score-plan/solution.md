# Solution

The authoritative design is `docs/complete-box-score-plan.md`.

Key decision: replace direct action-proxy counting with an immutable basketball
event graph and ledger. Traditional perception provides ball/rim/court/player
geometry and possession transitions; the on-device VLM reviews bounded event
windows; Codex performs scene/court/roster calibration, event annotation,
conflict review, negative labeling, and final reconciliation through append-only
review decisions.

External detection, tracking, pose, OCR, VLM, and annotation systems remain
optional adapters. AGU owns schemas, evidence fusion, state transitions, event
relationships, reconciliation, and aggregation.
