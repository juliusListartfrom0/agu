# Solution

Implement a reference-assisted audit adapter with three explicitly separated
measurements:

1. expand compressed detail rows into an immutable event ledger (an assist also
   implies the assisted made shot; a block also implies the blocked missed shot;
   a steal and an offensive foul imply turnovers);
2. aggregate the expanded ledger and reconcile every auditable player counting
   field against the supplied team CSVs;
3. fingerprint reference edit segments and locate them in the six raw period
   videos, then emit bounded Codex review windows and append-only decisions.

AGU owns event normalization, aggregation, reconciliation, evidence manifests,
and accuracy definitions. FFmpeg/OpenCV are replaceable local adapters for video
sampling and fingerprint matching. Reference files are read-only inputs and all
generated packages remain under ignored analysis output directories.

Wiki context read: `agu-complete-box-score-plan-2026-07-13` and
`agu-ground-truth-baseline-evaluator-2026-07-02`.
