# Gate Review

- Python-only offline adapter; no API or v3 preprocessing change.
- External download/probe remains optional and outside service business logic.
- Truth values are omitted from the plan; only rosters, evidence counts, paths,
  and hashes are recorded.
- Benchmark and enrollment YouTube IDs must be disjoint.
- At least two benchmark games are mandatory.
- The selected ATL-CHI and LAL-BOS pairs have complete active-roster coverage.

Decision: media and provenance gates are ready. Identity enrollment remains
fail-closed until each declared roster reaches at least 95% gallery coverage;
benchmark inference must not start before that freeze gate passes.
