# TASK-0041 Code Review

## Interim W5 review (2026-07-16)

- Existing `action_proxy_v1` and v3 preprocessing remain unchanged.
- Official aggregation accepts only confirmed event states and rejects broken
  assist/block/steal/rebound relations.
- The sealed prediction manifest contains raw-video content identity but not
  reference paths; edited/reference filename markers and non-video inputs fail.
- External detector loading is lazy and opt-in. Absence/failure cannot create an
  official event.
- Residual risk: E-BARD/COCO ball recall is insufficient; official contracts
  are not yet wired into the public analysis result; strict actor recognition
  remains unproven; Game B is partial-quarter truth only.
