# TASK-0258 P3 review-only validation handoff — 2026-08-22

Status: **not authorized / waiting on P2**

- Branch: `codex/agu-p3-review-only-audit`
- Base: `main` merge commit `7109fa9`
- Branch commit: `2160262` (pushed to the remote branch)
- Documentation merge: `main@5c1f560`
- P2 prerequisite: not satisfied; platform status remains
  `blocked_external_authorization`

## Local review-only validation

The following local graph checks passed:

```text
./.venv/bin/pytest -q tests/test_task0258_v2_capabilities.py \
  -k "static_input_loader_replays_complete_parent_graph_as_review_only or \
  review_rerun_authorization_requires_per_use_static_replay or \
  review_no_write_preflight or \
  run_admission_loader_can_bind_to_replayed_static_input_contract or \
  terminal_loader_can_bind_to_replayed_static_input_contract"
```

Result: `7 passed, 43 deselected`.

These tests prove that the review-only path replays the static-input graph,
requires per-use replay before binding rerun authorization, rejects mutated or
unbound admission spines, and binds admission/terminal artifacts to the
replayed contract. They do not issue a production authorization or consume a
run.

## Missing P3 authority

The exact-SHA v2 rerun authorization remains unissued because P2 has not
produced the required real Endpoint Security entitlement, approved System
Extension, authenticated kernel read-isolation evidence, and OS-enforced
review-sandbox evidence. Local fixtures, synthetic receipts, or a caller-made
authorization cannot satisfy this gate.

P4 model execution and P5 readiness remain prohibited. The next valid P3 step
is to create a new exact-SHA authorization only after the P2 platform evidence
is independently sealed.
