# TASK-0258 Amendment-001 — current fresh-context review handoff

Review target: `codex/agu-p2` at commit `e8b4077`  
Baseline: `main` / `origin/main` at `c5157f2`  
Diff SHA-256 (`git diff --binary c5157f2...e8b4077`):
`151000a99a98369d2b798ca265d09f44d060d404cb213f772b4c3e3a5b2783d7`  
Changed-file count: 43

## Purpose

This file is a handoff for a genuinely separate fresh-context implementation
review of the current remediation. It is not a review receipt, does not claim
`Critical=0 / Required=0`, and does not authorize P2, P3, P4, or P5. The review
can be performed without Apple Developer Program access; the real Endpoint
Security entitlement, user approval, and kernel read-isolation proof remain a
separate P2 platform gate.

## Required reviewer inputs

Read the current checkout at the exact target commit, then inspect these
repository artifacts together with the code:

- `docs/specs/TASK-0258-temporal-canary/amendment-001-implementation-plan.md`
- `docs/specs/TASK-0258-temporal-canary/code-review.md`
- `docs/specs/TASK-0258-temporal-canary/fresh-context-review-2026-08-20.md`
- `docs/specs/TASK-0258-temporal-canary/testing.md`
- `docs/harness/P0-P5-EXECUTION-PLAN.md`
- `docs/harness/TASK-0258-EXTERNAL-UNBLOCK-CHECKLIST.md`

Review the complete `c5157f2...e8b4077` scope, not only the latest two local
fixes. In particular, adversarially review receipt/marker replay, publication
locks and races, canonical/no-follow loaders, bootstrap and FD binding, worker
process-tree cleanup, diagnostic-vs-production admission, and every place a
synthetic or caller-supplied value could be promoted into authority.

## Reproducible verification commands

Use the canonical environment and record exact output:

```bash
.venv/bin/python -m pytest -q tests/test_task0258_*.py
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check app scripts tests
.venv/bin/python scripts/verify_harness.py
.venv/bin/python scripts/task0258_local_simulation.py --json
.venv/bin/python scripts/task0258_endpoint_security_capability.py --json
```

The current local evidence is 262 focused TASK-0258 tests and 2,130 full-suite
tests, with 5 skips and 15 warnings. The simulation and capability probe must
remain explicitly diagnostic-only and must report no production capability.

## Required review output

The reviewer must produce a new sealed receipt that records:

1. exact target commit and diff SHA above;
2. baseline/delta scope, review inputs, resource policy, and commands;
3. findings with severity and reproduction evidence;
4. `Critical=0 / Required=0` only if every such finding is actually closed;
5. reviewer/fresh-context provenance and review timestamp.

Until that independent receipt exists, the prior `Critical=7 / Required=5 /
Optional=3` review remains the authoritative failed review, regardless of the
green local test suite. No v2 rerun, production admission, runtime promotion,
readiness transition, or blind inference is authorized by this handoff.
