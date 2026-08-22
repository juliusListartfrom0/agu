# TASK-0258 Amendment-001 — current fresh-context review handoff

Review target: `codex/agu-p1-remediation-2` at commit `5549408`
Baseline: `main` / `origin/main` at `c5157f2`
Diff SHA-256 (`git diff --binary c5157f2...5549408`):
`5e88f55058e0563eef1cc1505d864ba7920d196eccced7c30face6b93b331f08`
Changed-file count: 46

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

Review the complete `c5157f2...5549408` scope, not only the latest local
fixes. In particular, adversarially review receipt/marker replay, publication
locks and races, canonical/no-follow loaders, bootstrap and FD binding, worker
process-tree cleanup, diagnostic-vs-production admission, and every place a
synthetic or caller-supplied value could be promoted into authority.

## Reproducible verification commands

Use the canonical environment and record exact output:

```bash
.venv/bin/python -m pytest -q tests/test_task0258_*.py
.venv/bin/python -m pytest -q
git diff --name-only c5157f2...5549408 -- '*.py' | \\
  xargs .venv/bin/python -m ruff check
.venv/bin/python scripts/verify_harness.py
.venv/bin/python scripts/task0258_local_simulation.py --json
.venv/bin/python scripts/task0258_endpoint_security_capability.py --json
```

The current local evidence is 284 focused TASK-0258 tests and 2,152 full-suite
tests, with 5 skips and 15 warnings; the local service hook also reached
`/health`/`/ready`, submitted a lightweight task, and observed `completed`.
The latest remediation closes the prior re-review's mutable `FlockHandle` seal
state and path-based registry write findings by using immutable seal state and
one locked, descriptor-relative registry transaction. It also validates all
ten candidate members by path-specific schema, canonical JSON/JSONL encoding,
resource-log receipts, and attempt/log counters; verification artifacts now
verify their own internal hashes; review locks use stable parent directory FDs;
and the candidate marker/bundle and pre-publication gate-history bindings are
checked separately. The follow-up remediation also binds the candidate gate's
verification-attempt receipt and producer chain to candidate members, ties
held locks to their parent-directory identities, proves the executed bootstrap
script is the inherited source FD, and requires complete read-isolation
allowlist/event coverage. The simulation and
capability probe must remain explicitly diagnostic-only and must report no
production capability.
The repository-wide `ruff check app scripts tests` baseline currently reports
52 existing findings outside the TASK-0258 diff; the diff-scoped Python check
above passes, and this task does not silently modify unrelated files.

## Latest independent review attempt (2026-08-22)

A fresh read-only reviewer inspected the exact `5549408` target and returned:
`scope_complete=false`, `sealed_receipt=false`, `Critical=0`, `Required=0`,
`Optional=0`, with no files modified. Focused TASK-0258 tests (`284 passed`),
diff-scoped Ruff, and diff checks passed. Its isolated target snapshot could
not run a complete full-suite/harness verification because the extracted
snapshot lacked the canonical `.venv` and repository-local generated
datasets/checkpoints; the current checkout separately passes `2,152 passed,
5 skipped, 15 warnings` and Harness `--run-tests`. This is an incomplete
review attempt, not a sealed 0/0 receipt; P1 remains open.

## Required review output

The reviewer must produce a new sealed receipt that records:

1. exact target commit and diff SHA above;
2. baseline/delta scope, review inputs, resource policy, and commands;
3. findings with severity and reproduction evidence;
4. `Critical=0 / Required=0` only if every such finding is actually closed;
5. reviewer/fresh-context provenance and review timestamp.

Until that independent receipt exists, the prior `Critical=7 / Required=5 /
Optional=3` review and the subsequent unsealed `Critical=1 / Required=1`
re-review remain failed review evidence, regardless of the green local test
suite. No v2 rerun, production admission, runtime promotion, readiness
transition, or blind inference is authorized by this handoff.
