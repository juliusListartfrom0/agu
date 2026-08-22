# TASK-0258 Amendment-001 — current fresh-context review handoff

Review target: `codex/agu-p1-remediation-2` at commit `b18c784`
Baseline: `main` / `origin/main` at `c5157f2`
Diff SHA-256 (`git diff --binary c5157f2...b18c784`):
`9d4f4bddc87b288f8de0a5e5b6996a8b7e67d7e3aced9306b2568130bd30dcdf`
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

Review the complete `c5157f2...b18c784` scope, not only the latest local
fixes. In particular, adversarially review receipt/marker replay, publication
locks and races, canonical/no-follow loaders, bootstrap and FD binding, worker
process-tree cleanup, diagnostic-vs-production admission, and every place a
synthetic or caller-supplied value could be promoted into authority.

## Reproducible verification commands

Use the canonical environment and record exact output:

```bash
.venv/bin/python -m pytest -q tests/test_task0258_*.py
.venv/bin/python -m pytest -q
git diff --name-only c5157f2...b18c784 -- '*.py' | \\
  xargs .venv/bin/python -m ruff check
.venv/bin/python scripts/verify_harness.py
.venv/bin/python scripts/task0258_local_simulation.py --json
.venv/bin/python scripts/task0258_endpoint_security_capability.py --json
```

The current local evidence is 297 focused TASK-0258 tests and 2,165 full-suite
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
allowlist/event coverage. The registry write path now also binds a held
history lock to the exact registry directory FD and rejects non-canonical
output-parent locks. The capability probe now requires an Apple-anchored
codesign designated requirement before classifying an artifact as signed, so
an arbitrary self-signed Authority cannot satisfy `--require-capability`. The
simulation and
capability probe must remain explicitly diagnostic-only and must report no
production capability.
The repository-wide `ruff check app scripts tests` baseline currently reports
52 existing findings outside the TASK-0258 diff; the diff-scoped Python check
above passes, and this task does not silently modify unrelated files.

## Previous independent review attempt (2026-08-22)

A fresh read-only reviewer inspected the exact `5549408` target before the
latest fixes and returned `scope_complete=false`, `sealed_receipt=false`,
`Critical=0`, `Required=2`, `Optional=1`, with no files modified. The two
Required findings were the missing registry-FD parent identity checks in
`seal_run_consumption_claim`/`seal_run_consumption_completed` and acceptance
of a caller-selected output lock in `create_run_history_registry`. The
Optional finding was that an arbitrary self-signed `Authority=` could be
classified as signed. These are now covered by the `313582b` and `d8569a8`
fixes plus regression tests. This is not a sealed 0/0 receipt; P1 remains
open pending a new review of `afbfef0`.

## Current remediation and review status (2026-08-22)

Commit `9fd347a` addresses the previous review's three Required findings:

1. Candidate bundle publication and replay now hold stable parent-directory
   FDs and use descriptor-relative lock, verification, read, and publication
   operations; the CLI no longer passes a caller-held path lock into replay.
2. Review-only capability objects now reject post-construction field mutation
   and recursively freeze nested JSON mappings/sequences while preserving
   `dict`/`list` verifier contracts.
3. Run-history claim/completion sealers acquire the exact history lock when the
   caller does not supply one, while still validating caller-held locks against
   the exact registry-directory FD.

`290` focused TASK-0258 tests, `2,158` full-suite tests, Harness, local service
curl, simulation, and the capability probe passed for `9fd347a`. Its
independent review was complete in scope but returned four Required and one
Optional finding, so it was not a sealed receipt. Those findings are addressed
by the current `afbfef0` target below; until a new sealed receipt reports
`scope_complete=true`, `Critical=0`, and `Required=0`, P1 remains open.

## Current remediation round (2026-08-22)

Commit `afbfef0` addresses the four Required and one Optional findings from
the independent review of `9fd347a`:

1. Review capabilities retain immutable internal snapshots and return fresh
   verifier-compatible dict/list copies; public-field tampering is detected on
   access instead of relying on mutable built-in subclasses.
2. Bundle publication and replay revalidate that the requested bundle parent
   path still names the opened directory after descriptor-relative work.
3. Bundle publication/replay register every opened descriptor immediately in an
   `ExitStack`, including failures during candidate-directory opening.
4. Registry sealers reject a caller-supplied registry FD unless it is paired
   with the matching held history lock.
5. The obsolete caller-held output-lock parameter was removed from bundle
   replay; the CLI already uses the stable self-owned lock transaction.

`297` focused TASK-0258 tests, `2,165` full-suite tests, Harness, simulation,
capability probe, and clean-archive replay (`297 passed`) pass. A fresh
independent review of `b18c784` is required; P1 remains open until its sealed
receipt reports `scope_complete=true`, `Critical=0`, and `Required=0`.

The follow-up `b18c784` additionally binds `candidate_dir.parent` and
`candidate_dir` to the opened output-root/candidate FDs before and after bundle
publication/replay, rejecting output-root or candidate-path replacement.

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
