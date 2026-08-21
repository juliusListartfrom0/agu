# TASK-0258 Module A Testing

## Automated verification

### Current P1 fresh-context review check (2026-08-20)

- TASK-0258 focused suite: `209 passed` (including the local simulation command,
  its fail-closed P5 boundary tests, symlinked-parent rejection tests, and
  pre-lock authorization, symlinked-lock, candidate-drift, and fixed-stage
  residue validation for candidate/bundle/result/failure publication). Candidate
  publication now validates and binds the candidate gate's exact rerun
  authorization SHA to its fixed stage and checks terminal occupancy under the
  output-root lock.
- The independent fresh-context implementation review: **failed** with
  `Critical=7 / Required=5 / Optional=3`.
- No model/video extraction or v2 rerun was performed.
- The passing focused suite does not prove receipt binding, kernel read
  isolation, authenticated bootstrap, or process-tree cleanup.
- Full repository pytest after the review-only no-write preflight increment: `2,077 passed, 5
  skipped, 15 warnings`.
- The bootstrap regression coverage now pins the four exact parent review
  command arrays, the fixed request/source FD map `202/203`, and no-follow
  canonical reopening of the standalone runtime contract through a
  descriptor-relative directory-FD chain, including a symlink-ancestor case;
  it also rejects closed, aliased, and non-regular inherited descriptors.
- The review-only capability/loader tests prove temporary-ancestor identity
  re-open, canonical artifact/file hashes, context separation, and wrong-context
  rejection. JSON manifest/bundle reads use bounded descriptor-relative
  no-follow traversal and reject symlinked temporary parents; they do not
  provide kernel-audit or production admission evidence.
- Run-history marker append validates the authorization SHA before touching a
  lock path, so malformed authorization input cannot create a path outside the
  registry directory.
- Marker append now replays the durable ledger under the registry lock and
  requires the new marker's authorization, run identity, run-root/nonce, and
  predecessor receipt to equal the durable claim/completion/head; forged
  predecessor and identity-drift regressions fail before publication.
- Shared v2 lock acquisition now creates/opens a regular lock file with
  `O_NOFOLLOW`/`O_CLOEXEC`/nonblocking flags; callers no longer pre-touch a
  potentially symlinked lock path.
- Candidate receipt-bundle sealing now takes the output-root and distinct
  bundle-parent locks, reopens and recomputes all candidate member receipts
  under those locks, rejects candidate drift and exact fixed-stage residue, and
  reopens the published bundle bytes. This is repository-local publication
  hardening, not a substitute for the external receipt issuer or OS sandbox.
- The public pipeline preflight now cross-binds the candidate gate's
  authorization, admission, and run-identity receipts to the actual admission
  and completion bytes, then requires the bundle and result to share the same
  authorization/admission/history/static-input tuple and exact candidate-member
  rows. Mismatches fail before registry, output-root, or bundle writes; tests
  cover admission drift and result-member drift with zero side effects.
- The review-only candidate bundle loader now requires the real absolute
  `candidate_v2` directory, reopens exact member coverage, recomputes all ten
  member receipts, and rejects candidate-byte drift. It remains a review-only
  loader and supplies no production admission or kernel evidence.
- The review-only run-admission loader now replays canonical claim,
  `run_admission.json`, and completion bytes, binds their receipt edges, and
  compares the completion's root/admission physical CAS to the reopened
  temporary files. The terminal loader then replays the candidate bundle and
  exactly one `verified_result_v2` or `postverification_failure_v2` sibling,
  binding history, admission, static-input, authorization, bundle, and member
  receipts. These loaders remain diagnostic-only.
- Candidate bundle replay now also reopens and validates the candidate gate
  schema; terminal replay binds its authorization, run-admission, static-input,
  and run-history provider slots to the independently loaded trust spine.
- Read-isolation policy/attestation binding now verifies the policy artifact
  hash, provider/run/worker/nonce tuple, deny-before-allow precedence, unique
  longest policy-row match, fixed inherited-FD locators, and returned regular
  file identity. It validates an externally produced attestation only; it
  never creates kernel evidence, and raw `fs_usage`/Python-row paths remain
  fail-closed.
- The review-only capability loader now reopens both policy and attestation
  artifacts through the bounded descriptor-relative no-follow JSON boundary,
  checks their exact file/internal hashes, and returns an opaque cross-bound
  pair. It remains diagnostic-only and cannot mint a kernel attestation or
  production admission.
- The verification-attempt loader now reopens the complete canonical attempt
  artifact, verifies its internal/file hashes, and invokes the existing
  attempt validator—including the nested read-isolation binder—before
  returning an opaque review-only artifact.
- The review-only trust-spine binder now cross-binds that attempt to the
  reopened admission, stable history identity/head, rerun authorization, and
  output-root physical CAS. It remains diagnostic-only and cannot issue
  production admission.
- The review-only no-write preflight now reopens output-root and registry
  identities, rejects reserved output/stage residue and candidate-bundle
  aliases, and returns a diagnostic write plan without creating locks,
  directories, or files. It remains diagnostic-only and cannot issue
  production authorization.
- `verify_verification_attempt` now invokes that binder as the attempt-level
  gate, so individually schema-valid but cross-boundary-drifting policy and
  attestation objects cannot enter a verification attempt.
- Canonical JSON reopening restores the schema's fixed authorization-provider
  iteration order only after canonical bytes and the internal artifact hash
  have been verified; this avoids treating JSON serializer key sorting as
  receipt drift.
- `verified_result_v2` and `postverification_failure_v2` sealing now derive
  deterministic stage-directory names from the rerun-authorization SHA, reject
  pre-existing stage residue, publish no-clobber, and reopen exact member
  coverage after rename. This also remains repository-local evidence.
- Terminal publication now replays the candidate-present topology under the
  output-root lock before creating a stage and rejects an existing terminal
  sibling, so result and postverification failure cannot coexist in one root.
- The read-isolation audit tests prove that both raw `fs_usage` rows and direct
  Python `verified_event_rows` injection fail closed with no attestation; the
  local policy↔attestation binder rejects tuple drift, unknown/denied events,
  and ambiguous policy matches; the external kernel-provider capability is
  still absent.

- Module-A focused tests: `67 passed`.
- Adjacent TASK-0257 export/probe/backbone/causal regression: `523 passed, 3 skipped`.
- Scoped Ruff check and Ruff format check passed.
- Targeted whitespace and repository harness checks passed.
- The three approved specification files were rehashed after implementation
  and remain byte-identical to the user-approved SHA tuple.
- The three Module-A CLIs return successful `--help` output under canonical
  `.venv` Python 3.11.

No model/video extraction or v2 rerun was performed, and no v2 terminal/result
artifact is claimed.

Endpoint Security capability probe: `compile_ok=true`, SDK header/stub present,
signature `adhoc`, required entitlement absent,
`external_user_approval_observed=false`, status
`blocked_external_authorization`. This is a platform blocker, not a
read-isolation attestation. The probe now accepts
`--signed-artifact <path> --json` to inspect the actual signed executable or
`.systemextension`; without that option it intentionally reports the
temporary ad-hoc compile and cannot observe an installed system extension.

## Repository-local simulation

The canonical local simulation command is:

```bash
.venv/bin/python scripts/task0258_local_simulation.py --json
```

The 2026-08-21 run exercised the synthetic discovery manifest and both
review-only state-machine contexts. It returned `evidence_class=diagnostic_only`
for the platform boundary, `production_capability=false`, and `p5_ready=false`.
The platform portion remained `blocked_external_authorization` with ad-hoc
signing, no Endpoint Security entitlement, no user approval, and no kernel
read-isolation attestation. This command is deliberately not a substitute for
an OS-enforced review sandbox or real Endpoint Security evidence.

## Local service curl hook

- Started the canonical service with `.venv/bin/python -m uvicorn app.main:app
  --host 127.0.0.1 --port 8765`.
- `GET /health` returned `{"status":"ok"}` and `GET /ready` returned
  `{"status":"ready"}`.
- `POST /api/v1/analysis/run` with `examples/lebron_shoots.mp4`,
  `vlm_mode=off`, `generate_video=false`, `segmented_analysis=false`, and
  `max_frames=60` returned a `task_id` with `status=pending`; polling the
  status endpoint reached `status=completed`, `progress=100`, `error=null`.
- An intentionally missing video path returned HTTP 400 as documented. No
  v2 extraction or TASK-0258 result artifact was produced by this service
  smoke.

## Real artifact verification

The terminal attempt record file SHA-256 is
`e5b6a00b1240bfd80f5a27ac590cdb1d2df25fe41492d365e7097082a47b11c0`;
its internal artifact SHA-256 is
`f1dcdac99e1056b47dabc3b178169ebedcf20874cce78ea08b00d9bafeeaef0a`.
The 49-row canonical resource log file SHA-256 is
`34418424008dd1d405e61e8dfe0b9960f04f10b7d3792396e74d2f83c66dc58d`.
The mechanical failure file/internal SHA pair is
`edfa96606d15ebe2a730e2804fd0000063b32682c413a101851c761986c21cec` /
`fce75703e40daba1cc03ced79a56fb6f9f098a02b00a802a7015941838ce7d21`.
All receipts, resource rows, terminal schemas, and the absence of `final_v1`
were verified after publication.

## Local service curl hook

The local service started successfully on `127.0.0.1:8765` using `.venv`.
`GET /health` returned `{"status":"ok"}` and `GET /ready` returned
`{"status":"ready"}`. `POST /api/v1/analysis/run` returned HTTP 200 with a
pending task ID and the documented response shape. The first status poll
returned a well-formed failed task: the unrelated runtime tried to download the
missing `yolov8n.pt` from GitHub and exhausted three retries with curl error 35
(`SSL_ERROR_SYSCALL`). The service shut down cleanly. Module A changes no public
API/startup/request/response contract, so README and API documentation require
no update.

## Result boundary

No Module-A accuracy, precision, recall, or balanced-accuracy result exists:
the disk gate failed before final publication. Readiness remains `not_ready`,
blind inference remains paused, and Module B remains unauthorized.
