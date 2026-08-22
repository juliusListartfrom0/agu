# TASK-0258 Amendment-001 — fresh-context implementation review

Review date: 2026-08-20

The current checkout was reviewed from a separate read-only fresh context at
commit `c5157f2` (`main` / `origin/main`). No model, video, or v2 extraction
was executed. The focused TASK-0258 suite was re-run with the canonical
`.venv` interpreter: `161 passed`.

## Verdict

The implementation review is **not ready / failed**. The required seal cannot
be issued because the findings are:

- Critical: 7
- Required: 5
- Optional: 3

The passing unit suite is schema and temporary-filesystem evidence only. It is
not evidence of the production trust boundary, read isolation, or a retained
v2 result.

## Critical findings

1. `run_v2_pipeline()` accepts caller-supplied candidate, bundle, and result
   payloads without receipt-bound validation and state-transition replay
   (`app/analysis/task0258_v2_pipeline_cli.py:45-78`).
2. Candidate, bundle, result, and failure validators do not replay nested
   receipts, provider bindings, artifact hashes, or canonical bytes
   (`app/analysis/task0258_v2_artifacts.py:342-496`).
3. Run-history writes do not durably append/replay the complete marker chain;
   `append_run_history_marker()` trusts the caller's `prior_event`
   (`app/analysis/task0258_v2_registry.py:32-62,89-111`).
4. Candidate publication lacks the required output-root/bundle-parent locks,
   fixed authorization-bound stage name, locked replay, and post-rename reopen
   (`app/analysis/task0258_v2_pipeline.py:169-176`).
5. Read-isolation policy/attestation and the Python audit are self-reportable,
   accept incomplete or arbitrary rows, and hardcode the unknown-read count to
   zero (`app/analysis/task0258_v2_read_isolation.py:82-147`,
   `app/analysis/task0258_v2_audit.py:116-180`).
6. The Endpoint Security client is unsigned/unentitled, not integrated, and
   does not provide authenticated lineage, exact syscall results, or overflow
   evidence (`scripts/endpoint_security/es_read_isolation_audit.c:98-197`).
7. Bootstrap/worker isolation does not authenticate request/source FDs or fix
   the target/argument allowlist, and timeout cleanup does not kill a process
   group (`scripts/task0258_module_a_verified_bootstrap.py:21-39`,
   `app/analysis/task0258_v2_bootstrap.py:58-123`,
   `app/analysis/task0258_v2_worker_runner.py:45-85`).

## Required follow-up

The next implementation slice must add the review-sandbox binders/traversal
APIs/verified loaders and production admission boundary defined by the
post-publication proof, obtain the external kernel-audit proof, issue a
separate exact-SHA rerun authorization, and retain either a verified result or
canonical terminal failure. It must also harden path publication, add
cross-receipt/marker-chain/bootstrap/descendant/audit/race adversarial tests,
and reconcile the implementation/testing documents with actual evidence.

Until a new independent review returns Critical=0 and Required=0, no v2 rerun,
Module B action, runtime promotion, readiness transition, or blind inference
is authorized.
