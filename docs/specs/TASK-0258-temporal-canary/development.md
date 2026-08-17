# TASK-0258 Module A Development

## Authorization boundary

The user approved only Module A with these exact SHA-256 values:

- requirement: `5c455408bd9ba730470b66189316578dc935b39a5cdc550c67e2a9c08d3f80d3`
- solution: `87f9a447112de8d294ffedbb1adfadb478713d94e00cdcf2e4f98999ea8a857d`
- gate review: `cc52d809f42abba25ccdfb6667325612ce2f8961842f0e375db060ea2b70fa8f`

The implementation does not authorize Module B, runtime use, formal evaluation,
promotion, readiness changes, or blind inference.

The user's explicit exact-SHA approval was received after the first v1 attempt.
The valid post-approval receipt is
`analysis_outputs/public_research/task0258_module_a_spec_registry_v2/module_a_spec_approval.json`;
its internal/file SHA-256 pair is
`42273c9db3e5c77da8f577ed0e5142e6210faf3caee53f572402229eac71944f` /
`0eb1759d992b9587ad47bf9ae08793a69bac266ada81d39258919ac61e06e9fb`.
The older `registry_v1` receipt was prepared before explicit approval and cannot
be used as evidence that the v1 execution was authorized.

## Implemented slice

Module A adds an exact-receipt temporal feature plan, batch-one MPS tiled-Swin
extraction, immutable attempt/resume state, sustained resource monitoring,
four-game nested LOGO retrospective screening, separate baseline and candidate
final evaluators, and exclusive generation publication. All public boundaries
fail closed on receipt, schema, source-video, resource, path-alias, or disk
budget drift.

The implementation uses canonical `.venv` Python 3.11. It does not use the
removed legacy `venv`, download new weights, or alter retained source videos,
existing TASK-0257 artifacts, runtime model defaults, or blind assets.

## Real execution outcome

The pre-approval guarded v1 run completed all 45 private tiled-Swin rows and 49 resource
samples in about 99.44 seconds. The final locked disk revalidation then observed
insufficient capacity for the required 1 GiB worst-case publication write plus
3.5 GiB reserve. No embedding or evaluator generation was published.

The immutable terminal generation is
`analysis_outputs/public_research/vru_causal_temporal_retrospective_v1/terminal_failure_v1/`.
Its mechanical failure has internal SHA-256
`fce75703e40daba1cc03ced79a56fb6f9f098a02b00a802a7015941838ce7d21`
and file SHA-256
`edfa96606d15ebe2a730e2804fd0000063b32682c413a101851c761986c21cec`.
The stored `stop_reason=determinism_failure` is a preserved v1 classification
defect; independent time, disk, and locked-gate evidence identify the actual
cause as the disk reserve. The implementation now uses a dedicated
`DiskWriteBudgetError` and maps future occurrences to `disk_failure`; the
immutable v1 terminal evidence was not rewritten.

Any rerun requires a new versioned boundary after a separate fresh-context
implementation review and explicit user direction.
