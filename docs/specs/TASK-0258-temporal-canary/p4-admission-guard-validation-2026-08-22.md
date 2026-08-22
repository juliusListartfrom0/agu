# TASK-0258 P4 admission-guard validation — 2026-08-22

Status: **not executed / waiting on P2 and P3 authority**

- Branch: `codex/agu-p4-admission-guard-audit`
- Base: `main` merge commit `5727ba4`
- P2 status: `blocked_external_authorization`
- P3 status: review-only replay passed; exact-SHA authorization unissued

## Fail-closed guard evidence

Command:

```text
./.venv/bin/pytest -q \
  tests/test_task0258_v2_pipeline_cli.py::test_run_v2_pipeline_end_to_end
```

Result: `1 passed`.

The test verifies that a production invocation without an external admission
issuer raises `PermissionError` before any registry, output root, candidate
bundle, or result file is created. A caller-supplied arbitrary context with
`synthetic_test_only=True` is also rejected. The explicitly issued synthetic
test context is exercised only in the test's diagnostic path and is not a
production authorization.

## P4 remains incomplete

The actual P4 gate still requires a separately issued exact-SHA v2 rerun
authorization, a valid P2 platform evidence bundle, and one authorized
Module-A producer/verification run that publishes a verified result or a
canonical terminal failure. No model/video extraction, production output, or
P4 result was created by this audit.

P5 remains prohibited until its independent rights, causal-evidence,
performance, and VLM gates pass.
