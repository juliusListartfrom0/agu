# TASK-0258 Module A Testing

## Automated verification

### Current P1 fresh-context review check (2026-08-20)

- TASK-0258 focused suite: `170 passed`.
- The independent fresh-context implementation review: **failed** with
  `Critical=7 / Required=5 / Optional=3`.
- No model/video extraction or v2 rerun was performed.
- The passing focused suite does not prove receipt binding, kernel read
  isolation, authenticated bootstrap, or process-tree cleanup.
- Full repository pytest after the remediation slice: `2,038 passed, 5
  skipped, 15 warnings`.
- The review-only capability/loader tests prove temporary-ancestor identity
  re-open, canonical artifact/file hashes, context separation, and wrong-context
  rejection; they do not provide kernel-audit or production admission evidence.

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
