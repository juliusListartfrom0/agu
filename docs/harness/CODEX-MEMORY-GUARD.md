# Codex memory guard

This repository includes a read-only guard for the Codex host process and its
JSONL session logs:

```bash
./.venv/bin/python scripts/monitor_codex_memory.py --json
```

The default hard stop is 8 GiB for aggregate Codex RSS, the largest Codex
process, or one session JSONL file. A session file at 100 MiB is a warning
because the previous host incident was caused by a multi-gigabyte session and
duplicated parallel sessions. The guard reads process RSS and file metadata
only; it never reads session contents, kills processes, deletes files, or
restarts Codex.

Exit codes:

- `0`: continue, while keeping outputs bounded and avoiding duplicate parallel
  sessions.
- `2`: stop the current task before another heavy tool call. Close duplicate
  sessions, archive oversized logs through the normal Codex maintenance path,
  and restart Codex if necessary; rerun the guard before continuing.

The report intentionally emits only process kind/PID/RSS and session basename/
size. It does not include full command lines or session contents, preventing
the monitor itself from inflating the Codex session log.

## Current baseline

On 2026-08-22 after the TASK-0258 full-suite run, the guard reported:

```text
status=ok
aggregate_codex_rss≈2.2 GiB
largest_codex_process≈447 MiB
largest_session_log≈70 MiB
```

No restart or deletion was necessary. Heavy commands remain sequential, large
tool outputs are bounded, and the active workspace stays at the repository
root rather than the home directory.
