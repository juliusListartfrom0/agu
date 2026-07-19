# Code Review

Review completed.

- Preserves FastAPI and v3 preprocessing contracts.
- Keeps user paths out of AGU code and committed docs.
- Separates CSV reconciliation, raw localization coverage, and reviewed
  localization accuracy.
- Rejects or queues weak matches instead of coercing them.
- FFmpeg and OpenCV remain replaceable local adapters with documented license
  boundaries and fallback behavior.
- Generated videos, frames, JSON, and workbook stay outside committed source.
