#!/usr/bin/env python3
"""TASK-0258 Amendment-001 authorized pre-import bootstrap (FD plumbing).

Trusted-runner launch form:
``[PYTHON, "-P", "-S", "/dev/fd/203", "--task0258-review-request-fd", "202",
"--task0258-source-fd", "203"]``

This entry reads the ``agu.task0258-review-driver-request.v1`` from the request
fd, constructs the exact three-entry ``sys.path`` from the request's runtime
snapshot, and dispatches only ``runpy.run_module`` with the fixed target argv.
The pure core lives in ``app.analysis.task0258_v2_bootstrap``.
"""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    from app.analysis.task0258_v2_bootstrap import (
        build_sys_path_from_entries,
        dispatch_module,
        parse_bootstrap_flags,
    )

    request_fd, source_fd = parse_bootstrap_flags(sys.argv[1:])
    if source_fd < 0:
        return 2
    with os.fdopen(request_fd, "r", closefd=False) as fh:
        request = json.load(fh)
    entries = request["runtime_snapshot_receipt"].get("ordered_python_sys_path_entries")
    if entries is None:
        return 2
    runtime_root = request["runtime_snapshot_receipt"]["runtime_root_absolute_path"]
    sys.path[:] = build_sys_path_from_entries(runtime_root, entries)
    dispatch_module(request["target_module"], request["target_argv"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
