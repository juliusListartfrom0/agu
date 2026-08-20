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

import hashlib
import json
import os
import stat
import sys


def main() -> int:
    from app.analysis.task0258_module_a_v2 import compact_canonical_json
    from app.analysis.task0258_v2_bootstrap import (
        build_sys_path_from_entries,
        dispatch_module,
        parse_bootstrap_flags,
        validate_review_driver_request,
    )

    request_fd, source_fd = parse_bootstrap_flags(sys.argv[1:])
    if source_fd < 0:
        return 2
    request_stat = os.fstat(request_fd)
    if stat.S_ISREG(request_stat.st_mode):
        if request_stat.st_size > 8_388_608:
            return 2
        os.lseek(request_fd, 0, os.SEEK_SET)
        request_bytes = os.read(request_fd, request_stat.st_size + 1)
    else:
        chunks: list[bytes] = []
        total = 0
        while total <= 8_388_608:
            chunk = os.read(request_fd, min(65_536, 8_388_609 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        request_bytes = b"".join(chunks)
    if not request_bytes or len(request_bytes) > 8_388_608 or not request_bytes.endswith(b"\n"):
        return 2
    try:
        request = json.loads(request_bytes.decode("utf-8"))
        validate_review_driver_request(request)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        return 2
    if request_bytes != (compact_canonical_json(request) + "\n").encode("utf-8"):
        return 2
    source_stat = os.fstat(source_fd)
    if not stat.S_ISREG(source_stat.st_mode) or source_stat.st_size != request["review_bootstrap_source_size_bytes"]:
        return 2
    os.lseek(source_fd, 0, os.SEEK_SET)
    source_bytes = os.read(source_fd, source_stat.st_size + 1)
    if len(source_bytes) != source_stat.st_size:
        return 2
    if hashlib.sha256(source_bytes).hexdigest() != request["review_bootstrap_source_sha256"]:
        return 2
    os.lseek(source_fd, 0, os.SEEK_SET)
    entries = request["runtime_snapshot_receipt"].get("ordered_python_sys_path_entries")
    if entries is None:
        return 2
    runtime_root = request["runtime_snapshot_receipt"]["runtime_root_absolute_path"]
    sys.path[:] = build_sys_path_from_entries(runtime_root, entries)
    dispatch_module(request["target_module"], request["target_argv"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
