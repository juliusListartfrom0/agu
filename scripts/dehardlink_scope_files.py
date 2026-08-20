#!/usr/bin/env python3
"""De-hardlink every scope regular file in the AGU repo to nlink=1.

TASK-0258 Amendment-001 (line 326-328) requires every scope regular file in the
implementation-scope baseline to have st_nlink == 1. The repo currently contains
4,616 hardlinked inodes (12,568 directory entries). This script keeps the first
lexical path of each group and rewrites the remaining entries as independent
byte-identical copies (new inode, nlink=1), preserving content SHA-256 and mode
bits. It is idempotent: re-running finds fewer hardlinks and continues.
"""

import hashlib
import json
import os
import shutil
import stat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(
    ROOT,
    "analysis_outputs/public_research/task0258_module_a_amendment_approval",
    "dehardlink_log.jsonl",
)
TMP_SUFFIX = ".dehardlink.tmp"


def content_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_groups():
    groups = {}
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in (".git", ".venv")]
        for f in fn:
            p = os.path.join(dp, f)
            if p.endswith(TMP_SUFFIX):
                continue
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode) and st.st_nlink > 1:
                groups.setdefault(st.st_ino, []).append(p)
    return groups


def main():
    # clean any leftover temp files from a prior interrupted run
    leftover = []
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in (".git", ".venv")]
        for f in fn:
            if f.endswith(TMP_SUFFIX):
                leftover.append(os.path.join(dp, f))
    for p in leftover:
        try:
            os.unlink(p)
        except OSError:
            pass

    groups = collect_groups()
    n_groups = len(groups)
    n_entries = sum(len(v) for v in groups.values())
    replaced = 0
    bytes_written = 0

    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "w") as log:
        for ino, paths in sorted(groups.items()):
            paths.sort()
            keep = paths[0]
            keep_sha = content_sha256(keep)
            keep_mode = stat.S_IMODE(os.lstat(keep).st_mode)
            for extra in paths[1:]:
                tmp = extra + TMP_SUFFIX
                shutil.copyfile(keep, tmp)
                os.chmod(tmp, keep_mode)
                os.replace(tmp, extra)
                extra_sha = content_sha256(extra)
                if extra_sha != keep_sha:
                    raise SystemExit(f"content mismatch after dehardlink: {extra}")
                new_nlink = os.lstat(extra).st_nlink
                size = os.lstat(extra).st_size
                bytes_written += size
                rec = {
                    "inode": ino,
                    "kept": keep,
                    "replaced": extra,
                    "size_bytes": size,
                    "new_nlink": new_nlink,
                    "content_sha256": extra_sha,
                }
                log.write(json.dumps(rec) + "\n")
                replaced += 1

    print(f"groups={n_groups} entries={n_entries} replaced={replaced}")
    print(f"bytes_written={bytes_written:,} ({bytes_written / 1e9:.2f} GB)")
    print(f"log={LOG}")


if __name__ == "__main__":
    main()
