#!/usr/bin/env python3
"""Second-pass de-hardlink: rewrite every remaining nlink>1 scope regular file
whose other hardlinks live OUTSIDE the repository root (external aliases).

For each such file, write a fresh byte-identical copy to a temp file and
os.replace it, so the repo entry gets a new inode with nlink=1. Content SHA-256
and mode bits are preserved. The external aliases (outside the repo) keep the
original inode and are not touched.
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
    "dehardlink_external_log.jsonl",
)
TMP_SUFFIX = ".dehardlink2.tmp"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    targets = []
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
                targets.append(p)

    replaced = 0
    bytes_written = 0
    with open(LOG, "w") as log:
        for p in sorted(targets):
            before = sha256_file(p)
            mode = stat.S_IMODE(os.lstat(p).st_mode)
            size = os.lstat(p).st_size
            tmp = p + TMP_SUFFIX
            shutil.copyfile(p, tmp)
            os.chmod(tmp, mode)
            os.replace(tmp, p)
            after = sha256_file(p)
            assert after == before, f"content mismatch: {p}"
            new_nlink = os.lstat(p).st_nlink
            assert new_nlink == 1, f"still nlink={new_nlink}: {p}"
            bytes_written += size
            log.write(
                json.dumps(
                    {
                        "path": p,
                        "size_bytes": size,
                        "new_nlink": new_nlink,
                        "content_sha256": after,
                    }
                )
                + "\n"
            )
            replaced += 1

    print(f"replaced={replaced}")
    print(f"bytes_written={bytes_written:,} ({bytes_written / 1e9:.2f} GB)")
    print(f"log={LOG}")


if __name__ == "__main__":
    main()
