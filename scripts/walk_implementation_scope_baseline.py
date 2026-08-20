#!/usr/bin/env python3
"""TASK-0258 Amendment-001 implementation-scope baseline walker (v2).

Produces `agu.task0258-module-a-implementation-scope-baseline.v1` for the AGU
repository root. Convention notes (documented for auditability):

- Walk is no-follow (symlinks never followed), excluding only the two exact root
  children `.git` and `.venv` (recorded as `excluded_directory`).
- `mode_bits` = `st_mode & 0o7777` for every existing entry; null for absent.
- `ordered_entry_receipts` sorted by repository-relative lexical path, root `.`
  first. One extra `absent` row for the future bootstrap
  `scripts/task0258_module_a_verified_bootstrap.py`.
- regular row: link_count=1, size_bytes, file_sha256, hardlink_group_sha256 =
  sha256(compact-canonical [path]); symlink_target/ordered_child_names null.
- directory row: mode_bits + ordered_child_names; other conditional fields null.
- symlink row: mode_bits + symlink_target_text (readlink); other fields null.
- excluded_directory row: mode_bits; other fields null.
- Compact-canonical JSON: sorted object keys, `(',', ':')` separators, no
  NaN/Infinity, NO trailing LF (matches verified `module_a_spec_approval.json`).
- artifact_sha256 = sha256(compact-canonical object minus artifact_sha256 field).
"""

import datetime
import hashlib
import json
import os
import stat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK_OUTPUT = os.path.join(os.path.dirname(ROOT), "agu-task0258-check-output")
BOOTSTRAP = "scripts/task0258_module_a_verified_bootstrap.py"
OUT = os.path.join(
    ROOT,
    "analysis_outputs/public_research/task0258_module_a_amendment_approval",
    "implementation_scope_baseline.json",
)
EXEC_EXTS = (".py", ".pyi", ".pyx", ".sh", ".zsh")


def canon(obj):
    if isinstance(obj, dict):
        return (
            "{"
            + ",".join(
                f"{json.dumps(k, separators=(',', ':'))}:{canon(v)}"
                for k, v in sorted(obj.items(), key=lambda kv: json.dumps(kv[0], separators=(",", ":")))
            )
            + "}"
        )
    if isinstance(obj, list):
        return "[" + ",".join(canon(v) for v in obj) + "]"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if obj is None:
        return "null"
    if isinstance(obj, (int, float)):
        return str(obj)
    if isinstance(obj, str):
        return json.dumps(obj, separators=(",", ":"))
    raise TypeError(type(obj))


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_entry(rel, kind, mode=None, link=None, size=None, file_sha=None, symlink=None, hardlink=None, children=None):
    return {
        "path": rel,
        "entry_kind": kind,
        "mode_bits": mode,
        "link_count": link,
        "size_bytes": size,
        "file_sha256": file_sha,
        "symlink_target_text": symlink,
        "hardlink_group_sha256": hardlink,
        "ordered_child_names": children,
    }


def main():
    # pre-flight: check-output dir must be empty, mode 0o700, no symlink
    ck_lst = os.lstat(CHECK_OUTPUT)
    if not stat.S_ISDIR(ck_lst.st_mode):
        raise SystemExit("check-output is not a directory")
    if stat.S_IMODE(ck_lst.st_mode) != 0o700:
        raise SystemExit(f"check-output mode != 0o700: {oct(ck_lst.st_mode)}")
    if os.listdir(CHECK_OUTPUT):
        raise SystemExit("check-output directory is not empty")

    entries = []
    symlink_to_exec = []

    def visit(rel, abs_path, is_root=False):
        st = os.lstat(abs_path)
        mode = st.st_mode & 0o7777
        if stat.S_ISDIR(st.st_mode):
            if (not is_root) and rel in (".git", ".venv") and "/" not in rel:
                entries.append(make_entry(rel, "excluded_directory", mode=mode))
                return
            children = sorted(os.listdir(abs_path))
            entries.append(make_entry(rel, "directory", mode=mode, children=children))
            for c in children:
                child_rel = c if rel == "." else rel + "/" + c
                visit(child_rel, os.path.join(abs_path, c))
        elif stat.S_ISREG(st.st_mode):
            if st.st_nlink != 1:
                raise SystemExit(f"nlink != 1: {rel} (nlink={st.st_nlink})")
            entries.append(
                make_entry(
                    rel,
                    "regular",
                    mode=mode,
                    link=1,
                    size=st.st_size,
                    file_sha=sha256_file(abs_path),
                    hardlink=sha256_bytes(canon([rel]).encode("utf-8")),
                )
            )
        elif stat.S_ISLNK(st.st_mode):
            target = os.readlink(abs_path)
            entries.append(make_entry(rel, "symlink", mode=mode, symlink=target))
            # detect symlink resolving to executable local code
            resolved = os.path.join(os.path.dirname(abs_path), target)
            try:
                r_st = os.stat(resolved)
            except OSError:
                r_st = None
            if r_st is not None and stat.S_ISREG(r_st.st_mode):
                base = os.path.basename(resolved)
                if base.endswith(EXEC_EXTS) or (r_st.st_mode & 0o111):
                    symlink_to_exec.append((rel, target))
        else:
            raise SystemExit(f"unknown entry kind: {rel}")

    visit(".", ROOT, is_root=True)
    entries.append(make_entry(BOOTSTRAP, "absent"))
    entries.sort(key=lambda e: e["path"])

    # executable subset: .py/.pyi/.pyx/.sh/.zsh, conftest.py, or execute bit
    executable = []
    for e in entries:
        base = e["path"].rsplit("/", 1)[-1]
        if e["entry_kind"] == "regular" and (
            base.endswith(EXEC_EXTS) or base == "conftest.py" or (e["mode_bits"] & 0o111)
        ):
            executable.append(
                {
                    "path": e["path"],
                    "size_bytes": e["size_bytes"],
                    "file_sha256": e["file_sha256"],
                    "mode_bits": e["mode_bits"],
                    "link_count": 1,
                }
            )
    executable.sort(key=lambda e: e["path"])

    root_st = os.lstat(ROOT)
    ck_st = os.lstat(CHECK_OUTPUT)

    baseline = {
        "schema_version": "agu.task0258-module-a-implementation-scope-baseline.v1",
        "module_id": "existing-45-temporal-retrospective",
        "repository_root_absolute_path": ROOT,
        "repository_root_device": root_st.st_dev,
        "repository_root_inode": root_st.st_ino,
        "ordered_root_paths": ["."],
        "check_output_directory_absolute_path": CHECK_OUTPUT,
        "check_output_directory_device": ck_st.st_dev,
        "check_output_directory_inode": ck_st.st_ino,
        "ordered_entry_receipts": entries,
        "ordered_repository_executable_receipts": executable,
        "captured_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    baseline["artifact_sha256"] = sha256_bytes(canon(baseline).encode("utf-8"))

    raw = (json.dumps(baseline, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with open(OUT, "wb") as fh:
        fh.write(raw)

    print("artifact_sha256 =", baseline["artifact_sha256"])
    print("file_sha256     =", sha256_bytes(raw))
    print("entries         =", len(entries))
    print("executables     =", len(executable))
    print("symlink_to_exec =", len(symlink_to_exec))
    for s, t in symlink_to_exec:
        print("  SYMLINK-TO-EXEC:", s, "->", t)
    print("wrote           =", OUT)


if __name__ == "__main__":
    main()
