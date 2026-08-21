"""TASK-0258 Amendment-001 v2 atomic filesystem publication primitives.

Self-contained, no-clobber atomic publication helpers used by the candidate,
failure, result, and run-history-registry transactions. These implement the
spec's publication discipline: write to a private stage, fsync, publish with a
no-clobber same-filesystem rename, then fsync the parent directory.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
import secrets
import stat
import sys
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from app.analysis.task0258_module_a_v2 import compact_canonical_json, is_safe_slug


def fsync_dir(path: Path) -> None:
    """fsync a directory by its no-follow opened descriptor."""
    _verify_no_symlink_ancestors(path)
    fd = os.open(
        os.fspath(path),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def exclusive_flock(path: Path):
    """Create/open and hold a no-follow exclusive flock on a regular file."""
    _verify_no_symlink_ancestors(path.parent)
    flags = (
        os.O_RDONLY
        | os.O_CREAT
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    fd = os.open(os.fspath(path), flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"lock path is not a regular file: {path}")
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def verify_absent(path: Path) -> None:
    """Raise when ``path`` exists (including a dangling symlink)."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"target already exists: {path}")


def _verify_no_symlink_ancestors(path: Path) -> None:
    """Reject symlinked path components before a publication transaction."""
    current = Path(path.anchor) if path.is_absolute() else Path()
    for part in path.parts[1:] if path.is_absolute() else path.parts:
        current /= part
        try:
            if current.is_symlink():
                raise ValueError(f"symlinked publication path component: {current}")
        except OSError as exc:
            raise ValueError(f"cannot inspect publication path component: {current}") from exc


def _validate_member_path(relative_path: str) -> None:
    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path or "//" in relative_path:
        raise ValueError("generation member path must be a non-empty POSIX relative path")
    posix = PurePosixPath(relative_path)
    if posix.is_absolute() or any(part in ("", ".", "..") for part in posix.parts):
        raise ValueError(f"generation member path escapes the generation root: {relative_path!r}")


def _rename_no_clobber_darwin(staged: Path, final: Path) -> None:
    """Atomic no-clobber rename via macOS ``renameatx_np(RENAME_EXCL)``.

    On success the stage is gone and only ``final`` exists (no crash window).
    On an existing target it raises ``FileExistsError`` and the stage remains.
    """
    # sys/stdio.h: RENAME_SECLUDE=0x1, RENAME_SWAP=0x2, RENAME_EXCL=0x4
    RENAME_EXCL = 0x4
    AT_FDCWD = -2
    libc = ctypes.CDLL("libc.dylib", use_errno=True)
    fn = libc.renameatx_np
    fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    fn.restype = ctypes.c_int
    rc = fn(AT_FDCWD, os.fsencode(staged), AT_FDCWD, os.fsencode(final), RENAME_EXCL)
    if rc != 0:
        err = ctypes.get_errno()
        if err == errno.EEXIST:
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(final))
        raise OSError(err, os.strerror(err), str(final))


def publish_no_clobber(staged: Path, final: Path) -> None:
    """Atomically publish ``staged`` to ``final`` without overwriting.

    On macOS this is ``renameatx_np(RENAME_EXCL)`` — an atomic no-clobber
    rename: ``final`` either appears exactly once or the stage is left intact.
    On other platforms it falls back to hardlink+unlink.
    """
    if sys.platform == "darwin":
        _rename_no_clobber_darwin(staged, final)
        return
    os.link(staged, final)
    os.unlink(staged)


def atomic_write_bytes(
    final: Path,
    data: bytes,
    *,
    mode: int = 0o600,
    stage_path: Path | None = None,
) -> None:
    """Write ``data`` to ``final`` atomically and without overwriting an existing file.

    Writes a private stage in the same directory, fsyncs it, publishes with
    ``publish_no_clobber``, then fsyncs the parent directory. Callers that have
    an externally frozen transaction name may supply ``stage_path``; it must be
    in the final file's parent and is still created with no-follow,
    no-clobber semantics.
    """
    parent = final.parent
    _verify_no_symlink_ancestors(parent)
    if not isinstance(data, bytes):
        raise TypeError("atomic_write_bytes data must be bytes")
    if stage_path is None:
        stage = parent / f".{final.name}.{secrets.token_hex(16)}.stage"
    else:
        stage = Path(stage_path)
        if stage.parent != parent or not stage.name or stage.name in {".", ".."}:
            raise ValueError("fixed atomic-write stage must be a sibling of the final path")
        _verify_no_symlink_ancestors(stage.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(os.fspath(stage), flags, mode)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        try:
            os.unlink(stage)
        except OSError:
            pass
        raise
    try:
        publish_no_clobber(stage, final)
    except BaseException:
        try:
            os.unlink(stage)
        except OSError:
            pass
        raise
    fsync_dir(parent)


def atomic_write_json(final: Path, payload: object, *, mode: int = 0o600) -> None:
    """Atomically write compact-canonical JSON + final LF to ``final``."""
    data = (compact_canonical_json(payload) + "\n").encode("utf-8")
    atomic_write_bytes(final, data, mode=mode)


def build_generation_directory(parent: Path, members: dict[str, bytes]) -> Path:
    """Create a private staging directory and write every member.

    ``members`` maps relative POSIX paths to exact bytes. The stage directory is
    named ``.task0258-<nonce>`` under ``parent`` and is created with mode 0o700.
    Every member is written with :func:`atomic_write_bytes` (mode 0o600). Returns
    the stage directory path; the caller is responsible for publishing it.
    """
    _verify_no_symlink_ancestors(parent)
    parent.mkdir(parents=True, exist_ok=True)
    _verify_no_symlink_ancestors(parent)
    for relpath, data in members.items():
        _validate_member_path(relpath)
        if not isinstance(data, bytes):
            raise TypeError("generation member data must be bytes")
    stage = parent / f".task0258-{secrets.token_hex(8)}"
    stage.mkdir(mode=0o700)
    try:
        for relpath, data in members.items():
            target = stage / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            _verify_no_symlink_ancestors(target.parent)
            atomic_write_bytes(target, data, mode=0o600)
    except BaseException:
        import shutil

        shutil.rmtree(stage, ignore_errors=True)
        raise
    return stage


def publish_generation_directory(staged: Path, final: Path, *, flock_path: Path) -> None:
    """Atomically publish a staged generation directory to ``final``.

    Under an exclusive flock on ``flock_path``, verifies ``final`` is absent then
    renames the stage (same-filesystem, atomic). The flock serializes concurrent
    publishers so the absence check is not TOCTOU-raced within this process
    family. Fsyncs the parent afterwards.
    """
    if not staged.is_dir() or staged.is_symlink():
        raise ValueError("staged generation must be a real directory")
    if not final.name or not is_safe_slug(final.name):
        raise ValueError("final generation name must be a safe slug")
    _verify_no_symlink_ancestors(staged)
    _verify_no_symlink_ancestors(final.parent)
    final.parent.mkdir(parents=True, exist_ok=True)
    _verify_no_symlink_ancestors(final.parent)
    with exclusive_flock(flock_path):
        verify_absent(final)
        if sys.platform == "darwin":
            _rename_no_clobber_darwin(staged, final)
        else:
            # A portable directory rename has no no-clobber primitive. Refuse
            # the trust-boundary operation instead of silently relying on the
            # cooperative flock against an unrelated publisher.
            raise OSError("atomic no-clobber directory publication is unavailable on this platform")
        fsync_dir(final.parent)


def verify_generation_directory(final: Path, expected_paths: tuple[str, ...]) -> None:
    """Re-open a published generation and verify exact no-symlink coverage."""
    if final.is_symlink() or not final.is_dir():
        raise ValueError("published generation is not a real directory")
    expected_files = {PurePosixPath(path) for path in expected_paths}
    expected_dirs = {PurePosixPath(".")}
    for path in expected_files:
        expected_dirs.update(PurePosixPath(*path.parts[:index]) for index in range(1, len(path.parts)))
    actual_files: set[PurePosixPath] = set()
    actual_dirs: set[PurePosixPath] = {PurePosixPath(".")}
    for entry in final.rglob("*"):
        relative = PurePosixPath(entry.relative_to(final).as_posix())
        if entry.is_symlink():
            raise ValueError(f"published generation contains a symlink: {relative}")
        if entry.is_dir():
            actual_dirs.add(relative)
        elif entry.is_file():
            actual_files.add(relative)
        else:
            raise ValueError(f"published generation contains a non-regular member: {relative}")
    if actual_files != expected_files or actual_dirs != expected_dirs:
        raise ValueError(
            f"published generation coverage drifted: expected_files={sorted(expected_files)!r} "
            f"actual_files={sorted(actual_files)!r} expected_dirs={sorted(expected_dirs)!r} "
            f"actual_dirs={sorted(actual_dirs)!r}"
        )


def seal_generation_directory(
    parent: Path,
    final_name: str,
    members: dict[str, bytes],
    expected_paths: tuple[str, ...],
    *,
    flock_path: Path,
) -> Path:
    """Validate exact member coverage and atomically publish a generation.

    ``members``' relative-path set must equal ``expected_paths`` (no missing,
    extra, renamed, or duplicate member). Then builds and publishes the
    generation directory under ``parent`` as ``final_name``. Returns the final
    path.
    """
    if not is_safe_slug(final_name):
        raise ValueError("generation final name must be a safe slug")
    if set(members) != set(expected_paths):
        raise ValueError(
            f"generation member coverage mismatch: "
            f"missing={sorted(set(expected_paths) - set(members))} "
            f"extra={sorted(set(members) - set(expected_paths))}"
        )
    for relpath in members:
        _validate_member_path(relpath)
    staged = build_generation_directory(parent, members)
    final = parent / final_name
    publish_generation_directory(staged, final, flock_path=flock_path)
    verify_generation_directory(final, expected_paths)
    return final


__all__ = [
    "fsync_dir",
    "exclusive_flock",
    "verify_absent",
    "publish_no_clobber",
    "atomic_write_bytes",
    "atomic_write_json",
    "build_generation_directory",
    "publish_generation_directory",
    "verify_generation_directory",
    "seal_generation_directory",
]
