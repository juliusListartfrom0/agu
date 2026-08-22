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
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path, PurePosixPath
from typing import Iterator

from app.analysis.task0258_module_a_v2 import compact_canonical_json, is_safe_slug

_LOCK_CAPABILITY = object()
_ACTIVE_FLOCKS: ContextVar[dict[str, "FlockHandle"]] = ContextVar("task0258_active_flocks", default={})


class FlockHandle:
    """Process-local capability for a currently-held fixed-path flock."""

    __slots__ = ("_capability", "_fd", "_path", "_parent_identity", "_sealed")

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_sealed" and hasattr(self, "_sealed"):
            raise AttributeError("FlockHandle seal state is immutable")
        if getattr(self, "_sealed", False) and name in {"_capability", "_fd", "_path", "_parent_identity"}:
            raise AttributeError("FlockHandle fields are immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name in {"_capability", "_fd", "_path", "_parent_identity", "_sealed"} and hasattr(self, name):
            raise AttributeError("FlockHandle fields are immutable")
        object.__delattr__(self, name)

    def __init__(
        self,
        path: Path,
        fd: int,
        capability: object,
        *,
        parent_identity: tuple[int, int],
    ) -> None:
        if capability is not _LOCK_CAPABILITY:
            raise TypeError("FlockHandle must be created by exclusive_flock")
        object.__setattr__(self, "_capability", capability)
        object.__setattr__(self, "_fd", fd)
        object.__setattr__(self, "_path", Path(path))
        object.__setattr__(self, "_parent_identity", parent_identity)
        object.__setattr__(self, "_sealed", True)

    def assert_held(self, path: Path, *, directory_fd: int | None = None) -> None:
        if self._capability is not _LOCK_CAPABILITY or self._fd < 0 or _lock_key(self._path) != _lock_key(path):
            raise ValueError("the supplied lock handle does not hold the required path")
        try:
            os.fstat(self._fd)
            if directory_fd is not None:
                directory_stat = os.fstat(directory_fd)
                if (directory_stat.st_dev, directory_stat.st_ino) != self._parent_identity:
                    raise ValueError("the supplied directory descriptor is not the lock parent")
        except OSError as exc:
            raise ValueError("the supplied lock handle is no longer active") from exc

    def _invalidate(self) -> None:
        object.__setattr__(self, "_fd", -1)


def _lock_key(path: Path) -> str:
    return os.path.normpath(os.fspath(Path(path)))


def active_flock(path: Path) -> FlockHandle | None:
    """Return the current context's verified handle for ``path``, if any."""
    handle = _ACTIVE_FLOCKS.get().get(_lock_key(path))
    if handle is not None:
        handle.assert_held(path)
    return handle


def _absolute_path_components(path: Path) -> tuple[str, ...]:
    absolute = os.path.abspath(os.fspath(Path(path)))
    if not absolute.startswith(os.sep):
        raise ValueError("filesystem path is not absolute")
    return tuple(component for component in absolute.split(os.sep)[1:] if component)


def _directory_open_flags() -> int:
    try:
        directory = os.O_DIRECTORY
        no_follow = os.O_NOFOLLOW
    except AttributeError as exc:
        raise OSError(errno.ENOTSUP, "platform lacks required directory no-follow flags") from exc
    return os.O_RDONLY | directory | no_follow | getattr(os, "O_CLOEXEC", 0)


def _open_existing_directory_no_follow(path: Path) -> int:
    """Open every directory component with O_NOFOLLOW and retain the leaf fd."""
    fd = os.open(os.sep, _directory_open_flags())
    try:
        for component in _absolute_path_components(path):
            next_fd = os.open(component, _directory_open_flags(), dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _ensure_directory_no_follow(path: Path, *, mode: int = 0o700) -> None:
    """Create missing directories through descriptor-relative no-follow opens."""
    fd = os.open(os.sep, _directory_open_flags())
    try:
        for component in _absolute_path_components(path):
            try:
                next_fd = os.open(component, _directory_open_flags(), dir_fd=fd)
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode, dir_fd=fd)
                except FileExistsError:
                    pass
                next_fd = os.open(component, _directory_open_flags(), dir_fd=fd)
            os.close(fd)
            fd = next_fd
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(f"directory path cannot be opened without following links: {path}") from exc
        raise
    finally:
        os.close(fd)


def _open_leaf_no_follow(path: Path, flags: int, mode: int = 0) -> int:
    path = Path(path)
    if not path.name or path.name in {".", ".."}:
        raise ValueError("filesystem leaf name is invalid")
    try:
        parent_fd = _open_existing_directory_no_follow(path.parent)
    except OSError as exc:
        raise ValueError(f"filesystem parent cannot be opened without following links: {path.parent}") from exc
    try:
        return os.open(path.name, flags, mode, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)


def _stat_snapshot(file_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        getattr(file_stat, "st_mtime_ns", 0),
        getattr(file_stat, "st_ctime_ns", 0),
    )


def fsync_dir(path: Path) -> None:
    """fsync a directory by its no-follow opened descriptor."""
    fd = _open_existing_directory_no_follow(Path(path))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def _hold_exclusive_flock(path: Path, fd: int, *, parent_identity: tuple[int, int]) -> Iterator[FlockHandle]:
    handle = FlockHandle(path, fd, _LOCK_CAPABILITY, parent_identity=parent_identity)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"lock path is not a regular file: {path}")
        fcntl.flock(fd, fcntl.LOCK_EX)
        active = dict(_ACTIVE_FLOCKS.get())
        active[_lock_key(path)] = handle
        context_token = _ACTIVE_FLOCKS.set(active)
        try:
            yield handle
        finally:
            _ACTIVE_FLOCKS.reset(context_token)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            handle._invalidate()
            os.close(fd)


def _flock_open_flags() -> int:
    return os.O_RDONLY | os.O_CREAT | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)


@contextmanager
def exclusive_flock(path: Path) -> Iterator[FlockHandle]:
    """Create/open and hold a no-follow exclusive flock on a regular file."""
    path = Path(path)
    parent_fd = _open_existing_directory_no_follow(path.parent)
    try:
        parent_stat = os.fstat(parent_fd)
        fd = os.open(path.name, _flock_open_flags(), 0o600, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    with _hold_exclusive_flock(
        path,
        fd,
        parent_identity=(parent_stat.st_dev, parent_stat.st_ino),
    ) as handle:
        yield handle


@contextmanager
def exclusive_flock_at(directory_fd: int, lock_name: str, path: Path) -> Iterator[FlockHandle]:
    """Hold a no-follow exclusive flock opened relative to a stable directory FD."""
    _validate_leaf_name(lock_name)
    directory_stat = os.fstat(directory_fd)
    fd = os.open(lock_name, _flock_open_flags(), 0o600, dir_fd=directory_fd)
    with _hold_exclusive_flock(
        Path(path),
        fd,
        parent_identity=(directory_stat.st_dev, directory_stat.st_ino),
    ) as handle:
        yield handle


def verify_absent(path: Path) -> None:
    """Raise when ``path`` exists (including a dangling symlink)."""
    path = Path(path)
    try:
        parent_fd = _open_existing_directory_no_follow(path.parent)
    except FileNotFoundError:
        return
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(f"target parent cannot be opened without following links: {path.parent}") from exc
        raise
    try:
        try:
            os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise FileExistsError(f"target already exists: {path}")
    finally:
        os.close(parent_fd)


def read_regular_file_no_follow(path: Path, *, maximum_bytes: int = 16_777_216) -> bytes:
    """Read one bounded regular file through an ``O_NOFOLLOW`` descriptor."""
    path = Path(path)
    if not isinstance(maximum_bytes, int) or isinstance(maximum_bytes, bool) or maximum_bytes < 0:
        raise ValueError("maximum_bytes is invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    try:
        fd = _open_leaf_no_follow(path, flags)
    except OSError as exc:
        raise ValueError(f"regular file cannot be opened without following links: {path}") from exc
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > maximum_bytes:
            raise ValueError(f"file is not a bounded regular file: {path}")
        remaining = file_stat.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(fd, min(1 << 20, remaining))
            if not chunk:
                raise ValueError(f"file ended before its recorded size: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        post_read_stat = os.fstat(fd)
        if _stat_snapshot(post_read_stat) != _stat_snapshot(file_stat):
            raise ValueError(f"file changed during bounded read: {path}")
        return b"".join(chunks)
    except OSError as exc:
        raise ValueError(f"regular file cannot be read: {path}") from exc
    finally:
        os.close(fd)


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


def _validate_fixed_stage_name(stage_name: str) -> None:
    if (
        not isinstance(stage_name, str)
        or not stage_name
        or stage_name in {".", ".."}
        or "/" in stage_name
        or "\\" in stage_name
        or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-" for char in stage_name)
    ):
        raise ValueError("fixed generation stage name is invalid")


def _rename_no_clobber_darwin(staged: Path, final: Path, *, directory_fd: int) -> None:
    """Atomic no-clobber rename via macOS ``renameatx_np(RENAME_EXCL)``.

    On success the stage is gone and only ``final`` exists (no crash window).
    On an existing target it raises ``FileExistsError`` and the stage remains.
    """
    # sys/stdio.h: RENAME_SECLUDE=0x1, RENAME_SWAP=0x2, RENAME_EXCL=0x4
    RENAME_EXCL = 0x4
    libc = ctypes.CDLL("libc.dylib", use_errno=True)
    fn = libc.renameatx_np
    fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    fn.restype = ctypes.c_int
    rc = fn(directory_fd, os.fsencode(staged.name), directory_fd, os.fsencode(final.name), RENAME_EXCL)
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
    staged = Path(staged)
    final = Path(final)
    if staged.parent != final.parent:
        raise ValueError("no-clobber publication requires sibling paths")
    parent_fd = _open_existing_directory_no_follow(staged.parent)
    try:
        if sys.platform == "darwin":
            _rename_no_clobber_darwin(staged, final, directory_fd=parent_fd)
        else:
            os.link(staged.name, final.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd, follow_symlinks=False)
            os.unlink(staged.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


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
    _ensure_directory_no_follow(parent)
    if not isinstance(data, bytes):
        raise TypeError("atomic_write_bytes data must be bytes")
    if stage_path is None:
        stage = parent / f".{final.name}.{secrets.token_hex(16)}.stage"
    else:
        stage = Path(stage_path)
        if stage.parent != parent or not stage.name or stage.name in {".", ".."}:
            raise ValueError("fixed atomic-write stage must be a sibling of the final path")
        stage_parent_fd = _open_existing_directory_no_follow(stage.parent)
        os.close(stage_parent_fd)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    fd = _open_leaf_no_follow(stage, flags, mode)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        try:
            _unlink_no_follow(stage)
        except OSError:
            pass
        raise
    try:
        publish_no_clobber(stage, final)
    except BaseException:
        try:
            _unlink_no_follow(stage)
        except OSError:
            pass
        raise
    fsync_dir(parent)


def _validate_leaf_name(name: str) -> None:
    if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("filesystem leaf name is invalid")


def _publish_no_clobber_at(directory_fd: int, staged_name: str, final_name: str) -> None:
    """Publish two sibling names through one already-open directory FD."""
    _validate_leaf_name(staged_name)
    _validate_leaf_name(final_name)
    if sys.platform == "darwin":
        _rename_no_clobber_darwin(Path(staged_name), Path(final_name), directory_fd=directory_fd)
    else:
        os.link(staged_name, final_name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd, follow_symlinks=False)
        os.unlink(staged_name, dir_fd=directory_fd)
    os.fsync(directory_fd)


def atomic_write_bytes_at(
    directory_fd: int,
    final_name: str,
    data: bytes,
    *,
    mode: int = 0o600,
) -> None:
    """Atomically publish one new file relative to a stable directory FD."""
    _validate_leaf_name(final_name)
    if not isinstance(data, bytes):
        raise TypeError("atomic_write_bytes data must be bytes")
    stage_name = f".{final_name}.{secrets.token_hex(16)}.stage"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(stage_name, flags, mode, dir_fd=directory_fd)
    except OSError:
        raise
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        try:
            os.unlink(stage_name, dir_fd=directory_fd)
        except OSError:
            pass
        raise
    try:
        _publish_no_clobber_at(directory_fd, stage_name, final_name)
    except BaseException:
        try:
            os.unlink(stage_name, dir_fd=directory_fd)
        except OSError:
            pass
        raise


def atomic_write_json_at(directory_fd: int, final_name: str, payload: object, *, mode: int = 0o600) -> None:
    """Atomically publish compact-canonical JSON relative to a directory FD."""
    data = (compact_canonical_json(payload) + "\n").encode("utf-8")
    atomic_write_bytes_at(directory_fd, final_name, data, mode=mode)


def _unlink_no_follow(path: Path) -> None:
    parent_fd = _open_existing_directory_no_follow(Path(path).parent)
    try:
        os.unlink(Path(path).name, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)


def atomic_write_json(final: Path, payload: object, *, mode: int = 0o600) -> None:
    """Atomically write compact-canonical JSON + final LF to ``final``."""
    data = (compact_canonical_json(payload) + "\n").encode("utf-8")
    atomic_write_bytes(final, data, mode=mode)


def build_generation_directory(
    parent: Path,
    members: dict[str, bytes],
    *,
    stage_name: str | None = None,
    parent_fd: int | None = None,
) -> Path:
    """Create a private staging directory and write every member.

    ``members`` maps relative POSIX paths to exact bytes. The stage directory is
    named ``.task0258-<nonce>`` under ``parent`` unless ``stage_name`` supplies
    a fixed transaction name, and is created with mode 0o700. Every member is
    written with :func:`atomic_write_bytes` (mode 0o600). Returns the stage
    directory path; the caller is responsible for publishing it.
    """
    owns_parent_fd = parent_fd is None
    if owns_parent_fd:
        _ensure_directory_no_follow(parent)
        parent_fd = _open_existing_directory_no_follow(parent)
    else:
        parent_stat = os.fstat(parent_fd)
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise ValueError("generation parent descriptor is not a directory")
    for relpath, data in members.items():
        _validate_member_path(relpath)
        if not isinstance(data, bytes):
            raise TypeError("generation member data must be bytes")
    if stage_name is not None:
        _validate_fixed_stage_name(stage_name)
    stage = parent / (stage_name or f".task0258-{secrets.token_hex(8)}")
    try:
        assert parent_fd is not None
        os.mkdir(stage.name, 0o700, dir_fd=parent_fd)
    finally:
        if owns_parent_fd:
            os.close(parent_fd)
    try:
        for relpath, data in members.items():
            target = stage / relpath
            _ensure_directory_no_follow(target.parent)
            atomic_write_bytes(target, data, mode=0o600)
    except BaseException:
        import shutil

        shutil.rmtree(stage, ignore_errors=True)
        raise
    return stage


def _verify_absent_at(directory_fd: int, name: str) -> None:
    _validate_leaf_name(name)
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise FileExistsError(f"target already exists: {name}")


def _publish_generation_directory_locked(staged: Path, final: Path) -> None:
    verify_absent(final)
    if sys.platform != "darwin":
        # A portable directory rename has no no-clobber primitive. Refuse
        # the trust-boundary operation instead of silently relying on the
        # cooperative flock against an unrelated publisher.
        raise OSError("atomic no-clobber directory publication is unavailable on this platform")
    publish_no_clobber(staged, final)


def _publish_generation_directory_locked_at(directory_fd: int, staged_name: str, final_name: str) -> None:
    """Publish a generation using one stable parent descriptor."""
    _verify_absent_at(directory_fd, final_name)
    if sys.platform != "darwin":
        raise OSError("atomic no-clobber directory publication is unavailable on this platform")
    _publish_no_clobber_at(directory_fd, staged_name, final_name)


def publish_generation_directory(
    staged: Path,
    final: Path,
    *,
    flock_path: Path,
    under_lock_validator: Callable[[], None] | None = None,
) -> None:
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
    if staged.parent != final.parent:
        raise ValueError("staged and final generations must share one parent")
    _ensure_directory_no_follow(final.parent)
    parent_fd = _open_existing_directory_no_follow(final.parent)
    lock_parent_fd = _open_existing_directory_no_follow(Path(flock_path).parent)
    try:
        with exclusive_flock_at(lock_parent_fd, Path(flock_path).name, flock_path):
            if under_lock_validator is not None:
                under_lock_validator()
            _publish_generation_directory_locked_at(parent_fd, staged.name, final.name)
    finally:
        os.close(lock_parent_fd)
        os.close(parent_fd)


def verify_generation_directory(final: Path, expected_paths: tuple[str, ...]) -> None:
    """Re-open a published generation and verify exact no-symlink coverage."""
    final = Path(final)
    expected_files = {PurePosixPath(path) for path in expected_paths}
    expected_dirs = {PurePosixPath(".")}
    for path in expected_files:
        expected_dirs.update(PurePosixPath(*path.parts[:index]) for index in range(1, len(path.parts)))
    actual_files: set[PurePosixPath] = set()
    actual_dirs: set[PurePosixPath] = {PurePosixPath(".")}
    try:
        root_fd = _open_existing_directory_no_follow(final)
    except OSError as exc:
        raise ValueError("published generation is not a real directory") from exc

    def walk(directory_fd: int, relative_parent: PurePosixPath) -> None:
        try:
            names = os.listdir(directory_fd)
        except OSError as exc:
            raise ValueError("published generation cannot be listed safely") from exc
        for name in names:
            relative = PurePosixPath(name) if relative_parent == PurePosixPath(".") else relative_parent / name
            try:
                entry_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                raise ValueError(f"published generation member cannot be inspected: {relative}") from exc
            if stat.S_ISLNK(entry_stat.st_mode):
                raise ValueError(f"published generation contains a symlink: {relative}")
            if stat.S_ISDIR(entry_stat.st_mode):
                actual_dirs.add(relative)
                child_fd = os.open(name, _directory_open_flags(), dir_fd=directory_fd)
                try:
                    walk(child_fd, relative)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(entry_stat.st_mode):
                actual_files.add(relative)
            else:
                raise ValueError(f"published generation contains a non-regular member: {relative}")

    try:
        walk(root_fd, PurePosixPath("."))
    finally:
        os.close(root_fd)
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
    stage_name: str | None = None,
    pre_publish_validator: Callable[[], None] | None = None,
    held_lock: FlockHandle | None = None,
    parent_fd: int | None = None,
) -> Path:
    """Validate exact member coverage and atomically publish a generation.

    ``members``' relative-path set must equal ``expected_paths`` (no missing,
    extra, renamed, or duplicate member). Then builds and publishes the
    generation directory under ``parent`` as ``final_name``. When supplied,
    ``stage_name`` is rejected if it already exists and is checked for absence
    after publication. Returns the final path.
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
    if held_lock is not None and not isinstance(held_lock, FlockHandle):
        raise TypeError("held_lock must be an active FlockHandle")
    parent = Path(parent)
    owns_parent_fd = parent_fd is None
    if owns_parent_fd:
        _ensure_directory_no_follow(parent)
        parent_fd = _open_existing_directory_no_follow(parent)
    else:
        parent_stat = os.fstat(parent_fd)
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise ValueError("generation parent descriptor is not a directory")
    lock_parent_fd = _open_existing_directory_no_follow(Path(flock_path).parent)
    final = parent / final_name
    try:
        if held_lock is None:
            held_lock = active_flock(flock_path)
        if held_lock is not None:
            held_lock.assert_held(flock_path, directory_fd=lock_parent_fd)
        lock_context = (
            nullcontext(held_lock)
            if held_lock is not None
            else exclusive_flock_at(lock_parent_fd, Path(flock_path).name, flock_path)
        )
        with lock_context as _active_lock:
            if pre_publish_validator is not None:
                pre_publish_validator()
            staged = build_generation_directory(parent, members, stage_name=stage_name, parent_fd=parent_fd)
            try:
                _publish_generation_directory_locked_at(parent_fd, staged.name, final_name)
                verify_generation_directory(final, expected_paths)
                if stage_name is not None and (staged.exists() or staged.is_symlink()):
                    raise ValueError("fixed generation stage remained after publication")
            except BaseException:
                if staged.exists() or staged.is_symlink():
                    import shutil

                    shutil.rmtree(staged, ignore_errors=True)
                raise
        return final
    finally:
        os.close(lock_parent_fd)
        if owns_parent_fd:
            os.close(parent_fd)


__all__ = [
    "FlockHandle",
    "active_flock",
    "fsync_dir",
    "exclusive_flock",
    "exclusive_flock_at",
    "verify_absent",
    "read_regular_file_no_follow",
    "publish_no_clobber",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_bytes_at",
    "atomic_write_json_at",
    "build_generation_directory",
    "publish_generation_directory",
    "verify_generation_directory",
    "seal_generation_directory",
]
