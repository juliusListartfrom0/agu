"""Tests for the TASK-0258 v2 atomic filesystem primitives."""

from __future__ import annotations

import json

import pytest

from app.analysis.task0258_module_a_v2 import (
    canonical_artifact_sha256,
    compact_canonical_json,
)
from app.analysis.task0258_v2_fs import (
    atomic_write_bytes,
    atomic_write_json,
    publish_no_clobber,
    verify_absent,
)


def test_compact_canonical_json_sorted_and_compact():
    assert compact_canonical_json({"b": 1, "a": [1, 2], "c": None, "d": True}) == '{"a":[1,2],"b":1,"c":null,"d":true}'


def test_canonical_artifact_sha256_stable_and_no_lf():
    payload = {"schema_version": "x", "artifact_sha256": "0" * 64}
    without = {k: v for k, v in payload.items() if k != "artifact_sha256"}
    expected = __import__("hashlib").sha256(compact_canonical_json(without).encode()).hexdigest()
    assert canonical_artifact_sha256(without) == expected
    # deterministic
    assert canonical_artifact_sha256({"b": 1, "a": 2}) == canonical_artifact_sha256({"a": 2, "b": 1})


def test_verify_absent(tmp_path):
    p = tmp_path / "nope"
    verify_absent(p)
    p.write_text("x")
    with pytest.raises(FileExistsError):
        verify_absent(p)


def test_publish_no_clobber(tmp_path):
    staged = tmp_path / "stage"
    staged.write_text("data")
    final = tmp_path / "final"
    publish_no_clobber(staged, final)
    assert final.read_text() == "data"
    assert not staged.exists()
    # second publish must fail
    staged2 = tmp_path / "stage2"
    staged2.write_text("other")
    with pytest.raises(FileExistsError):
        publish_no_clobber(staged2, final)
    assert final.read_text() == "data"  # unchanged


def test_atomic_write_bytes(tmp_path):
    final = tmp_path / "f.json"
    atomic_write_bytes(final, b'{"a":1}\n')
    assert final.read_bytes() == b'{"a":1}\n'
    # no leftover stage files
    assert list(tmp_path.iterdir()) == [final]
    # existing final must not be overwritten
    with pytest.raises(FileExistsError):
        atomic_write_bytes(final, b"clobber\n")
    assert final.read_bytes() == b'{"a":1}\n'


def test_atomic_write_json(tmp_path):
    final = tmp_path / "g.json"
    atomic_write_json(final, {"b": 1, "a": 2})
    assert json.loads(final.read_text()) == {"a": 2, "b": 1}
    # bytes are compact canonical + final LF
    assert final.read_text() == '{"a":2,"b":1}\n'


def test_generation_directory_transaction(tmp_path):
    from pathlib import Path

    from app.analysis.task0258_v2_artifacts import CANDIDATE_MEMBER_PATHS
    from app.analysis.task0258_v2_fs import (
        build_generation_directory,
        publish_generation_directory,
    )

    output_root = tmp_path / "out"
    members = {p: f"bytes-{i}\n".encode() for i, p in enumerate(CANDIDATE_MEMBER_PATHS)}
    staged = build_generation_directory(output_root, members)
    assert staged.name.startswith(".task0258-")

    final = output_root / "candidate_v2"
    lock = output_root / ".lock"
    lock.write_text("")
    publish_generation_directory(staged, final, flock_path=lock)

    # final has exactly the 10 members
    written = sorted(str(Path(r).relative_to(final)) for r in final.rglob("*") if r.is_file())
    assert written == sorted(CANDIDATE_MEMBER_PATHS)

    # second publish must fail (final exists)
    staged2 = build_generation_directory(output_root, {"x.json": b"1"})
    with pytest.raises(FileExistsError):
        publish_generation_directory(staged2, final, flock_path=lock)


def test_seal_generation_directory_coverage(tmp_path):
    from app.analysis.task0258_v2_artifacts import CANDIDATE_MEMBER_PATHS
    from app.analysis.task0258_v2_fs import seal_generation_directory

    root = tmp_path / "out"
    lock = root / ".lock"
    lock.mkdir(parents=True, exist_ok=True)  # placeholder
    flock = root / ".lockf"
    flock.write_text("")
    members = {p: b"x\n" for p in CANDIDATE_MEMBER_PATHS}
    final = seal_generation_directory(root, "candidate_v2", members, CANDIDATE_MEMBER_PATHS, flock_path=flock)
    assert final == root / "candidate_v2"
    assert final.is_dir()
    # missing member fails before publish
    with pytest.raises(ValueError):
        seal_generation_directory(
            root,
            "other",
            {p: b"x\n" for p in CANDIDATE_MEMBER_PATHS[:-1]},
            CANDIDATE_MEMBER_PATHS,
            flock_path=flock,
        )


def test_generation_rejects_traversal_and_symlinked_parent(tmp_path):
    from app.analysis.task0258_v2_fs import build_generation_directory

    with pytest.raises(ValueError):
        build_generation_directory(tmp_path / "out", {"../escape.json": b"x"})
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError):
        build_generation_directory(link, {"member.json": b"x"})
