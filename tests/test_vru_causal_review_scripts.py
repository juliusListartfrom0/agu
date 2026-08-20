from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import scripts.build_vru_causal_review as build_vru_causal_review
import scripts.seal_vru_causal_review as seal_vru_causal_review
from scripts.build_vru_causal_review import build_plan
from scripts.materialize_vru_causal_review_frames import (
    _read_frame_sequence,
    _safe_output_path,
)
from scripts.seal_vru_causal_review import DECISION_SCHEMA, seal_review


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 36),
    )
    assert writer.isOpened()
    for index in range(12):
        writer.write(np.full((36, 64, 3), index * 10, dtype=np.uint8))
    writer.release()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sparse_frame_sha256s(path: Path, indexes: list[int]) -> dict[str, str]:
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    hashes: dict[str, str] = {}
    try:
        for index in indexes:
            assert capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            assert ok
            hashes[str(index)] = hashlib.sha256(frame.tobytes()).hexdigest()
    finally:
        capture.release()
    return hashes


def _manifest(video_path: Path) -> dict[str, object]:
    return {
        "schema_version": "agu.vru-basketball-source-manifest.v1",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "videos": [
            {
                "clip_id": "clip-a",
                "path": video_path.name,
                "sha256": _sha256(video_path),
                "fps": 10,
                "frame_count": 12,
            }
        ],
    }


def test_read_frame_sequence_uses_one_seek_and_preserves_sparse_frames() -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.next_index = 0
            self.set_calls: list[int] = []
            self.read_calls = 0

        def set(self, property_id: int, value: int) -> bool:
            assert property_id == cv2.CAP_PROP_POS_FRAMES
            self.set_calls.append(int(value))
            self.next_index = int(value)
            return True

        def read(self) -> tuple[bool, np.ndarray]:
            self.read_calls += 1
            index = self.next_index
            self.next_index += 1
            return True, np.full((2, 3, 1), index, dtype=np.uint8)

    capture = FakeCapture()
    frames = _read_frame_sequence(capture, [2, 5, 8], video_path=Path("clip-a.webm"))

    assert capture.set_calls == [2]
    assert capture.read_calls == 7
    assert [int(frame[0, 0, 0]) for frame in frames] == [2, 5, 8]


def test_build_plan_hashes_exact_source_frames(tmp_path: Path) -> None:
    video_path = tmp_path / "clip-a.avi"
    _write_video(video_path)
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps(_manifest(video_path)), encoding="utf-8")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "examples": [
                    {
                        "review_id": "vru-causal-0001",
                        "clip_id": "clip-a",
                        "frame_indexes": [0, 3, 6],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    plan = build_plan(manifest_path=manifest_path, spec_path=spec_path)

    assert plan["runtime_consumable"] is False
    assert plan["source_manifest_sha256"] == _sha256(manifest_path)
    row = plan["examples"][0]
    assert set(row["frame_sha256s"]) == {"0", "3", "6"}
    assert row["source_video_sha256"] == _sha256(video_path)


def test_build_plan_hashes_shared_source_once_and_preserves_frame_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "clip-a.avi"
    _write_video(video_path)
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps(_manifest(video_path)), encoding="utf-8")
    examples = [
        {
            "review_id": "vru-causal-0001",
            "clip_id": "clip-a",
            "frame_indexes": [0, 3, 6],
        },
        {
            "review_id": "vru-causal-0002",
            "clip_id": "clip-a",
            "frame_indexes": [1, 4, 9],
        },
    ]
    expected_hashes = {row["review_id"]: _sparse_frame_sha256s(video_path, row["frame_indexes"]) for row in examples}
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"examples": examples}), encoding="utf-8")

    hashed_paths: list[Path] = []
    real_file_sha256 = build_vru_causal_review._file_sha256
    real_video_capture = cv2.VideoCapture
    opened_paths: list[Path] = []
    seek_indexes: list[int] = []

    def tracked_file_sha256(path: Path) -> str:
        hashed_paths.append(path.resolve())
        return real_file_sha256(path)

    class TrackedCapture:
        def __init__(self, path: str) -> None:
            opened_paths.append(Path(path).resolve())
            self._capture = real_video_capture(path)

        def isOpened(self) -> bool:
            return self._capture.isOpened()

        def set(self, property_id: int, value: int) -> bool:
            if property_id == cv2.CAP_PROP_POS_FRAMES:
                seek_indexes.append(int(value))
            return self._capture.set(property_id, value)

        def read(self) -> tuple[bool, np.ndarray]:
            return self._capture.read()

        def release(self) -> None:
            self._capture.release()

    monkeypatch.setattr(build_vru_causal_review, "_file_sha256", tracked_file_sha256)
    monkeypatch.setattr(build_vru_causal_review.cv2, "VideoCapture", TrackedCapture)

    plan = build_vru_causal_review.build_plan(manifest_path=manifest_path, spec_path=spec_path)

    assert hashed_paths.count(video_path.resolve()) == 1
    assert hashed_paths.count(manifest_path.resolve()) == 1
    assert opened_paths == [video_path.resolve()]
    assert seek_indexes == [0, 1]
    assert {row["review_id"]: row["frame_sha256s"] for row in plan["examples"]} == expected_hashes


def test_frame_hashing_splits_widely_sparse_indexes_into_bounded_runs() -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.next_index = 0
            self.set_calls: list[int] = []
            self.read_calls = 0

        def set(self, property_id: int, value: int) -> bool:
            assert property_id == cv2.CAP_PROP_POS_FRAMES
            self.set_calls.append(int(value))
            self.next_index = int(value)
            return True

        def read(self) -> tuple[bool, np.ndarray]:
            self.read_calls += 1
            index = self.next_index
            self.next_index += 1
            return True, np.full((2, 3, 1), index % 256, dtype=np.uint8)

    capture = FakeCapture()

    hashes = build_vru_causal_review._frame_sha256s(capture, Path("clip-a.webm"), [0, 9, 18])

    assert set(hashes) == {"0", "9", "18"}
    assert capture.set_calls == [0, 9, 18]
    assert capture.read_calls == 3


@pytest.mark.parametrize("review_id", ("../escape", 7, True, None))
def test_build_plan_rejects_invalid_review_id(tmp_path: Path, review_id: object) -> None:
    video_path = tmp_path / "clip-a.avi"
    _write_video(video_path)
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps(_manifest(video_path)), encoding="utf-8")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "examples": [
                    {
                        "review_id": review_id,
                        "clip_id": "clip-a",
                        "frame_indexes": [0, 3, 6],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="review ID.*path-safe"):
        build_plan(manifest_path=manifest_path, spec_path=spec_path)


@pytest.mark.parametrize("escaped_path", ("../clip-a.avi", "/tmp/clip-a.avi", "C:/clip-a.avi"))
def test_build_plan_rejects_manifest_video_path_escape(
    tmp_path: Path,
    escaped_path: str,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    outside = tmp_path / "clip-a.avi"
    _write_video(outside)
    manifest = _manifest(outside)
    manifest["videos"][0]["path"] = escaped_path  # type: ignore[index]
    manifest_path = bundle / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    spec_path = bundle / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "examples": [
                    {
                        "review_id": "vru-causal-0001",
                        "clip_id": "clip-a",
                        "frame_indexes": [0, 3, 6],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relative|manifest root"):
        build_plan(manifest_path=manifest_path, spec_path=spec_path)


def test_build_plan_rejects_manifest_video_symlink_escape(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    outside = tmp_path / "clip-a.avi"
    _write_video(outside)
    (bundle / outside.name).symlink_to(outside)
    manifest_path = bundle / "source_manifest.json"
    manifest_path.write_text(json.dumps(_manifest(outside)), encoding="utf-8")
    spec_path = bundle / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "examples": [
                    {
                        "review_id": "vru-causal-0001",
                        "clip_id": "clip-a",
                        "frame_indexes": [0, 3, 6],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest root"):
        build_plan(manifest_path=manifest_path, spec_path=spec_path)


@pytest.mark.parametrize("input_name", ("manifest", "spec", "video"))
@pytest.mark.parametrize("alias_kind", ("direct", "symlink", "hardlink", "casefold"))
def test_build_plan_cli_rejects_output_input_alias_before_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_name: str,
    alias_kind: str,
) -> None:
    video = tmp_path / "clip-a.avi"
    _write_video(video)
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(json.dumps(_manifest(video)), encoding="utf-8")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"examples": []}), encoding="utf-8")
    target = {"manifest": manifest, "spec": spec, "video": video}[input_name]
    output = target
    if alias_kind == "symlink":
        output = tmp_path / f"{input_name}-alias.json"
        output.symlink_to(target)
    elif alias_kind == "hardlink":
        output = tmp_path / f"{input_name}-alias.json"
        output.hardlink_to(target)
    elif alias_kind == "casefold":
        output = target.with_name(target.name.upper())
        if output.exists():
            pytest.skip("case-insensitive filesystem already resolves the alias")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_vru_causal_review.py",
            "--manifest",
            str(manifest),
            "--spec",
            str(spec),
            "--output",
            str(output),
        ],
    )
    built = False

    def _unexpected_build(**_kwargs: object) -> dict[str, object]:
        nonlocal built
        built = True
        return {}

    monkeypatch.setattr(build_vru_causal_review, "build_plan", _unexpected_build)
    with pytest.raises(ValueError, match="alias"):
        build_vru_causal_review.main()
    assert built is False


def test_build_plan_atomic_replace_failure_preserves_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "plan.json"
    output.write_text("old\n", encoding="utf-8")

    def _fail_replace(_source: Path, target: Path) -> Path:
        assert target == output
        raise OSError("injected replace failure")

    monkeypatch.setattr(Path, "replace", _fail_replace)
    with pytest.raises(OSError, match="injected"):
        build_vru_causal_review._atomic_write_json(output, {"new": True})
    assert output.read_text(encoding="utf-8") == "old\n"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_materializer_output_path_stays_inside_frame_root(tmp_path: Path) -> None:
    output_dir = tmp_path / "frames"
    output_dir.mkdir()

    with pytest.raises(ValueError, match="outside.*frame root"):
        _safe_output_path(output_dir, Path("../escape.jpg"))

    assert _safe_output_path(output_dir, Path("safe.jpg")) == (output_dir / "safe.jpg").resolve()


def test_seal_review_requires_training_only_decision_provenance(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "clip-a.avi"
    _write_video(video_path)
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps(_manifest(video_path)), encoding="utf-8")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "examples": [
                    {
                        "review_id": "vru-causal-0001",
                        "clip_id": "clip-a",
                        "frame_indexes": [0, 3, 6],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    plan = build_plan(manifest_path=manifest_path, spec_path=spec_path)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(
        json.dumps(
            {
                "schema_version": DECISION_SCHEMA,
                "runtime_consumable": False,
                "codex_runtime_answer_used": False,
                "labels_hidden_from_reviewer": True,
                "plan_sha256": plan["artifact_sha256"],
                "reviews": [
                    {
                        "review_id": "vru-causal-0001",
                        "shot_sequence": "not_a_shot",
                        "release_position": None,
                        "rim_position": None,
                        "outcome": "not_applicable",
                        "confidence": "high",
                        "evidence": {
                            "controlled_ball_before_release": False,
                            "ball_separated_from_hand": False,
                            "ball_progresses_toward_rim": False,
                            "rim_proximity_visible": False,
                            "rim_contact_visible": False,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    review = seal_review(plan_path=plan_path, decisions_path=decisions_path)

    assert review["reviews"][0]["source_video_filename"] == "clip-a.avi"
    assert review["runtime_consumable"] is False
    assert review["plan_sha256"] == plan["artifact_sha256"]

    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    decisions["runtime_consumable"] = True
    decisions_path.write_text(json.dumps(decisions), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        seal_review(plan_path=plan_path, decisions_path=decisions_path)


@pytest.mark.parametrize("input_name", ("plan", "decisions"))
@pytest.mark.parametrize("alias_kind", ("direct", "output_symlink"))
def test_seal_cli_rejects_output_input_alias_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_name: str,
    alias_kind: str,
) -> None:
    input_paths = {
        "plan": tmp_path / "plan.json",
        "decisions": tmp_path / "decisions.json",
    }
    for name, path in input_paths.items():
        path.write_text(f"{name} evidence\n", encoding="utf-8")
    output = input_paths[input_name]
    if alias_kind == "output_symlink":
        output = tmp_path / f"{input_name}-output-alias.json"
        output.symlink_to(input_paths[input_name])
    snapshots = {name: path.read_bytes() for name, path in input_paths.items()}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "seal_vru_causal_review.py",
            "--plan",
            str(input_paths["plan"]),
            "--decisions",
            str(input_paths["decisions"]),
            "--output",
            str(output),
        ],
    )

    def reject_read(_path: Path) -> dict[str, object]:
        raise AssertionError("output/input alias must be rejected before JSON reading")

    monkeypatch.setattr(seal_vru_causal_review, "_read_json", reject_read)

    with pytest.raises(ValueError, match="output.*input"):
        seal_vru_causal_review.main()

    assert {name: path.read_bytes() for name, path in input_paths.items()} == snapshots


def test_seal_json_writer_stages_and_fsyncs_before_atomic_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "sealed.json"
    destination.write_text("old artifact\n", encoding="utf-8")
    writer = getattr(seal_vru_causal_review, "_write_json_atomic", None)
    assert callable(writer)
    original_replace = Path.replace
    replace_observations: list[Path] = []
    fsync_calls: list[int] = []

    def observe_replace(source: Path, target: Path) -> Path:
        assert source.parent == destination.parent
        assert source.suffix == ".tmp"
        assert json.loads(source.read_text(encoding="utf-8")) == {"new": True}
        assert destination.read_text(encoding="utf-8") == "old artifact\n"
        replace_observations.append(source)
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", observe_replace)
    monkeypatch.setattr(
        seal_vru_causal_review.os,
        "fsync",
        lambda file_descriptor: fsync_calls.append(file_descriptor),
    )

    writer(destination, {"new": True})

    assert replace_observations
    assert fsync_calls
    assert json.loads(destination.read_text(encoding="utf-8")) == {"new": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_seal_cli_uses_frozen_resolved_paths_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = tmp_path / "plan.json"
    decisions_path = tmp_path / "decisions.json"
    plan_path.write_text("{}\n", encoding="utf-8")
    decisions_path.write_text("{}\n", encoding="utf-8")
    first_destination = tmp_path / "first-destination"
    second_destination = tmp_path / "second-destination"
    first_destination.mkdir()
    second_destination.mkdir()
    output_parent = tmp_path / "output-parent"
    output_parent.symlink_to(first_destination, target_is_directory=True)
    output = output_parent / "sealed.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "seal_vru_causal_review.py",
            "--plan",
            str(plan_path),
            "--decisions",
            str(decisions_path),
            "--output",
            str(output),
        ],
    )

    def redirect_after_validation(
        *,
        plan_path: Path,
        decisions_path: Path,
        expected_plan_artifact_sha256: str | None,
    ) -> dict[str, object]:
        assert plan_path == plan_path.resolve()
        assert decisions_path == decisions_path.resolve()
        assert expected_plan_artifact_sha256 is None
        output_parent.unlink()
        output_parent.symlink_to(second_destination, target_is_directory=True)
        return {"reviews": [], "artifact_sha256": "a" * 64}

    monkeypatch.setattr(
        seal_vru_causal_review,
        "seal_review",
        redirect_after_validation,
    )

    assert seal_vru_causal_review.main() == 0

    assert json.loads((first_destination / "sealed.json").read_text(encoding="utf-8"))["artifact_sha256"] == "a" * 64
    assert not (second_destination / "sealed.json").exists()
