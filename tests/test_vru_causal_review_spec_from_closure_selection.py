from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts import build_vru_causal_review as review_plan_builder
from scripts.build_vru_causal_closure_selection import (
    build_causal_closure_selection,
    canonical_sha256,
)
from scripts.build_vru_causal_review_spec_from_closure_selection import (
    build_spec_from_closure_selection,
    main,
)

ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_ROOT = ROOT / "analysis_outputs" / "public_research"
SOURCE_MANIFEST = (
    ANNOTATION_ROOT
    / "wikimedia_hctv_hazen_randolph_vtv_pilot_annotation_v35_rv"
    / "source_manifest.json"
)


@pytest.fixture(scope="module")
def real_selection() -> dict[str, Any]:
    return build_causal_closure_selection(
        annotation_root=ANNOTATION_ROOT,
        source_manifest_path=SOURCE_MANIFEST,
        max_version=41,
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_selection(
    tmp_path: Path,
    selection: dict[str, Any],
) -> Path:
    path = tmp_path / "selection.json"
    _write_json(path, selection)
    return path


def _reseal(selection: dict[str, Any]) -> dict[str, Any]:
    selection["artifact_sha256"] = canonical_sha256(selection)
    return selection


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_selection_expands_to_24_neutral_unique_64_frame_specs(
    tmp_path: Path,
    real_selection: dict[str, Any],
) -> None:
    selection_path = _write_selection(tmp_path, real_selection)

    spec = build_spec_from_closure_selection(
        selection_path=selection_path,
        expected_selection_artifact_sha256=real_selection["artifact_sha256"],
        source_manifest_path=SOURCE_MANIFEST,
    )

    assert spec["schema_version"] == "agu.vru-causal-review-spec.v1"
    assert spec["source_selection_artifact_sha256"] == real_selection[
        "artifact_sha256"
    ]
    assert spec["expected_selection_artifact_sha256"] == real_selection[
        "artifact_sha256"
    ]
    assert spec["source_manifest_sha256"] == _file_sha256(SOURCE_MANIFEST)
    assert spec["sample_count"] == 64
    assert spec["sample_period_seconds"] == 0.125
    assert spec["interval_semantics"] == "half_open"
    assert len(spec["examples"]) == 24
    assert [row["review_id"] for row in spec["examples"]] == [
        row["review_id"] for row in real_selection["windows"]
    ]
    for window, example in zip(
        real_selection["windows"], spec["examples"], strict=True
    ):
        indexes = example["frame_indexes"]
        assert example["review_id"].startswith("closure-")
        assert example["clip_id"] == window["clip_id"]
        assert len(indexes) == 64
        assert indexes == sorted(set(indexes))
        assert indexes[32] == window["anchor_frame"]
        assert indexes[-1] < round(
            window["end_seconds"] * window["source_fps"]
        )


def test_expected_selection_sha_is_an_external_required_pin(
    tmp_path: Path,
    real_selection: dict[str, Any],
) -> None:
    selection_path = _write_selection(tmp_path, real_selection)

    with pytest.raises(ValueError, match="expected selection artifact SHA-256"):
        build_spec_from_closure_selection(
            selection_path=selection_path,
            expected_selection_artifact_sha256="0" * 64,
            source_manifest_path=SOURCE_MANIFEST,
        )


def test_output_spec_is_consumable_by_the_existing_review_plan_builder(
    tmp_path: Path,
    real_selection: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection_path = _write_selection(tmp_path, real_selection)
    spec = build_spec_from_closure_selection(
        selection_path=selection_path,
        expected_selection_artifact_sha256=real_selection["artifact_sha256"],
        source_manifest_path=SOURCE_MANIFEST,
    )
    spec_path = tmp_path / "review_spec.json"
    _write_json(spec_path, spec)
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    hashes_by_filename = {
        Path(row["path"]).name: row["sha256"] for row in manifest["videos"]
    }
    safe_root = tmp_path / "safe_manifest_bundle"
    safe_raw = safe_root / "raw"
    safe_raw.mkdir(parents=True)
    for row in manifest["videos"]:
        filename = Path(row["path"]).name
        (safe_raw / filename).write_bytes(b"test source placeholder")
        row["path"] = f"raw/{filename}"
    safe_manifest = safe_root / "source_manifest.json"
    _write_json(safe_manifest, manifest)

    def bound_file_sha256(path: Path) -> str:
        if path == safe_manifest:
            return _file_sha256(path)
        return hashes_by_filename[path.name]

    class _Capture:
        def isOpened(self) -> bool:  # noqa: N802 - mirrors OpenCV's API
            return True

        def release(self) -> None:
            return None

    monkeypatch.setattr(review_plan_builder, "_file_sha256", bound_file_sha256)
    monkeypatch.setattr(
        review_plan_builder.cv2,
        "VideoCapture",
        lambda _path: _Capture(),
    )
    monkeypatch.setattr(
        review_plan_builder,
        "_frame_sha256s",
        lambda _capture, _path, _indexes: {},
    )

    plan = review_plan_builder.build_plan(
        manifest_path=safe_manifest,
        spec_path=spec_path,
    )

    assert len(plan["examples"]) == 24
    assert [row["review_id"] for row in plan["examples"]] == [
        row["review_id"] for row in spec["examples"]
    ]
    assert all(len(row["frame_indexes"]) == 64 for row in plan["examples"])


def test_source_manifest_file_sha_must_match_the_sealed_selection(
    tmp_path: Path,
    real_selection: dict[str, Any],
) -> None:
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    manifest["purpose"] = "tampered"
    manifest_path = tmp_path / "source_manifest.json"
    _write_json(manifest_path, manifest)
    selection_path = _write_selection(tmp_path, real_selection)

    with pytest.raises(ValueError, match="source manifest file SHA-256"):
        build_spec_from_closure_selection(
            selection_path=selection_path,
            expected_selection_artifact_sha256=real_selection["artifact_sha256"],
            source_manifest_path=manifest_path,
        )


def test_each_window_must_match_its_manifest_source_binding(
    tmp_path: Path,
    real_selection: dict[str, Any],
) -> None:
    selection = copy.deepcopy(real_selection)
    selection["windows"][0]["clip_id"] = "wrong_but_path_safe_clip"
    _reseal(selection)
    selection_path = _write_selection(tmp_path, selection)

    with pytest.raises(ValueError, match="window/manifest source binding"):
        build_spec_from_closure_selection(
            selection_path=selection_path,
            expected_selection_artifact_sha256=selection["artifact_sha256"],
            source_manifest_path=SOURCE_MANIFEST,
        )


@pytest.mark.parametrize(
    "unsafe_path",
    ("/etc/passwd", "../../../../../../../../etc/passwd"),
)
def test_manifest_video_paths_must_not_escape_allowed_roots(
    tmp_path: Path,
    real_selection: dict[str, Any],
    unsafe_path: str,
) -> None:
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    manifest["videos"][0]["path"] = unsafe_path
    manifest_path = tmp_path / "source_manifest.json"
    _write_json(manifest_path, manifest)

    selection = copy.deepcopy(real_selection)
    selection["source_manifest_sha256"] = _file_sha256(manifest_path)
    _reseal(selection)
    selection_path = _write_selection(tmp_path, selection)

    with pytest.raises(ValueError, match="path-safe|allowed roots"):
        build_spec_from_closure_selection(
            selection_path=selection_path,
            expected_selection_artifact_sha256=selection["artifact_sha256"],
            source_manifest_path=manifest_path,
        )


def test_cli_writes_a_complete_same_directory_temp_before_replace(
    tmp_path: Path,
    real_selection: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection_path = _write_selection(tmp_path, real_selection)
    output = tmp_path / "review_spec.json"
    observed: list[tuple[Path, Path]] = []
    original_replace = Path.replace

    def observe_replace(source: Path, target: Path) -> Path:
        staged = json.loads(source.read_text(encoding="utf-8"))
        assert staged["schema_version"] == "agu.vru-causal-review-spec.v1"
        assert len(staged["examples"]) == 24
        observed.append((source, target))
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", observe_replace)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_vru_causal_review_spec_from_closure_selection.py",
            "--selection",
            str(selection_path),
            "--expected-selection-artifact-sha256",
            real_selection["artifact_sha256"],
            "--source-manifest",
            str(SOURCE_MANIFEST),
            "--output",
            str(output),
        ],
    )

    assert main() == 0
    assert output.is_file()
    assert len(observed) == 1
    temporary, target = observed[0]
    assert temporary.parent == output.parent
    assert target == output
    assert not temporary.exists()


def test_cli_refuses_to_overwrite_a_verified_input(
    tmp_path: Path,
    real_selection: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection_path = _write_selection(tmp_path, real_selection)
    original = selection_path.read_bytes()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_vru_causal_review_spec_from_closure_selection.py",
            "--selection",
            str(selection_path),
            "--expected-selection-artifact-sha256",
            real_selection["artifact_sha256"],
            "--source-manifest",
            str(SOURCE_MANIFEST),
            "--output",
            str(selection_path),
        ],
    )

    with pytest.raises(ValueError, match="must not overwrite an input"):
        main()
    assert selection_path.read_bytes() == original
