from __future__ import annotations

import copy
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from app.analysis.independent_shot_vlm import (
    canonical_sha256,
    seal_independent_shot_vlm_plan,
)
from app.analysis.shot_validity_scene_state import (
    PHASE_FRACTIONS,
    seal_scene_embedding_artifact,
    verify_scene_embedding_artifact,
)
from app.analysis.shot_validity_video_backbone import (
    get_video_backbone_spec,
    seal_video_embedding_artifact,
    verify_video_embedding_artifact,
)
from scripts import subset_shot_validity_embeddings_by_plan as subset_module
from scripts.subset_shot_validity_embeddings_by_plan import (
    _write_json,
    subset_embedding_artifacts_by_plan,
    subset_embedding_files_by_plan,
)

TRAINING_MANIFEST_SHA256 = "f" * 64
GAME_A_SHA256 = "a" * 64
GAME_B_SHA256 = "b" * 64
GAME_C_SHA256 = "c" * 64
BUNDLE_A_SHA256 = "1" * 64
BUNDLE_B_SHA256 = "2" * 64
BUNDLE_C_SHA256 = "3" * 64

KEY_A = (GAME_A_SHA256, BUNDLE_A_SHA256, "event-a")
KEY_B = (GAME_B_SHA256, BUNDLE_B_SHA256, "event-b")
KEY_C = (GAME_C_SHA256, BUNDLE_C_SHA256, "event-c-extra")


def _key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["source_video_sha256"]),
        str(row["candidate_bundle_sha256"]),
        str(row["event_id"]),
    )


def _plan(*, keys: Sequence[tuple[str, str, str]] = (KEY_B, KEY_A)) -> dict[str, Any]:
    source_count = len({source_sha256 for source_sha256, _bundle, _event in keys})
    return seal_independent_shot_vlm_plan(
        {
            "training_manifest_sha256": TRAINING_MANIFEST_SHA256,
            "selection": {
                "method": "sha256_rank_within_video_and_boolean_class",
                "positive_per_video": 1,
                "negative_per_video": 0,
                "video_count": source_count,
                "example_count": len(keys),
            },
            "input_contract": {
                "raw_frames_only": True,
                "labels_or_review_notes_exposed_to_model": False,
                "chronological_even_sampling": True,
                "max_frames": 6,
                "image_width": 704,
            },
            "fusion_screen_contract": {},
            "examples": [
                {
                    "source_video_sha256": source_sha256,
                    "source_video_filename": f"{source_sha256}.mp4",
                    "candidate_bundle_sha256": bundle_sha256,
                    "event_id": event_id,
                    "start_frame": index * 100,
                    "end_frame": index * 100 + 99,
                    "source_fps": 30.0,
                }
                for index, (source_sha256, bundle_sha256, event_id) in enumerate(keys)
            ],
        }
    )


def _scene_artifact(
    *,
    keys: Sequence[tuple[str, str, str]] = (KEY_A, KEY_C, KEY_B),
    manifest_sha256: str = TRAINING_MANIFEST_SHA256,
) -> dict[str, Any]:
    return seal_scene_embedding_artifact(
        {
            "purpose": "scene_state_screening_training_only",
            "training_manifest_sha256": manifest_sha256,
            "backbone": "torchvision/mobilenet_v3_small/imagenet1k_v1",
            "backbone_sha256": "d" * 64,
            "embedding_dimension": 576,
            "phase_fractions": list(PHASE_FRACTIONS),
            "examples": [
                {
                    "source_video_sha256": source_sha256,
                    "candidate_bundle_sha256": bundle_sha256,
                    "event_id": event_id,
                    "event_present": event_id != "event-b",
                    "phase_embeddings": [[float(index)] * 576 for _ in PHASE_FRACTIONS],
                }
                for index, (source_sha256, bundle_sha256, event_id) in enumerate(keys, start=1)
            ],
        }
    )


def _video_artifact(
    *,
    keys: Sequence[tuple[str, str, str]] = (KEY_C, KEY_B, KEY_A),
    manifest_sha256: str = TRAINING_MANIFEST_SHA256,
) -> dict[str, Any]:
    backbone = "torchvision/mvit_v2_s/kinetics400_v1"
    dimension = get_video_backbone_spec(backbone).embedding_dimension
    return seal_video_embedding_artifact(
        {
            "purpose": "backbone_screening_training_only",
            "training_manifest_sha256": manifest_sha256,
            "backbone": backbone,
            "backbone_sha256": "e" * 64,
            "embedding_dimension": dimension,
            "clip_frames": 16,
            "examples": [
                {
                    "source_video_sha256": source_sha256,
                    "candidate_bundle_sha256": bundle_sha256,
                    "event_id": event_id,
                    "event_present": event_id != "event-b",
                    "embedding": [float(index)] * dimension,
                }
                for index, (source_sha256, bundle_sha256, event_id) in enumerate(keys, start=1)
            ],
        }
    )


def _reseal_scene(artifact: Mapping[str, Any], *, examples: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        key: copy.deepcopy(value) for key, value in artifact.items() if key not in {"artifact_sha256", "examples"}
    }
    payload["examples"] = examples
    return seal_scene_embedding_artifact(payload)


def _reseal_video(artifact: Mapping[str, Any], *, examples: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        key: copy.deepcopy(value) for key, value in artifact.items() if key not in {"artifact_sha256", "examples"}
    }
    payload["examples"] = examples
    return seal_video_embedding_artifact(payload)


def test_subset_reorders_every_artifact_to_plan_and_seals_provenance() -> None:
    plan = _plan()
    scene = _scene_artifact()
    video = _video_artifact()

    result = subset_embedding_artifacts_by_plan(plan=plan, scene=scene, videos=[video])

    expected_keys = [KEY_B, KEY_A]
    subset_scene = result["scene"]
    subset_video = result["videos"][0]
    assert [_key(row) for row in subset_scene["examples"]] == expected_keys
    assert [_key(row) for row in subset_video["examples"]] == expected_keys
    assert subset_scene["plan_sha256"] == plan["plan_sha256"]
    assert subset_video["plan_sha256"] == plan["plan_sha256"]
    assert subset_scene["source_embedding_artifact_sha256"] == scene["artifact_sha256"]
    assert subset_video["source_embedding_artifact_sha256"] == video["artifact_sha256"]
    assert subset_scene["runtime_consumable"] is False
    assert subset_video["runtime_consumable"] is False
    assert verify_scene_embedding_artifact(subset_scene) == subset_scene
    assert verify_video_embedding_artifact(subset_video) == subset_video


@pytest.mark.parametrize("artifact_name", ["scene", "video"])
def test_subset_rejects_a_plan_key_missing_from_an_embedding_artifact(artifact_name: str) -> None:
    plan = _plan()
    scene = _scene_artifact()
    video = _video_artifact()
    if artifact_name == "scene":
        scene = _scene_artifact(keys=(KEY_A, KEY_C))
    else:
        video = _video_artifact(keys=(KEY_A, KEY_C))

    with pytest.raises(ValueError, match=rf"{artifact_name}.*missing.*event-b"):
        subset_embedding_artifacts_by_plan(plan=plan, scene=scene, videos=[video])


@pytest.mark.parametrize("artifact_name", ["scene", "video"])
def test_subset_rejects_duplicate_embedding_keys(artifact_name: str) -> None:
    plan = _plan()
    scene = _scene_artifact()
    video = _video_artifact()
    if artifact_name == "scene":
        examples = copy.deepcopy(scene["examples"])
        examples.append(copy.deepcopy(examples[0]))
        scene = _reseal_scene(scene, examples=examples)
    else:
        examples = copy.deepcopy(video["examples"])
        examples.append(copy.deepcopy(examples[0]))
        video = _reseal_video(video, examples=examples)

    with pytest.raises(ValueError, match=rf"{artifact_name}.*duplicate"):
        subset_embedding_artifacts_by_plan(plan=plan, scene=scene, videos=[video])


def test_subset_rejects_duplicate_plan_keys_even_when_hash_is_resealed() -> None:
    plan = _plan()
    duplicate = copy.deepcopy(plan)
    duplicate["examples"].append(copy.deepcopy(duplicate["examples"][0]))
    duplicate["plan_sha256"] = canonical_sha256(duplicate, hash_field="plan_sha256")

    with pytest.raises(ValueError, match="plan contains duplicate examples"):
        subset_embedding_artifacts_by_plan(
            plan=duplicate,
            scene=_scene_artifact(),
            videos=[_video_artifact()],
        )


@pytest.mark.parametrize(
    "key",
    (
        ("not-a-sha", BUNDLE_A_SHA256, "event-a"),
        (GAME_A_SHA256, "not-a-sha", "event-a"),
    ),
)
def test_subset_rejects_non_sha_three_part_key_bindings(
    key: tuple[str, str, str],
) -> None:
    plan = _plan(keys=(key,))
    scene = _scene_artifact(keys=(key,))
    video = _video_artifact(keys=(key,))

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        subset_embedding_artifacts_by_plan(plan=plan, scene=scene, videos=[video])


def test_subset_rejects_int_string_event_id_alias() -> None:
    plan_key = (GAME_A_SHA256, BUNDLE_A_SHA256, 1)
    artifact_key = (GAME_A_SHA256, BUNDLE_A_SHA256, "1")
    plan = _plan(keys=(plan_key,))

    with pytest.raises(ValueError, match="key fields must be strings"):
        subset_embedding_artifacts_by_plan(
            plan=plan,
            scene=_scene_artifact(keys=(artifact_key,)),
            videos=[_video_artifact(keys=(artifact_key,))],
        )


def test_subset_rejects_label_exposed_frozen_plan() -> None:
    plan = _plan()
    payload = copy.deepcopy(plan)
    payload.pop("plan_sha256")
    payload["input_contract"][
        "labels_or_review_notes_exposed_to_model"
    ] = True
    with pytest.raises(ValueError, match="unsafe provenance"):
        seal_independent_shot_vlm_plan(payload)


@pytest.mark.parametrize("artifact_name", ["scene", "video"])
def test_subset_rejects_training_manifest_mismatch(artifact_name: str) -> None:
    scene = _scene_artifact(manifest_sha256="0" * 64 if artifact_name == "scene" else TRAINING_MANIFEST_SHA256)
    video = _video_artifact(manifest_sha256="0" * 64 if artifact_name == "video" else TRAINING_MANIFEST_SHA256)

    with pytest.raises(ValueError, match=rf"{artifact_name}.*training manifest"):
        subset_embedding_artifacts_by_plan(plan=_plan(), scene=scene, videos=[video])


def test_subset_rejects_scene_video_label_disagreement() -> None:
    video = _video_artifact()
    examples = copy.deepcopy(video["examples"])
    row = next(row for row in examples if _key(row) == KEY_A)
    row["event_present"] = False
    video = _reseal_video(video, examples=examples)

    with pytest.raises(ValueError, match="labels disagree.*event-a"):
        subset_embedding_artifacts_by_plan(
            plan=_plan(),
            scene=_scene_artifact(),
            videos=[video],
        )


def test_cli_writes_verified_subsets_and_reports_hashes(tmp_path: Path) -> None:
    plan = _plan()
    scene = _scene_artifact()
    video = _video_artifact()
    plan_path = tmp_path / "plan.json"
    scene_input = tmp_path / "scene-input.json"
    video_input = tmp_path / "video-input.json"
    scene_output = tmp_path / "scene-output.json"
    video_output = tmp_path / "video-output.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    scene_input.write_text(json.dumps(scene), encoding="utf-8")
    video_input.write_text(json.dumps(video), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/subset_shot_validity_embeddings_by_plan.py",
            "--plan",
            str(plan_path),
            "--scene-input",
            str(scene_input),
            "--scene-output",
            str(scene_output),
            "--video-pair",
            str(video_input),
            str(video_output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)
    written_scene = verify_scene_embedding_artifact(json.loads(scene_output.read_text(encoding="utf-8")))
    written_video = verify_video_embedding_artifact(json.loads(video_output.read_text(encoding="utf-8")))
    assert summary["plan_sha256"] == plan["plan_sha256"]
    assert summary["scene_artifact_sha256"] == written_scene["artifact_sha256"]
    assert summary["video_artifacts"][0]["artifact_sha256"] == written_video["artifact_sha256"]
    assert [_key(row) for row in written_scene["examples"]] == [KEY_B, KEY_A]
    assert [_key(row) for row in written_video["examples"]] == [KEY_B, KEY_A]


def test_json_write_stages_a_complete_same_directory_temp_before_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact.json"
    destination.write_text("old artifact\n", encoding="utf-8")
    original_replace = Path.replace
    observations: list[tuple[Path, Path]] = []

    def observe_replace(source: Path, target: Path) -> Path:
        assert source.parent == destination.parent
        assert source.suffix == ".tmp"
        assert json.loads(source.read_text(encoding="utf-8")) == {"new": True}
        assert destination.read_text(encoding="utf-8") == "old artifact\n"
        observations.append((source, target))
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", observe_replace)

    _write_json(destination, {"new": True})

    assert observations
    assert json.loads(destination.read_text(encoding="utf-8")) == {"new": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_file_pipeline_rejects_duplicate_output_paths_before_writing(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    scene_input = tmp_path / "scene.json"
    video_input = tmp_path / "video.json"
    shared_output = tmp_path / "shared-output.json"
    plan_path.write_text(json.dumps(_plan()), encoding="utf-8")
    scene_input.write_text(json.dumps(_scene_artifact()), encoding="utf-8")
    video_input.write_text(json.dumps(_video_artifact()), encoding="utf-8")

    with pytest.raises(ValueError, match="output paths must be unique"):
        subset_embedding_files_by_plan(
            plan_path=plan_path,
            scene_input=scene_input,
            scene_output=shared_output,
            video_pairs=[(video_input, shared_output)],
        )

    assert not shared_output.exists()


@pytest.mark.parametrize("output_name", ("scene", "video"))
@pytest.mark.parametrize("input_name", ("plan", "scene", "video"))
@pytest.mark.parametrize(
    "alias_kind",
    (
        "direct",
        "output_symlink",
        "input_symlink",
        "output_parent_symlink",
        "input_parent_symlink",
    ),
)
def test_file_pipeline_rejects_output_input_alias_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_name: str,
    input_name: str,
    alias_kind: str,
) -> None:
    input_paths = {
        "plan": tmp_path / "plan.json",
        "scene": tmp_path / "scene-input.json",
        "video": tmp_path / "video-input.json",
    }
    for name, path in input_paths.items():
        path.write_text(f"{name} evidence\n", encoding="utf-8")
    scene_output = tmp_path / "scene-output.json"
    video_output = tmp_path / "video-output.json"
    conflicting_output = input_paths[input_name]
    if alias_kind == "output_symlink":
        conflicting_output = tmp_path / f"{output_name}-{input_name}-alias.json"
        conflicting_output.symlink_to(input_paths[input_name])
    elif alias_kind == "input_symlink":
        physical_input = tmp_path / f"{input_name}-physical.json"
        input_paths[input_name].replace(physical_input)
        input_paths[input_name].symlink_to(physical_input)
        conflicting_output = physical_input
    elif alias_kind == "output_parent_symlink":
        output_parent = tmp_path / "output-parent-alias"
        output_parent.symlink_to(tmp_path, target_is_directory=True)
        conflicting_output = output_parent / input_paths[input_name].name
    elif alias_kind == "input_parent_symlink":
        physical_parent = tmp_path / "physical-input-parent"
        physical_parent.mkdir()
        physical_input = physical_parent / input_paths[input_name].name
        input_paths[input_name].replace(physical_input)
        input_parent = tmp_path / "input-parent-alias"
        input_parent.symlink_to(physical_parent, target_is_directory=True)
        input_paths[input_name] = input_parent / physical_input.name
        conflicting_output = physical_input
    snapshots = {name: path.read_bytes() for name, path in input_paths.items()}
    if output_name == "scene":
        scene_output = conflicting_output
    else:
        video_output = conflicting_output

    reads: list[Path] = []

    def reject_read(path: Path) -> dict[str, Any]:
        reads.append(path)
        raise AssertionError("output/input alias must be rejected before JSON reading")

    monkeypatch.setattr(subset_module, "_read_json", reject_read)

    with pytest.raises(ValueError, match="output.*input"):
        subset_module.subset_embedding_files_by_plan(
            plan_path=input_paths["plan"],
            scene_input=input_paths["scene"],
            scene_output=scene_output,
            video_pairs=[(input_paths["video"], video_output)],
        )

    assert reads == []
    assert {
        name: path.read_bytes() for name, path in input_paths.items()
    } == snapshots
    assert not (tmp_path / "scene-output.json").exists()
    assert not (tmp_path / "video-output.json").exists()


def test_file_pipeline_uses_frozen_resolved_output_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_destination = tmp_path / "first-destination"
    second_destination = tmp_path / "second-destination"
    first_destination.mkdir()
    second_destination.mkdir()
    output_parent = tmp_path / "output-parent"
    output_parent.symlink_to(first_destination, target_is_directory=True)
    scene_output = output_parent / "scene.json"
    video_output = tmp_path / "video-output.json"
    plan_path = tmp_path / "plan.json"
    scene_input = tmp_path / "scene-input.json"
    video_input = tmp_path / "video-input.json"
    for path in (plan_path, scene_input, video_input):
        path.write_text("{}\n", encoding="utf-8")

    def redirect_after_validation(
        **_kwargs: object,
    ) -> dict[str, Any]:
        output_parent.unlink()
        output_parent.symlink_to(second_destination, target_is_directory=True)
        return {
            "plan_sha256": "a" * 64,
            "scene": {"artifact_sha256": "b" * 64},
            "videos": [
                {
                    "backbone": "test-backbone",
                    "artifact_sha256": "c" * 64,
                }
            ],
        }

    monkeypatch.setattr(subset_module, "_read_json", lambda _path: {})
    monkeypatch.setattr(
        subset_module,
        "subset_embedding_artifacts_by_plan",
        redirect_after_validation,
    )

    subset_module.subset_embedding_files_by_plan(
        plan_path=plan_path,
        scene_input=scene_input,
        scene_output=scene_output,
        video_pairs=[(video_input, video_output)],
    )

    assert json.loads(
        (first_destination / "scene.json").read_text(encoding="utf-8")
    ) == {"artifact_sha256": "b" * 64}
    assert not (second_destination / "scene.json").exists()
