from __future__ import annotations

import copy
import itertools
import json
import sys
from pathlib import Path

import pytest

from app.analysis.independent_shot_vlm import (
    canonical_sha256,
    seal_independent_shot_vlm_plan,
    seal_independent_shot_vlm_predictions,
)
from app.analysis.independent_shot_vlm_evidence_gate import (
    fuse_vlm_with_frozen_auxiliary,
)
from app.analysis.shot_validity_scene_state import (
    PHASE_FRACTIONS,
    phase_frame_indexes,
    seal_scene_embedding_artifact,
    verify_scene_embedding_artifact,
)
from app.analysis.shot_validity_video_backbone import (
    get_video_backbone_spec,
    seal_video_embedding_artifact,
)
from scripts import screen_shot_validity_scene_fusion as scene_fusion_module
from scripts.screen_shot_validity_scene_fusion import screen_scene_fusion

GAME_A_SHA256 = "a" * 64
GAME_B_SHA256 = "b" * 64
GAME_C_SHA256 = "c" * 64
BUNDLE_A_SHA256 = "1" * 64
BUNDLE_B_SHA256 = "2" * 64
BUNDLE_C_SHA256 = "3" * 64
TRAINING_MANIFEST_SHA256 = "f" * 64
SUBSET_SELECTION_PROTOCOL = "sealed_independent_shot_vlm_plan_three_part_key_order_v1"
THRESHOLD_SELECTION = "nested_inner_game_held_minimum_recall_0_85"
PROMOTION_REQUIREMENTS = {
    "minimum_pooled_precision": 0.85,
    "minimum_pooled_recall": 0.85,
    "minimum_per_game_precision": 0.85,
    "minimum_per_game_recall": 0.85,
}


def _scene_artifact() -> dict[str, object]:
    examples = []
    for game, bundle, label, offset in (
        (GAME_A_SHA256, BUNDLE_A_SHA256, "game-a", 0.0),
        (GAME_B_SHA256, BUNDLE_B_SHA256, "game-b", 0.1),
        (GAME_C_SHA256, BUNDLE_C_SHA256, "game-c", -0.1),
    ):
        for index in range(4):
            examples.append(
                {
                    "source_video_sha256": game,
                    "candidate_bundle_sha256": bundle,
                    "event_id": f"{label}-positive-{index}",
                    "event_present": True,
                    "phase_embeddings": [
                        [2.0 + offset + index * 0.1] * 576,
                        [2.5 + offset + index * 0.1] * 576,
                        [3.0 + offset + index * 0.1] * 576,
                    ],
                }
            )
            examples.append(
                {
                    "source_video_sha256": game,
                    "candidate_bundle_sha256": bundle,
                    "event_id": f"{label}-negative-{index}",
                    "event_present": False,
                    "phase_embeddings": [
                        [-2.0 + offset - index * 0.1] * 576,
                        [-2.5 + offset - index * 0.1] * 576,
                        [-3.0 + offset - index * 0.1] * 576,
                    ],
                }
            )
    return seal_scene_embedding_artifact(
        {
            "purpose": "scene_state_screening_training_only",
            "training_manifest_sha256": TRAINING_MANIFEST_SHA256,
            "backbone": "torchvision/mobilenet_v3_small/imagenet1k_v1",
            "backbone_sha256": "checkpoint",
            "embedding_dimension": 576,
            "phase_fractions": list(PHASE_FRACTIONS),
            "examples": examples,
        }
    )


def _video_artifact(scene: dict[str, object]) -> dict[str, object]:
    backbone = "torchvision/mvit_v2_s/kinetics400_v1"
    dimension = get_video_backbone_spec(backbone).embedding_dimension
    payload: dict[str, object] = {
        "purpose": "backbone_screening_training_only",
        "training_manifest_sha256": scene["training_manifest_sha256"],
        "backbone": backbone,
        "backbone_sha256": "video-checkpoint",
        "embedding_dimension": dimension,
        "clip_frames": 16,
        "examples": [
            {
                "source_video_sha256": row["source_video_sha256"],
                "candidate_bundle_sha256": row["candidate_bundle_sha256"],
                "event_id": row["event_id"],
                "event_present": row["event_present"],
                "embedding": [1.0 if row["event_present"] else -1.0] * dimension,
            }
            for row in scene["examples"]
        ],
    }
    return seal_video_embedding_artifact(payload)


def _fusion_screen_contract(
    *,
    scene: dict[str, object],
    videos: list[dict[str, object]],
) -> dict[str, object]:
    ordered_videos = sorted(
        videos,
        key=lambda video: (
            str(video["backbone"]),
            str(video["artifact_sha256"]),
        ),
    )
    source_videos = [
        {
            "backbone": str(video["backbone"]),
            "artifact_sha256": str(video["artifact_sha256"]),
        }
        for video in ordered_videos
    ]
    scene_dimension = 6 * len(scene["examples"][0]["phase_embeddings"][0])
    variant_specs = [{"name": "scene_phase_all", "input_dimension": scene_dimension}]
    for count in range(1, len(ordered_videos) + 1):
        for combination in itertools.combinations(ordered_videos, count):
            variant_specs.append(
                {
                    "name": "scene_phase_all+" + "+".join(str(row["backbone"]) for row in combination),
                    "input_dimension": scene_dimension
                    + sum(len(row["examples"][0]["embedding"]) for row in combination),
                }
            )
    return {
        "scene_embedding_artifact_sha256": scene["artifact_sha256"],
        "video_embedding_artifacts": source_videos,
        "predeclared_variants": variant_specs,
        "pca_components": 2,
        "regularization_c": 0.01,
        "threshold_selection": THRESHOLD_SELECTION,
        "promotion_requirements": dict(PROMOTION_REQUIREMENTS),
    }


def _independent_plan(
    scene: dict[str, object],
    *,
    fusion_screen_contract: dict[str, object],
) -> dict[str, object]:
    source_video_count = len({row["source_video_sha256"] for row in scene["examples"]})
    return seal_independent_shot_vlm_plan(
        {
            "training_manifest_sha256": scene["training_manifest_sha256"],
            "selection": {
                "method": "sha256_rank_within_video_and_boolean_class",
                "positive_per_video": 4,
                "negative_per_video": 4,
                "video_count": source_video_count,
                "example_count": len(scene["examples"]),
            },
            "input_contract": {
                "raw_frames_only": True,
                "labels_or_review_notes_exposed_to_model": False,
                "chronological_even_sampling": True,
                "max_frames": 6,
                "image_width": 704,
            },
            "fusion_screen_contract": fusion_screen_contract,
            "examples": [
                {
                    "source_video_sha256": row["source_video_sha256"],
                    "source_video_filename": f"{row['source_video_sha256']}.mp4",
                    "candidate_bundle_sha256": row["candidate_bundle_sha256"],
                    "event_id": row["event_id"],
                    "start_frame": index * 100,
                    "end_frame": index * 100 + 99,
                    "source_fps": 30.0,
                }
                for index, row in enumerate(scene["examples"])
            ],
        }
    )


def _seal_plan_bound_scene_subset(
    source_scene: dict[str, object],
    *,
    plan: dict[str, object],
) -> dict[str, object]:
    payload = copy.deepcopy(source_scene)
    source_sha256 = payload.pop("artifact_sha256")
    payload.update(
        {
            "plan_sha256": plan["plan_sha256"],
            "subset_selection_protocol": SUBSET_SELECTION_PROTOCOL,
            "source_embedding_artifact_sha256": payload.get("source_embedding_artifact_sha256", source_sha256),
        }
    )
    return seal_scene_embedding_artifact(payload)


def _seal_plan_bound_video_subset(
    source_video: dict[str, object],
    *,
    plan: dict[str, object],
) -> dict[str, object]:
    payload = copy.deepcopy(source_video)
    source_sha256 = payload.pop("artifact_sha256")
    payload.update(
        {
            "plan_sha256": plan["plan_sha256"],
            "subset_selection_protocol": SUBSET_SELECTION_PROTOCOL,
            "source_embedding_artifact_sha256": payload.get("source_embedding_artifact_sha256", source_sha256),
        }
    )
    return seal_video_embedding_artifact(payload)


def _plan_bound_subset_artifacts() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    unbound_scene = _scene_artifact()
    unbound_video = _video_artifact(unbound_scene)
    plan = _independent_plan(
        unbound_scene,
        fusion_screen_contract=_fusion_screen_contract(
            scene=unbound_scene,
            videos=[unbound_video],
        ),
    )
    scene = _seal_plan_bound_scene_subset(unbound_scene, plan=plan)
    video = _seal_plan_bound_video_subset(unbound_video, plan=plan)
    return scene, video, plan


def _screen_nested_v2(
    *,
    tmp_path: Path,
    prefix: str,
    scene: dict[str, object],
    video: dict[str, object] | None,
    plan: dict[str, object],
    pca_components: int = 2,
    regularization_c: float = 0.01,
) -> dict[str, object]:
    scene_path = tmp_path / f"{prefix}-scene.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    video_paths = []
    if video is not None:
        video_path = tmp_path / f"{prefix}-video.json"
        video_path.write_text(json.dumps(video), encoding="utf-8")
        video_paths.append(video_path)
    return scene_fusion_module.screen_nested_outer_game_held_fusion(  # type: ignore[attr-defined]
        scene_path=scene_path,
        video_embedding_paths=video_paths,
        plan=plan,
        pca_components=pca_components,
        regularization_c=regularization_c,
    )


def test_phase_frame_indexes_are_deterministic_and_include_three_states() -> None:
    assert PHASE_FRACTIONS == (0.15, 0.5, 0.85)
    assert phase_frame_indexes(100, 200) == (115, 150, 185)
    assert phase_frame_indexes(12, 12) == (12, 12, 12)
    with pytest.raises(ValueError, match="before"):
        phase_frame_indexes(20, 10)


def test_scene_embedding_artifact_is_training_only_and_hash_bound() -> None:
    artifact = _scene_artifact()

    assert verify_scene_embedding_artifact(artifact)["runtime_consumable"] is False
    artifact["phase_fractions"] = [0.1, 0.5, 0.9]
    with pytest.raises(ValueError, match="phase fractions"):
        verify_scene_embedding_artifact(artifact)
    artifact = _scene_artifact()
    artifact["backbone_sha256"] = "tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_scene_embedding_artifact(artifact)


def test_scene_fusion_uses_game_held_oof_and_strict_per_game_gate(
    tmp_path: Path,
) -> None:
    scene = _scene_artifact()
    video = _video_artifact(scene)
    scene_path = tmp_path / "scene.json"
    video_path = tmp_path / "video.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    video_path.write_text(json.dumps(video), encoding="utf-8")

    result = screen_scene_fusion(
        scene_path=scene_path,
        video_embedding_paths=[video_path],
        pca_components=2,
        regularization_c=0.01,
    )

    assert result["runtime_consumable"] is False
    assert result["selection_protocol"] == "game_held_oof_predeclared_variants"
    assert {row["name"] for row in result["variants"]} == {
        "scene_phase_all",
        "scene_phase_all+torchvision/mvit_v2_s/kinetics400_v1",
    }
    assert result["legacy_provenance_only"] is True
    assert result["best_variant"]["gate"]["promoted"] is False
    assert all(row["gate"]["promoted"] is False for row in result["variants"])
    assert result["best_variant"]["observed_metrics"]["legacy_gate_would_have_promoted"] is True
    assert all(
        fold["held_game_sha256"] not in fold["training_game_sha256s"] for fold in result["best_variant"]["folds"]
    )
    assert min(row["recall"] for row in result["best_variant"]["observed_metrics"]["per_game"].values()) >= 0.85


def test_scene_fusion_cli_requires_explicit_nested_or_legacy_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = _scene_artifact()
    scene_path = tmp_path / "scene.json"
    output_path = tmp_path / "screen.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "screen_shot_validity_scene_fusion.py",
            "--scene-embeddings",
            str(scene_path),
            "--output",
            str(output_path),
        ],
    )

    with pytest.raises(SystemExit) as raised:
        scene_fusion_module.main()

    assert raised.value.code == 2
    assert not output_path.exists()


def test_scene_fusion_cli_legacy_mode_is_never_promotable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scene = _scene_artifact()
    video = _video_artifact(scene)
    scene_path = tmp_path / "scene.json"
    video_path = tmp_path / "video.json"
    output_path = tmp_path / "legacy-screen.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    video_path.write_text(json.dumps(video), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "screen_shot_validity_scene_fusion.py",
            "--scene-embeddings",
            str(scene_path),
            "--video-embeddings",
            str(video_path),
            "--legacy-v1-provenance-only",
            "--output",
            str(output_path),
            "--pca-components",
            "2",
        ],
    )

    exit_code = scene_fusion_module.main()
    result = json.loads(output_path.read_text(encoding="utf-8"))
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert result["legacy_provenance_only"] is True
    assert result["best_variant"]["observed_metrics"]["legacy_gate_would_have_promoted"] is True
    assert result["best_variant"]["gate"]["promoted"] is False
    assert summary["legacy_provenance_only"] is True
    assert summary["gate"]["promoted"] is False


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
def test_scene_fusion_cli_rejects_output_input_alias_before_reading_or_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_name: str,
    alias_kind: str,
) -> None:
    input_paths = {
        "plan": tmp_path / "plan.json",
        "scene": tmp_path / "scene.json",
        "video": tmp_path / "video.json",
    }
    for name, path in input_paths.items():
        path.write_text(f"{name} evidence\n", encoding="utf-8")
    output = input_paths[input_name]
    if alias_kind == "output_symlink":
        output = tmp_path / f"{input_name}-output-alias.json"
        output.symlink_to(input_paths[input_name])
    elif alias_kind == "input_symlink":
        physical_input = tmp_path / f"{input_name}-physical.json"
        input_paths[input_name].replace(physical_input)
        input_paths[input_name].symlink_to(physical_input)
        output = physical_input
    elif alias_kind == "output_parent_symlink":
        output_parent = tmp_path / "output-parent-alias"
        output_parent.symlink_to(tmp_path, target_is_directory=True)
        output = output_parent / input_paths[input_name].name
    elif alias_kind == "input_parent_symlink":
        physical_parent = tmp_path / "physical-input-parent"
        physical_parent.mkdir()
        physical_input = physical_parent / input_paths[input_name].name
        input_paths[input_name].replace(physical_input)
        input_parent = tmp_path / "input-parent-alias"
        input_parent.symlink_to(physical_parent, target_is_directory=True)
        input_paths[input_name] = input_parent / physical_input.name
        output = physical_input
    snapshots = {name: path.read_bytes() for name, path in input_paths.items()}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "screen_shot_validity_scene_fusion.py",
            "--scene-embeddings",
            str(input_paths["scene"]),
            "--video-embeddings",
            str(input_paths["video"]),
            "--plan",
            str(input_paths["plan"]),
            "--output",
            str(output),
        ],
    )

    def reject_read(_path: Path, *_args: object, **_kwargs: object) -> str:
        raise AssertionError("output/input alias must be rejected before JSON reading")

    def reject_training(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("output/input alias must be rejected before training")

    monkeypatch.setattr(Path, "read_text", reject_read)
    monkeypatch.setattr(
        scene_fusion_module,
        "screen_nested_outer_game_held_fusion",
        reject_training,
    )

    with pytest.raises(ValueError, match="output.*input"):
        scene_fusion_module.main()

    assert {name: path.read_bytes() for name, path in input_paths.items()} == snapshots


def test_scene_fusion_cli_uses_frozen_resolved_output_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_paths = {
        "plan": tmp_path / "plan.json",
        "scene": tmp_path / "scene.json",
        "video": tmp_path / "video.json",
    }
    for path in input_paths.values():
        path.write_text("{}\n", encoding="utf-8")
    first_destination = tmp_path / "first-destination"
    second_destination = tmp_path / "second-destination"
    first_destination.mkdir()
    second_destination.mkdir()
    output_parent = tmp_path / "output-parent"
    output_parent.symlink_to(first_destination, target_is_directory=True)
    output = output_parent / "screen.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "screen_shot_validity_scene_fusion.py",
            "--scene-embeddings",
            str(input_paths["scene"]),
            "--video-embeddings",
            str(input_paths["video"]),
            "--plan",
            str(input_paths["plan"]),
            "--output",
            str(output),
        ],
    )

    def redirect_after_validation(**_kwargs: object) -> dict[str, object]:
        output_parent.unlink()
        output_parent.symlink_to(second_destination, target_is_directory=True)
        return {
            "selection_protocol": "test_nested_protocol",
            "outer_folds": [],
            "gate": {"promoted": True},
            "artifact_sha256": "a" * 64,
        }

    monkeypatch.setattr(
        scene_fusion_module,
        "screen_nested_outer_game_held_fusion",
        redirect_after_validation,
    )

    assert scene_fusion_module.main() == 0

    assert json.loads((first_destination / "screen.json").read_text(encoding="utf-8"))["artifact_sha256"] == "a" * 64
    assert not (second_destination / "screen.json").exists()


def test_scene_fusion_writer_stages_fsyncs_and_atomically_replaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "screen.json"
    destination.write_text("old artifact\n", encoding="utf-8")
    protected = tmp_path / "protected.txt"
    protected.write_text("do-not-overwrite\n", encoding="utf-8")
    legacy_temporary = destination.with_suffix(destination.suffix + ".tmp")
    legacy_temporary.symlink_to(protected)
    original_replace = Path.replace
    events: list[str] = []
    replacements: list[tuple[Path, Path]] = []

    def observe_fsync(file_descriptor: int) -> None:
        assert file_descriptor >= 0
        events.append("fsync")

    def observe_replace(source: Path, target: Path) -> Path:
        assert source.parent == destination.parent
        assert source.suffix == ".tmp"
        assert source != legacy_temporary
        expected_value = len(replacements) + 1
        assert json.loads(source.read_text(encoding="utf-8")) == {"new": expected_value}
        if not replacements:
            assert destination.read_text(encoding="utf-8") == "old artifact\n"
        else:
            assert json.loads(destination.read_text(encoding="utf-8")) == {"new": expected_value - 1}
        events.append("replace")
        replacements.append((source, target))
        return original_replace(source, target)

    monkeypatch.setattr(scene_fusion_module.os, "fsync", observe_fsync)
    monkeypatch.setattr(Path, "replace", observe_replace)

    scene_fusion_module._write_json_atomic(destination, {"new": 1})
    scene_fusion_module._write_json_atomic(destination, {"new": 2})

    assert events == ["fsync", "replace", "fsync", "replace"]
    assert len({source.name for source, _target in replacements}) == 2
    assert json.loads(destination.read_text(encoding="utf-8")) == {"new": 2}
    assert protected.read_text(encoding="utf-8") == "do-not-overwrite\n"
    assert legacy_temporary.is_symlink()


def test_nested_outer_held_labels_do_not_change_fold_threshold_or_predictions(
    tmp_path: Path,
) -> None:
    scene, video, plan = _plan_bound_subset_artifacts()
    relabeled_scene_payload = copy.deepcopy(scene)
    relabeled_scene_payload.pop("artifact_sha256")
    for row in relabeled_scene_payload["examples"]:
        if row["source_video_sha256"] == GAME_A_SHA256:
            row["event_present"] = not row["event_present"]
    relabeled_scene = seal_scene_embedding_artifact(relabeled_scene_payload)
    relabeled_video_payload = copy.deepcopy(video)
    relabeled_video_payload.pop("artifact_sha256")
    for row in relabeled_video_payload["examples"]:
        if row["source_video_sha256"] == GAME_A_SHA256:
            row["event_present"] = not row["event_present"]
    relabeled_video = seal_video_embedding_artifact(relabeled_video_payload)

    original = _screen_nested_v2(
        tmp_path=tmp_path,
        prefix="original",
        scene=scene,
        video=video,
        plan=plan,
    )
    relabeled = _screen_nested_v2(
        tmp_path=tmp_path,
        prefix="relabeled",
        scene=relabeled_scene,
        video=relabeled_video,
        plan=plan,
    )
    original_fold = next(row for row in original["outer_folds"] if row["held_game_sha256"] == GAME_A_SHA256)
    relabeled_fold = next(row for row in relabeled["outer_folds"] if row["held_game_sha256"] == GAME_A_SHA256)

    def prediction_projection(fold: dict[str, object]) -> list[tuple[object, ...]]:
        return [
            (
                row["source_video_sha256"],
                row["candidate_bundle_sha256"],
                row["event_id"],
                row["probability"],
            )
            for row in fold["oof_predictions"]
        ]

    assert original["schema_version"] == "agu.shot-validity-scene-fusion-screen.v2"
    assert original["selection_protocol"] == "nested_outer_game_held_v2"
    assert original_fold["selected_variant"] == relabeled_fold["selected_variant"]
    assert original_fold["threshold"] == pytest.approx(relabeled_fold["threshold"])
    assert prediction_projection(original_fold) == prediction_projection(relabeled_fold)


def test_nested_outer_folds_bind_plan_and_exclude_held_game_from_selection_and_fit(
    tmp_path: Path,
) -> None:
    scene, video, plan = _plan_bound_subset_artifacts()

    result = _screen_nested_v2(
        tmp_path=tmp_path,
        prefix="nested-contract",
        scene=scene,
        video=video,
        plan=plan,
    )

    assert result["plan_sha256"] == plan["plan_sha256"]
    assert result["training_manifest_sha256"] == plan["training_manifest_sha256"]
    assert result["fusion_screen_contract"] == plan["fusion_screen_contract"]
    assert result["predeclared_variants"] == plan["fusion_screen_contract"]["predeclared_variants"]
    assert (
        result["scene_embedding_artifact_sha256"] == plan["fusion_screen_contract"]["scene_embedding_artifact_sha256"]
    )
    assert result["video_embedding_artifact_sha256s"] == [
        row["artifact_sha256"] for row in plan["fusion_screen_contract"]["video_embedding_artifacts"]
    ]
    assert result["scene_subset_embedding_artifact_sha256"] == scene["artifact_sha256"]
    assert result["video_subset_embedding_artifact_sha256s"] == [video["artifact_sha256"]]
    assert (
        scene["source_embedding_artifact_sha256"] == plan["fusion_screen_contract"]["scene_embedding_artifact_sha256"]
    )
    assert (
        video["source_embedding_artifact_sha256"]
        == plan["fusion_screen_contract"]["video_embedding_artifacts"][0]["artifact_sha256"]
    )
    for fold in result["outer_folds"]:
        held = fold["held_game_sha256"]
        assert held not in fold["inner_selection_game_sha256s"]
        assert held not in fold["fit_game_sha256s"]
        assert fold["oof_predictions"]
        assert all(
            {
                "source_video_sha256",
                "candidate_bundle_sha256",
                "event_id",
                "probability",
            }.issubset(row)
            for row in fold["oof_predictions"]
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (("start_frame", 1), ("source_fps", 29.97)),
)
def test_nested_fusion_rejects_same_keys_bound_to_a_different_plan(
    tmp_path: Path,
    field: str,
    replacement: int | float,
) -> None:
    scene, video, plan = _plan_bound_subset_artifacts()
    changed_plan_payload = copy.deepcopy(plan)
    changed_plan_payload.pop("plan_sha256")
    changed_plan_payload["examples"][0][field] = replacement
    changed_plan = seal_independent_shot_vlm_plan(changed_plan_payload)

    assert {
        (
            row["source_video_sha256"],
            row["candidate_bundle_sha256"],
            row["event_id"],
        )
        for row in changed_plan["examples"]
    } == {
        (
            row["source_video_sha256"],
            row["candidate_bundle_sha256"],
            row["event_id"],
        )
        for row in plan["examples"]
    }
    assert changed_plan["plan_sha256"] != plan["plan_sha256"]
    with pytest.raises(ValueError, match="plan"):
        _screen_nested_v2(
            tmp_path=tmp_path,
            prefix=f"changed-{field}",
            scene=scene,
            video=video,
            plan=changed_plan,
        )


@pytest.mark.parametrize("artifact_name", ("scene", "video"))
def test_nested_fusion_rejects_plan_bound_subset_row_reordering(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    scene, video, plan = _plan_bound_subset_artifacts()
    if artifact_name == "scene":
        payload = copy.deepcopy(scene)
        payload.pop("artifact_sha256")
        payload["examples"] = list(reversed(payload["examples"]))
        scene = seal_scene_embedding_artifact(payload)
    else:
        payload = copy.deepcopy(video)
        payload.pop("artifact_sha256")
        payload["examples"] = list(reversed(payload["examples"]))
        video = seal_video_embedding_artifact(payload)

    with pytest.raises(ValueError, match="order|frozen plan"):
        _screen_nested_v2(
            tmp_path=tmp_path,
            prefix=f"reordered-{artifact_name}",
            scene=scene,
            video=video,
            plan=plan,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("source_video_sha256", int("4" * 64)),
        ("candidate_bundle_sha256", int("5" * 64)),
        ("event_id", 1),
    ),
)
def test_nested_fusion_rejects_non_string_three_part_keys(
    tmp_path: Path,
    field: str,
    replacement: int,
) -> None:
    scene, video, plan = _plan_bound_subset_artifacts()
    plan_payload = copy.deepcopy(plan)
    plan_payload.pop("plan_sha256")
    plan_payload["examples"][0][field] = replacement
    plan = seal_independent_shot_vlm_plan(plan_payload)
    scene_payload = copy.deepcopy(scene)
    scene_payload.pop("artifact_sha256")
    scene_payload["plan_sha256"] = plan["plan_sha256"]
    scene_payload["examples"][0][field] = replacement
    scene = seal_scene_embedding_artifact(scene_payload)
    video_payload = copy.deepcopy(video)
    video_payload.pop("artifact_sha256")
    video_payload["plan_sha256"] = plan["plan_sha256"]
    video_payload["examples"][0][field] = replacement
    video = seal_video_embedding_artifact(video_payload)

    with pytest.raises(ValueError, match="strings|SHA-256"):
        _screen_nested_v2(
            tmp_path=tmp_path,
            prefix=f"non-string-{field}",
            scene=scene,
            video=video,
            plan=plan,
        )


def test_nested_fusion_rejects_label_exposed_plan_before_model_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene, video, plan = _plan_bound_subset_artifacts()
    plan_payload = copy.deepcopy(plan)
    plan_payload.pop("plan_sha256")
    plan_payload["input_contract"]["labels_or_review_notes_exposed_to_model"] = True
    plan_payload["plan_sha256"] = canonical_sha256(
        plan_payload,
        hash_field="plan_sha256",
    )
    plan = plan_payload
    for artifact in (scene, video):
        artifact.pop("artifact_sha256")
        artifact["plan_sha256"] = plan["plan_sha256"]
    scene = seal_scene_embedding_artifact(scene)
    video = seal_video_embedding_artifact(video)

    def reject_fit(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("model fitting must not start for a label-exposed plan")

    monkeypatch.setattr(scene_fusion_module, "_fit_probability_model", reject_fit)
    with pytest.raises(ValueError, match="unsafe provenance|input contract"):
        _screen_nested_v2(
            tmp_path=tmp_path,
            prefix="label-exposed-plan",
            scene=scene,
            video=video,
            plan=plan,
        )


@pytest.mark.parametrize(
    ("artifact_name", "field", "replacement"),
    (
        ("scene", "plan_sha256", "0" * 64),
        ("video", "plan_sha256", "0" * 64),
        ("scene", "subset_selection_protocol", "unordered_three_part_keys"),
        ("video", "subset_selection_protocol", "unordered_three_part_keys"),
        ("scene", "source_embedding_artifact_sha256", "0" * 64),
        ("video", "source_embedding_artifact_sha256", "0" * 64),
    ),
)
def test_nested_fusion_rejects_unbound_subset_metadata(
    tmp_path: Path,
    artifact_name: str,
    field: str,
    replacement: str,
) -> None:
    scene, video, plan = _plan_bound_subset_artifacts()
    if artifact_name == "scene":
        scene_payload = copy.deepcopy(scene)
        scene_payload.pop("artifact_sha256")
        scene_payload[field] = replacement
        scene = seal_scene_embedding_artifact(scene_payload)
    else:
        video_payload = copy.deepcopy(video)
        video_payload.pop("artifact_sha256")
        video_payload[field] = replacement
        video = seal_video_embedding_artifact(video_payload)

    with pytest.raises(
        ValueError,
        match="plan|subset selection protocol|fusion screen contract",
    ):
        _screen_nested_v2(
            tmp_path=tmp_path,
            prefix=f"unbound-{artifact_name}-{field}",
            scene=scene,
            video=video,
            plan=plan,
        )


@pytest.mark.parametrize(
    ("pca_components", "regularization_c"),
    ((3, 0.01), (2, 0.1)),
)
def test_nested_fusion_rejects_call_parameter_drift_from_sealed_contract(
    tmp_path: Path,
    pca_components: int,
    regularization_c: float,
) -> None:
    scene, video, plan = _plan_bound_subset_artifacts()

    with pytest.raises(ValueError, match="fusion screen contract"):
        _screen_nested_v2(
            tmp_path=tmp_path,
            prefix=f"parameter-drift-{pca_components}-{regularization_c}",
            scene=scene,
            video=video,
            plan=plan,
            pca_components=pca_components,
            regularization_c=regularization_c,
        )


def test_nested_fusion_rejects_video_set_drift_from_sealed_contract(
    tmp_path: Path,
) -> None:
    scene, _video, plan = _plan_bound_subset_artifacts()

    with pytest.raises(ValueError, match="fusion screen contract"):
        _screen_nested_v2(
            tmp_path=tmp_path,
            prefix="missing-contracted-video",
            scene=scene,
            video=None,
            plan=plan,
        )


def test_nested_producer_artifact_is_accepted_by_structural_fuser(
    tmp_path: Path,
) -> None:
    scene, video, plan = _plan_bound_subset_artifacts()
    auxiliary = _screen_nested_v2(
        tmp_path=tmp_path,
        prefix="producer-consumer-integration",
        scene=scene,
        video=video,
        plan=plan,
    )
    vlm = seal_independent_shot_vlm_predictions(
        {
            "plan_sha256": plan["plan_sha256"],
            "model": {
                "name": "independent-vlm-integration-test",
                "independent_from_codex": True,
            },
            "predictions": [
                {
                    "source_video_sha256": row["source_video_sha256"],
                    "candidate_bundle_sha256": row["candidate_bundle_sha256"],
                    "event_id": row["event_id"],
                    "field_goal_state": "live_field_goal",
                    "confidence": 0.9,
                    "observables": {
                        "continuous_live_play": True,
                        "controlled_ball_before_release": True,
                        "ball_separates_from_hands": True,
                        "ball_moves_toward_rim": True,
                        "replay_or_highlight": False,
                        "free_throw": False,
                    },
                }
                for row in plan["examples"]
            ],
        }
    )

    fused = fuse_vlm_with_frozen_auxiliary(
        plan=plan,
        vlm_predictions=vlm,
        auxiliary_artifact=auxiliary,
        expected_auxiliary_artifact_sha256=str(auxiliary["artifact_sha256"]),
    )

    assert fused["coverage"]["vlm_event_count"] == len(plan["examples"])
    assert fused["coverage"]["missing_auxiliary_keys"] == []
    assert fused["model"]["auxiliary_provenance_verification"] == "structural_only"
    assert fused["model"]["formal_promotion_eligible"] is False


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        (
            "predeclared_variants",
            [{"name": "scene_phase_all", "input_dimension": 3456}],
        ),
        ("threshold_selection", "outer_label_selected_threshold"),
        (
            "promotion_requirements",
            {
                **PROMOTION_REQUIREMENTS,
                "minimum_per_game_recall": 0.8,
            },
        ),
    ),
)
def test_nested_fusion_rejects_variant_or_policy_drift_from_sealed_contract(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    scene, video, plan = _plan_bound_subset_artifacts()
    changed_plan_payload = copy.deepcopy(plan)
    changed_plan_payload.pop("plan_sha256")
    changed_plan_payload["fusion_screen_contract"][field] = replacement
    changed_plan = seal_independent_shot_vlm_plan(changed_plan_payload)
    scene = _seal_plan_bound_scene_subset(scene, plan=changed_plan)
    video = _seal_plan_bound_video_subset(video, plan=changed_plan)

    with pytest.raises(ValueError, match="fusion screen contract"):
        _screen_nested_v2(
            tmp_path=tmp_path,
            prefix=f"contract-drift-{field}",
            scene=scene,
            video=video,
            plan=changed_plan,
        )


def test_nested_fusion_rejects_variant_dimension_drift_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene, video, plan = _plan_bound_subset_artifacts()
    changed_plan_payload = copy.deepcopy(plan)
    changed_plan_payload.pop("plan_sha256")
    changed_plan_payload["fusion_screen_contract"]["predeclared_variants"][0]["input_dimension"] = 999
    changed_plan = seal_independent_shot_vlm_plan(changed_plan_payload)
    scene = _seal_plan_bound_scene_subset(scene, plan=changed_plan)
    video = _seal_plan_bound_video_subset(video, plan=changed_plan)

    def fail_if_training_starts(**_kwargs: object) -> object:
        pytest.fail("variant dimensions must be validated before model fitting")

    monkeypatch.setattr(
        scene_fusion_module,
        "_fit_probability_model",
        fail_if_training_starts,
    )
    with pytest.raises(ValueError, match="fusion screen contract"):
        _screen_nested_v2(
            tmp_path=tmp_path,
            prefix="contract-drift-input-dimension",
            scene=scene,
            video=video,
            plan=changed_plan,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "expected_error"),
    (
        ("pca_components", True, "PCA components must be a positive integer"),
        ("regularization_c", True, "regularization C must be a positive number"),
    ),
)
def test_nested_fusion_rejects_boolean_hyperparameters(
    tmp_path: Path,
    field: str,
    replacement: object,
    expected_error: str,
) -> None:
    scene, video, plan = _plan_bound_subset_artifacts()
    changed_plan_payload = copy.deepcopy(plan)
    changed_plan_payload.pop("plan_sha256")
    changed_plan_payload["fusion_screen_contract"][field] = replacement
    changed_plan = seal_independent_shot_vlm_plan(changed_plan_payload)
    scene = _seal_plan_bound_scene_subset(scene, plan=changed_plan)
    video = _seal_plan_bound_video_subset(video, plan=changed_plan)

    kwargs = {field: replacement}
    with pytest.raises(ValueError, match=expected_error):
        _screen_nested_v2(
            tmp_path=tmp_path,
            prefix=f"boolean-{field}",
            scene=scene,
            video=video,
            plan=changed_plan,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("pca_components", "regularization_c", "expected_error"),
    (
        (True, 0.01, "PCA components must be a positive integer"),
        (2, True, "regularization C must be a positive number"),
    ),
)
def test_legacy_scene_fusion_rejects_boolean_hyperparameters(
    tmp_path: Path,
    pca_components: object,
    regularization_c: object,
    expected_error: str,
) -> None:
    scene = _scene_artifact()
    scene_path = tmp_path / "legacy-boolean-scene.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_error):
        screen_scene_fusion(
            scene_path=scene_path,
            pca_components=pca_components,
            regularization_c=regularization_c,
        )


def test_scene_fusion_rejects_misaligned_video_examples(tmp_path: Path) -> None:
    scene = _scene_artifact()
    video = _video_artifact(scene)
    video["examples"] = video["examples"][:-1]
    video = seal_video_embedding_artifact({key: value for key, value in video.items() if key != "artifact_sha256"})
    scene_path = tmp_path / "scene.json"
    video_path = tmp_path / "video.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    video_path.write_text(json.dumps(video), encoding="utf-8")

    with pytest.raises(ValueError, match="identical examples"):
        screen_scene_fusion(scene_path=scene_path, video_embedding_paths=[video_path])
