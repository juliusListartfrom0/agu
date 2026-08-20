import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import pytest

from app.analysis.vru_causal_review import seal_vru_causal_review
from app.analysis.vru_causal_source_groups import seal_vru_causal_source_groups
from app.analysis.vru_causal_training import encode_json_artifact
from app.analysis.vru_causal_training_v2 import (
    FROZEN_HARWOOD_ARTIFACT_SHA256,
    FROZEN_HARWOOD_FILE_SHA256,
    FROZEN_PARENT_ARTIFACT_SHA256,
    FROZEN_PARENT_CHILD_SHA256,
    FROZEN_PARENT_FILE_SHA256,
    TRAINING_EXPORT_V2_SCHEMA,
    ContinuousCausalTrainingChain,
    _build_harwood_candidate,
    _verify_harwood_chain,
    _verify_harwood_raw_manifest,
    _verify_harwood_reference,
    _verify_harwood_review_partition,
    _verify_parent_chain,
    _verify_source_groups_file,
    _verify_v2_index,
    build_vru_causal_shot_validity_training_extension_export,
    canonical_sha256,
    verify_vru_causal_shot_validity_training_extension_export,
    verify_vru_causal_shot_validity_training_extension_index,
)
from scripts import export_vru_causal_shot_validity_training_v2 as export_script

ROOT = Path(__file__).resolve().parents[1]
PARENT_ROOT = ROOT / "analysis_outputs/public_research/vru_causal_closure_training_v1"
HARWOOD_ROOT = ROOT / "analysis_outputs/public_research/wikimedia_hctv_harwood_causal_review_v2"
FROZEN_SOURCE_GROUP_ARTIFACT_SHA256 = "029cb1b4c56eebe2cf36495618e12966944883e52a706a5bb05feb0e81e674b0"
FROZEN_SOURCE_GROUP_FILE_SHA256 = "d48f990790b57b11389186b407078f52fce3d34a6410a9101001cd5d3221c05e"


def _harwood_chain_kwargs(*, source_manifest_path: Path | None = None) -> dict[str, object]:
    return {
        "harwood_selection_path": HARWOOD_ROOT / "selection.json",
        "expected_harwood_selection_artifact_sha256": FROZEN_HARWOOD_ARTIFACT_SHA256["selection"],
        "harwood_review_plan_path": HARWOOD_ROOT / "review_plan.json",
        "expected_harwood_review_plan_artifact_sha256": FROZEN_HARWOOD_ARTIFACT_SHA256["review_plan"],
        "harwood_sealed_review_path": HARWOOD_ROOT / "sealed_review.json",
        "expected_harwood_sealed_review_artifact_sha256": FROZEN_HARWOOD_ARTIFACT_SHA256["sealed_review"],
        "harwood_raw_frame_manifest_path": HARWOOD_ROOT / "raw_frames_manifest.json",
        "expected_harwood_raw_frame_manifest_artifact_sha256": FROZEN_HARWOOD_ARTIFACT_SHA256["raw_frame_manifest"],
        "harwood_source_manifest_path": source_manifest_path or HARWOOD_ROOT / "source_manifest.json",
    }


def test_v2_export_contract_is_additive() -> None:
    assert TRAINING_EXPORT_V2_SCHEMA == ("agu.vru-causal-shot-validity-training-export.v2")
    assert tuple(ContinuousCausalTrainingChain.__dataclass_fields__) == (
        "index",
        "extension_candidate_bundles",
        "extension_label_files",
        "source_groups",
        "resolved_examples",
        "validated_input_paths",
    )
    assert callable(verify_vru_causal_shot_validity_training_extension_index)
    assert callable(verify_vru_causal_shot_validity_training_extension_export)
    assert callable(export_script.main)


def test_parent_chain_replays_index_and_all_six_children() -> None:
    parent = _verify_parent_chain(
        parent_export_path=PARENT_ROOT / "training_export.json",
        expected_parent_artifact_sha256=FROZEN_PARENT_ARTIFACT_SHA256,
        expected_parent_file_sha256=FROZEN_PARENT_FILE_SHA256,
        parent_asset_root=PARENT_ROOT,
    )

    assert len(parent.candidate_bundles) == 3
    assert len(parent.label_files) == 3
    assert len(parent.resolved_examples) == 22
    assert sum(row["event_present"] for row in parent.resolved_examples) == 14
    assert len(parent.validated_paths) == 7


def test_parent_chain_rejects_wrong_receipt_before_read(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not TASK-0254"):
        _verify_parent_chain(
            parent_export_path=tmp_path / "missing.json",
            expected_parent_artifact_sha256="0" * 64,
            expected_parent_file_sha256=FROZEN_PARENT_FILE_SHA256,
            parent_asset_root=tmp_path,
        )


def test_parent_chain_rejects_child_content_drift(tmp_path: Path) -> None:
    for filename in (
        "hazen_candidate_bundle_v1.json",
        "hazen_shot_validity_labels_v1.json",
        "randolph_candidate_bundle_v1.json",
        "randolph_shot_validity_labels_v1.json",
        "vtv_candidate_bundle_v1.json",
        "vtv_shot_validity_labels_v1.json",
    ):
        shutil.copyfile(PARENT_ROOT / filename, tmp_path / filename)
    with (tmp_path / "hazen_shot_validity_labels_v1.json").open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(ValueError, match="file receipt mismatch"):
        _verify_parent_chain(
            parent_export_path=PARENT_ROOT / "training_export.json",
            expected_parent_artifact_sha256=FROZEN_PARENT_ARTIFACT_SHA256,
            expected_parent_file_sha256=FROZEN_PARENT_FILE_SHA256,
            parent_asset_root=tmp_path,
        )


def test_parent_chain_rejects_wrong_parent_file_bytes_or_external_receipt(tmp_path: Path) -> None:
    drifted_index = tmp_path / "training_export.json"
    drifted_index.write_bytes((PARENT_ROOT / "training_export.json").read_bytes() + b"\n")
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        _verify_parent_chain(
            parent_export_path=drifted_index,
            expected_parent_artifact_sha256=FROZEN_PARENT_ARTIFACT_SHA256,
            expected_parent_file_sha256=FROZEN_PARENT_FILE_SHA256,
            parent_asset_root=PARENT_ROOT,
        )

    with pytest.raises(ValueError, match="not TASK-0254"):
        _verify_parent_chain(
            parent_export_path=PARENT_ROOT / "training_export.json",
            expected_parent_artifact_sha256=FROZEN_PARENT_ARTIFACT_SHA256,
            expected_parent_file_sha256="0" * 64,
            parent_asset_root=PARENT_ROOT,
        )


@pytest.mark.parametrize("mutation", ("missing", "renamed"))
def test_parent_chain_rejects_missing_or_renamed_child(tmp_path: Path, mutation: str) -> None:
    for filename in FROZEN_PARENT_CHILD_SHA256:
        shutil.copyfile(PARENT_ROOT / filename, tmp_path / filename)
    victim = tmp_path / "hazen_candidate_bundle_v1.json"
    if mutation == "missing":
        victim.unlink()
    else:
        victim.rename(tmp_path / "renamed_candidate_bundle_v1.json")

    with pytest.raises(ValueError, match="missing|cannot read"):
        _verify_parent_chain(
            parent_export_path=PARENT_ROOT / "training_export.json",
            expected_parent_artifact_sha256=FROZEN_PARENT_ARTIFACT_SHA256,
            expected_parent_file_sha256=FROZEN_PARENT_FILE_SHA256,
            parent_asset_root=tmp_path,
        )


def test_harwood_chain_replays_source_video_and_all_jpegs() -> None:
    harwood = _verify_harwood_chain(
        harwood_selection_path=HARWOOD_ROOT / "selection.json",
        expected_harwood_selection_artifact_sha256=(FROZEN_HARWOOD_ARTIFACT_SHA256["selection"]),
        harwood_review_plan_path=HARWOOD_ROOT / "review_plan.json",
        expected_harwood_review_plan_artifact_sha256=(FROZEN_HARWOOD_ARTIFACT_SHA256["review_plan"]),
        harwood_sealed_review_path=HARWOOD_ROOT / "sealed_review.json",
        expected_harwood_sealed_review_artifact_sha256=(FROZEN_HARWOOD_ARTIFACT_SHA256["sealed_review"]),
        harwood_raw_frame_manifest_path=HARWOOD_ROOT / "raw_frames_manifest.json",
        expected_harwood_raw_frame_manifest_artifact_sha256=(FROZEN_HARWOOD_ARTIFACT_SHA256["raw_frame_manifest"]),
        harwood_source_manifest_path=HARWOOD_ROOT / "source_manifest.json",
    )

    assert len(harwood.plan["examples"]) == 24
    assert len(harwood.jpeg_paths) == 1536
    assert harwood.reference["source_frame_manifest"]["jpeg_count"] == 1536

    changed_review = copy.deepcopy(harwood.review)
    changed_row = changed_review["reviews"][13]
    changed_row["outcome"] = "made"
    changed_row["confidence"] = "low"
    changed_row["notes"] = "deliberately changed label detail"
    changed_row["release_position"] = 0
    changed_row["rim_position"] = 1
    changed_row["release_frame"] = 1
    changed_row["rim_frame"] = 2
    changed_row["evidence"] = {key: False for key in changed_row["evidence"]}
    original_candidate = _build_harwood_candidate(harwood)
    changed_candidate = _build_harwood_candidate(replace(harwood, review=changed_review))
    assert encode_json_artifact(changed_candidate) == encode_json_artifact(original_candidate)


def test_harwood_chain_rejects_wrong_external_receipt_before_read(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="not TASK-0256"):
        _verify_harwood_chain(
            harwood_selection_path=missing,
            expected_harwood_selection_artifact_sha256="0" * 64,
            harwood_review_plan_path=missing,
            expected_harwood_review_plan_artifact_sha256=(FROZEN_HARWOOD_ARTIFACT_SHA256["review_plan"]),
            harwood_sealed_review_path=missing,
            expected_harwood_sealed_review_artifact_sha256=(FROZEN_HARWOOD_ARTIFACT_SHA256["sealed_review"]),
            harwood_raw_frame_manifest_path=missing,
            expected_harwood_raw_frame_manifest_artifact_sha256=(FROZEN_HARWOOD_ARTIFACT_SHA256["raw_frame_manifest"]),
            harwood_source_manifest_path=missing,
        )


@pytest.mark.parametrize("mutation", ("shot_negative_swap", "uncertain_to_negative"))
def test_self_resealed_harwood_label_reclassification_is_rejected(mutation: str) -> None:
    plan = json.loads((HARWOOD_ROOT / "review_plan.json").read_text(encoding="utf-8"))
    original = json.loads((HARWOOD_ROOT / "sealed_review.json").read_text(encoding="utf-8"))
    reviews = copy.deepcopy(original["reviews"])
    if mutation == "shot_negative_swap":
        shot = next(row for row in reviews if row["shot_sequence"] == "shot")
        negative = next(row for row in reviews if row["shot_sequence"] == "not_a_shot")
        for field in (
            "shot_sequence",
            "release_position",
            "rim_position",
            "outcome",
            "evidence",
        ):
            shot[field], negative[field] = negative[field], shot[field]
    else:
        uncertain = next(row for row in reviews if row["shot_sequence"] == "uncertain")
        uncertain.update(
            {
                "shot_sequence": "not_a_shot",
                "release_position": None,
                "rim_position": None,
                "outcome": "not_applicable",
                "evidence": {key: False for key in uncertain["evidence"]},
            }
        )
    resealed = seal_vru_causal_review(
        {"reviewer": original["reviewer"], "reviews": reviews},
        plan=plan,
        expected_plan_artifact_sha256=FROZEN_HARWOOD_ARTIFACT_SHA256["review_plan"],
    )

    with pytest.raises(ValueError, match="frozen|partition"):
        _verify_harwood_review_partition(review=resealed, plan=plan)


def test_harwood_chain_rejects_source_video_size_and_hash_drift(tmp_path: Path) -> None:
    manifest_path = tmp_path / "source_manifest.json"
    shutil.copyfile(HARWOOD_ROOT / "source_manifest.json", manifest_path)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    video_path = source_dir / "hctv_harwood_2026.webm"

    video_path.write_bytes(b"wrong-size")
    with pytest.raises(ValueError, match="source video size does not match"):
        _verify_harwood_chain(**_harwood_chain_kwargs(source_manifest_path=manifest_path))

    video_path.unlink()
    subprocess.run(
        [
            "cp",
            "-c",
            os.fspath(HARWOOD_ROOT / "source/hctv_harwood_2026.webm"),
            os.fspath(video_path),
        ],
        check=True,
    )
    with video_path.open("r+b") as handle:
        original = handle.read(1)
        handle.seek(0)
        handle.write(bytes([original[0] ^ 0xFF]))
        handle.flush()
        os.fsync(handle.fileno())
    with pytest.raises(ValueError, match="source video SHA-256 does not match"):
        _verify_harwood_chain(**_harwood_chain_kwargs(source_manifest_path=manifest_path))


def test_harwood_raw_manifest_rejects_missing_changed_and_extra_jpegs(tmp_path: Path) -> None:
    raw_manifest = json.loads((HARWOOD_ROOT / "raw_frames_manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((HARWOOD_ROOT / "review_plan.json").read_text(encoding="utf-8"))
    raw_root = tmp_path / raw_manifest["frame_root"]
    raw_root.mkdir()
    source_root = HARWOOD_ROOT / raw_manifest["frame_root"]
    for row in raw_manifest["frames"]:
        destination = raw_root / row["relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source_root / row["relative_path"], destination)

    verifier_kwargs = {
        "manifest_path": tmp_path / "raw_frames_manifest.json",
        "plan": plan,
        "expected_artifact_sha256": FROZEN_HARWOOD_ARTIFACT_SHA256["raw_frame_manifest"],
        "source_manifest_sha256": FROZEN_HARWOOD_FILE_SHA256["source_manifest"],
    }
    first_relative = raw_manifest["frames"][0]["relative_path"]
    first_source = source_root / first_relative
    first_copy = raw_root / first_relative

    first_copy.unlink()
    with pytest.raises(ValueError, match="missing|aliased"):
        _verify_harwood_raw_manifest(raw_manifest, **verifier_kwargs)
    os.link(first_source, first_copy)

    first_copy.unlink()
    shutil.copyfile(first_source, first_copy)
    with first_copy.open("r+b") as handle:
        original = handle.read(1)
        handle.seek(0)
        handle.write(bytes([original[0] ^ 0xFF]))
    with pytest.raises(ValueError, match="receipt mismatch"):
        _verify_harwood_raw_manifest(raw_manifest, **verifier_kwargs)
    first_copy.unlink()
    os.link(first_source, first_copy)

    extra = raw_root / "unexpected.jpg"
    extra.write_bytes(b"extra JPEG")
    with pytest.raises(ValueError, match="missing or extra"):
        _verify_harwood_raw_manifest(raw_manifest, **verifier_kwargs)


def test_real_chain_builds_only_additive_harwood_children(tmp_path: Path) -> None:
    frozen_v1_paths = {
        "training_export.json": PARENT_ROOT / "training_export.json",
        **{filename: PARENT_ROOT / filename for filename in FROZEN_PARENT_CHILD_SHA256},
    }
    v1_before = {filename: hashlib.sha256(path.read_bytes()).hexdigest() for filename, path in frozen_v1_paths.items()}
    assert v1_before == {
        "training_export.json": FROZEN_PARENT_FILE_SHA256,
        **dict(FROZEN_PARENT_CHILD_SHA256),
    }
    parent_index = json.loads((PARENT_ROOT / "training_export.json").read_text(encoding="utf-8"))
    hashes = {row["source_id"]: row["source_video_sha256"] for row in parent_index["sources"]}
    hashes["harwood"] = "bf7f3135774027aec3c2827cfa282c95ac4f958dd2bccd93e42f370d4b1f8b81"
    source_groups = seal_vru_causal_source_groups(
        {
            "schema_version": "agu.vru-causal-source-groups-spec.v1",
            "games": [
                {
                    "game_id": game_id,
                    "source_video_sha256": hashes[game_id],
                    "production_family": family,
                }
                for game_id, family in (
                    ("hazen", "hctv"),
                    ("randolph", "hctv"),
                    ("vtv", "vtv"),
                    ("harwood", "hctv"),
                )
            ],
        }
    )
    source_groups_path = tmp_path / "source_groups.json"
    source_group_bytes = encode_json_artifact(source_groups)
    source_groups_path.write_bytes(source_group_bytes)
    assert source_groups["artifact_sha256"] == FROZEN_SOURCE_GROUP_ARTIFACT_SHA256
    assert hashlib.sha256(source_group_bytes).hexdigest() == FROZEN_SOURCE_GROUP_FILE_SHA256
    replayed_source_groups, replayed_source_group_file_sha = _verify_source_groups_file(
        source_groups_path,
        expected_artifact_sha256=FROZEN_SOURCE_GROUP_ARTIFACT_SHA256,
    )
    assert replayed_source_groups == source_groups
    assert replayed_source_group_file_sha == FROZEN_SOURCE_GROUP_FILE_SHA256
    with pytest.raises(ValueError, match="expected|TASK-0257"):
        _verify_source_groups_file(
            source_groups_path,
            expected_artifact_sha256="f" * 64,
        )
    noncanonical_source_groups_path = tmp_path / "source_groups_noncanonical.json"
    noncanonical_source_groups_path.write_text(json.dumps(source_groups), encoding="utf-8")
    with pytest.raises(ValueError, match="file.*frozen|canonical"):
        _verify_source_groups_file(
            noncanonical_source_groups_path,
            expected_artifact_sha256=FROZEN_SOURCE_GROUP_ARTIFACT_SHA256,
        )

    extension_root = tmp_path / "extension"
    output_path = extension_root / "training_export_v2.json"
    preflight_inputs, preflight_output, child_paths, preflight_protected = export_script._preflight_paths(
        argparse.Namespace(
            parent_export=PARENT_ROOT / "training_export.json",
            parent_asset_root=PARENT_ROOT,
            harwood_selection=HARWOOD_ROOT / "selection.json",
            harwood_review_plan=HARWOOD_ROOT / "review_plan.json",
            harwood_sealed_review=HARWOOD_ROOT / "sealed_review.json",
            harwood_raw_frame_manifest=HARWOOD_ROOT / "raw_frames_manifest.json",
            harwood_source_manifest=HARWOOD_ROOT / "source_manifest.json",
            source_groups=source_groups_path,
            output=output_path,
        )
    )
    assert preflight_output == output_path
    assert len(preflight_protected) == 1552

    build_kwargs = {
        "parent_export_path": preflight_inputs["parent_export"],
        "expected_parent_artifact_sha256": FROZEN_PARENT_ARTIFACT_SHA256,
        "expected_parent_file_sha256": FROZEN_PARENT_FILE_SHA256,
        "parent_asset_root": preflight_inputs["parent_asset_root"],
        "harwood_selection_path": preflight_inputs["harwood_selection"],
        "expected_harwood_selection_artifact_sha256": FROZEN_HARWOOD_ARTIFACT_SHA256["selection"],
        "harwood_review_plan_path": preflight_inputs["harwood_review_plan"],
        "expected_harwood_review_plan_artifact_sha256": FROZEN_HARWOOD_ARTIFACT_SHA256["review_plan"],
        "harwood_sealed_review_path": preflight_inputs["harwood_sealed_review"],
        "expected_harwood_sealed_review_artifact_sha256": FROZEN_HARWOOD_ARTIFACT_SHA256["sealed_review"],
        "harwood_raw_frame_manifest_path": preflight_inputs["harwood_raw_frame_manifest"],
        "expected_harwood_raw_frame_manifest_artifact_sha256": FROZEN_HARWOOD_ARTIFACT_SHA256["raw_frame_manifest"],
        "harwood_source_manifest_path": preflight_inputs["harwood_source_manifest"],
        "source_groups_path": preflight_inputs["source_groups"],
        "expected_source_groups_artifact_sha256": FROZEN_SOURCE_GROUP_ARTIFACT_SHA256,
    }
    chain = build_vru_causal_shot_validity_training_extension_export(**build_kwargs)
    repeated_chain = build_vru_causal_shot_validity_training_extension_export(**build_kwargs)
    assert encode_json_artifact(repeated_chain.index) == encode_json_artifact(chain.index)
    assert {
        name: encode_json_artifact(payload) for name, payload in repeated_chain.extension_candidate_bundles.items()
    } == {name: encode_json_artifact(payload) for name, payload in chain.extension_candidate_bundles.items()}
    assert {name: encode_json_artifact(payload) for name, payload in repeated_chain.extension_label_files.items()} == {
        name: encode_json_artifact(payload) for name, payload in chain.extension_label_files.items()
    }

    assert chain.index["counts"] == {
        "selected": 48,
        "exported": 45,
        "positive": 17,
        "negative": 28,
        "excluded_uncertain": 3,
        "game_groups": 4,
        "production_families": 2,
    }
    assert tuple(chain.extension_candidate_bundles) == ("harwood_candidate_bundle_v2.json",)
    assert tuple(chain.extension_label_files) == ("harwood_shot_validity_labels_v2.json",)
    assert len(chain.resolved_examples) == 45
    assert sum(row["event_present"] for row in chain.resolved_examples) == 17
    game_by_source = {row["source_video_sha256"]: row["game_id"] for row in chain.source_groups["games"]}
    assert [game_by_source[row["source_video_sha256"]] for row in chain.resolved_examples] == ["hazen"] * 8 + [
        "randolph"
    ] * 7 + ["vtv"] * 7 + ["harwood"] * 23
    assert {row["event_id"] for row in chain.resolved_examples}.isdisjoint(chain.index["excluded_review_ids"])

    promoted = copy.deepcopy(chain.index)
    promoted["promoted"] = True
    promoted["artifact_sha256"] = canonical_sha256(promoted)
    with pytest.raises(ValueError, match="consumption flags"):
        verify_vru_causal_shot_validity_training_extension_index(promoted)

    reordered = copy.deepcopy(chain.index)
    reordered["parent"]["sources"].reverse()
    reordered["artifact_sha256"] = canonical_sha256(reordered)
    with pytest.raises(ValueError, match="canonically ordered"):
        _verify_v2_index(reordered)

    unknown = copy.deepcopy(chain.index)
    unknown["unexpected"] = True
    unknown["artifact_sha256"] = canonical_sha256(unknown)
    with pytest.raises(ValueError, match="fields are not canonical"):
        verify_vru_causal_shot_validity_training_extension_index(unknown)
    with pytest.raises(ValueError, match="expected training v2"):
        verify_vru_causal_shot_validity_training_extension_index(
            chain.index,
            expected_artifact_sha256="f" * 64,
        )

    for field, value in (("size_bytes", 1), ("sha256", "0" * 64)):
        drifted_source = copy.deepcopy(chain.index)
        drifted_source["harwood_evidence"]["source_video"][field] = value
        drifted_source["artifact_sha256"] = canonical_sha256(drifted_source)
        with pytest.raises(ValueError, match="source-video reference"):
            _verify_v2_index(drifted_source)

    verified_harwood_reference = copy.deepcopy(chain.index["harwood_evidence"])
    _verify_harwood_reference(verified_harwood_reference)
    candidate = next(iter(chain.extension_candidate_bundles.values()))
    for event in candidate["events"]:
        assert event["confidence"] == 0.5
        assert event["release_frame"] is None
        assert event["outcome_frame"] is None
        assert event["outcome"] is None
        assert event["reviewer"] is None
        assert event["evidence"] == []

    export_script._write_generation_atomic(
        output_path=output_path,
        child_paths=child_paths,
        chain=chain,
        protected_inputs=(*preflight_protected, *chain.validated_input_paths),
    )
    assert json.loads(output_path.read_text(encoding="utf-8")) == chain.index
    published_bytes = {path.name: path.read_bytes() for path in extension_root.iterdir()}
    export_script._write_generation_atomic(
        output_path=output_path,
        child_paths=child_paths,
        chain=chain,
        protected_inputs=(*preflight_protected, *chain.validated_input_paths),
    )
    assert {path.name: path.read_bytes() for path in extension_root.iterdir()} == published_bytes
    verify_kwargs = {
        "extension_asset_root": extension_root,
        "parent_export_path": PARENT_ROOT / "training_export.json",
        "expected_parent_artifact_sha256": FROZEN_PARENT_ARTIFACT_SHA256,
        "expected_parent_file_sha256": FROZEN_PARENT_FILE_SHA256,
        "parent_asset_root": PARENT_ROOT,
        **_harwood_chain_kwargs(),
        "source_groups_path": source_groups_path,
    }
    with pytest.raises(ValueError, match="expected training v2"):
        verify_vru_causal_shot_validity_training_extension_export(
            chain.index,
            expected_artifact_sha256="f" * 64,
            expected_source_groups_artifact_sha256=FROZEN_SOURCE_GROUP_ARTIFACT_SHA256,
            **verify_kwargs,
        )
    with pytest.raises(ValueError, match="expected|TASK-0257"):
        verify_vru_causal_shot_validity_training_extension_export(
            chain.index,
            expected_artifact_sha256=chain.index["artifact_sha256"],
            expected_source_groups_artifact_sha256="f" * 64,
            **verify_kwargs,
        )
    verified = verify_vru_causal_shot_validity_training_extension_export(
        chain.index,
        expected_artifact_sha256=chain.index["artifact_sha256"],
        expected_source_groups_artifact_sha256=FROZEN_SOURCE_GROUP_ARTIFACT_SHA256,
        **verify_kwargs,
    )
    assert verified.index == chain.index
    assert verified.resolved_examples == chain.resolved_examples
    v1_after = {filename: hashlib.sha256(path.read_bytes()).hexdigest() for filename, path in frozen_v1_paths.items()}
    assert v1_after == v1_before


def test_generation_target_guard_rejects_nested_casefold_symlink_and_hardlink_aliases(
    tmp_path: Path,
) -> None:
    protected_root = tmp_path / "protected"
    protected_root.mkdir()
    protected_file = protected_root / "input.json"
    protected_file.write_text("input", encoding="utf-8")

    nested_target = protected_root / "generation"
    with pytest.raises(ValueError, match="aliases or nests"):
        export_script._require_generation_paths_disjoint(
            target_dir=nested_target,
            outputs=(nested_target / "index.json",),
            protected_inputs=(protected_root,),
        )

    ancestor_target = tmp_path / "ancestor"
    with pytest.raises(ValueError, match="aliases or nests"):
        export_script._require_generation_paths_disjoint(
            target_dir=ancestor_target,
            outputs=(ancestor_target / "index.json",),
            protected_inputs=(ancestor_target / "nested/input.json",),
        )

    casefold_target = tmp_path / "Generation"
    with pytest.raises(ValueError, match="aliases or nests"):
        export_script._require_generation_paths_disjoint(
            target_dir=casefold_target,
            outputs=(casefold_target / "index.json",),
            protected_inputs=(tmp_path / "generation/input.json",),
        )
    with pytest.raises(ValueError, match="casefold-unique"):
        export_script._require_generation_paths_disjoint(
            target_dir=casefold_target,
            outputs=(casefold_target / "Result.json", casefold_target / "result.json"),
            protected_inputs=(),
        )

    symlink_target = tmp_path / "symlink-generation"
    symlink_target.symlink_to(protected_root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        export_script._require_generation_paths_disjoint(
            target_dir=symlink_target,
            outputs=(symlink_target / "index.json",),
            protected_inputs=(),
        )

    hardlink_target = tmp_path / "hardlink-generation"
    hardlink_target.mkdir()
    hardlink_output = hardlink_target / "index.json"
    os.link(protected_file, hardlink_output)
    with pytest.raises(ValueError, match="hardlink"):
        export_script._require_generation_paths_disjoint(
            target_dir=hardlink_target,
            outputs=(hardlink_output,),
            protected_inputs=(protected_file,),
        )


def test_cli_capped_json_reader_rejects_duplicate_keys_and_oversize_input(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"videos": [], "videos": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        export_script._read_json_object(duplicate, "source manifest")

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (export_script._MAX_JSON_BYTES + 1))
    with pytest.raises(ValueError, match="size limit"):
        export_script._read_json_object(oversized, "source manifest")


def test_preflight_explicit_alias_fails_before_resolver_discovery_or_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_root = tmp_path / "inputs"
    parent_root.mkdir()
    parent_export = parent_root / "training_export.json"
    parent_export.write_text("{}", encoding="utf-8")
    args = argparse.Namespace(
        parent_export=parent_export,
        parent_asset_root=parent_root,
        harwood_selection=parent_root / "selection.json",
        harwood_review_plan=parent_root / "review_plan.json",
        harwood_sealed_review=parent_root / "sealed_review.json",
        harwood_raw_frame_manifest=parent_root / "raw_frames_manifest.json",
        harwood_source_manifest=parent_root / "source_manifest.json",
        source_groups=parent_root / "source_groups.json",
        output=parent_export,
    )
    calls: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> None:
        calls.append("read")
        raise AssertionError("phase-one alias guard must run before content reads")

    monkeypatch.setattr(export_script, "_read_json_object", forbidden)
    monkeypatch.setattr(export_script, "_discover_raw_frame_inputs", forbidden)
    monkeypatch.setattr(export_script, "_resolve_source_video_input", forbidden, raising=False)
    monkeypatch.setattr(
        export_script,
        "resolve_continuous_source_video_path",
        forbidden,
        raising=False,
    )

    with pytest.raises(ValueError, match="aliases|nests protected"):
        export_script._preflight_paths(args)

    assert calls == []


def test_generation_directory_publish_uses_one_replace_and_exact_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "generation"
    encoded = {
        "harwood_candidate_bundle_v2.json": b"candidate\n",
        "harwood_shot_validity_labels_v2.json": b"labels\n",
        "training_export_v2.json": b"index\n",
    }
    replacements: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def record_replace(source: str | Path, destination: str | Path) -> None:
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(export_script.os, "replace", record_replace)
    assert export_script._publish_generation(target_dir=target, encoded_by_name=encoded) is True
    assert len(replacements) == 1
    assert replacements[0][1] == target
    assert replacements[0][0].parent == target.parent
    assert replacements[0][0].name.startswith(f".{target.name}.")
    assert {path.name for path in target.iterdir()} == set(encoded)
    assert {name: (target / name).read_bytes() for name in encoded} == encoded
    assert not list(tmp_path.glob(f".{target.name}.*.stage"))

    assert export_script._publish_generation(target_dir=target, encoded_by_name=encoded) is False
    assert len(replacements) == 1


def test_existing_generation_is_immutable_and_rejects_partial_or_drifted_state(
    tmp_path: Path,
) -> None:
    encoded = {
        "harwood_candidate_bundle_v2.json": b"candidate\n",
        "harwood_shot_validity_labels_v2.json": b"labels\n",
        "training_export_v2.json": b"index\n",
    }
    target = tmp_path / "generation"
    export_script._publish_generation(target_dir=target, encoded_by_name=encoded)
    drifted = target / "harwood_candidate_bundle_v2.json"
    drifted.write_bytes(b"different\n")

    with pytest.raises(FileExistsError, match="immutable generation"):
        export_script._publish_generation(target_dir=target, encoded_by_name=encoded)
    assert drifted.read_bytes() == b"different\n"

    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "training_export_v2.json").write_bytes(encoded["training_export_v2.json"])
    with pytest.raises(FileExistsError, match="immutable generation"):
        export_script._publish_generation(target_dir=partial, encoded_by_name=encoded)
    assert {path.name for path in partial.iterdir()} == {"training_export_v2.json"}

    extra_target = tmp_path / "extra"
    export_script._publish_generation(target_dir=extra_target, encoded_by_name=encoded)
    (extra_target / "unexpected.json").write_bytes(b"extra\n")
    with pytest.raises(FileExistsError, match="immutable generation"):
        export_script._publish_generation(target_dir=extra_target, encoded_by_name=encoded)
    assert (extra_target / "unexpected.json").read_bytes() == b"extra\n"

    symlink_child_target = tmp_path / "symlink-child"
    symlink_child_target.mkdir()
    real_child = tmp_path / "real-candidate.json"
    real_child.write_bytes(encoded["harwood_candidate_bundle_v2.json"])
    (symlink_child_target / "harwood_candidate_bundle_v2.json").symlink_to(real_child)
    for name in ("harwood_shot_validity_labels_v2.json", "training_export_v2.json"):
        (symlink_child_target / name).write_bytes(encoded[name])
    with pytest.raises(FileExistsError, match="immutable generation"):
        export_script._publish_generation(target_dir=symlink_child_target, encoded_by_name=encoded)

    aliased = tmp_path / "aliased"
    aliased.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        export_script._publish_generation(target_dir=aliased, encoded_by_name=encoded)


@pytest.mark.parametrize(
    ("crashpoint", "expect_complete"),
    (
        ("during_stage_write", False),
        ("before_rename", False),
        ("after_rename_before_parent_fsync", True),
    ),
)
def test_generation_crashes_publish_only_absent_or_complete(
    tmp_path: Path,
    crashpoint: str,
    expect_complete: bool,
) -> None:
    target = tmp_path / crashpoint / "generation"
    target.parent.mkdir()
    encoded = {
        "harwood_candidate_bundle_v2.json": b"candidate\n",
        "harwood_shot_validity_labels_v2.json": b"labels\n",
        "training_export_v2.json": b"index\n",
    }
    program = textwrap.dedent(
        """
        import os
        import sys
        from pathlib import Path
        from scripts import export_vru_causal_shot_validity_training_v2 as module

        crashpoint = sys.argv[2]
        def checkpoint(name):
            if name == crashpoint:
                os._exit(91)

        module._publication_checkpoint = checkpoint
        module._publish_generation(
            target_dir=Path(sys.argv[1]),
            encoded_by_name={
                "harwood_candidate_bundle_v2.json": b"candidate\\n",
                "harwood_shot_validity_labels_v2.json": b"labels\\n",
                "training_export_v2.json": b"index\\n",
            },
        )
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", program, os.fspath(target), crashpoint],
        cwd=ROOT,
        check=False,
    )

    assert completed.returncode == 91
    assert target.exists() is expect_complete
    if expect_complete:
        assert target.is_dir() and not target.is_symlink()
        assert {path.name for path in target.iterdir()} == set(encoded)
        assert {name: (target / name).read_bytes() for name in encoded} == encoded
    else:
        assert not target.exists()
