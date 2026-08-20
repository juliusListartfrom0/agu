from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from app.analysis.official_evaluation import verify_raw_only_bundle
from app.analysis.vru_causal_review import (
    build_vru_causal_review_plan,
    seal_vru_causal_review,
)
from app.analysis.vru_causal_training import (
    TRAINING_EXPORT_SCHEMA,
    build_vru_causal_shot_validity_training_export,
    verify_vru_causal_shot_validity_training_export,
)
from scripts import export_vru_causal_shot_validity_training as export_script
from scripts.build_vru_causal_closure_selection import (
    FROZEN_INCLUDED_INPUT_IDS,
    PRIOR_POOL_QUOTA_PAIRS,
    RANKING_NAMESPACE,
    SAMPLE_COUNT,
    SAMPLE_RATE_HZ,
    SOURCE_IDS,
    TEMPORAL_BUCKET_QUOTAS,
    WINDOW_SECONDS,
    canonical_sha256,
)


@dataclass(frozen=True)
class _Fixture:
    selection: Path
    plan: Path
    review: Path
    raw_manifest: Path
    source_manifest: Path
    expected_selection_sha256: str
    expected_plan_sha256: str
    expected_review_sha256: str
    expected_raw_manifest_sha256: str
    videos: tuple[Path, ...]
    first_jpeg: Path


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _artifact_sha256(payload: dict[str, object]) -> str:
    normalized = dict(payload)
    normalized.pop("artifact_sha256", None)
    return _sha256_bytes(
        json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _ranking_sha256(source_sha256: str, anchor_frame: int) -> str:
    return _sha256_bytes(
        f"{RANKING_NAMESPACE}\0{source_sha256}\0{anchor_frame}".encode()
    )


def _selection(
    *, source_manifest_sha256: str, sources: list[dict[str, object]]
) -> dict[str, object]:
    windows: list[dict[str, object]] = []
    buckets = ("early", "early", "early", "middle", "middle", "late", "late", "late")
    for source in sources:
        source_id = str(source["location"])
        for ordinal, bucket in enumerate(buckets, start=1):
            center = float(10 + (ordinal - 1) * 20)
            anchor = round(center * float(source["fps"]))
            windows.append(
                {
                    "review_id": f"closure-{source_id}-{ordinal:04d}",
                    "source_id": source_id,
                    "clip_id": source["clip_id"],
                    "source_video_sha256": source["sha256"],
                    "source_video_filename": Path(str(source["path"])).name,
                    "source_fps": source["fps"],
                    "frame_count": source["frame_count"],
                    "anchor_frame": anchor,
                    "center_seconds": center,
                    "start_seconds": center - WINDOW_SECONDS / 2.0,
                    "end_seconds": center + WINDOW_SECONDS / 2.0,
                    "temporal_bucket": bucket,
                    "ranking_sha256": _ranking_sha256(
                        str(source["sha256"]), anchor
                    ),
                }
            )
    payload: dict[str, object] = {
        "schema_version": "agu.vru-causal-closure-selection.v1",
        "purpose": "offline_causal_closure_selection_freeze",
        "runtime_consumable": False,
        "training_consumable": False,
        "codex_runtime_answer_used": False,
        "formal_evaluation_eligible": False,
        "labels_hidden_from_reviewer": True,
        "prior_labels_used_for_stratified_selection": True,
        "prior_labels_used_as_final_8s_targets": False,
        "selection_provenance_verification": "structural_only_without_source_inputs",
        "source_manifest_sha256": source_manifest_sha256,
        "selection": {
            "ranking_namespace": RANKING_NAMESPACE,
            "ranking_direction": "ascending_sha256",
            "cell_selection_order": (
                "prior_pool_then_available_count_then_early_middle_late"
            ),
            "physical_window_supersession": (
                "highest_formal_version_then_input_id_then_review_id"
            ),
            "source_ids": list(SOURCE_IDS),
            "per_source": 8,
            "window_seconds": WINDOW_SECONDS,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "sample_count": SAMPLE_COUNT,
            "interval_semantics": "half_open",
            "prior_pool_quota_pairs": {
                source_id: list(PRIOR_POOL_QUOTA_PAIRS[source_id])
                for source_id in SOURCE_IDS
            },
            "temporal_bucket_quotas": {
                source_id: dict(TEMPORAL_BUCKET_QUOTAS)
                for source_id in SOURCE_IDS
            },
        },
        "inputs": [
            {
                "input_id": input_id,
                "plan_sha256": _sha256_bytes(f"{input_id}:plan".encode()),
                "sealed_review_sha256": _sha256_bytes(
                    f"{input_id}:review".encode()
                ),
            }
            for input_id in FROZEN_INCLUDED_INPUT_IDS
        ],
        "audit": {
            "included_input_count": len(FROZEN_INCLUDED_INPUT_IDS),
            "excluded_input_ids": ["v22_smoke"],
            "excluded_inputs": [
                {
                    "input_id": "v22_smoke",
                    "plan_sha256": _sha256_bytes(b"v22_smoke:plan"),
                    "sealed_review_sha256": _sha256_bytes(b"v22_smoke:review"),
                }
            ],
            "observation_count": 24,
            "physical_window_count": 24,
            "collision_group_count": 1,
            "conflicting_physical_window_count": 1,
            "candidate_physical_window_count": 24,
            "selectable_physical_window_count": 24,
        },
        "windows": windows,
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    return payload


def _review_rows(plan: dict[str, object]) -> list[dict[str, object]]:
    dispositions = {
        "hazen": [
            "shot",
            "not_a_shot",
            "shot",
            "not_a_shot",
            "shot",
            "shot",
            "shot",
            "shot",
        ],
        "randolph": [
            "not_a_shot",
            "uncertain",
            "shot",
            "shot",
            "not_a_shot",
            "not_a_shot",
            "shot",
            "shot",
        ],
        "vtv": [
            "shot",
            "shot",
            "uncertain",
            "shot",
            "not_a_shot",
            "not_a_shot",
            "not_a_shot",
            "shot",
        ],
    }
    rows: list[dict[str, object]] = []
    for example in plan["examples"]:  # type: ignore[index]
        review_id = str(example["review_id"])
        source_id = review_id.split("-")[1]
        ordinal = int(review_id.rsplit("-", 1)[1])
        disposition = dispositions[source_id][ordinal - 1]
        if disposition == "shot":
            rows.append(
                {
                    "review_id": review_id,
                    "shot_sequence": "shot",
                    "release_position": 20,
                    "rim_position": 40,
                    "outcome": "made" if ordinal % 2 else "missed",
                    "confidence": "high",
                    "evidence": {
                        "controlled_ball_before_release": True,
                        "ball_separated_from_hand": True,
                        "ball_progresses_toward_rim": True,
                        "rim_proximity_visible": True,
                        "rim_contact_visible": True,
                    },
                    "notes": "review-only truth that must not enter geometry",
                }
            )
        elif disposition == "not_a_shot":
            rows.append(
                {
                    "review_id": review_id,
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
                    "notes": "review-only negative note",
                }
            )
        else:
            rows.append(
                {
                    "review_id": review_id,
                    "shot_sequence": "uncertain",
                    "release_position": None,
                    "rim_position": None,
                    "outcome": "unknown",
                    "confidence": "medium",
                    "evidence": {
                        "controlled_ball_before_release": False,
                        "ball_separated_from_hand": False,
                        "ball_progresses_toward_rim": False,
                        "rim_proximity_visible": False,
                        "rim_contact_visible": False,
                    },
                    "notes": "must be excluded, never mapped negative",
                }
            )
    return rows


def _fixture(tmp_path: Path) -> _Fixture:
    tmp_path.mkdir(parents=True, exist_ok=True)
    videos: list[Path] = []
    source_rows: list[dict[str, object]] = []
    for source_id in SOURCE_IDS:
        video = tmp_path / f"{source_id}.webm"
        video.write_bytes(f"fixture-video:{source_id}".encode())
        videos.append(video)
        source_rows.append(
            {
                "location": source_id,
                "clip_id": f"{source_id}_fixture",
                "path": f"nested/../{video.name}",
                "size_bytes": video.stat().st_size,
                "sha256": _file_sha256(video),
                "fps": 8.0,
                "frame_count": 1600,
                "duration_seconds": 200.0,
            }
        )
    source_manifest = tmp_path / "source_manifest.json"
    _write_json(
        source_manifest,
        {
            "schema_version": "agu.vru-basketball-source-manifest.v1",
            "purpose": "offline_test_fixture",
            "runtime_consumable": False,
            "codex_runtime_answer_used": False,
            "videos": source_rows,
        },
    )
    selection_payload = _selection(
        source_manifest_sha256=_file_sha256(source_manifest),
        sources=source_rows,
    )
    selection = tmp_path / "selection.json"
    _write_json(selection, selection_payload)

    examples: list[dict[str, object]] = []
    for window in selection_payload["windows"]:  # type: ignore[index]
        start = float(window["start_seconds"])
        fps = float(window["source_fps"])
        indexes = [
            round((start + sample / SAMPLE_RATE_HZ) * fps)
            for sample in range(SAMPLE_COUNT)
        ]
        examples.append(
            {
                "review_id": window["review_id"],
                "source_video_sha256": window["source_video_sha256"],
                "source_video_filename": window["source_video_filename"],
                "source_fps": fps,
                "frame_count": window["frame_count"],
                "frame_indexes": indexes,
                "frame_sha256s": {
                    str(index): _sha256_bytes(
                        f"{window['review_id']}:{index}:raw".encode()
                    )
                    for index in indexes
                },
            }
        )
    plan_payload = build_vru_causal_review_plan(
        source_manifest_sha256=_file_sha256(source_manifest),
        examples=examples,
    )
    plan = tmp_path / "review_plan.json"
    _write_json(plan, plan_payload)

    review_payload = seal_vru_causal_review(
        {"reviewer": "fixture", "reviews": _review_rows(plan_payload)},
        plan=plan_payload,
    )
    review = tmp_path / "sealed_review.json"
    _write_json(review, review_payload)

    frame_root = tmp_path / "raw_frames"
    frame_root.mkdir()
    frame_rows: list[dict[str, object]] = []
    first_jpeg: Path | None = None
    for example in plan_payload["examples"]:
        review_id = str(example["review_id"])
        for position, frame_index in enumerate(example["frame_indexes"]):
            relative_path = f"{review_id}_{position:03d}_{frame_index}.jpg"
            jpeg_path = frame_root / relative_path
            jpeg_path.write_bytes(
                b"\xff\xd8" + f"{review_id}:{position}:{frame_index}".encode() + b"\xff\xd9"
            )
            first_jpeg = first_jpeg or jpeg_path
            frame_rows.append(
                {
                    "review_id": review_id,
                    "position": position,
                    "frame_index": frame_index,
                    "raw_frame_sha256": example["frame_sha256s"][str(frame_index)],
                    "relative_path": relative_path,
                    "jpeg_bytes": jpeg_path.stat().st_size,
                    "jpeg_sha256": _file_sha256(jpeg_path),
                }
            )
    raw_payload: dict[str, object] = {
        "schema_version": "agu.vru-causal-review-frame-manifest.v1",
        "purpose": "offline_codex_visual_review_of_original_vru_frames",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "labels_hidden_from_reviewer": True,
        "source_manifest_sha256": _file_sha256(source_manifest),
        "review_plan_sha256": plan_payload["artifact_sha256"],
        "frame_root": frame_root.name,
        "jpeg_quality": 95,
        "frame_count": len(frame_rows),
        "frames": frame_rows,
    }
    raw_payload["artifact_sha256"] = _artifact_sha256(raw_payload)
    raw_manifest = tmp_path / "raw_frames_manifest.json"
    _write_json(raw_manifest, raw_payload)
    assert first_jpeg is not None
    return _Fixture(
        selection=selection,
        plan=plan,
        review=review,
        raw_manifest=raw_manifest,
        source_manifest=source_manifest,
        expected_selection_sha256=str(selection_payload["artifact_sha256"]),
        expected_plan_sha256=str(plan_payload["artifact_sha256"]),
        expected_review_sha256=str(review_payload["artifact_sha256"]),
        expected_raw_manifest_sha256=str(raw_payload["artifact_sha256"]),
        videos=tuple(videos),
        first_jpeg=first_jpeg,
    )


def _build(inputs: _Fixture):
    return build_vru_causal_shot_validity_training_export(
        selection_path=inputs.selection,
        expected_selection_artifact_sha256=inputs.expected_selection_sha256,
        review_plan_path=inputs.plan,
        expected_review_plan_artifact_sha256=inputs.expected_plan_sha256,
        sealed_review_path=inputs.review,
        expected_sealed_review_artifact_sha256=inputs.expected_review_sha256,
        raw_frame_manifest_path=inputs.raw_manifest,
        expected_raw_frame_manifest_artifact_sha256=(
            inputs.expected_raw_manifest_sha256
        ),
        source_manifest_path=inputs.source_manifest,
    )


def test_export_maps_24_windows_to_22_label_hidden_examples(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)

    artifacts = _build(inputs)
    export = verify_vru_causal_shot_validity_training_export(artifacts.index)

    assert export["schema_version"] == TRAINING_EXPORT_SCHEMA
    assert export["purpose"] == "development_diagnostic_only"
    assert export["runtime_consumable"] is False
    assert export["training_consumable"] is True
    assert export["formal_evaluation_eligible"] is False
    assert export["counts"] == {
        "selected": 24,
        "exported": 22,
        "positive": 14,
        "negative": 8,
        "excluded_uncertain": 2,
    }
    assert export["excluded_review_ids"] == [
        "closure-randolph-0002",
        "closure-vtv-0003",
    ]
    assert len(export["examples"]) == 22
    assert sum(row["event_present"] for row in export["examples"]) == 14
    assert {source["source_id"] for source in export["sources"]} == set(SOURCE_IDS)
    assert export["mapping_policy"] == {
        "shot": True,
        "not_a_shot": False,
        "uncertain": "excluded",
        "outcome_used_for_shot_validity": False,
        "target_conditioned_anchor_used": False,
    }
    assert export["training_routes"] == {
        "scene_representation": True,
        "video_representation": True,
        "traditional_feature_extra_trees": False,
        "runtime_inference": False,
    }

    label_rows = [
        row
        for payload in artifacts.label_files.values()
        for row in payload["examples"]
    ]
    candidate_ids = {
        event["event_id"]
        for payload in artifacts.candidate_bundles.values()
        for event in payload["events"]
    }
    assert len(candidate_ids) == 24
    assert set(export["excluded_review_ids"]) <= candidate_ids
    assert len(label_rows) == 22
    assert all(set(row) == {"event_id", "event_present"} for row in label_rows)
    assert {row["event_id"] for row in label_rows}.isdisjoint(
        export["excluded_review_ids"]
    )


def test_candidate_geometry_has_no_target_or_reviewed_anchor_leakage(
    tmp_path: Path,
) -> None:
    artifacts = _build(_fixture(tmp_path))

    for bundle in artifacts.candidate_bundles.values():
        serialized = json.dumps(bundle, sort_keys=True).lower()
        assert "source_review" not in serialized
        assert "shot_sequence" not in serialized
        assert "event_present" not in serialized
        assert "rim_frame" not in serialized
        assert "candidate_event_frame" not in serialized
        for event in bundle["events"]:
            assert event["release_frame"] is None
            assert event["outcome_frame"] is None
            assert event["outcome"] is None
            assert event["reviewer"] is None
            assert event["evidence"] == []
            assert event["start_frame"] < event["end_frame"]


def test_review_outcome_and_positions_do_not_change_candidate_geometry(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    before = _build(inputs)
    plan_payload = json.loads(inputs.plan.read_text(encoding="utf-8"))
    review_payload = json.loads(inputs.review.read_text(encoding="utf-8"))
    review_inputs = []
    for row in review_payload["reviews"]:
        normalized = {
            key: row[key]
            for key in (
                "review_id",
                "shot_sequence",
                "release_position",
                "rim_position",
                "outcome",
                "confidence",
                "evidence",
                "notes",
            )
        }
        if normalized["shot_sequence"] == "shot":
            normalized["release_position"] = 19
            normalized["rim_position"] = 41
            normalized["outcome"] = (
                "missed" if normalized["outcome"] == "made" else "made"
            )
        review_inputs.append(normalized)
    changed = seal_vru_causal_review(
        {"reviewer": "fixture", "reviews": review_inputs},
        plan=plan_payload,
    )
    _write_json(inputs.review, changed)

    after = _build(
        replace(
            inputs,
            expected_review_sha256=str(changed["artifact_sha256"]),
        )
    )

    assert before.candidate_bundles == after.candidate_bundles
    assert [
        source["candidate_bundle"]["bundle_sha256"]
        for source in before.index["sources"]
    ] == [
        source["candidate_bundle"]["bundle_sha256"]
        for source in after.index["sources"]
    ]


def test_export_rejects_resealed_label_swap_against_frozen_review_sha(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    plan = json.loads(inputs.plan.read_text(encoding="utf-8"))
    rows = _review_rows(plan)
    shot = next(
        row
        for row in rows
        if row["review_id"].startswith("closure-hazen-")
        and row["shot_sequence"] == "shot"
    )
    negative = next(
        row
        for row in rows
        if row["review_id"].startswith("closure-hazen-")
        and row["shot_sequence"] == "not_a_shot"
    )
    review_fields = set(shot) - {"review_id"}
    shot_values = {field: copy.deepcopy(shot[field]) for field in review_fields}
    negative_values = {
        field: copy.deepcopy(negative[field]) for field in review_fields
    }
    shot.update(negative_values)
    negative.update(shot_values)
    resealed = seal_vru_causal_review(
        {"reviewer": "fixture", "reviews": rows},
        plan=plan,
    )
    assert resealed["artifact_sha256"] != inputs.expected_review_sha256
    _write_json(inputs.review, resealed)

    with pytest.raises(ValueError, match="expected sealed review.*SHA|review.*receipt"):
        _build(inputs)


def test_export_rejects_resealed_plan_against_frozen_plan_sha(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    plan = json.loads(inputs.plan.read_text(encoding="utf-8"))
    plan["review_rules"]["release"] += "; resealed mutation"
    plan["artifact_sha256"] = _artifact_sha256(plan)
    _write_json(inputs.plan, plan)

    sealed = json.loads(inputs.review.read_text(encoding="utf-8"))
    review_fields = (
        "review_id",
        "shot_sequence",
        "release_position",
        "rim_position",
        "outcome",
        "confidence",
        "evidence",
        "notes",
    )
    resealed_review = seal_vru_causal_review(
        {
            "reviewer": sealed["reviewer"],
            "reviews": [
                {field: row[field] for field in review_fields}
                for row in sealed["reviews"]
            ],
        },
        plan=plan,
    )
    _write_json(inputs.review, resealed_review)
    raw_manifest = json.loads(inputs.raw_manifest.read_text(encoding="utf-8"))
    raw_manifest["review_plan_sha256"] = plan["artifact_sha256"]
    raw_manifest["artifact_sha256"] = _artifact_sha256(raw_manifest)
    _write_json(inputs.raw_manifest, raw_manifest)

    with pytest.raises(ValueError, match="expected review plan.*SHA|plan.*receipt"):
        _build(inputs)


def test_export_rejects_resealed_jpeg_manifest_against_frozen_raw_manifest_sha(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path)
    inputs.first_jpeg.write_bytes(b"\xff\xd8unrelated-resealed-jpeg\xff\xd9")
    raw_manifest = json.loads(inputs.raw_manifest.read_text(encoding="utf-8"))
    first_row = raw_manifest["frames"][0]
    first_row["jpeg_bytes"] = inputs.first_jpeg.stat().st_size
    first_row["jpeg_sha256"] = _file_sha256(inputs.first_jpeg)
    raw_manifest["artifact_sha256"] = _artifact_sha256(raw_manifest)
    assert raw_manifest["artifact_sha256"] != inputs.expected_raw_manifest_sha256
    _write_json(inputs.raw_manifest, raw_manifest)

    with pytest.raises(ValueError, match="expected raw frame manifest.*SHA|frame.*receipt"):
        _build(inputs)


@pytest.mark.parametrize(
    ("target_source", "target_asset", "donor_source", "donor_asset"),
    (
        (1, "candidate_bundle", 0, "candidate_bundle"),
        (1, "labels", 0, "labels"),
        (1, "labels", 0, "candidate_bundle"),
    ),
)
def test_export_verifier_rejects_rehashed_child_filename_alias(
    tmp_path: Path,
    target_source: int,
    target_asset: str,
    donor_source: int,
    donor_asset: str,
) -> None:
    export = copy.deepcopy(_build(_fixture(tmp_path)).index)
    export.pop("artifact_sha256")
    donor_name = export["sources"][donor_source][donor_asset]["filename"]
    export["sources"][target_source][target_asset]["filename"] = donor_name.upper()
    export["artifact_sha256"] = _artifact_sha256(export)

    with pytest.raises(ValueError, match="child.*filename|asset.*alias|globally unique"):
        verify_vru_causal_shot_validity_training_export(export)


def test_export_fails_closed_on_every_input_chain_tamper(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    json_inputs = (
        inputs.selection,
        inputs.plan,
        inputs.review,
        inputs.raw_manifest,
    )
    for path in json_inputs:
        original = path.read_bytes()
        payload = json.loads(original)
        payload["purpose"] = "tampered"
        _write_json(path, payload)
        with pytest.raises(ValueError):
            _build(inputs)
        path.write_bytes(original)

    jpeg_bytes = inputs.first_jpeg.read_bytes()
    inputs.first_jpeg.write_bytes(jpeg_bytes + b"tampered")
    with pytest.raises(ValueError, match="JPEG.*mismatch"):
        _build(inputs)
    inputs.first_jpeg.write_bytes(jpeg_bytes)

    video_bytes = inputs.videos[0].read_bytes()
    inputs.videos[0].write_bytes(video_bytes + b"tampered")
    with pytest.raises(ValueError, match="source video.*mismatch"):
        _build(inputs)
    inputs.videos[0].write_bytes(video_bytes)


def test_export_rejects_frame_manifest_coverage_and_path_escape(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    original = json.loads(inputs.raw_manifest.read_text(encoding="utf-8"))

    missing = copy.deepcopy(original)
    missing["frames"].pop()
    missing["frame_count"] -= 1
    missing["artifact_sha256"] = _artifact_sha256(missing)
    _write_json(inputs.raw_manifest, missing)
    with pytest.raises(ValueError, match="exactly cover|coverage"):
        _build(inputs)

    escaped = copy.deepcopy(original)
    escaped["frames"][0]["relative_path"] = "../escape.jpg"
    escaped["artifact_sha256"] = _artifact_sha256(escaped)
    _write_json(inputs.raw_manifest, escaped)
    with pytest.raises(ValueError, match="outside|path"):
        _build(inputs)


def test_uncertain_cannot_be_reclassified_as_negative(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    plan = json.loads(inputs.plan.read_text(encoding="utf-8"))
    sealed = json.loads(inputs.review.read_text(encoding="utf-8"))
    rows = []
    changed = False
    for row in sealed["reviews"]:
        normalized = {
            key: row[key]
            for key in (
                "review_id",
                "shot_sequence",
                "release_position",
                "rim_position",
                "outcome",
                "confidence",
                "evidence",
                "notes",
            )
        }
        if not changed and normalized["shot_sequence"] == "uncertain":
            changed = True
            normalized.update(
                {
                    "shot_sequence": "not_a_shot",
                    "release_position": None,
                    "rim_position": None,
                    "outcome": "not_applicable",
                    "confidence": "high",
                    "evidence": {key: False for key in normalized["evidence"]},
                }
            )
        rows.append(normalized)
    _write_json(
        inputs.review,
        seal_vru_causal_review(
            {"reviewer": "fixture", "reviews": rows}, plan=plan
        ),
    )

    changed_review = json.loads(inputs.review.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="14.*8.*2|frozen.*mapping"):
        _build(
            replace(
                inputs,
                expected_review_sha256=str(changed_review["artifact_sha256"]),
            )
        )


def test_export_is_deterministic_and_expected_sha_bound(tmp_path: Path) -> None:
    artifacts = _build(_fixture(tmp_path))

    repeated = _build(_fixture(tmp_path / "repeated"))

    assert artifacts.index == repeated.index
    assert artifacts.candidate_bundles == repeated.candidate_bundles
    assert artifacts.label_files == repeated.label_files
    assert verify_vru_causal_shot_validity_training_export(
        artifacts.index,
        expected_artifact_sha256=artifacts.index["artifact_sha256"],
    ) == artifacts.index
    with pytest.raises(ValueError, match="expected.*SHA"):
        verify_vru_causal_shot_validity_training_export(
            artifacts.index,
            expected_artifact_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    "alias_kind",
    (
        "direct",
        "output_symlink",
        "child_symlink",
        "output_hardlink",
        "child_hardlink",
        "output_case_variant",
    ),
)
def test_export_cli_rejects_output_input_alias_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_kind: str,
) -> None:
    inputs = {
        "selection": tmp_path / "selection.json",
        "plan": tmp_path / "plan.json",
        "review": tmp_path / "review.json",
        "raw-frame-manifest": tmp_path / "raw.json",
        "source-manifest": tmp_path / "sources.json",
    }
    for name, path in inputs.items():
        path.write_text(f"{name}\n", encoding="utf-8")
    output = tmp_path / "out" / "training_export_v1.json"
    output.parent.mkdir()
    if alias_kind == "direct":
        output = inputs["selection"]
    elif alias_kind == "output_symlink":
        output.symlink_to(inputs["selection"])
    elif alias_kind == "child_symlink":
        (output.parent / "hazen_candidate_bundle_v1.json").symlink_to(
            inputs["selection"]
        )
    elif alias_kind == "output_hardlink":
        os.link(inputs["selection"], output)
    elif alias_kind == "child_hardlink":
        os.link(
            inputs["selection"],
            output.parent / "hazen_candidate_bundle_v1.json",
        )
    else:
        output = inputs["selection"].with_name("SELECTION.JSON")
    argv = ["export_vru_causal_shot_validity_training.py"]
    for name, path in inputs.items():
        argv.extend((f"--{name}", str(path)))
    argv.extend(
        (
            "--expected-selection-artifact-sha256",
            "a" * 64,
            "--expected-review-plan-artifact-sha256",
            "b" * 64,
            "--expected-sealed-review-artifact-sha256",
            "c" * 64,
            "--expected-raw-frame-manifest-artifact-sha256",
            "d" * 64,
            "--output",
            str(output),
        )
    )
    monkeypatch.setattr(sys, "argv", argv)

    def reject_build(**_kwargs: object):
        raise AssertionError("alias must be rejected before reading or building")

    monkeypatch.setattr(
        export_script,
        "build_vru_causal_shot_validity_training_export",
        reject_build,
    )
    with pytest.raises(ValueError, match="output.*input"):
        export_script.main()


def test_export_cli_rejects_casefold_output_child_collision_before_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = {
        "selection": tmp_path / "selection.json",
        "plan": tmp_path / "plan.json",
        "review": tmp_path / "review.json",
        "raw-frame-manifest": tmp_path / "raw.json",
        "source-manifest": tmp_path / "sources.json",
    }
    for path in inputs.values():
        path.write_text("not read\n", encoding="utf-8")
    output = tmp_path / "out" / "Hazen_candidate_bundle_v1.json"
    argv = ["export_vru_causal_shot_validity_training.py"]
    for name, path in inputs.items():
        argv.extend((f"--{name}", str(path)))
    argv.extend(
        (
            "--expected-selection-artifact-sha256",
            "a" * 64,
            "--expected-review-plan-artifact-sha256",
            "b" * 64,
            "--expected-sealed-review-artifact-sha256",
            "c" * 64,
            "--expected-raw-frame-manifest-artifact-sha256",
            "d" * 64,
            "--output",
            str(output),
        )
    )
    monkeypatch.setattr(sys, "argv", argv)

    def reject_build(**_kwargs: object):
        raise AssertionError("output collision must fail before building")

    monkeypatch.setattr(
        export_script,
        "build_vru_causal_shot_validity_training_export",
        reject_build,
    )

    with pytest.raises(ValueError, match="output.*unique|collision|disjoint"):
        export_script.main()


def test_export_cli_rejects_output_alias_to_derived_source_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _fixture(tmp_path)
    original_video = inputs.videos[0].read_bytes()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_vru_causal_shot_validity_training.py",
            "--selection",
            str(inputs.selection),
            "--expected-selection-artifact-sha256",
            inputs.expected_selection_sha256,
            "--expected-review-plan-artifact-sha256",
            inputs.expected_plan_sha256,
            "--expected-sealed-review-artifact-sha256",
            inputs.expected_review_sha256,
            "--expected-raw-frame-manifest-artifact-sha256",
            inputs.expected_raw_manifest_sha256,
            "--plan",
            str(inputs.plan),
            "--review",
            str(inputs.review),
            "--raw-frame-manifest",
            str(inputs.raw_manifest),
            "--source-manifest",
            str(inputs.source_manifest),
            "--output",
            str(inputs.videos[0]),
        ],
    )

    with pytest.raises(ValueError, match="output.*input"):
        export_script.main()

    assert inputs.videos[0].read_bytes() == original_video


@pytest.mark.parametrize("alias_kind", ("hardlink", "case_variant"))
def test_export_cli_rejects_identity_alias_to_derived_source_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_kind: str,
) -> None:
    inputs = _fixture(tmp_path)
    original_video = inputs.videos[0].read_bytes()
    if alias_kind == "hardlink":
        output = tmp_path / "video-output-hardlink.json"
        os.link(inputs.videos[0], output)
    else:
        output = inputs.videos[0].with_name(inputs.videos[0].name.upper())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_vru_causal_shot_validity_training.py",
            "--selection",
            str(inputs.selection),
            "--expected-selection-artifact-sha256",
            inputs.expected_selection_sha256,
            "--expected-review-plan-artifact-sha256",
            inputs.expected_plan_sha256,
            "--expected-sealed-review-artifact-sha256",
            inputs.expected_review_sha256,
            "--expected-raw-frame-manifest-artifact-sha256",
            inputs.expected_raw_manifest_sha256,
            "--plan",
            str(inputs.plan),
            "--review",
            str(inputs.review),
            "--raw-frame-manifest",
            str(inputs.raw_manifest),
            "--source-manifest",
            str(inputs.source_manifest),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(ValueError, match="output.*input|alias"):
        export_script.main()

    assert inputs.videos[0].read_bytes() == original_video


def test_export_json_writer_stages_fsyncs_and_atomically_replaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "export.json"
    destination.write_text("old artifact\n", encoding="utf-8")
    original_replace = Path.replace
    events: list[str] = []

    def observe_fsync(file_descriptor: int) -> None:
        assert file_descriptor >= 0
        events.append("fsync")

    def observe_replace(source: Path, target: Path) -> Path:
        assert source.parent == destination.parent
        assert source.suffix == ".tmp"
        assert json.loads(source.read_text(encoding="utf-8")) == {"new": True}
        assert destination.read_text(encoding="utf-8") == "old artifact\n"
        events.append("replace")
        return original_replace(source, target)

    monkeypatch.setattr(export_script.os, "fsync", observe_fsync)
    monkeypatch.setattr(Path, "replace", observe_replace)

    export_script._write_json_atomic(destination, {"new": True})

    assert events == ["fsync", "replace"]
    assert json.loads(destination.read_text(encoding="utf-8")) == {"new": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_export_cli_writes_hash_bound_self_consistent_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _fixture(tmp_path / "inputs")
    output = tmp_path / "export" / "training_export_v1.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_vru_causal_shot_validity_training.py",
            "--selection",
            str(inputs.selection),
            "--expected-selection-artifact-sha256",
            inputs.expected_selection_sha256,
            "--expected-review-plan-artifact-sha256",
            inputs.expected_plan_sha256,
            "--expected-sealed-review-artifact-sha256",
            inputs.expected_review_sha256,
            "--expected-raw-frame-manifest-artifact-sha256",
            inputs.expected_raw_manifest_sha256,
            "--plan",
            str(inputs.plan),
            "--review",
            str(inputs.review),
            "--raw-frame-manifest",
            str(inputs.raw_manifest),
            "--source-manifest",
            str(inputs.source_manifest),
            "--output",
            str(output),
        ],
    )

    assert export_script.main() == 0

    export = verify_vru_causal_shot_validity_training_export(
        json.loads(output.read_text(encoding="utf-8"))
    )
    for source in export["sources"]:
        candidate_asset = source["candidate_bundle"]
        candidate_path = output.parent / candidate_asset["filename"]
        assert candidate_path.stat().st_size == candidate_asset["size_bytes"]
        assert _file_sha256(candidate_path) == candidate_asset["sha256"]
        candidate = verify_raw_only_bundle(
            json.loads(candidate_path.read_text(encoding="utf-8"))
        )
        assert candidate.bundle_sha256 == candidate_asset["bundle_sha256"]

        label_asset = source["labels"]
        label_path = output.parent / label_asset["filename"]
        assert label_path.stat().st_size == label_asset["size_bytes"]
        assert _file_sha256(label_path) == label_asset["sha256"]
    assert not list(output.parent.glob(".*.tmp"))
