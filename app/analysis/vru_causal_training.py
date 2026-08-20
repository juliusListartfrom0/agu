"""Training-only export boundary for the sealed VRU causal closure.

The source selection deliberately remains non-training-consumable.  This module
is the explicit, hash-bound conversion boundary that validates the complete
selection/review/frame/source chain and emits only label-hidden window geometry
plus separate binary shot-validity labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.analysis.official_evaluation import verify_raw_only_bundle
from app.analysis.schemas import GameEventResponse
from app.analysis.shot_validity import SHOT_VALIDITY_LABEL_SCHEMA
from app.analysis.vru_causal_review import (
    verify_vru_causal_review,
    verify_vru_causal_review_plan,
)

TRAINING_EXPORT_SCHEMA = "agu.vru-causal-shot-validity-training-export.v1"
FRAME_MANIFEST_SCHEMA = "agu.vru-causal-review-frame-manifest.v1"
SOURCE_MANIFEST_SCHEMA = "agu.vru-basketball-source-manifest.v1"
EXPORT_PURPOSE = "development_diagnostic_only"
SOURCE_IDS = ("hazen", "randolph", "vtv")
SAMPLE_COUNT = 64
EXPECTED_EXCLUDED_REVIEW_IDS = (
    "closure-randolph-0002",
    "closure-vtv-0003",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_EXPORT_FIELDS = {
    "artifact_sha256",
    "codex_runtime_answer_used",
    "counts",
    "examples",
    "excluded_review_ids",
    "formal_evaluation_eligible",
    "mapping_policy",
    "purpose",
    "runtime_consumable",
    "schema_version",
    "source_frame_manifest",
    "source_manifest",
    "source_review",
    "source_review_plan",
    "source_selection",
    "sources",
    "training_consumable",
    "training_routes",
}
_MAPPING_POLICY = {
    "shot": True,
    "not_a_shot": False,
    "uncertain": "excluded",
    "outcome_used_for_shot_validity": False,
    "target_conditioned_anchor_used": False,
}
_TRAINING_ROUTES = {
    "scene_representation": True,
    "video_representation": True,
    "traditional_feature_extra_trees": False,
    "runtime_inference": False,
}
_SOURCE_FIELDS = {
    "candidate_bundle",
    "example_count",
    "labels",
    "negative_count",
    "positive_count",
    "source_id",
    "source_video_filename",
    "source_video_sha256",
    "source_video_size_bytes",
}
_CANDIDATE_ASSET_FIELDS = {
    "bundle_sha256",
    "filename",
    "sha256",
    "size_bytes",
}
_LABEL_ASSET_FIELDS = {"filename", "sha256", "size_bytes"}
_EXAMPLE_FIELDS = {
    "candidate_bundle_sha256",
    "event_id",
    "event_present",
    "source_video_sha256",
}
_RAW_FRAME_FIELDS = {
    "frame_index",
    "jpeg_bytes",
    "jpeg_sha256",
    "position",
    "raw_frame_sha256",
    "relative_path",
    "review_id",
}
_RAW_MANIFEST_FIELDS = {
    "artifact_sha256",
    "codex_runtime_answer_used",
    "frame_count",
    "frame_root",
    "frames",
    "jpeg_quality",
    "labels_hidden_from_reviewer",
    "purpose",
    "review_plan_sha256",
    "runtime_consumable",
    "schema_version",
    "source_manifest_sha256",
}
_PLAN_FIELDS = {
    "annotation_scope",
    "artifact_sha256",
    "codex_runtime_answer_used",
    "examples",
    "labels_hidden_from_reviewer",
    "purpose",
    "review_rules",
    "reviewer_visible_fields",
    "runtime_consumable",
    "schema_version",
    "source_manifest_sha256",
    "source_video_sha256s",
}
_PLAN_EXAMPLE_FIELDS = {
    "frame_count",
    "frame_indexes",
    "frame_sha256s",
    "review_id",
    "source_fps",
    "source_video_filename",
    "source_video_sha256",
}
_EXPECTED_SOURCE_COUNTS = {
    "hazen": (8, 6, 2),
    "randolph": (7, 4, 3),
    "vtv": (7, 4, 3),
}
_FIXED_COMPLETED_AT = "2026-08-15T00:00:00+00:00"


@dataclass(frozen=True)
class VruCausalShotValidityTrainingArtifacts:
    """In-memory export plus its separately writable child artifacts."""

    index: dict[str, Any]
    candidate_bundles: dict[str, dict[str, Any]]
    label_files: dict[str, dict[str, Any]]
    validated_input_paths: tuple[Path, ...]


@dataclass(frozen=True)
class _Source:
    source_id: str
    clip_id: str
    filename: str
    path: Path
    sha256: str
    size_bytes: int
    fps: float
    frame_count: int


def build_vru_causal_shot_validity_training_export(
    *,
    selection_path: Path,
    expected_selection_artifact_sha256: str,
    review_plan_path: Path,
    expected_review_plan_artifact_sha256: str,
    sealed_review_path: Path,
    expected_sealed_review_artifact_sha256: str,
    raw_frame_manifest_path: Path,
    expected_raw_frame_manifest_artifact_sha256: str,
    source_manifest_path: Path,
) -> VruCausalShotValidityTrainingArtifacts:
    """Validate the complete frozen chain and build deterministic derivatives."""

    # The legacy selection owner remains a CLI module.  Import it only on the
    # export-building path so downstream consumers can use the pure index
    # verifier without depending on the scripts package.
    from scripts.build_vru_causal_closure_selection import (
        verify_causal_closure_selection,
    )

    expected_selection_sha = _require_sha(
        expected_selection_artifact_sha256,
        "expected selection artifact",
    )
    expected_plan_sha = _require_sha(
        expected_review_plan_artifact_sha256,
        "expected review plan artifact",
    )
    expected_review_sha = _require_sha(
        expected_sealed_review_artifact_sha256,
        "expected sealed review artifact",
    )
    expected_raw_manifest_sha = _require_sha(
        expected_raw_frame_manifest_artifact_sha256,
        "expected raw frame manifest artifact",
    )
    selection_payload, selection_file_sha = _read_mapping_with_sha(
        selection_path, "causal closure selection"
    )
    selection = verify_causal_closure_selection(selection_payload)
    if selection["artifact_sha256"] != expected_selection_sha:
        raise ValueError("expected selection artifact SHA-256 does not match")

    source_manifest, source_manifest_file_sha = _read_mapping_with_sha(
        source_manifest_path, "VRU source manifest"
    )
    if selection["source_manifest_sha256"] != source_manifest_file_sha:
        raise ValueError("selection/source manifest file SHA-256 mismatch")
    sources = _verify_source_manifest(
        source_manifest,
        manifest_path=source_manifest_path,
        selection=selection,
    )

    plan_payload, plan_file_sha = _read_mapping_with_sha(
        review_plan_path, "VRU causal review plan"
    )
    plan = verify_vru_causal_review_plan(plan_payload)
    if plan["artifact_sha256"] != expected_plan_sha:
        raise ValueError("expected review plan artifact SHA-256 does not match")
    _verify_selection_plan(
        selection=selection,
        plan=plan,
        source_manifest_sha256=source_manifest_file_sha,
    )

    review_payload, review_file_sha = _read_mapping_with_sha(
        sealed_review_path, "VRU sealed causal review"
    )
    review = verify_vru_causal_review(review_payload, plan=plan)
    if review["artifact_sha256"] != expected_review_sha:
        raise ValueError("expected sealed review artifact SHA-256 does not match")

    raw_payload, raw_file_sha = _read_mapping_with_sha(
        raw_frame_manifest_path, "VRU raw frame manifest"
    )
    raw_manifest, review_jpeg_paths = _verify_raw_frame_manifest(
        raw_payload,
        manifest_path=raw_frame_manifest_path,
        plan=plan,
        source_manifest_sha256=source_manifest_file_sha,
    )
    if raw_manifest["artifact_sha256"] != expected_raw_manifest_sha:
        raise ValueError("expected raw frame manifest artifact SHA-256 does not match")

    reviews_by_id = {str(row["review_id"]): row for row in review["reviews"]}
    positives = sorted(
        review_id
        for review_id, row in reviews_by_id.items()
        if row["shot_sequence"] == "shot"
    )
    negatives = sorted(
        review_id
        for review_id, row in reviews_by_id.items()
        if row["shot_sequence"] == "not_a_shot"
    )
    excluded = sorted(
        review_id
        for review_id, row in reviews_by_id.items()
        if row["shot_sequence"] == "uncertain"
    )
    if (
        len(positives) != 14
        or len(negatives) != 8
        or excluded != list(EXPECTED_EXCLUDED_REVIEW_IDS)
    ):
        raise ValueError(
            "frozen mapping requires 14 positive, 8 negative, and 2 uncertain"
        )

    source_refs = {
        "source_selection": {
            "schema_version": selection["schema_version"],
            "artifact_sha256": selection["artifact_sha256"],
            "file_sha256": selection_file_sha,
            "training_consumable": False,
            "prior_labels_used_for_stratified_selection": True,
            "prior_labels_used_as_final_8s_targets": False,
        },
        "source_review_plan": _artifact_ref(plan, plan_file_sha),
        "source_review": _artifact_ref(review, review_file_sha),
        "source_frame_manifest": _artifact_ref(raw_manifest, raw_file_sha),
        "source_manifest": {
            "schema_version": source_manifest["schema_version"],
            "file_sha256": source_manifest_file_sha,
        },
    }
    plan_by_id = {str(row["review_id"]): row for row in plan["examples"]}
    candidate_bundles: dict[str, dict[str, Any]] = {}
    label_files: dict[str, dict[str, Any]] = {}
    source_rows: list[dict[str, Any]] = []
    index_examples: list[dict[str, Any]] = []
    for source_id in SOURCE_IDS:
        source = sources[source_id]
        planned = [
            plan_by_id[str(window["review_id"])]
            for window in selection["windows"]
            if window["source_id"] == source_id
        ]
        candidate = _build_candidate_bundle(
            source=source,
            plan_rows=planned,
            selection_sha256=selection["artifact_sha256"],
        )
        candidate_filename = f"{source_id}_candidate_bundle_v1.json"
        candidate_bytes = encode_json_artifact(candidate)
        candidate_bundles[candidate_filename] = candidate

        label_examples = [
            {
                "event_id": str(row["review_id"]),
                "event_present": (
                    reviews_by_id[str(row["review_id"])]["shot_sequence"] == "shot"
                ),
            }
            for row in planned
            if reviews_by_id[str(row["review_id"])]["shot_sequence"] != "uncertain"
        ]
        labels = {
            "schema_version": SHOT_VALIDITY_LABEL_SCHEMA,
            "purpose": EXPORT_PURPOSE,
            "runtime_consumable": False,
            "formal_evaluation_eligible": False,
            "source_video_sha256": source.sha256,
            "candidate_bundle_sha256": candidate["bundle_sha256"],
            "source_training_export_provenance": {
                key: value["artifact_sha256"]
                for key, value in source_refs.items()
                if "artifact_sha256" in value
            },
            "mapping_policy": dict(_MAPPING_POLICY),
            "examples": label_examples,
        }
        labels_filename = f"{source_id}_shot_validity_labels_v1.json"
        labels_bytes = encode_json_artifact(labels)
        label_files[labels_filename] = labels
        positive_count = sum(
            bool(row["event_present"]) for row in label_examples
        )
        negative_count = len(label_examples) - positive_count
        source_rows.append(
            {
                "source_id": source_id,
                "source_video_filename": source.filename,
                "source_video_sha256": source.sha256,
                "source_video_size_bytes": source.size_bytes,
                "candidate_bundle": {
                    "filename": candidate_filename,
                    "sha256": _sha256_bytes(candidate_bytes),
                    "size_bytes": len(candidate_bytes),
                    "bundle_sha256": candidate["bundle_sha256"],
                },
                "labels": {
                    "filename": labels_filename,
                    "sha256": _sha256_bytes(labels_bytes),
                    "size_bytes": len(labels_bytes),
                },
                "example_count": len(label_examples),
                "positive_count": positive_count,
                "negative_count": negative_count,
            }
        )
        index_examples.extend(
            {
                "source_video_sha256": source.sha256,
                "candidate_bundle_sha256": candidate["bundle_sha256"],
                "event_id": str(row["event_id"]),
                "event_present": bool(row["event_present"]),
            }
            for row in label_examples
        )

    index_examples.sort(
        key=lambda row: (
            row["source_video_sha256"],
            row["candidate_bundle_sha256"],
            row["event_id"],
        )
    )
    index: dict[str, Any] = {
        "schema_version": TRAINING_EXPORT_SCHEMA,
        "purpose": EXPORT_PURPOSE,
        "runtime_consumable": False,
        "training_consumable": True,
        "formal_evaluation_eligible": False,
        "codex_runtime_answer_used": False,
        **source_refs,
        "mapping_policy": dict(_MAPPING_POLICY),
        "training_routes": dict(_TRAINING_ROUTES),
        "counts": {
            "selected": 24,
            "exported": 22,
            "positive": 14,
            "negative": 8,
            "excluded_uncertain": 2,
        },
        "excluded_review_ids": excluded,
        "sources": source_rows,
        "examples": index_examples,
    }
    index["artifact_sha256"] = canonical_sha256(index)
    return VruCausalShotValidityTrainingArtifacts(
        index=verify_vru_causal_shot_validity_training_export(index),
        candidate_bundles=candidate_bundles,
        label_files=label_files,
        validated_input_paths=tuple(
            sorted(
                {
                    selection_path.resolve(),
                    review_plan_path.resolve(),
                    sealed_review_path.resolve(),
                    raw_frame_manifest_path.resolve(),
                    source_manifest_path.resolve(),
                    *(source.path for source in sources.values()),
                    *review_jpeg_paths,
                },
                key=lambda path: path.as_posix(),
            )
        ),
    )


def verify_vru_causal_shot_validity_training_export(
    payload: Mapping[str, Any],
    *,
    expected_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify the sealed export index and its label/route contract."""

    if set(payload) != _EXPORT_FIELDS:
        raise ValueError("VRU causal training export fields are not canonical")
    artifact = dict(payload)
    claimed = _require_sha(
        artifact.pop("artifact_sha256", None), "training export artifact"
    )
    if claimed != canonical_sha256(artifact):
        raise ValueError("VRU causal training export hash mismatch")
    if expected_artifact_sha256 is not None:
        expected = _require_sha(expected_artifact_sha256, "expected export artifact")
        if claimed != expected:
            raise ValueError("expected export artifact SHA-256 does not match")
    if artifact.get("schema_version") != TRAINING_EXPORT_SCHEMA:
        raise ValueError("unsupported VRU causal training export schema")
    if artifact.get("purpose") != EXPORT_PURPOSE:
        raise ValueError("VRU causal training export purpose is invalid")
    expected_flags = (
        artifact.get("runtime_consumable") is False
        and artifact.get("training_consumable") is True
        and artifact.get("formal_evaluation_eligible") is False
        and artifact.get("codex_runtime_answer_used") is False
    )
    if not expected_flags:
        raise ValueError("VRU causal training export consumption flags are invalid")
    if artifact.get("mapping_policy") != _MAPPING_POLICY:
        raise ValueError("VRU causal training export mapping policy is invalid")
    if artifact.get("training_routes") != _TRAINING_ROUTES:
        raise ValueError("VRU causal training export routes are invalid")
    _verify_export_references(artifact)
    _verify_export_rows(artifact)
    artifact["artifact_sha256"] = claimed
    return artifact


def encode_json_artifact(payload: Mapping[str, Any]) -> bytes:
    """Encode one artifact exactly as the CLI writes it."""

    return (
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
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


def _build_candidate_bundle(
    *,
    source: _Source,
    plan_rows: Sequence[Mapping[str, Any]],
    selection_sha256: str,
) -> dict[str, Any]:
    events = [
        GameEventResponse(
            event_id=str(row["review_id"]),
            revision=1,
            event_type="field_goal_attempt",
            source_video_id="video_001",
            start_frame=int(row["frame_indexes"][0]),
            end_frame=int(row["frame_indexes"][-1]),
            status="candidate",
            confidence=0.5,
            reason="label-hidden full-window geometry; offline training only",
        ).model_dump(mode="json")
        for row in plan_rows
    ]
    config = {
        "source_selection_artifact_sha256": selection_sha256,
        "geometry": "full_half_open_eight_second_window",
        "label_selection_not_used": True,
        "target_conditioned_anchor_used": False,
    }
    bundle: dict[str, Any] = {
        "schema_version": "agu.raw-only.v1",
        "game_id": f"vru-causal-closure-{source.source_id}",
        "raw_videos": [
            {
                "video_id": "video_001",
                "filename": source.filename,
                "sha256": source.sha256,
                "size_bytes": source.size_bytes,
            }
        ],
        "config_sha256": _json_sha256(config),
        "model_provenance": {
            "producer": "agu",
            "purpose": "development_diagnostic_geometry_only",
            "training_route": "scene_video_representation_only",
            "traditional_feature_training_eligible": "false",
            "training_geometry": "label_hidden_window_only",
        },
        "events": events,
        "inference_completed_at": _FIXED_COMPLETED_AT,
        "events_sha256": _json_sha256(events),
    }
    bundle["bundle_sha256"] = _json_sha256(bundle)
    return verify_raw_only_bundle(bundle).model_dump(mode="json")


def _verify_source_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    selection: Mapping[str, Any],
) -> dict[str, _Source]:
    if manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA:
        raise ValueError("unsupported VRU source manifest schema")
    if manifest.get("runtime_consumable") is not False:
        raise ValueError("VRU source manifest must remain offline-only")
    if manifest.get("codex_runtime_answer_used") is not False:
        raise ValueError("VRU source manifest cannot be a runtime answer channel")
    videos = manifest.get("videos")
    if not isinstance(videos, list) or len(videos) != len(SOURCE_IDS):
        raise ValueError("VRU source manifest must contain the three frozen videos")
    selection_by_source = {
        source_id: [
            row for row in selection["windows"] if row["source_id"] == source_id
        ]
        for source_id in SOURCE_IDS
    }
    result: dict[str, _Source] = {}
    seen_paths: set[Path] = set()
    seen_hashes: set[str] = set()
    seen_clip_ids: set[str] = set()
    for value in videos:
        if not isinstance(value, Mapping):
            raise ValueError("VRU source manifest rows must be objects")
        source_id = _safe_id(value.get("location"), "source ID")
        if source_id not in SOURCE_IDS or source_id in result:
            raise ValueError("VRU source manifest source IDs are invalid")
        relative = _manifest_relative_path(value.get("path"), "source video path")
        resolved = (manifest_path.parent.resolve() / relative).resolve()
        allowed_roots = (manifest_path.parent.resolve(), (Path(__file__).parents[2] / "dataset").resolve())
        if not any(resolved.is_relative_to(root) for root in allowed_roots):
            raise ValueError("source video path is outside its allowed roots")
        if resolved in seen_paths or not resolved.is_file():
            raise ValueError("source video path is missing or duplicated")
        filename = resolved.name
        sha256 = _require_sha(value.get("sha256"), "source video")
        size_bytes = _positive_int(value.get("size_bytes"), "source video size")
        if resolved.stat().st_size != size_bytes:
            raise ValueError("source video size mismatch")
        if _file_sha256(resolved) != sha256:
            raise ValueError("source video SHA-256 mismatch")
        if sha256 in seen_hashes:
            raise ValueError("source video hashes must be unique")
        fps = _positive_float(value.get("fps"), "source FPS")
        frame_count = _positive_int(value.get("frame_count"), "source frame count")
        clip_id = _safe_id(value.get("clip_id"), "source clip ID")
        if clip_id in seen_clip_ids:
            raise ValueError("source clip IDs must be unique")
        windows = selection_by_source[source_id]
        if len(windows) != 8 or any(
            row["clip_id"] != clip_id
            or row["source_video_filename"] != filename
            or row["source_video_sha256"] != sha256
            or row["source_fps"] != fps
            or row["frame_count"] != frame_count
            for row in windows
        ):
            raise ValueError("selection/source video binding mismatch")
        result[source_id] = _Source(
            source_id=source_id,
            clip_id=clip_id,
            filename=filename,
            path=resolved,
            sha256=sha256,
            size_bytes=size_bytes,
            fps=fps,
            frame_count=frame_count,
        )
        seen_paths.add(resolved)
        seen_hashes.add(sha256)
        seen_clip_ids.add(clip_id)
    if set(result) != set(SOURCE_IDS):
        raise ValueError("VRU source manifest must cover every frozen source")
    return result


def _verify_selection_plan(
    *,
    selection: Mapping[str, Any],
    plan: Mapping[str, Any],
    source_manifest_sha256: str,
) -> None:
    if set(plan) != _PLAN_FIELDS:
        raise ValueError("review plan fields are not canonical")
    if plan.get("source_manifest_sha256") != source_manifest_sha256:
        raise ValueError("review plan/source manifest SHA-256 mismatch")
    windows = selection["windows"]
    examples = plan["examples"]
    if [row["review_id"] for row in examples] != [row["review_id"] for row in windows]:
        raise ValueError("selection/review plan coverage mismatch")
    expected_source_hashes = sorted(
        {str(row["source_video_sha256"]) for row in windows}
    )
    if plan.get("source_video_sha256s") != expected_source_hashes:
        raise ValueError("review plan source-video set mismatch")
    policy = selection["selection"]
    sample_count = int(policy["sample_count"])
    sample_rate = float(policy["sample_rate_hz"])
    if sample_count != SAMPLE_COUNT or policy["interval_semantics"] != "half_open":
        raise ValueError("review plan must retain the frozen half-open sample policy")
    for window, example in zip(windows, examples, strict=True):
        if set(example) != _PLAN_EXAMPLE_FIELDS:
            raise ValueError("review plan example fields are not canonical")
        if any(
            example[field] != window[field]
            for field in (
                "review_id",
                "source_video_sha256",
                "source_video_filename",
                "source_fps",
                "frame_count",
            )
        ):
            raise ValueError("selection/review plan source binding mismatch")
        expected_indexes = [
            round(
                (float(window["start_seconds"]) + index / sample_rate)
                * float(window["source_fps"])
            )
            for index in range(sample_count)
        ]
        if example["frame_indexes"] != expected_indexes:
            raise ValueError("selection/review plan frame mapping mismatch")
        frame_hashes = example["frame_sha256s"]
        if set(frame_hashes) != {str(index) for index in expected_indexes}:
            raise ValueError("review plan raw-frame hash coverage mismatch")
        for digest in frame_hashes.values():
            _require_sha(digest, "review plan raw frame")


def _verify_raw_frame_manifest(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
    plan: Mapping[str, Any],
    source_manifest_sha256: str,
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    if set(payload) != _RAW_MANIFEST_FIELDS:
        raise ValueError("VRU raw frame manifest fields are not canonical")
    artifact = dict(payload)
    claimed = _require_sha(
        artifact.pop("artifact_sha256", None), "raw frame manifest artifact"
    )
    if claimed != canonical_sha256(artifact):
        raise ValueError("VRU raw frame manifest hash mismatch")
    if artifact.get("schema_version") != FRAME_MANIFEST_SCHEMA:
        raise ValueError("unsupported VRU raw frame manifest schema")
    if artifact.get("purpose") != "offline_codex_visual_review_of_original_vru_frames":
        raise ValueError("VRU raw frame manifest purpose is invalid")
    if (
        artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("labels_hidden_from_reviewer") is not True
    ):
        raise ValueError("VRU raw frame manifest provenance is invalid")
    if artifact.get("source_manifest_sha256") != source_manifest_sha256:
        raise ValueError("raw frame/source manifest SHA-256 mismatch")
    if artifact.get("review_plan_sha256") != plan["artifact_sha256"]:
        raise ValueError("raw frame/review plan SHA-256 mismatch")
    quality = _positive_int(artifact.get("jpeg_quality"), "JPEG quality")
    if quality > 100:
        raise ValueError("JPEG quality must not exceed 100")
    frame_root_value = _safe_relative_path(artifact.get("frame_root"), "frame root")
    if len(frame_root_value.parts) != 1:
        raise ValueError("frame root must be one path-safe directory")
    manifest_parent = manifest_path.parent.resolve()
    frame_root = (manifest_parent / frame_root_value).resolve()
    if not frame_root.is_relative_to(manifest_parent) or not frame_root.is_dir():
        raise ValueError("frame root is outside the raw-frame manifest directory")
    expected = [
        (
            str(example["review_id"]),
            position,
            int(frame_index),
            str(example["frame_sha256s"][str(frame_index)]),
        )
        for example in plan["examples"]
        for position, frame_index in enumerate(example["frame_indexes"])
    ]
    rows = artifact.get("frames")
    if (
        not isinstance(rows, list)
        or artifact.get("frame_count") != len(rows)
        or len(rows) != len(expected)
    ):
        raise ValueError("raw frame manifest must exactly cover the review plan")
    resolved_paths: set[Path] = set()
    for row, expected_row in zip(rows, expected, strict=True):
        if not isinstance(row, Mapping) or set(row) != _RAW_FRAME_FIELDS:
            raise ValueError("raw frame manifest rows are not canonical")
        actual = (
            str(row.get("review_id") or ""),
            _non_negative_int(row.get("position"), "raw frame position"),
            _non_negative_int(row.get("frame_index"), "raw frame index"),
            _require_sha(row.get("raw_frame_sha256"), "raw frame"),
        )
        if actual != expected_row:
            raise ValueError("raw frame manifest coverage does not match the plan")
        relative = _safe_relative_path(row.get("relative_path"), "review JPEG path")
        jpeg_path = (frame_root / relative).resolve()
        if not jpeg_path.is_relative_to(frame_root):
            raise ValueError("review JPEG path is outside the frame root")
        if jpeg_path in resolved_paths or not jpeg_path.is_file():
            raise ValueError("review JPEG path is missing or aliased")
        expected_bytes = _positive_int(row.get("jpeg_bytes"), "review JPEG size")
        if jpeg_path.stat().st_size != expected_bytes:
            raise ValueError("review JPEG size mismatch")
        expected_sha = _require_sha(row.get("jpeg_sha256"), "review JPEG")
        if _file_sha256(jpeg_path) != expected_sha:
            raise ValueError("review JPEG SHA-256 mismatch")
        resolved_paths.add(jpeg_path)
    artifact["artifact_sha256"] = claimed
    return artifact, tuple(sorted(resolved_paths, key=lambda path: path.as_posix()))


def _verify_export_references(artifact: Mapping[str, Any]) -> None:
    selection = artifact.get("source_selection")
    if not isinstance(selection, Mapping) or set(selection) != {
        "artifact_sha256",
        "file_sha256",
        "prior_labels_used_as_final_8s_targets",
        "prior_labels_used_for_stratified_selection",
        "schema_version",
        "training_consumable",
    }:
        raise ValueError("training export source selection reference is invalid")
    if (
        selection.get("schema_version") != "agu.vru-causal-closure-selection.v1"
        or selection.get("training_consumable") is not False
        or selection.get("prior_labels_used_for_stratified_selection") is not True
        or selection.get("prior_labels_used_as_final_8s_targets") is not False
    ):
        raise ValueError("training export source selection boundary is invalid")
    _require_sha(selection.get("artifact_sha256"), "source selection artifact")
    _require_sha(selection.get("file_sha256"), "source selection file")
    expected_schemas = {
        "source_review_plan": "agu.vru-causal-review-plan.v1",
        "source_review": "agu.vru-causal-offline-review.v1",
        "source_frame_manifest": FRAME_MANIFEST_SCHEMA,
    }
    for field, schema in expected_schemas.items():
        reference = artifact.get(field)
        if (
            not isinstance(reference, Mapping)
            or set(reference) != {"artifact_sha256", "file_sha256", "schema_version"}
            or reference.get("schema_version") != schema
        ):
            raise ValueError(f"training export {field} reference is invalid")
        _require_sha(reference.get("artifact_sha256"), f"{field} artifact")
        _require_sha(reference.get("file_sha256"), f"{field} file")
    source_manifest = artifact.get("source_manifest")
    if (
        not isinstance(source_manifest, Mapping)
        or set(source_manifest) != {"file_sha256", "schema_version"}
        or source_manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA
    ):
        raise ValueError("training export source manifest reference is invalid")
    _require_sha(source_manifest.get("file_sha256"), "source manifest file")


def _verify_export_rows(artifact: Mapping[str, Any]) -> None:
    excluded = artifact.get("excluded_review_ids")
    if excluded != list(EXPECTED_EXCLUDED_REVIEW_IDS):
        raise ValueError("training export uncertain exclusions are invalid")
    sources = artifact.get("sources")
    if not isinstance(sources, list) or len(sources) != len(SOURCE_IDS):
        raise ValueError("training export requires three source rows")
    if [row.get("source_id") for row in sources if isinstance(row, Mapping)] != list(SOURCE_IDS):
        raise ValueError("training export source rows are not canonically ordered")
    source_keys: dict[tuple[str, str], Mapping[str, Any]] = {}
    asset_filenames: set[str] = set()
    for row in sources:
        if not isinstance(row, Mapping) or set(row) != _SOURCE_FIELDS:
            raise ValueError("training export source row fields are invalid")
        _safe_id(row.get("source_id"), "training source ID")
        filename = _safe_relative_path(
            row.get("source_video_filename"), "training source filename"
        )
        if len(filename.parts) != 1:
            raise ValueError("training source filename must be path-safe")
        source_sha = _require_sha(row.get("source_video_sha256"), "training source video")
        _positive_int(row.get("source_video_size_bytes"), "training source size")
        candidate = row.get("candidate_bundle")
        labels = row.get("labels")
        if not isinstance(candidate, Mapping) or set(candidate) != _CANDIDATE_ASSET_FIELDS:
            raise ValueError("training export candidate asset is invalid")
        if not isinstance(labels, Mapping) or set(labels) != _LABEL_ASSET_FIELDS:
            raise ValueError("training export label asset is invalid")
        for filename in (
            _verify_file_asset(candidate, "candidate bundle"),
            _verify_file_asset(labels, "shot-validity labels"),
        ):
            normalized_filename = filename.casefold()
            if normalized_filename in asset_filenames:
                raise ValueError(
                    "training export child asset filenames must be globally unique"
                )
            asset_filenames.add(normalized_filename)
        bundle_sha = _require_sha(candidate.get("bundle_sha256"), "candidate bundle")
        key = (source_sha, bundle_sha)
        if key in source_keys:
            raise ValueError("training export source/bundle keys must be unique")
        source_keys[key] = row
    examples = artifact.get("examples")
    if not isinstance(examples, list) or len(examples) != 22:
        raise ValueError("training export must contain exactly 22 examples")
    expected_order = sorted(
        examples,
        key=lambda row: (
            str(row.get("source_video_sha256")),
            str(row.get("candidate_bundle_sha256")),
            str(row.get("event_id")),
        ),
    )
    if examples != expected_order:
        raise ValueError("training export examples must use ordered triple keys")
    counts_by_source: Counter[tuple[str, str]] = Counter()
    positives_by_source: Counter[tuple[str, str]] = Counter()
    keys: set[tuple[str, str, str]] = set()
    for row in examples:
        if not isinstance(row, Mapping) or set(row) != _EXAMPLE_FIELDS:
            raise ValueError("training export example fields are invalid")
        source_sha = _require_sha(row.get("source_video_sha256"), "example source")
        bundle_sha = _require_sha(row.get("candidate_bundle_sha256"), "example bundle")
        event_id = _safe_id(row.get("event_id"), "example event ID")
        event_present = row.get("event_present")
        if not isinstance(event_present, bool):
            raise ValueError("training export event_present must be boolean")
        source_key = (source_sha, bundle_sha)
        key = (*source_key, event_id)
        if source_key not in source_keys or key in keys or event_id in excluded:
            raise ValueError("training export example key is invalid or duplicated")
        keys.add(key)
        counts_by_source[source_key] += 1
        positives_by_source[source_key] += int(event_present)
    positive = sum(bool(row["event_present"]) for row in examples)
    expected_counts = {
        "selected": 24,
        "exported": 22,
        "positive": 14,
        "negative": 8,
        "excluded_uncertain": 2,
    }
    if artifact.get("counts") != expected_counts or positive != 14:
        raise ValueError("training export 24-to-22 class counts are invalid")
    for key, source in source_keys.items():
        count = counts_by_source[key]
        source_positive = positives_by_source[key]
        expected = _EXPECTED_SOURCE_COUNTS[str(source["source_id"])]
        if (
            source.get("example_count") != count
            or source.get("positive_count") != source_positive
            or source.get("negative_count") != count - source_positive
            or (count, source_positive, count - source_positive) != expected
        ):
            raise ValueError("training export per-source counts are invalid")


def _verify_file_asset(value: Mapping[str, Any], field: str) -> str:
    filename = _safe_relative_path(value.get("filename"), f"{field} filename")
    if len(filename.parts) != 1:
        raise ValueError(f"{field} filename must be path-safe")
    _require_sha(value.get("sha256"), f"{field} file")
    _positive_int(value.get("size_bytes"), f"{field} size")
    return filename.as_posix()


def _artifact_ref(payload: Mapping[str, Any], file_sha256: str) -> dict[str, str]:
    return {
        "schema_version": str(payload["schema_version"]),
        "artifact_sha256": _require_sha(payload["artifact_sha256"], "artifact"),
        "file_sha256": _require_sha(file_sha256, "artifact file"),
    }


def _read_mapping_with_sha(
    path: Path, description: str
) -> tuple[dict[str, Any], str]:
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object")
    return payload, _sha256_bytes(encoded)


def _safe_relative_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{field} must be a path-safe relative path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a path-safe relative path")
    return path


def _manifest_relative_path(value: Any, field: str) -> Path:
    """Accept canonical ``..`` components only when root resolution stays safe."""

    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{field} must be a relative path")
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{field} must be a relative path")
    return path


def _safe_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{field} must be a path-safe scalar")
    return value


def _require_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} SHA-256 must be a lowercase hex digest")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _positive_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite and positive")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return number


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_sha256(payload: object) -> str:
    return _sha256_bytes(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
