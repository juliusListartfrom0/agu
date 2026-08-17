"""Additive Harwood extension for the frozen VRU causal training export."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from app.analysis.continuous_causal_selection import (
    neutral_continuous_review_id,
    resolve_continuous_source_video_path,
    verify_continuous_causal_selection_receipt,
)
from app.analysis.official_evaluation import verify_raw_only_bundle
from app.analysis.schemas import GameEventResponse
from app.analysis.shot_validity import SHOT_VALIDITY_LABEL_SCHEMA
from app.analysis.vru_causal_review import (
    neutral_vru_source_filename,
    verify_vru_causal_review,
    verify_vru_causal_review_plan,
)
from app.analysis.vru_causal_source_groups import (
    SOURCE_GROUP_SCHEMA,
    verify_vru_causal_source_groups,
)
from app.analysis.vru_causal_training import (
    TRAINING_EXPORT_SCHEMA,
    encode_json_artifact,
    verify_vru_causal_shot_validity_training_export,
)

TRAINING_EXPORT_V2_SCHEMA = "agu.vru-causal-shot-validity-training-export.v2"
EXPORT_PURPOSE = "development_diagnostic_only"
HARWOOD_SOURCE_ID = "harwood"
HARWOOD_GAME_ID = "harwood"
PARENT_SOURCE_IDS = ("hazen", "randolph", "vtv")
GAME_IDS = (*PARENT_SOURCE_IDS, HARWOOD_GAME_ID)
HARWOOD_CANDIDATE_FILENAME = "harwood_candidate_bundle_v2.json"
HARWOOD_LABEL_FILENAME = "harwood_shot_validity_labels_v2.json"
EXPECTED_EXCLUDED_REVIEW_IDS = (
    "closure-randolph-0002",
    "closure-vtv-0003",
    "closure-84dc8a546ac128ce09978c0f-0023",
)
FROZEN_PARENT_ARTIFACT_SHA256 = "1b0bcef2c67f0d0e1fa35b79042584c43d938f05e893e89a4a64266ab9d0e464"
FROZEN_PARENT_FILE_SHA256 = "afb406ef84ae57b8d6cbe565a848b03bcefc2fdd449d8c223374e04d7f7dc3e0"
FROZEN_PARENT_CHILD_SHA256 = MappingProxyType(
    {
        "hazen_candidate_bundle_v1.json": "378a08604962a8733898312478d9e3ad762791563d00f7608cd4353cf2dbaaa9",
        "hazen_shot_validity_labels_v1.json": "acabe7e07ad8f98350ef492622bea8d7230ce9dfe1b8d278a4919248faad8dda",
        "randolph_candidate_bundle_v1.json": "2d7d0d3b6d872cb1b94c43c8c595df1d98c1485b82faa72ba43f3157dafc8cd8",
        "randolph_shot_validity_labels_v1.json": "44b1588e583edb1e4e66ab270511245938c19e8197cd96df0a09aec3da8dda09",
        "vtv_candidate_bundle_v1.json": "65c0fb60b8338512d50ee3c33ca01b93c3c9d6f4c126a27ca311f8e8709f22f2",
        "vtv_shot_validity_labels_v1.json": "5274a0ce6b2aae635b763b138a88487d28da61f9945f5e779526efbb52ba1b98",
    }
)
FROZEN_HARWOOD_FILE_SHA256 = MappingProxyType(
    {
        "source_manifest": "8786b64fc02f4d6105acbe097617a733f7276469a60a721b883ac64062d51ded",
        "selection": "35126baa4ae15a888391ff69840c5410c4195f48a0608b25f7ec1a680a576138",
        "review_plan": "bbf12a7f7de808d467c6057b2e93170765e0098b47fd4465180ac8c9b9575c24",
        "raw_frame_manifest": "984365ba89c71d781a1e36566a5c5327006349c9c0c6744d0246449933a5bcca",
        "sealed_review": "ce9ad5d00db77baed53d874cc11ee7deb1ba960e75592ddf2c591a2490319fff",
    }
)
FROZEN_HARWOOD_ARTIFACT_SHA256 = MappingProxyType(
    {
        "selection": "5a7eaa1ecefa295c5d97632e05e7aea2187eafac0082ba6ea6341fc6b372a80f",
        "review_plan": "87c9b92c548000527f253082c6190ec82022934d9c3db244be32e50505244911",
        "raw_frame_manifest": "0757e339ffd5acb97d5b8f42790cfced2de825e44deea20095abd7bedfeae776",
        "sealed_review": "5f2219a4cdcc7e8d4e40dde060c8ba82b5070eb6db471c1ce2855cea9d3e5cb6",
    }
)
FROZEN_HARWOOD_VIDEO_SHA256 = "bf7f3135774027aec3c2827cfa282c95ac4f958dd2bccd93e42f370d4b1f8b81"
FROZEN_HARWOOD_VIDEO_SIZE_BYTES = 1_376_342_882
FROZEN_HARWOOD_UNCERTAIN_ID = "closure-84dc8a546ac128ce09978c0f-0023"
FROZEN_HARWOOD_COMPLETED_AT = "2026-08-16T00:00:00+00:00"
_FROZEN_HARWOOD_SHOT_SEQUENCE_SHA256 = "0a164aa9c135aca2abc54186882c28000f3f62f843dd32cf7ef3fa69f14a6506"
_FROZEN_SOURCE_GROUP_ARTIFACT_SHA256 = "029cb1b4c56eebe2cf36495618e12966944883e52a706a5bb05feb0e81e674b0"
_FROZEN_SOURCE_GROUP_FILE_SHA256 = "d48f990790b57b11389186b407078f52fce3d34a6410a9101001cd5d3221c05e"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,127}")
_MAX_JSON_BYTES = 2 * 1024 * 1024
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
_INDEX_FIELDS = {
    "artifact_sha256",
    "codex_runtime_answer_used",
    "counts",
    "excluded_review_ids",
    "extension",
    "formal_evaluation_eligible",
    "harwood_evidence",
    "mapping_policy",
    "parent",
    "promoted",
    "promotion_eligible",
    "purpose",
    "runtime_consumable",
    "schema_version",
    "source_groups",
    "training_consumable",
    "training_routes",
}
_ASSET_FIELDS = {"filename", "sha256", "size_bytes"}
_CANDIDATE_ASSET_FIELDS = _ASSET_FIELDS | {"bundle_sha256"}
_PARENT_FIELDS = {
    "artifact_sha256",
    "file_sha256",
    "schema_version",
    "sources",
}
_PARENT_SOURCE_FIELDS = {
    "candidate_bundle",
    "labels",
    "source_id",
    "source_video_sha256",
}
_EVIDENCE_FIELDS = {
    "source_frame_manifest",
    "source_manifest",
    "source_review",
    "source_review_plan",
    "source_selection",
    "source_video",
}
_ARTIFACT_REF_FIELDS = {"artifact_sha256", "file_sha256", "schema_version"}
_SOURCE_MANIFEST_REF_FIELDS = {"file_sha256", "schema_version"}
_SOURCE_VIDEO_REF_FIELDS = {"filename", "sha256", "size_bytes"}
_RAW_REF_FIELDS = _ARTIFACT_REF_FIELDS | {"jpeg_count", "jpeg_set_sha256"}
_EXTENSION_FIELDS = {
    "candidate_bundle",
    "example_count",
    "excluded_uncertain",
    "game_id",
    "labels",
    "negative_count",
    "positive_count",
    "selected_count",
    "source_id",
    "source_video_sha256",
}
_SOURCE_GROUP_REF_FIELDS = {"artifact_sha256", "file_sha256", "schema_version"}
_LABEL_FIELDS = {
    "candidate_bundle_sha256",
    "examples",
    "formal_evaluation_eligible",
    "mapping_policy",
    "purpose",
    "runtime_consumable",
    "schema_version",
    "source_training_export_provenance",
    "source_video_sha256",
}
_LABEL_EXAMPLE_FIELDS = {"event_id", "event_present"}
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
_RAW_FRAME_FIELDS = {
    "frame_index",
    "jpeg_bytes",
    "jpeg_sha256",
    "position",
    "raw_frame_sha256",
    "relative_path",
    "review_id",
}


@dataclass(frozen=True)
class ContinuousCausalTrainingChain:
    """Verified stored edges plus the resolved four-game training view."""

    index: dict[str, Any]
    extension_candidate_bundles: dict[str, dict[str, Any]]
    extension_label_files: dict[str, dict[str, Any]]
    source_groups: dict[str, Any]
    resolved_examples: tuple[dict[str, Any], ...]
    validated_input_paths: tuple[Path, ...]


@dataclass(frozen=True)
class _VerifiedParent:
    index: dict[str, Any]
    candidate_bundles: dict[str, dict[str, Any]]
    label_files: dict[str, dict[str, Any]]
    resolved_examples: tuple[dict[str, Any], ...]
    validated_paths: tuple[Path, ...]
    reference: dict[str, Any]


@dataclass(frozen=True)
class _VerifiedHarwood:
    selection: dict[str, Any]
    plan: dict[str, Any]
    review: dict[str, Any]
    raw_manifest: dict[str, Any]
    source_video_path: Path
    jpeg_paths: tuple[Path, ...]
    reference: dict[str, Any]
    validated_paths: tuple[Path, ...]


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash canonical JSON while excluding the artifact's own receipt."""

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


def _verify_v2_index(
    payload: Mapping[str, Any],
    *,
    expected_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != _INDEX_FIELDS:
        raise ValueError("VRU causal training v2 index fields are not canonical")
    artifact = dict(payload)
    claimed = _require_sha(artifact.pop("artifact_sha256", None), "training v2 artifact")
    if claimed != canonical_sha256(artifact):
        raise ValueError("VRU causal training v2 artifact SHA-256 mismatch")
    if expected_artifact_sha256 is not None and claimed != _require_sha(
        expected_artifact_sha256, "expected training v2 artifact"
    ):
        raise ValueError("expected training v2 artifact SHA-256 does not match")
    if artifact.get("schema_version") != TRAINING_EXPORT_V2_SCHEMA or artifact.get("purpose") != EXPORT_PURPOSE:
        raise ValueError("unsupported VRU causal training v2 contract")
    if (
        artifact.get("runtime_consumable") is not False
        or artifact.get("training_consumable") is not True
        or artifact.get("formal_evaluation_eligible") is not False
        or artifact.get("promotion_eligible") is not False
        or artifact.get("promoted") is not False
        or artifact.get("codex_runtime_answer_used") is not False
    ):
        raise ValueError("VRU causal training v2 consumption flags are invalid")
    if artifact.get("mapping_policy") != _MAPPING_POLICY:
        raise ValueError("VRU causal training v2 mapping policy is invalid")
    if artifact.get("training_routes") != _TRAINING_ROUTES:
        raise ValueError("VRU causal training v2 routes are invalid")
    if artifact.get("counts") != {
        "selected": 48,
        "exported": 45,
        "positive": 17,
        "negative": 28,
        "excluded_uncertain": 3,
        "game_groups": 4,
        "production_families": 2,
    }:
        raise ValueError("VRU causal training v2 counts are invalid")
    if artifact.get("excluded_review_ids") != list(EXPECTED_EXCLUDED_REVIEW_IDS):
        raise ValueError("VRU causal training v2 uncertain exclusions are invalid")
    _verify_parent_reference(artifact.get("parent"))
    _verify_harwood_reference(artifact.get("harwood_evidence"))
    _verify_extension_reference(artifact.get("extension"))
    _verify_source_group_reference(artifact.get("source_groups"))
    artifact["artifact_sha256"] = claimed
    return artifact


def verify_vru_causal_shot_validity_training_extension_index(
    payload: Mapping[str, Any],
    *,
    expected_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify the canonical additive-v2 index without replaying child files."""

    return _verify_v2_index(
        payload,
        expected_artifact_sha256=expected_artifact_sha256,
    )


def _verify_parent_reference(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != _PARENT_FIELDS:
        raise ValueError("training v2 parent reference is invalid")
    if value.get("schema_version") != TRAINING_EXPORT_SCHEMA:
        raise ValueError("training v2 parent schema is invalid")
    if value.get("artifact_sha256") != FROZEN_PARENT_ARTIFACT_SHA256:
        raise ValueError("training v2 parent artifact receipt is not frozen")
    if value.get("file_sha256") != FROZEN_PARENT_FILE_SHA256:
        raise ValueError("training v2 parent file receipt is not frozen")
    sources = value.get("sources")
    if not isinstance(sources, list) or len(sources) != 3:
        raise ValueError("training v2 parent must bind three sources")
    if [row.get("source_id") for row in sources if isinstance(row, Mapping)] != list(PARENT_SOURCE_IDS):
        raise ValueError("training v2 parent sources are not canonically ordered")
    seen_names: set[str] = set()
    for row in sources:
        if not isinstance(row, Mapping) or set(row) != _PARENT_SOURCE_FIELDS:
            raise ValueError("training v2 parent source reference is invalid")
        _require_sha(row.get("source_video_sha256"), "parent source video")
        candidate = row.get("candidate_bundle")
        labels = row.get("labels")
        if not isinstance(candidate, Mapping) or set(candidate) != _CANDIDATE_ASSET_FIELDS:
            raise ValueError("training v2 parent candidate reference is invalid")
        if not isinstance(labels, Mapping) or set(labels) != _ASSET_FIELDS:
            raise ValueError("training v2 parent label reference is invalid")
        candidate_name = _verify_asset(candidate, "parent candidate")
        label_name = _verify_asset(labels, "parent labels")
        _require_sha(candidate.get("bundle_sha256"), "parent candidate bundle")
        for filename, ref in ((candidate_name, candidate), (label_name, labels)):
            folded = filename.casefold()
            if folded in seen_names:
                raise ValueError("training v2 parent child filenames are duplicated")
            seen_names.add(folded)
            if FROZEN_PARENT_CHILD_SHA256.get(filename) != ref.get("sha256"):
                raise ValueError("training v2 parent child receipt is not frozen")
    if set(FROZEN_PARENT_CHILD_SHA256) != {
        name for name in FROZEN_PARENT_CHILD_SHA256 if name.casefold() in seen_names
    }:
        raise ValueError("training v2 parent child set is incomplete")


def _verify_harwood_reference(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != _EVIDENCE_FIELDS:
        raise ValueError("training v2 Harwood evidence reference is invalid")
    schemas = {
        "source_selection": "agu.continuous-causal-review-selection.v1",
        "source_review_plan": "agu.vru-causal-review-plan.v1",
        "source_review": "agu.vru-causal-offline-review.v1",
    }
    receipt_names = {
        "source_selection": "selection",
        "source_review_plan": "review_plan",
        "source_review": "sealed_review",
    }
    for field, schema in schemas.items():
        ref = value.get(field)
        if not isinstance(ref, Mapping) or set(ref) != _ARTIFACT_REF_FIELDS or ref.get("schema_version") != schema:
            raise ValueError(f"training v2 {field} reference is invalid")
        receipt_name = receipt_names[field]
        if ref.get("artifact_sha256") != FROZEN_HARWOOD_ARTIFACT_SHA256[receipt_name]:
            raise ValueError(f"training v2 {field} artifact receipt is not frozen")
        if ref.get("file_sha256") != FROZEN_HARWOOD_FILE_SHA256[receipt_name]:
            raise ValueError(f"training v2 {field} file receipt is not frozen")
    raw = value.get("source_frame_manifest")
    if (
        not isinstance(raw, Mapping)
        or set(raw) != _RAW_REF_FIELDS
        or raw.get("schema_version") != "agu.vru-causal-review-frame-manifest.v1"
        or raw.get("artifact_sha256") != FROZEN_HARWOOD_ARTIFACT_SHA256["raw_frame_manifest"]
        or raw.get("file_sha256") != FROZEN_HARWOOD_FILE_SHA256["raw_frame_manifest"]
        or raw.get("jpeg_count") != 1536
    ):
        raise ValueError("training v2 Harwood raw-frame reference is invalid")
    _require_sha(raw.get("jpeg_set_sha256"), "Harwood JPEG set")
    manifest = value.get("source_manifest")
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != _SOURCE_MANIFEST_REF_FIELDS
        or manifest.get("schema_version") != "agu.vru-basketball-source-manifest.v1"
        or manifest.get("file_sha256") != FROZEN_HARWOOD_FILE_SHA256["source_manifest"]
    ):
        raise ValueError("training v2 Harwood source-manifest reference is invalid")
    video = value.get("source_video")
    if (
        not isinstance(video, Mapping)
        or set(video) != _SOURCE_VIDEO_REF_FIELDS
        or video.get("filename") != "hctv_harwood_2026.webm"
        or video.get("sha256") != FROZEN_HARWOOD_VIDEO_SHA256
        or video.get("size_bytes") != FROZEN_HARWOOD_VIDEO_SIZE_BYTES
    ):
        raise ValueError("training v2 Harwood source-video reference is invalid")


def _verify_extension_reference(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != _EXTENSION_FIELDS:
        raise ValueError("training v2 extension reference is invalid")
    if (
        value.get("source_id") != HARWOOD_SOURCE_ID
        or value.get("game_id") != HARWOOD_GAME_ID
        or value.get("source_video_sha256") != FROZEN_HARWOOD_VIDEO_SHA256
        or value.get("selected_count") != 24
        or value.get("example_count") != 23
        or value.get("positive_count") != 3
        or value.get("negative_count") != 20
        or value.get("excluded_uncertain") != 1
    ):
        raise ValueError("training v2 extension counts or identity are invalid")
    candidate = value.get("candidate_bundle")
    labels = value.get("labels")
    if (
        not isinstance(candidate, Mapping)
        or set(candidate) != _CANDIDATE_ASSET_FIELDS
        or _verify_asset(candidate, "Harwood candidate") != HARWOOD_CANDIDATE_FILENAME
    ):
        raise ValueError("training v2 Harwood candidate reference is invalid")
    _require_sha(candidate.get("bundle_sha256"), "Harwood candidate bundle")
    if (
        not isinstance(labels, Mapping)
        or set(labels) != _ASSET_FIELDS
        or _verify_asset(labels, "Harwood labels") != HARWOOD_LABEL_FILENAME
    ):
        raise ValueError("training v2 Harwood label reference is invalid")


def _verify_source_group_reference(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != _SOURCE_GROUP_REF_FIELDS:
        raise ValueError("training v2 source-group reference is invalid")
    if value.get("schema_version") != "agu.vru-causal-source-groups.v1":
        raise ValueError("training v2 source-group schema is invalid")
    _require_sha(value.get("artifact_sha256"), "source-group artifact")
    _require_sha(value.get("file_sha256"), "source-group file")


def _verify_asset(value: Mapping[str, Any], field: str) -> str:
    filename = _safe_basename(value.get("filename"), f"{field} filename")
    _require_sha(value.get("sha256"), f"{field} file")
    _positive_int(value.get("size_bytes"), f"{field} size")
    return filename


def _safe_basename(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or Path(value).is_absolute()
        or len(Path(value).parts) != 1
        or value in {".", ".."}
    ):
        raise ValueError(f"{field} must be a path-safe basename")
    return value


def _require_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} SHA-256 must be a lowercase hex digest")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _verify_parent_chain(
    *,
    parent_export_path: Path,
    expected_parent_artifact_sha256: str,
    expected_parent_file_sha256: str,
    parent_asset_root: Path,
) -> _VerifiedParent:
    expected_artifact = _require_sha(expected_parent_artifact_sha256, "expected parent artifact")
    expected_file = _require_sha(expected_parent_file_sha256, "expected parent file")
    if expected_artifact != FROZEN_PARENT_ARTIFACT_SHA256:
        raise ValueError("expected parent artifact receipt is not TASK-0254")
    if expected_file != FROZEN_PARENT_FILE_SHA256:
        raise ValueError("expected parent file receipt is not TASK-0254")

    parent_payload, parent_bytes = _read_json_object(parent_export_path, "parent v1 training export")
    if _sha256_bytes(parent_bytes) != expected_file:
        raise ValueError("parent v1 training export file SHA-256 mismatch")
    parent = verify_vru_causal_shot_validity_training_export(
        parent_payload,
        expected_artifact_sha256=expected_artifact,
    )
    root = parent_asset_root.resolve()
    if not root.is_dir():
        raise ValueError("parent v1 asset root must be a directory")

    candidates: dict[str, dict[str, Any]] = {}
    labels_by_filename: dict[str, dict[str, Any]] = {}
    reconstructed: list[dict[str, Any]] = []
    validated_paths = [parent_export_path.resolve()]
    provenance = {
        field: parent[field]["artifact_sha256"]
        for field in (
            "source_selection",
            "source_review_plan",
            "source_review",
            "source_frame_manifest",
        )
    }
    reference_sources: list[dict[str, Any]] = []
    for source in parent["sources"]:
        source_id = str(source["source_id"])
        candidate_ref = dict(source["candidate_bundle"])
        label_ref = dict(source["labels"])
        candidate_path = _resolve_child_path(root, candidate_ref["filename"], "parent candidate")
        label_path = _resolve_child_path(root, label_ref["filename"], "parent labels")
        candidate_payload, candidate_bytes = _read_json_object(candidate_path, "parent candidate bundle")
        label_payload, label_bytes = _read_json_object(label_path, "parent shot-validity labels")
        _verify_file_bytes(candidate_bytes, candidate_ref, "parent candidate bundle")
        _verify_file_bytes(label_bytes, label_ref, "parent shot-validity labels")
        if FROZEN_PARENT_CHILD_SHA256.get(candidate_path.name) != _sha256_bytes(
            candidate_bytes
        ) or FROZEN_PARENT_CHILD_SHA256.get(label_path.name) != _sha256_bytes(label_bytes):
            raise ValueError("parent child bytes do not match frozen TASK-0254 receipts")

        candidate_model = verify_raw_only_bundle(candidate_payload)
        candidate = candidate_model.model_dump(mode="json")
        if candidate != candidate_payload:
            raise ValueError("parent candidate bundle is not canonical")
        if (
            candidate["bundle_sha256"] != candidate_ref["bundle_sha256"]
            or candidate["game_id"] != f"vru-causal-closure-{source_id}"
            or len(candidate["raw_videos"]) != 1
        ):
            raise ValueError("parent candidate bundle/index binding mismatch")
        raw_video = candidate["raw_videos"][0]
        if (
            raw_video["sha256"] != source["source_video_sha256"]
            or raw_video["filename"] != source["source_video_filename"]
            or raw_video["size_bytes"] != source["source_video_size_bytes"]
        ):
            raise ValueError("parent candidate source-video binding mismatch")
        event_ids = [str(event["event_id"]) for event in candidate["events"]]
        if (
            len(event_ids) != len(set(event_ids))
            or len(event_ids) != source["example_count"] + int(source_id in {"randolph", "vtv"})
            or any(event["event_type"] != "field_goal_attempt" for event in candidate["events"])
        ):
            raise ValueError("parent candidate event coverage is invalid")
        labels = _verify_label_payload(
            label_payload,
            source_video_sha256=str(source["source_video_sha256"]),
            candidate_bundle_sha256=str(candidate["bundle_sha256"]),
            expected_provenance=provenance,
            allowed_event_ids=event_ids,
            expected_example_count=int(source["example_count"]),
        )
        label_ids = [str(row["event_id"]) for row in labels["examples"]]
        if label_ids != [event_id for event_id in event_ids if event_id in set(label_ids)]:
            raise ValueError("parent labels are not ordered by candidate geometry")
        reconstructed.extend(
            {
                "source_video_sha256": str(source["source_video_sha256"]),
                "candidate_bundle_sha256": str(candidate["bundle_sha256"]),
                "event_id": str(row["event_id"]),
                "event_present": bool(row["event_present"]),
            }
            for row in labels["examples"]
        )
        candidates[candidate_path.name] = candidate
        labels_by_filename[label_path.name] = labels
        validated_paths.extend((candidate_path, label_path))
        reference_sources.append(
            {
                "source_id": source_id,
                "source_video_sha256": str(source["source_video_sha256"]),
                "candidate_bundle": candidate_ref,
                "labels": label_ref,
            }
        )

    reconstructed.sort(key=_example_sort_key)
    if reconstructed != parent["examples"]:
        raise ValueError("parent child labels do not reconstruct the parent index")
    reference = {
        "schema_version": TRAINING_EXPORT_SCHEMA,
        "artifact_sha256": parent["artifact_sha256"],
        "file_sha256": expected_file,
        "sources": reference_sources,
    }
    _verify_parent_reference(reference)
    return _VerifiedParent(
        index=parent,
        candidate_bundles=candidates,
        label_files=labels_by_filename,
        resolved_examples=tuple(reconstructed),
        validated_paths=tuple(validated_paths),
        reference=reference,
    )


def _verify_label_payload(
    payload: Mapping[str, Any],
    *,
    source_video_sha256: str,
    candidate_bundle_sha256: str,
    expected_provenance: Mapping[str, str],
    allowed_event_ids: Sequence[str],
    expected_example_count: int,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != _LABEL_FIELDS:
        raise ValueError("shot-validity label fields are not canonical")
    labels = dict(payload)
    if (
        labels.get("schema_version") != SHOT_VALIDITY_LABEL_SCHEMA
        or labels.get("purpose") != EXPORT_PURPOSE
        or labels.get("runtime_consumable") is not False
        or labels.get("formal_evaluation_eligible") is not False
        or labels.get("source_video_sha256") != source_video_sha256
        or labels.get("candidate_bundle_sha256") != candidate_bundle_sha256
        or labels.get("source_training_export_provenance") != dict(expected_provenance)
        or labels.get("mapping_policy") != _MAPPING_POLICY
    ):
        raise ValueError("shot-validity label provenance is invalid")
    examples = labels.get("examples")
    if not isinstance(examples, list) or len(examples) != expected_example_count:
        raise ValueError("shot-validity label count is invalid")
    allowed = set(allowed_event_ids)
    seen: set[str] = set()
    for row in examples:
        if not isinstance(row, Mapping) or set(row) != _LABEL_EXAMPLE_FIELDS:
            raise ValueError("shot-validity label rows are not canonical")
        event_id = _safe_id(row.get("event_id"), "shot-validity event ID")
        if event_id not in allowed or event_id in seen:
            raise ValueError("shot-validity label event coverage is invalid")
        if not isinstance(row.get("event_present"), bool):
            raise ValueError("shot-validity labels must be boolean")
        seen.add(event_id)
    return labels


def _verify_file_bytes(encoded: bytes, reference: Mapping[str, Any], description: str) -> None:
    if len(encoded) != reference.get("size_bytes") or _sha256_bytes(encoded) != reference.get("sha256"):
        raise ValueError(f"{description} file receipt mismatch")


def _read_json_object(path: Path, description: str) -> tuple[dict[str, Any], bytes]:
    try:
        if path.stat().st_size > _MAX_JSON_BYTES:
            raise ValueError(f"{description} exceeds the JSON size limit")
        encoded = path.read_bytes()
        payload = json.loads(encoded, object_pairs_hook=_unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object")
    return payload, encoded


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _resolve_child_path(root: Path, filename: Any, description: str) -> Path:
    basename = _safe_basename(filename, f"{description} filename")
    unresolved = root / basename
    if unresolved.is_symlink():
        raise ValueError(f"{description} must not be a symlink")
    path = unresolved.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"{description} is missing or outside its asset root")
    return path


def _safe_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{field} must be a path-safe scalar")
    return value


def _example_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["source_video_sha256"]),
        str(row["candidate_bundle_sha256"]),
        str(row["event_id"]),
    )


def _verify_harwood_chain(
    *,
    harwood_selection_path: Path,
    expected_harwood_selection_artifact_sha256: str,
    harwood_review_plan_path: Path,
    expected_harwood_review_plan_artifact_sha256: str,
    harwood_sealed_review_path: Path,
    expected_harwood_sealed_review_artifact_sha256: str,
    harwood_raw_frame_manifest_path: Path,
    expected_harwood_raw_frame_manifest_artifact_sha256: str,
    harwood_source_manifest_path: Path,
) -> _VerifiedHarwood:
    expected = {
        "selection": _require_sha(
            expected_harwood_selection_artifact_sha256,
            "expected Harwood selection artifact",
        ),
        "review_plan": _require_sha(
            expected_harwood_review_plan_artifact_sha256,
            "expected Harwood review-plan artifact",
        ),
        "sealed_review": _require_sha(
            expected_harwood_sealed_review_artifact_sha256,
            "expected Harwood sealed-review artifact",
        ),
        "raw_frame_manifest": _require_sha(
            expected_harwood_raw_frame_manifest_artifact_sha256,
            "expected Harwood raw-frame artifact",
        ),
    }
    for name, value in expected.items():
        frozen_name = "sealed_review" if name == "sealed_review" else name
        if value != FROZEN_HARWOOD_ARTIFACT_SHA256[frozen_name]:
            raise ValueError(f"expected Harwood {name} receipt is not TASK-0256")

    source_manifest, source_manifest_bytes = _read_json_object(harwood_source_manifest_path, "Harwood source manifest")
    selection_payload, selection_bytes = _read_json_object(harwood_selection_path, "Harwood continuous selection")
    plan_payload, plan_bytes = _read_json_object(harwood_review_plan_path, "Harwood causal review plan")
    review_payload, review_bytes = _read_json_object(harwood_sealed_review_path, "Harwood sealed causal review")
    raw_payload, raw_bytes = _read_json_object(harwood_raw_frame_manifest_path, "Harwood raw-frame manifest")
    encoded_by_name = {
        "source_manifest": source_manifest_bytes,
        "selection": selection_bytes,
        "review_plan": plan_bytes,
        "sealed_review": review_bytes,
        "raw_frame_manifest": raw_bytes,
    }
    for name, encoded in encoded_by_name.items():
        if _sha256_bytes(encoded) != FROZEN_HARWOOD_FILE_SHA256[name]:
            raise ValueError(f"Harwood {name} file SHA-256 is not frozen")

    selection = verify_continuous_causal_selection_receipt(
        selection_payload,
        expected_artifact_sha256=expected["selection"],
        source_manifest_path=harwood_source_manifest_path,
    )
    source = selection["source"]
    if (
        source["source_id"] != HARWOOD_SOURCE_ID
        or source["source_video_sha256"] != FROZEN_HARWOOD_VIDEO_SHA256
        or source["source_video_size_bytes"] != FROZEN_HARWOOD_VIDEO_SIZE_BYTES
        or source["source_video_filename"] != "hctv_harwood_2026.webm"
        or selection["source_manifest_sha256"] != FROZEN_HARWOOD_FILE_SHA256["source_manifest"]
        or len(selection["windows"]) != 24
    ):
        raise ValueError("Harwood selection does not bind the frozen source")
    source_video_path = resolve_continuous_source_video_path(
        harwood_source_manifest_path,
        source_id=HARWOOD_SOURCE_ID,
    ).resolve()

    plan = verify_vru_causal_review_plan(
        plan_payload,
        expected_artifact_sha256=expected["review_plan"],
    )
    _verify_harwood_selection_plan(selection=selection, plan=plan)
    review = verify_vru_causal_review(
        review_payload,
        plan=plan,
        expected_plan_artifact_sha256=expected["review_plan"],
    )
    if review["artifact_sha256"] != expected["sealed_review"]:
        raise ValueError("Harwood sealed-review artifact SHA-256 mismatch")
    _verify_harwood_review_partition(review=review, plan=plan)
    raw_manifest, jpeg_paths, jpeg_set_sha = _verify_harwood_raw_manifest(
        raw_payload,
        manifest_path=harwood_raw_frame_manifest_path,
        plan=plan,
        expected_artifact_sha256=expected["raw_frame_manifest"],
        source_manifest_sha256=FROZEN_HARWOOD_FILE_SHA256["source_manifest"],
    )

    reference = {
        "source_manifest": {
            "schema_version": str(source_manifest["schema_version"]),
            "file_sha256": FROZEN_HARWOOD_FILE_SHA256["source_manifest"],
        },
        "source_video": {
            "filename": str(source["source_video_filename"]),
            "sha256": str(source["source_video_sha256"]),
            "size_bytes": int(source["source_video_size_bytes"]),
        },
        "source_selection": _artifact_reference(selection, FROZEN_HARWOOD_FILE_SHA256["selection"]),
        "source_review_plan": _artifact_reference(plan, FROZEN_HARWOOD_FILE_SHA256["review_plan"]),
        "source_frame_manifest": {
            **_artifact_reference(
                raw_manifest,
                FROZEN_HARWOOD_FILE_SHA256["raw_frame_manifest"],
            ),
            "jpeg_count": len(jpeg_paths),
            "jpeg_set_sha256": jpeg_set_sha,
        },
        "source_review": _artifact_reference(review, FROZEN_HARWOOD_FILE_SHA256["sealed_review"]),
    }
    _verify_harwood_reference(reference)
    return _VerifiedHarwood(
        selection=selection,
        plan=plan,
        review=review,
        raw_manifest=raw_manifest,
        source_video_path=source_video_path,
        jpeg_paths=jpeg_paths,
        reference=reference,
        validated_paths=(
            harwood_selection_path.resolve(),
            harwood_review_plan_path.resolve(),
            harwood_sealed_review_path.resolve(),
            harwood_raw_frame_manifest_path.resolve(),
            harwood_source_manifest_path.resolve(),
            source_video_path,
            *jpeg_paths,
        ),
    )


def _verify_harwood_selection_plan(*, selection: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    if (
        plan.get("source_selection_artifact_sha256") != selection["artifact_sha256"]
        or plan.get("source_manifest_sha256") != selection["source_manifest_sha256"]
        or plan.get("source_video_sha256s") != [FROZEN_HARWOOD_VIDEO_SHA256]
    ):
        raise ValueError("Harwood selection/review-plan receipt binding mismatch")
    windows = selection["windows"]
    examples = plan["examples"]
    if len(examples) != len(windows):
        raise ValueError("Harwood selection/review-plan coverage mismatch")
    sample_count = int(selection["selection"]["sample_count"])
    sample_rate = float(selection["selection"]["sample_rate_hz"])
    if sample_count != 64 or sample_rate != 8.0 or selection["selection"]["interval_semantics"] != "half_open":
        raise ValueError("Harwood selection sampling contract is invalid")
    for ordinal, (window, example) in enumerate(zip(windows, examples, strict=True), start=1):
        expected_indexes = [
            round((float(window["start_seconds"]) + position / sample_rate) * float(window["source_fps"]))
            for position in range(sample_count)
        ]
        if (
            window["review_id"] != neutral_continuous_review_id(FROZEN_HARWOOD_VIDEO_SHA256, ordinal)
            or example["review_id"] != window["review_id"]
            or example["source_video_sha256"] != window["source_video_sha256"]
            or example["source_video_filename"] != neutral_vru_source_filename(FROZEN_HARWOOD_VIDEO_SHA256)
            or example["source_fps"] != window["source_fps"]
            or example["frame_count"] != window["frame_count"]
            or example["frame_indexes"] != expected_indexes
            or set(example["frame_sha256s"]) != {str(index) for index in expected_indexes}
        ):
            raise ValueError("Harwood selection/review-plan geometry mismatch")


def _verify_harwood_review_partition(*, review: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    planned_ids = [str(row["review_id"]) for row in plan["examples"]]
    rows = review["reviews"]
    if [str(row["review_id"]) for row in rows] != planned_ids:
        raise ValueError("Harwood review must canonically cover the review plan")
    counts = Counter(str(row["shot_sequence"]) for row in rows)
    uncertain = [str(row["review_id"]) for row in rows if row["shot_sequence"] == "uncertain"]
    label_identity = [
        {
            "review_id": str(row["review_id"]),
            "shot_sequence": str(row["shot_sequence"]),
        }
        for row in rows
    ]
    if (
        counts != Counter({"shot": 3, "not_a_shot": 20, "uncertain": 1})
        or uncertain != [FROZEN_HARWOOD_UNCERTAIN_ID]
        or _json_sha256(label_identity) != _FROZEN_HARWOOD_SHOT_SEQUENCE_SHA256
    ):
        raise ValueError("Harwood review requires the frozen 3/20/1 partition")


def _verify_harwood_raw_manifest(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
    plan: Mapping[str, Any],
    expected_artifact_sha256: str,
    source_manifest_sha256: str,
) -> tuple[dict[str, Any], tuple[Path, ...], str]:
    if not isinstance(payload, Mapping) or set(payload) != _RAW_MANIFEST_FIELDS:
        raise ValueError("Harwood raw-frame manifest fields are not canonical")
    artifact = dict(payload)
    claimed = _require_sha(artifact.pop("artifact_sha256", None), "Harwood raw-frame artifact")
    if claimed != canonical_sha256(artifact) or claimed != expected_artifact_sha256:
        raise ValueError("Harwood raw-frame manifest artifact SHA-256 mismatch")
    if (
        artifact.get("schema_version") != "agu.vru-causal-review-frame-manifest.v1"
        or artifact.get("purpose") != "offline_codex_visual_review_of_original_vru_frames"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("labels_hidden_from_reviewer") is not True
        or artifact.get("source_manifest_sha256") != source_manifest_sha256
        or artifact.get("review_plan_sha256") != plan["artifact_sha256"]
        or artifact.get("jpeg_quality") != 95
    ):
        raise ValueError("Harwood raw-frame manifest provenance is invalid")
    frame_root_name = _safe_basename(artifact.get("frame_root"), "frame root")
    manifest_parent = manifest_path.parent.resolve()
    unresolved_frame_root = manifest_parent / frame_root_name
    if unresolved_frame_root.is_symlink():
        raise ValueError("Harwood frame root cannot be a symlink")
    frame_root = unresolved_frame_root.resolve()
    if not frame_root.is_relative_to(manifest_parent) or not frame_root.is_dir():
        raise ValueError("Harwood frame root is missing or escapes its manifest")
    expected_rows = [
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
        or artifact.get("frame_count") != 1536
        or len(rows) != len(expected_rows)
        or len(rows) != 1536
    ):
        raise ValueError("Harwood raw-frame manifest must cover exactly 1,536 JPEGs")
    paths: list[Path] = []
    path_keys: set[str] = set()
    identities: set[tuple[int, int]] = set()
    descriptors: list[dict[str, Any]] = []
    for row, expected_row in zip(rows, expected_rows, strict=True):
        if not isinstance(row, Mapping) or set(row) != _RAW_FRAME_FIELDS:
            raise ValueError("Harwood raw-frame rows are not canonical")
        actual = (
            str(row.get("review_id") or ""),
            _non_negative_int(row.get("position"), "raw-frame position"),
            _non_negative_int(row.get("frame_index"), "raw-frame index"),
            _require_sha(row.get("raw_frame_sha256"), "raw frame"),
        )
        if actual != expected_row:
            raise ValueError("Harwood raw-frame coverage does not match the plan")
        relative = _safe_relative_path(row.get("relative_path"), "review JPEG path")
        unresolved_jpeg_path = frame_root / relative
        jpeg_path = unresolved_jpeg_path.resolve()
        key = jpeg_path.as_posix().casefold()
        if (
            not jpeg_path.is_relative_to(frame_root)
            or not jpeg_path.is_file()
            or unresolved_jpeg_path.is_symlink()
            or key in path_keys
        ):
            raise ValueError("Harwood review JPEG path is missing or aliased")
        status = jpeg_path.stat()
        identity = (status.st_dev, status.st_ino)
        if identity in identities:
            raise ValueError("Harwood review JPEGs must be physically unique")
        jpeg_bytes = _positive_int(row.get("jpeg_bytes"), "review JPEG size")
        jpeg_sha = _require_sha(row.get("jpeg_sha256"), "review JPEG")
        if status.st_size != jpeg_bytes or _file_sha256(jpeg_path) != jpeg_sha:
            raise ValueError("Harwood review JPEG receipt mismatch")
        paths.append(jpeg_path)
        path_keys.add(key)
        identities.add(identity)
        descriptors.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": jpeg_bytes,
                "sha256": jpeg_sha,
            }
        )
    actual_entries = [path for path in frame_root.rglob("*") if path.is_file()]
    if any(path.is_symlink() for path in actual_entries):
        raise ValueError("Harwood frame root cannot contain symlinked JPEGs")
    actual_files = {path.resolve() for path in actual_entries}
    if actual_files != set(paths):
        raise ValueError("Harwood frame root contains missing or extra files")
    artifact["artifact_sha256"] = claimed
    return artifact, tuple(paths), _json_sha256(descriptors)


def _artifact_reference(payload: Mapping[str, Any], file_sha256: str) -> dict[str, str]:
    return {
        "schema_version": str(payload["schema_version"]),
        "artifact_sha256": _require_sha(payload["artifact_sha256"], "artifact"),
        "file_sha256": _require_sha(file_sha256, "artifact file"),
    }


def _safe_relative_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{field} must be a path-safe relative path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a path-safe relative path")
    return path


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: object) -> str:
    return _sha256_bytes(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def build_vru_causal_shot_validity_training_extension_export(
    *,
    parent_export_path: Path,
    expected_parent_artifact_sha256: str,
    expected_parent_file_sha256: str,
    parent_asset_root: Path,
    harwood_selection_path: Path,
    expected_harwood_selection_artifact_sha256: str,
    harwood_review_plan_path: Path,
    expected_harwood_review_plan_artifact_sha256: str,
    harwood_sealed_review_path: Path,
    expected_harwood_sealed_review_artifact_sha256: str,
    harwood_raw_frame_manifest_path: Path,
    expected_harwood_raw_frame_manifest_artifact_sha256: str,
    harwood_source_manifest_path: Path,
    source_groups_path: Path,
    expected_source_groups_artifact_sha256: str,
) -> ContinuousCausalTrainingChain:
    """Build the additive Harwood children and receipt-only v2 index."""

    source_groups, source_groups_file_sha = _verify_source_groups_file(
        source_groups_path,
        expected_artifact_sha256=expected_source_groups_artifact_sha256,
    )
    parent = _verify_parent_chain(
        parent_export_path=parent_export_path,
        expected_parent_artifact_sha256=expected_parent_artifact_sha256,
        expected_parent_file_sha256=expected_parent_file_sha256,
        parent_asset_root=parent_asset_root,
    )
    harwood = _verify_harwood_chain(
        harwood_selection_path=harwood_selection_path,
        expected_harwood_selection_artifact_sha256=(expected_harwood_selection_artifact_sha256),
        harwood_review_plan_path=harwood_review_plan_path,
        expected_harwood_review_plan_artifact_sha256=(expected_harwood_review_plan_artifact_sha256),
        harwood_sealed_review_path=harwood_sealed_review_path,
        expected_harwood_sealed_review_artifact_sha256=(expected_harwood_sealed_review_artifact_sha256),
        harwood_raw_frame_manifest_path=harwood_raw_frame_manifest_path,
        expected_harwood_raw_frame_manifest_artifact_sha256=(expected_harwood_raw_frame_manifest_artifact_sha256),
        harwood_source_manifest_path=harwood_source_manifest_path,
    )
    _verify_source_group_bindings(
        source_groups=source_groups,
        parent=parent,
        harwood=harwood,
    )
    candidate = _build_harwood_candidate(harwood)
    labels = _build_harwood_labels(harwood=harwood, candidate=candidate)
    candidate_bytes = encode_json_artifact(candidate)
    label_bytes = encode_json_artifact(labels)
    extension_ref = {
        "source_id": HARWOOD_SOURCE_ID,
        "game_id": HARWOOD_GAME_ID,
        "source_video_sha256": FROZEN_HARWOOD_VIDEO_SHA256,
        "candidate_bundle": {
            "filename": HARWOOD_CANDIDATE_FILENAME,
            "sha256": _sha256_bytes(candidate_bytes),
            "size_bytes": len(candidate_bytes),
            "bundle_sha256": candidate["bundle_sha256"],
        },
        "labels": {
            "filename": HARWOOD_LABEL_FILENAME,
            "sha256": _sha256_bytes(label_bytes),
            "size_bytes": len(label_bytes),
        },
        "selected_count": 24,
        "example_count": 23,
        "positive_count": 3,
        "negative_count": 20,
        "excluded_uncertain": 1,
    }
    source_group_ref = {
        "schema_version": SOURCE_GROUP_SCHEMA,
        "artifact_sha256": source_groups["artifact_sha256"],
        "file_sha256": source_groups_file_sha,
    }
    index: dict[str, Any] = {
        "schema_version": TRAINING_EXPORT_V2_SCHEMA,
        "purpose": EXPORT_PURPOSE,
        "runtime_consumable": False,
        "training_consumable": True,
        "formal_evaluation_eligible": False,
        "promotion_eligible": False,
        "promoted": False,
        "codex_runtime_answer_used": False,
        "parent": parent.reference,
        "harwood_evidence": harwood.reference,
        "extension": extension_ref,
        "source_groups": source_group_ref,
        "mapping_policy": dict(_MAPPING_POLICY),
        "training_routes": dict(_TRAINING_ROUTES),
        "counts": {
            "selected": 48,
            "exported": 45,
            "positive": 17,
            "negative": 28,
            "excluded_uncertain": 3,
            "game_groups": 4,
            "production_families": 2,
        },
        "excluded_review_ids": list(EXPECTED_EXCLUDED_REVIEW_IDS),
    }
    index["artifact_sha256"] = canonical_sha256(index)
    verified_index = _verify_v2_index(index)
    resolved = _resolve_examples(
        parent=parent,
        candidate=candidate,
        labels=labels,
        source_groups=source_groups,
    )
    return ContinuousCausalTrainingChain(
        index=verified_index,
        extension_candidate_bundles={HARWOOD_CANDIDATE_FILENAME: candidate},
        extension_label_files={HARWOOD_LABEL_FILENAME: labels},
        source_groups=source_groups,
        resolved_examples=resolved,
        validated_input_paths=_canonical_paths(
            (
                *parent.validated_paths,
                *harwood.validated_paths,
                source_groups_path.resolve(),
            )
        ),
    )


def verify_vru_causal_shot_validity_training_extension_export(
    payload: Mapping[str, Any],
    *,
    expected_artifact_sha256: str,
    extension_asset_root: Path,
    parent_export_path: Path,
    expected_parent_artifact_sha256: str,
    expected_parent_file_sha256: str,
    parent_asset_root: Path,
    harwood_selection_path: Path,
    expected_harwood_selection_artifact_sha256: str,
    harwood_review_plan_path: Path,
    expected_harwood_review_plan_artifact_sha256: str,
    harwood_sealed_review_path: Path,
    expected_harwood_sealed_review_artifact_sha256: str,
    harwood_raw_frame_manifest_path: Path,
    expected_harwood_raw_frame_manifest_artifact_sha256: str,
    harwood_source_manifest_path: Path,
    source_groups_path: Path,
    expected_source_groups_artifact_sha256: str,
) -> ContinuousCausalTrainingChain:
    """Replay every stored edge and return the resolved four-game view."""

    index = _verify_v2_index(
        payload,
        expected_artifact_sha256=expected_artifact_sha256,
    )
    source_groups, source_groups_file_sha = _verify_source_groups_file(
        source_groups_path,
        expected_artifact_sha256=expected_source_groups_artifact_sha256,
    )
    parent = _verify_parent_chain(
        parent_export_path=parent_export_path,
        expected_parent_artifact_sha256=expected_parent_artifact_sha256,
        expected_parent_file_sha256=expected_parent_file_sha256,
        parent_asset_root=parent_asset_root,
    )
    harwood = _verify_harwood_chain(
        harwood_selection_path=harwood_selection_path,
        expected_harwood_selection_artifact_sha256=(expected_harwood_selection_artifact_sha256),
        harwood_review_plan_path=harwood_review_plan_path,
        expected_harwood_review_plan_artifact_sha256=(expected_harwood_review_plan_artifact_sha256),
        harwood_sealed_review_path=harwood_sealed_review_path,
        expected_harwood_sealed_review_artifact_sha256=(expected_harwood_sealed_review_artifact_sha256),
        harwood_raw_frame_manifest_path=harwood_raw_frame_manifest_path,
        expected_harwood_raw_frame_manifest_artifact_sha256=(expected_harwood_raw_frame_manifest_artifact_sha256),
        harwood_source_manifest_path=harwood_source_manifest_path,
    )
    _verify_source_group_bindings(
        source_groups=source_groups,
        parent=parent,
        harwood=harwood,
    )
    expected_source_group_ref = {
        "schema_version": SOURCE_GROUP_SCHEMA,
        "artifact_sha256": source_groups["artifact_sha256"],
        "file_sha256": source_groups_file_sha,
    }
    if (
        index["parent"] != parent.reference
        or index["harwood_evidence"] != harwood.reference
        or index["source_groups"] != expected_source_group_ref
    ):
        raise ValueError("training v2 index evidence references do not replay")

    root = extension_asset_root.resolve()
    if not root.is_dir():
        raise ValueError("training v2 extension asset root must be a directory")
    extension = index["extension"]
    candidate_ref = extension["candidate_bundle"]
    label_ref = extension["labels"]
    candidate_path = _resolve_child_path(root, candidate_ref["filename"], "Harwood v2 candidate")
    label_path = _resolve_child_path(root, label_ref["filename"], "Harwood v2 labels")
    _require_extension_paths_disjoint(
        extension_paths=(candidate_path, label_path),
        evidence_paths=(
            *parent.validated_paths,
            *harwood.validated_paths,
            source_groups_path.resolve(),
        ),
    )
    candidate_payload, candidate_bytes = _read_json_object(candidate_path, "Harwood v2 candidate bundle")
    label_payload, label_bytes = _read_json_object(label_path, "Harwood v2 shot-validity labels")
    _verify_file_bytes(candidate_bytes, candidate_ref, "Harwood v2 candidate")
    _verify_file_bytes(label_bytes, label_ref, "Harwood v2 labels")
    expected_candidate = _build_harwood_candidate(harwood)
    if candidate_payload != expected_candidate:
        raise ValueError("Harwood v2 candidate does not replay label-hidden geometry")
    expected_labels = _build_harwood_labels(
        harwood=harwood,
        candidate=expected_candidate,
    )
    if label_payload != expected_labels:
        raise ValueError("Harwood v2 labels do not replay the sealed review")
    if candidate_ref["bundle_sha256"] != expected_candidate["bundle_sha256"] or extension != {
        "source_id": HARWOOD_SOURCE_ID,
        "game_id": HARWOOD_GAME_ID,
        "source_video_sha256": FROZEN_HARWOOD_VIDEO_SHA256,
        "candidate_bundle": {
            "filename": HARWOOD_CANDIDATE_FILENAME,
            "sha256": _sha256_bytes(candidate_bytes),
            "size_bytes": len(candidate_bytes),
            "bundle_sha256": expected_candidate["bundle_sha256"],
        },
        "labels": {
            "filename": HARWOOD_LABEL_FILENAME,
            "sha256": _sha256_bytes(label_bytes),
            "size_bytes": len(label_bytes),
        },
        "selected_count": 24,
        "example_count": 23,
        "positive_count": 3,
        "negative_count": 20,
        "excluded_uncertain": 1,
    }:
        raise ValueError("training v2 extension reference does not match its children")
    resolved = _resolve_examples(
        parent=parent,
        candidate=expected_candidate,
        labels=expected_labels,
        source_groups=source_groups,
    )
    return ContinuousCausalTrainingChain(
        index=index,
        extension_candidate_bundles={HARWOOD_CANDIDATE_FILENAME: expected_candidate},
        extension_label_files={HARWOOD_LABEL_FILENAME: expected_labels},
        source_groups=source_groups,
        resolved_examples=resolved,
        validated_input_paths=_canonical_paths(
            (
                *parent.validated_paths,
                *harwood.validated_paths,
                source_groups_path.resolve(),
                candidate_path,
                label_path,
            )
        ),
    )


def _require_extension_paths_disjoint(*, extension_paths: Sequence[Path], evidence_paths: Sequence[Path]) -> None:
    extension_casefolds = [path.as_posix().casefold() for path in extension_paths]
    if len(extension_casefolds) != len(set(extension_casefolds)):
        raise ValueError("training v2 extension child paths are aliased")
    evidence_casefolds = {path.as_posix().casefold() for path in evidence_paths}
    if any(path in evidence_casefolds for path in extension_casefolds):
        raise ValueError("training v2 extension child aliases evidence")
    evidence_identities = {
        (status.st_dev, status.st_ino) for path in evidence_paths if (status := _stat_or_none(path)) is not None
    }
    extension_identities: set[tuple[int, int]] = set()
    for path in extension_paths:
        status = path.stat()
        identity = (status.st_dev, status.st_ino)
        if identity in evidence_identities or identity in extension_identities:
            raise ValueError("training v2 extension child is hardlink-aliased")
        extension_identities.add(identity)


def _stat_or_none(path: Path) -> Any:
    try:
        return path.stat()
    except FileNotFoundError:
        return None


def _verify_source_groups_file(path: Path, *, expected_artifact_sha256: str) -> tuple[dict[str, Any], str]:
    expected = _require_sha(expected_artifact_sha256, "expected source-group artifact")
    if expected != _FROZEN_SOURCE_GROUP_ARTIFACT_SHA256:
        raise ValueError("expected source-group artifact receipt is not TASK-0257")
    payload, encoded = _read_json_object(path, "source-group manifest")
    file_sha = _sha256_bytes(encoded)
    if file_sha != _FROZEN_SOURCE_GROUP_FILE_SHA256:
        raise ValueError("source-group manifest file receipt is not frozen")
    verified = verify_vru_causal_source_groups(
        payload,
        expected_artifact_sha256=expected,
    )
    return verified, file_sha


def _verify_source_group_bindings(
    *,
    source_groups: Mapping[str, Any],
    parent: _VerifiedParent,
    harwood: _VerifiedHarwood,
) -> None:
    parent_hashes = {str(row["source_id"]): str(row["source_video_sha256"]) for row in parent.index["sources"]}
    expected_hashes = {
        **parent_hashes,
        HARWOOD_GAME_ID: str(harwood.selection["source"]["source_video_sha256"]),
    }
    actual_hashes = {str(row["game_id"]): str(row["source_video_sha256"]) for row in source_groups["games"]}
    if actual_hashes != expected_hashes:
        raise ValueError("source-group manifest does not bind the four training games")


def _build_harwood_candidate(harwood: _VerifiedHarwood) -> dict[str, Any]:
    source = harwood.selection["source"]
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
        for row in harwood.plan["examples"]
    ]
    config = {
        "source_selection_artifact_sha256": harwood.selection["artifact_sha256"],
        "geometry": "full_half_open_eight_second_window",
        "label_selection_not_used": True,
        "target_conditioned_anchor_used": False,
    }
    candidate: dict[str, Any] = {
        "schema_version": "agu.raw-only.v1",
        "game_id": HARWOOD_GAME_ID,
        "raw_videos": [
            {
                "video_id": "video_001",
                "filename": str(source["source_video_filename"]),
                "sha256": str(source["source_video_sha256"]),
                "size_bytes": int(source["source_video_size_bytes"]),
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
        "inference_completed_at": FROZEN_HARWOOD_COMPLETED_AT,
        "events_sha256": _json_sha256(events),
    }
    candidate["bundle_sha256"] = _json_sha256(candidate)
    verified = verify_raw_only_bundle(candidate).model_dump(mode="json")
    if verified != candidate:
        raise ValueError("Harwood candidate bundle is not canonical")
    return candidate


def _build_harwood_labels(*, harwood: _VerifiedHarwood, candidate: Mapping[str, Any]) -> dict[str, Any]:
    reviews = {str(row["review_id"]): row for row in harwood.review["reviews"]}
    examples = [
        {
            "event_id": str(row["review_id"]),
            "event_present": reviews[str(row["review_id"])]["shot_sequence"] == "shot",
        }
        for row in harwood.plan["examples"]
        if reviews[str(row["review_id"])]["shot_sequence"] != "uncertain"
    ]
    labels = {
        "schema_version": SHOT_VALIDITY_LABEL_SCHEMA,
        "purpose": EXPORT_PURPOSE,
        "runtime_consumable": False,
        "formal_evaluation_eligible": False,
        "source_video_sha256": FROZEN_HARWOOD_VIDEO_SHA256,
        "candidate_bundle_sha256": candidate["bundle_sha256"],
        "source_training_export_provenance": {
            "source_selection": harwood.selection["artifact_sha256"],
            "source_review_plan": harwood.plan["artifact_sha256"],
            "source_review": harwood.review["artifact_sha256"],
            "source_frame_manifest": harwood.raw_manifest["artifact_sha256"],
        },
        "mapping_policy": dict(_MAPPING_POLICY),
        "examples": examples,
    }
    _verify_label_payload(
        labels,
        source_video_sha256=FROZEN_HARWOOD_VIDEO_SHA256,
        candidate_bundle_sha256=str(candidate["bundle_sha256"]),
        expected_provenance=labels["source_training_export_provenance"],
        allowed_event_ids=[str(row["review_id"]) for row in harwood.plan["examples"]],
        expected_example_count=23,
    )
    if sum(bool(row["event_present"]) for row in examples) != 3 or {str(row["event_id"]) for row in examples} & {
        FROZEN_HARWOOD_UNCERTAIN_ID
    }:
        raise ValueError("Harwood labels do not preserve the frozen partition")
    return labels


def _resolve_examples(
    *,
    parent: _VerifiedParent,
    candidate: Mapping[str, Any],
    labels: Mapping[str, Any],
    source_groups: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    game_by_source_sha = {str(row["source_video_sha256"]): str(row["game_id"]) for row in source_groups["games"]}
    rows = [dict(row) for row in parent.resolved_examples]
    rows.extend(
        {
            "source_video_sha256": FROZEN_HARWOOD_VIDEO_SHA256,
            "candidate_bundle_sha256": str(candidate["bundle_sha256"]),
            "event_id": str(row["event_id"]),
            "event_present": bool(row["event_present"]),
        }
        for row in labels["examples"]
    )
    order = {game_id: index for index, game_id in enumerate(GAME_IDS)}
    rows.sort(
        key=lambda row: (
            order[game_by_source_sha[str(row["source_video_sha256"])]],
            str(row["event_id"]),
        )
    )
    seen: set[tuple[str, str, str]] = set()
    counts_by_game: Counter[str] = Counter()
    positives_by_game: Counter[str] = Counter()
    for row in rows:
        if set(row) != {
            "source_video_sha256",
            "candidate_bundle_sha256",
            "event_id",
            "event_present",
        }:
            raise ValueError("resolved training row fields are not canonical")
        source_sha = _require_sha(row["source_video_sha256"], "resolved source")
        bundle_sha = _require_sha(row["candidate_bundle_sha256"], "resolved candidate bundle")
        event_id = _safe_id(row["event_id"], "resolved event ID")
        if not isinstance(row["event_present"], bool):
            raise ValueError("resolved event_present must be boolean")
        key = (source_sha, bundle_sha, event_id)
        if key in seen or event_id in EXPECTED_EXCLUDED_REVIEW_IDS:
            raise ValueError("resolved training row is duplicated or excluded")
        seen.add(key)
        game_id = game_by_source_sha.get(source_sha)
        if game_id is None:
            raise ValueError("resolved source is absent from source groups")
        counts_by_game[game_id] += 1
        positives_by_game[game_id] += int(row["event_present"])
    expected_counts = {
        "hazen": (8, 6),
        "randolph": (7, 4),
        "vtv": (7, 4),
        "harwood": (23, 3),
    }
    if (
        len(rows) != 45
        or sum(bool(row["event_present"]) for row in rows) != 17
        or any(
            (counts_by_game[game_id], positives_by_game[game_id]) != values
            for game_id, values in expected_counts.items()
        )
    ):
        raise ValueError("resolved four-game training counts are invalid")
    return tuple(rows)


def _canonical_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    resolved = {path.resolve() for path in paths}
    return tuple(sorted(resolved, key=lambda path: path.as_posix()))
