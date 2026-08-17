#!/usr/bin/env python3
"""Extract sealed training-only embeddings with a registered video backbone."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.official_evaluation import verify_raw_only_bundle  # noqa: E402
from app.analysis.schemas import RawOnlyPredictionBundleResponse  # noqa: E402
from app.analysis.shot_validity import SHOT_VALIDITY_LABEL_SCHEMA  # noqa: E402
from app.analysis.shot_validity_sampling_window import (  # noqa: E402
    SHOT_SAMPLING_PROTOCOL,
    resolve_shot_sampling_window,
)
from app.analysis.shot_validity_video_backbone import (  # noqa: E402
    VIDEO_BACKBONES,
    extract_video_embeddings,
    get_video_backbone_spec,
    load_video_backbone,
    seal_video_embedding_artifact,
    verify_video_embedding_artifact,
)
from app.analysis.training_annotation import verify_training_annotation_manifest  # noqa: E402

GIB = 1024**3
DEFAULT_MAX_SYSTEM_MEMORY_PERCENT = 90.0
DEFAULT_MIN_AVAILABLE_MEMORY_GIB = 2.0
DEFAULT_MIN_OUTPUT_FREE_GIB = 2.0
MAX_JSON_SNAPSHOT_BYTES = 64 * 1024**2
MAX_BACKBONE_SNAPSHOT_BYTES = 200 * 1024**2
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MANIFEST_ASSET_FIELDS = frozenset({"filename", "sha256", "size_bytes"})
_TRAINING_EXPORT_INDEX_FILENAMES = frozenset({"training_export.json", "training_export_v2.json"})
_EMBEDDING_EXAMPLE_FIELDS = frozenset(
    {
        "source_video_sha256",
        "candidate_bundle_sha256",
        "event_id",
        "event_present",
        "sampling_window",
        "embedding",
    }
)
_SAMPLING_WINDOW_FIELDS = frozenset(
    {
        "start_frame",
        "end_frame",
        "anchor_frame",
        "anchor_source",
        "protocol",
    }
)
_EMBEDDING_ARTIFACT_FIELDS = frozenset(
    {
        "artifact_sha256",
        "schema_version",
        "runtime_consumable",
        "purpose",
        "producer",
        "training_manifest_sha256",
        "training_annotation_sha256",
        "source_video_sha256",
        "backbone",
        "backbone_sha256",
        "backbone_license",
        "backbone_weights_url",
        "embedding_dimension",
        "clip_frames",
        "sampling_protocol",
        "examples",
    }
)


@dataclass(frozen=True)
class _ByteSnapshot:
    path: Path
    payload: bytes
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class _ValidatedSource:
    video_path: Path
    video_sha256: str
    video_size_bytes: int
    bundle_sha256: str
    annotation_sha256: str
    rows: tuple[dict[str, object], ...]
    events: tuple[object, ...]


@dataclass(frozen=True)
class _PreparedRow:
    video_path: Path
    source_video_sha256: str
    candidate_bundle_sha256: str
    event: object
    event_present: bool
    sampling_window: object


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not bool")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number")
    return parsed


def _validate_max_system_memory_percent(value: object) -> float:
    parsed = _finite_float(value, name="max_system_memory_percent")
    if not 0 < parsed <= DEFAULT_MAX_SYSTEM_MEMORY_PERCENT:
        raise ValueError("max_system_memory_percent must be in (0, 90] and may not weaken the default preflight")
    return parsed


def _validate_minimum_gib(value: object, *, name: str) -> float:
    parsed = _finite_float(value, name=name)
    if parsed < 2.0:
        raise ValueError(f"{name} must be at least 2 GiB")
    return parsed


def _arg_max_system_memory_percent(value: str) -> float:
    try:
        return _validate_max_system_memory_percent(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _arg_minimum_available_gib(value: str) -> float:
    try:
        return _validate_minimum_gib(value, name="min_available_memory_gib")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _arg_minimum_output_free_gib(value: str) -> float:
    try:
        return _validate_minimum_gib(value, name="min_output_free_gib")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _arg_unit_batch_size(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("batch size must be exactly 1") from error
    if parsed != 1:
        raise argparse.ArgumentTypeError("batch size must be exactly 1")
    return parsed


def _validate_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact lowercase SHA-256")
    return value


def _arg_sha256(value: str) -> str:
    try:
        return _validate_sha256(value, name="SHA-256 receipt")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-bundle", type=Path, action="append", required=True)
    parser.add_argument("--annotation", type=Path, action="append", required=True)
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--backbone", choices=sorted(VIDEO_BACKBONES), required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-backbone-sha256", type=_arg_sha256, required=True)
    parser.add_argument(
        "--expected-training-manifest-sha256",
        type=_arg_sha256,
        required=True,
    )
    parser.add_argument(
        "--expected-training-manifest-file-sha256",
        type=_arg_sha256,
        required=True,
    )
    parser.add_argument("--resume-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-resume-checkpoint-sha256", type=_arg_sha256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=_arg_unit_batch_size, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--max-memory-percent",
        dest="max_system_memory_percent",
        type=_arg_max_system_memory_percent,
        default=DEFAULT_MAX_SYSTEM_MEMORY_PERCENT,
    )
    parser.add_argument(
        "--min-available-gib",
        type=_arg_minimum_available_gib,
        default=DEFAULT_MIN_AVAILABLE_MEMORY_GIB,
    )
    parser.add_argument(
        "--min-output-free-gib",
        type=_arg_minimum_output_free_gib,
        default=DEFAULT_MIN_OUTPUT_FREE_GIB,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from the separately hash-verified staging checkpoint",
    )
    args = parser.parse_args()
    if args.resume and args.expected_resume_checkpoint_sha256 is None:
        parser.error("--resume requires --expected-resume-checkpoint-sha256")
    if not args.resume and args.expected_resume_checkpoint_sha256 is not None:
        parser.error("--expected-resume-checkpoint-sha256 requires --resume")
    return args


def extract_labeled_video_embeddings(
    *,
    manifest_path: Path,
    candidate_bundle_paths: list[Path],
    annotation_paths: list[Path],
    video_paths: list[Path],
    backbone_name: str,
    backbone_checkpoint: Path,
    clip_frames: int,
    batch_size: int,
    device_name: str,
    expected_backbone_sha256: str,
    expected_training_manifest_sha256: str,
    expected_training_manifest_file_sha256: str,
    checkpoint_path: Path | None = None,
    resume_checkpoint_path: Path | None = None,
    output_path: Path | None = None,
    resume: bool = False,
    expected_resume_checkpoint_sha256: str | None = None,
    max_system_memory_percent: float = DEFAULT_MAX_SYSTEM_MEMORY_PERCENT,
    min_available_memory_gib: float = DEFAULT_MIN_AVAILABLE_MEMORY_GIB,
    min_output_free_gib: float = DEFAULT_MIN_OUTPUT_FREE_GIB,
) -> dict[str, object]:
    if type(batch_size) is not int or batch_size != 1:
        raise ValueError("video embedding extraction requires batch size exactly 1")
    if not (len(candidate_bundle_paths) == len(annotation_paths) == len(video_paths)):
        raise ValueError("candidate bundles, annotations, and videos must be paired")
    max_system_memory_percent = _validate_max_system_memory_percent(max_system_memory_percent)
    min_available_memory_gib = _validate_minimum_gib(
        min_available_memory_gib,
        name="min_available_memory_gib",
    )
    min_output_free_gib = _validate_minimum_gib(
        min_output_free_gib,
        name="min_output_free_gib",
    )
    (
        manifest_path,
        candidate_bundle_paths,
        annotation_paths,
        video_paths,
        backbone_checkpoint,
        resume_checkpoint_path,
        output_path,
    ) = _resolved_disjoint_paths(
        manifest_path=manifest_path,
        candidate_bundle_paths=candidate_bundle_paths,
        annotation_paths=annotation_paths,
        video_paths=video_paths,
        backbone_checkpoint=backbone_checkpoint,
        checkpoint_path=checkpoint_path,
        resume_checkpoint_path=resume_checkpoint_path,
        output_path=output_path,
    )
    if output_path is not None and output_path.exists() and not output_path.is_file():
        raise ValueError("final output path must be absent or a regular file")
    snapshot_parent = _snapshot_parent(
        resume_checkpoint_path=resume_checkpoint_path,
        output_path=output_path,
        backbone_checkpoint=backbone_checkpoint,
    )
    _run_resource_preflight(
        output_paths=[path for path in (resume_checkpoint_path, output_path) if path is not None],
        snapshot_parent=snapshot_parent,
        max_system_memory_percent=max_system_memory_percent,
        min_available_memory_gib=min_available_memory_gib,
        min_output_free_gib=min_output_free_gib,
    )
    expected_backbone_sha256 = _validate_sha256(
        expected_backbone_sha256,
        name="expected_backbone_sha256",
    )
    _validate_manifest_receipts(
        expected_training_manifest_sha256=expected_training_manifest_sha256,
        expected_training_manifest_file_sha256=expected_training_manifest_file_sha256,
    )
    if resume:
        if resume_checkpoint_path is None:
            raise ValueError("resume requires a separate resume checkpoint path")
        if expected_resume_checkpoint_sha256 is None:
            raise ValueError("resume requires an expected checkpoint SHA-256 receipt")
        expected_resume_checkpoint_sha256 = _validate_sha256(
            expected_resume_checkpoint_sha256,
            name="expected_resume_checkpoint_sha256",
        )
        if not resume_checkpoint_path.is_file():
            raise ValueError("resume checkpoint does not exist")
    elif expected_resume_checkpoint_sha256 is not None:
        raise ValueError("expected resume checkpoint SHA-256 requires resume=true")
    elif resume_checkpoint_path is not None and resume_checkpoint_path.exists():
        raise ValueError("staging checkpoint already exists; use resume with its receipt")

    manifest_snapshot = _read_byte_snapshot(
        manifest_path,
        label="training manifest",
        maximum_bytes=MAX_JSON_SNAPSHOT_BYTES,
    )
    if manifest_snapshot.sha256 != expected_training_manifest_file_sha256:
        raise ValueError("training manifest file SHA-256 mismatch")
    manifest = verify_training_annotation_manifest(_json_object_from_snapshot(manifest_snapshot))
    if manifest["manifest_sha256"] != expected_training_manifest_sha256:
        raise ValueError("training manifest internal SHA-256 mismatch")
    if "shot_validity" not in manifest.get("task_types", []):
        raise ValueError("training manifest does not authorize shot_validity")
    del manifest_snapshot

    manifest_sources = _strict_manifest_assets(
        manifest.get("source_videos"),
        label="training manifest source video",
    )
    manifest_annotations = _strict_manifest_assets(
        manifest.get("annotation_files"),
        label="training manifest annotation",
    )
    if len(manifest_sources) not in (3, 4):
        raise ValueError("training manifest must describe a complete v1 or v2 source set")
    if len(video_paths) != len(manifest_sources):
        raise ValueError("candidate/annotation/video pairs must cover every training manifest source")
    if len(manifest_annotations) != len(manifest_sources) + 1:
        raise ValueError("training manifest must contain one label per source plus one export index")

    backbone_snapshot = _read_byte_snapshot(
        backbone_checkpoint,
        label="backbone checkpoint",
        maximum_bytes=MAX_BACKBONE_SNAPSHOT_BYTES,
    )
    if backbone_snapshot.sha256 != expected_backbone_sha256:
        raise ValueError("backbone checkpoint SHA-256 mismatch")
    backbone_sha256 = backbone_snapshot.sha256

    allowed_sources = set(manifest_sources)
    allowed_annotations = set(manifest_annotations)
    provided_sources: list[tuple[str, str, int]] = []
    provided_annotations: list[tuple[str, str, int]] = []
    validated_sources: list[_ValidatedSource] = []
    for bundle_path, annotation_path, video_path in zip(
        candidate_bundle_paths,
        annotation_paths,
        video_paths,
        strict=True,
    ):
        bundle_snapshot = _read_byte_snapshot(
            bundle_path,
            label="candidate bundle",
            maximum_bytes=MAX_JSON_SNAPSHOT_BYTES,
        )
        bundle = verify_raw_only_bundle(RawOnlyPredictionBundleResponse.model_validate_json(bundle_snapshot.payload))
        del bundle_snapshot
        if len(bundle.raw_videos) != 1:
            raise ValueError("video embedding extraction requires one video per bundle")
        annotation_snapshot = _read_byte_snapshot(
            annotation_path,
            label="shot-validity annotation",
            maximum_bytes=MAX_JSON_SNAPSHOT_BYTES,
        )
        annotation_asset = (
            annotation_path.name,
            annotation_snapshot.sha256,
            annotation_snapshot.size_bytes,
        )
        if annotation_asset not in allowed_annotations:
            raise ValueError("annotation is not SHA-bound by training manifest")
        provided_annotations.append(annotation_asset)
        annotation_sha256 = annotation_snapshot.sha256
        labels = _json_object_from_snapshot(annotation_snapshot)
        del annotation_snapshot
        source_hash, source_size = _stream_file_sha256(video_path)
        source_asset = bundle.raw_videos[0]
        provided_source = (video_path.name, source_hash, source_size)
        if (
            provided_source not in allowed_sources
            or source_asset.filename != video_path.name
            or source_asset.sha256 != source_hash
            or source_asset.size_bytes != source_size
        ):
            raise ValueError("video is not SHA-bound to bundle and training manifest")
        provided_sources.append(provided_source)
        rows, events = _validate_label_rows(
            labels=labels,
            bundle=bundle,
            source_hash=source_hash,
        )
        validated_sources.append(
            _ValidatedSource(
                video_path=video_path,
                video_sha256=source_hash,
                video_size_bytes=source_size,
                bundle_sha256=bundle.bundle_sha256,
                annotation_sha256=annotation_sha256,
                rows=tuple(rows),
                events=tuple(events),
            )
        )
    if len(set(provided_sources)) != len(provided_sources) or set(provided_sources) != allowed_sources:
        raise ValueError("passed videos must exactly cover the training manifest sources")
    provided_annotation_set = set(provided_annotations)
    remaining_annotations = allowed_annotations - provided_annotation_set
    if (
        len(provided_annotation_set) != len(provided_annotations)
        or len(remaining_annotations) != 1
        or provided_annotation_set | remaining_annotations != allowed_annotations
    ):
        raise ValueError("passed annotations must exactly cover the training manifest labels")
    export_index = next(iter(remaining_annotations))
    if export_index[0] not in _TRAINING_EXPORT_INDEX_FILENAMES:
        raise ValueError("training manifest must retain one recognized v1/v2 export index")
    if not validated_sources or not any(source.rows for source in validated_sources):
        raise ValueError("video embedding extraction requires at least one labeled row")

    resume_artifact: dict[str, object] | None = None
    if resume and resume_checkpoint_path is not None:
        resume_snapshot = _read_byte_snapshot(
            resume_checkpoint_path,
            label="resume checkpoint",
            maximum_bytes=MAX_JSON_SNAPSHOT_BYTES,
        )
        if resume_snapshot.sha256 != expected_resume_checkpoint_sha256:
            raise ValueError("resume checkpoint file SHA-256 mismatch")
        resume_artifact = verify_video_embedding_artifact(_json_object_from_snapshot(resume_snapshot))
        del resume_snapshot

    spec = get_video_backbone_spec(backbone_name)
    if type(clip_frames) is not int or clip_frames < spec.minimum_clip_frames:
        raise ValueError(f"clip_frames must be an integer >= {spec.minimum_clip_frames}")
    annotation_hashes = sorted({source.annotation_sha256 for source in validated_sources})
    source_hashes = sorted({source.video_sha256 for source in validated_sources})
    if resume_artifact is not None:
        _validate_resume_provenance_and_rows(
            artifact=resume_artifact,
            validated_sources=validated_sources,
            manifest_sha256=str(manifest["manifest_sha256"]),
            annotation_hashes=annotation_hashes,
            source_hashes=source_hashes,
            backbone_name=backbone_name,
            backbone_sha256=backbone_sha256,
            backbone_license=spec.license,
            backbone_weights_url=spec.weights_url,
            embedding_dimension=spec.embedding_dimension,
            clip_frames=clip_frames,
        )

    prepared_rows: list[_PreparedRow] = []
    for source in validated_sources:
        fps = _read_video_fps(source.video_path)
        for row, event in zip(source.rows, source.events, strict=True):
            prepared_rows.append(
                _PreparedRow(
                    video_path=source.video_path,
                    source_video_sha256=source.video_sha256,
                    candidate_bundle_sha256=source.bundle_sha256,
                    event=event,
                    event_present=bool(row["event_present"]),
                    sampling_window=resolve_shot_sampling_window(event, fps=fps),
                )
            )

    examples: list[dict[str, object]] = []
    if resume_artifact is not None:
        _validate_resume_sampling_windows(
            artifact=resume_artifact,
            prepared_rows=prepared_rows,
        )
        examples.extend(dict(row) for row in resume_artifact["examples"])

    def current_artifact() -> dict[str, object]:
        return seal_video_embedding_artifact(
            {
                "purpose": "backbone_screening_training_only",
                "producer": "agu",
                "training_manifest_sha256": manifest["manifest_sha256"],
                "training_annotation_sha256": annotation_hashes,
                "source_video_sha256": source_hashes,
                "backbone": backbone_name,
                "backbone_sha256": backbone_sha256,
                "backbone_license": spec.license,
                "backbone_weights_url": spec.weights_url,
                "embedding_dimension": spec.embedding_dimension,
                "clip_frames": clip_frames,
                "sampling_protocol": SHOT_SAMPLING_PROTOCOL,
                "examples": examples,
            }
        )

    if len(examples) == len(prepared_rows):
        _verify_source_videos_unchanged(validated_sources)
        artifact = current_artifact()
        _publish_final_artifact(
            output_path=output_path,
            resume_checkpoint_path=resume_checkpoint_path,
            artifact=artifact,
        )
        return artifact

    device = _resolve_device(device_name)
    backbone, transform = _load_backbone_snapshot(
        backbone_name=backbone_name,
        backbone_snapshot=backbone_snapshot,
        snapshot_parent=snapshot_parent,
        device=device,
    )
    del backbone_snapshot

    row_index = 0
    for source in validated_sources:
        source_rows = prepared_rows[row_index : row_index + len(source.rows)]
        for prepared in source_rows:
            row_index += 1
            if len(examples) >= row_index:
                continue
            embeddings = extract_video_embeddings(
                video_path=prepared.video_path,
                events=[prepared.event],
                backbone=backbone,
                transform=transform,
                device=device,
                clip_frames=clip_frames,
                batch_size=1,
                sampling_windows=[prepared.sampling_window],
            )
            if len(embeddings) != 1:
                raise ValueError("video backbone must return exactly one embedding per row")
            examples.append(
                {
                    "source_video_sha256": prepared.source_video_sha256,
                    "candidate_bundle_sha256": prepared.candidate_bundle_sha256,
                    "event_id": prepared.event.event_id,
                    "event_present": prepared.event_present,
                    "sampling_window": asdict(prepared.sampling_window),
                    "embedding": embeddings[0],
                }
            )
            if resume_checkpoint_path is not None:
                _write_json_atomic(resume_checkpoint_path, current_artifact())
            print(
                json.dumps(
                    {
                        "stage": "embedding_row",
                        "event_id": prepared.event.event_id,
                        "total_examples": len(examples),
                    }
                ),
                flush=True,
            )

    if len(examples) != len(prepared_rows):
        raise ValueError("embedding extraction did not produce every validated row")
    try:
        _verify_source_videos_unchanged(validated_sources)
    except ValueError:
        if resume_checkpoint_path is not None:
            resume_checkpoint_path.unlink(missing_ok=True)
        raise
    artifact = current_artifact()
    _publish_final_artifact(
        output_path=output_path,
        resume_checkpoint_path=resume_checkpoint_path,
        artifact=artifact,
    )
    return artifact


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _read_video_fps(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open source video: {video_path.name}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if fps <= 0:
        raise ValueError(f"source video has invalid fps: {video_path.name}")
    return fps


def _validate_manifest_receipts(
    *,
    expected_training_manifest_sha256: str,
    expected_training_manifest_file_sha256: str,
) -> None:
    _validate_sha256(
        expected_training_manifest_sha256,
        name="expected_training_manifest_sha256",
    )
    _validate_sha256(
        expected_training_manifest_file_sha256,
        name="expected_training_manifest_file_sha256",
    )


def _strict_manifest_assets(
    value: object,
    *,
    label: str,
) -> tuple[tuple[str, str, int], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} assets must be a non-empty list")
    assets: list[tuple[str, str, int]] = []
    casefolded_filenames: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != _MANIFEST_ASSET_FIELDS:
            raise ValueError(f"{label} assets require exact fields")
        filename = item.get("filename")
        size_bytes = item.get("size_bytes")
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise ValueError(f"{label} filename is invalid")
        if type(size_bytes) is not int or size_bytes < 0:
            raise ValueError(f"{label} size must be a non-negative integer")
        sha256 = _validate_sha256(item.get("sha256"), name=f"{label} SHA-256")
        folded_filename = filename.casefold()
        if folded_filename in casefolded_filenames:
            raise ValueError(f"{label} filenames must be unique")
        casefolded_filenames.add(folded_filename)
        assets.append((filename, sha256, size_bytes))
    if len(set(assets)) != len(assets):
        raise ValueError(f"{label} assets must be unique")
    return tuple(assets)


def _read_byte_snapshot(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
) -> _ByteSnapshot:
    try:
        handle = path.open("rb")
    except OSError as error:
        raise ValueError(f"{label} cannot be opened: {path.name}") from error
    with handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file: {path.name}")
        if before.st_size > maximum_bytes:
            raise ValueError(f"{label} exceeds the {maximum_bytes} byte immutable snapshot limit")
        payload = handle.read(maximum_bytes + 1)
        after = os.fstat(handle.fileno())
    if _stat_snapshot(before) != _stat_snapshot(after) or len(payload) != after.st_size or len(payload) > maximum_bytes:
        raise ValueError(f"{label} changed while its immutable snapshot was read")
    return _ByteSnapshot(
        path=path,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _json_object_from_snapshot(snapshot: _ByteSnapshot) -> dict[str, object]:
    try:
        payload = json.loads(snapshot.payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON object: {snapshot.path.name}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {snapshot.path.name}")
    return payload


def _stream_file_sha256(path: Path) -> tuple[str, int]:
    try:
        handle = path.open("rb")
    except OSError as error:
        raise ValueError(f"source video cannot be opened: {path.name}") from error
    with handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"source video must be a regular file: {path.name}")
        digest = hashlib.sha256()
        size_bytes = 0
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size_bytes += len(chunk)
        after = os.fstat(handle.fileno())
    if _stat_snapshot(before) != _stat_snapshot(after) or size_bytes != after.st_size:
        raise ValueError(f"source video changed while hashing: {path.name}")
    return digest.hexdigest(), size_bytes


def _stat_snapshot(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _verify_source_videos_unchanged(
    validated_sources: list[_ValidatedSource],
) -> None:
    for source in validated_sources:
        current_hash, current_size = _stream_file_sha256(source.video_path)
        if current_hash != source.video_sha256 or current_size != source.video_size_bytes:
            raise ValueError("source video changed before final artifact publication")


def _validate_label_rows(
    *,
    labels: dict[str, object],
    bundle: RawOnlyPredictionBundleResponse,
    source_hash: str,
) -> tuple[list[dict[str, object]], list[object]]:
    if labels.get("schema_version") != SHOT_VALIDITY_LABEL_SCHEMA:
        raise ValueError("unsupported shot-validity labels")
    if labels.get("runtime_consumable") is not False:
        raise ValueError("training labels must not be runtime-consumable")
    if labels.get("source_video_sha256") != source_hash:
        raise ValueError("annotation source video hash mismatch")
    if labels.get("candidate_bundle_sha256") != bundle.bundle_sha256:
        raise ValueError("annotation candidate bundle hash mismatch")
    raw_rows = labels.get("examples")
    if not isinstance(raw_rows, list):
        raise ValueError("video shot labels require an examples list")
    events_by_id = {event.event_id: event for event in bundle.events}
    rows: list[dict[str, object]] = []
    events: list[object] = []
    seen_event_ids: set[str] = set()
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict) or set(raw_row) != {
            "event_id",
            "event_present",
        }:
            raise ValueError("video shot label rows require exact event fields")
        event_id = str(raw_row.get("event_id") or "")
        if not event_id or event_id in seen_event_ids:
            raise ValueError("video shot labels contain an empty or duplicate event ID")
        event = events_by_id.get(event_id)
        if event is None or event.event_type != "field_goal_attempt":
            raise ValueError("video label does not name a field-goal candidate")
        if not isinstance(raw_row.get("event_present"), bool):
            raise ValueError("video shot labels must be boolean")
        seen_event_ids.add(event_id)
        rows.append(dict(raw_row))
        events.append(event)
    return rows, events


def _validate_resume_provenance_and_rows(
    *,
    artifact: dict[str, object],
    validated_sources: list[_ValidatedSource],
    manifest_sha256: str,
    annotation_hashes: list[str],
    source_hashes: list[str],
    backbone_name: str,
    backbone_sha256: str,
    backbone_license: str,
    backbone_weights_url: str,
    embedding_dimension: int,
    clip_frames: int,
) -> None:
    if set(artifact) != _EMBEDDING_ARTIFACT_FIELDS:
        raise ValueError("resume checkpoint requires exact top-level fields")
    if (
        artifact.get("purpose") != "backbone_screening_training_only"
        or artifact.get("producer") != "agu"
        or artifact.get("training_manifest_sha256") != manifest_sha256
        or artifact.get("training_annotation_sha256") != annotation_hashes
        or artifact.get("source_video_sha256") != source_hashes
        or artifact.get("backbone") != backbone_name
        or artifact.get("backbone_sha256") != backbone_sha256
        or artifact.get("backbone_license") != backbone_license
        or artifact.get("backbone_weights_url") != backbone_weights_url
        or artifact.get("embedding_dimension") != embedding_dimension
        or artifact.get("clip_frames") != clip_frames
        or artifact.get("sampling_protocol") != SHOT_SAMPLING_PROTOCOL
    ):
        raise ValueError("resume checkpoint provenance does not match extraction")
    expected_rows = [
        (
            source.video_sha256,
            source.bundle_sha256,
            event.event_id,
            bool(row["event_present"]),
        )
        for source in validated_sources
        for row, event in zip(source.rows, source.events, strict=True)
    ]
    checkpoint_rows = artifact.get("examples")
    if not isinstance(checkpoint_rows, list) or len(checkpoint_rows) > len(expected_rows):
        raise ValueError("resume checkpoint row count does not match extraction")
    for index, checkpoint_row in enumerate(checkpoint_rows):
        if not isinstance(checkpoint_row, dict):
            raise ValueError("resume checkpoint contains a non-object row")
        _validate_resume_example_schema(
            checkpoint_row,
            embedding_dimension=embedding_dimension,
        )
        source_sha256, bundle_sha256, event_id, event_present = expected_rows[index]
        expected_identity = (source_sha256, bundle_sha256, event_id)
        checkpoint_identity = (
            checkpoint_row.get("source_video_sha256"),
            checkpoint_row.get("candidate_bundle_sha256"),
            checkpoint_row.get("event_id"),
        )
        if checkpoint_identity != expected_identity:
            raise ValueError("resume checkpoint rows must be an exact extraction prefix")
        if checkpoint_row.get("event_present") is not event_present:
            raise ValueError("resume checkpoint label does not match annotation")


def _validate_resume_example_schema(
    row: dict[str, object],
    *,
    embedding_dimension: int,
) -> None:
    if set(row) != _EMBEDDING_EXAMPLE_FIELDS:
        raise ValueError("resume checkpoint examples require exact fields")
    if not isinstance(row.get("event_present"), bool):
        raise ValueError("resume checkpoint examples require boolean labels")
    sampling_window = row.get("sampling_window")
    if not isinstance(sampling_window, dict) or set(sampling_window) != _SAMPLING_WINDOW_FIELDS:
        raise ValueError("resume checkpoint sampling windows require exact fields")
    start_frame = sampling_window.get("start_frame")
    end_frame = sampling_window.get("end_frame")
    anchor_frame = sampling_window.get("anchor_frame")
    if any(type(value) is not int for value in (start_frame, end_frame, anchor_frame)):
        raise ValueError("resume checkpoint sampling frames must be integers")
    if start_frame < 0 or end_frame < start_frame or not start_frame <= anchor_frame <= end_frame:
        raise ValueError("resume checkpoint sampling window bounds are invalid")
    anchor_source = sampling_window.get("anchor_source")
    if not isinstance(anchor_source, str) or not anchor_source:
        raise ValueError("resume checkpoint sampling anchor source is invalid")
    if sampling_window.get("protocol") != SHOT_SAMPLING_PROTOCOL:
        raise ValueError("resume checkpoint sampling protocol is invalid")
    embedding = row.get("embedding")
    if not isinstance(embedding, list) or len(embedding) != embedding_dimension:
        raise ValueError("resume checkpoint embedding dimension is invalid")
    if any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in embedding):
        raise ValueError("resume checkpoint embeddings must be finite non-boolean numbers")


def _validate_resume_sampling_windows(
    *,
    artifact: dict[str, object],
    prepared_rows: list[_PreparedRow],
) -> None:
    checkpoint_rows = artifact["examples"]
    if not isinstance(checkpoint_rows, list):  # verified before this helper
        raise ValueError("resume checkpoint examples must be a list")
    for index, checkpoint_row in enumerate(checkpoint_rows):
        if not isinstance(checkpoint_row, dict):  # verified before this helper
            raise ValueError("resume checkpoint contains a non-object row")
        expected = prepared_rows[index]
        if checkpoint_row.get("sampling_window") != asdict(expected.sampling_window):
            raise ValueError("resume checkpoint sampling window does not match plan")


def _snapshot_parent(
    *,
    resume_checkpoint_path: Path | None,
    output_path: Path | None,
    backbone_checkpoint: Path,
) -> Path:
    preferred = resume_checkpoint_path or output_path or backbone_checkpoint
    return _nearest_existing_parent(preferred.parent)


def _load_backbone_snapshot(
    *,
    backbone_name: str,
    backbone_snapshot: _ByteSnapshot,
    snapshot_parent: Path,
    device: torch.device,
) -> tuple[object, object]:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=snapshot_parent,
            prefix=".agu-backbone-snapshot.",
            suffix=".pth",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(backbone_snapshot.payload)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary_path.stat().st_size != backbone_snapshot.size_bytes:
            raise ValueError("backbone checkpoint snapshot write was incomplete")
        return load_video_backbone(
            backbone_name,
            temporary_path,
            device=device,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _resolved_disjoint_paths(
    *,
    manifest_path: Path,
    candidate_bundle_paths: list[Path],
    annotation_paths: list[Path],
    video_paths: list[Path],
    backbone_checkpoint: Path,
    checkpoint_path: Path | None = None,
    resume_checkpoint_path: Path | None = None,
    output_path: Path | None = None,
) -> tuple[
    Path,
    list[Path],
    list[Path],
    list[Path],
    Path,
    Path | None,
    Path | None,
]:
    if checkpoint_path is not None and resume_checkpoint_path is not None:
        raise ValueError("legacy checkpoint_path and resume_checkpoint_path cannot both be used")
    staging_path = resume_checkpoint_path or checkpoint_path
    resolved_manifest = manifest_path.resolve()
    resolved_bundles = [path.resolve() for path in candidate_bundle_paths]
    resolved_annotations = [path.resolve() for path in annotation_paths]
    resolved_videos = [path.resolve() for path in video_paths]
    resolved_backbone = backbone_checkpoint.resolve()
    resolved_staging = staging_path.resolve() if staging_path is not None else None
    resolved_output = output_path.resolve() if output_path is not None else None
    resolved_inputs = [
        resolved_manifest,
        *resolved_bundles,
        *resolved_annotations,
        *resolved_videos,
        resolved_backbone,
    ]
    resolved_outputs = [path for path in (resolved_staging, resolved_output) if path is not None]
    for candidate in resolved_outputs:
        if any(_paths_alias_or_overlap(candidate, source) for source in resolved_inputs):
            raise ValueError("embedding output/staging path must not alias or overlap an input path")
    if len(resolved_outputs) == 2 and _paths_alias_or_overlap(
        resolved_outputs[0],
        resolved_outputs[1],
    ):
        raise ValueError("embedding output and staging paths must not alias or overlap")
    return (
        resolved_manifest,
        resolved_bundles,
        resolved_annotations,
        resolved_videos,
        resolved_backbone,
        resolved_staging,
        resolved_output,
    )


def _existing_path_identity(path: Path) -> tuple[int, int] | None:
    try:
        status = path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return None
    return status.st_dev, status.st_ino


def _paths_alias_or_overlap(left: Path, right: Path) -> bool:
    left_parts = tuple(part.casefold() for part in left.parts)
    right_parts = tuple(part.casefold() for part in right.parts)
    if (
        left_parts == right_parts
        or _parts_are_ancestor(left_parts, right_parts)
        or _parts_are_ancestor(right_parts, left_parts)
    ):
        return True
    left_identity = _existing_path_identity(left)
    right_identity = _existing_path_identity(right)
    return left_identity is not None and left_identity == right_identity


def _parts_are_ancestor(
    candidate: tuple[str, ...],
    descendant: tuple[str, ...],
) -> bool:
    return len(candidate) < len(descendant) and descendant[: len(candidate)] == candidate


def _run_resource_preflight(
    *,
    output_paths: list[Path],
    snapshot_parent: Path,
    max_system_memory_percent: float,
    min_available_memory_gib: float,
    min_output_free_gib: float,
) -> None:
    memory = psutil.virtual_memory()
    memory_percent = _finite_float(memory.percent, name="system memory percent")
    available_bytes = int(memory.available)
    if memory_percent > max_system_memory_percent:
        raise RuntimeError(
            f"system memory preflight failed: {memory_percent:.3f}% exceeds {max_system_memory_percent:.3f}%"
        )
    minimum_available_bytes = int(min_available_memory_gib * GIB)
    if available_bytes < minimum_available_bytes:
        raise RuntimeError(
            "available memory preflight failed: "
            f"{available_bytes / GIB:.3f} GiB is below "
            f"{min_available_memory_gib:.3f} GiB"
        )
    minimum_free_bytes = int(min_output_free_gib * GIB)
    output_parents = [
        snapshot_parent,
        *(_nearest_existing_parent(path.parent) for path in output_paths),
    ]
    checked_parents: set[Path] = set()
    for output_parent in output_parents:
        if output_parent in checked_parents:
            continue
        checked_parents.add(output_parent)
        free_bytes = int(shutil.disk_usage(output_parent).free)
        if free_bytes < minimum_free_bytes:
            raise RuntimeError(
                f"output filesystem preflight failed: {free_bytes / GIB:.3f} GiB is below {min_output_free_gib:.3f} GiB"
            )


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise RuntimeError("output filesystem preflight cannot resolve a volume")
        candidate = parent
    if not candidate.is_dir():
        raise RuntimeError("output filesystem preflight parent must be a directory")
    return candidate


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _publish_final_artifact(
    *,
    output_path: Path | None,
    resume_checkpoint_path: Path | None,
    artifact: dict[str, object],
) -> None:
    if output_path is None:
        return
    _write_json_atomic(output_path, artifact)
    if resume_checkpoint_path is not None:
        resume_checkpoint_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    artifact = extract_labeled_video_embeddings(
        manifest_path=args.manifest.resolve(),
        candidate_bundle_paths=[path.resolve() for path in args.candidate_bundle],
        annotation_paths=[path.resolve() for path in args.annotation],
        video_paths=[path.resolve() for path in args.video],
        backbone_name=args.backbone,
        backbone_checkpoint=args.backbone_checkpoint.resolve(),
        expected_backbone_sha256=args.expected_backbone_sha256,
        expected_training_manifest_sha256=args.expected_training_manifest_sha256,
        expected_training_manifest_file_sha256=(args.expected_training_manifest_file_sha256),
        clip_frames=args.clip_frames,
        batch_size=args.batch_size,
        device_name=args.device,
        resume_checkpoint_path=args.resume_checkpoint.resolve(),
        output_path=args.output.resolve(),
        resume=args.resume,
        expected_resume_checkpoint_sha256=(args.expected_resume_checkpoint_sha256),
        max_system_memory_percent=args.max_system_memory_percent,
        min_available_memory_gib=args.min_available_gib,
        min_output_free_gib=args.min_output_free_gib,
    )
    print(
        json.dumps(
            {
                "examples": len(artifact["examples"]),
                "backbone": artifact["backbone"],
                "artifact_sha256": artifact["artifact_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
