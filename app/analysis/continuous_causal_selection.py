"""Deterministically freeze label-hidden windows from one continuous game."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path, PureWindowsPath
from typing import Any

SCHEMA_VERSION = "agu.continuous-causal-review-selection.v1"
PURPOSE = "offline_continuous_game_label_hidden_causal_review"
RANKING_NAMESPACE = "agu.continuous-causal-selection.v1|2026-08-16|8s|8fps"
NEUTRAL_REVIEW_NAMESPACE = "agu.continuous-causal-neutral-review-id.v1"
SOURCE_SELECTION_PROTOCOL = "predeclared_single_continuous_source_geometry_only_v1"
SELECTION_PROVENANCE = "exact_geometry_replay_v1"
GRID_SPACING_SECONDS = 30.0
WINDOW_SECONDS = 8.0
SAMPLE_RATE_HZ = 8.0
SAMPLE_COUNT = 64
TEMPORAL_BUCKETS = ("early", "middle", "late")
TEMPORAL_BUCKET_QUOTAS = {bucket: 8 for bucket in TEMPORAL_BUCKETS}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SAFE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,127}")

MANIFEST_FIELDS = {
    "schema_version",
    "purpose",
    "license",
    "source_urls",
    "selection",
    "runtime_consumable",
    "codex_runtime_answer_used",
    "provenance",
    "videos",
}
MANIFEST_VIDEO_FIELDS = {
    "location",
    "clip_id",
    "path",
    "size_bytes",
    "sha256",
    "fps",
    "frame_count",
    "duration_seconds",
}
SOURCE_FIELDS = {
    "source_id",
    "clip_id",
    "source_video_filename",
    "source_video_sha256",
    "source_video_size_bytes",
    "source_fps",
    "frame_count",
    "duration_seconds",
    "provenance",
}
PROVENANCE_FIELDS = {
    "provider",
    "publisher",
    "publisher_channel_id",
    "source_page_url",
    "original_source_url",
    "original_source_id",
    "title",
    "event_date",
    "license_spdx",
    "attribution",
    "license_review_warning",
    "production_family",
}
POLICY_FIELDS = {
    "ranking_namespace",
    "ranking_direction",
    "candidate_grid_spacing_seconds",
    "window_seconds",
    "sample_rate_hz",
    "sample_count",
    "interval_semantics",
    "temporal_bucket_quotas",
}
AUDIT_FIELDS = {"candidate_count", "candidate_count_by_bucket"}
WINDOW_FIELDS = {
    "review_id",
    "source_id",
    "clip_id",
    "source_video_filename",
    "source_video_sha256",
    "source_fps",
    "frame_count",
    "anchor_frame",
    "center_seconds",
    "start_seconds",
    "end_seconds",
    "temporal_bucket",
    "ranking_sha256",
}
TOP_LEVEL_FIELDS = {
    "schema_version",
    "purpose",
    "runtime_consumable",
    "training_consumable",
    "formal_evaluation_eligible",
    "codex_runtime_answer_used",
    "labels_hidden_from_reviewer",
    "selection_provenance_verification",
    "source_manifest_sha256",
    "source",
    "selection",
    "audit",
    "windows",
    "artifact_sha256",
}


def build_continuous_causal_selection(
    *,
    source_manifest_path: Path,
    source_id: str,
) -> dict[str, Any]:
    """Build and seal the frozen selection using source geometry only."""

    manifest_bytes = source_manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError("continuous source manifest is not valid JSON") from exc
    source = _source_from_manifest(
        manifest,
        source_id=source_id,
        manifest_path=source_manifest_path,
    )
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    artifact = _build_payload(source, source_manifest_sha256=manifest_sha256)
    artifact["artifact_sha256"] = _artifact_sha256(artifact)
    return verify_continuous_causal_selection(artifact)


def resolve_continuous_source_video_path(
    source_manifest_path: Path,
    *,
    source_id: str,
) -> Path:
    """Resolve the selected video early so CLIs can protect it from overwrite."""

    try:
        manifest = json.loads(
            source_manifest_path.read_bytes(),
            object_pairs_hook=_unique_object,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("continuous source manifest is not valid JSON") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("continuous source manifest must be an object")
    videos = manifest.get("videos")
    if not isinstance(videos, list) or len(videos) != 1:
        raise ValueError("continuous source manifest requires exactly one video")
    row = videos[0]
    if not isinstance(row, Mapping) or set(row) != MANIFEST_VIDEO_FIELDS or row.get("location") != source_id:
        raise ValueError("continuous source manifest video fields are not canonical")
    path = _resolve_source_video_path(
        manifest_path=source_manifest_path,
        value=row["path"],
    )
    if not path.is_file():
        raise ValueError("continuous source video must be a regular file")
    return path


def verify_continuous_causal_selection(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Structurally verify the seal and exact geometry replay.

    Evidence consumers must call :func:`verify_continuous_causal_selection_receipt`
    so a self-resealed source cannot substitute for the externally frozen input.
    """

    if not isinstance(payload, Mapping) or set(payload) != TOP_LEVEL_FIELDS:
        raise ValueError("continuous causal selection top-level fields are not canonical")
    artifact = dict(payload)
    if artifact["schema_version"] != SCHEMA_VERSION or artifact["purpose"] != PURPOSE:
        raise ValueError("unsupported continuous causal selection contract")
    for field in (
        "runtime_consumable",
        "training_consumable",
        "formal_evaluation_eligible",
        "codex_runtime_answer_used",
    ):
        if artifact[field] is not False:
            raise ValueError(f"continuous causal selection {field} must be false")
    if artifact["labels_hidden_from_reviewer"] is not True:
        raise ValueError("continuous causal selection must remain label hidden")
    if artifact["selection_provenance_verification"] != SELECTION_PROVENANCE:
        raise ValueError("continuous causal selection provenance claim is invalid")

    claimed_sha = _require_sha(artifact["artifact_sha256"], "artifact SHA-256")
    if _artifact_sha256(artifact) != claimed_sha:
        raise ValueError("continuous causal selection artifact SHA-256 mismatch")

    source_manifest_sha = _require_sha(
        artifact["source_manifest_sha256"],
        "source manifest SHA-256",
    )
    source = _verify_source(artifact["source"])
    _verify_policy(artifact["selection"])
    _verify_audit_shape(artifact["audit"])
    _verify_window_shape(artifact["windows"])
    expected = _build_payload(source, source_manifest_sha256=source_manifest_sha)
    observed = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    if observed != expected:
        raise ValueError("continuous causal selection does not match exact geometry replay")
    return artifact


def verify_continuous_causal_selection_receipt(
    payload: Mapping[str, Any],
    *,
    expected_artifact_sha256: str,
    source_manifest_path: Path,
) -> dict[str, Any]:
    """Verify a frozen receipt against the exact manifest and retained video."""

    artifact = verify_continuous_causal_selection(payload)
    expected_sha = _require_sha(
        expected_artifact_sha256,
        "expected artifact SHA-256",
    )
    if artifact["artifact_sha256"] != expected_sha:
        raise ValueError("expected artifact SHA-256 does not match")
    manifest_bytes = source_manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != artifact["source_manifest_sha256"]:
        raise ValueError("source manifest file SHA-256 does not match selection")
    try:
        manifest = json.loads(manifest_bytes, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError("continuous source manifest is not valid JSON") from exc
    source = _source_from_manifest(
        manifest,
        source_id=artifact["source"]["source_id"],
        manifest_path=source_manifest_path,
    )
    if source != artifact["source"]:
        raise ValueError("source manifest/video do not match frozen selection")
    return artifact


def _source_from_manifest(
    value: Any,
    *,
    source_id: str,
    manifest_path: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != MANIFEST_FIELDS:
        raise ValueError("continuous source manifest fields are not canonical")
    if value["schema_version"] != "agu.vru-basketball-source-manifest.v1":
        raise ValueError("unsupported continuous source manifest")
    if value["runtime_consumable"] is not False:
        raise ValueError("continuous source manifest must remain offline-only")
    if value["codex_runtime_answer_used"] is not False:
        raise ValueError("continuous source manifest cannot be a runtime answer channel")
    for field in ("purpose", "license"):
        if not isinstance(value[field], str) or not value[field]:
            raise ValueError(f"continuous source manifest {field} is invalid")
    if value["selection"] != SOURCE_SELECTION_PROTOCOL:
        raise ValueError("continuous source manifest selection protocol is invalid")
    urls = value["source_urls"]
    if not isinstance(urls, list) or not urls or any(not isinstance(url, str) or not url for url in urls):
        raise ValueError("continuous source manifest URLs are invalid")
    if not isinstance(source_id, str) or SAFE_ID_PATTERN.fullmatch(source_id) is None:
        raise ValueError("continuous source ID must be a path-safe scalar")
    videos = value["videos"]
    if not isinstance(videos, list) or len(videos) != 1:
        raise ValueError("continuous source manifest requires exactly one video")
    if not isinstance(videos[0], Mapping) or set(videos[0]) != MANIFEST_VIDEO_FIELDS:
        raise ValueError("continuous source manifest video fields are not canonical")
    matches = [row for row in videos if isinstance(row, Mapping) and row.get("location") == source_id]
    if len(matches) != 1:
        raise ValueError("continuous source ID must resolve to exactly one video")
    row = matches[0]
    video_path = _resolve_source_video_path(
        manifest_path=manifest_path,
        value=row["path"],
    )
    expected_size = _positive_int(row["size_bytes"], "source video size")
    if video_path.stat().st_size != expected_size:
        raise ValueError("continuous source video size does not match manifest")
    expected_video_sha = _require_sha(row["sha256"], "source video SHA-256")
    if _file_sha256(video_path) != expected_video_sha:
        raise ValueError("continuous source video SHA-256 does not match manifest")
    source = {
        "source_id": _safe_id(row["location"], "source ID"),
        "clip_id": _safe_id(row["clip_id"], "clip ID"),
        "source_video_filename": _safe_basename(video_path.name),
        "source_video_sha256": expected_video_sha,
        "source_video_size_bytes": expected_size,
        "source_fps": _positive_number(row["fps"], "source FPS"),
        "frame_count": _positive_int(row["frame_count"], "source frame count"),
        "duration_seconds": _positive_number(row["duration_seconds"], "source duration"),
        "provenance": _verify_provenance(value["provenance"]),
    }
    _verify_probed_geometry(source, _probe_video_geometry(video_path))
    return _verify_source(source)


def _build_payload(
    source: Mapping[str, Any],
    *,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    candidates = _candidate_windows(source)
    counts = Counter(row["temporal_bucket"] for row in candidates)
    selected: list[dict[str, Any]] = []
    for bucket in TEMPORAL_BUCKETS:
        ranked = sorted(
            (row for row in candidates if row["temporal_bucket"] == bucket),
            key=lambda row: row["ranking_sha256"],
        )
        if len(ranked) < TEMPORAL_BUCKET_QUOTAS[bucket]:
            raise ValueError(f"source lacks enough {bucket} candidates")
        selected.extend(ranked[: TEMPORAL_BUCKET_QUOTAS[bucket]])
    selected.sort(key=lambda row: row["center_seconds"])
    for index, row in enumerate(selected, start=1):
        row["review_id"] = neutral_continuous_review_id(
            source["source_video_sha256"],
            index,
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "purpose": PURPOSE,
        "runtime_consumable": False,
        "training_consumable": False,
        "formal_evaluation_eligible": False,
        "codex_runtime_answer_used": False,
        "labels_hidden_from_reviewer": True,
        "selection_provenance_verification": SELECTION_PROVENANCE,
        "source_manifest_sha256": source_manifest_sha256,
        "source": dict(source),
        "selection": _policy(),
        "audit": {
            "candidate_count": len(candidates),
            "candidate_count_by_bucket": {bucket: counts[bucket] for bucket in TEMPORAL_BUCKETS},
        },
        "windows": selected,
    }


def _candidate_windows(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    fps = float(source["source_fps"])
    duration = float(source["duration_seconds"])
    frame_count = int(source["frame_count"])
    candidates: list[dict[str, Any]] = []
    nominal_center = GRID_SPACING_SECONDS
    while nominal_center + WINDOW_SECONDS / 2.0 <= duration:
        anchor = round(nominal_center * fps)
        center = round(anchor / fps, 6)
        start = round(center - WINDOW_SECONDS / 2.0, 6)
        end = round(center + WINDOW_SECONDS / 2.0, 6)
        indexes = [round((start + index / SAMPLE_RATE_HZ) * fps) for index in range(SAMPLE_COUNT)]
        if (
            start >= 0.0
            and indexes == sorted(set(indexes))
            and indexes[0] >= 0
            and indexes[-1] < frame_count
            and indexes[SAMPLE_COUNT // 2] == anchor
            and indexes[-1] < round(end * fps)
        ):
            candidates.append(
                {
                    "review_id": "",
                    "source_id": source["source_id"],
                    "clip_id": source["clip_id"],
                    "source_video_filename": source["source_video_filename"],
                    "source_video_sha256": source["source_video_sha256"],
                    "source_fps": source["source_fps"],
                    "frame_count": source["frame_count"],
                    "anchor_frame": anchor,
                    "center_seconds": center,
                    "start_seconds": start,
                    "end_seconds": end,
                    "temporal_bucket": _temporal_bucket(center, duration),
                    "ranking_sha256": _ranking_sha256(source["source_video_sha256"], anchor),
                }
            )
        nominal_center += GRID_SPACING_SECONDS
    return candidates


def _policy() -> dict[str, Any]:
    return {
        "ranking_namespace": RANKING_NAMESPACE,
        "ranking_direction": "ascending_sha256",
        "candidate_grid_spacing_seconds": GRID_SPACING_SECONDS,
        "window_seconds": WINDOW_SECONDS,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "sample_count": SAMPLE_COUNT,
        "interval_semantics": "half_open",
        "temporal_bucket_quotas": dict(TEMPORAL_BUCKET_QUOTAS),
    }


def _verify_policy(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != POLICY_FIELDS or dict(value) != _policy():
        raise ValueError("continuous causal selection policy is not canonical")


def _verify_source(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != SOURCE_FIELDS:
        raise ValueError("continuous causal selection source fields are not canonical")
    source = {
        "source_id": _safe_id(value["source_id"], "source ID"),
        "clip_id": _safe_id(value["clip_id"], "clip ID"),
        "source_video_filename": _safe_basename(value["source_video_filename"]),
        "source_video_sha256": _require_sha(value["source_video_sha256"], "source video SHA-256"),
        "source_video_size_bytes": _positive_int(value["source_video_size_bytes"], "source video size"),
        "source_fps": _positive_number(value["source_fps"], "source FPS"),
        "frame_count": _positive_int(value["frame_count"], "source frame count"),
        "duration_seconds": _positive_number(value["duration_seconds"], "source duration"),
        "provenance": _verify_provenance(value["provenance"]),
    }
    if abs(source["frame_count"] - round(source["duration_seconds"] * source["source_fps"])) > 1:
        raise ValueError("continuous source frame count and duration disagree")
    return source


def _verify_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != PROVENANCE_FIELDS:
        raise ValueError("continuous source provenance fields are not canonical")
    provenance: dict[str, Any] = {}
    for field in PROVENANCE_FIELDS - {"license_review_warning"}:
        item = value[field]
        if not isinstance(item, str) or not item or any(ord(character) < 32 for character in item):
            raise ValueError(f"continuous source provenance {field} is invalid")
        provenance[field] = item
    if value["license_review_warning"] is not True:
        raise ValueError("continuous source license review warning must remain explicit")
    provenance["license_review_warning"] = True
    return provenance


def _verify_audit_shape(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != AUDIT_FIELDS:
        raise ValueError("continuous causal selection audit fields are not canonical")
    _positive_int(value["candidate_count"], "candidate count")
    counts = value["candidate_count_by_bucket"]
    if not isinstance(counts, Mapping) or set(counts) != set(TEMPORAL_BUCKETS):
        raise ValueError("continuous causal candidate bucket counts are not canonical")
    for bucket in TEMPORAL_BUCKETS:
        _positive_int(counts[bucket], f"{bucket} candidate count")


def _verify_window_shape(value: Any) -> None:
    if not isinstance(value, list) or len(value) != 24:
        raise ValueError("continuous causal selection requires exactly 24 windows")
    for row in value:
        if not isinstance(row, Mapping) or set(row) != WINDOW_FIELDS:
            raise ValueError("continuous causal selection window fields are not canonical")


def _artifact_sha256(artifact: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    return hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _ranking_sha256(source_sha256: str, anchor_frame: int) -> str:
    return hashlib.sha256(f"{RANKING_NAMESPACE}\0{source_sha256}\0{anchor_frame}".encode("utf-8")).hexdigest()


def _temporal_bucket(center_seconds: float, duration_seconds: float) -> str:
    fraction = center_seconds / duration_seconds
    if fraction < 1.0 / 3.0:
        return "early"
    if fraction < 2.0 / 3.0:
        return "middle"
    return "late"


def _safe_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or SAFE_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a path-safe scalar")
    return value


def _safe_video_path(value: Any) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("source video path must be a safe scalar")
    path = Path(value)
    if path.is_absolute() or PureWindowsPath(value).drive or any(part in {".", ".."} for part in path.parts):
        raise ValueError("source video path must be relative to the manifest root")
    _safe_basename(path.name)
    return path


def _resolve_source_video_path(*, manifest_path: Path, value: Any) -> Path:
    root = manifest_path.parent.resolve(strict=True)
    resolved = (root / _safe_video_path(value)).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("source video path must remain inside the manifest root") from exc
    if not resolved.is_file():
        raise ValueError("continuous source video must be a regular file")
    return resolved


def neutral_continuous_review_id(source_video_sha256: str, ordinal: int) -> str:
    """Return an opaque reviewer ID that cannot copy manifest identifiers."""

    source_sha = _require_sha(source_video_sha256, "source video SHA-256")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal <= 0:
        raise ValueError("review ordinal must be a positive integer")
    token = hashlib.sha256(f"{NEUTRAL_REVIEW_NAMESPACE}\0{source_sha}".encode("utf-8")).hexdigest()[:24]
    return f"closure-{token}-{ordinal:04d}"


def _safe_basename(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or Path(value).name != value
    ):
        raise ValueError("source video filename must be a safe basename")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_video_geometry(path: Path) -> dict[str, float | int]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,nb_read_packets,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("continuous source video ffprobe failed") from exc
    if completed.returncode != 0:
        raise ValueError("continuous source video ffprobe failed")
    try:
        payload = json.loads(completed.stdout, object_pairs_hook=_unique_object)
        streams = payload["streams"]
        stream = streams[0]
        raw_rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
        fps = float(Fraction(raw_rate))
        frame_count = int(stream["nb_read_packets"])
        duration = float(stream.get("duration") or payload["format"]["duration"])
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("continuous source video ffprobe geometry is invalid") from exc
    if not isinstance(streams, list) or len(streams) != 1:
        raise ValueError("continuous source video requires exactly one probed video stream")
    return {
        "fps": _positive_number(fps, "probed source FPS"),
        "frame_count": _positive_int(frame_count, "probed source frame count"),
        "duration_seconds": _positive_number(duration, "probed source duration"),
    }


def _verify_probed_geometry(
    source: Mapping[str, Any],
    probed: Mapping[str, Any],
) -> None:
    if set(probed) != {"fps", "frame_count", "duration_seconds"}:
        raise ValueError("continuous source video probe fields are not canonical")
    probed_fps = float(_positive_number(probed["fps"], "probed source FPS"))
    manifest_fps = float(source["source_fps"])
    if not math.isclose(probed_fps, manifest_fps, rel_tol=1e-6, abs_tol=1e-3):
        raise ValueError("continuous source video probe FPS does not match manifest")
    probed_frames = _positive_int(probed["frame_count"], "probed source frame count")
    if abs(probed_frames - int(source["frame_count"])) > 1:
        raise ValueError("continuous source video probe frame count does not match manifest")
    probed_duration = float(_positive_number(probed["duration_seconds"], "probed source duration"))
    duration_tolerance = max(0.05, 2.0 / probed_fps)
    if abs(probed_duration - float(source["duration_seconds"])) > duration_tolerance:
        raise ValueError("continuous source video probe duration does not match manifest")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key is forbidden: {key}")
        value[key] = child
    return value


def _require_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _positive_number(value: Any, field: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite positive number")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{field} must be a finite positive number")
    return value
