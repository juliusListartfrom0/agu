"""Label-hidden multi-frame MUVS source contracts for temporal screening.

This module deliberately stops at offline source materialisation and review.  It
does not expose a runtime prediction path and does not read any AGU evaluation
or blind-game answer.  A dense sequence is a new source coordinate with a
fixed, label-independent set of temporal offsets; the old two-frame plans are
accepted only as exclusion manifests.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from app.analysis.muvs_event_state import (
    _balanced_indices,
    _basketball_event_files,
    _canonical_sha256,
    _file_sha256,
    _read_annotations,
    verify_muvs_event_state_plan,
)

MUVS_DENSE_SEQUENCE_PLAN_SCHEMA = "agu.muvs-dense-sequence-plan.v1"
MUVS_DENSE_SEQUENCE_FRAMES_SCHEMA = "agu.muvs-dense-sequence-frames.v1"
MUVS_DENSE_SEQUENCE_REVIEW_SCHEMA = "agu.muvs-dense-sequence-review.v1"
MUVS_DENSE_SEQUENCE_STATES = {
    "live_play",
    "free_throw_setup",
    "dead_ball_timeout",
    "uncertain",
}
MUVS_RECORD_ID = 20_708_683
MUVS_RECORD_URL = f"https://zenodo.org/records/{MUVS_RECORD_ID}"
MUVS_VIDEO_BASE_URL = (
    "https://storage.googleapis.com/dataset-ugv-sports/public/RAW_DATA"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAMPLE_ID = re.compile(r"muvs-dense-(\d{4})")
_REVIEWER_FIELDS = ["sample_id", "frames"]
_DEFAULT_FRAME_OFFSETS = (0.25, 0.75, 1.25, 1.75, 2.25)


def build_muvs_dense_sequence_plan(
    data_root: Path,
    *,
    samples_per_event: int,
    expected_basketball_events: int = 12,
    frame_offsets_sec: Sequence[float] = _DEFAULT_FRAME_OFFSETS,
    included_event_ids: Sequence[str] | None = None,
    excluded_plans: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, label-hidden sequence plan.

    ``frame_offsets_sec`` is relative to the selected three-second MUVS
    fragment after the camera offset.  It is fixed before any source frame is
    reviewed, and at least three points are required to prevent silently
    falling back to the already-rejected two-frame protocol.
    """

    data_root = data_root.resolve()
    offsets = _validate_offsets(frame_offsets_sec)
    if samples_per_event < 1 or expected_basketball_events < 1:
        raise ValueError("MUVS dense sampling limits must be positive")

    excluded_coordinates: set[tuple[str, int, int, str]] = set()
    for payload in excluded_plans or ():
        excluded_coordinates.update(_plan_coordinates(payload))

    event_files = _basketball_event_files(data_root)
    if included_event_ids is not None:
        requested = {str(event_id) for event_id in included_event_ids}
        if not requested or len(requested) != len(included_event_ids):
            raise ValueError("MUVS requested event ids must be unique")
        available = {event_id for event_id, *_ in event_files}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"MUVS requested event is absent: {', '.join(missing)}")
        event_files = [row for row in event_files if row[0] in requested]
    if len(event_files) != expected_basketball_events:
        raise ValueError(
            "MUVS basketball event count mismatch: "
            f"expected {expected_basketball_events}, got {len(event_files)}"
        )

    samples: list[dict[str, Any]] = []
    for event_id, split, annotation_path, fragments_path in event_files:
        annotations = _read_annotations(
            annotation_path,
            expected_event=event_id,
            expected_split=split,
        )
        eligible: list[dict[str, Any]] = []
        for row in annotations:
            camera_id = str(row["camera_selected"])
            coordinate = (
                event_id,
                int(row["period"]),
                int(row["fragment_number"]),
                camera_id,
            )
            if coordinate in excluded_coordinates:
                continue
            base_time = (int(row["fragment_number"]) - 1) * 3.0 + float(
                row["offsets"][camera_id]
            )
            frame_times = [round(base_time + offset, 6) for offset in offsets]
            if frame_times[0] < 0.0 or not all(
                math.isfinite(value) for value in frame_times
            ):
                continue
            eligible.append({**row, "frame_times_sec": frame_times})
        if len(eligible) < samples_per_event:
            raise ValueError(
                f"MUVS event {event_id} has fewer unused dense rows than requested"
            )
        selected = [
            eligible[index]
            for index in _balanced_indices(len(eligible), samples_per_event)
        ]
        annotation_sha256 = _file_sha256(annotation_path)
        fragments_sha256 = _file_sha256(fragments_path)
        for row in selected:
            camera_id = str(row["camera_selected"])
            samples.append(
                {
                    "sample_id": f"muvs-dense-{len(samples) + 1:04d}",
                    "split": split,
                    "event_id": event_id,
                    "period": int(row["period"]),
                    "fragment_number": int(row["fragment_number"]),
                    "camera_id": camera_id,
                    "camera_offset_sec": float(row["offsets"][camera_id]),
                    "frame_times_sec": row["frame_times_sec"],
                    "video_url": (
                        f"{MUVS_VIDEO_BASE_URL}/basketball/{event_id}/"
                        f"{camera_id}/VIDEOS/P{int(row['period'])}/video.mp4"
                    ),
                    "annotation_sha256": annotation_sha256,
                    "fragments_sha256": fragments_sha256,
                }
            )

    artifact: dict[str, Any] = {
        "schema_version": MUVS_DENSE_SEQUENCE_PLAN_SCHEMA,
        "purpose": "codex_offline_muvs_dense_sequence_annotation",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "source": {
            "record_id": MUVS_RECORD_ID,
            "record_url": MUVS_RECORD_URL,
            "record_license": "CC-BY-4.0",
            "package_readme_license": "TO BE ADDED",
            "license_basis": "Zenodo API record metadata",
            "media_redistribution": "not_performed",
        },
        "reviewer_visible_fields": list(_REVIEWER_FIELDS),
        "event_count": len(event_files),
        "samples_per_event": samples_per_event,
        "sample_count": len(samples),
        "sequence_length": len(offsets),
        "frame_offsets_sec": list(offsets),
        "samples": samples,
    }
    return seal_muvs_dense_sequence_plan(artifact)


def seal_muvs_dense_sequence_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    _validate_plan(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_muvs_dense_sequence_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_plan(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("MUVS dense sequence plan hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def seal_muvs_dense_sequence_frames(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    verified_plan = verify_muvs_dense_sequence_plan(plan)
    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["schema_version"] = MUVS_DENSE_SEQUENCE_FRAMES_SCHEMA
    artifact["purpose"] = "codex_offline_muvs_dense_sequence_annotation"
    artifact["runtime_consumable"] = False
    artifact["codex_runtime_answer_used"] = False
    artifact["plan_sha256"] = verified_plan["artifact_sha256"]
    files = artifact.get("files")
    artifact["frame_count"] = (
        sum(len(row.get("frames", [])) for row in files)
        if isinstance(files, list)
        else 0
    )
    _validate_frames(artifact, plan=verified_plan)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_muvs_dense_sequence_frames(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    verified_plan = verify_muvs_dense_sequence_plan(plan)
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_frames(artifact, plan=verified_plan)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("MUVS dense sequence frames hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def seal_muvs_dense_sequence_review(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    frames: Mapping[str, Any],
) -> dict[str, Any]:
    verified_plan = verify_muvs_dense_sequence_plan(plan)
    verified_frames = verify_muvs_dense_sequence_frames(frames, plan=verified_plan)
    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["schema_version"] = MUVS_DENSE_SEQUENCE_REVIEW_SCHEMA
    artifact["purpose"] = "codex_offline_muvs_dense_sequence_annotation"
    artifact["runtime_consumable"] = False
    artifact["codex_runtime_answer_used"] = False
    artifact["plan_sha256"] = verified_plan["artifact_sha256"]
    artifact["frames_sha256"] = verified_frames["artifact_sha256"]
    _validate_review(artifact, plan=verified_plan, frames=verified_frames)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_muvs_dense_sequence_review(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    frames: Mapping[str, Any],
) -> dict[str, Any]:
    verified_plan = verify_muvs_dense_sequence_plan(plan)
    verified_frames = verify_muvs_dense_sequence_frames(frames, plan=verified_plan)
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_review(artifact, plan=verified_plan, frames=verified_frames)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("MUVS dense sequence review hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _validate_offsets(values: Sequence[float]) -> list[float]:
    offsets = [float(value) for value in values]
    if len(offsets) < 3:
        raise ValueError("MUVS dense sequence requires three frame offsets")
    if any(not math.isfinite(value) or value < 0.0 for value in offsets):
        raise ValueError("MUVS dense frame offsets must be finite and non-negative")
    if any(right <= left for left, right in zip(offsets, offsets[1:])):
        raise ValueError("MUVS dense frame offsets must be strictly increasing")
    if offsets[-1] >= 3.0:
        raise ValueError("MUVS dense frame offsets must fit a three-second fragment")
    return [round(value, 6) for value in offsets]


def _plan_coordinates(
    payload: Mapping[str, Any],
) -> set[tuple[str, int, int, str]]:
    schema = str(payload.get("schema_version") or "")
    if schema == MUVS_DENSE_SEQUENCE_PLAN_SCHEMA:
        verified = verify_muvs_dense_sequence_plan(payload)
    elif schema == "agu.muvs-event-state-plan.v1":
        verified = verify_muvs_event_state_plan(payload)
    else:
        raise ValueError("unsupported MUVS exclusion plan schema")
    coordinates = set()
    for row in verified["samples"]:
        coordinates.add(
            (
                str(row["event_id"]),
                int(row["period"]),
                int(row["fragment_number"]),
                str(row["camera_id"]),
            )
        )
    return coordinates


def _validate_plan(artifact: Mapping[str, Any]) -> None:
    source = artifact.get("source")
    samples = artifact.get("samples")
    offsets = artifact.get("frame_offsets_sec")
    sequence_length = int(artifact.get("sequence_length", 0))
    if (
        artifact.get("schema_version") != MUVS_DENSE_SEQUENCE_PLAN_SCHEMA
        or artifact.get("purpose") != "codex_offline_muvs_dense_sequence_annotation"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("reviewer_visible_fields") != _REVIEWER_FIELDS
        or not isinstance(source, Mapping)
        or source.get("record_id") != MUVS_RECORD_ID
        or source.get("record_url") != MUVS_RECORD_URL
        or source.get("record_license") != "CC-BY-4.0"
        or source.get("package_readme_license") != "TO BE ADDED"
        or source.get("media_redistribution") != "not_performed"
        or not isinstance(offsets, list)
        or _validate_offsets(offsets) != offsets
        or sequence_length != len(offsets)
        or not isinstance(samples, list)
        or not samples
        or int(artifact.get("event_count", 0)) < 1
        or int(artifact.get("samples_per_event", 0)) < 1
        or len(samples) != int(artifact["event_count"]) * int(artifact["samples_per_event"])
        or artifact.get("sample_count") != len(samples)
    ):
        raise ValueError("MUVS dense sequence plan policy is invalid")
    events: set[str] = set()
    coordinates: set[tuple[str, int, int, str]] = set()
    for index, row in enumerate(samples, start=1):
        times = row.get("frame_times_sec")
        sample_id = str(row.get("sample_id") or "")
        coordinate = (
            str(row.get("event_id") or ""),
            int(row.get("period", 0)),
            int(row.get("fragment_number", 0)),
            str(row.get("camera_id") or ""),
        )
        if (
            set(row)
            != {
                "sample_id",
                "split",
                "event_id",
                "period",
                "fragment_number",
                "camera_id",
                "camera_offset_sec",
                "frame_times_sec",
                "video_url",
                "annotation_sha256",
                "fragments_sha256",
            }
            or _SAMPLE_ID.fullmatch(sample_id) is None
            or sample_id != f"muvs-dense-{index:04d}"
            or row.get("split") not in {"train", "test"}
            or not coordinate[0]
            or coordinate[1] < 1
            or coordinate[2] < 1
            or re.fullmatch(r"cam_\d+", coordinate[3]) is None
            or not math.isfinite(float(row.get("camera_offset_sec", 0.0)))
            or not isinstance(times, list)
            or len(times) != sequence_length
            or any(not math.isfinite(float(value)) or float(value) < 0 for value in times)
            or any(float(right) <= float(left) for left, right in zip(times, times[1:]))
            or not str(row.get("video_url") or "").startswith(
                f"{MUVS_VIDEO_BASE_URL}/basketball/"
            )
            or coordinate in coordinates
        ):
            raise ValueError("invalid MUVS dense sequence sample")
        _require_sha256(row.get("annotation_sha256"))
        _require_sha256(row.get("fragments_sha256"))
        coordinates.add(coordinate)
        events.add(coordinate[0])
    if len(events) != int(artifact["event_count"]):
        raise ValueError("MUVS dense sequence event coverage is invalid")


def _validate_frames(artifact: Mapping[str, Any], *, plan: Mapping[str, Any]) -> None:
    files = artifact.get("files")
    planned_ids = [row["sample_id"] for row in plan["samples"]]
    sequence_length = int(plan["sequence_length"])
    if (
        artifact.get("schema_version") != MUVS_DENSE_SEQUENCE_FRAMES_SCHEMA
        or artifact.get("purpose") != "codex_offline_muvs_dense_sequence_annotation"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("plan_sha256") != plan["artifact_sha256"]
        or not isinstance(files, list)
        or [str(row.get("sample_id") or "") for row in files] != planned_ids
        or artifact.get("frame_count") != len(files) * sequence_length
    ):
        raise ValueError("MUVS dense sequence frames policy is invalid")
    for row in files:
        if set(row) != {"sample_id", "frames"} or not isinstance(row["frames"], list):
            raise ValueError("invalid MUVS dense sequence frame row")
        if len(row["frames"]) != sequence_length:
            raise ValueError("invalid MUVS dense sequence frame row")
        for frame in row["frames"]:
            path = Path(str(frame.get("path") or ""))
            if (
                set(frame) != {"path", "sha256", "size_bytes", "width", "height"}
                or path.is_absolute()
                or ".." in path.parts
                or path.suffix.lower() != ".png"
                or int(frame.get("size_bytes", 0)) < 1
                or int(frame.get("width", 0)) < 1
                or int(frame.get("height", 0)) < 1
            ):
                raise ValueError("invalid MUVS dense sequence frame row")
            _require_sha256(frame.get("sha256"))


def _validate_review(
    artifact: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    frames: Mapping[str, Any],
) -> None:
    decisions = artifact.get("decisions")
    planned_ids = [row["sample_id"] for row in plan["samples"]]
    if (
        artifact.get("schema_version") != MUVS_DENSE_SEQUENCE_REVIEW_SCHEMA
        or artifact.get("purpose") != "codex_offline_muvs_dense_sequence_annotation"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("plan_sha256") != plan["artifact_sha256"]
        or artifact.get("frames_sha256") != frames["artifact_sha256"]
        or artifact.get("reviewer") != "codex_offline_source_annotation"
        or not isinstance(decisions, list)
        or [str(row.get("sample_id") or "") for row in decisions] != planned_ids
    ):
        raise ValueError("MUVS dense sequence review policy is invalid")
    for row in decisions:
        if (
            set(row) != {"sample_id", "state", "notes"}
            or row.get("state") not in MUVS_DENSE_SEQUENCE_STATES
            or not isinstance(row.get("notes"), str)
            or not row["notes"].strip()
        ):
            raise ValueError("invalid MUVS dense sequence state decision")


def _require_sha256(value: object) -> str:
    text = str(value or "")
    if _SHA256.fullmatch(text) is None:
        raise ValueError("invalid MUVS dense sequence SHA-256")
    return text
