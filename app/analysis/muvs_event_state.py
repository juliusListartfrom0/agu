"""License-bound, source-only MUVS basketball event-state review contracts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

MUVS_EVENT_STATE_PLAN_SCHEMA = "agu.muvs-event-state-plan.v1"
MUVS_EVENT_STATE_FRAMES_SCHEMA = "agu.muvs-event-state-frames.v1"
MUVS_EVENT_STATE_REVIEW_SCHEMA = "agu.muvs-event-state-review.v1"
MUVS_EVENT_STATES = {
    "live_play",
    "free_throw_setup",
    "dead_ball_timeout",
    "uncertain",
}
MUVS_RECORD_ID = 20_708_683
MUVS_RECORD_URL = f"https://zenodo.org/records/{MUVS_RECORD_ID}"
MUVS_VIDEO_BASE_URL = "https://storage.googleapis.com/dataset-ugv-sports/public/RAW_DATA"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAMPLE_ID = re.compile(r"muvs-state-(\d{4})")
_REVIEWER_FIELDS = ["sample_id", "frame_before", "frame_after"]


def build_muvs_event_state_plan(
    data_root: Path,
    *,
    samples_per_event: int,
    expected_basketball_events: int = 12,
    included_event_ids: Sequence[str] | None = None,
    excluded_plans: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a deterministic label-hidden plan from official MUVS CSV files."""

    data_root = data_root.resolve()
    if samples_per_event < 1 or expected_basketball_events < 1:
        raise ValueError("MUVS sampling limits must be positive")
    excluded_coordinates: set[tuple[str, int, int, str]] = set()
    for payload in excluded_plans or ():
        excluded = verify_muvs_event_state_plan(payload)
        excluded_coordinates.update(
            _sample_source_coordinate(row) for row in excluded["samples"]
        )
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
            f"MUVS basketball event count mismatch: expected {expected_basketball_events}, got {len(event_files)}"
        )

    samples: list[dict[str, Any]] = []
    for event_id, split, annotation_path, fragments_path in event_files:
        annotations = _read_annotations(
            annotation_path,
            expected_event=event_id,
            expected_split=split,
        )
        annotations = [
            row
            for row in annotations
            if (
                event_id,
                int(row["period"]),
                int(row["fragment_number"]),
                str(row["camera_selected"]),
            )
            not in excluded_coordinates
        ]
        if len(annotations) < samples_per_event:
            raise ValueError(
                f"MUVS event {event_id} has fewer unused rows than requested samples"
            )
        selected = [
            annotations[index]
            for index in _balanced_indices(
                len(annotations),
                samples_per_event,
            )
        ]
        annotation_sha256 = _file_sha256(annotation_path)
        fragments_sha256 = _file_sha256(fragments_path)
        for row in selected:
            camera_id = str(row["camera_selected"])
            offsets = row["offsets"]
            camera_offset = float(offsets[camera_id])
            fragment_number = int(row["fragment_number"])
            period = int(row["period"])
            base_time = (fragment_number - 1) * 3.0 + camera_offset
            frame_times = [
                round(base_time + 1.0, 6),
                round(base_time + 2.0, 6),
            ]
            if frame_times[0] < 0:
                raise ValueError("MUVS selected frame time precedes period video")
            sample_id = f"muvs-state-{len(samples) + 1:04d}"
            video_url = f"{MUVS_VIDEO_BASE_URL}/basketball/{event_id}/{camera_id}/VIDEOS/P{period}/video.mp4"
            samples.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "event_id": event_id,
                    "period": period,
                    "fragment_number": fragment_number,
                    "camera_id": camera_id,
                    "camera_offset_sec": camera_offset,
                    "frame_times_sec": frame_times,
                    "video_url": video_url,
                    "annotation_sha256": annotation_sha256,
                    "fragments_sha256": fragments_sha256,
                }
            )

    artifact: dict[str, Any] = {
        "schema_version": MUVS_EVENT_STATE_PLAN_SCHEMA,
        "purpose": "codex_offline_muvs_source_event_state_annotation",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "source": {
            "record_id": MUVS_RECORD_ID,
            "record_url": MUVS_RECORD_URL,
            "record_license": "CC-BY-4.0",
            "package_readme_license": "TO BE ADDED",
            "license_basis": "Zenodo record metadata",
            "media_redistribution": "not_performed",
        },
        "reviewer_visible_fields": list(_REVIEWER_FIELDS),
        "event_count": len(event_files),
        "samples_per_event": samples_per_event,
        "sample_count": len(samples),
        "samples": samples,
    }
    return seal_muvs_event_state_plan(artifact)


def _sample_source_coordinate(
    row: Mapping[str, Any],
) -> tuple[str, int, int, str]:
    return (
        str(row["event_id"]),
        int(row["period"]),
        int(row["fragment_number"]),
        str(row["camera_id"]),
    )


def seal_muvs_event_state_plan(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and hash a label-hidden MUVS source sampling plan."""

    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    _validate_plan(artifact)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_muvs_event_state_plan(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_plan(artifact)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("MUVS event-state plan hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def seal_muvs_event_state_review(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    frames: Mapping[str, Any],
) -> dict[str, Any]:
    verified_plan = verify_muvs_event_state_plan(plan)
    verified_frames = verify_muvs_event_state_frames(
        frames,
        plan=verified_plan,
    )
    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["schema_version"] = MUVS_EVENT_STATE_REVIEW_SCHEMA
    artifact["purpose"] = "codex_offline_muvs_source_event_state_annotation"
    artifact["runtime_consumable"] = False
    artifact["codex_runtime_answer_used"] = False
    artifact["plan_sha256"] = verified_plan["artifact_sha256"]
    artifact["frames_sha256"] = verified_frames["artifact_sha256"]
    _validate_review(
        artifact,
        plan=verified_plan,
        frames=verified_frames,
    )
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_muvs_event_state_review(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    frames: Mapping[str, Any],
) -> dict[str, Any]:
    verified_plan = verify_muvs_event_state_plan(plan)
    verified_frames = verify_muvs_event_state_frames(
        frames,
        plan=verified_plan,
    )
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_review(
        artifact,
        plan=verified_plan,
        frames=verified_frames,
    )
    if claimed != _canonical_sha256(artifact):
        raise ValueError("MUVS event-state review hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def seal_muvs_event_state_frames(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    verified_plan = verify_muvs_event_state_plan(plan)
    artifact = dict(payload)
    artifact.pop("artifact_sha256", None)
    artifact["schema_version"] = MUVS_EVENT_STATE_FRAMES_SCHEMA
    artifact["purpose"] = "codex_offline_muvs_source_event_state_annotation"
    artifact["runtime_consumable"] = False
    artifact["codex_runtime_answer_used"] = False
    artifact["plan_sha256"] = verified_plan["artifact_sha256"]
    files = artifact.get("files")
    artifact["frame_count"] = 2 * len(files) if isinstance(files, list) else 0
    _validate_frames(artifact, plan=verified_plan)
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def verify_muvs_event_state_frames(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    verified_plan = verify_muvs_event_state_plan(plan)
    artifact = dict(payload)
    claimed = str(artifact.pop("artifact_sha256", ""))
    _validate_frames(artifact, plan=verified_plan)
    if claimed != _canonical_sha256(artifact):
        raise ValueError("MUVS event-state frames hash mismatch")
    artifact["artifact_sha256"] = claimed
    return artifact


def _basketball_event_files(
    data_root: Path,
) -> list[tuple[str, str, Path, Path]]:
    annotations_root = data_root / "annotations"
    fragments_root = data_root / "fragments"
    if not annotations_root.is_dir() or not fragments_root.is_dir():
        raise FileNotFoundError("MUVS data root is missing annotations/fragments")
    found: list[tuple[str, str, Path, Path]] = []
    for annotation_path in sorted(annotations_root.glob("*_ANNOTATIONS.csv")):
        event_id = annotation_path.name.removesuffix("_ANNOTATIONS.csv")
        matches = [
            (split, fragments_root / split / f"{event_id}_FRAGMENTS.csv")
            for split in ("train", "test")
            if (fragments_root / split / f"{event_id}_FRAGMENTS.csv").is_file()
        ]
        if len(matches) != 1:
            raise ValueError(f"MUVS event {event_id} must have exactly one fragment CSV")
        split, fragments_path = matches[0]
        with fragments_path.open(
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(handle)
            first = next(reader, None)
        if first is None:
            raise ValueError(f"MUVS fragment CSV is empty: {event_id}")
        if str(first.get("sport_genre") or "").lower() == "basketball":
            found.append((event_id, split, annotation_path, fragments_path))
    return found


def _read_annotations(
    path: Path,
    *,
    expected_event: str,
    expected_split: str,
) -> list[dict[str, Any]]:
    required = {
        "dataset_type",
        "event",
        "period",
        "fragment_number",
        "camera_selected",
        "offsets",
    }
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("MUVS annotation CSV is missing required columns")
        for raw in reader:
            try:
                period = int(raw["period"])
                fragment_number = int(raw["fragment_number"])
                offsets = json.loads(raw["offsets"])
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("invalid MUVS annotation row") from exc
            camera_id = str(raw["camera_selected"])
            if (
                raw["dataset_type"] != expected_split
                or raw["event"] != f"{expected_event}_filtered"
                or period < 1
                or fragment_number < 1
                or not re.fullmatch(r"cam_\d+", camera_id)
                or not isinstance(offsets, dict)
                or camera_id not in offsets
                or not math.isfinite(float(offsets[camera_id]))
            ):
                raise ValueError("invalid MUVS annotation row")
            rows.append(
                {
                    "period": period,
                    "fragment_number": fragment_number,
                    "camera_selected": camera_id,
                    "offsets": offsets,
                }
            )
    rows.sort(
        key=lambda row: (
            int(row["period"]),
            int(row["fragment_number"]),
            str(row["camera_selected"]),
        )
    )
    return rows


def _balanced_indices(size: int, count: int) -> list[int]:
    if count < 1 or size < count:
        raise ValueError("invalid balanced MUVS sample request")
    indices = [((2 * index + 1) * size) // (2 * count) for index in range(count)]
    if len(set(indices)) != count:
        raise ValueError("MUVS balanced sampling produced duplicate rows")
    return indices


def _validate_plan(artifact: Mapping[str, Any]) -> None:
    source = artifact.get("source")
    samples = artifact.get("samples")
    if (
        artifact.get("schema_version") != MUVS_EVENT_STATE_PLAN_SCHEMA
        or artifact.get("purpose") != "codex_offline_muvs_source_event_state_annotation"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("reviewer_visible_fields") != _REVIEWER_FIELDS
        or not isinstance(source, Mapping)
        or source.get("record_id") != MUVS_RECORD_ID
        or source.get("record_url") != MUVS_RECORD_URL
        or source.get("record_license") != "CC-BY-4.0"
        or source.get("package_readme_license") != "TO BE ADDED"
        or source.get("media_redistribution") != "not_performed"
        or not isinstance(samples, list)
        or not samples
        or artifact.get("sample_count") != len(samples)
        or int(artifact.get("event_count", 0)) < 1
        or int(artifact.get("samples_per_event", 0)) < 1
        or len(samples) != int(artifact["event_count"]) * int(artifact["samples_per_event"])
    ):
        raise ValueError("MUVS event-state plan policy is invalid")
    expected_keys = {
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
    events = set()
    for index, row in enumerate(samples, start=1):
        sample_id = str(row.get("sample_id") or "")
        times = row.get("frame_times_sec")
        if (
            set(row) != expected_keys
            or _SAMPLE_ID.fullmatch(sample_id) is None
            or sample_id != f"muvs-state-{index:04d}"
            or row.get("split") not in {"train", "test"}
            or not str(row.get("event_id") or "")
            or int(row.get("period", 0)) < 1
            or int(row.get("fragment_number", 0)) < 1
            or re.fullmatch(r"cam_\d+", str(row.get("camera_id") or "")) is None
            or not isinstance(times, list)
            or len(times) != 2
            or not all(math.isfinite(float(value)) and float(value) >= 0 for value in times)
            or float(times[1]) <= float(times[0])
            or not str(row.get("video_url") or "").startswith(f"{MUVS_VIDEO_BASE_URL}/basketball/")
        ):
            raise ValueError("invalid MUVS event-state sample")
        _require_sha256(row.get("annotation_sha256"))
        _require_sha256(row.get("fragments_sha256"))
        events.add(str(row["event_id"]))
    if len(events) != int(artifact["event_count"]):
        raise ValueError("MUVS event-state plan event coverage is invalid")


def _validate_review(
    artifact: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    frames: Mapping[str, Any],
) -> None:
    decisions = artifact.get("decisions")
    planned_ids = [row["sample_id"] for row in plan["samples"]]
    if (
        artifact.get("schema_version") != MUVS_EVENT_STATE_REVIEW_SCHEMA
        or artifact.get("purpose") != "codex_offline_muvs_source_event_state_annotation"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("plan_sha256") != plan["artifact_sha256"]
        or artifact.get("frames_sha256") != frames["artifact_sha256"]
        or artifact.get("reviewer") != "codex_offline_source_annotation"
        or not isinstance(decisions, list)
        or [str(row.get("sample_id") or "") for row in decisions] != planned_ids
    ):
        raise ValueError("MUVS event-state review policy is invalid")
    for row in decisions:
        if (
            row.get("state") not in MUVS_EVENT_STATES
            or not isinstance(row.get("notes"), str)
            or not row["notes"].strip()
        ):
            raise ValueError("invalid MUVS event-state state decision")


def _validate_frames(
    artifact: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
) -> None:
    files = artifact.get("files")
    planned_ids = [row["sample_id"] for row in plan["samples"]]
    if (
        artifact.get("schema_version") != MUVS_EVENT_STATE_FRAMES_SCHEMA
        or artifact.get("purpose") != "codex_offline_muvs_source_event_state_annotation"
        or artifact.get("runtime_consumable") is not False
        or artifact.get("codex_runtime_answer_used") is not False
        or artifact.get("plan_sha256") != plan["artifact_sha256"]
        or not isinstance(files, list)
        or [str(row.get("sample_id") or "") for row in files] != planned_ids
        or artifact.get("frame_count") != 2 * len(files)
    ):
        raise ValueError("MUVS event-state frames policy is invalid")
    for row in files:
        if set(row) != {"sample_id", "frame_before", "frame_after"}:
            raise ValueError("invalid MUVS event-state frame row")
        for key in ("frame_before", "frame_after"):
            frame = row.get(key)
            if not isinstance(frame, Mapping):
                raise ValueError("invalid MUVS event-state frame row")
            path = Path(str(frame.get("path") or ""))
            if (
                set(frame)
                != {
                    "path",
                    "sha256",
                    "size_bytes",
                    "width",
                    "height",
                }
                or path.is_absolute()
                or ".." in path.parts
                or path.suffix.lower() != ".png"
                or int(frame.get("size_bytes", 0)) < 1
                or int(frame.get("width", 0)) < 1
                or int(frame.get("height", 0)) < 1
            ):
                raise ValueError("invalid MUVS event-state frame row")
            _require_sha256(frame.get("sha256"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object) -> str:
    text = str(value or "")
    if _SHA256.fullmatch(text) is None:
        raise ValueError("invalid MUVS source SHA-256")
    return text


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
