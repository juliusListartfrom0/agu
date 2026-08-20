"""Prepare hash-bound, training-only ReID crops from sealed MOT catalogs."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2

from app.analysis.open_tracking_datasets import verify_open_tracking_catalog

REID_TRAINING_DATASET_SCHEMA = "agu.reid-training-crops.v1"


def build_reid_training_crops(
    *,
    catalog_payload: dict[str, Any],
    dataset_root: Path,
    output_root: Path,
    maximum_crops_per_track: int = 12,
    minimum_crop_width: int = 24,
    minimum_crop_height: int = 48,
    jpeg_quality: int = 90,
    identity_scope: str = "sequence_track",
) -> dict[str, Any]:
    """Crop MOT identities without assigning roster names or acceptance answers."""

    catalog = verify_open_tracking_catalog(catalog_payload)
    if maximum_crops_per_track <= 0 or minimum_crop_width <= 0 or minimum_crop_height <= 0:
        raise ValueError("ReID crop count and dimensions must be positive")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in [1,100]")
    if identity_scope not in {"sequence_track", "dataset_track"}:
        raise ValueError("identity_scope must be sequence_track or dataset_track")
    if catalog.get("training_only") is not True or catalog.get("acceptance_media_allowed") is not False:
        raise ValueError("ReID crops require a training-only, acceptance-forbidden catalog")

    samples: list[dict[str, Any]] = []
    identity_summaries: dict[str, dict[str, Any]] = {}
    for sequence in catalog.get("sequences") or []:
        annotation_path = dataset_root / str(sequence["annotation_path"])
        media_value = sequence.get("media_path")
        if not media_value:
            raise ValueError(f"MOT sequence has no media: {sequence.get('sequence')}")
        media_path = dataset_root / str(media_value)
        grouped = _load_mot_boxes(annotation_path)
        selected = {
            track_id: _evenly_select(rows, maximum_crops_per_track)
            for track_id, rows in grouped.items()
        }
        frame_requests: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
        for track_id, rows in selected.items():
            for frame_id, box in rows:
                frame_requests[frame_id].append((track_id, box))
        written_by_track = _write_requested_crops(
            media_path=media_path,
            output_root=output_root,
            sequence_name=str(sequence["sequence"]),
            frame_requests=frame_requests,
            minimum_crop_width=minimum_crop_width,
            minimum_crop_height=minimum_crop_height,
            jpeg_quality=jpeg_quality,
        )
        for track_id, written in sorted(written_by_track.items()):
            identity_id = (
                f"{sequence['sequence']}:{track_id}"
                if identity_scope == "sequence_track"
                else f"track:{track_id}"
            )
            summary = identity_summaries.setdefault(
                identity_id,
                {
                    "identity_id": identity_id,
                    "sequences": [],
                    "source_track_id": track_id,
                    "sample_count": 0,
                },
            )
            if sequence["sequence"] not in summary["sequences"]:
                summary["sequences"].append(sequence["sequence"])
            summary["sample_count"] += len(written)
            for frame_id, path in written:
                samples.append(
                    {
                        "identity_id": identity_id,
                        "frame_id": frame_id,
                        "path": path.relative_to(output_root).as_posix(),
                        "sha256": _file_sha256(path),
                    }
                )

    payload: dict[str, Any] = {
        "schema_version": REID_TRAINING_DATASET_SCHEMA,
        "source_catalog_sha256": catalog["catalog_sha256"],
        "dataset_id": catalog["dataset_id"],
        "data_license": catalog["data_license"],
        "annotation_producer": "upstream_mot_ground_truth",
        "training_only": True,
        "runtime_consumable": False,
        "acceptance_media_allowed": False,
        "benchmark_disjoint": True,
        "maximum_crops_per_track": maximum_crops_per_track,
        "identity_scope": identity_scope,
        "identity_count": len(identity_summaries),
        "sample_count": len(samples),
        "identities": sorted(identity_summaries.values(), key=lambda item: item["identity_id"]),
        "samples": samples,
    }
    payload["manifest_sha256"] = _json_sha256(payload)
    return payload


def _load_mot_boxes(path: Path) -> dict[int, list[tuple[int, tuple[float, float, float, float]]]]:
    grouped: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        frame_id = int(float(values[0]))
        track_id = int(float(values[1]))
        x, y, width, height = (float(value) for value in values[2:6])
        grouped[track_id].append((frame_id, (x, y, width, height)))
    return {track_id: sorted(rows) for track_id, rows in grouped.items()}


def _evenly_select(values: list[Any], maximum: int) -> list[Any]:
    if len(values) <= maximum:
        return list(values)
    if maximum == 1:
        return [values[len(values) // 2]]
    return [values[round(index * (len(values) - 1) / (maximum - 1))] for index in range(maximum)]


def _write_requested_crops(
    *,
    media_path: Path,
    output_root: Path,
    sequence_name: str,
    frame_requests: dict[int, list[tuple[int, tuple[float, float, float, float]]]],
    minimum_crop_width: int,
    minimum_crop_height: int,
    jpeg_quality: int,
) -> dict[int, list[tuple[int, Path]]]:
    capture = cv2.VideoCapture(str(media_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open MOT media: {media_path}")
    written: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    try:
        for frame_id in sorted(frame_requests):
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_id - 1))
            ok, frame = capture.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            for track_id, (x, y, box_width, box_height) in frame_requests[frame_id]:
                x1 = max(0, min(width, int(round(x))))
                y1 = max(0, min(height, int(round(y))))
                x2 = max(0, min(width, int(round(x + box_width))))
                y2 = max(0, min(height, int(round(y + box_height))))
                if x2 - x1 < minimum_crop_width or y2 - y1 < minimum_crop_height:
                    continue
                crop = frame[y1:y2, x1:x2]
                path = output_root / sequence_name / f"track_{track_id:04d}" / f"frame_{frame_id:06d}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                    raise RuntimeError(f"unable to write ReID crop: {path}")
                written[track_id].append((frame_id, path))
    finally:
        capture.release()
    return dict(written)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
