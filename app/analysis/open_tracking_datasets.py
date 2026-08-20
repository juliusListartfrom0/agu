"""License-aware MOT annotation catalogs for open basketball pretraining."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

OPEN_TRACKING_CATALOG_SCHEMA = "agu.open-tracking-dataset.v1"

DATASET_POLICIES = {
    "trackid3x3": {
        "source_url": "https://github.com/open-starlab/TrackID3x3",
        "data_license": "CC-BY-4.0",
        "role": "basketball_detection_tracking_pose_and_reid_pretraining",
        "third_party_note": (
            "optional jersey-number-pipeline is CC-BY-NC-3.0 and remains "
            "isolated from the core adapter"
        ),
    },
    "teamtrack": {
        "source_url": "https://github.com/AtomScott/TeamTrack",
        "data_license": "MIT-as-declared-by-dataset-distribution",
        "role": "basketball_detection_and_tracking_pretraining",
        "third_party_note": "verify the license included with the downloaded dataset copy",
    },
}


def build_open_tracking_catalog(
    dataset_root: str | Path,
    *,
    dataset_id: str,
    source_revision: str,
) -> dict[str, Any]:
    """Validate MOT ground truth and build a sealed training-only catalog."""

    policy = DATASET_POLICIES.get(dataset_id)
    if policy is None:
        raise ValueError(f"unsupported open tracking dataset: {dataset_id}")
    if not source_revision.strip():
        raise ValueError("open tracking import requires an immutable source revision")

    root = Path(dataset_root)
    annotation_paths = sorted(root.rglob("gt.txt"))
    if dataset_id == "trackid3x3":
        annotation_paths.extend(
            sorted(root.glob("ground_truth/*/MOT/*.txt"))
        )
        annotation_paths = sorted(set(annotation_paths))
    if not annotation_paths:
        raise ValueError("no supported MOT annotations found")

    sequences = [
        _load_mot_sequence(root, annotation_path, dataset_id=dataset_id)
        for annotation_path in annotation_paths
    ]
    payload: dict[str, Any] = {
        "schema_version": OPEN_TRACKING_CATALOG_SCHEMA,
        "dataset_id": dataset_id,
        "source_url": policy["source_url"],
        "source_revision": source_revision,
        "data_license": policy["data_license"],
        "third_party_note": policy["third_party_note"],
        "role": policy["role"],
        "training_only": True,
        "runtime_consumable": False,
        "acceptance_media_allowed": False,
        "sequence_count": len(sequences),
        "annotation_count": sum(item["annotation_count"] for item in sequences),
        "track_count": sum(item["track_count"] for item in sequences),
        "sequences": sequences,
    }
    payload["catalog_sha256"] = _json_sha256(payload)
    return payload


def verify_open_tracking_catalog(payload: dict[str, Any]) -> dict[str, Any]:
    catalog = dict(payload)
    claimed_hash = str(catalog.pop("catalog_sha256", ""))
    if catalog.get("schema_version") != OPEN_TRACKING_CATALOG_SCHEMA:
        raise ValueError("unsupported open tracking catalog schema")
    if catalog.get("training_only") is not True:
        raise ValueError("open tracking annotations must remain training-only")
    if catalog.get("runtime_consumable") is not False:
        raise ValueError("open tracking annotations cannot be runtime-consumable")
    if catalog.get("acceptance_media_allowed") is not False:
        raise ValueError("open tracking training data cannot become acceptance media")
    if _json_sha256(catalog) != claimed_hash:
        raise ValueError("open tracking catalog hash mismatch")
    catalog["catalog_sha256"] = claimed_hash
    return catalog


def _load_mot_sequence(
    root: Path,
    annotation_path: Path,
    *,
    dataset_id: str,
) -> dict[str, Any]:
    frame_ids: set[int] = set()
    track_ids: set[int] = set()
    annotation_count = 0
    minimum_x = float("inf")
    minimum_y = float("inf")
    maximum_x = 0.0
    maximum_y = 0.0

    for line_number, line in enumerate(
        annotation_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        values = [value.strip() for value in line.split(",")]
        if len(values) < 6:
            raise ValueError(
                f"MOT row needs at least six columns: {annotation_path}:{line_number}"
            )
        try:
            frame_id = int(float(values[0]))
            track_id = int(float(values[1]))
            x, y, width, height = (float(value) for value in values[2:6])
        except ValueError as error:
            raise ValueError(
                f"invalid MOT numeric value: {annotation_path}:{line_number}"
            ) from error
        if frame_id < 0 or track_id < 0:
            raise ValueError(f"negative MOT identity: {annotation_path}:{line_number}")
        if width <= 0 or height <= 0 or x < 0 or y < 0:
            raise ValueError(f"invalid MOT bounding box: {annotation_path}:{line_number}")
        if len(values) >= 7:
            confidence = float(values[6])
            if confidence < -1 or confidence > 1:
                raise ValueError(
                    f"invalid MOT confidence: {annotation_path}:{line_number}"
                )

        annotation_count += 1
        frame_ids.add(frame_id)
        track_ids.add(track_id)
        minimum_x = min(minimum_x, x)
        minimum_y = min(minimum_y, y)
        maximum_x = max(maximum_x, x + width)
        maximum_y = max(maximum_y, y + height)

    if not annotation_count:
        raise ValueError(f"empty MOT annotation file: {annotation_path}")

    if dataset_id == "trackid3x3" and annotation_path.parent.name == "MOT":
        sequence = annotation_path.stem
        subset = annotation_path.parent.parent.name
        media = _find_trackid3x3_media(root, subset=subset, sequence=sequence)
        if media is None:
            raise ValueError(
                f"TrackID3x3 named MOT annotation requires its matching video: "
                f"{sequence}"
            )
    else:
        sequence_root = annotation_path.parent.parent
        sequence = sequence_root.name
        media = _find_sequence_media(sequence_root)
    return {
        "sequence": sequence,
        "annotation_path": annotation_path.relative_to(root).as_posix(),
        "annotation_sha256": _file_sha256(annotation_path),
        "annotation_count": annotation_count,
        "frame_count": len(frame_ids),
        "first_frame": min(frame_ids),
        "last_frame": max(frame_ids),
        "track_count": len(track_ids),
        "coordinate_extent": [minimum_x, minimum_y, maximum_x, maximum_y],
        "media_path": media.relative_to(root).as_posix() if media else None,
        "media_sha256": _file_sha256(media) if media else None,
    }


def _find_trackid3x3_media(
    root: Path,
    *,
    subset: str,
    sequence: str,
) -> Path | None:
    candidates = [
        path
        for base in (root / subset / "raw", root / "videos" / subset / "raw")
        for suffix in (".mp4", ".mov", ".mkv")
        if (path := base / f"{sequence}{suffix}").is_file()
    ]
    if len(candidates) > 1:
        raise ValueError(f"multiple TrackID3x3 videos found for {sequence}")
    return candidates[0] if candidates else None


def _find_sequence_media(sequence_root: Path) -> Path | None:
    candidates = sorted(
        path
        for pattern in ("*.mp4", "*.mov", "*.mkv")
        for path in sequence_root.glob(pattern)
        if path.is_file()
    )
    if len(candidates) > 1:
        raise ValueError(f"multiple sequence videos found: {sequence_root}")
    return candidates[0] if candidates else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
