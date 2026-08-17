#!/usr/bin/env python3
"""Materialize sparse, hash-bound original frames for offline VRU review."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2

cv2.setNumThreads(1)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.vru_causal_review import verify_vru_causal_review_plan  # noqa: E402

FRAME_MANIFEST_SCHEMA = "agu.vru-causal-review-frame-manifest.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--expected-plan-artifact-sha256",
        help="externally recorded plan SHA-256; required for selection-bound plans",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def materialize_frames(
    *,
    manifest_path: Path,
    plan_path: Path,
    output_dir: Path,
    jpeg_quality: int = 95,
    expected_plan_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != "agu.vru-basketball-source-manifest.v1":
        raise ValueError("unsupported VRU source manifest")
    if manifest.get("runtime_consumable") is not False:
        raise ValueError("VRU source manifest must remain offline-only")
    videos = manifest.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("VRU source manifest requires videos")
    videos_by_clip: dict[str, dict[str, Any]] = {}
    for video in videos:
        if not isinstance(video, dict):
            raise ValueError("VRU source manifest video rows must be objects")
        clip_id = str(video.get("clip_id") or "")
        if not clip_id or clip_id in videos_by_clip:
            raise ValueError("VRU source manifest clip IDs must be unique")
        videos_by_clip[clip_id] = video

    plan = verify_vru_causal_review_plan(
        _read_json(plan_path),
        expected_artifact_sha256=expected_plan_artifact_sha256,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_output_dir = output_dir.resolve()
    rows: list[dict[str, Any]] = []
    source_caps: dict[str, cv2.VideoCapture] = {}
    try:
        for example in plan["examples"]:
            review_id = str(example["review_id"])
            source_sha = str(example["source_video_sha256"])
            video = _find_video(videos_by_clip, source_sha)
            video_path = (manifest_path.parent / Path(str(video["path"]))).resolve()
            if not video_path.is_file():
                raise ValueError(f"VRU source video is missing: {video_path}")
            cap = source_caps.get(source_sha)
            if cap is None:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    raise ValueError(f"unable to open VRU source video: {video_path}")
                source_caps[source_sha] = cap
            frame_indexes = [int(index) for index in example["frame_indexes"]]
            frames = _read_frame_sequence(cap, frame_indexes, video_path=video_path)
            for position, (frame_index, frame) in enumerate(zip(frame_indexes, frames, strict=True)):
                frame_key = str(frame_index)
                expected_raw_sha = str(example["frame_sha256s"].get(frame_key) or "")
                if not expected_raw_sha:
                    raise ValueError(f"plan has no raw frame hash for {review_id}:{frame_index}")
                raw_sha = hashlib.sha256(frame.tobytes()).hexdigest()
                if raw_sha != expected_raw_sha:
                    raise ValueError(f"VRU source frame hash mismatch for {review_id}:{frame_index}")
                relative_path = Path(f"{review_id}_{position:03d}_{int(frame_index)}.jpg")
                destination = _safe_output_path(
                    resolved_output_dir,
                    relative_path,
                )
                if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                    raise ValueError(f"unable to write review frame: {destination}")
                rows.append(
                    {
                        "review_id": review_id,
                        "position": position,
                        "frame_index": int(frame_index),
                        "raw_frame_sha256": raw_sha,
                        "relative_path": relative_path.as_posix(),
                        "jpeg_bytes": destination.stat().st_size,
                        "jpeg_sha256": _file_sha256(destination),
                    }
                )
    finally:
        for cap in source_caps.values():
            cap.release()

    artifact: dict[str, Any] = {
        "schema_version": FRAME_MANIFEST_SCHEMA,
        "purpose": "offline_codex_visual_review_of_original_vru_frames",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "labels_hidden_from_reviewer": True,
        "source_manifest_sha256": _file_sha256(manifest_path),
        "review_plan_sha256": plan["artifact_sha256"],
        "frame_root": output_dir.name,
        "jpeg_quality": jpeg_quality,
        "frame_count": len(rows),
        "frames": rows,
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def _read_frame_sequence(cap: cv2.VideoCapture, frame_indexes: list[int], *, video_path: Path) -> list[Any]:
    """Decode one sorted sparse sequence with a single seek per review window."""

    if not frame_indexes:
        return []
    if frame_indexes != sorted(set(frame_indexes)):
        raise ValueError(f"frame indexes must be sorted and unique: {video_path}")
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indexes[0]):
        raise ValueError(f"unable to seek VRU source frame {frame_indexes[0]}: {video_path}")
    wanted = set(frame_indexes)
    frames: list[Any] = []
    current = frame_indexes[0]
    last = frame_indexes[-1]
    while current <= last:
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"unable to decode VRU source frame {current}: {video_path}")
        if current in wanted:
            frames.append(frame)
        current += 1
    return frames


def _find_video(videos: dict[str, dict[str, Any]], source_sha: str) -> dict[str, Any]:
    matches = [row for row in videos.values() if str(row.get("sha256") or "") == source_sha]
    if len(matches) != 1:
        raise ValueError("review plan source hash is not uniquely present in source manifest")
    return matches[0]


def _safe_output_path(output_dir: Path, relative_path: Path) -> Path:
    """Resolve a frame destination and reject any path traversal."""

    frame_root = output_dir.resolve()
    destination = (frame_root / relative_path).resolve()
    if not destination.is_relative_to(frame_root):
        raise ValueError("review frame destination is outside the frame root")
    return destination


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("artifact_sha256", None)
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    args = parse_args()
    output_dir = args.output
    artifact = materialize_frames(
        manifest_path=args.manifest,
        plan_path=args.plan,
        output_dir=output_dir,
        jpeg_quality=args.jpeg_quality,
        expected_plan_artifact_sha256=args.expected_plan_artifact_sha256,
    )
    manifest_path = output_dir.parent / f"{output_dir.name}_manifest.json"
    manifest_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "frame_manifest": str(manifest_path),
                "frame_count": artifact["frame_count"],
                "artifact_sha256": artifact["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
