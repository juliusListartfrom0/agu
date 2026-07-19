#!/usr/bin/env python3
"""Extract YuNet+SFace face clusters from benchmark-disjoint raw videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.face_enrollment import (  # noqa: E402
    EnrollmentFaceObservation,
    cluster_enrollment_faces,
    select_enrollment_samples,
)
from app.analysis.face_identity import OpenCvSFaceIdentityAdapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--detector-model", type=Path, required=True)
    parser.add_argument("--recognizer-model", type=Path, required=True)
    parser.add_argument("--sample-interval-sec", type=float, default=1.0)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float)
    parser.add_argument("--minimum-face-size", type=int, default=32)
    parser.add_argument("--detection-threshold", type=float, default=0.75)
    parser.add_argument("--cluster-cosine", type=float, default=0.58)
    parser.add_argument("--minimum-cluster-samples", type=int, default=2)
    parser.add_argument("--maximum-samples-per-cluster", type=int, default=6)
    parser.add_argument("--benchmark-disjoint", action="store_true", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_interval_sec <= 0 or args.start_sec < 0:
        raise ValueError("sampling interval must be positive and start non-negative")
    if args.end_sec is not None and args.end_sec <= args.start_sec:
        raise ValueError("end-sec must be after start-sec")
    if args.minimum_cluster_samples < 2:
        raise ValueError("minimum-cluster-samples must be at least two")
    adapter = OpenCvSFaceIdentityAdapter(
        str(args.detector_model),
        str(args.recognizer_model),
        score_threshold=args.detection_threshold,
    )
    observations: list[EnrollmentFaceObservation] = []
    sources = []
    for video_index, video_path in enumerate(args.video, start=1):
        source_video_id = f"enrollment_{video_index:03d}"
        sources.append(
            {
                "source_video_id": source_video_id,
                "filename": video_path.name,
                "sha256": _file_sha256(video_path),
                "size_bytes": video_path.stat().st_size,
            }
        )
        observations.extend(
            _extract_video_faces(
                video_path,
                source_video_id=source_video_id,
                adapter=adapter,
                start_sec=args.start_sec,
                end_sec=args.end_sec,
                sample_interval_sec=args.sample_interval_sec,
                minimum_face_size=args.minimum_face_size,
            )
        )
    groups = cluster_enrollment_faces(observations, minimum_cosine=args.cluster_cosine)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clusters = []
    retained_index = 0
    for group in groups:
        distinct_frames = {(item.source_video_id, item.frame) for item in group}
        if len(distinct_frames) < args.minimum_cluster_samples:
            continue
        retained_index += 1
        cluster_id = f"face-cluster-{retained_index:04d}"
        samples = []
        sample_images = []
        for sample_index, item in enumerate(
            select_enrollment_samples(group, maximum=args.maximum_samples_per_cluster),
            start=1,
        ):
            filename = f"{cluster_id}-{sample_index:02d}.jpg"
            output_path = args.output_dir / filename
            if not cv2.imwrite(str(output_path), item.crop):
                raise RuntimeError(f"unable to write enrollment crop: {output_path}")
            sample_images.append(item.crop)
            samples.append(
                {
                    "path": str(output_path.resolve()),
                    "source_video_id": item.source_video_id,
                    "frame": item.frame,
                    "bbox": list(item.bbox),
                    "detection_confidence": item.confidence,
                    "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                }
            )
        review_sheet = args.output_dir / f"{cluster_id}-review.jpg"
        _write_review_sheet(cluster_id, sample_images, review_sheet)
        pair_scores = [
            float(group[left].embedding @ group[right].embedding)
            for left in range(len(group))
            for right in range(left + 1, len(group))
        ]
        clusters.append(
            {
                "cluster_id": cluster_id,
                "observation_count": len(group),
                "minimum_pair_cosine": min(pair_scores) if pair_scores else 1.0,
                "review_sheet": str(review_sheet.resolve()),
                "samples": samples,
            }
        )
    payload = {
        "schema_version": "agu.face-enrollment-candidates.v1",
        "benchmark_disjoint": True,
        "producer": "agu_yunet_sface",
        "model_id": adapter.model_id,
        "sources": sources,
        "config": {
            "sample_interval_sec": args.sample_interval_sec,
            "start_sec": args.start_sec,
            "end_sec": args.end_sec,
            "minimum_face_size": args.minimum_face_size,
            "detection_threshold": args.detection_threshold,
            "cluster_cosine": args.cluster_cosine,
            "minimum_cluster_samples": args.minimum_cluster_samples,
        },
        "clusters": clusters,
    }
    payload["manifest_sha256"] = _canonical_sha256({**payload, "manifest_sha256": ""})
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest_sha256": payload["manifest_sha256"],
                "detected_face_count": len(observations),
                "retained_cluster_count": len(clusters),
            },
            indent=2,
        )
    )
    return 0


def _extract_video_faces(
    video_path: Path,
    *,
    source_video_id: str,
    adapter: OpenCvSFaceIdentityAdapter,
    start_sec: float,
    end_sec: float | None,
    sample_interval_sec: float,
    minimum_face_size: int,
) -> list[EnrollmentFaceObservation]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open enrollment video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError(f"invalid enrollment video metadata: {video_path}")
    start_frame = max(0, int(round(start_sec * fps)))
    end_frame = min(
        frame_count - 1,
        int(round(end_sec * fps)) if end_sec is not None else frame_count - 1,
    )
    stride = max(1, int(round(sample_interval_sec * fps)))
    result: list[EnrollmentFaceObservation] = []
    try:
        for frame_number in range(start_frame, end_frame + 1, stride):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            for face in adapter.detect_frame_faces(frame, minimum_face_size=minimum_face_size):
                crop = _padded_player_face_crop(frame, face.bbox)
                if crop.size == 0:
                    continue
                result.append(
                    EnrollmentFaceObservation(
                        source_video_id=source_video_id,
                        frame=frame_number,
                        bbox=face.bbox,
                        confidence=face.confidence,
                        embedding=face.embedding,
                        crop=crop,
                    )
                )
    finally:
        capture.release()
    return result


def _padded_player_face_crop(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Keep the face in the upper third so the runtime player-crop gate is reproduced."""

    x, y, width, height = bbox
    frame_height, frame_width = frame.shape[:2]
    x1 = max(0, int(round(x - width * 0.7)))
    x2 = min(frame_width, int(round(x + width * 1.7)))
    y1 = max(0, int(round(y - height * 0.4)))
    y2 = min(frame_height, int(round(y + height * 2.4)))
    return frame[y1:y2, x1:x2].copy()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_review_sheet(cluster_id: str, images: list[np.ndarray], output_path: Path) -> None:
    tiles = []
    for image in images:
        resized = cv2.resize(image, (160, 160), interpolation=cv2.INTER_AREA)
        tile = cv2.copyMakeBorder(resized, 28, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(
            tile,
            cluster_id,
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    if not tiles or not cv2.imwrite(str(output_path), cv2.hconcat(tiles)):
        raise RuntimeError(f"unable to write enrollment review sheet: {output_path}")


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
