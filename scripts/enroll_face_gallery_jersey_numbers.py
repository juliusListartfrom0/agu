#!/usr/bin/env python3
"""Attach benchmark-disjoint AGU jersey reads to an annotated face gallery."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.face_gallery import parse_face_gallery, seal_face_gallery_payload  # noqa: E402
from app.analysis.jersey_identity import CachedJerseyNumberReader  # noqa: E402
from app.analysis.schemas import JerseyNumberCandidateResponse  # noqa: E402
from app.analysis.vlm import OllamaVLMVerifier  # noqa: E402
from app.config import get_settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-gallery", type=Path, required=True)
    parser.add_argument("--annotated-face-list", type=Path, required=True)
    parser.add_argument("--face-candidate-manifest", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", default=settings.ollama_model)
    parser.add_argument("--host", default=settings.ollama_host)
    parser.add_argument("--timeout", type=float, default=settings.official_vlm_timeout)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--context-length", type=int, default=settings.official_vlm_context_length)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maximum-samples", type=int, default=8)
    parser.add_argument("--minimum-confidence", type=float, default=0.90)
    parser.add_argument("--minimum-consensus", type=int, default=2)
    return parser.parse_args()


def enroll_jersey_numbers(
    *,
    gallery_payload: dict[str, Any],
    annotated_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    video_path: Path,
    reader: CachedJerseyNumberReader,
    maximum_samples: int,
    minimum_confidence: float,
    minimum_consensus: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    parse_face_gallery(gallery_payload)
    if annotated_payload.get("benchmark_disjoint") is not True:
        raise ValueError("annotated face list must be benchmark-disjoint")
    source = _matching_source(candidate_payload, video_path)
    observations = _candidate_observations(candidate_payload)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open enrollment video: {video_path}")
    enrolled: dict[str, str] = {}
    try:
        for person in annotated_payload.get("persons") or []:
            person_id = str(person.get("person_id") or "").strip()
            candidates = [
                observations[str(sample.get("path"))]
                for sample in person.get("samples") or []
                if str(sample.get("path")) in observations
            ]
            selected = _spread_samples(candidates, maximum_samples=maximum_samples)
            crops = [_read_torso_crop(capture, item) for item in selected]
            crops = [crop for crop in crops if crop is not None]
            number = _consensus_number(
                reader,
                crops,
                scope=f"enrollment:{source['source_video_id']}:{person_id}",
                minimum_confidence=minimum_confidence,
                minimum_consensus=minimum_consensus,
            )
            if number is not None:
                enrolled[person_id] = number
            print(
                json.dumps(
                    {"person_id": person_id, "jersey_number": number, "processed": len(enrolled)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        capture.release()

    output = json.loads(json.dumps(gallery_payload))
    for entry in output.get("entries") or []:
        entry.pop("jersey_number", None)
        person_id = str(entry.get("person_id") or "")
        if person_id in enrolled:
            entry["jersey_number"] = enrolled[person_id]
    output["jersey_enrollment"] = {
        "role": "benchmark_disjoint_identity_registration",
        "annotation_boundary": "face_identity_anchor_only_no_event_labels",
        "source_video_sha256": source["sha256"],
        "reader_model": reader.model,
        "minimum_confidence": minimum_confidence,
        "minimum_consensus": minimum_consensus,
        "enrolled_person_count": len(enrolled),
    }
    return seal_face_gallery_payload(output), enrolled


def _matching_source(candidate_payload: dict[str, Any], video_path: Path) -> dict[str, Any]:
    digest = _sha256_file(video_path)
    matches = [
        source
        for source in candidate_payload.get("sources") or []
        if source.get("filename") == video_path.name and source.get("sha256") == digest
    ]
    if len(matches) != 1:
        raise ValueError("face candidate manifest does not uniquely bind the enrollment video")
    return matches[0]


def _candidate_observations(candidate_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(sample["path"]): sample
        for cluster in candidate_payload.get("clusters") or []
        for sample in cluster.get("samples") or []
        if sample.get("path")
    }


def _spread_samples(samples: Sequence[dict[str, Any]], *, maximum_samples: int) -> list[dict[str, Any]]:
    if maximum_samples <= 0:
        raise ValueError("maximum samples must be positive")
    unique = {int(item["frame"]): item for item in samples if item.get("frame") is not None}
    ordered = [unique[frame] for frame in sorted(unique)]
    if len(ordered) <= maximum_samples:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, maximum_samples, dtype=int)
    return [ordered[int(index)] for index in indices]


def _read_torso_crop(capture: cv2.VideoCapture, sample: dict[str, Any]) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(sample["frame"]))
    ok, frame = capture.read()
    if not ok or frame is None:
        return None
    x, y, width, height = (float(value) for value in sample["bbox"])
    frame_height, frame_width = frame.shape[:2]
    x1 = max(0, int(round(x - 1.75 * width)))
    x2 = min(frame_width, int(round(x + 2.75 * width)))
    y1 = max(0, int(round(y - 0.50 * height)))
    y2 = min(frame_height, int(round(y + 4.00 * height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _consensus_number(
    reader: CachedJerseyNumberReader,
    crops: Sequence[np.ndarray],
    *,
    scope: str,
    minimum_confidence: float,
    minimum_consensus: int,
) -> str | None:
    if len(crops) < minimum_consensus or minimum_consensus <= 0:
        return None
    accepted: list[JerseyNumberCandidateResponse] = []
    for index in range(minimum_consensus):
        partition = list(crops[index::minimum_consensus])
        visible = [
            item
            for item in reader.read_jersey_number(
                partition,
                scope=f"{scope}:consensus-{index + 1}-of-{minimum_consensus}",
            )
            if item.visible
            and item.number is not None
            and item.number.isdigit()
            and item.confidence >= minimum_confidence
        ]
        if len(visible) != 1:
            return None
        accepted.append(visible[0])
    numbers = {item.number for item in accepted}
    return accepted[0].number if len(numbers) == 1 else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    reader = CachedJerseyNumberReader(
        OllamaVLMVerifier(
            model=args.model,
            host=args.host,
            timeout=args.timeout,
            image_width=args.image_width,
            context_length=args.context_length,
            seed=args.seed,
        ),
        args.cache,
    )
    output, enrolled = enroll_jersey_numbers(
        gallery_payload=json.loads(args.face_gallery.read_text(encoding="utf-8")),
        annotated_payload=json.loads(args.annotated_face_list.read_text(encoding="utf-8")),
        candidate_payload=json.loads(args.face_candidate_manifest.read_text(encoding="utf-8")),
        video_path=args.video,
        reader=reader,
        maximum_samples=args.maximum_samples,
        minimum_confidence=args.minimum_confidence,
        minimum_consensus=args.minimum_consensus,
    )
    _write_json_atomic(args.output, output)
    print(json.dumps({"gallery_sha256": output["gallery_sha256"], "jersey_numbers": enrolled}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
