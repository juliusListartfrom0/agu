#!/usr/bin/env python3
"""Build a sealed SFace gallery from a benchmark-disjoint annotated face list."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.face_gallery import seal_face_gallery_payload  # noqa: E402
from app.analysis.face_identity import OpenCvSFaceIdentityAdapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector-model", type=Path, required=True)
    parser.add_argument("--recognizer-model", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != "agu.annotated-face-list.v1":
        raise ValueError("unsupported annotated face list schema")
    if manifest.get("benchmark_disjoint") is not True:
        raise ValueError("face gallery sources must be explicitly benchmark-disjoint")
    adapter = OpenCvSFaceIdentityAdapter(
        str(args.detector_model),
        str(args.recognizer_model),
        score_threshold=args.score_threshold,
    )
    entries = []
    for person in manifest.get("persons") or []:
        person_id = str(person.get("person_id") or "").strip()
        if not person_id:
            raise ValueError("annotated face person_id is required")
        crops = []
        sample_hashes = []
        for raw_sample in person.get("samples") or []:
            sample = {"path": raw_sample} if isinstance(raw_sample, str) else dict(raw_sample)
            path = Path(str(sample.get("path") or ""))
            if not path.is_absolute():
                path = args.manifest.parent / path
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"unable to read annotated face sample: {path}")
            bbox = sample.get("bbox")
            if bbox is not None:
                x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
                height, width = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"invalid annotated face bbox: {path}")
                image = image[y1:y2, x1:x2].copy()
            crops.append(image)
            sample_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        result = adapter.embed_player_crops(crops)
        if result is None:
            raise ValueError(f"insufficient consistent face samples for {person_id}")
        entries.append(
            {
                "person_id": person_id,
                "team_id": person.get("team_id"),
                "embedding": [float(value) for value in result.embedding.tolist()],
                "sample_count": result.sample_count,
                "quality": result.quality,
                "sample_sha256": sample_hashes,
            }
        )
    payload = seal_face_gallery_payload(
        {
            "model_id": adapter.model_id,
            "entries": entries,
            "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "benchmark_disjoint": True,
            "annotation_producer": str(manifest.get("annotation_producer") or "unspecified"),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gallery_sha256": payload["gallery_sha256"], "person_count": len(entries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
