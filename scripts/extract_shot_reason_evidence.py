#!/usr/bin/env python3
"""Build label-free continuous shot-reason evidence from existing pose artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.shot_reason_evidence import (  # noqa: E402
    build_reason_evidence_artifact,
)
from app.analysis.shot_validity_scene_state import file_sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-embeddings", type=Path, required=True)
    parser.add_argument("--pose", type=Path, action="append", required=True)
    parser.add_argument(
        "--candidate-bundle", type=Path, action="append", required=True
    )
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pose_artifacts = [_read_json(path) for path in args.pose]
    candidate_bundles = [_read_json(path) for path in args.candidate_bundle]
    dimensions = {}
    for path in args.video:
        source_sha = file_sha256(path)
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f"cannot open source video: {path.name}")
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid source dimensions: {path.name}")
        dimensions[source_sha] = (width, height)
    artifact = build_reason_evidence_artifact(
        scene_artifact=_read_json(args.scene_embeddings),
        pose_artifacts=pose_artifacts,
        pose_source_sha256s=[file_sha256(path) for path in args.pose],
        candidate_bundles=candidate_bundles,
        video_dimensions=dimensions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "example_count": len(artifact["examples"]),
                "feature_count": len(artifact["feature_names"]),
                "artifact_sha256": artifact["artifact_sha256"],
            }
        )
    )
    return 0


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
