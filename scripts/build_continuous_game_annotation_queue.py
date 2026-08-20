#!/usr/bin/env python3
"""Build a label-hidden coverage queue from a retained VRU source manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.continuous_game_annotation_queue import (  # noqa: E402
    build_continuous_game_annotation_queue,
    canonical_sha256,
    verify_continuous_game_annotation_queue,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stride-seconds", type=float, default=10.0)
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_queue(*, manifest_path: Path, stride_seconds: float, window_seconds: float) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "agu.vru-basketball-source-manifest.v1":
        raise ValueError("unsupported continuous-game source manifest")
    if manifest.get("runtime_consumable") is not False:
        raise ValueError("source manifest must remain offline-only")
    if manifest.get("codex_runtime_answer_used") is not False:
        raise ValueError("source manifest cannot be a runtime answer channel")
    videos = manifest.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("source manifest requires videos")
    rows: list[dict[str, Any]] = []
    for video in videos:
        if not isinstance(video, dict):
            raise ValueError("source manifest video rows must be objects")
        path = (manifest_path.parent / str(video.get("path") or "")).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha = file_sha256(path)
        if actual_sha != str(video.get("sha256") or ""):
            raise ValueError(f"source video hash mismatch: {path}")
        rows.append(
            {
                "source_id": str(video.get("location") or video.get("clip_id") or ""),
                "source_video_filename": path.name,
                "source_video_sha256": actual_sha,
                "source_fps": float(video.get("fps")),
                "frame_count": int(video.get("frame_count")),
                "duration_seconds": float(video.get("duration_seconds")),
            }
        )
    artifact = build_continuous_game_annotation_queue(
        rows,
        stride_seconds=stride_seconds,
        window_seconds=window_seconds,
    )
    artifact["source_manifest"] = manifest_path.as_posix()
    artifact["source_manifest_sha256"] = file_sha256(manifest_path)
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return verify_continuous_game_annotation_queue(artifact)


def main() -> int:
    args = parse_args()
    artifact = build_queue(
        manifest_path=args.manifest,
        stride_seconds=args.stride_seconds,
        window_seconds=args.window_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "sources": len(artifact["sources"]),
                "windows": len(artifact["windows"]),
                "artifact_sha256": artifact["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
