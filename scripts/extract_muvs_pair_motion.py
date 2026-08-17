#!/usr/bin/env python3
"""Extract camera-compensated pair-motion features from sealed MUVS frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.muvs_pair_motion import (  # noqa: E402
    build_muvs_pair_motion_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frame(
    metadata: Mapping[str, Any],
    *,
    frame_root: Path,
):
    path = (frame_root / str(metadata["path"])).resolve()
    try:
        path.relative_to(frame_root.resolve())
    except ValueError as exc:
        raise ValueError("MUVS pair motion frame escapes artifact root") from exc
    if (
        not path.is_file()
        or path.stat().st_size != int(metadata["size_bytes"])
        or _sha256(path) != metadata["sha256"]
    ):
        raise ValueError(f"MUVS pair motion frame binding mismatch: {path.name}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if (
        image is None
        or image.shape[1] != int(metadata["width"])
        or image.shape[0] != int(metadata["height"])
    ):
        raise ValueError(f"MUVS pair motion frame decode mismatch: {path.name}")
    return image


def main() -> int:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    frames = json.loads(args.frames.read_text(encoding="utf-8"))
    review = json.loads(args.review.read_text(encoding="utf-8"))
    if not all(isinstance(row, dict) for row in (plan, frames, review)):
        raise ValueError("expected MUVS JSON objects")
    artifact = build_muvs_pair_motion_features(
        plan=plan,
        frames=frames,
        review=review,
        frame_loader=lambda metadata: _load_frame(
            metadata,
            frame_root=args.frames.parent,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "examples": len(artifact["examples"]),
                "homography_applied": sum(
                    row["homography_applied"]
                    for row in artifact["examples"]
                ),
                "artifact_sha256": artifact["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
