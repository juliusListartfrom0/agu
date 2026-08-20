#!/usr/bin/env python3
"""Attach a hash-sealed identity-only jersey annotation to a face gallery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.face_gallery import attach_face_jersey_annotations  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-gallery", type=Path, required=True)
    parser.add_argument("--jersey-annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = attach_face_jersey_annotations(
        json.loads(args.face_gallery.read_text(encoding="utf-8")),
        json.loads(args.jersey_annotations.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"gallery_sha256": output["gallery_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
