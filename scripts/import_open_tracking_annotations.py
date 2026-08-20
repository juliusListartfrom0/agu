#!/usr/bin/env python3
"""Import TrackID3x3 or TeamTrack MOT annotations into a sealed AGU catalog."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.open_tracking_datasets import (  # noqa: E402
    DATASET_POLICIES,
    build_open_tracking_catalog,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-id", choices=sorted(DATASET_POLICIES), required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = build_open_tracking_catalog(
        args.dataset_root,
        dataset_id=args.dataset_id,
        source_revision=args.source_revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "catalog_sha256": catalog["catalog_sha256"],
                "sequence_count": catalog["sequence_count"],
                "annotation_count": catalog["annotation_count"],
                "track_count": catalog["track_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
