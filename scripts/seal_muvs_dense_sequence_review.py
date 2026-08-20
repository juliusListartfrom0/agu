#!/usr/bin/env python3
"""Seal manually authored, source-only MUVS dense-sequence decisions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.muvs_dense_sequence import (  # noqa: E402
    seal_muvs_dense_sequence_review,
    verify_muvs_dense_sequence_frames,
    verify_muvs_dense_sequence_plan,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = verify_muvs_dense_sequence_plan(_read_json(args.plan))
    frames = verify_muvs_dense_sequence_frames(
        _read_json(args.frames),
        plan=plan,
    )
    review = seal_muvs_dense_sequence_review(
        _read_json(args.decisions),
        plan=plan,
        frames=frames,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    counts = Counter(row["state"] for row in review["decisions"])
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sample_count": len(review["decisions"]),
                "state_counts": dict(sorted(counts.items())),
                "artifact_sha256": review["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
