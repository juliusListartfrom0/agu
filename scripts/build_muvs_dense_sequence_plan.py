#!/usr/bin/env python3
"""Build a hash-bound, label-hidden MUVS multi-frame sequence plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.muvs_dense_sequence import (  # noqa: E402
    build_muvs_dense_sequence_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-event", type=int, default=2)
    parser.add_argument("--expected-basketball-events", type=int, default=12)
    parser.add_argument(
        "--frame-offset-sec",
        action="append",
        type=float,
        default=None,
        help="Fixed offset inside each three-second fragment; repeat at least 3 times.",
    )
    parser.add_argument(
        "--event-id",
        action="append",
        default=None,
        help="Limit sampling to a named event; may be repeated.",
    )
    parser.add_argument(
        "--exclude-plan",
        action="append",
        type=Path,
        default=[],
        help="Prior sealed two-/multi-frame plan whose source coordinates are excluded.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    excluded_plans = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.exclude_plan
    ]
    if not all(isinstance(payload, dict) for payload in excluded_plans):
        raise ValueError("excluded MUVS plans must be JSON objects")
    offsets = (
        tuple(args.frame_offset_sec)
        if args.frame_offset_sec is not None
        else (0.25, 0.75, 1.25, 1.75, 2.25)
    )
    artifact = build_muvs_dense_sequence_plan(
        args.data_root,
        samples_per_event=args.samples_per_event,
        expected_basketball_events=args.expected_basketball_events,
        frame_offsets_sec=offsets,
        included_event_ids=args.event_id,
        excluded_plans=excluded_plans,
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
                "event_count": artifact["event_count"],
                "sample_count": artifact["sample_count"],
                "sequence_length": artifact["sequence_length"],
                "artifact_sha256": artifact["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
