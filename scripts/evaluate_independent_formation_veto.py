#!/usr/bin/env python3
"""Evaluate a fixed independent-formation veto over frozen OOF predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.independent_formation_vlm import (  # noqa: E402
    evaluate_fixed_formation_veto,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formation-predictions", type=Path, required=True)
    parser.add_argument("--upstream-screen", type=Path, required=True)
    parser.add_argument("--variant", default="base+broadcast_raw")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    upstream = json.loads(args.upstream_screen.read_text(encoding="utf-8"))
    variants = {
        str(row.get("name") or ""): row
        for row in upstream.get("variants", [])
    }
    variant = variants.get(args.variant)
    if variant is None:
        raise ValueError(f"upstream variant is missing: {args.variant}")
    evaluation = evaluate_fixed_formation_veto(
        formation_predictions=json.loads(
            args.formation_predictions.read_text(encoding="utf-8")
        ),
        upstream_artifact_sha256=str(upstream["artifact_sha256"]),
        upstream_predictions=variant["oof_predictions"],
        threshold=float(variant["threshold"]),
        upstream_variant=args.variant,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evaluation, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "artifact_sha256": evaluation["artifact_sha256"],
                "baseline": evaluation["baseline"],
                "formation_veto": evaluation["formation_veto"],
                "accepted": evaluation["accepted"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
