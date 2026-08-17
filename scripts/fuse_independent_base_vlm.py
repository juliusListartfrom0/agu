#!/usr/bin/env python3
"""Apply one frozen non-learned rule to AGU v3 and independent VLM outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.analysis.independent_base_vlm_fusion import (
    fuse_base_and_vlm_predictions,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--vlm-predictions", type=Path, required=True)
    parser.add_argument(
        "--rule",
        choices=("base_only", "vlm_only", "both_confirm", "either_confirms"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifact = fuse_base_and_vlm_predictions(
        plan=json.loads(args.plan.read_text(encoding="utf-8")),
        base_predictions=json.loads(
            args.base_predictions.read_text(encoding="utf-8")
        ),
        vlm_predictions=json.loads(
            args.vlm_predictions.read_text(encoding="utf-8")
        ),
        rule=args.rule,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rule": args.rule,
                "artifact_sha256": artifact["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
