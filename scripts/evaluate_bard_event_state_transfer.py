#!/usr/bin/env python3
"""Evaluate frozen BARD transfer predictions against delayed labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.analysis.bard_event_state_transfer import (
    evaluate_bard_transfer_predictions,
)
from app.analysis.causal_phase_dense_temporal import (
    verify_dense_causal_temporal_screen,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--acceptance-threshold", type=float, default=0.85)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = verify_dense_causal_temporal_screen(
        json.loads(args.reference.read_text(encoding="utf-8"))
    )
    artifact = evaluate_bard_transfer_predictions(
        prediction_artifact=json.loads(
            args.predictions.read_text(encoding="utf-8")
        ),
        reference_predictions=reference["predictions"],
        reference_artifact_sha256=reference["artifact_sha256"],
        acceptance_threshold=args.acceptance_threshold,
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
                "metrics": artifact["metrics"],
                "accepted": artifact["accepted"],
                "artifact_sha256": artifact["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
