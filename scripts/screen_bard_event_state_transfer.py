#!/usr/bin/env python3
"""Fit on BARD only and seal label-free AGU target predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.analysis.bard_event_state_transfer import (
    train_bard_transfer_predictor,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bard-embeddings", type=Path, required=True)
    parser.add_argument("--target-embeddings", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifact = train_bard_transfer_predictor(
        bard_embedding_artifact=json.loads(
            args.bard_embeddings.read_text(encoding="utf-8")
        ),
        target_embedding_artifact=json.loads(
            args.target_embeddings.read_text(encoding="utf-8")
        ),
        configurations=[
            {"representation": representation, "c": c}
            for representation in ("mean", "mean_std", "quarters")
            for c in (0.01, 0.1, 1.0)
        ],
        seed=args.seed,
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
                "source_oof_metrics": artifact["source_oof_metrics"],
                "selected_configuration": artifact[
                    "selected_configuration"
                ],
                "artifact_sha256": artifact["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
