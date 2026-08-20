#!/usr/bin/env python3
"""Derive a label-free target embedding set from frozen resolved IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.analysis.causal_phase_dense_temporal import (
    seal_dense_causal_frame_embeddings,
    verify_dense_causal_frame_embeddings,
    verify_dense_causal_temporal_screen,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--resolved-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    embeddings = verify_dense_causal_frame_embeddings(
        json.loads(args.embeddings.read_text(encoding="utf-8"))
    )
    reference = verify_dense_causal_temporal_screen(
        json.loads(args.resolved_reference.read_text(encoding="utf-8"))
    )
    allowed = {
        (
            str(row["source_video_sha256"]),
            str(row["event_id"]),
            str(row["phase_review_id"]),
        )
        for row in reference["predictions"]
    }
    selected = [
        row
        for row in embeddings["examples"]
        if (
            str(row["source_video_sha256"]),
            str(row["event_id"]),
            str(row["phase_review_id"]),
        )
        in allowed
    ]
    selected_keys = {
        (
            str(row["source_video_sha256"]),
            str(row["event_id"]),
            str(row["phase_review_id"]),
        )
        for row in selected
    }
    if selected_keys != allowed:
        raise ValueError(
            "resolved target IDs do not exactly join the dense embeddings"
        )
    artifact = seal_dense_causal_frame_embeddings(
        {
            **{
                key: value
                for key, value in embeddings.items()
                if key
                not in {
                    "artifact_sha256",
                    "examples",
                    "source_video_sha256s",
                    "purpose",
                }
            },
            "purpose": "training_only_bard_transfer_target_embeddings",
            "source_video_sha256s": sorted(
                {row["source_video_sha256"] for row in selected}
            ),
            "target_reference_artifact_sha256": reference[
                "artifact_sha256"
            ],
            "examples": selected,
        }
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
                "examples": len(selected),
                "artifact_sha256": artifact["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
