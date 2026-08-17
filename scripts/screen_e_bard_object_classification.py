#!/usr/bin/env python3
"""Audit the downloaded E-BARD object-role classification archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.analysis.ball_candidate_review import seal_artifact
from app.analysis.e_bard_object_classification import (
    summarize_object_classification_archive,
)

SOURCE_URL = "https://huggingface.co/datasets/GabrieleGiudici/E-BARD-ObjectClassification"
SOURCE_REVISION = "a0919d0bfbf0ff57502b1f2ec515899cc9c5e818"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_screen_artifact(archive: Path) -> dict[str, object]:
    summary = summarize_object_classification_archive(archive)
    return seal_artifact(
        {
            "schema_version": "agu.e-bard-object-classification-screen.v1",
            "purpose": "offline_object_role_vlm_pretraining_screen",
            "source_url": SOURCE_URL,
            "source_revision": SOURCE_REVISION,
            "data_license": "CC-BY-4.0",
            "archive_sha256": summary["archive_sha256"],
            "archive_size_bytes": summary["archive_size_bytes"],
            "retention": {
                "archive_retained": True,
                "extracted_duplicate_retained": False,
                "reason": "small licensed crop archive; retain one hash-bound copy",
            },
            "training_scope": {
                "offline_object_role_crop_classifier": bool(
                    summary["offline_object_role_training_ready"]
                ),
                "continuous_ball_tracking": False,
                "cross_game_benchmark": bool(summary["cross_game_benchmark_ready"]),
                "runtime_consumable": False,
            },
            "accepted_for_offline_object_role_training": bool(
                summary["offline_object_role_training_ready"]
            ),
            "accepted_for_cross_game_benchmark": bool(
                summary["cross_game_benchmark_ready"]
            ),
            "audit": summary,
            "runtime_consumable": False,
            "codex_runtime_answer_used": False,
        }
    )


def main() -> int:
    args = parse_args()
    artifact = build_screen_artifact(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "accepted_for_offline_object_role_training": artifact[
                    "accepted_for_offline_object_role_training"
                ],
                "accepted_for_cross_game_benchmark": artifact[
                    "accepted_for_cross_game_benchmark"
                ],
                "archive_sha256": artifact["archive_sha256"],
                "artifact_sha256": artifact["artifact_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
