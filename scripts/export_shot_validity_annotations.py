#!/usr/bin/env python3
"""Export a truth-free review template for sealed shot candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.official_evaluation import verify_raw_only_bundle  # noqa: E402
from app.analysis.shot_validity import (  # noqa: E402
    SHOT_VALIDITY_LABEL_SCHEMA,
    extract_shot_validity_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def export_annotation_template(candidate_bundle_path: Path) -> dict[str, object]:
    bundle = verify_raw_only_bundle(
        json.loads(candidate_bundle_path.read_text(encoding="utf-8"))
    )
    if len(bundle.raw_videos) != 1:
        raise ValueError("shot-validity annotation requires exactly one raw video")
    examples = []
    for event in bundle.events:
        if event.event_type != "field_goal_attempt":
            continue
        examples.append(
            {
                "event_id": event.event_id,
                "start_frame": event.start_frame,
                "end_frame": event.end_frame,
                "release_frame": event.release_frame,
                "event_present": None,
                "review_note": "",
                "raw_feature_snapshot": extract_shot_validity_features(event),
            }
        )
    return {
        "schema_version": SHOT_VALIDITY_LABEL_SCHEMA,
        "purpose": "training_annotation_only",
        "runtime_consumable": False,
        "source_video_sha256": bundle.raw_videos[0].sha256,
        "candidate_bundle_sha256": bundle.bundle_sha256,
        "instructions": (
            "Review the bound raw-video window and replace event_present=null with true "
            "only for a complete real shot attempt, otherwise false. Do not use this file "
            "during AGU runtime inference."
        ),
        "examples": examples,
    }


def main() -> int:
    args = parse_args()
    payload = export_annotation_template(args.candidate_bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"example_count": len(payload["examples"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
