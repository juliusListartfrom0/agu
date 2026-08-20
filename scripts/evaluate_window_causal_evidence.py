#!/usr/bin/env python3
"""Evaluate window-scoped causal evidence against training-only review labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.official_evaluation import verify_raw_only_bundle  # noqa: E402
from app.analysis.shot_validity import SHOT_VALIDITY_LABEL_SCHEMA  # noqa: E402

CAUSAL_EVIDENCE_VERSION = "agu_window_ball_rim_pose_v1"


def evaluate_causal_evidence(
    bundle_payload: dict[str, Any],
    labels: dict[str, Any],
) -> dict[str, float | int]:
    bundle = verify_raw_only_bundle(bundle_payload)
    if labels.get("schema_version") != SHOT_VALIDITY_LABEL_SCHEMA:
        raise ValueError("unsupported shot-validity labels")
    if labels.get("runtime_consumable") is not False:
        raise ValueError("causal evaluation requires training-only labels")
    if labels.get("candidate_bundle_sha256") != bundle.bundle_sha256:
        raise ValueError("labels are not bound to the evaluated candidate bundle")
    if (
        len(bundle.raw_videos) != 1
        or labels.get("source_video_sha256") != bundle.raw_videos[0].sha256
    ):
        raise ValueError("labels are not bound to the evaluated raw video")
    events = {event.event_id: event for event in bundle.events}
    true_positive = false_positive = false_negative = true_negative = 0
    seen: set[str] = set()
    for row in labels.get("examples", []):
        event_id = str(row.get("event_id") or "")
        present = row.get("event_present")
        if event_id in seen or event_id not in events or not isinstance(present, bool):
            raise ValueError("labels contain duplicate, absent or invalid examples")
        seen.add(event_id)
        predicted = any(
            evidence.kind == "ball_rim_proximity_cluster"
            and evidence.details.get("causal_evidence_version")
            == CAUSAL_EVIDENCE_VERSION
            for evidence in events[event_id].evidence
        )
        true_positive += int(predicted and present)
        false_positive += int(predicted and not present)
        false_negative += int(not predicted and present)
        true_negative += int(not predicted and not present)
    if not seen:
        raise ValueError("causal evaluation requires labeled examples")
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    return {
        "labeled_count": len(seen),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()
    metrics = evaluate_causal_evidence(
        json.loads(args.bundle.read_text(encoding="utf-8")),
        json.loads(args.labels.read_text(encoding="utf-8")),
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
