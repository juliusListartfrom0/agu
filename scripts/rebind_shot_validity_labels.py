#!/usr/bin/env python3
"""Rebind training-only shot labels to an evidence-enriched candidate bundle."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--enriched-bundle", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def rebind_labels(
    source_bundle_payload: dict[str, Any],
    enriched_bundle_payload: dict[str, Any],
    labels: dict[str, Any],
) -> dict[str, Any]:
    """Move labels only when candidate identity and raw media are unchanged."""

    source = verify_raw_only_bundle(source_bundle_payload)
    enriched = verify_raw_only_bundle(enriched_bundle_payload)
    if labels.get("schema_version") != SHOT_VALIDITY_LABEL_SCHEMA:
        raise ValueError("unsupported shot-validity labels")
    if labels.get("runtime_consumable") is not False:
        raise ValueError("training annotations must not be runtime-consumable")
    if labels.get("candidate_bundle_sha256") != source.bundle_sha256:
        raise ValueError("labels are not bound to the source candidate bundle")
    if (
        len(source.raw_videos) != 1
        or labels.get("source_video_sha256") != source.raw_videos[0].sha256
    ):
        raise ValueError("labels are not bound to the source raw video")
    if source.game_id != enriched.game_id or source.raw_videos != enriched.raw_videos:
        raise ValueError("enriched bundle changed game or raw-video identity")

    source_events = {event.event_id: event for event in source.events}
    enriched_events = {event.event_id: event for event in enriched.events}
    if source_events.keys() != enriched_events.keys():
        raise ValueError("enriched bundle changed the candidate event set")
    for event_id, source_event in source_events.items():
        enriched_event = enriched_events[event_id]
        source_identity = source_event.model_dump(exclude={"evidence", "reason"})
        enriched_identity = enriched_event.model_dump(exclude={"evidence", "reason"})
        if source_identity != enriched_identity:
            raise ValueError(f"enriched bundle changed candidate identity: {event_id}")

    examples = labels.get("examples", [])
    labeled_ids = [str(row.get("event_id") or "") for row in examples]
    if (
        not labeled_ids
        or len(set(labeled_ids)) != len(labeled_ids)
        or not set(labeled_ids).issubset(source_events)
        or any(not isinstance(row.get("event_present"), bool) for row in examples)
    ):
        raise ValueError("labels contain absent or empty candidate IDs")
    rebound = dict(labels)
    rebound["candidate_bundle_sha256"] = enriched.bundle_sha256
    rebound["evidence_rebind"] = {
        "source_candidate_bundle_sha256": source.bundle_sha256,
        "enriched_candidate_bundle_sha256": enriched.bundle_sha256,
        "invariant": "same_raw_video_and_candidate_identity_evidence_only",
    }
    return rebound


def main() -> int:
    args = parse_args()
    payload = rebind_labels(
        json.loads(args.source_bundle.read_text(encoding="utf-8")),
        json.loads(args.enriched_bundle.read_text(encoding="utf-8")),
        json.loads(args.labels.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["evidence_rebind"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
