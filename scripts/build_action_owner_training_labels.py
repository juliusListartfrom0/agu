#!/usr/bin/env python3
"""Materialize Codex-reviewed action-owner labels from non-benchmark candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.official_evaluation import verify_raw_only_bundle  # noqa: E402
from app.analysis.schemas import RawOnlyPredictionBundleResponse  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_action_owner_labels(
    *,
    candidate_bundle_path: Path,
    video_path: Path,
    selections_path: Path,
) -> dict[str, Any]:
    bundle = verify_raw_only_bundle(
        RawOnlyPredictionBundleResponse.model_validate_json(
            candidate_bundle_path.read_text(encoding="utf-8")
        )
    )
    if len(bundle.raw_videos) != 1:
        raise ValueError("action-owner labels currently require one raw video")
    asset = bundle.raw_videos[0]
    if asset.filename != video_path.name or asset.sha256 != _file_sha256(video_path):
        raise ValueError("candidate bundle does not match the selected raw video")
    selections = json.loads(selections_path.read_text(encoding="utf-8"))
    if selections.get("schema_version") != "agu.codex-action-owner-selections.v1":
        raise ValueError("unsupported Codex action-owner selection schema")
    if selections.get("producer") != "codex" or selections.get("purpose") != "model_training_only":
        raise ValueError("action-owner selections must be explicit Codex training labels")
    events = {event.event_id: event for event in bundle.events}
    examples = []
    for selection in selections.get("selections", []):
        event_id = str(selection.get("event_id") or "")
        event = events.get(event_id)
        if event is None or event.event_type != "field_goal_attempt":
            raise ValueError(f"selection does not name a shot candidate: {event_id}")
        observations = next(
            (
                evidence.details.get("candidate_player_observations")
                for evidence in event.evidence
                if evidence.details.get("candidate_player_observations")
            ),
            None,
        )
        if not isinstance(observations, list):
            raise ValueError(f"shot candidate has no player observations: {event_id}")
        observations = _bounded_observations(observations, selection, event_id=event_id)
        positive = str(selection.get("positive_player_id") or "")
        candidate_ids = {str(item.get("player_id") or "") for item in observations}
        if positive not in candidate_ids:
            raise ValueError(f"selected player is not grounded in candidate observations: {event_id}")
        examples.append(
            {
                "event_id": event_id,
                "event_type": "shot_release_actor",
                "anchor_frame": int(selection["anchor_frame"]),
                "positive_player_id": positive,
                "candidate_player_observations": observations,
                "review_note": str(selection.get("review_note") or ""),
            }
        )
    if not examples:
        raise ValueError("at least one Codex-reviewed training selection is required")
    return {
        "schema_version": "agu.action-ownership-labels.v1",
        "purpose": "model_training_only",
        "producer": "codex",
        "runtime_consumable": False,
        "source_video_sha256": asset.sha256,
        "candidate_bundle_sha256": bundle.bundle_sha256,
        "selection_file_sha256": _file_sha256(selections_path),
        "examples": examples,
    }


def _bounded_observations(
    observations: list[dict[str, Any]],
    selection: dict[str, Any],
    *,
    event_id: str,
) -> list[dict[str, Any]]:
    start_value = selection.get("observation_start_frame")
    end_value = selection.get("observation_end_frame")
    if start_value is None and end_value is None:
        return observations
    if start_value is None or end_value is None:
        raise ValueError(f"selection observation bounds must be paired: {event_id}")
    start_frame = int(start_value)
    end_frame = int(end_value)
    anchor_frame = int(selection["anchor_frame"])
    if start_frame > anchor_frame or anchor_frame > end_frame:
        raise ValueError(f"selection observation bounds must contain anchor: {event_id}")
    bounded = [
        item
        for item in observations
        if start_frame <= int(item.get("frame", -1)) <= end_frame
    ]
    if not bounded:
        raise ValueError(f"selection observation bounds contain no candidates: {event_id}")
    return bounded


def main() -> int:
    args = parse_args()
    payload = build_action_owner_labels(
        candidate_bundle_path=args.candidate_bundle,
        video_path=args.video,
        selections_path=args.selections,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"examples": len(payload["examples"]), "source": payload["source_video_sha256"]}, indent=2))
    return 0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
