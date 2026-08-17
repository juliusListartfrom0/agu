#!/usr/bin/env python3
"""Select a deterministic truth-free candidate probe for model calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.face_gallery import load_face_gallery  # noqa: E402
from app.analysis.official_evaluation import (  # noqa: E402
    seal_raw_only_predictions,
    verify_raw_only_bundle,
)
from app.analysis.schemas import (  # noqa: E402
    GameEventResponse,
    RawOnlyPredictionBundleResponse,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-type", action="append", required=True)
    parser.add_argument("--maximum-per-type", type=int, default=9)
    parser.add_argument(
        "--face-gallery",
        type=Path,
        help="Select only events with a traditional-model player candidate in this sealed gallery.",
    )
    parser.add_argument(
        "--include-related-events",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include causal parent events referenced by the selected candidates.",
    )
    parser.add_argument(
        "--unresolved-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Select only needs-review candidates whose outcome is still unknown",
    )
    return parser.parse_args()


def select_probe_events(
    events: Sequence[GameEventResponse],
    *,
    event_types: Sequence[str],
    maximum_per_type: int,
    unresolved_only: bool = False,
    allowed_player_ids: frozenset[str] | None = None,
) -> list[GameEventResponse]:
    """Select strong candidates across the full timeline without label access."""

    if maximum_per_type <= 0:
        raise ValueError("maximum_per_type must be positive")
    selected: list[GameEventResponse] = []
    for event_type in dict.fromkeys(event_types):
        candidates = sorted(
            (
                event
                for event in events
                if event.event_type == event_type
                and (
                    allowed_player_ids is None
                    or bool(_event_player_candidates(event) & allowed_player_ids)
                )
                and (
                    not unresolved_only
                    or (event.status == "needs_review" and event.outcome in {None, "unknown"})
                )
            ),
            key=lambda event: (event.start_frame, event.end_frame, event.event_id),
        )
        if len(candidates) <= maximum_per_type:
            selected.extend(candidates)
            continue
        for index in range(maximum_per_type):
            start = index * len(candidates) // maximum_per_type
            end = (index + 1) * len(candidates) // maximum_per_type
            selected.append(max(candidates[start:end], key=_probe_evidence_score))
    return sorted(selected, key=lambda event: (event.start_frame, event.event_type, event.event_id))


def build_probe_bundle(
    source: RawOnlyPredictionBundleResponse,
    *,
    video_paths: Sequence[Path],
    event_types: Sequence[str],
    maximum_per_type: int,
    unresolved_only: bool = False,
    allowed_player_ids: frozenset[str] | None = None,
    include_related_events: bool = False,
) -> RawOnlyPredictionBundleResponse:
    verified = verify_raw_only_bundle(source)
    events = select_probe_events(
        verified.events,
        event_types=event_types,
        maximum_per_type=maximum_per_type,
        unresolved_only=unresolved_only,
        allowed_player_ids=allowed_player_ids,
    )
    if include_related_events:
        events = _with_related_events(events, verified.events)
    return seal_raw_only_predictions(
        game_id=f"{verified.game_id}-truth-free-probe",
        raw_video_paths=video_paths,
        events=events,
        config={
            "pipeline": "truth_free_temporal_probe_v1",
            "source_candidate_bundle_sha256": verified.bundle_sha256,
            "event_types": list(dict.fromkeys(event_types)),
            "maximum_per_type": maximum_per_type,
            "selection_uses_ground_truth": False,
            "unresolved_only": unresolved_only,
            "registered_candidate_only": allowed_player_ids is not None,
            "include_related_events": include_related_events,
        },
        model_provenance={
            "producer": "agu",
            "artifact_role": "candidate_probe_only",
            "source_candidate_bundle_sha256": verified.bundle_sha256,
            "candidate_backend": verified.model_provenance.get("candidate_backend", ""),
            "registered_candidate_only": str(allowed_player_ids is not None).lower(),
            "include_related_events": str(include_related_events).lower(),
        },
    )


def _with_related_events(
    selected: Sequence[GameEventResponse],
    all_events: Sequence[GameEventResponse],
) -> list[GameEventResponse]:
    by_id = {event.event_id: event for event in all_events}
    included = {event.event_id: event for event in selected}
    pending = [event_id for event in selected for event_id in event.related_event_ids]
    while pending:
        event_id = pending.pop()
        if event_id in included or event_id not in by_id:
            continue
        related = by_id[event_id]
        included[event_id] = related
        pending.extend(related.related_event_ids)
    return sorted(included.values(), key=lambda event: (event.start_frame, event.event_type, event.event_id))


def _event_player_candidates(event: GameEventResponse) -> frozenset[str]:
    candidates = {
        str(value)
        for value in (event.primary_player_id, event.secondary_player_id)
        if value
    }
    for evidence in event.evidence:
        candidates.update(
            str(value)
            for value in (evidence.details.get("candidate_player_ids") or [])
            if value
        )
    return frozenset(candidates)


def _probe_evidence_score(event: GameEventResponse) -> tuple[float, ...]:
    details = [evidence.details for evidence in event.evidence]
    hit_count = max((int(item.get("hit_count") or 0) for item in details), default=0)
    approach_points = max(
        (int(item.get("maximum_approach_point_count") or 0) for item in details),
        default=0,
    )
    approach_rise = max(
        (float(item.get("maximum_approach_rise_px") or 0.0) for item in details),
        default=0.0,
    )
    resolved_trajectory = float(event.outcome in {"made", "missed"})
    return (
        resolved_trajectory,
        float(hit_count),
        float(approach_points),
        approach_rise,
        event.confidence,
        -float(event.start_frame),
    )


def main() -> int:
    args = parse_args()
    source = RawOnlyPredictionBundleResponse.model_validate_json(
        args.candidate_bundle.read_text(encoding="utf-8")
    )
    allowed_player_ids = None
    if args.face_gallery:
        gallery = load_face_gallery(args.face_gallery)
        allowed_player_ids = frozenset(entry.person_id for entry in gallery.entries)
    result = build_probe_bundle(
        source,
        video_paths=args.video,
        event_types=args.event_type,
        maximum_per_type=args.maximum_per_type,
        unresolved_only=args.unresolved_only,
        allowed_player_ids=allowed_player_ids,
        include_related_events=args.include_related_events,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "bundle_sha256": result.bundle_sha256,
                "event_count": len(result.events),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
