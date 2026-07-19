#!/usr/bin/env python3
"""Evaluate a sealed raw-only official bundle after inference has completed."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.official_evaluation import (  # noqa: E402
    AUTOMATIC_CONFIRMED_STATUSES,
    REVIEWED_CONFIRMED_STATUSES,
    RawOnlyEvaluationError,
    evaluate_identity_aligned_strict_events,
    evaluate_official_tiers,
    load_strict_truth_csv,
    strict_events_from_game_events,
    verify_raw_only_bundle,
    verify_agu_autonomous_bundle,
)
from app.analysis.schemas import RawOnlyPredictionBundleResponse  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--tolerance-frames", type=int, default=90)
    parser.add_argument("--identity-alignment", type=Path)
    parser.add_argument(
        "--require-agu-autonomous",
        action="store_true",
        help="Reject Codex/human/reference-derived predictions and score only AGU automatic events",
    )
    parser.add_argument(
        "--event-type",
        action="append",
        dest="event_types",
        help="Explicit truth-covered event type; repeat only for a declared partial-category benchmark",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def evaluate_bundle(
    *,
    bundle_path: Path,
    truth_path: Path,
    fps: float,
    tolerance_frames: int,
    identity_alignment_path: Path | None = None,
    event_types: Sequence[str] | None = None,
    require_agu_autonomous: bool = False,
) -> dict[str, object]:
    if fps <= 0 or tolerance_frames < 0:
        raise ValueError("fps must be positive and tolerance must be non-negative")
    bundle = verify_raw_only_bundle(
        RawOnlyPredictionBundleResponse.model_validate_json(bundle_path.read_text(encoding="utf-8"))
    )
    if require_agu_autonomous:
        bundle = verify_agu_autonomous_bundle(bundle)
    truth_sha256 = _file_sha256(truth_path)
    all_truth = load_strict_truth_csv(truth_path, source_video_id="video_001", fps=fps)
    scope = tuple(dict.fromkeys(str(item) for item in (event_types or ())))
    truth = [event for event in all_truth if not scope or event.event_type in scope]
    evaluated_events = [event for event in bundle.events if not scope or event.event_type in scope]
    report: dict[str, object] = {
        "schema_version": "agu.official-evaluation.v1",
        "prediction_bundle_sha256": bundle.bundle_sha256,
        "truth_sha256": truth_sha256,
        "truth_total_event_count": len(all_truth),
        "truth_event_count": len(truth),
        "evaluation_scope": {
            "event_types": list(scope) if scope else "all",
            "explicit_partial_category_scope": bool(scope),
        },
        "acceptance_mode": "agu_autonomous" if require_agu_autonomous else "audit_all_tiers",
        "tiers": evaluate_official_tiers(truth, evaluated_events, tolerance_frames=tolerance_frames),
    }
    if identity_alignment_path is not None:
        alignment = json.loads(identity_alignment_path.read_text(encoding="utf-8"))
        if not isinstance(alignment, dict) or alignment.get("schema_version") != "agu.identity-alignment.v1":
            raise RawOnlyEvaluationError("unsupported identity alignment schema")
        if alignment.get("prediction_bundle_sha256") != bundle.bundle_sha256:
            raise RawOnlyEvaluationError("identity alignment prediction bundle hash mismatch")
        if alignment.get("truth_sha256") != truth_sha256:
            raise RawOnlyEvaluationError("identity alignment truth hash mismatch")
        players = alignment.get("player_id_map")
        teams = alignment.get("team_id_map")
        if not isinstance(players, dict) or not isinstance(teams, dict):
            raise RawOnlyEvaluationError("identity alignment maps must be objects")
        reviewed_events = strict_events_from_game_events(
            event for event in evaluated_events if event.status in REVIEWED_CONFIRMED_STATUSES
        )
        report["reviewed_identity_aligned_strict"] = evaluate_identity_aligned_strict_events(
            truth,
            reviewed_events,
            player_id_map={str(key): str(value) for key, value in players.items()},
            team_id_map={str(key): str(value) for key, value in teams.items()},
            tolerance_frames=tolerance_frames,
        )
        if require_agu_autonomous:
            automatic_events = strict_events_from_game_events(
                event for event in evaluated_events if event.status in AUTOMATIC_CONFIRMED_STATUSES
            )
            report["autonomous_identity_aligned_strict"] = evaluate_identity_aligned_strict_events(
                truth,
                automatic_events,
                player_id_map={str(key): str(value) for key, value in players.items()},
                team_id_map={str(key): str(value) for key, value in teams.items()},
                tolerance_frames=tolerance_frames,
            )
        report["identity_alignment_file_sha256"] = _file_sha256(identity_alignment_path)
    return report


def main() -> int:
    args = parse_args()
    report = evaluate_bundle(
        bundle_path=args.bundle,
        truth_path=args.truth,
        fps=args.fps,
        tolerance_frames=args.tolerance_frames,
        identity_alignment_path=args.identity_alignment,
        event_types=args.event_types,
        require_agu_autonomous=args.require_agu_autonomous,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = (
        report.get("autonomous_identity_aligned_strict")
        or report.get("reviewed_identity_aligned_strict")
        or report["tiers"]
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
