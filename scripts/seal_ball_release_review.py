#!/usr/bin/env python3
"""Seal Codex offline ball-release decisions against a review-sheet plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.ball_release_annotation import (  # noqa: E402
    seal_ball_release_review,
    verify_ball_release_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sheet-manifest", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def seal_review(
    *,
    plan_path: Path,
    sheet_manifest_path: Path,
    decisions_path: Path,
) -> dict[str, object]:
    plan = verify_ball_release_plan(_read_json(plan_path))
    sheets = _read_json(sheet_manifest_path)
    if sheets.get("schema_version") != "agu.ball-release-review-sheets.v1":
        raise ValueError("unsupported ball-release sheet manifest")
    if sheets.get("runtime_consumable") is not False:
        raise ValueError("review sheets must remain training-only")
    if sheets.get("plan_sha256") != plan["artifact_sha256"]:
        raise ValueError("review sheets do not match the plan")
    decisions = _read_json(decisions_path)
    if decisions.get("schema_version") != "agu.ball-release-codex-decisions.v1":
        raise ValueError("unsupported ball-release decision schema")
    if decisions.get("runtime_consumable") is not False:
        raise ValueError("Codex decisions must remain training-only")
    if decisions.get("plan_sha256") != plan["artifact_sha256"]:
        raise ValueError("Codex decisions do not match the plan")

    planned_by_id = {str(row["review_id"]): row for row in plan["examples"]}
    sheets_by_id = {str(row["review_id"]): row for row in sheets["records"]}
    decisions_by_id = {
        str(row["review_id"]): row for row in decisions.get("decisions", [])
    }
    if (
        set(sheets_by_id) != set(planned_by_id)
        or set(decisions_by_id) != set(planned_by_id)
        or len(decisions_by_id) != len(decisions.get("decisions", []))
    ):
        raise ValueError("sheets and decisions must exactly cover the plan")

    reviews = []
    for review_id, planned in planned_by_id.items():
        decision = decisions_by_id[review_id]
        sheet = sheets_by_id[review_id]
        frame_numbers = [int(value) for value in planned["frame_numbers"]]
        visible = _frame_subset(decision, "ball_visible_frames", frame_numbers)
        not_visible = _frame_subset(
            decision, "ball_not_visible_frames", frame_numbers
        )
        path_correct = _frame_subset(
            decision, "selected_path_correct_frames", frame_numbers
        )
        path_incorrect = _frame_subset(
            decision, "selected_path_incorrect_frames", frame_numbers
        )
        if visible & not_visible:
            raise ValueError(f"{review_id} has conflicting ball visibility")
        if path_correct & path_incorrect:
            raise ValueError(f"{review_id} has conflicting path decisions")
        path_observations = {
            int(row["planned_frame"]): row
            for row in sheet["selected_path_observations"]
        }
        if set(path_correct | path_incorrect) - {
            frame
            for frame, row in path_observations.items()
            if row.get("detection_frame") is not None
        }:
            raise ValueError(
                f"{review_id} labels a selected path that was not rendered"
            )
        frame_observations = []
        for frame in frame_numbers:
            path_row = path_observations[frame]
            frame_observations.append(
                {
                    "frame": frame,
                    "ball_visible": (
                        True
                        if frame in visible
                        else False
                        if frame in not_visible
                        else None
                    ),
                    "selected_path_is_ball": (
                        True
                        if frame in path_correct
                        else False
                        if frame in path_incorrect
                        else None
                    ),
                    "ball_bbox": (
                        path_row["bbox"] if frame in path_correct else None
                    ),
                }
            )
        reviews.append(
            {
                "source_video_sha256": planned["source_video_sha256"],
                "candidate_bundle_sha256": planned[
                    "candidate_bundle_sha256"
                ],
                "event_id": planned["event_id"],
                "release_observed": decision["release_observed"],
                "ball_moves_toward_rim": decision["ball_moves_toward_rim"],
                "rim_arrival_observed": decision["rim_arrival_observed"],
                "free_throw_formation": decision["free_throw_formation"],
                "replay_or_stoppage": decision["replay_or_stoppage"],
                "review_confidence": decision["review_confidence"],
                "review_notes": str(decision.get("review_notes") or ""),
                "frame_observations": frame_observations,
            }
        )
    return seal_ball_release_review(
        {
            "plan_sha256": plan["artifact_sha256"],
            "review_sheet_artifact_sha256": sheets["artifact_sha256"],
            "reviewer": "codex_offline_visual_review",
            "reviews": reviews,
        },
        plan=plan,
    )


def _frame_subset(
    decision: dict[str, Any], field: str, planned: list[int]
) -> set[int]:
    values = decision.get(field, [])
    if not isinstance(values, list):
        raise ValueError(f"{field} must be a list")
    output = {int(value) for value in values}
    if output - set(planned):
        raise ValueError(f"{field} contains an unplanned frame")
    return output


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return payload


def main() -> int:
    args = parse_args()
    artifact = seal_review(
        plan_path=args.plan,
        sheet_manifest_path=args.sheet_manifest,
        decisions_path=args.decisions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "reviews": len(artifact["reviews"]),
                "artifact_sha256": artifact["artifact_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
