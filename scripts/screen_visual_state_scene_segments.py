#!/usr/bin/env python3
"""Screen cut-aware DINOv2 visual-state features with strict game-held splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.analysis.pbp_visual_state_review import (  # noqa: E402
    verify_visual_state_label_corrections,
    verify_visual_state_review_plan,
)
from app.analysis.visual_state_scene_screen import (  # noqa: E402
    build_visual_state_scene_examples,
    screen_visual_state_scene_examples,
)
from app.analysis.visual_state_temporal_screen import (  # noqa: E402
    build_visual_state_temporal_examples,
)


def run_screen(
    *,
    pre_anchor_path: Path,
    anchor_path: Path,
    post_anchor_path: Path,
    wide_anchor_path: Path,
    base_plan_path: Path,
    base_corrections_path: Path,
    followup_plan_path: Path,
    followup_corrections_path: Path,
    transition_paths: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    base_plan = verify_visual_state_review_plan(_read_json(base_plan_path))
    base_corrections = verify_visual_state_label_corrections(
        _read_json(base_corrections_path),
        plan=base_plan,
    )
    followup_plan = verify_visual_state_review_plan(_read_json(followup_plan_path))
    followup_corrections = verify_visual_state_label_corrections(
        _read_json(followup_corrections_path),
        plan=followup_plan,
    )
    if (
        followup_plan.get("parent_plan_sha256") != base_plan["artifact_sha256"]
        or followup_plan.get("parent_corrections_sha256")
        != base_corrections["artifact_sha256"]
        or followup_plan.get("sealed_blind_video_sha256s")
        != base_plan["sealed_blind_video_sha256s"]
    ):
        raise ValueError("visual-state follow-up chain is invalid")

    temporal_examples, provenance = build_visual_state_temporal_examples(
        pre_anchor_artifact=_read_json(pre_anchor_path),
        anchor_artifact=_read_json(anchor_path),
        post_anchor_artifact=_read_json(post_anchor_path),
        wide_anchor_artifact=_read_json(wide_anchor_path),
        base_decisions=base_corrections["decisions"],
        followup_decisions=followup_corrections["decisions"],
    )
    expected_event_keys = {
        (
            str(row["source_video_sha256"]),
            str(row["candidate_bundle_sha256"]),
            str(row["event_id"]),
        )
        for row in base_plan["examples"]
    }
    scene_examples, transition_hashes = build_visual_state_scene_examples(
        temporal_examples=temporal_examples,
        transition_artifacts=[_read_json(path) for path in transition_paths],
        expected_event_keys=expected_event_keys,
        sealed_blind_video_sha256s=base_plan["sealed_blind_video_sha256s"],
    )
    provenance["transition_artifact_sha256s"] = transition_hashes
    artifact = screen_visual_state_scene_examples(
        examples=scene_examples,
        provenance=provenance,
        label_correction_artifact_sha256s=[
            base_corrections["artifact_sha256"],
            followup_corrections["artifact_sha256"],
        ],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return artifact


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path.name}")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-anchor", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--post-anchor", type=Path, required=True)
    parser.add_argument("--wide-anchor", type=Path, required=True)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--base-corrections", type=Path, required=True)
    parser.add_argument("--followup-plan", type=Path, required=True)
    parser.add_argument("--followup-corrections", type=Path, required=True)
    parser.add_argument(
        "--transition-artifact",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    artifact = run_screen(
        pre_anchor_path=arguments.pre_anchor,
        anchor_path=arguments.anchor,
        post_anchor_path=arguments.post_anchor,
        wide_anchor_path=arguments.wide_anchor,
        base_plan_path=arguments.base_plan,
        base_corrections_path=arguments.base_corrections,
        followup_plan_path=arguments.followup_plan,
        followup_corrections_path=arguments.followup_corrections,
        transition_paths=arguments.transition_artifact,
        output_path=arguments.output,
    )
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
