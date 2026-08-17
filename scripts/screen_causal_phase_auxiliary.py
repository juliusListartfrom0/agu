#!/usr/bin/env python3
"""Screen causal phase supervision with doubly game-disjoint stacking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.causal_phase_auxiliary_screen import (  # noqa: E402
    join_causal_phase_targets,
    screen_causal_phase_auxiliary_examples,
)
from app.analysis.causal_shot_phase_review import (  # noqa: E402
    verify_causal_shot_phase_review,
    verify_causal_shot_phase_review_plan,
)
from app.analysis.pbp_visual_state_review import (  # noqa: E402
    verify_visual_state_label_corrections,
    verify_visual_state_review_plan,
)
from app.analysis.visual_state_temporal_screen import (  # noqa: E402
    build_visual_state_temporal_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-anchor", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--post-anchor", type=Path, required=True)
    parser.add_argument("--wide-anchor", type=Path, required=True)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--base-corrections", type=Path, required=True)
    parser.add_argument("--followup-plan", type=Path, required=True)
    parser.add_argument("--followup-corrections", type=Path, required=True)
    parser.add_argument("--phase-plan", type=Path, required=True)
    parser.add_argument("--phase-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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
    phase_plan_path: Path,
    phase_review_path: Path,
) -> dict[str, Any]:
    base_plan = verify_visual_state_review_plan(_read_json(base_plan_path))
    base_corrections = verify_visual_state_label_corrections(
        _read_json(base_corrections_path),
        plan=base_plan,
    )
    followup_plan = verify_visual_state_review_plan(
        _read_json(followup_plan_path)
    )
    followup_corrections = verify_visual_state_label_corrections(
        _read_json(followup_corrections_path),
        plan=followup_plan,
    )
    if (
        followup_plan.get("parent_plan_sha256")
        != base_plan["artifact_sha256"]
        or followup_plan.get("parent_corrections_sha256")
        != base_corrections["artifact_sha256"]
    ):
        raise ValueError("visual-state correction chain is invalid")

    examples, provenance = build_visual_state_temporal_examples(
        pre_anchor_artifact=_read_json(pre_anchor_path),
        anchor_artifact=_read_json(anchor_path),
        post_anchor_artifact=_read_json(post_anchor_path),
        wide_anchor_artifact=_read_json(wide_anchor_path),
        base_decisions=base_corrections["decisions"],
        followup_decisions=followup_corrections["decisions"],
    )
    phase_plan = verify_causal_shot_phase_review_plan(
        _read_json(phase_plan_path)
    )
    phase_review = verify_causal_shot_phase_review(
        _read_json(phase_review_path),
        plan=phase_plan,
    )
    if (
        set(provenance["sealed_blind_video_sha256s"])
        != set(phase_plan["sealed_blind_video_sha256s"])
        or {row["source_video_sha256"] for row in examples}
        != set(phase_plan["source_video_sha256s"])
    ):
        raise ValueError("phase and temporal source provenance mismatch")
    joined = join_causal_phase_targets(examples, phase_review["reviews"])
    return screen_causal_phase_auxiliary_examples(
        examples=joined,
        provenance={
            **provenance,
            "label_correction_artifact_sha256s": [
                base_corrections["artifact_sha256"],
                followup_corrections["artifact_sha256"],
            ],
            "phase_review_artifact_sha256": phase_review[
                "artifact_sha256"
            ],
            "phase_review_example_count": len(phase_review["reviews"]),
            "excluded_unresolved_target_count": (
                len(phase_review["reviews"]) - len(joined)
            ),
        },
        configurations=[
            {
                "representation": representation,
                "c": c_value,
                "pca_components": components,
            }
            for representation in ("wide_summary", "wide_flat")
            for c_value in (0.01, 0.1, 1.0)
            for components in (8, 16)
        ],
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return value


def main() -> int:
    args = parse_args()
    artifact = run_screen(
        pre_anchor_path=args.pre_anchor,
        anchor_path=args.anchor,
        post_anchor_path=args.post_anchor,
        wide_anchor_path=args.wide_anchor,
        base_plan_path=args.base_plan,
        base_corrections_path=args.base_corrections,
        followup_plan_path=args.followup_plan,
        followup_corrections_path=args.followup_corrections,
        phase_plan_path=args.phase_plan,
        phase_review_path=args.phase_review,
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
                "baseline": artifact["variants"]["baseline"]["metrics"],
                "phase_auxiliary": artifact["variants"][
                    "phase_auxiliary"
                ]["metrics"],
                "accepted": artifact["accepted"],
                "artifact_sha256": artifact["artifact_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
