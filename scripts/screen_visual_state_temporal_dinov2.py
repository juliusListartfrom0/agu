#!/usr/bin/env python3
"""Screen existing pre/anchor/post DINOv2 embeddings on reviewed visual states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.analysis.visual_state_temporal_screen import (  # noqa: E402
    build_visual_state_temporal_screen,
)


def run_screen(
    *,
    pre_anchor_path: Path,
    anchor_path: Path,
    post_anchor_path: Path,
    wide_anchor_path: Path | None,
    base_plan_path: Path,
    base_corrections_path: Path,
    followup_plan_path: Path,
    followup_corrections_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    artifact = build_visual_state_temporal_screen(
        pre_anchor_artifact=_read_json(pre_anchor_path),
        anchor_artifact=_read_json(anchor_path),
        post_anchor_artifact=_read_json(post_anchor_path),
        wide_anchor_artifact=(
            None if wide_anchor_path is None else _read_json(wide_anchor_path)
        ),
        base_review_plan=_read_json(base_plan_path),
        base_label_corrections=_read_json(base_corrections_path),
        followup_review_plan=_read_json(followup_plan_path),
        followup_label_corrections=_read_json(followup_corrections_path),
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
    parser.add_argument("--wide-anchor", type=Path)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--base-corrections", type=Path, required=True)
    parser.add_argument("--followup-plan", type=Path, required=True)
    parser.add_argument("--followup-corrections", type=Path, required=True)
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
        output_path=arguments.output,
    )
    print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
