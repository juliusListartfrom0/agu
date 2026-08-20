#!/usr/bin/env python3
"""Screen cut-aligned dense-video state features with nested held-game folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.cut_aligned_video_state_screen import (  # noqa: E402
    build_cut_aligned_video_state_examples,
    screen_cut_aligned_video_state_examples,
)
from app.analysis.pbp_visual_state_review import (  # noqa: E402
    verify_visual_state_label_corrections,
    verify_visual_state_review_plan,
)


def run_screen(
    *,
    embedding_paths: list[Path],
    base_plan_path: Path,
    base_corrections_path: Path,
    followup_plan_path: Path,
    followup_corrections_path: Path,
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

    examples, provenance = build_cut_aligned_video_state_examples(
        embedding_artifacts=[_read_json(path) for path in embedding_paths],
        base_decisions=base_corrections["decisions"],
        followup_decisions=followup_corrections["decisions"],
        sealed_blind_video_sha256s=base_plan["sealed_blind_video_sha256s"],
    )
    if (
        provenance["review_plan_sha256"] != base_plan["artifact_sha256"]
        or {row["source_video_sha256"] for row in examples}
        != set(base_plan["source_video_sha256s"])
    ):
        raise ValueError("cut-aligned embeddings do not match review-plan provenance")
    artifact = screen_cut_aligned_video_state_examples(
        examples=examples,
        provenance=provenance,
        label_correction_artifact_sha256s=[
            base_corrections["artifact_sha256"],
            followup_corrections["artifact_sha256"],
        ],
    )
    _write_json(output_path, artifact)
    return artifact


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-artifact",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--base-corrections", type=Path, required=True)
    parser.add_argument("--followup-plan", type=Path, required=True)
    parser.add_argument("--followup-corrections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    artifact = run_screen(
        embedding_paths=args.embedding_artifact,
        base_plan_path=args.base_plan,
        base_corrections_path=args.base_corrections,
        followup_plan_path=args.followup_plan,
        followup_corrections_path=args.followup_corrections,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "accepted": artifact["accepted"],
                "metrics": artifact["metrics"],
                "artifact_sha256": artifact["artifact_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
