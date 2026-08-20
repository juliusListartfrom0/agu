#!/usr/bin/env python3
"""Build wider raw-frame sheets for unresolved PBP visual-state reviews."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.pbp_visual_state_review import (  # noqa: E402
    build_visual_state_followup_review_plan,
)
from scripts.build_pbp_visual_state_review import (  # noqa: E402
    render_review_plan_package,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--base-corrections", type=Path, required=True)
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel-width", type=int, default=240)
    parser.add_argument("--rows-per-sheet", type=int, default=5)
    return parser.parse_args()


def build_followup_package(
    *,
    base_plan_path: Path,
    base_corrections_path: Path,
    video_paths: list[Path],
    output_dir: Path,
    panel_width: int = 240,
    rows_per_sheet: int = 5,
) -> dict[str, Any]:
    plan = build_visual_state_followup_review_plan(
        _read_json(base_plan_path),
        corrections=_read_json(base_corrections_path),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "plan.json"
    _write_json(plan_path, plan)
    return render_review_plan_package(
        plan_path=plan_path,
        video_paths=video_paths,
        output_dir=output_dir,
        panel_width=panel_width,
        rows_per_sheet=rows_per_sheet,
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    manifest = build_followup_package(
        base_plan_path=args.base_plan,
        base_corrections_path=args.base_corrections,
        video_paths=args.video,
        output_dir=args.output_dir,
        panel_width=args.panel_width,
        rows_per_sheet=args.rows_per_sheet,
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "sheets": len(manifest["records"]),
                "examples": sum(
                    len(row["review_ids"])
                    for row in manifest["records"]
                ),
                "artifact_sha256": manifest["artifact_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
