#!/usr/bin/env python3
"""Check whether a sealed face gallery covers an expected player roster."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.face_gallery import assess_face_gallery_coverage, load_face_gallery  # noqa: E402
from app.analysis.disjoint_identity_roster import (  # noqa: E402
    identity_coverage_person_ids,
    verify_disjoint_identity_roster,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-coverage", type=float, default=0.95)
    parser.add_argument("--minimum-entry-quality", type=float, default=0.50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roster = json.loads(args.roster.read_text(encoding="utf-8"))
    verify_disjoint_identity_roster(roster)
    person_ids = list(identity_coverage_person_ids(roster))
    assessment = assess_face_gallery_coverage(
        load_face_gallery(args.gallery),
        person_ids,
        minimum_coverage=args.minimum_coverage,
        minimum_entry_quality=args.minimum_entry_quality,
    )
    payload = {
        "schema_version": "agu.face-gallery-coverage.v1",
        "coverage_scope": (
            "enrollment_game_visual_participants"
            if roster.get("visual_identity_scope") is not None
            else "complete_roster"
        ),
        "full_roster_person_count": len(roster.get("players") or []),
        "metadata_only_person_ids": [
            str(item["person_id"])
            for item in (roster.get("visual_identity_scope") or {}).get(
                "not_required_persons", []
            )
        ],
        **asdict(assessment),
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if assessment.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
