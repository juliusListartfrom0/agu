#!/usr/bin/env python3
"""Bind enrollment faces to player tracks and select a team-consistent subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.face_player_context import (  # noqa: E402
    attach_face_player_context,
    select_team_consistent_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--perception", type=Path, action="append", required=True)
    parser.add_argument(
        "--team-map",
        action="append",
        required=True,
        help="Map a perception team label to a roster team ID (RAW=TEAM_ID).",
    )
    parser.add_argument("--output-context", type=Path, required=True)
    parser.add_argument("--team-id")
    parser.add_argument("--output-candidates", type=Path)
    parser.add_argument("--minimum-team-observations", type=int, default=2)
    parser.add_argument("--minimum-team-consensus", type=float, default=0.75)
    return parser.parse_args()


def _parse_team_map(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        raw_id, separator, team_id = value.partition("=")
        raw_id, team_id = raw_id.strip(), team_id.strip()
        if separator != "=" or not raw_id or not team_id or raw_id in result:
            raise ValueError(f"invalid or duplicate team mapping: {value}")
        result[raw_id] = team_id
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if (args.team_id is None) != (args.output_candidates is None):
        raise ValueError("--team-id and --output-candidates must be provided together")
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    perceptions = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.perception
    ]
    context = attach_face_player_context(
        candidates,
        perceptions,
        perception_sha256s=[_sha256(path) for path in args.perception],
        team_id_map=_parse_team_map(args.team_map),
        minimum_team_observations=args.minimum_team_observations,
        minimum_team_consensus=args.minimum_team_consensus,
    )
    _write_json(args.output_context, context)
    subset = None
    if args.team_id is not None:
        subset = select_team_consistent_candidates(
            candidates,
            context,
            team_id=args.team_id,
        )
        _write_json(args.output_candidates, subset)
    print(
        json.dumps(
            {
                "context_manifest_sha256": context["manifest_sha256"],
                "context_cluster_count": len(context["clusters"]),
                "selected_cluster_count": (
                    None if subset is None else len(subset["clusters"])
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
