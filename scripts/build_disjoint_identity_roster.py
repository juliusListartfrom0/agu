#!/usr/bin/env python3
"""Build an identity-only face roster from benchmark-disjoint NBA games."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.disjoint_identity_roster import (  # noqa: E402
    IdentityRosterSource,
    build_disjoint_identity_roster,
    build_single_face_reference_manifest,
    download_identity_portraits,
)
from app.analysis.face_identity import OpenCvSFaceIdentityAdapter  # noqa: E402

USER_AGENT = "AGU-open-research/1.0 (identity-only NBA portrait enrollment)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="TRICODE=TEAM_ID=BOX_SCORE_JSONL",
    )
    parser.add_argument("--forbidden-game-id", action="append", required=True)
    parser.add_argument(
        "--supplemental-identity-source",
        action="append",
        type=Path,
        default=[],
        help="Benchmark-disjoint identity-only JSON supplement.",
    )
    parser.add_argument("--roster-output", type=Path, required=True)
    parser.add_argument("--portrait-dir", type=Path, required=True)
    parser.add_argument("--annotated-output", type=Path, required=True)
    parser.add_argument("--single-reference-output", type=Path)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--exclude-duplicate-portraits",
        action="store_true",
        help="Exclude all IDs sharing identical portrait bytes instead of failing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.retries <= 0:
        raise ValueError("retries must be positive")
    sources = [_parse_source(value) for value in args.source]
    roster = build_disjoint_identity_roster(
        sources,
        forbidden_game_ids=set(args.forbidden_game_id),
        supplemental_identity_paths=args.supplemental_identity_source,
    )
    args.roster_output.parent.mkdir(parents=True, exist_ok=True)
    args.roster_output.write_text(
        json.dumps(roster, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    annotated = download_identity_portraits(
        roster,
        output_dir=args.portrait_dir,
        fetch_bytes=lambda url: _request_bytes(url, retries=args.retries),
        duplicate_policy="exclude" if args.exclude_duplicate_portraits else "reject",
    )
    args.annotated_output.parent.mkdir(parents=True, exist_ok=True)
    args.annotated_output.write_text(
        json.dumps(annotated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    reference_manifest_sha256 = None
    if args.single_reference_output is not None:
        references = build_single_face_reference_manifest(
            annotated,
            model_id=OpenCvSFaceIdentityAdapter.model_id,
        )
        args.single_reference_output.parent.mkdir(parents=True, exist_ok=True)
        args.single_reference_output.write_text(
            json.dumps(references, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        reference_manifest_sha256 = references["manifest_sha256"]
    print(
        json.dumps(
            {
                "roster_sha256": roster["roster_sha256"],
                "manifest_sha256": annotated["manifest_sha256"],
                "reference_manifest_sha256": reference_manifest_sha256,
                "person_count": len(roster["players"]),
            },
            indent=2,
        )
    )
    return 0


def _parse_source(value: str) -> IdentityRosterSource:
    team_tricode, separator, remainder = value.partition("=")
    team_id, second_separator, raw_path = remainder.partition("=")
    if not separator or not second_separator or not raw_path:
        raise ValueError("source must use TRICODE=TEAM_ID=BOX_SCORE_JSONL")
    return IdentityRosterSource(
        path=Path(raw_path),
        team_tricode=team_tricode,
        team_id=team_id,
    )


def _request_bytes(url: str, *, retries: int) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except Exception as exc:  # pragma: no cover - exercised against the network
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"portrait request failed after {retries} attempts: {url}") from last_error


if __name__ == "__main__":
    raise SystemExit(main())
