#!/usr/bin/env python3
"""Build a source-balanced, label-hidden held-out annotation batch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.continuous_game_annotation_queue import verify_continuous_game_annotation_queue  # noqa: E402
from app.analysis.heldout_annotation_batch import (  # noqa: E402
    build_heldout_annotation_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--exclude-anchor", action="append", default=[], metavar="SOURCE_ID=SECONDS")
    parser.add_argument(
        "--source-id",
        action="append",
        default=None,
        metavar="SOURCE_ID",
        help="restrict selection to these queue sources; repeat for a source-disjoint batch",
    )
    parser.add_argument("--per-source", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--exclusion-radius-seconds", type=float, default=20.0)
    parser.add_argument(
        "--temporal-bucket-quota",
        action="append",
        default=[],
        metavar="SOURCE_ID=BUCKET=COUNT",
        help="repeat for early/middle/late quotas; enables stratified selection",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_batch(
    *,
    queue_path: Path,
    anchors: list[str],
    per_source: int,
    seed: int,
    source_ids: list[str] | None = None,
    radius: float,
    temporal_bucket_quotas: list[str] | None = None,
) -> dict[str, Any]:
    queue = verify_continuous_game_annotation_queue(_read_json(queue_path))
    parsed: list[dict[str, Any]] = []
    for value in anchors:
        source_id, separator, seconds = value.partition("=")
        if separator != "=" or not source_id or not seconds:
            raise ValueError("--exclude-anchor must use SOURCE_ID=SECONDS")
        parsed.append({"source_id": source_id, "center_seconds": float(seconds)})
    quotas: dict[str, dict[str, int]] = {}
    for value in temporal_bucket_quotas or []:
        source_id, separator, remainder = value.partition("=")
        bucket, separator2, count = remainder.partition("=")
        if (
            not separator
            or not separator2
            or not source_id
            or bucket not in {"early", "middle", "late"}
            or not count
        ):
            raise ValueError("--temporal-bucket-quota must use SOURCE_ID=BUCKET=COUNT")
        try:
            parsed_count = int(count)
        except ValueError as exc:
            raise ValueError("temporal bucket quota count must be an integer") from exc
        quotas.setdefault(source_id, {})[bucket] = parsed_count
    return build_heldout_annotation_batch(
        queue,
        per_source=per_source,
        seed=seed,
        source_ids=source_ids,
        excluded_anchors=parsed,
        exclusion_radius_seconds=radius,
        temporal_bucket_quotas=quotas or None,
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def main() -> int:
    args = parse_args()
    artifact = build_batch(
        queue_path=args.queue,
        anchors=args.exclude_anchor,
        per_source=args.per_source,
        seed=args.seed,
        source_ids=args.source_id,
        radius=args.exclusion_radius_seconds,
        temporal_bucket_quotas=args.temporal_bucket_quota,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "windows": len(artifact["windows"]), "artifact_sha256": artifact["artifact_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
