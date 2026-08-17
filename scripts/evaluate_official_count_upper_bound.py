#!/usr/bin/env python3
"""Audit count-only F1 upper bounds for a sealed AGU blind prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.official_count_audit import build_blind_count_audit  # noqa: E402
from app.analysis.official_evaluation import verify_agu_autonomous_bundle  # noqa: E402
from app.analysis.schemas import RawOnlyPredictionBundleResponse  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--official-json", type=Path, required=True)
    parser.add_argument("--target-f1", type=float, default=0.85)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = verify_agu_autonomous_bundle(
        RawOnlyPredictionBundleResponse.model_validate_json(
            args.bundle.read_text(encoding="utf-8")
        )
    )
    official_payload = json.loads(args.official_json.read_text(encoding="utf-8"))
    report = build_blind_count_audit(
        prediction_bundle_sha256=bundle.bundle_sha256,
        events=bundle.events,
        official_payload=official_payload,
        target_f1=args.target_f1,
    )
    report["prediction_file_sha256"] = _file_sha256(args.bundle)
    report["official_file_sha256"] = _file_sha256(args.official_json)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
