#!/usr/bin/env python3
"""Screen continuous shot-causal support with nested game-held thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.shot_causal_support import (  # noqa: E402
    DEFAULT_CAUSAL_FEATURE_NAMES,
    screen_shot_causal_support,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-fusion", type=Path, required=True)
    parser.add_argument("--reason-evidence", type=Path, required=True)
    parser.add_argument(
        "--causal-feature",
        action="append",
        dest="causal_features",
        default=None,
        help="Feature name to include; may be repeated.",
    )
    parser.add_argument("--regularization-c", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = screen_shot_causal_support(
        base_artifact=_read_json(args.base_fusion),
        reason_evidence_artifact=_read_json(args.reason_evidence),
        causal_feature_names=tuple(args.causal_features or DEFAULT_CAUSAL_FEATURE_NAMES),
        regularization_c=args.regularization_c,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "accepted": result["accepted"],
                "metrics": result["metrics"],
                "artifact_sha256": result["artifact_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
