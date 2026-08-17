#!/usr/bin/env python3
"""Build label-free broadcast-state evidence from raw clock OCR artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.shot_broadcast_state import (  # noqa: E402
    build_broadcast_state_evidence_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-embeddings", type=Path, required=True)
    parser.add_argument("--clock-evidence", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact = build_broadcast_state_evidence_artifact(
        scene_artifact=_read_json(args.scene_embeddings),
        clock_artifacts=[_read_json(path) for path in args.clock_evidence],
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
                "example_count": len(artifact["examples"]),
                "feature_count": len(artifact["feature_names"]),
                "artifact_sha256": artifact["artifact_sha256"],
            }
        )
    )
    return 0


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
