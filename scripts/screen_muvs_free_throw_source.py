#!/usr/bin/env python3
"""Screen reviewed MUVS free-throw features with event-grouped OOF folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.muvs_event_state_transfer import (  # noqa: E402
    screen_muvs_free_throw_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    embeddings = [json.loads(path.read_text(encoding="utf-8")) for path in args.embeddings]
    if not all(isinstance(payload, dict) for payload in embeddings):
        raise ValueError("expected MUVS embedding JSON objects")
    artifact = screen_muvs_free_throw_source(embeddings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "accepted": artifact["accepted"],
                "best_candidate": artifact["best_candidate"],
                "artifact_sha256": artifact["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
