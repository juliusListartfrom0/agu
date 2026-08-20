#!/usr/bin/env python3
"""Audit a label-hidden independent VLM/auxiliary transfer pair.

The command only reads two sealed, development-only artifacts.  It joins them
by their exact source/candidate/event keys and records whether their frozen
plan contracts match.  No annotation or review path is accepted, and no
runtime or training artifact is emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.independent_shot_vlm_consistency import (  # noqa: E402
    canonical_sha256,
    screen_cross_model_consistency,
    verify_consistency_artifact,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact(*, vlm_path: Path, auxiliary_path: Path) -> dict[str, Any]:
    vlm = json.loads(vlm_path.read_text(encoding="utf-8"))
    auxiliary = json.loads(auxiliary_path.read_text(encoding="utf-8"))
    artifact = screen_cross_model_consistency(
        vlm_predictions=vlm,
        auxiliary_transfer=auxiliary,
    )
    artifact["contracts"].update(
        {
            "vlm_predictions": vlm_path.as_posix(),
            "vlm_predictions_file_sha256": file_sha256(vlm_path),
            "auxiliary_transfer": auxiliary_path.as_posix(),
            "auxiliary_transfer_file_sha256": file_sha256(auxiliary_path),
        }
    )
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return verify_consistency_artifact(artifact)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm-predictions", type=Path, required=True)
    parser.add_argument("--auxiliary-transfer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = build_artifact(
        vlm_path=args.vlm_predictions,
        auxiliary_path=args.auxiliary_transfer,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "artifact_sha256": artifact["artifact_sha256"],
                **artifact["summary"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
