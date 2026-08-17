#!/usr/bin/env python3
"""Seal complete offline review decisions for raw-audio action candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.audio_evidence import (  # noqa: E402
    AudioEvidenceArtifact,
    seal_audio_action_review,
)


def seal_audio_action_review_file(
    evidence_path: Path,
    decisions_path: Path,
    output_path: Path,
) -> dict[str, object]:
    evidence = AudioEvidenceArtifact.model_validate_json(
        evidence_path.read_text(encoding="utf-8")
    )
    source = json.loads(decisions_path.read_text(encoding="utf-8"))
    artifact = seal_audio_action_review(
        evidence,
        action=str(source.get("action") or ""),
        decisions=source.get("decisions") or [],
        producer=str(source.get("producer") or ""),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = seal_audio_action_review_file(args.evidence, args.decisions, args.output)
    print(
        json.dumps(
            {
                "artifact_sha256": artifact["artifact_sha256"],
                "reviewed_count": artifact["reviewed_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
