#!/usr/bin/env python3
"""Convert a truth-free registration roster into AGU's sealed audio roster."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.audio_evidence import (  # noqa: E402
    AudioRosterArtifact,
    seal_audio_roster_from_registration,
    sha256_file,
)


def build_audio_roster(
    registration_path: Path,
    output_path: Path,
) -> AudioRosterArtifact:
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    artifact = seal_audio_roster_from_registration(
        registration,
        source_roster_sha256=sha256_file(registration_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration-roster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = build_audio_roster(args.registration_roster, args.output)
    print(
        json.dumps(
            {
                "artifact_sha256": artifact.artifact_sha256,
                "player_count": len(artifact.players),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
