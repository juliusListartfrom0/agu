from __future__ import annotations

import json
from pathlib import Path

from app.analysis.audio_evidence import AudioRosterArtifact, verify_audio_roster
from scripts.build_audio_roster import build_audio_roster


def test_build_audio_roster_writes_a_hash_bound_registration_artifact(tmp_path: Path) -> None:
    registration_path = tmp_path / "face_roster.json"
    output_path = tmp_path / "audio_roster.json"
    registration_path.write_text(
        json.dumps(
            {
                "schema_version": "agu.face-roster.v1",
                "benchmark_answers_included": False,
                "players": [
                    {
                        "person_id": "977",
                        "team_id": "LAL",
                        "display_name": "Kobe Bryant",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    artifact = build_audio_roster(registration_path, output_path)

    loaded = AudioRosterArtifact.model_validate_json(output_path.read_text(encoding="utf-8"))
    verify_audio_roster(loaded)
    assert loaded == artifact
    assert loaded.source_roster_sha256
