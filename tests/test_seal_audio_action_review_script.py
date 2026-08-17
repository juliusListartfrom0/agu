from __future__ import annotations

import json
from pathlib import Path

from app.analysis.audio_evidence import (
    AudioRosterPlayer,
    build_audio_evidence,
    seal_audio_roster,
    seal_transcript_artifact,
    verify_audio_action_review,
)
from scripts.seal_audio_action_review import seal_audio_action_review_file


def test_seal_audio_action_review_file_binds_decisions_to_evidence(tmp_path: Path) -> None:
    roster = seal_audio_roster(
        [AudioRosterPlayer(person_id="977", team_id="LAL", display_name="Kobe Bryant")],
        source_roster_sha256="roster",
    )
    transcript = seal_transcript_artifact(
        {
            "schema_version": "agu.raw-audio-transcript.v1",
            "video_sha256": "video",
            "roster_source_sha256": "roster",
            "model": "fixture",
            "transcript": {
                "segments": [{"start": 1, "end": 2, "text": "Bryant draws the foul."}]
            },
        }
    )
    evidence = build_audio_evidence(
        transcript_artifact=transcript,
        roster=roster,
        raw_video_sha256="video",
        fps=30,
    )
    evidence_path = tmp_path / "evidence.json"
    decisions_path = tmp_path / "decisions.json"
    output_path = tmp_path / "review.json"
    evidence_path.write_text(evidence.model_dump_json(indent=2), encoding="utf-8")
    decisions_path.write_text(
        json.dumps(
            {
                "action": "foul",
                "producer": "codex_offline_review",
                "decisions": [
                    {
                        "mention_id": evidence.mentions[0].mention_id,
                        "label": "live_current_event",
                        "note": "fixture",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    artifact = seal_audio_action_review_file(evidence_path, decisions_path, output_path)

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert verify_audio_action_review(loaded) == artifact
