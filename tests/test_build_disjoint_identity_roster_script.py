from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import scripts.build_disjoint_identity_roster as cli


def test_cli_builds_sealed_roster_and_portrait_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "box-score.jsonl"
    source.write_text(
        json.dumps(
            {
                "source": "nba.com",
                "row_type": "player",
                "game_id": "disjoint-game",
                "game_date": "1995-05-20",
                "team_id": 1610612745,
                "team_tricode": "HOU",
                "person_id": 165,
                "first_name": "Hakeem",
                "family_name": "Olajuwon",
                "player_slug": "hakeem-olajuwon",
                "jersey_num": "34",
                "statistics": {"points": 40},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    image = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "/x8AAusB9Wl2nJ0AAAAASUVORK5CYII="
    )
    monkeypatch.setattr(cli, "_request_bytes", lambda _url, *, retries: image)
    roster_output = tmp_path / "roster.json"
    annotated_output = tmp_path / "annotated.json"
    reference_output = tmp_path / "references.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_disjoint_identity_roster.py",
            "--source",
            f"HOU=1610612745={source}",
            "--forbidden-game-id",
            "blind-game-1",
            "--roster-output",
            str(roster_output),
            "--portrait-dir",
            str(tmp_path / "portraits"),
            "--annotated-output",
            str(annotated_output),
            "--single-reference-output",
            str(reference_output),
        ],
    )

    assert cli.main() == 0

    roster = json.loads(roster_output.read_text(encoding="utf-8"))
    annotated = json.loads(annotated_output.read_text(encoding="utf-8"))
    references = json.loads(reference_output.read_text(encoding="utf-8"))
    assert roster["players"][0]["person_id"] == "165"
    assert annotated["persons"][0]["person_id"] == "165"
    assert references["references"][0]["person_id"] == "165"
    assert Path(annotated["persons"][0]["samples"][0]["path"]).is_file()
