from __future__ import annotations

import json
from pathlib import Path

from app.analysis.nba_games_local_mirror import (
    NBA_GAMES_LOCAL_MIRROR_SCHEMA,
    build_nba_games_local_mirror_audit,
    verify_nba_games_local_mirror_audit,
)


def _write_game(root: Path, game_id: str, *, pbp_rows: list[dict[str, object]]) -> None:
    game = root / "games" / game_id
    game.mkdir(parents=True)
    metadata = {
        "source_game": {
            "id": game_id + "-youtube",
            "url": "https://www.youtube.com/watch?v=" + game_id,
            "title": "Away vs. Home",
            "date": "2025-01-01",
            "duration": 6000,
        },
        "official_game": {
            "game_id": game_id,
            "game_date": "2025-01-01",
            "away_abbr": "AWY",
            "home_abbr": "HME",
        },
    }
    (game / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (game / "box-score.jsonl").write_text(
        json.dumps({"row_type": "team", "game_id": game_id}) + "\n",
        encoding="utf-8",
    )
    (game / "play-by-play.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in pbp_rows),
        encoding="utf-8",
    )


def test_local_mirror_audit_binds_index_and_structured_files(tmp_path: Path) -> None:
    index = tmp_path / "nba_games.jsonl"
    rows = [
        {"id": "game-a-youtube", "title": "Away vs. Home", "date": "2025-01-01", "duration": 6000},
        {"id": "game-b-youtube", "title": "Away vs. Home", "date": "2025-01-01", "duration": 6100},
    ]
    index.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    _write_game(mirror, "game-a", pbp_rows=[{"actionType": "Made Shot"}])
    _write_game(mirror, "game-b", pbp_rows=[])

    audit = build_nba_games_local_mirror_audit(index, mirror, generated_on="2026-08-13")

    assert audit["schema_version"] == NBA_GAMES_LOCAL_MIRROR_SCHEMA
    assert audit["index_game_count"] == 2
    assert audit["metadata_game_count"] == 2
    assert audit["games_with_nonempty_play_by_play"] == 1
    assert audit["play_by_play_rows"] == 1
    assert audit["video_file_count"] == 0
    assert audit["runtime_consumable"] is False
    assert verify_nba_games_local_mirror_audit(audit)["audit_sha256"] == audit["audit_sha256"]


def test_local_mirror_audit_rejects_index_mismatch(tmp_path: Path) -> None:
    index = tmp_path / "nba_games.jsonl"
    index.write_text(json.dumps({"id": "other", "title": "x", "date": "2025-01-01", "duration": 1}) + "\n")
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    _write_game(mirror, "game-a", pbp_rows=[])

    try:
        build_nba_games_local_mirror_audit(index, mirror, generated_on="2026-08-13")
    except ValueError as exc:
        assert "index" in str(exc)
    else:
        raise AssertionError("index mismatch must be rejected")
