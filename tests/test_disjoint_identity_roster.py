from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from app.analysis.disjoint_identity_roster import (
    IdentityRosterSource,
    _canonical_sha256,
    build_disjoint_identity_roster,
    build_single_face_reference_manifest,
    download_identity_portraits,
    identity_coverage_person_ids,
    verify_disjoint_identity_roster,
)


def _write_box_score(path: Path, *, game_id: str, team: str, person_id: int) -> None:
    rows = [
        {
            "row_type": "team",
            "game_id": game_id,
            "game_date": "1995-05-20",
            "team_tricode": team,
            "nba_game_url": f"https://www.nba.com/game/{game_id}",
            "statistics": {"points": 100},
        },
        {
            "row_type": "player",
            "game_id": game_id,
            "game_date": "1995-05-20",
            "team_id": 1610612745,
            "team_tricode": team,
            "person_id": person_id,
            "first_name": "Test",
            "family_name": "Player",
            "player_slug": "test-player",
            "jersey_num": "34",
            "statistics": {"points": 42, "assists": 8},
        },
        {
            "row_type": "player",
            "game_id": game_id,
            "game_date": "1995-05-20",
            "team_id": 999,
            "team_tricode": "OPP",
            "person_id": 999,
            "first_name": "Opponent",
            "family_name": "Player",
            "jersey_num": "1",
            "statistics": {"points": 1},
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_roster_uses_only_identity_fields_from_disjoint_official_game(tmp_path: Path) -> None:
    source_path = tmp_path / "box-score.jsonl"
    _write_box_score(source_path, game_id="source-game", team="HOU", person_id=165)

    roster = build_disjoint_identity_roster(
        [
            IdentityRosterSource(
                path=source_path,
                team_tricode="HOU",
                team_id="1610612745",
            )
        ],
        forbidden_game_ids={"blind-game-1", "blind-game-2"},
    )

    assert roster["schema_version"] == "agu.face-roster.v1"
    assert roster["benchmark_disjoint"] is True
    assert roster["identity_only"] is True
    assert roster["runtime_consumable"] is False
    assert roster["sources"][0]["game_id"] == "source-game"
    assert roster["players"] == [
        {
            "person_id": "165",
            "display_name": "Test Player",
            "team_id": "1610612745",
            "team_tricode": "HOU",
            "jersey_number": "34",
            "player_slug": "test-player",
            "portrait_url": (
                "https://cdn.nba.com/headshots/nba/latest/1040x760/165.png"
            ),
        }
    ]
    assert "statistics" not in json.dumps(roster)
    assert verify_disjoint_identity_roster(roster)["roster_sha256"]


def test_roster_rejects_a_forbidden_benchmark_game(tmp_path: Path) -> None:
    source_path = tmp_path / "box-score.jsonl"
    _write_box_score(source_path, game_id="blind-game-1", team="HOU", person_id=165)

    with pytest.raises(ValueError, match="forbidden benchmark game"):
        build_disjoint_identity_roster(
            [
                IdentityRosterSource(
                    path=source_path,
                    team_tricode="HOU",
                    team_id="1610612745",
                )
            ],
            forbidden_game_ids={"blind-game-1", "blind-game-2"},
        )


def test_roster_verifier_rejects_tampering(tmp_path: Path) -> None:
    source_path = tmp_path / "box-score.jsonl"
    _write_box_score(source_path, game_id="source-game", team="HOU", person_id=165)
    roster = build_disjoint_identity_roster(
        [
            IdentityRosterSource(
                path=source_path,
                team_tricode="HOU",
                team_id="1610612745",
            )
        ],
        forbidden_game_ids={"blind-game-1"},
    )

    roster["players"][0]["display_name"] = "Tampered"

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_disjoint_identity_roster(roster)


def test_roster_adds_hash_bound_identity_supplement_and_visual_scope(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "box-score.jsonl"
    _write_box_score(source_path, game_id="source-game", team="HOU", person_id=165)
    supplement_path = tmp_path / "identity-supplement.json"
    supplement_path.write_text(
        json.dumps(
            {
                "schema_version": "agu.face-roster-supplement.v1",
                "benchmark_disjoint": True,
                "identity_only": True,
                "runtime_consumable": False,
                "evidence_sources": [
                    {
                        "provider": "NBA",
                        "source_url": "https://www.nba.com/stats/player/279/career",
                        "claim": "official player identity",
                    }
                ],
                "players": [
                    {
                        "person_id": "279",
                        "display_name": "Charles Jones",
                        "team_id": "1610612745",
                        "team_tricode": "HOU",
                        "jersey_number": "27",
                        "player_slug": "charles-jones",
                    }
                ],
                "visual_identity_scope": {
                    "basis": "benchmark_disjoint_enrollment_game_participation",
                    "source_game_ids": ["enrollment-game"],
                    "required_person_ids": ["165", "279"],
                    "not_required_persons": [],
                },
            }
        ),
        encoding="utf-8",
    )

    roster = build_disjoint_identity_roster(
        [
            IdentityRosterSource(
                path=source_path,
                team_tricode="HOU",
                team_id="1610612745",
            )
        ],
        forbidden_game_ids={"blind-game-1"},
        supplemental_identity_paths=[supplement_path],
    )

    assert [player["person_id"] for player in roster["players"]] == ["279", "165"]
    assert roster["supplemental_sources"][0]["filename"] == supplement_path.name
    assert len(roster["supplemental_sources"][0]["file_sha256"]) == 64
    assert roster["visual_identity_scope"]["required_person_ids"] == ["165", "279"]
    assert identity_coverage_person_ids(roster) == ("165", "279")
    assert verify_disjoint_identity_roster(roster)["roster_sha256"]


def test_visual_scope_requires_exhaustive_roster_partition(tmp_path: Path) -> None:
    source_path = tmp_path / "box-score.jsonl"
    _write_box_score(source_path, game_id="source-game", team="HOU", person_id=165)
    supplement_path = tmp_path / "identity-supplement.json"
    supplement_path.write_text(
        json.dumps(
            {
                "schema_version": "agu.face-roster-supplement.v1",
                "benchmark_disjoint": True,
                "identity_only": True,
                "runtime_consumable": False,
                "evidence_sources": [
                    {
                        "provider": "NBA",
                        "source_url": "https://www.nba.com/stats/player/279/career",
                        "claim": "official player identity",
                    }
                ],
                "players": [
                    {
                        "person_id": "279",
                        "display_name": "Charles Jones",
                        "team_id": "1610612745",
                        "team_tricode": "HOU",
                        "jersey_number": "27",
                        "player_slug": "charles-jones",
                    }
                ],
                "visual_identity_scope": {
                    "basis": "benchmark_disjoint_enrollment_game_participation",
                    "source_game_ids": ["enrollment-game"],
                    "required_person_ids": ["165"],
                    "not_required_persons": [],
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="partition"):
        build_disjoint_identity_roster(
            [
                IdentityRosterSource(
                    path=source_path,
                    team_tricode="HOU",
                    team_id="1610612745",
                )
            ],
            forbidden_game_ids={"blind-game-1"},
            supplemental_identity_paths=[supplement_path],
        )


def test_visual_scope_can_exempt_metadata_only_players(tmp_path: Path) -> None:
    source_path = tmp_path / "box-score.jsonl"
    _write_box_score(source_path, game_id="source-game", team="HOU", person_id=165)
    roster = build_disjoint_identity_roster(
        [
            IdentityRosterSource(
                path=source_path,
                team_tricode="HOU",
                team_id="1610612745",
            )
        ],
        forbidden_game_ids={"blind-game-1"},
    )
    roster["visual_identity_scope"] = {
        "basis": "benchmark_disjoint_enrollment_game_participation",
        "source_game_ids": ["enrollment-game"],
        "required_person_ids": [],
        "not_required_persons": [
            {
                "person_id": "165",
                "reason": "did_not_play_in_enrollment_games",
            }
        ],
        "source_file_sha256": "a" * 64,
    }
    roster["roster_sha256"] = ""
    roster["roster_sha256"] = _canonical_sha256(roster)

    with pytest.raises(ValueError, match="at least one required"):
        identity_coverage_person_ids(roster)


def test_portrait_download_builds_gallery_input_without_answer_fields(tmp_path: Path) -> None:
    source_path = tmp_path / "box-score.jsonl"
    _write_box_score(source_path, game_id="source-game", team="HOU", person_id=165)
    roster = build_disjoint_identity_roster(
        [
            IdentityRosterSource(
                path=source_path,
                team_tricode="HOU",
                team_id="1610612745",
            )
        ],
        forbidden_game_ids={"blind-game-1"},
    )
    one_pixel_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "/x8AAusB9Wl2nJ0AAAAASUVORK5CYII="
    )

    manifest = download_identity_portraits(
        roster,
        output_dir=tmp_path / "portraits",
        fetch_bytes=lambda _url: one_pixel_png,
    )

    assert manifest["schema_version"] == "agu.annotated-face-list.v1"
    assert manifest["benchmark_disjoint"] is True
    assert manifest["identity_only"] is True
    assert manifest["annotation_producer"] == "agu_official_nba_portrait_identity_v1"
    assert manifest["persons"][0]["person_id"] == "165"
    assert manifest["persons"][0]["team_id"] == "1610612745"
    sample = manifest["persons"][0]["samples"][0]
    assert Path(sample["path"]).read_bytes() == one_pixel_png
    assert len(sample["sha256"]) == 64
    assert "statistics" not in json.dumps(manifest)


def test_portrait_download_rejects_non_image_response(tmp_path: Path) -> None:
    source_path = tmp_path / "box-score.jsonl"
    _write_box_score(source_path, game_id="source-game", team="HOU", person_id=165)
    roster = build_disjoint_identity_roster(
        [
            IdentityRosterSource(
                path=source_path,
                team_tricode="HOU",
                team_id="1610612745",
            )
        ],
        forbidden_game_ids={"blind-game-1"},
    )

    with pytest.raises(ValueError, match="not a supported image"):
        download_identity_portraits(
            roster,
            output_dir=tmp_path / "portraits",
            fetch_bytes=lambda _url: b"<html>not an image</html>",
        )


def test_portrait_download_rejects_one_image_reused_for_multiple_people(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "box-score.jsonl"
    _write_box_score(source_path, game_id="source-game", team="HOU", person_id=165)
    rows = [json.loads(line) for line in source_path.read_text().splitlines()]
    second_player = dict(rows[1])
    second_player.update(
        {
            "person_id": 17,
            "first_name": "Second",
            "family_name": "Player",
            "player_slug": "second-player",
            "jersey_num": "22",
        }
    )
    rows.append(second_player)
    source_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    roster = build_disjoint_identity_roster(
        [
            IdentityRosterSource(
                path=source_path,
                team_tricode="HOU",
                team_id="1610612745",
            )
        ],
        forbidden_game_ids={"blind-game-1"},
    )
    placeholder_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "/x8AAusB9Wl2nJ0AAAAASUVORK5CYII="
    )

    with pytest.raises(ValueError, match="reused for multiple people"):
        download_identity_portraits(
            roster,
            output_dir=tmp_path / "portraits",
            fetch_bytes=lambda _url: placeholder_png,
        )


def test_portrait_download_can_exclude_all_members_of_duplicate_content_group(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "box-score.jsonl"
    _write_box_score(source_path, game_id="source-game", team="HOU", person_id=165)
    rows = [json.loads(line) for line in source_path.read_text().splitlines()]
    for person_id in (17, 53):
        player = dict(rows[1])
        player.update(
            {
                "person_id": person_id,
                "first_name": f"Player{person_id}",
                "family_name": "Test",
                "player_slug": f"player-{person_id}",
                "jersey_num": str(person_id),
            }
        )
        rows.append(player)
    source_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    roster = build_disjoint_identity_roster(
        [
            IdentityRosterSource(
                path=source_path,
                team_tricode="HOU",
                team_id="1610612745",
            )
        ],
        forbidden_game_ids={"blind-game-1"},
    )
    base_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "/x8AAusB9Wl2nJ0AAAAASUVORK5CYII="
    )

    manifest = download_identity_portraits(
        roster,
        output_dir=tmp_path / "portraits",
        fetch_bytes=lambda url: base_png + (b"unique" if "/53.png" in url else b""),
        duplicate_policy="exclude",
    )

    assert [person["person_id"] for person in manifest["persons"]] == ["53"]
    assert {item["person_id"] for item in manifest["excluded_duplicate_portraits"]} == {
        "17",
        "165",
    }
    assert not (tmp_path / "portraits" / "17.png").exists()
    assert not (tmp_path / "portraits" / "165.png").exists()


def test_portrait_manifest_converts_to_offline_single_reference_ranking_input(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "box-score.jsonl"
    _write_box_score(source_path, game_id="source-game", team="HOU", person_id=165)
    roster = build_disjoint_identity_roster(
        [
            IdentityRosterSource(
                path=source_path,
                team_tricode="HOU",
                team_id="1610612745",
            )
        ],
        forbidden_game_ids={"blind-game-1"},
    )
    image = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "/x8AAusB9Wl2nJ0AAAAASUVORK5CYII="
    )
    annotated = download_identity_portraits(
        roster,
        output_dir=tmp_path / "portraits",
        fetch_bytes=lambda _url: image,
    )

    references = build_single_face_reference_manifest(
        annotated,
        model_id="opencv_yunet+sface",
    )

    assert references["schema_version"] == "agu.single-face-reference-list.v1"
    assert references["benchmark_disjoint"] is True
    assert references["runtime_consumable"] is False
    assert references["model_id"] == "opencv_yunet+sface"
    assert references["references"] == [
        {
            "person_id": "165",
            "team_id": "1610612745",
            "path": annotated["persons"][0]["samples"][0]["path"],
            "sha256": annotated["persons"][0]["samples"][0]["sha256"],
            "source_url": annotated["persons"][0]["samples"][0]["source_url"],
        }
    ]
    assert len(references["manifest_sha256"]) == 64
