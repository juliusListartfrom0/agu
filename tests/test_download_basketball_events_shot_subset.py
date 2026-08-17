from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "download_basketball_events_shot_subset.py"
    spec = importlib.util.spec_from_file_location(
        "download_basketball_events_shot_subset", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(game: str) -> dict[str, object]:
    clips = []
    for index, (result, start) in enumerate(
        (("make", 0), ("miss", 10), ("make", 20), ("miss", 30), ("make", 40))
    ):
        clips.append(
            {
                "clip_file": f"{game}_clip{index:03d}.mp4",
                "clip_start": float(start),
                "clip_end": float(start + 7),
                "duration": 7.0,
                "events": [
                    {
                        "event": "shot",
                        "shot_type": "2-pt jump shot",
                        "result": result,
                    }
                ],
            }
        )
    clips.append(
        {
            "clip_file": f"{game}_ambiguous.mp4",
            "clip_start": 100.0,
            "clip_end": 107.0,
            "duration": 7.0,
            "events": [
                {"event": "shot", "shot_type": "3-pt jump shot", "result": "make"},
                {"event": "shot", "shot_type": "2-pt jump shot", "result": "miss"},
            ],
        }
    )
    return {"game_id": game, "clips": clips}


def test_select_balanced_shot_clips_is_per_game_and_excludes_ambiguous_rows() -> None:
    module = _module()
    games = [_payload("AAA_vs_BBB_20200101"), _payload("CCC_vs_DDD_20200102")]
    files = [
        {
            "path": f"clips/{payload['game_id']}/{clip['clip_file']}",
            "size": 100 + index,
            "oid": f"oid-{index}",
        }
        for index, (payload, clip) in enumerate(
            (payload, clip)
            for payload in games
            for clip in payload["clips"]
        )
    ]

    selected = module.select_balanced_shot_clips(
        games,
        files,
        per_game=4,
    )

    assert len(selected) == 8
    assert {row["game_id"] for row in selected} == {
        "AAA_vs_BBB_20200101",
        "CCC_vs_DDD_20200102",
    }
    for game in {row["game_id"] for row in selected}:
        rows = [row for row in selected if row["game_id"] == game]
        assert [row["shot_result"] for row in rows].count("make") == 2
        assert [row["shot_result"] for row in rows].count("miss") == 2
        assert all("ambiguous" not in row["source_path"] for row in rows)


def test_select_balanced_shot_clips_honors_exclusions_and_byte_budget() -> None:
    module = _module()
    payload = _payload("AAA_vs_BBB_20200101")
    files = [
        {
            "path": f"clips/{payload['game_id']}/{clip['clip_file']}",
            "size": 80,
            "oid": f"oid-{index}",
        }
        for index, clip in enumerate(payload["clips"])
    ]

    selected = module.select_balanced_shot_clips(
        [payload],
        files,
        per_game=4,
        excluded_paths={files[0]["path"]},
        max_bytes=240,
    )

    assert len(selected) == 3
    assert files[0]["path"] not in {row["source_path"] for row in selected}
    assert sum(row["size_bytes"] for row in selected) <= 240

    with pytest.raises(ValueError, match="even"):
        module.select_balanced_shot_clips([payload], files, per_game=3)


def test_resolve_source_url_rejects_traversal_and_quotes_path() -> None:
    module = _module()
    assert module.resolve_source_url("rev", "clips/GAME/clip 01.mp4").endswith(
        "/clips/GAME/clip%2001.mp4"
    )
    with pytest.raises(ValueError, match="unsafe"):
        module.resolve_source_url("rev", "clips/../escape.mp4")


def test_seal_manifest_is_stable_and_verifiable() -> None:
    module = _module()
    payload = {"schema_version": module.SCHEMA_VERSION, "files": [{"path": "a.mp4"}]}
    sealed = module.seal_manifest(payload)

    assert len(sealed["manifest_sha256"]) == 64
    assert module.verify_manifest(sealed) == sealed
    sealed["files"][0]["path"] = "changed.mp4"
    with pytest.raises(ValueError, match="hash mismatch"):
        module.verify_manifest(sealed)
