from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.analysis.bard_event_state import (
    build_bard_embedded_video_plan,
    build_bard_event_state_plan,
    materialize_bard_embedded_subset,
    materialize_bard_event_state_subset,
    parse_bard_benchmark_csv,
    parse_bard_dataset_csv,
    seal_bard_video_resolution_manifest,
    select_bard_embedded_state_rows,
    select_bard_event_state_rows,
    verify_bard_embedded_subset,
    verify_bard_event_state_plan,
    verify_bard_event_state_subset,
)


def _csv_fixture() -> str:
    rows = [
        (
            "https://www.nba.com/stats/events/?GameEventID=1&"
            "GameID=0022400001&Season=2024-25&flag=1",
            "[{'player': '1', 'action': '2PT Shot', 'result': True, "
            "'assisted': False, 'other_player': None, 'color': 'blue'}]",
        ),
        (
            "https://www.nba.com/stats/events/?GameEventID=2&"
            "GameID=0022400002&Season=2024-25&flag=1",
            "[{'player': '2', 'action': '3PT Shot', 'result': False, "
            "'assisted': False, 'other_player': None, 'color': 'white'}]",
        ),
        (
            "https://www.nba.com/stats/events/?GameEventID=3&"
            "GameID=0022400001&Season=2024-25&flag=1",
            "[{'player': '3', 'action': 'Free Throw', 'result': True, "
            "'assisted': None, 'other_player': None, 'color': 'blue'}]",
        ),
        (
            "https://www.nba.com/stats/events/?GameEventID=4&"
            "GameID=0022400002&Season=2024-25&flag=1",
            "[{'player': '4', 'action': 'Free Throw', 'result': False, "
            "'assisted': None, 'other_player': None, 'color': 'white'}]",
        ),
        (
            "https://www.nba.com/stats/events/?GameEventID=5&"
            "GameID=0022400001&Season=2024-25&flag=1",
            "[{'player': '5', 'action': 'Foul', 'result': None, "
            "'assisted': None, 'other_player': None, 'color': 'blue'}]",
        ),
        (
            "https://www.nba.com/stats/events/?GameEventID=6&"
            "GameID=0022400002&Season=2024-25&flag=1",
            "[{'player': '6', 'action': 'Foul', 'result': None, "
            "'assisted': None, 'other_player': None, 'color': 'white'}]",
        ),
    ]
    return "urls;actions;numerosity\n" + "".join(
        f'{url};"{actions.replace(chr(34), chr(34) * 2)}";1\n'
        for url, actions in rows
    )


def test_parse_bard_csv_recovers_exact_event_state_and_outcome() -> None:
    rows = parse_bard_dataset_csv(_csv_fixture())

    assert [row.clip_id for row in rows] == [
        "0022400001-1",
        "0022400002-2",
        "0022400001-3",
        "0022400002-4",
        "0022400001-5",
        "0022400002-6",
    ]
    assert [row.event_state for row in rows] == [
        "live_field_goal",
        "live_field_goal",
        "free_throw",
        "free_throw",
        "foul_only",
        "foul_only",
    ]
    assert [row.shot_outcome for row in rows] == [
        "made",
        "missed",
        "made",
        "missed",
        None,
        None,
    ]


def test_parse_bard_csv_rejects_code_and_blind_game_rows() -> None:
    malicious = (
        "urls;actions;numerosity\n"
        "https://www.nba.com/stats/events/?GameEventID=1&"
        "GameID=0022400001&Season=2024-25;"
        "\"__import__('os').system('false')\";1\n"
    )
    with pytest.raises(ValueError, match="actions"):
        parse_bard_dataset_csv(malicious)

    blind = _csv_fixture().replace("0022400001", "0049400070")
    with pytest.raises(ValueError, match="blind"):
        parse_bard_dataset_csv(blind)


def test_bard_selection_is_deterministic_and_spreads_games() -> None:
    rows = parse_bard_dataset_csv(_csv_fixture())

    first = select_bard_event_state_rows(rows, per_class=2, seed=17)
    second = select_bard_event_state_rows(reversed(rows), per_class=2, seed=17)

    assert first == second
    assert len(first) == 6
    assert {
        state: len({row.game_id for row in first if row.event_state == state})
        for state in ("live_field_goal", "free_throw", "foul_only")
    } == {
        "live_field_goal": 2,
        "free_throw": 2,
        "foul_only": 2,
    }


def test_plan_and_resolution_manifest_are_exact_hash_bound_and_blind_safe() -> None:
    rows = select_bard_event_state_rows(
        parse_bard_dataset_csv(_csv_fixture()),
        per_class=1,
        seed=3,
    )
    plan = build_bard_event_state_plan(
        rows,
        source_revision="a" * 40,
        source_csv_sha256="b" * 64,
        per_class=1,
        seed=3,
    )
    assert verify_bard_event_state_plan(json.loads(json.dumps(plan))) == plan
    assert plan["runtime_consumable"] is False
    assert plan["labels_exposed_to_runtime"] is False

    resolutions = {
        row.clip_id: (
            "https://videos.nba.com/nba/pbp/media/2025/01/01/"
            f"{row.game_id}/{row.event_id}/{row.clip_id}_1280x720.mp4"
        )
        for row in rows
    }
    manifest = seal_bard_video_resolution_manifest(
        plan=plan,
        video_urls=resolutions,
    )
    assert manifest["plan_sha256"] == plan["plan_sha256"]

    incomplete = dict(resolutions)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(ValueError, match="exactly"):
        seal_bard_video_resolution_manifest(
            plan=plan,
            video_urls=incomplete,
        )

    bad_host = dict(resolutions)
    bad_host[next(iter(bad_host))] = "https://example.com/video.mp4"
    with pytest.raises(ValueError, match="NBA"):
        seal_bard_video_resolution_manifest(
            plan=plan,
            video_urls=bad_host,
        )

    plan["examples"][0]["game_id"] = "0049400071"
    with pytest.raises(ValueError, match="hash mismatch|blind"):
        verify_bard_event_state_plan(plan)


def test_materialized_subset_verifies_every_downloaded_clip(tmp_path: Path) -> None:
    rows = select_bard_event_state_rows(
        parse_bard_dataset_csv(_csv_fixture()),
        per_class=1,
        seed=5,
    )
    plan = build_bard_event_state_plan(
        rows,
        source_revision="c" * 40,
        source_csv_sha256="d" * 64,
        per_class=1,
        seed=5,
    )
    resolutions = seal_bard_video_resolution_manifest(
        plan=plan,
        video_urls={
            row.clip_id: (
                "https://videos.nba.com/nba/pbp/media/2025/01/01/"
                f"{row.game_id}/{row.event_id}/{row.clip_id}_1280x720.mp4"
            )
            for row in rows
        },
    )

    subset = materialize_bard_event_state_subset(
        plan=plan,
        resolution_manifest=resolutions,
        output_dir=tmp_path,
        fetch_video=lambda url: b"\x00\x00\x00\x18ftypmp42" + url.encode(),
    )
    verified = verify_bard_event_state_subset(
        json.loads(json.dumps(subset)),
        root=tmp_path,
    )

    assert verified["runtime_consumable"] is False
    assert len(verified["clips"]) == 3
    assert all((tmp_path / row["relative_path"]).is_file() for row in verified["clips"])

    clip = tmp_path / verified["clips"][0]["relative_path"]
    clip.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash|MP4"):
        verify_bard_event_state_subset(verified, root=tmp_path)


def test_materialization_rejects_duplicate_media_across_distinct_events(
    tmp_path: Path,
) -> None:
    rows = select_bard_event_state_rows(
        parse_bard_dataset_csv(_csv_fixture()),
        per_class=1,
        seed=9,
    )
    plan = build_bard_event_state_plan(
        rows,
        source_revision="e" * 40,
        source_csv_sha256="f" * 64,
        per_class=1,
        seed=9,
    )
    resolutions = seal_bard_video_resolution_manifest(
        plan=plan,
        video_urls={
            row.clip_id: (
                "https://videos.nba.com/nba/pbp/media/2025/01/01/"
                f"{row.game_id}/{row.event_id}/{row.clip_id}_1280x720.mp4"
            )
            for row in rows
        },
    )

    with pytest.raises(ValueError, match="duplicate media"):
        materialize_bard_event_state_subset(
            plan=plan,
            resolution_manifest=resolutions,
            output_dir=tmp_path,
            fetch_video=lambda _: b"\x00\x00\x00\x18ftypmp42same-video",
        )


def _benchmark_fixture() -> str:
    return (
        "files,actions_name,number_actions\n"
        "../validation/2024/multi/aaa-vs-bbb-0022300001_10.mp4,"
        "\"[{'player': '1', 'action': '2PT Shot', 'result': True, "
        "'assisted': False, 'other_player': None, 'color': 'blue'}]\",1\n"
        "../validation/2024/multi/ccc-vs-ddd-0022300002_20.mp4,"
        "\"[{'player': '2', 'action': '3PT Shot', 'result': False, "
        "'assisted': False, 'other_player': None, 'color': 'white'}]\",1\n"
        "../validation/2024/multi/aaa-vs-bbb-0022300001_30.mp4,"
        "\"[{'player': '3', 'action': 'Free Throw', 'result': True, "
        "'assisted': None, 'other_player': None, 'color': 'blue'}]\",1\n"
        "../validation/2024/multi/ccc-vs-ddd-0022300002_40.mp4,"
        "\"[{'player': '4', 'action': 'Free Throw', 'result': False, "
        "'assisted': None, 'other_player': None, 'color': 'white'}]\",1\n"
    )


def test_parse_embedded_benchmark_and_balanced_selection() -> None:
    rows = parse_bard_benchmark_csv(_benchmark_fixture(), year=2024)

    assert [row.repository_path for row in rows] == [
        "validation/2024/multi/aaa-vs-bbb-0022300001_10.mp4",
        "validation/2024/multi/ccc-vs-ddd-0022300002_20.mp4",
        "validation/2024/multi/aaa-vs-bbb-0022300001_30.mp4",
        "validation/2024/multi/ccc-vs-ddd-0022300002_40.mp4",
    ]
    assert [row.event_state for row in rows] == [
        "live_field_goal",
        "live_field_goal",
        "free_throw",
        "free_throw",
    ]
    selected = select_bard_embedded_state_rows(
        reversed(rows),
        per_class=2,
        seed=11,
    )
    assert len(selected) == 4
    assert all(
        len({row.game_id for row in selected if row.event_state == state}) == 2
        for state in ("live_field_goal", "free_throw")
    )


def test_embedded_plan_requires_exact_git_blob_identity() -> None:
    rows = parse_bard_benchmark_csv(_benchmark_fixture(), year=2024)
    selected = select_bard_embedded_state_rows(rows, per_class=1, seed=4)
    tree = {
        row.repository_path: {
            "sha": "a" * 40 if index == 0 else "b" * 40,
            "size": 123 + index,
        }
        for index, row in enumerate(selected)
    }

    plan = build_bard_embedded_video_plan(
        selected,
        tree_entries=tree,
        source_revision="c" * 40,
        benchmark_sha256s={"2024": "d" * 64},
        per_class=1,
        seed=4,
    )

    assert len(plan["examples"]) == 2
    assert plan["runtime_consumable"] is False
    assert all(row["git_blob_sha1"] in {"a" * 40, "b" * 40} for row in plan["examples"])

    with pytest.raises(ValueError, match="tree"):
        build_bard_embedded_video_plan(
            selected,
            tree_entries={},
            source_revision="c" * 40,
            benchmark_sha256s={"2024": "d" * 64},
            per_class=1,
            seed=4,
        )


def test_embedded_subset_checks_git_blob_sha_and_unique_media(
    tmp_path: Path,
) -> None:
    rows = parse_bard_benchmark_csv(_benchmark_fixture(), year=2024)
    selected = select_bard_embedded_state_rows(rows, per_class=1, seed=8)
    payloads = {
        row.repository_path: (
            b"\x00\x00\x00\x18ftypmp42" + row.clip_id.encode()
        )
        for row in selected
    }

    def git_blob_sha(payload: bytes) -> str:
        import hashlib

        return hashlib.sha1(
            f"blob {len(payload)}\0".encode() + payload
        ).hexdigest()

    plan = build_bard_embedded_video_plan(
        selected,
        tree_entries={
            row.repository_path: {
                "sha": git_blob_sha(payloads[row.repository_path]),
                "size": len(payloads[row.repository_path]),
            }
            for row in selected
        },
        source_revision="e" * 40,
        benchmark_sha256s={"2024": "f" * 64},
        per_class=1,
        seed=8,
    )
    by_url = {
        row["download_url"]: payloads[row["repository_path"]]
        for row in plan["examples"]
    }
    subset = materialize_bard_embedded_subset(
        plan=plan,
        output_dir=tmp_path,
        fetch_video=by_url.__getitem__,
    )
    assert verify_bard_embedded_subset(
        json.loads(json.dumps(subset)),
        root=tmp_path,
    ) == subset

    with pytest.raises(ValueError, match="size|Git blob"):
        materialize_bard_embedded_subset(
            plan=plan,
            output_dir=tmp_path / "bad",
            fetch_video=lambda _: b"\x00\x00\x00\x18ftypmp42wrong",
        )
