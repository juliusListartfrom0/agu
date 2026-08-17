from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "download_public_research_media.py"
    spec = importlib.util.spec_from_file_location("download_public_research_media", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_video_jobs_deduplicates_and_omits_truth() -> None:
    module = _module()
    game = {
        "slug": "game-a",
        "youtube_id": "video-a",
        "youtube_url": "https://www.youtube.com/watch?v=video-a",
        "truth_artifacts": {"box_score_path": "forbidden-at-runtime"},
    }
    jobs = module.collect_video_jobs(
        {
            "benchmarks": [
                {"benchmark": game, "enrollment_games": [game]},
            ]
        }
    )

    assert jobs == [
        {
            "slug": "game-a",
            "youtube_id": "video-a",
            "youtube_url": "https://www.youtube.com/watch?v=video-a",
        }
    ]
    assert "truth_artifacts" not in jobs[0]


def test_complete_media_ignores_partial_files_and_retry_is_capped(
    tmp_path: Path,
) -> None:
    module = _module()
    (tmp_path / "video-a.f135.mp4.part").write_bytes(b"partial")
    (tmp_path / "video-a.f135.mp4").write_bytes(b"video-stream-only")
    (tmp_path / "video-a.f251.webm").write_bytes(b"audio-stream-only")
    assert module.find_complete_media(tmp_path, "video-a") is None

    complete = tmp_path / "video-a.mp4"
    complete.write_bytes(b"complete")
    assert module.find_complete_media(tmp_path, "video-a") == complete
    assert module.retry_delay(1, base_seconds=10, maximum_seconds=300) == 10
    assert module.retry_delay(20, base_seconds=10, maximum_seconds=300) == 300

    with pytest.raises(ValueError, match="attempt"):
        module.retry_delay(0, base_seconds=10, maximum_seconds=300)


def test_download_command_uses_finite_inner_retries_and_optional_ipv4() -> None:
    module = _module()

    command = module._download_command(
        "yt-dlp",
        output_dir=Path("media"),
        video_url="https://www.youtube.com/watch?v=video-a",
        maximum_height=480,
        retries=3,
        force_ipv4=True,
        external_downloader="curl",
    )

    assert command[1:3] == ["--downloader", "curl"]
    assert "--force-ipv4" in command
    assert "--retry-all-errors" in command[4]
    assert command[command.index("--retries") + 1] == "3"
    assert command[command.index("--fragment-retries") + 1] == "3"
    assert command[command.index("--file-access-retries") + 1] == "3"
    assert "infinite" not in command


def test_collect_video_jobs_supports_truth_free_archive_override() -> None:
    module = _module()
    jobs = module.collect_video_jobs(
        {
            "benchmarks": [
                {
                    "benchmark": {
                        "slug": "game-a",
                        "youtube_id": "source-video-a",
                        "youtube_url": "https://www.youtube.com/watch?v=source-video-a",
                        "truth_artifacts": {"box_score_path": "forbidden-at-runtime"},
                        "media_source": {
                            "provider": "internet_archive",
                            "identifier": "archive-item",
                            "filename": "complete game.ogv",
                            "url": (
                                "https://archive.org/download/archive-item/"
                                "complete%20game.ogv"
                            ),
                            "declared_license": None,
                            "match_basis": "date_teams_and_game_number",
                            "redistribution_permitted": False,
                        },
                    },
                    "enrollment_games": [],
                }
            ]
        }
    )

    assert jobs == [
        {
            "slug": "game-a",
            "youtube_id": "source-video-a",
            "youtube_url": "https://www.youtube.com/watch?v=source-video-a",
            "media_provider": "internet_archive",
            "media_url": (
                "https://archive.org/download/archive-item/"
                "complete%20game.ogv"
            ),
            "media_filename": "complete game.ogv",
        }
    ]
    assert "truth_artifacts" not in jobs[0]


def test_direct_download_command_is_resumable_and_uses_sealed_name(
    tmp_path: Path,
) -> None:
    module = _module()

    command = module._direct_download_command(
        "curl",
        output_path=tmp_path / "source-video-a.ogv",
        media_url=(
            "https://archive.org/download/archive-item/complete%20game.ogv"
        ),
        retries=4,
    )

    assert command[:3] == ["curl", "--location", "--fail"]
    assert command[command.index("--continue-at") + 1] == "-"
    assert command[command.index("--retry") + 1] == "4"
    assert command[command.index("--output") + 1].endswith("source-video-a.ogv")


def test_filter_video_jobs_selects_archive_without_weakening_plan() -> None:
    module = _module()
    jobs = [
        {
            "slug": "archive-game",
            "youtube_id": "archive-source-id",
            "youtube_url": "https://youtube.example/archive",
            "media_provider": "internet_archive",
            "media_url": "https://archive.org/download/item/game.ogv",
            "media_filename": "game.ogv",
        },
        {
            "slug": "youtube-game",
            "youtube_id": "youtube-id",
            "youtube_url": "https://youtube.example/youtube",
        },
    ]

    assert module.filter_video_jobs(jobs, only_provider="internet_archive") == [
        jobs[0]
    ]
    assert module.filter_video_jobs(jobs, only_provider=None) == jobs
    with pytest.raises(ValueError, match="no media jobs"):
        module.filter_video_jobs(jobs, only_provider="missing")
