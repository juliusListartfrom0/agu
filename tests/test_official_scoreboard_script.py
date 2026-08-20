from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_official_scoreboard_evidence import _load_cache


def test_scoreboard_cache_is_resumable_when_header_matches(tmp_path: Path) -> None:
    path = tmp_path / "scoreboard-cache.json"
    header = {
        "schema_version": "agu.broadcast-scoreboard-cache.v1",
        "raw_video_sha256": "raw-sha",
        "team_ids": ["ATL", "CHI"],
    }
    path.write_text(json.dumps({**header, "frames": {"30": None}}), encoding="utf-8")

    loaded = _load_cache(path, expected_header=header)

    assert loaded["frames"] == {"30": None}


def test_scoreboard_cache_rejects_raw_video_or_config_drift(tmp_path: Path) -> None:
    path = tmp_path / "scoreboard-cache.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "agu.broadcast-scoreboard-cache.v1",
                "raw_video_sha256": "old-sha",
                "frames": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="raw_video_sha256"):
        _load_cache(
            path,
            expected_header={
                "schema_version": "agu.broadcast-scoreboard-cache.v1",
                "raw_video_sha256": "new-sha",
            },
        )


def test_scoreboard_cache_can_reuse_raw_reads_for_a_new_candidate_bundle(tmp_path: Path) -> None:
    path = tmp_path / "scoreboard-cache.json"
    header = {
        "schema_version": "agu.broadcast-scoreboard-cache.v1",
        "raw_video_sha256": "raw-sha",
        "team_ids": ["ATL", "CHI"],
    }
    path.write_text(
        json.dumps(
            {
                **header,
                "source_candidate_bundle_sha256": "older-candidate-sha",
                "frames": {"30": {"frame": 30, "scores": {"ATL": 2, "CHI": 0}}},
            }
        ),
        encoding="utf-8",
    )

    loaded = _load_cache(path, expected_header=header)

    assert loaded["frames"]["30"]["scores"] == {"ATL": 2, "CHI": 0}
