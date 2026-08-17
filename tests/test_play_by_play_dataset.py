from __future__ import annotations

import json
from pathlib import Path

from app.analysis.play_by_play_dataset import (
    build_play_by_play_basketball_audit,
    verify_play_by_play_basketball_audit,
)


def _write_clip(root: Path, name: str, label: str, frames: list[dict]) -> None:
    path = root / "play-by-play-dataset" / "basketball" / "match-0" / label / name
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"field": {"left_area": []}, "frames": frames, "label": label}),
        encoding="utf-8",
    )


def test_play_by_play_audit_counts_sparse_ball_and_standardized_frames(tmp_path: Path) -> None:
    root = tmp_path / "annotations"
    _write_clip(
        root,
        "clip-0.json",
        "left_attack",
        [
            {
                "balls": [{"x": 0.2, "y": 0.3}],
                "last_balls": [{"x": 0.2, "y": 0.3}],
                "players": [{"x": 0.1, "y": 0.2, "x_velocity": 0.0, "y_velocity": 0.0}],
                "stopped_balls": [],
            },
            {
                "balls": [],
                "last_balls": [{"x": 0.2, "y": 0.3}],
                "players": [],
                "stopped_balls": [{"x": 0.2, "y": 0.3}],
            },
        ],
    )
    _write_clip(
        root,
        "clip-1.json",
        "timeout",
        [
            {
                "balls": [],
                "last_balls": [{"x": 0.0, "y": 0.0}],
                "players": [],
                "stopped_balls": [],
            }
        ],
    )
    (root / "play-by-play-dataset" / "basketball" / "info.json").write_text(
        json.dumps({"test_set": ["match-0"]}), encoding="utf-8"
    )

    audit = build_play_by_play_basketball_audit(
        root,
        source_url="https://example.invalid/play-by-play",
        source_archive_md5="a" * 32,
        source_archive_bytes=123,
    )

    assert audit["clip_count"] == 2
    assert audit["frame_count"] == 3
    assert audit["label_counts"] == {"left_attack": 1, "timeout": 1}
    assert audit["ball_observation_frame_count"] == 1
    assert audit["ball_observation_count"] == 1
    assert audit["stopped_ball_observation_count"] == 1
    assert audit["player_observation_count"] == 1
    assert audit["missing_test_match_ids"] == []
    assert audit["runtime_consumable"] is False
    assert verify_play_by_play_basketball_audit(audit)["artifact_sha256"] == audit["artifact_sha256"]
