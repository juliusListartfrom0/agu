from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from app.analysis.ball_release_annotation import seal_ball_release_plan
from scripts.render_ball_release_review_sheets import (
    _file_sha256,
    render_review_sheets,
)
from scripts.seal_ball_release_review import seal_review


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (320, 180),
    )
    assert writer.isOpened()
    for frame in range(10):
        image = np.full((180, 320, 3), frame * 20, dtype=np.uint8)
        writer.write(image)
    writer.release()


def test_review_sheets_preserve_raw_frames_without_exposing_labels(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "source.avi"
    _write_video(video_path)
    source_sha = _file_sha256(video_path)
    plan = seal_ball_release_plan(
        {
            "purpose": "codex_offline_ball_release_training_annotation_plan",
            "annotation_scope": "ball_visibility_path_and_release_geometry_only",
            "training_manifest_sha256": "manifest",
            "scene_artifact_sha256": "scene",
            "fusion_artifact_sha256": "fusion",
            "selection_protocol": "hidden_labels",
            "per_game_per_class": 1,
            "frame_fractions": [
                0.05,
                0.1625,
                0.275,
                0.3875,
                0.5,
                0.6125,
                0.725,
                0.8375,
                0.95,
            ],
            "selection_summary": {
                "games": 1,
                "examples": 1,
                "positive_hard_examples": 1,
                "negative_hard_examples": 0,
            },
            "examples": [
                {
                    "review_id": "ball-release-0001",
                    "source_video_sha256": source_sha,
                    "candidate_bundle_sha256": "bundle",
                    "event_id": "event",
                    "start_frame": 0,
                    "end_frame": 9,
                    "frame_numbers": list(range(9)),
                }
            ],
        }
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    manifest = render_review_sheets(
        plan_path=plan_path,
        video_paths=[video_path],
        detection_paths=[],
        output_dir=tmp_path / "sheets",
        panel_width=320,
    )

    assert manifest["plan_sha256"] == plan["artifact_sha256"]
    assert manifest["runtime_consumable"] is False
    assert "event_present" not in json.dumps(manifest)
    assert (tmp_path / "sheets/raw/ball-release-0001.jpg").is_file()
    assert (
        tmp_path / "sheets/selected_path/ball-release-0001.jpg"
    ).is_file()

    decisions = {
        "schema_version": "agu.ball-release-codex-decisions.v1",
        "runtime_consumable": False,
        "plan_sha256": plan["artifact_sha256"],
        "decisions": [
            {
                "review_id": "ball-release-0001",
                "release_observed": None,
                "ball_moves_toward_rim": None,
                "rim_arrival_observed": None,
                "free_throw_formation": False,
                "replay_or_stoppage": False,
                "review_confidence": "uncertain",
                "ball_visible_frames": [],
                "ball_not_visible_frames": [],
                "selected_path_correct_frames": [],
                "selected_path_incorrect_frames": [],
            }
        ],
    }
    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(json.dumps(decisions), encoding="utf-8")
    review = seal_review(
        plan_path=plan_path,
        sheet_manifest_path=tmp_path / "sheets/manifest.json",
        decisions_path=decisions_path,
    )
    assert review["reviews"][0]["frame_observations"][0][
        "ball_visible"
    ] is None
