from __future__ import annotations

import json

import pytest

from app.analysis.cross_game_shot_windows import seal_shot_window_spec
from scripts.build_cross_game_shot_review_artifacts import build_artifacts


def _spec(source_sha: str) -> dict[str, object]:
    return seal_shot_window_spec(
        {
            "schema_version": "agu.cross-game-shot-window-spec.v1",
            "purpose": "offline_training_annotation_only",
            "runtime_consumable": False,
            "source_video_sha256": source_sha,
            "source_video_filename": "clip.mp4",
            "video_frame_count": 200,
            "video_fps": 25.0,
            "video_width": 640,
            "video_height": 360,
            "selection_seed": 1,
            "selection_protocol": "seeded_uniform_nonoverlap_windows_v1",
            "window_frames": 20,
            "minimum_center_spacing_frames": 20,
            "label_selection_not_used": True,
            "candidates": [
                {
                    "event_id": "raw-shot-001",
                    "start_frame": 10,
                    "end_frame": 29,
                    "anchor_frame": 20,
                    "source_fps": 25.0,
                }
            ],
        }
    )


def test_build_artifacts_rejects_video_hash_mismatch(tmp_path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not-a-video")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(_spec("0" * 64)), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="hash-bound candidate spec"):
        build_artifacts(
            video_path=video,
            spec_path=spec_path,
            labels_path=tmp_path / "labels.json",
            base_manifest_path=tmp_path / "base.json",
            output_dir=tmp_path / "out",
            game_id="test-game",
        )


def test_build_artifacts_requires_boolean_labels(tmp_path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"clip")
    import hashlib

    source_sha = hashlib.sha256(video.read_bytes()).hexdigest()
    spec = _spec(source_sha)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "source_video_sha256": source_sha,
                "candidate_spec_sha256": spec["artifact_sha256"],
                "examples": [
                    {"event_id": "raw-shot-001", "event_present": None}
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="event_present label must be boolean"):
        build_artifacts(
            video_path=video,
            spec_path=spec_path,
            labels_path=labels_path,
            base_manifest_path=tmp_path / "base.json",
            output_dir=tmp_path / "out",
            game_id="test-game",
        )
