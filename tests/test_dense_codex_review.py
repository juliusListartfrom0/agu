import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.analysis.review import (
    DenseReviewError,
    import_dense_review,
    seal_dense_game_reviews,
    seal_dense_review_result,
)
from scripts.build_dense_codex_review import build_dense_review_package


def test_dense_review_rejects_invalid_window_before_opening_video(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        build_dense_review_package(
            video_path=tmp_path / "missing.mov",
            output_dir=tmp_path / "review",
            window_sec=0,
            sample_fps=1,
        )


def test_dense_review_requires_every_window_and_seals_confirmed_event(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "period1.mp4", fps=10, frame_count=20)
    package = tmp_path / "review"
    manifest = build_dense_review_package(
        video_path=video,
        output_dir=package,
        window_sec=1,
        sample_fps=2,
        panel_width=64,
    )
    manifest_hash = _json_sha256(manifest)
    first, second = manifest["windows"]
    decisions = [
        {
            "decision_id": "d1",
            "window_id": first["window_id"],
            "reviewer": "codex-test",
            "input_sha256": manifest_hash,
            "events": [
                {
                    "event_id": "shot-1",
                    "event_type": "field_goal_attempt",
                    "start_sec": 0.2,
                    "end_sec": 0.8,
                    "team_id": "dark",
                    "primary_player_id": "p1",
                    "shot_value": 2,
                    "outcome": "made",
                    "status": "codex_confirmed",
                    "confidence": 0.97,
                    "evidence_times_sec": [0.5],
                }
            ],
        },
        {
            "decision_id": "d2",
            "window_id": second["window_id"],
            "reviewer": "codex-test",
            "input_sha256": manifest_hash,
            "no_event": True,
        },
    ]
    _write_jsonl(package / "codex_events.jsonl", decisions)

    result = import_dense_review(package_dir=package, raw_video_path=video)
    bundle, score = seal_dense_review_result(
        game_id="g1",
        raw_video_path=video,
        result=result,
        config={"pipeline": "test"},
        expected_team_points={"dark": 2},
    )

    assert result.complete_video_coverage is True
    assert result.reviewed_window_count == result.total_window_count == 2
    assert bundle.events[0].evidence[0].details["window_id"] == first["window_id"]
    assert bundle.model_provenance["review_manifest_sha256"] == manifest_hash
    assert score.status == "official"
    assert score.players[0].points == 2


def test_dense_review_rejects_missing_window_and_out_of_window_event(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "period1.mp4", fps=10, frame_count=20)
    package = tmp_path / "review"
    manifest = build_dense_review_package(
        video_path=video,
        output_dir=package,
        window_sec=1,
        sample_fps=2,
        panel_width=64,
    )
    manifest_hash = _json_sha256(manifest)
    decision = {
        "decision_id": "d1",
        "window_id": manifest["windows"][0]["window_id"],
        "reviewer": "codex-test",
        "input_sha256": manifest_hash,
        "events": [
            {
                "event_id": "shot-1",
                "event_type": "field_goal_attempt",
                "start_sec": 0.5,
                "end_sec": 1.2,
                "status": "needs_review",
            }
        ],
    }
    _write_jsonl(package / "codex_events.jsonl", [decision])

    with pytest.raises(DenseReviewError, match="unreviewed windows"):
        import_dense_review(package_dir=package, raw_video_path=video)
    with pytest.raises(DenseReviewError, match="outside"):
        import_dense_review(package_dir=package, raw_video_path=video, require_all_windows=False)


def test_allowed_incomplete_window_decisions_still_fail_complete_coverage(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "period1.mp4", fps=10, frame_count=20)
    package = tmp_path / "review"
    manifest = build_dense_review_package(
        video_path=video,
        output_dir=package,
        window_sec=1,
        sample_fps=2,
        panel_width=64,
    )
    _write_jsonl(
        package / "codex_events.jsonl",
        [
            {
                "decision_id": "d1",
                "window_id": manifest["windows"][0]["window_id"],
                "reviewer": "codex-test",
                "input_sha256": _json_sha256(manifest),
                "no_event": True,
            }
        ],
    )

    result = import_dense_review(
        package_dir=package,
        raw_video_path=video,
        require_all_windows=False,
    )
    bundle, score = seal_dense_review_result(
        game_id="g1", raw_video_path=video, result=result, config={}
    )

    assert result.reviewed_window_count == 1
    assert result.total_window_count == 2
    assert result.complete_video_coverage is False
    assert score.status == "needs_review"
    assert score.reconciliation.valid is False
    assert score.reconciliation.issues[-1].code == "incomplete_video_coverage"
    assert bundle.model_provenance["review_complete_video_coverage"] == "false"
    assert bundle.model_provenance["reviewed_window_count"] == "1"
    assert bundle.model_provenance["total_window_count"] == "2"


def test_dense_review_allows_event_across_adjacent_reviewed_windows(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "period1.mp4", fps=10, frame_count=20)
    package = tmp_path / "review"
    manifest = build_dense_review_package(
        video_path=video,
        output_dir=package,
        window_sec=1,
        sample_fps=2,
        panel_width=64,
    )
    manifest_hash = _json_sha256(manifest)
    first, second = manifest["windows"]
    _write_jsonl(
        package / "codex_events.jsonl",
        [
            {
                "decision_id": "d1",
                "window_id": first["window_id"],
                "reviewer": "codex-test",
                "input_sha256": manifest_hash,
                "events": [
                    {
                        "event_id": "boundary-shot",
                        "event_type": "field_goal_attempt",
                        "start_sec": 0.8,
                        "end_sec": 1.2,
                        "supporting_window_ids": [second["window_id"]],
                        "status": "needs_review",
                    }
                ],
            },
            {
                "decision_id": "d2",
                "window_id": second["window_id"],
                "reviewer": "codex-test",
                "input_sha256": manifest_hash,
                "no_event": True,
            },
        ],
    )

    result = import_dense_review(package_dir=package, raw_video_path=video)

    evidence = result.events[0].evidence[0]
    assert evidence.start_frame == 8
    assert evidence.end_frame == 12
    assert evidence.details["supporting_window_ids"] == [first["window_id"], second["window_id"]]


def test_partial_dense_review_cannot_be_marked_official(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "period1.mp4", fps=10, frame_count=30)
    package = tmp_path / "review"
    manifest = build_dense_review_package(
        video_path=video,
        output_dir=package,
        window_sec=1,
        sample_fps=2,
        start_sec=1,
        duration_sec=1,
        panel_width=64,
    )
    _write_jsonl(
        package / "codex_events.jsonl",
        [
            {
                "decision_id": "d1",
                "window_id": manifest["windows"][0]["window_id"],
                "reviewer": "codex-test",
                "input_sha256": _json_sha256(manifest),
                "no_event": True,
            }
        ],
    )

    result = import_dense_review(package_dir=package, raw_video_path=video)
    _, score = seal_dense_review_result(game_id="g1", raw_video_path=video, result=result, config={})

    assert result.complete_video_coverage is False
    assert score.status == "needs_review"
    assert score.reconciliation.valid is False
    assert score.reconciliation.issues[-1].code == "incomplete_video_coverage"


def test_dense_steal_candidate_gets_reviewable_turnover_companion(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "period1.mp4", fps=10, frame_count=10)
    package = tmp_path / "review"
    manifest = build_dense_review_package(
        video_path=video,
        output_dir=package,
        window_sec=1,
        sample_fps=2,
        panel_width=64,
    )
    _write_jsonl(
        package / "codex_events.jsonl",
        [
            {
                "decision_id": "d1",
                "window_id": manifest["windows"][0]["window_id"],
                "reviewer": "codex-test",
                "input_sha256": _json_sha256(manifest),
                "events": [
                    {
                        "event_id": "steal-candidate",
                        "event_type": "steal",
                        "start_sec": 0.2,
                        "end_sec": 0.8,
                        "status": "needs_review",
                    }
                ],
            }
        ],
    )

    result = import_dense_review(package_dir=package, raw_video_path=video)

    assert [event.event_type for event in result.events] == ["turnover", "steal"]
    turnover, steal = result.events
    assert steal.related_event_ids == [turnover.event_id]
    assert turnover.status == steal.status == "needs_review"
    assert turnover.primary_player_id is None and turnover.team_id is None


def test_dense_game_review_remaps_each_raw_video_and_evidence(tmp_path: Path) -> None:
    pairs = []
    for index in (1, 2):
        video = _write_video(tmp_path / f"period{index}.mp4", fps=10, frame_count=10)
        package = tmp_path / f"review{index}"
        manifest = build_dense_review_package(
            video_path=video,
            output_dir=package,
            window_sec=1,
            sample_fps=2,
            panel_width=64,
        )
        event = {
            "event_id": f"shot-{index}",
            "event_type": "field_goal_attempt",
            "start_sec": 0.2,
            "end_sec": 0.8,
            "status": "needs_review",
        }
        _write_jsonl(
            package / "codex_events.jsonl",
            [
                {
                    "decision_id": f"d{index}",
                    "window_id": manifest["windows"][0]["window_id"],
                    "reviewer": "codex-test",
                    "input_sha256": _json_sha256(manifest),
                    "events": [event],
                }
            ],
        )
        pairs.append((package, video))

    result = seal_dense_game_reviews(game_id="g1", package_video_pairs=pairs, config={})

    assert result.complete_video_coverage is True
    assert [event.source_video_id for event in result.events] == ["video_001", "video_002"]
    assert [event.evidence[0].source_video_id for event in result.events] == ["video_001", "video_002"]
    assert [asset.filename for asset in result.bundle.raw_videos] == ["period1.mp4", "period2.mp4"]


def _write_video(path: Path, *, fps: int, frame_count: int) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 36))
    if not writer.isOpened():
        pytest.skip("OpenCV mp4 writer is unavailable")
    for index in range(frame_count):
        frame = np.full((36, 64, 3), index * 5 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
