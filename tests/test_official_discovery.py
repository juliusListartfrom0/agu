import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from app.analysis.official_discovery import (
    DiscoveredEvent,
    discovered_events_to_candidates,
    parse_discovered_events,
)
from app.analysis.schemas import BoundingBoxResponse, PerceptionDetectionResponse
from scripts.run_official_dense_vlm_discovery import run_dense_discovery


class CountingDiscoverer:
    name = "counting-discoverer"
    model = "fixture-model"

    def __init__(self) -> None:
        self.calls = 0

    def discover(self, frames: object, *, window_duration_sec: float) -> list[DiscoveredEvent]:
        self.calls += 1
        return [DiscoveredEvent("turnover", min(0.5, window_duration_sec), 0.8, "fixture")]


def test_dense_discovery_parser_rejects_unknown_and_out_of_window_events() -> None:
    events = parse_discovered_events(
        {
            "events": [
                {"event_type": "field_goal_attempt", "relative_time_sec": 2.5, "confidence": 0.9},
                {"event_type": "dunkish", "relative_time_sec": 3, "confidence": 1},
                {"event_type": "rebound", "relative_time_sec": 20, "confidence": 1},
            ]
        },
        window_duration_sec=8,
    )

    assert len(events) == 1
    assert events[0].event_type == "field_goal_attempt"


def test_dense_discovery_candidates_bind_nearby_traditional_identities() -> None:
    discovered = parse_discovered_events(
        {"events": [{"event_type": "rebound", "relative_time_sec": 2, "confidence": 0.8}]},
        window_duration_sec=8,
    )
    detection = PerceptionDetectionResponse(
        detection_id="player-1",
        frame=160,
        object_type="player",
        bbox=BoundingBoxResponse(x1=0, y1=0, x2=20, y2=40),
        confidence=0.9,
        track_id="track-7",
        player_id="raw-player-track-7",
        team_id="raw-dark",
        backend="traditional",
    )

    candidates = discovered_events_to_candidates(
        discovered,
        source_video_id="video_001",
        window_start_frame=100,
        fps=30,
        detections=[detection],
        event_id_prefix="window-1",
    )

    assert candidates[0].status == "needs_review"
    assert candidates[0].evidence[0].details["candidate_player_ids"] == ["raw-player-track-7"]
    assert candidates[0].evidence[0].details["candidate_team_ids"] == ["raw-dark"]


def test_dense_discovery_cache_resumes_without_repeating_vlm_calls(tmp_path: Path) -> None:
    video = tmp_path / "game.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
    assert writer.isOpened()
    for _ in range(20):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    perception = tmp_path / "perception.json"
    perception.write_text(
        json.dumps(
            {
                "schema_version": "agu.official-perception.v1",
                "raw_video": {
                    "filename": video.name,
                    "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                    "source_fps": 10.0,
                    "frame_count": 20,
                },
                "detector": {"model_sha256": "fixture"},
                "detections": [],
            }
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "discovery-cache.json"
    first = CountingDiscoverer()
    run_dense_discovery(
        video_path=video,
        perception_path=perception,
        game_id="game",
        discoverer=first,
        window_sec=1.0,
        overlap_sec=0.0,
        sample_fps=1.0,
        start_sec=0.0,
        end_sec=2.0,
        minimum_confidence=0.5,
        cache_path=cache,
    )
    assert first.calls == 2

    resumed = CountingDiscoverer()
    bundle = run_dense_discovery(
        video_path=video,
        perception_path=perception,
        game_id="game",
        discoverer=resumed,
        window_sec=1.0,
        overlap_sec=0.0,
        sample_fps=1.0,
        start_sec=0.0,
        end_sec=2.0,
        minimum_confidence=0.5,
        cache_path=cache,
    )
    assert resumed.calls == 0
    assert len(bundle.events) == 2
