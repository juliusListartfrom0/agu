from __future__ import annotations

import hashlib
import json

import pytest

from app.analysis.face_player_context import (
    attach_face_player_context,
    select_team_consistent_candidates,
)


def _seal(payload: dict) -> dict:
    payload = {**payload, "manifest_sha256": ""}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    payload["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _candidates() -> dict:
    return _seal(
        {
            "schema_version": "agu.face-enrollment-candidates.v1",
            "benchmark_disjoint": True,
            "model_id": "test-face-model",
            "sources": [
                {
                    "source_video_id": "enrollment_001",
                    "filename": "game3.mp4",
                    "sha256": "video-3",
                }
            ],
            "config": {},
            "clusters": [
                {
                    "cluster_id": "face-001",
                    "samples": [
                        {
                            "path": "face-001-a.jpg",
                            "source_video_id": "enrollment_001",
                            "frame": 100,
                            "bbox": [30, 20, 10, 10],
                        },
                        {
                            "path": "face-001-b.jpg",
                            "source_video_id": "enrollment_001",
                            "frame": 105,
                            "bbox": [32, 21, 10, 10],
                        },
                    ],
                },
                {
                    "cluster_id": "face-002",
                    "samples": [
                        {
                            "path": "face-002-a.jpg",
                            "source_video_id": "enrollment_001",
                            "frame": 100,
                            "bbox": [80, 20, 10, 10],
                        }
                    ],
                },
            ],
        }
    )


def _perception() -> dict:
    return {
        "schema_version": "agu.official-perception.v1",
        "raw_video": {
            "filename": "game3.mp4",
            "sha256": "video-3",
        },
        "detections": [
            {
                "frame": 100,
                "object_type": "player",
                "bbox": {"x1": 20, "y1": 10, "x2": 60, "y2": 90},
                "confidence": 0.9,
                "track_id": "track-dark",
                "team_id": "raw-dark",
            },
            {
                "frame": 100,
                "object_type": "player",
                "bbox": {"x1": 0, "y1": 0, "x2": 200, "y2": 200},
                "confidence": 0.99,
                "track_id": "overlap",
                "team_id": "raw-light",
            },
            {
                "frame": 100,
                "object_type": "player",
                "bbox": {"x1": 70, "y1": 10, "x2": 110, "y2": 90},
                "confidence": 0.8,
                "track_id": "track-light",
                "team_id": "raw-light",
            },
            {
                "frame": 105,
                "object_type": "player",
                "bbox": {"x1": 20, "y1": 10, "x2": 60, "y2": 90},
                "confidence": 0.88,
                "track_id": "track-dark",
                "team_id": "raw-dark",
            },
        ],
    }


def test_face_context_prefers_smallest_containing_player_and_maps_team() -> None:
    context = attach_face_player_context(
        _candidates(),
        [_perception()],
        perception_sha256s=["perception-3"],
        team_id_map={
            "raw-dark": "1610612753",
            "raw-light": "1610612745",
        },
        minimum_team_observations=2,
        minimum_team_consensus=0.75,
    )

    first = context["clusters"][0]
    assert first["cluster_id"] == "face-001"
    assert first["matched_sample_count"] == 2
    assert first["dominant_team_id"] == "1610612753"
    assert first["dominant_team_fraction"] == 1.0
    assert first["track_ids"] == ["track-dark"]
    assert first["samples"][0]["track_id"] == "track-dark"
    assert first["samples"][0]["raw_team_id"] == "raw-dark"
    assert first["samples"][0]["team_id"] == "1610612753"

    second = context["clusters"][1]
    assert second["matched_sample_count"] == 1
    assert second["dominant_team_id"] is None
    assert context["manifest_sha256"]


def test_team_consistent_subset_is_sealed_and_preserves_only_matching_clusters() -> None:
    candidates = _candidates()
    context = attach_face_player_context(
        candidates,
        [_perception()],
        perception_sha256s=["perception-3"],
        team_id_map={
            "raw-dark": "1610612753",
            "raw-light": "1610612745",
        },
        minimum_team_observations=2,
        minimum_team_consensus=0.75,
    )

    subset = select_team_consistent_candidates(
        candidates,
        context,
        team_id="1610612753",
    )

    assert [row["cluster_id"] for row in subset["clusters"]] == ["face-001"]
    assert subset["config"]["team_consistent_selection"]["team_id"] == "1610612753"
    assert (
        subset["config"]["team_consistent_selection"]["source_context_manifest_sha256"]
        == context["manifest_sha256"]
    )
    assert subset["manifest_sha256"]


def test_face_context_rejects_unbound_perception_or_tampered_source() -> None:
    candidates = _candidates()
    bad_perception = _perception()
    bad_perception["raw_video"]["sha256"] = "different-video"

    with pytest.raises(ValueError, match="bind"):
        attach_face_player_context(
            candidates,
            [bad_perception],
            perception_sha256s=["perception-3"],
            team_id_map={"raw-dark": "1610612753"},
        )

    candidates["manifest_sha256"] = "tampered"
    with pytest.raises(ValueError, match="hash"):
        attach_face_player_context(
            candidates,
            [_perception()],
            perception_sha256s=["perception-3"],
            team_id_map={"raw-dark": "1610612753"},
        )
