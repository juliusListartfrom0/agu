from scripts.prefill_official_jersey_cache import (
    _candidate_source_counts,
    _select_tracklets,
)


def test_candidate_source_counts_are_bound_to_requested_perception_namespace() -> None:
    payload = {
        "schema_version": "agu.raw-only.v1",
        "events": [
            {
                "evidence": [
                    {
                        "details": {
                            "candidate_player_ids": [
                                "perception-3:raw-player-track-7",
                                "perception-2:raw-player-track-7",
                            ],
                            "candidate_player_observations": [
                                {"player_id": "perception-3:raw-player-track-9"},
                            ],
                        }
                    }
                ]
            },
            {
                "evidence": [
                    {
                        "details": {
                            "candidate_player_ids": [
                                "perception-3:raw-player-track-7",
                                "perception-3:raw-player-track-7",
                            ]
                        }
                    }
                ]
            },
        ],
    }

    counts = _candidate_source_counts(payload, source_prefix="perception-3:")

    assert counts == {"raw-player-track-7": 2, "raw-player-track-9": 1}


def test_select_tracklets_prefers_event_relevance_and_one_best_segment_per_source() -> None:
    identity_payload = {
        "schema_version": "agu.official-identity.v1",
        "tracklets": [
            {
                "tracklet_id": "video_001:raw-player-track-7:001",
                "source_player_id": "raw-player-track-7",
                "observation_count": 50,
                "crop_count": 8,
                "sampled_boxes": [
                    {"frame": 1, "x1": 0, "y1": 0, "x2": 20, "y2": 80},
                    {"frame": 2, "x1": 0, "y1": 0, "x2": 20, "y2": 80},
                    {"frame": 3, "x1": 0, "y1": 0, "x2": 20, "y2": 80},
                    {"frame": 4, "x1": 0, "y1": 0, "x2": 20, "y2": 80},
                ],
                "gallery_person_id": None,
            },
            {
                "tracklet_id": "video_001:raw-player-track-7:002",
                "source_player_id": "raw-player-track-7",
                "observation_count": 80,
                "crop_count": 8,
                "sampled_boxes": [
                    {"frame": 5, "x1": 0, "y1": 0, "x2": 30, "y2": 120},
                    {"frame": 6, "x1": 0, "y1": 0, "x2": 30, "y2": 120},
                    {"frame": 7, "x1": 0, "y1": 0, "x2": 30, "y2": 120},
                    {"frame": 8, "x1": 0, "y1": 0, "x2": 30, "y2": 120},
                ],
                "gallery_person_id": None,
            },
            {
                "tracklet_id": "video_001:raw-player-track-9:001",
                "source_player_id": "raw-player-track-9",
                "observation_count": 120,
                "crop_count": 8,
                "sampled_boxes": [
                    {"frame": 9, "x1": 0, "y1": 0, "x2": 40, "y2": 150},
                    {"frame": 10, "x1": 0, "y1": 0, "x2": 40, "y2": 150},
                    {"frame": 11, "x1": 0, "y1": 0, "x2": 40, "y2": 150},
                    {"frame": 12, "x1": 0, "y1": 0, "x2": 40, "y2": 150},
                ],
                "gallery_person_id": None,
            },
            {
                "tracklet_id": "video_001:raw-player-track-11:001",
                "source_player_id": "raw-player-track-11",
                "observation_count": 200,
                "crop_count": 8,
                "sampled_boxes": [
                    {"frame": 13, "x1": 0, "y1": 0, "x2": 60, "y2": 200},
                    {"frame": 14, "x1": 0, "y1": 0, "x2": 60, "y2": 200},
                    {"frame": 15, "x1": 0, "y1": 0, "x2": 60, "y2": 200},
                    {"frame": 16, "x1": 0, "y1": 0, "x2": 60, "y2": 200},
                ],
                "gallery_person_id": "already-registered",
            },
            {
                "tracklet_id": "video_001:raw-player-track-13:001",
                "source_player_id": "raw-player-track-13",
                "observation_count": 300,
                "crop_count": 3,
                "sampled_boxes": [
                    {"frame": 5, "x1": 0, "y1": 0, "x2": 80, "y2": 240},
                ],
                "gallery_person_id": None,
            },
            {
                "tracklet_id": "video_001:raw-player-track-15:001",
                "source_player_id": "raw-player-track-15",
                "observation_count": 20,
                "crop_count": 4,
                "sampled_boxes": [
                    {"frame": 17, "x1": 0, "y1": 0, "x2": 100, "y2": 400},
                    {"frame": 18, "x1": 0, "y1": 0, "x2": 100, "y2": 400},
                    {"frame": 19, "x1": 0, "y1": 0, "x2": 100, "y2": 400},
                    {"frame": 20, "x1": 0, "y1": 0, "x2": 100, "y2": 400},
                ],
                "gallery_person_id": None,
            },
        ],
    }

    selected = _select_tracklets(
        identity_payload,
        candidate_counts={
            "raw-player-track-7": 3,
            "raw-player-track-9": 2,
            "raw-player-track-15": 1,
        },
        maximum_tracklets=2,
        minimum_crops=4,
    )

    assert [item["tracklet_id"] for item in selected] == [
        "video_001:raw-player-track-15:001",
        "video_001:raw-player-track-7:002",
    ]
