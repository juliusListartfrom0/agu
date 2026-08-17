from __future__ import annotations

from scripts.build_official_identity_graph import _candidate_player_ids


def test_candidate_player_ids_collects_ranked_and_observed_tracks() -> None:
    payload = {
        "schema_version": "agu.raw-only.v1",
        "events": [
            {
                "evidence": [
                    {
                        "details": {
                            "candidate_player_ids": ["raw-1", "raw-2"],
                            "candidate_player_observations": [
                                {"player_id": "raw-3"},
                                {"player_id": "raw-1"},
                            ],
                        }
                    }
                ]
            }
        ],
    }

    assert _candidate_player_ids(payload) == {"raw-1", "raw-2", "raw-3"}


def test_candidate_player_ids_can_select_and_strip_perception_namespace() -> None:
    payload = {
        "schema_version": "agu.raw-only.v1",
        "events": [
            {
                "evidence": [
                    {
                        "details": {
                            "candidate_player_ids": [
                                "raw-primary",
                                "perception-3:raw-five-fps",
                            ],
                            "candidate_player_observations": [
                                {"player_id": "perception-3:raw-observed"}
                            ],
                        }
                    }
                ]
            }
        ],
    }

    assert _candidate_player_ids(payload, source_prefix="perception-3:") == {
        "raw-five-fps",
        "raw-observed",
    }
