from __future__ import annotations

from scripts.screen_deepball_large_windows import summarize_window_rows
from scripts.seal_deepball_large_screen import evaluate_hou_probe


def test_summarize_window_rows_uses_event_presence_as_ground_truth() -> None:
    rows = [
        {"event_present": True, "prediction_present": True},
        {"event_present": True, "prediction_present": False},
        {"event_present": False, "prediction_present": True},
        {"event_present": False, "prediction_present": False},
    ]

    metrics = summarize_window_rows(rows)

    assert metrics == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "tn": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "window_count": 4,
    }


def test_evaluate_hou_probe_uses_fixed_window_pbp_matches() -> None:
    probe = {
        "detections": [
            {"frame": 100, "detections": [{"score": 0.8}]},
            {"frame": 250, "detections": []},
        ]
    }
    base_screen = {
        "hou_sac_screen": {
            "interval": {
                "candidate_matches": [
                    {
                        "event_id": "positive",
                        "start_frame": 90,
                        "end_frame": 110,
                        "matched_shot_frames": [100],
                    },
                    {
                        "event_id": "negative",
                        "start_frame": 200,
                        "end_frame": 210,
                        "matched_shot_frames": [],
                    },
                ]
            }
        }
    }

    result = evaluate_hou_probe(probe, base_screen)

    assert result["metrics"]["tp"] == 1
    assert result["metrics"]["tn"] == 1
