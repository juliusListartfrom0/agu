from __future__ import annotations

from scripts.calibrate_audio_score_candidates import calibrate, truth_player_for_delta


def test_truth_player_requires_one_exact_scoring_transition() -> None:
    play_by_play = [
        {
            "actionType": "period",
            "location": "h",
            "teamTricode": "BOS",
            "scoreHome": "0",
            "scoreAway": "0",
        },
        {
            "actionType": "Jump Ball",
            "location": "v",
            "teamTricode": "LAL",
            "scoreHome": "0",
            "scoreAway": "0",
        },
        {
            "actionType": "Made Shot",
            "location": "v",
            "teamTricode": "LAL",
            "personId": 977,
            "scoreHome": "0",
            "scoreAway": "2",
        },
    ]
    delta = {
        "team_id": "LAL",
        "points": 2,
        "before_scores": {"BOS": 0, "LAL": 0},
        "after_scores": {"BOS": 0, "LAL": 2},
    }

    assert truth_player_for_delta(delta, play_by_play) == "977"
    assert truth_player_for_delta({**delta, "points": 3}, play_by_play) is None


def test_truth_player_rejects_scoreboard_jump_spanning_multiple_scores() -> None:
    play_by_play = [
        {"actionType": "period", "location": "h", "teamTricode": "BOS"},
        {"actionType": "Jump Ball", "location": "v", "teamTricode": "LAL"},
        {
            "actionType": "Made Shot",
            "location": "v",
            "teamTricode": "LAL",
            "personId": 977,
            "scoreHome": "0",
            "scoreAway": "2",
        },
        {
            "actionType": "Made Shot",
            "location": "v",
            "teamTricode": "LAL",
            "personId": 1885,
            "scoreHome": "0",
            "scoreAway": "4",
        },
    ]
    merged_delta = {
        "team_id": "LAL",
        "points": 4,
        "before_scores": {"BOS": 0, "LAL": 0},
        "after_scores": {"BOS": 0, "LAL": 4},
    }

    assert truth_player_for_delta(merged_delta, play_by_play) is None


def test_calibration_does_not_promote_without_evidence() -> None:
    assert calibrate([], minimum_precision=0.85)["promoted"] is False
