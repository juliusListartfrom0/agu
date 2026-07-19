from scripts.export_action_owner_review_sheets import _select_temporal_observations


def test_review_crop_sampling_preserves_temporal_coverage() -> None:
    observations = [
        {"frame": frame, "ball_player_distance": float(10 - frame)} for frame in range(10)
    ]

    selected = _select_temporal_observations(observations, maximum=3)

    assert [item["frame"] for item in selected] == [0, 4, 9]


def test_single_review_crop_prefers_closest_ball_observation() -> None:
    observations = [
        {"frame": 10, "ball_player_distance": 0.8},
        {"frame": 20, "ball_player_distance": 0.2},
        {"frame": 30, "ball_player_distance": 0.5},
    ]

    assert _select_temporal_observations(observations, maximum=1)[0]["frame"] == 20
