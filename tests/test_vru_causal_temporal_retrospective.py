from __future__ import annotations

import copy

import numpy as np
import pytest

from app.analysis import vru_causal_temporal_retrospective as temporal_module
from app.analysis.vru_causal_temporal_retrospective import (
    TEMPORAL_REPRESENTATION_CONTRACT,
    build_temporal_retrospective,
    screen_temporal_outer_fold,
)


@pytest.fixture(autouse=True)
def _verified_fit_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        temporal_module,
        "observe_temporal_environment_contract",
        lambda: temporal_module.TEMPORAL_ENVIRONMENT_CONTRACT,
    )


GAME_COUNTS = {
    "hazen": (6, 2),
    "randolph": (4, 3),
    "vtv": (4, 3),
    "harwood": (3, 20),
}


def _synthetic_rows() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    labels: list[bool] = []
    games: list[str] = []
    features: list[list[float]] = []
    for game_index, (game, (positive_count, negative_count)) in enumerate(GAME_COUNTS.items()):
        for label, count in ((True, positive_count), (False, negative_count)):
            for row_index in range(count):
                labels.append(label)
                games.append(game)
                features.append(
                    [
                        (1.0 if label else -1.0) + game_index * 0.05,
                        float(row_index) / 100.0,
                        float(game_index),
                    ]
                )
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(labels, dtype=np.bool_),
        tuple(games),
    )


def _non_truth_projection(fold: dict[str, object]) -> dict[str, object]:
    projection = copy.deepcopy(fold)
    projection.pop("held_metrics")
    for row in projection["held_predictions"]:
        row.pop("truth")
    return projection


def test_outer_fold_threshold_and_predictions_ignore_held_labels() -> None:
    features, labels, games = _synthetic_rows()
    original = screen_temporal_outer_fold(
        feature_matrix=features,
        labels=labels,
        game_ids=games,
        held_game_id="harwood",
        representation_name=TEMPORAL_REPRESENTATION_CONTRACT["name"],
    )

    changed_labels = labels.copy()
    held_indexes = [index for index, game in enumerate(games) if game == "harwood"]
    positive_index = next(index for index in held_indexes if changed_labels[index])
    negative_index = next(index for index in held_indexes if not changed_labels[index])
    changed_labels[positive_index] = False
    changed_labels[negative_index] = True
    changed = screen_temporal_outer_fold(
        feature_matrix=features,
        labels=changed_labels,
        game_ids=games,
        held_game_id="harwood",
        representation_name=TEMPORAL_REPRESENTATION_CONTRACT["name"],
    )

    assert _non_truth_projection(original) == _non_truth_projection(changed)
    assert original["held_metrics"] != changed["held_metrics"]


def test_outer_fold_uses_frozen_grid_and_three_inner_logo_groups() -> None:
    features, labels, games = _synthetic_rows()
    fold = screen_temporal_outer_fold(
        feature_matrix=features,
        labels=labels,
        game_ids=games,
        held_game_id="hazen",
        representation_name=TEMPORAL_REPRESENTATION_CONTRACT["name"],
    )

    assert fold["held_game_id"] == "hazen"
    assert fold["fit_game_ids"] == ["randolph", "vtv", "harwood"]
    assert fold["inner_selection_game_ids"] == ["randolph", "vtv", "harwood"]
    assert len(fold["threshold_candidates"]) == 21
    assert [row["threshold"] for row in fold["threshold_candidates"]] == [index / 20 for index in range(21)]
    assert len(fold["held_predictions"]) == GAME_COUNTS["hazen"][0] + GAME_COUNTS["hazen"][1]
    assert fold["selected_threshold"] in [index / 20 for index in range(21)]


def test_retrospective_builds_all_four_folds_and_frozen_error_decision() -> None:
    _features, labels, games = _synthetic_rows()
    matrix = np.zeros((45, 1536), dtype=np.float64)
    matrix[:, 0] = np.where(labels, 2.0, -2.0)
    keys = [
        {
            "source_video_sha256": f"{list(GAME_COUNTS).index(game) + 1:064x}",
            "candidate_bundle_sha256": f"{list(GAME_COUNTS).index(game) + 11:064x}",
            "event_id": f"{game}-{index:04d}",
        }
        for index, game in enumerate(games)
    ]

    result = build_temporal_retrospective(
        feature_matrix=matrix,
        labels=labels,
        game_ids=games,
        ordered_keys=keys,
        input_receipts={"plan": "1" * 64},
        archived_task0257_comparator={"artifact_sha256": "2" * 64},
    )

    assert [fold["held_game_id"] for fold in result["outer_folds"]] == list(GAME_COUNTS)
    assert result["observed_metrics"]["pooled"]["fp"] == 0
    assert result["observed_metrics"]["pooled"]["fn"] == 0
    assert result["error_bound_result"]["passed"] is True
    assert result["temporal_hypothesis_decision"] == "within_frozen_error_bounds"
    assert result["formal_evaluation_eligible"] is False
