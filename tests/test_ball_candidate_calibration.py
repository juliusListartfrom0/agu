from __future__ import annotations

import numpy as np

from app.analysis.ball_candidate_calibration import (
    CALIBRATION_FEATURE_NAMES,
    build_nonlinear_track_features,
)
from app.analysis.ball_candidate_verifier import BALL_TRACK_FEATURE_NAMES
from scripts.screen_nonlinear_ball_calibration import _fit_variant


def test_nonlinear_track_features_are_named_and_finite() -> None:
    raw = {
        "detection-1": np.asarray(
            [0.4, 2.0, 1.2, 3.0, 1.0, 4.0, 6.0, 12.0, 0.4, 0.8, 2.0, 4.0, 1.0, 3.0],
            dtype=np.float64,
        )
    }

    names, derived = build_nonlinear_track_features(
        BALL_TRACK_FEATURE_NAMES,
        raw,
    )

    assert names == CALIBRATION_FEATURE_NAMES
    assert len(names) == derived["detection-1"].shape[0]
    assert np.isfinite(derived["detection-1"]).all()


def test_nonlinear_track_features_reject_unknown_width() -> None:
    try:
        build_nonlinear_track_features(
            BALL_TRACK_FEATURE_NAMES,
            {"bad": np.zeros(len(BALL_TRACK_FEATURE_NAMES) - 1)},
        )
    except ValueError as exc:
        assert "feature width" in str(exc)
    else:
        raise AssertionError("expected a feature-width validation error")


def test_grouped_calibration_fits_without_saving_checkpoint() -> None:
    source_x = np.asarray(
        [
            [0.1, 0.0],
            [0.8, 1.0],
            [0.2, 0.1],
            [0.9, 1.1],
            [0.3, 0.2],
            [1.0, 1.2],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    groups = np.asarray(["g1", "g1", "g2", "g2", "g3", "g3"])

    evaluation, scores = _fit_variant(
        "nonlinear_logistic",
        source_x,
        labels,
        groups,
        source_x,
        recall_floor=0.5,
        tree_count=8,
    )

    assert evaluation["checkpoint_saved"] is False
    assert evaluation["selection"]["recall"] >= 0.5
    assert scores.shape == (6,)
