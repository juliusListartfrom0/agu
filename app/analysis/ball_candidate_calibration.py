"""Leakage-safe nonlinear features for offline ball-candidate calibration.

The feature builder only transforms detector and track metadata already present
in a perception artifact.  It never reads pixels, labels, or runtime state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from app.analysis.ball_candidate_verifier import BALL_TRACK_FEATURE_NAMES

CALIBRATION_FEATURE_NAMES = BALL_TRACK_FEATURE_NAMES + (
    "logit_detection_confidence",
    "log_candidate_count_same_frame",
    "log_track_visible_points",
    "track_visibility_ratio",
    "track_span_per_visible_point",
    "speed_displacement_ratio",
    "speed_range_px_per_frame",
    "confidence_track_interaction",
    "aspect_area_interaction",
    "track_confidence_gap",
)


def _safe_logit(value: float) -> float:
    clipped = min(1.0 - 1e-6, max(1e-6, value))
    return float(np.log(clipped / (1.0 - clipped)))


def build_nonlinear_track_features(
    raw_names: Sequence[str],
    raw_by_id: Mapping[str, np.ndarray],
) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    """Add bounded nonlinear detector/track interactions to raw features.

    ``raw_by_id`` is expected to come from :func:`ball_track_features`.  The
    function deliberately requires the exact raw feature contract so a caller
    cannot silently mix incompatible perception artifacts.
    """

    if tuple(raw_names) != BALL_TRACK_FEATURE_NAMES:
        raise ValueError("unexpected ball-track feature names")
    output: dict[str, np.ndarray] = {}
    for detection_id, values in raw_by_id.items():
        vector = np.asarray(values, dtype=np.float64)
        if vector.ndim != 1 or vector.shape[0] != len(BALL_TRACK_FEATURE_NAMES):
            raise ValueError("feature width does not match ball-track contract")
        if not np.isfinite(vector).all():
            raise ValueError("ball-track features must be finite")
        (
            confidence,
            log_area,
            aspect,
            candidate_count,
            track_found,
            visible_points,
            total_points,
            span_frames,
            mean_confidence,
            max_confidence,
            mean_speed,
            max_speed,
            speed_std,
            displacement,
        ) = vector.tolist()
        derived = np.asarray(
            [
                _safe_logit(confidence),
                float(np.log1p(max(0.0, candidate_count))),
                float(np.log1p(max(0.0, visible_points))),
                visible_points / max(1.0, total_points),
                span_frames / max(1.0, visible_points),
                mean_speed / max(1e-6, displacement),
                max_speed - mean_speed,
                confidence * track_found,
                aspect * log_area,
                max_confidence - mean_confidence,
            ],
            dtype=np.float64,
        )
        combined = np.concatenate([vector, derived])
        if not np.isfinite(combined).all():
            raise ValueError("derived ball-track features must be finite")
        output[str(detection_id)] = combined
    return CALIBRATION_FEATURE_NAMES, output
