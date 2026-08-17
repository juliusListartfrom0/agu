from __future__ import annotations

import io

import numpy as np
from PIL import Image

from app.analysis.e_bard_role_transfer import (
    choose_precision_threshold,
    image_role_features,
    split_group_indices,
)


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (12, 8), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_image_role_features_are_fixed_finite_and_color_sensitive() -> None:
    orange = image_role_features(_jpeg_bytes((220, 100, 20)))
    blue = image_role_features(_jpeg_bytes((20, 80, 180)))

    assert orange.shape == blue.shape
    assert orange.ndim == 1
    assert np.isfinite(orange).all()
    assert np.linalg.norm(orange - blue) > 0.1


def test_split_group_indices_keeps_groups_disjoint_and_covers_rows() -> None:
    groups = np.asarray(["a", "a", "b", "c", "c", "d"])

    train, valid = split_group_indices(groups, holdout_fraction=0.5)

    assert set(train).isdisjoint(set(valid))
    assert sorted((*train, *valid)) == list(range(len(groups)))
    assert set(groups[train]).isdisjoint(set(groups[valid]))


def test_choose_precision_threshold_prefers_highest_recall_above_gate() -> None:
    labels = np.asarray([True, True, False, False])
    scores = np.asarray([0.90, 0.60, 0.55, 0.10])

    threshold = choose_precision_threshold(labels, scores, minimum_precision=1.0)

    assert threshold == 0.60
