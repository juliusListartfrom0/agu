from __future__ import annotations

import numpy as np
import pytest

from app.analysis.face_enrollment import (
    EnrollmentFaceObservation,
    cluster_enrollment_faces,
    select_enrollment_samples,
)


def _observation(
    source: str,
    frame: int,
    vector: tuple[float, float],
    confidence: float = 0.9,
) -> EnrollmentFaceObservation:
    embedding = np.asarray(vector, dtype=np.float32)
    embedding /= np.linalg.norm(embedding)
    return EnrollmentFaceObservation(
        source_video_id=source,
        frame=frame,
        bbox=(1, 2, 30, 30),
        confidence=confidence,
        embedding=embedding,
        crop=np.zeros((112, 112, 3), dtype=np.uint8),
    )


def test_enrollment_clustering_uses_complete_link_similarity() -> None:
    first = _observation("a", 1, (1.0, 0.0))
    second = _observation("a", 2, (0.9, 0.436))
    bridge = _observation("a", 3, (0.55, 0.835))

    groups = cluster_enrollment_faces([first, second, bridge], minimum_cosine=0.7)

    assert sorted(len(group) for group in groups) == [1, 2]


def test_enrollment_clustering_never_merges_faces_from_same_frame() -> None:
    first = _observation("a", 1, (1.0, 0.0), confidence=0.95)
    second = _observation("a", 1, (1.0, 0.0), confidence=0.94)

    groups = cluster_enrollment_faces([first, second], minimum_cosine=0.5)

    assert [len(group) for group in groups] == [1, 1]


def test_enrollment_sample_selection_prefers_distinct_videos() -> None:
    items = [
        _observation("a", 1, (1.0, 0.0), confidence=0.99),
        _observation("a", 2, (1.0, 0.0), confidence=0.98),
        _observation("b", 1, (1.0, 0.0), confidence=0.80),
    ]

    selected = select_enrollment_samples(items, maximum=2)

    assert [item.source_video_id for item in selected] == ["a", "b"]


def test_enrollment_threshold_validation() -> None:
    with pytest.raises(ValueError, match="minimum_cosine"):
        cluster_enrollment_faces([], minimum_cosine=1.1)
