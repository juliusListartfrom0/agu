"""Conservative clustering for benchmark-disjoint face enrollment candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class EnrollmentFaceObservation:
    source_video_id: str
    frame: int
    bbox: tuple[int, int, int, int]
    confidence: float
    embedding: np.ndarray
    crop: np.ndarray


def cluster_enrollment_faces(
    observations: Sequence[EnrollmentFaceObservation],
    *,
    minimum_cosine: float = 0.55,
) -> list[list[EnrollmentFaceObservation]]:
    """Complete-link cluster faces while forbidding simultaneous-person merges."""

    if not 0.0 <= minimum_cosine <= 1.0:
        raise ValueError("minimum_cosine must be in [0,1]")
    groups: list[list[EnrollmentFaceObservation]] = []
    ordered = sorted(
        observations,
        key=lambda item: (-item.confidence, item.source_video_id, item.frame, item.bbox),
    )
    for observation in ordered:
        candidates: list[tuple[float, int]] = []
        for index, group in enumerate(groups):
            if any(
                item.source_video_id == observation.source_video_id and item.frame == observation.frame
                for item in group
            ):
                continue
            scores = [float(observation.embedding @ item.embedding) for item in group]
            minimum = min(scores)
            if minimum >= minimum_cosine:
                candidates.append((minimum, index))
        if candidates:
            _score, index = max(candidates, key=lambda item: (item[0], -item[1]))
            groups[index].append(observation)
        else:
            groups.append([observation])
    return sorted(
        groups,
        key=lambda group: (
            -len(group),
            group[0].source_video_id,
            min(item.frame for item in group),
        ),
    )


def select_enrollment_samples(
    observations: Sequence[EnrollmentFaceObservation],
    *,
    maximum: int,
) -> list[EnrollmentFaceObservation]:
    """Choose high-confidence samples spread across source frames and videos."""

    if maximum <= 0:
        raise ValueError("maximum must be positive")
    ranked = sorted(
        observations,
        key=lambda item: (-item.confidence, item.source_video_id, item.frame, item.bbox),
    )
    selected: list[EnrollmentFaceObservation] = []
    used_videos: set[str] = set()
    for item in ranked:
        if item.source_video_id not in used_videos:
            selected.append(item)
            used_videos.add(item.source_video_id)
            if len(selected) >= maximum:
                return selected
    for item in ranked:
        if not any(item is chosen for chosen in selected):
            selected.append(item)
            if len(selected) >= maximum:
                break
    return selected
