from __future__ import annotations

from pathlib import Path

import numpy as np

from app.analysis.schemas import JerseyNumberCandidateResponse
from scripts.enroll_face_gallery_jersey_numbers import _consensus_number, _spread_samples


class FixtureReader:
    model = "fixture"

    def __init__(self, numbers: list[str | None]) -> None:
        self.numbers = iter(numbers)

    def read_jersey_number(self, frames, scope=""):
        number = next(self.numbers)
        if number is None:
            return []
        return [
            JerseyNumberCandidateResponse(
                number=number,
                confidence=0.95,
                visible=True,
                reason=scope,
            )
        ]


def test_spread_samples_is_deterministic_and_deduplicates_frames() -> None:
    samples = [{"frame": frame, "path": str(Path(f"{frame}.jpg"))} for frame in [9, 1, 5, 5, 7]]

    selected = _spread_samples(samples, maximum_samples=3)

    assert [item["frame"] for item in selected] == [1, 5, 9]


def test_enrollment_jersey_requires_partition_consensus() -> None:
    crops = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(4)]

    assert _consensus_number(
        FixtureReader(["7", "7"]),
        crops,
        scope="player-7",
        minimum_confidence=0.90,
        minimum_consensus=2,
    ) == "7"
    assert _consensus_number(
        FixtureReader(["7", "1"]),
        crops,
        scope="ambiguous",
        minimum_confidence=0.90,
        minimum_consensus=2,
    ) is None
