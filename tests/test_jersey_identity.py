from pathlib import Path

import numpy as np

from app.analysis.jersey_identity import CachedJerseyNumberReader
from app.analysis.schemas import JerseyNumberCandidateResponse


class CountingReader:
    model = "fixture-model"

    def __init__(self) -> None:
        self.calls = 0

    def read_jersey_number(
        self,
        frames: list[np.ndarray],
        scope: str = "",
    ) -> list[JerseyNumberCandidateResponse]:
        del frames, scope
        self.calls += 1
        return [JerseyNumberCandidateResponse(number="6", confidence=0.95, visible=True)]


def test_jersey_reader_cache_reuses_raw_crop_fingerprint(tmp_path: Path) -> None:
    delegate = CountingReader()
    cache_path = tmp_path / "jersey-cache.json"
    reader = CachedJerseyNumberReader(delegate, cache_path)
    frames = [np.zeros((12, 8, 3), dtype=np.uint8)]

    first = reader.read_jersey_number(frames, scope="track-1")
    second = reader.read_jersey_number(frames, scope="track-1")
    reloaded = CachedJerseyNumberReader(delegate, cache_path).read_jersey_number(
        frames,
        scope="track-1",
    )

    assert delegate.calls == 1
    assert first == second == reloaded
