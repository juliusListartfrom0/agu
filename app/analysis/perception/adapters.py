from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Protocol

from app.analysis.schemas import PerceptionDetectionResponse


class DetectorAdapter(Protocol):
    """Backend-neutral detector contract; external model details stop here."""

    @property
    def name(self) -> str: ...

    def available(self) -> bool: ...

    def detect(self, frames: Sequence[Any], frame_numbers: Sequence[int]) -> Iterable[PerceptionDetectionResponse]: ...


class PoseAdapter(Protocol):
    """Backend-neutral human-pose contract used by action ownership models."""

    @property
    def name(self) -> str: ...

    def available(self) -> bool: ...

    def estimate(
        self, frames: Sequence[Any], frame_numbers: Sequence[int]
    ) -> Iterable[PerceptionDetectionResponse]: ...


class PerceptionAdapterRegistry:
    def __init__(self) -> None:
        self._detectors: dict[str, DetectorAdapter] = {}

    def register(self, detector: DetectorAdapter) -> None:
        if not detector.name:
            raise ValueError("detector adapter must have a name")
        if detector.name in self._detectors:
            raise ValueError(f"detector adapter already registered: {detector.name}")
        self._detectors[detector.name] = detector

    def get(self, name: str) -> DetectorAdapter:
        try:
            detector = self._detectors[name]
        except KeyError as exc:
            raise ValueError(f"unknown detector adapter: {name}") from exc
        if not detector.available():
            raise RuntimeError(f"detector adapter is unavailable: {name}")
        return detector

    def diagnostics(self) -> dict[str, bool]:
        return {name: detector.available() for name, detector in sorted(self._detectors.items())}
