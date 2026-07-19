"""Resumable AGU jersey-number recognition helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np

from app.analysis.schemas import JerseyNumberCandidateResponse

JERSEY_CACHE_SCHEMA = "agu.jersey-number-cache.v1"


class JerseyReader(Protocol):
    model: str

    def read_jersey_number(
        self,
        frames: Sequence[np.ndarray],
        scope: str = "",
    ) -> list[JerseyNumberCandidateResponse]: ...


class CachedJerseyNumberReader:
    """Cache raw-crop-bound VLM reads without storing images or manual answers."""

    def __init__(self, reader: JerseyReader, cache_path: Path) -> None:
        self.reader = reader
        self.model = reader.model
        self.cache_path = cache_path
        self._entries: dict[str, list[dict[str, object]]] = {}
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != JERSEY_CACHE_SCHEMA:
                raise ValueError("unsupported jersey-number cache schema")
            if payload.get("model") != self.model:
                raise ValueError("jersey-number cache model mismatch")
            self._entries = dict(payload.get("entries") or {})

    def read_jersey_number(
        self,
        frames: Sequence[np.ndarray],
        scope: str = "",
    ) -> list[JerseyNumberCandidateResponse]:
        key = _crop_cache_key(self.model, scope, frames)
        cached = self._entries.get(key)
        if cached is not None:
            return [JerseyNumberCandidateResponse.model_validate(item) for item in cached]
        result = self.reader.read_jersey_number(frames, scope=scope)
        self._entries[key] = [item.model_dump(mode="json") for item in result]
        self._write()
        return result

    def _write(self) -> None:
        payload = {
            "schema_version": JERSEY_CACHE_SCHEMA,
            "model": self.model,
            "entries": self._entries,
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.cache_path)


def _crop_cache_key(model: str, scope: str, frames: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256(f"{model}|{scope}|jersey-v1".encode("utf-8"))
    for frame in frames:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            raise ValueError("unable to encode jersey crop for cache fingerprint")
        digest.update(encoded.tobytes())
    return digest.hexdigest()
