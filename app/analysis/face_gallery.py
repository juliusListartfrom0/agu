"""Hash-sealed annotated face gallery matching for canonical player identity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "agu.face-gallery.v1"


@dataclass(frozen=True)
class FaceGalleryEntry:
    person_id: str
    team_id: str | None
    embedding: tuple[float, ...]
    sample_count: int
    quality: float
    sample_sha256: tuple[str, ...] = ()


@dataclass(frozen=True)
class FaceGallery:
    model_id: str
    entries: tuple[FaceGalleryEntry, ...]
    source_manifest_sha256: str
    gallery_sha256: str


@dataclass(frozen=True)
class FaceGalleryMatch:
    person_id: str
    team_id: str | None
    confidence: float
    runner_up_confidence: float
    gallery_sha256: str


def load_face_gallery(path: str | Path) -> FaceGallery:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return parse_face_gallery(payload)


def parse_face_gallery(payload: Mapping[str, Any]) -> FaceGallery:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported face gallery schema")
    if payload.get("benchmark_disjoint") is not True:
        raise ValueError("face gallery must be explicitly benchmark-disjoint")
    claimed_sha256 = str(payload.get("gallery_sha256") or "")
    expected_sha256 = _canonical_sha256({**dict(payload), "gallery_sha256": ""})
    if not claimed_sha256 or claimed_sha256 != expected_sha256:
        raise ValueError("face gallery hash mismatch")
    model_id = str(payload.get("model_id") or "").strip()
    if not model_id:
        raise ValueError("face gallery model_id is required")
    source_manifest_sha256 = str(payload.get("source_manifest_sha256") or "").strip()
    if not source_manifest_sha256:
        raise ValueError("face gallery source manifest hash is required")

    entries: list[FaceGalleryEntry] = []
    dimensions: set[int] = set()
    person_ids: set[str] = set()
    for raw in payload.get("entries") or []:
        person_id = str(raw.get("person_id") or "").strip()
        if not person_id or person_id in person_ids:
            raise ValueError("face gallery person_id values must be non-empty and unique")
        vector = np.asarray(raw.get("embedding") or [], dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError(f"invalid face embedding for {person_id}")
        norm = float(np.linalg.norm(vector))
        if norm < 0.99 or norm > 1.01:
            raise ValueError(f"face embedding for {person_id} must be L2 normalized")
        sample_count = int(raw.get("sample_count") or 0)
        quality = float(raw.get("quality") or 0.0)
        if sample_count < 2 or not 0.0 <= quality <= 1.0:
            raise ValueError(f"invalid face gallery evidence for {person_id}")
        entries.append(
            FaceGalleryEntry(
                person_id=person_id,
                team_id=str(raw["team_id"]).strip() if raw.get("team_id") else None,
                embedding=tuple(float(value) for value in vector),
                sample_count=sample_count,
                quality=quality,
                sample_sha256=tuple(str(value) for value in raw.get("sample_sha256") or []),
            )
        )
        dimensions.add(int(vector.size))
        person_ids.add(person_id)
    if not entries or len(dimensions) != 1:
        raise ValueError("face gallery must contain compatible entries")
    return FaceGallery(
        model_id=model_id,
        entries=tuple(entries),
        source_manifest_sha256=source_manifest_sha256,
        gallery_sha256=claimed_sha256,
    )


def match_face_gallery(
    gallery: FaceGallery,
    embedding: Sequence[float] | np.ndarray,
    *,
    model_id: str,
    team_id: str | None = None,
    similarity_threshold: float = 0.45,
    minimum_margin: float = 0.08,
) -> FaceGalleryMatch | None:
    if model_id != gallery.model_id:
        raise ValueError("face gallery and query embedding models do not match")
    if not 0.0 <= similarity_threshold <= 1.0 or not 0.0 <= minimum_margin <= 1.0:
        raise ValueError("face gallery thresholds must be in [0,1]")
    query = np.asarray(embedding, dtype=np.float32)
    if query.ndim != 1 or query.size == 0:
        return None
    norm = float(np.linalg.norm(query))
    if norm <= 0.0:
        return None
    query = query / norm
    compatible = [entry for entry in gallery.entries if not team_id or not entry.team_id or entry.team_id == team_id]
    scores = sorted(
        ((float(query @ np.asarray(entry.embedding, dtype=np.float32)), entry) for entry in compatible),
        key=lambda item: (item[0], item[1].person_id),
        reverse=True,
    )
    if not scores:
        return None
    best_score, best_entry = scores[0]
    runner_up = scores[1][0] if len(scores) > 1 else -1.0
    if best_score < similarity_threshold or best_score - runner_up < minimum_margin:
        return None
    return FaceGalleryMatch(
        person_id=best_entry.person_id,
        team_id=best_entry.team_id,
        confidence=max(0.0, min(1.0, best_score)),
        runner_up_confidence=max(-1.0, min(1.0, runner_up)),
        gallery_sha256=gallery.gallery_sha256,
    )


def seal_face_gallery_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = {**dict(payload), "schema_version": SCHEMA_VERSION, "gallery_sha256": ""}
    sealed["gallery_sha256"] = _canonical_sha256(sealed)
    return sealed


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
