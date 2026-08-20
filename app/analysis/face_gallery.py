"""Hash-sealed annotated face gallery matching for canonical player identity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "agu.face-gallery.v1"
JERSEY_ANNOTATION_SCHEMA_VERSION = "agu.face-jersey-annotation.v1"


@dataclass(frozen=True)
class FaceGalleryEntry:
    person_id: str
    team_id: str | None
    jersey_number: str | None
    embedding: tuple[float, ...]
    sample_count: int
    quality: float
    sample_sha256: tuple[str, ...] = ()
    prototypes: tuple[tuple[float, ...], ...] = ()
    prototype_qualities: tuple[float, ...] = ()
    prototype_sample_counts: tuple[int, ...] = ()


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
    jersey_number: str | None
    confidence: float
    runner_up_confidence: float
    gallery_sha256: str


@dataclass(frozen=True)
class FaceGalleryCoverage:
    expected_person_count: int
    enrolled_person_count: int
    covered_person_count: int
    coverage: float
    minimum_coverage: float
    ready: bool
    covered_person_ids: tuple[str, ...]
    missing_person_ids: tuple[str, ...]
    extra_person_ids: tuple[str, ...]
    low_quality_person_ids: tuple[str, ...]
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
        jersey_number = _normalize_jersey_number(raw.get("jersey_number"), person_id=person_id)
        if sample_count < 2 or not 0.0 <= quality <= 1.0:
            raise ValueError(f"invalid face gallery evidence for {person_id}")
        raw_prototypes = raw.get("prototypes") or [vector.tolist()]
        prototypes = [np.asarray(item, dtype=np.float32) for item in raw_prototypes]
        prototype_qualities = tuple(
            float(value) for value in (raw.get("prototype_qualities") or [quality] * len(prototypes))
        )
        prototype_sample_counts = tuple(
            int(value) for value in (raw.get("prototype_sample_counts") or [sample_count] * len(prototypes))
        )
        if not prototypes or not (
            len(prototypes) == len(prototype_qualities) == len(prototype_sample_counts)
        ):
            raise ValueError(f"invalid face prototypes for {person_id}")
        for prototype, prototype_quality, prototype_support in zip(
            prototypes, prototype_qualities, prototype_sample_counts
        ):
            prototype_norm = float(np.linalg.norm(prototype))
            if (
                prototype.ndim != 1
                or prototype.size != vector.size
                or not np.all(np.isfinite(prototype))
                or prototype_norm < 0.99
                or prototype_norm > 1.01
                or not 0.0 <= prototype_quality <= 1.0
                or prototype_support < 2
            ):
                raise ValueError(f"invalid face prototype evidence for {person_id}")
        entries.append(
            FaceGalleryEntry(
                person_id=person_id,
                team_id=str(raw["team_id"]).strip() if raw.get("team_id") else None,
                jersey_number=jersey_number,
                embedding=tuple(float(value) for value in vector),
                sample_count=sample_count,
                quality=quality,
                sample_sha256=tuple(str(value) for value in raw.get("sample_sha256") or []),
                prototypes=tuple(tuple(float(value) for value in item) for item in prototypes),
                prototype_qualities=prototype_qualities,
                prototype_sample_counts=prototype_sample_counts,
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
    minimum_entry_quality: float = 0.0,
) -> FaceGalleryMatch | None:
    if model_id != gallery.model_id:
        raise ValueError("face gallery and query embedding models do not match")
    if not (
        0.0 <= similarity_threshold <= 1.0
        and 0.0 <= minimum_margin <= 1.0
        and 0.0 <= minimum_entry_quality <= 1.0
    ):
        raise ValueError("face gallery thresholds must be in [0,1]")
    query = np.asarray(embedding, dtype=np.float32)
    if query.ndim != 1 or query.size == 0:
        return None
    norm = float(np.linalg.norm(query))
    if norm <= 0.0:
        return None
    query = query / norm
    compatible = [
        entry
        for entry in gallery.entries
        if entry.quality >= minimum_entry_quality
        and (not team_id or not entry.team_id or entry.team_id == team_id)
    ]
    scores = sorted(
        (
            (
                max(float(query @ np.asarray(prototype, dtype=np.float32)) for prototype in entry.prototypes),
                entry,
            )
            for entry in compatible
        ),
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
        jersey_number=best_entry.jersey_number,
        confidence=max(0.0, min(1.0, best_score)),
        runner_up_confidence=max(-1.0, min(1.0, runner_up)),
        gallery_sha256=gallery.gallery_sha256,
    )


def assess_face_gallery_coverage(
    gallery: FaceGallery,
    expected_person_ids: Sequence[str],
    *,
    minimum_coverage: float = 0.95,
    minimum_entry_quality: float = 0.50,
) -> FaceGalleryCoverage:
    """Fail-closed roster coverage gate for benchmark-disjoint face enrollment."""

    if not 0.0 <= minimum_coverage <= 1.0 or not 0.0 <= minimum_entry_quality <= 1.0:
        raise ValueError("face gallery coverage thresholds must be in [0,1]")
    normalized = tuple(str(value).strip() for value in expected_person_ids)
    if not normalized or any(not value for value in normalized):
        raise ValueError("expected face roster must contain non-empty person IDs")
    if len(set(normalized)) != len(normalized):
        raise ValueError("expected face roster person IDs must be unique")

    expected = set(normalized)
    all_enrolled = {entry.person_id for entry in gallery.entries}
    eligible = {entry.person_id for entry in gallery.entries if entry.quality >= minimum_entry_quality}
    low_quality = all_enrolled - eligible
    covered = expected & eligible
    coverage = len(covered) / len(expected)
    return FaceGalleryCoverage(
        expected_person_count=len(expected),
        enrolled_person_count=len(all_enrolled),
        covered_person_count=len(covered),
        coverage=coverage,
        minimum_coverage=minimum_coverage,
        ready=coverage >= minimum_coverage,
        covered_person_ids=tuple(sorted(covered)),
        missing_person_ids=tuple(sorted(expected - eligible)),
        extra_person_ids=tuple(sorted(eligible - expected)),
        low_quality_person_ids=tuple(sorted(low_quality)),
        gallery_sha256=gallery.gallery_sha256,
    )


def seal_face_gallery_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = {**dict(payload), "schema_version": SCHEMA_VERSION, "gallery_sha256": ""}
    sealed["gallery_sha256"] = _canonical_sha256(sealed)
    return sealed


def parse_face_jersey_annotations(
    payload: Mapping[str, Any],
    gallery: FaceGallery,
) -> dict[str, str]:
    """Validate identity-only jersey labels before they can anchor runtime reads."""

    if payload.get("schema_version") != JERSEY_ANNOTATION_SCHEMA_VERSION:
        raise ValueError("unsupported face jersey annotation schema")
    if payload.get("benchmark_disjoint") is not True or payload.get("role") != "identity_enrollment":
        raise ValueError("face jersey annotations must be benchmark-disjoint identity enrollment")
    claimed = str(payload.get("artifact_sha256") or "")
    expected = _canonical_sha256({**dict(payload), "artifact_sha256": ""})
    if not claimed or claimed != expected:
        raise ValueError("face jersey annotation hash mismatch")
    gallery_by_person = {entry.person_id: entry for entry in gallery.entries}
    result: dict[str, str] = {}
    team_numbers: set[tuple[str, str]] = set()
    for raw in payload.get("annotations") or []:
        person_id = str(raw.get("person_id") or "").strip()
        team_id = str(raw.get("team_id") or "").strip()
        if not person_id or person_id in result or person_id not in gallery_by_person:
            raise ValueError("jersey annotations must name unique enrolled people")
        entry = gallery_by_person[person_id]
        if not team_id or entry.team_id != team_id:
            raise ValueError(f"jersey annotation team mismatch for {person_id}")
        number = _normalize_jersey_number(raw.get("jersey_number"), person_id=person_id)
        if number is None or (team_id, number) in team_numbers:
            raise ValueError("jersey annotations must be unique within each team")
        result[person_id] = number
        team_numbers.add((team_id, number))
    if not result:
        raise ValueError("face jersey annotations are empty")
    return result


def seal_face_jersey_annotation_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = {
        **dict(payload),
        "schema_version": JERSEY_ANNOTATION_SCHEMA_VERSION,
        "artifact_sha256": "",
    }
    sealed["artifact_sha256"] = _canonical_sha256(sealed)
    return sealed


def attach_face_jersey_annotations(
    gallery_payload: Mapping[str, Any],
    annotation_payload: Mapping[str, Any],
) -> dict[str, Any]:
    gallery = parse_face_gallery(gallery_payload)
    annotations = parse_face_jersey_annotations(annotation_payload, gallery)
    output = json.loads(json.dumps(gallery_payload))
    for entry in output.get("entries") or []:
        entry.pop("jersey_number", None)
        person_id = str(entry.get("person_id") or "")
        if person_id in annotations:
            entry["jersey_number"] = annotations[person_id]
    output["jersey_enrollment"] = {
        "role": "benchmark_disjoint_identity_registration",
        "annotation_boundary": "face_identity_anchor_only_no_event_labels",
        "annotation_artifact_sha256": annotation_payload["artifact_sha256"],
        "enrolled_person_count": len(annotations),
    }
    return seal_face_gallery_payload(output)


def _normalize_jersey_number(value: object, *, person_id: str) -> str | None:
    if value is None:
        return None
    number = str(value).strip()
    if not number or not number.isdigit() or len(number) > 2:
        raise ValueError(f"invalid jersey number for {person_id}")
    return number


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
