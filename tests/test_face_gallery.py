from __future__ import annotations

import numpy as np
import pytest

from app.analysis.face_gallery import (
    assess_face_gallery_coverage,
    attach_face_jersey_annotations,
    match_face_gallery,
    parse_face_gallery,
    seal_face_gallery_payload,
    seal_face_jersey_annotation_payload,
)


def _gallery():
    return parse_face_gallery(
        seal_face_gallery_payload(
            {
                "model_id": "sface-test",
                "source_manifest_sha256": "manifest",
                "benchmark_disjoint": True,
                "entries": [
                    {
                        "person_id": "black-player-01",
                        "team_id": "raw-dark",
                        "embedding": [1.0, 0.0],
                        "sample_count": 3,
                        "quality": 0.91,
                    },
                    {
                        "person_id": "black-player-02",
                        "team_id": "raw-dark",
                        "embedding": [0.0, 1.0],
                        "sample_count": 2,
                        "quality": 0.88,
                    },
                ],
            }
        )
    )


def test_face_gallery_matches_with_threshold_and_margin() -> None:
    match = match_face_gallery(
        _gallery(),
        np.array([0.99, 0.05], dtype=np.float32),
        model_id="sface-test",
        team_id="raw-dark",
    )

    assert match is not None
    assert match.person_id == "black-player-01"
    assert match.confidence > 0.99


def test_face_gallery_match_preserves_optional_enrollment_jersey_number() -> None:
    payload = seal_face_gallery_payload(
        {
            "schema_version": "agu.face-gallery.v1",
            "benchmark_disjoint": True,
            "model_id": "test-model",
            "source_manifest_sha256": "source-hash",
            "entries": [
                {
                    "person_id": "player-1",
                    "team_id": "dark",
                    "jersey_number": "07",
                    "embedding": [1.0, 0.0],
                    "sample_count": 2,
                    "quality": 0.9,
                }
            ],
        }
    )
    gallery = parse_face_gallery(payload)
    match = match_face_gallery(gallery, [1.0, 0.0], model_id="test-model")

    assert match is not None
    assert match.jersey_number == "07"


def test_face_gallery_rejects_invalid_enrollment_jersey_number() -> None:
    payload = seal_face_gallery_payload(
        {
            "schema_version": "agu.face-gallery.v1",
            "benchmark_disjoint": True,
            "model_id": "test-model",
            "source_manifest_sha256": "source-hash",
            "entries": [
                {
                    "person_id": "player-1",
                    "team_id": "dark",
                    "jersey_number": "seven",
                    "embedding": [1.0, 0.0],
                    "sample_count": 2,
                    "quality": 0.9,
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="invalid jersey number"):
        parse_face_gallery(payload)


def test_hash_sealed_jersey_annotations_attach_only_to_matching_face_entries() -> None:
    gallery_payload = seal_face_gallery_payload(
        {
            "benchmark_disjoint": True,
            "model_id": "test-model",
            "source_manifest_sha256": "source-hash",
            "entries": [
                {
                    "person_id": "player-1",
                    "team_id": "dark",
                    "embedding": [1.0, 0.0],
                    "sample_count": 2,
                    "quality": 0.9,
                }
            ],
        }
    )
    annotations = seal_face_jersey_annotation_payload(
        {
            "benchmark_disjoint": True,
            "role": "identity_enrollment",
            "annotation_producer": "codex",
            "annotations": [{"person_id": "player-1", "team_id": "dark", "jersey_number": "7"}],
        }
    )

    attached = attach_face_jersey_annotations(gallery_payload, annotations)
    gallery = parse_face_gallery(attached)

    assert gallery.entries[0].jersey_number == "7"
    assert attached["jersey_enrollment"]["annotation_artifact_sha256"] == annotations["artifact_sha256"]


def test_jersey_annotations_reject_team_mismatch() -> None:
    gallery = _gallery()
    gallery_payload = seal_face_gallery_payload(
        {
            "benchmark_disjoint": True,
            "model_id": gallery.model_id,
            "source_manifest_sha256": gallery.source_manifest_sha256,
            "entries": [
                {
                    "person_id": entry.person_id,
                    "team_id": entry.team_id,
                    "embedding": list(entry.embedding),
                    "sample_count": entry.sample_count,
                    "quality": entry.quality,
                }
                for entry in gallery.entries
            ],
        }
    )
    annotations = seal_face_jersey_annotation_payload(
        {
            "benchmark_disjoint": True,
            "role": "identity_enrollment",
            "annotations": [
                {"person_id": "black-player-01", "team_id": "wrong", "jersey_number": "7"}
            ],
        }
    )

    with pytest.raises(ValueError, match="team mismatch"):
        attach_face_jersey_annotations(gallery_payload, annotations)


def test_face_gallery_matches_supported_view_prototype_instead_of_weak_average() -> None:
    payload = seal_face_gallery_payload(
        {
            "model_id": "sface-test",
            "source_manifest_sha256": "manifest",
            "benchmark_disjoint": True,
            "entries": [
                {
                    "person_id": "player-a",
                    "embedding": [0.70710677, 0.70710677],
                    "sample_count": 4,
                    "quality": 0.90,
                    "prototypes": [[1.0, 0.0], [0.0, 1.0]],
                    "prototype_qualities": [0.92, 0.88],
                    "prototype_sample_counts": [2, 2],
                },
                {
                    "person_id": "player-b",
                    "embedding": [0.6, 0.8],
                    "sample_count": 2,
                    "quality": 0.85,
                },
            ],
        }
    )

    match = match_face_gallery(
        parse_face_gallery(payload),
        [1.0, 0.0],
        model_id="sface-test",
        similarity_threshold=0.90,
        minimum_margin=0.10,
    )

    assert match is not None
    assert match.person_id == "player-a"


def test_face_gallery_matching_excludes_low_quality_enrollment_entries() -> None:
    assert (
        match_face_gallery(
            _gallery(),
            [0.0, 1.0],
            model_id="sface-test",
            minimum_entry_quality=0.90,
        )
        is None
    )


def test_face_gallery_rejects_ambiguous_or_wrong_model() -> None:
    assert (
        match_face_gallery(
            _gallery(),
            np.array([1.0, 1.0], dtype=np.float32),
            model_id="sface-test",
            minimum_margin=0.10,
        )
        is None
    )
    with pytest.raises(ValueError, match="models do not match"):
        match_face_gallery(_gallery(), [1.0, 0.0], model_id="other")


def test_face_gallery_detects_tampering() -> None:
    payload = seal_face_gallery_payload(
        {
            "model_id": "sface-test",
            "source_manifest_sha256": "manifest",
            "benchmark_disjoint": True,
            "entries": [
                {
                    "person_id": "p1",
                    "embedding": [1.0, 0.0],
                    "sample_count": 2,
                    "quality": 0.9,
                }
            ],
        }
    )
    payload["entries"][0]["person_id"] = "changed"
    with pytest.raises(ValueError, match="hash mismatch"):
        parse_face_gallery(payload)


def test_face_gallery_requires_benchmark_disjoint_provenance() -> None:
    payload = seal_face_gallery_payload(
        {
            "model_id": "sface-test",
            "source_manifest_sha256": "manifest",
            "benchmark_disjoint": False,
            "entries": [
                {
                    "person_id": "p1",
                    "embedding": [1.0, 0.0],
                    "sample_count": 2,
                    "quality": 0.9,
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="benchmark-disjoint"):
        parse_face_gallery(payload)


def test_face_gallery_coverage_fails_closed_on_missing_or_low_quality_people() -> None:
    gallery = _gallery()

    assessment = assess_face_gallery_coverage(
        gallery,
        ["black-player-01", "black-player-02", "black-player-03"],
        minimum_coverage=0.95,
        minimum_entry_quality=0.90,
    )

    assert assessment.ready is False
    assert assessment.coverage == pytest.approx(1 / 3)
    assert assessment.covered_person_ids == ("black-player-01",)
    assert assessment.missing_person_ids == ("black-player-02", "black-player-03")
    assert assessment.low_quality_person_ids == ("black-player-02",)


def test_face_gallery_coverage_accepts_complete_unique_roster() -> None:
    assessment = assess_face_gallery_coverage(
        _gallery(),
        ["black-player-01", "black-player-02"],
    )

    assert assessment.ready is True
    assert assessment.coverage == 1.0
    with pytest.raises(ValueError, match="must be unique"):
        assess_face_gallery_coverage(_gallery(), ["black-player-01", "black-player-01"])
