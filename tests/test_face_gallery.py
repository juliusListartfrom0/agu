from __future__ import annotations

import numpy as np
import pytest

from app.analysis.face_gallery import match_face_gallery, parse_face_gallery, seal_face_gallery_payload


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
