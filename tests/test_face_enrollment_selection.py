from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from app.analysis.face_gallery import parse_face_gallery, seal_face_gallery_payload
from scripts.rank_face_enrollment_candidates import (
    _load_cluster_images,
    rank_face_enrollment_candidates,
    rank_face_enrollment_reference_embeddings,
)
from scripts.select_face_enrollment_candidates import select_face_enrollment_candidates


def _seal(payload: dict[str, object]) -> dict[str, object]:
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _candidates() -> dict[str, object]:
    return _seal(
        {
            "schema_version": "agu.face-enrollment-candidates.v1",
            "benchmark_disjoint": True,
            "producer": "test",
            "model_id": "sface-test",
            "sources": [{"source_video_id": "enrollment"}],
            "config": {"interval": 2},
            "clusters": [
                {"cluster_id": "face-1", "samples": [{"path": "a.jpg"}, {"path": "b.jpg"}]},
                {"cluster_id": "face-2", "samples": [{"path": "c.jpg"}, {"path": "d.jpg"}]},
            ],
            "manifest_sha256": "",
        }
    )


def test_select_face_candidates_preserves_sealed_provenance() -> None:
    source = _candidates()

    subset = select_face_enrollment_candidates(source, ["face-2"])

    assert [item["cluster_id"] for item in subset["clusters"]] == ["face-2"]
    assert subset["config"]["source_candidate_manifest_sha256"] == source["manifest_sha256"]
    claimed = subset["manifest_sha256"]
    assert claimed == hashlib.sha256(
        json.dumps({**subset, "manifest_sha256": ""}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_select_face_candidates_rejects_unknown_duplicate_or_tampered_input() -> None:
    source = _candidates()
    with pytest.raises(ValueError, match="unknown"):
        select_face_enrollment_candidates(source, ["missing"])
    with pytest.raises(ValueError, match="must be unique"):
        select_face_enrollment_candidates(source, ["face-1", "face-1"])

    source["producer"] = "tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        select_face_enrollment_candidates(source, ["face-1"])


def _gallery():
    return parse_face_gallery(
        seal_face_gallery_payload(
            {
                "model_id": "sface-test",
                "source_manifest_sha256": "portraits",
                "benchmark_disjoint": True,
                "entries": [
                    {
                        "person_id": "p1",
                        "team_id": "CHI",
                        "embedding": [1.0, 0.0],
                        "sample_count": 2,
                        "quality": 0.9,
                        "prototypes": [[1.0, 0.0]],
                        "prototype_qualities": [0.9],
                        "prototype_sample_counts": [2],
                    },
                    {
                        "person_id": "p2",
                        "team_id": "UTA",
                        "embedding": [0.0, 1.0],
                        "sample_count": 2,
                        "quality": 0.8,
                        "prototypes": [[0.0, 1.0]],
                        "prototype_qualities": [0.8],
                        "prototype_sample_counts": [2],
                    },
                ],
            }
        )
    )


def test_rank_face_candidates_is_review_only_and_hash_bound() -> None:
    source = _candidates()

    ranked = rank_face_enrollment_candidates(
        source,
        _gallery(),
        {
            "face-1": np.asarray([0.99, 0.01], dtype=np.float32),
            "face-2": np.asarray([0.20, 0.98], dtype=np.float32),
        },
        top_k=1,
    )

    assert ranked["runtime_consumable"] is False
    assert ranked["identity_verified"] is False
    assert ranked["codex_runtime_answer_used"] is False
    assert ranked["source_candidate_manifest_sha256"] == source["manifest_sha256"]
    assert [row["person_id"] for row in ranked["rankings"]] == ["p1", "p2"]
    assert ranked["rankings"][0]["candidates"][0]["cluster_id"] == "face-1"
    assert ranked["rankings"][1]["candidates"][0]["cluster_id"] == "face-2"
    assert ranked["rankings"][0]["candidates"][0]["person_margin"] > 0.9
    claimed = ranked["manifest_sha256"]
    assert claimed == hashlib.sha256(
        json.dumps(
            {**ranked, "manifest_sha256": ""},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def test_rank_face_candidates_records_missing_embeddings_and_rejects_unsafe_inputs() -> None:
    source = _candidates()

    ranked = rank_face_enrollment_candidates(
        source,
        _gallery(),
        {"face-1": np.asarray([1.0, 0.0], dtype=np.float32)},
        top_k=2,
    )

    assert ranked["unembedded_cluster_ids"] == ["face-2"]
    assert len(ranked["rankings"][0]["candidates"]) == 1

    with pytest.raises(ValueError, match="unknown cluster"):
        rank_face_enrollment_candidates(
            source,
            _gallery(),
            {"face-3": np.asarray([1.0, 0.0], dtype=np.float32)},
        )
    with pytest.raises(ValueError, match="top_k"):
        rank_face_enrollment_candidates(source, _gallery(), {}, top_k=0)

    source["model_id"] = "other-model"
    source["manifest_sha256"] = ""
    source = _seal(source)
    with pytest.raises(ValueError, match="models do not match"):
        rank_face_enrollment_candidates(source, _gallery(), {})


def test_rank_face_candidates_verifies_crop_bytes_before_embedding(
    tmp_path, monkeypatch
) -> None:
    crop = tmp_path / "crop.jpg"
    crop.write_bytes(b"sealed-crop")
    monkeypatch.setattr(
        "scripts.rank_face_enrollment_candidates.cv2.imread",
        lambda _path: np.zeros((16, 16, 3), dtype=np.uint8),
    )
    cluster = {
        "cluster_id": "face-1",
        "samples": [
            {
                "path": "crop.jpg",
                "sha256": hashlib.sha256(b"sealed-crop").hexdigest(),
            }
        ],
    }

    images = _load_cluster_images(cluster, base_dir=tmp_path)

    assert len(images) == 1
    cluster["samples"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sample hash mismatch"):
        _load_cluster_images(cluster, base_dir=tmp_path)


def test_single_face_references_rank_candidates_without_becoming_a_gallery() -> None:
    source = _candidates()
    references = {
        "schema_version": "agu.single-face-reference-list.v1",
        "benchmark_disjoint": True,
        "runtime_consumable": False,
        "model_id": "sface-test",
        "references": [
            {"person_id": "p1", "team_id": "CHI", "path": "portrait.jpg", "sha256": "abc"}
        ],
        "manifest_sha256": "",
    }
    references["manifest_sha256"] = hashlib.sha256(
        json.dumps(references, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    ranked = rank_face_enrollment_reference_embeddings(
        source,
        references,
        {"p1": np.asarray([1.0, 0.0], dtype=np.float32)},
        {
            "face-1": np.asarray([0.99, 0.01], dtype=np.float32),
            "face-2": np.asarray([0.10, 0.90], dtype=np.float32),
        },
        top_k=1,
    )

    assert ranked["schema_version"] == "agu.face-enrollment-single-reference-ranking.v1"
    assert ranked["runtime_consumable"] is False
    assert ranked["identity_verified"] is False
    assert ranked["reference_mode"] == "single_verified_portrait_ranking_only"
    assert ranked["rankings"][0]["candidates"][0]["cluster_id"] == "face-1"


def test_single_face_reference_ranking_rejects_tampering() -> None:
    source = _candidates()
    references = {
        "schema_version": "agu.single-face-reference-list.v1",
        "benchmark_disjoint": True,
        "runtime_consumable": False,
        "model_id": "sface-test",
        "references": [{"person_id": "p1", "team_id": "CHI"}],
        "manifest_sha256": "bad",
    }

    with pytest.raises(ValueError, match="reference manifest hash mismatch"):
        rank_face_enrollment_reference_embeddings(source, references, {}, {})
