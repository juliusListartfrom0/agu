from __future__ import annotations

import hashlib
import json

import pytest

from scripts.approve_face_enrollment_candidates import approve_candidates


def _candidates() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "agu.face-enrollment-candidates.v1",
        "benchmark_disjoint": True,
        "clusters": [
            {
                "cluster_id": "c1",
                "samples": [{"path": "a.jpg"}, {"path": "b.jpg"}],
            },
            {
                "cluster_id": "c2",
                "samples": [{"path": "c.jpg"}, {"path": "d.jpg"}],
            },
            {
                "cluster_id": "c3",
                "samples": [{"path": "e.jpg"}, {"path": "f.jpg"}],
            },
        ],
        "manifest_sha256": "",
    }
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def test_face_enrollment_approval_merges_clusters_and_is_exhaustive() -> None:
    candidates = _candidates()
    approved = approve_candidates(
        candidates,
        {
            "schema_version": "agu.face-enrollment-review.v1",
            "candidate_manifest_sha256": candidates["manifest_sha256"],
            "reviewer": "codex-assisted-face-only",
            "persons": [{"person_id": "p1", "cluster_ids": ["c1", "c2"]}],
            "rejected_cluster_ids": ["c3"],
        },
    )

    assert approved["benchmark_disjoint"] is True
    assert approved["persons"][0]["source_cluster_ids"] == ["c1", "c2"]
    assert len(approved["persons"][0]["samples"]) == 4


def test_face_enrollment_approval_rejects_unreviewed_cluster() -> None:
    candidates = _candidates()
    with pytest.raises(ValueError, match="not exhaustive"):
        approve_candidates(
            candidates,
            {
                "schema_version": "agu.face-enrollment-review.v1",
                "candidate_manifest_sha256": candidates["manifest_sha256"],
                "reviewer": "codex-assisted-face-only",
                "persons": [{"person_id": "p1", "cluster_ids": ["c1"]}],
                "rejected_cluster_ids": ["c2"],
            },
        )


def test_face_enrollment_approval_rejects_duplicate_cluster_claim() -> None:
    candidates = _candidates()
    with pytest.raises(ValueError, match="more than once"):
        approve_candidates(
            candidates,
            {
                "schema_version": "agu.face-enrollment-review.v1",
                "candidate_manifest_sha256": candidates["manifest_sha256"],
                "reviewer": "codex-assisted-face-only",
                "persons": [
                    {"person_id": "p1", "cluster_ids": ["c1"]},
                    {"person_id": "p2", "cluster_ids": ["c1"]},
                ],
                "rejected_cluster_ids": ["c2", "c3"],
            },
        )
