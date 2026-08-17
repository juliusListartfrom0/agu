from __future__ import annotations

import pytest

from scripts.merge_annotated_face_lists import merge_annotated_face_lists


def _manifest(source_hash: str, person_id: str) -> dict[str, object]:
    return {
        "schema_version": "agu.annotated-face-list.v1",
        "benchmark_disjoint": True,
        "source_candidate_manifest_sha256": source_hash,
        "persons": [
            {
                "person_id": person_id,
                "team_id": None,
                "samples": [{"path": f"{person_id}-1.jpg"}, {"path": f"{person_id}-2.jpg"}],
            }
        ],
    }


def test_merge_face_lists_combines_cross_source_identity() -> None:
    merged = merge_annotated_face_lists(
        [_manifest("hash-a", "a"), _manifest("hash-b", "b")],
        {
            "schema_version": "agu.annotated-face-merge.v1",
            "source_candidate_manifest_sha256s": ["hash-a", "hash-b"],
            "reviewer": "codex-face-only",
            "persons": [
                {
                    "person_id": "roster-01",
                    "source_person_ids": ["source_001:a", "source_002:b"],
                }
            ],
        },
    )

    assert merged["benchmark_disjoint"] is True
    assert len(merged["persons"]) == 1
    assert len(merged["persons"][0]["samples"]) == 4
    assert {sample["prototype_source_id"] for sample in merged["persons"][0]["samples"]} == {
        "source_001:a",
        "source_002:b",
    }


def test_merge_face_lists_requires_exhaustive_hash_bound_decisions() -> None:
    manifests = [_manifest("hash-a", "a"), _manifest("hash-b", "b")]
    decisions = {
        "schema_version": "agu.annotated-face-merge.v1",
        "source_candidate_manifest_sha256s": ["hash-a", "hash-b"],
        "reviewer": "codex-face-only",
        "persons": [{"person_id": "roster-01", "source_person_ids": ["source_001:a"]}],
    }

    with pytest.raises(ValueError, match="not exhaustive"):
        merge_annotated_face_lists(manifests, decisions)

    decisions["source_candidate_manifest_sha256s"] = ["wrong", "hash-b"]
    with pytest.raises(ValueError, match="do not match"):
        merge_annotated_face_lists(manifests, decisions)
