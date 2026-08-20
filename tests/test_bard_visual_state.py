from __future__ import annotations

import copy

import pytest

from app.analysis.bard_visual_state import (
    bard_clip_frame_indexes,
    build_bard_visual_state_subset_manifest,
    seal_bard_visual_state_embedding_artifact,
    verify_bard_visual_state_embedding_artifact,
    verify_bard_visual_state_subset_manifest,
)


def _rows() -> list[dict]:
    return [
        {
            "path": "validation/2024/multi/bos-vs-ind-0022300507_364.mp4",
            "actions": ["Free Throw"],
        },
        {
            "path": "validation/2024/multi/bos-vs-ind-0022300507_351.mp4",
            "actions": ["2PT Shot", "Rebound"],
        },
        {
            "path": "validation/2024/multi/bos-vs-ind-0022300507_370.mp4",
            "actions": ["3PT Shot"],
        },
    ]


def _blobs() -> dict[str, str]:
    return {
        row["path"]: f"{index + 1:040x}"
        for index, row in enumerate(_rows())
    }


def test_bard_subset_pairs_each_free_throw_with_nearest_same_game_shot() -> None:
    manifest = build_bard_visual_state_subset_manifest(
        rows=_rows(),
        git_blob_sha1_by_path=_blobs(),
        source_revision="a" * 40,
        benchmark_sha256s=["b" * 64],
    )

    assert manifest["schema_version"] == "agu.bard-visual-state-subset.v1"
    assert manifest["license"] == "CC-BY-4.0"
    assert manifest["runtime_consumable"] is False
    assert manifest["summary"] == {
        "candidate_free_throw": 1,
        "excluded_unpaired_free_throw": 0,
        "pairs": 1,
        "free_throw": 1,
        "field_goal": 1,
        "examples": 2,
    }
    assert manifest["pairs"][0]["free_throw_path"].endswith("_364.mp4")
    assert manifest["pairs"][0]["field_goal_path"].endswith("_370.mp4")
    verify_bard_visual_state_subset_manifest(manifest)


def test_bard_subset_excludes_ambiguous_rows_and_rejects_path_traversal() -> None:
    rows = _rows()
    rows.append(
        {
            "path": "validation/2024/multi/bos-vs-ind-0022300507_380.mp4",
            "actions": ["Free Throw", "2PT Shot"],
        }
    )
    blobs = _blobs()
    blobs[rows[-1]["path"]] = "4" * 40

    manifest = build_bard_visual_state_subset_manifest(
        rows=rows,
        git_blob_sha1_by_path=blobs,
        source_revision="a" * 40,
        benchmark_sha256s=["b" * 64],
    )

    assert manifest["summary"]["examples"] == 2

    rows[0]["path"] = "../outside.mp4"
    with pytest.raises(ValueError, match="path"):
        build_bard_visual_state_subset_manifest(
            rows=rows,
            git_blob_sha1_by_path=blobs,
            source_revision="a" * 40,
            benchmark_sha256s=["b" * 64],
        )


def test_bard_subset_records_and_excludes_free_throws_without_same_game_shots() -> None:
    rows = _rows()
    unmatched = {
        "path": "validation/2025/multi/atl-vs-orl-0022401149_16.mp4",
        "actions": ["Free Throw"],
    }
    rows.append(unmatched)
    blobs = _blobs()
    blobs[unmatched["path"]] = "4" * 40

    manifest = build_bard_visual_state_subset_manifest(
        rows=rows,
        git_blob_sha1_by_path=blobs,
        source_revision="a" * 40,
        benchmark_sha256s=["b" * 64],
    )

    assert manifest["summary"]["candidate_free_throw"] == 2
    assert manifest["summary"]["excluded_unpaired_free_throw"] == 1
    assert manifest["summary"]["pairs"] == 1


def test_bard_subset_hash_detects_tampering() -> None:
    manifest = build_bard_visual_state_subset_manifest(
        rows=_rows(),
        git_blob_sha1_by_path=_blobs(),
        source_revision="a" * 40,
        benchmark_sha256s=["b" * 64],
    )
    tampered = copy.deepcopy(manifest)
    tampered["pairs"][0]["event_distance"] += 1

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_bard_visual_state_subset_manifest(tampered)


def test_bard_embedding_artifact_is_training_only_and_hash_bound() -> None:
    artifact = seal_bard_visual_state_embedding_artifact(
        {
            "purpose": "test",
            "acquisition_artifact_sha256": "1" * 64,
            "backbone": "facebook/dinov2-small",
            "backbone_sha256": "2" * 64,
            "sealed_blind_video_sha256s": ["3" * 64],
            "examples": [
                {
                    "source_video_sha256": "4" * 64,
                    "pair_id": "pair-1",
                    "state": "free_throw",
                    "frame_indexes": [1, 2, 3, 4, 5],
                    "embeddings": [[float(index)] * 384 for index in range(5)],
                }
            ],
        }
    )

    assert artifact["runtime_consumable"] is False
    assert artifact["codex_runtime_answer_used"] is False
    verify_bard_visual_state_embedding_artifact(artifact)

    tampered = copy.deepcopy(artifact)
    tampered["examples"][0]["embeddings"][0][0] = -1.0
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_bard_visual_state_embedding_artifact(tampered)


def test_bard_clip_frame_indexes_are_fixed_unique_quantiles() -> None:
    assert bard_clip_frame_indexes(frame_count=101) == (5, 27, 50, 72, 95)

    with pytest.raises(ValueError, match="at least five"):
        bard_clip_frame_indexes(frame_count=4)
