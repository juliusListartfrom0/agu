from __future__ import annotations

import pytest

from app.analysis.basketball_detector_dataset import (
    assign_e_bard_game_splits,
    filter_e_bard_ball_labels,
    parse_e_bard_game_id,
    seal_basketball_detector_dataset_manifest,
    verify_basketball_detector_dataset_manifest,
)


def test_parse_e_bard_game_id_requires_canonical_filename() -> None:
    assert (
        parse_e_bard_game_id("bos-vs-was-0022401217_000124.jpg")
        == "0022401217"
    )
    with pytest.raises(ValueError, match="canonical"):
        parse_e_bard_game_id("frame_000124.jpg")
    with pytest.raises(ValueError, match="canonical"):
        parse_e_bard_game_id("bos-vs-was-0022401217_000124.png")


def test_assign_e_bard_game_splits_is_deterministic_and_disjoint() -> None:
    game_ids = [f"{index:010d}" for index in range(60)]

    assignments = assign_e_bard_game_splits(
        game_ids,
        validation_game_count=8,
        test_game_count=8,
        seed="sealed-seed",
    )

    assert assignments == assign_e_bard_game_splits(
        reversed(game_ids),
        validation_game_count=8,
        test_game_count=8,
        seed="sealed-seed",
    )
    assert list(assignments.values()).count("train") == 44
    assert list(assignments.values()).count("val") == 8
    assert list(assignments.values()).count("test") == 8
    assert set(assignments) == set(game_ids)
    with pytest.raises(ValueError, match="unique"):
        assign_e_bard_game_splits(
            ["0000000001", "0000000001"],
            validation_game_count=1,
            test_game_count=0,
            seed="sealed-seed",
        )


def test_filter_e_bard_ball_labels_keeps_ball_and_validates_every_row() -> None:
    labels = filter_e_bard_ball_labels(
        [
            "0 0.5 0.4 0.02 0.03",
            "2 0.5 0.5 0.4 0.8",
            "3 0.1 0.2 0.1 0.2",
        ]
    )

    assert labels == ["0 0.5 0.4 0.02 0.03"]
    with pytest.raises(ValueError, match="class"):
        filter_e_bard_ball_labels(["4 0.5 0.5 0.1 0.1"])
    with pytest.raises(ValueError, match="normalized"):
        filter_e_bard_ball_labels(["0 1.1 0.5 0.1 0.1"])
    with pytest.raises(ValueError, match="positive area"):
        filter_e_bard_ball_labels(["0 0.5 0.5 0.1 0"])
    assert filter_e_bard_ball_labels(["2 0.9 0.1 0.01 0"]) == []
    with pytest.raises(ValueError, match="five"):
        filter_e_bard_ball_labels(["0 0.5 0.5 0.1"])


def test_combined_detector_manifest_is_hash_bound_and_game_disjoint() -> None:
    payload = {
        "e_bard_source_manifest_sha256": "1" * 64,
        "muvy_source_manifest_sha256": "2" * 64,
        "split_seed": "sealed-seed",
        "e_bard_split_game_ids": {
            "train": ["0000000001"],
            "val": ["0000000002"],
            "test": ["0000000003"],
        },
        "examples": [
            {
                "source": "e_bard",
                "group_id": f"000000000{index}",
                "split": split,
                "image_path": f"images/{split}/{index}.jpg",
                "image_sha256": f"{index + 2:x}" * 64,
                "label_path": f"labels/{split}/{index}.txt",
                "label_sha256": f"{index + 5:x}" * 64,
                "ball_box_count": int(split != "test"),
            }
            for index, split in enumerate(("train", "val", "test"), start=1)
        ],
    }

    manifest = seal_basketball_detector_dataset_manifest(payload)

    assert verify_basketball_detector_dataset_manifest(manifest) == manifest
    assert manifest["split_counts"] == {"train": 1, "val": 1, "test": 1}
    leaked = dict(payload)
    leaked["e_bard_split_game_ids"] = {
        "train": ["0000000001"],
        "val": ["0000000001"],
        "test": ["0000000003"],
    }
    with pytest.raises(ValueError, match="disjoint"):
        seal_basketball_detector_dataset_manifest(leaked)
