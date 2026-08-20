from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from app.analysis.open_tracking_datasets import build_open_tracking_catalog
from app.analysis.reid_training_data import build_reid_training_crops


def test_build_reid_training_crops_is_hash_bound_and_training_only(tmp_path: Path) -> None:
    root = tmp_path / "mot"
    sequence = root / "sequence-a"
    (sequence / "gt").mkdir(parents=True)
    (sequence / "gt" / "gt.txt").write_text(
        "1,1,5,5,30,60,1,-1,-1,-1\n"
        "2,1,6,5,30,60,1,-1,-1,-1\n"
        "3,1,7,5,30,60,1,-1,-1,-1\n",
        encoding="utf-8",
    )
    video = sequence / "img1.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (80, 80))
    assert writer.isOpened()
    for value in (40, 80, 120):
        writer.write(np.full((80, 80, 3), value, dtype=np.uint8))
    writer.release()
    catalog = build_open_tracking_catalog(root, dataset_id="teamtrack", source_revision="fixture-v1")

    payload = build_reid_training_crops(
        catalog_payload=catalog,
        dataset_root=root,
        output_root=tmp_path / "crops",
        maximum_crops_per_track=2,
    )

    assert payload["training_only"] is True
    assert payload["runtime_consumable"] is False
    assert payload["acceptance_media_allowed"] is False
    assert payload["benchmark_disjoint"] is True
    assert payload["identity_count"] == 1
    assert payload["sample_count"] == 2
    for sample in payload["samples"]:
        path = tmp_path / "crops" / sample["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sample["sha256"]


def test_dataset_track_scope_merges_declared_stable_ids_across_sequences(tmp_path: Path) -> None:
    root = tmp_path / "mot"
    for sequence_name, value in (("clip-a", 40), ("clip-b", 120)):
        sequence = root / sequence_name
        (sequence / "gt").mkdir(parents=True)
        (sequence / "gt" / "gt.txt").write_text(
            "1,7,5,5,30,60,1,-1,-1,-1\n",
            encoding="utf-8",
        )
        video = sequence / "img1.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (80, 80))
        assert writer.isOpened()
        writer.write(np.full((80, 80, 3), value, dtype=np.uint8))
        writer.release()
    catalog = build_open_tracking_catalog(root, dataset_id="teamtrack", source_revision="fixture-v2")

    payload = build_reid_training_crops(
        catalog_payload=catalog,
        dataset_root=root,
        output_root=tmp_path / "crops",
        identity_scope="dataset_track",
    )

    assert payload["identity_scope"] == "dataset_track"
    assert payload["identity_count"] == 1
    assert payload["identities"][0]["identity_id"] == "track:7"
    assert payload["identities"][0]["sequences"] == ["clip-a", "clip-b"]
    assert payload["identities"][0]["sample_count"] == 2
