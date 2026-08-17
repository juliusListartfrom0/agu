from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from PIL import Image

from app.analysis.e_bard_object_classification import (
    OBJECT_CLASSES,
    summarize_object_classification_archive,
)


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _row(image: str, label: str) -> dict[str, object]:
    return {
        "image": f"/{image}",
        "conversations": [
            {"from": "human", "value": "<image> What object?"},
            {"from": "gpt", "value": label},
        ],
    }


def _write_archive(path: Path) -> None:
    manifests = {
        "train_classification_dataset.json": [
            _row("train/game-a_000001_obj1.jpg", "basketball"),
            _row("train/game-a_000002_obj2.jpg", "player"),
        ],
        "valid_classification_dataset.json": [
            _row("valid/game-a_000003_obj3.jpg", "hoop"),
        ],
        "test_classification_dataset.json": [
            _row("test/game-b_000004_obj4.jpg", "referee"),
        ],
    }
    colors = ((220, 100, 20), (20, 80, 180), (180, 180, 20), (30, 150, 60))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for manifest_name, rows in manifests.items():
            archive.writestr(manifest_name, json.dumps(rows))
        for index, image in enumerate(
            (
                "train/game-a_000001_obj1.jpg",
                "train/game-a_000002_obj2.jpg",
                "valid/game-a_000003_obj3.jpg",
                "test/game-b_000004_obj4.jpg",
            )
        ):
            archive.writestr(image, _jpeg_bytes(colors[index]))


def test_summarize_object_classification_archive_seals_labels_and_game_overlap(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "all.zip"
    _write_archive(archive)

    summary = summarize_object_classification_archive(archive)

    assert summary["schema_version"] == "agu.e-bard-object-classification-audit.v1"
    assert summary["total_examples"] == 4
    assert summary["class_counts"] == {
        "basketball": 1,
        "hoop": 1,
        "player": 1,
        "referee": 1,
    }
    assert summary["image_reference_count"] == 4
    assert summary["missing_image_references"] == []
    assert summary["decode_failures"] == []
    assert summary["split_game_counts"] == {"test": 1, "train": 1, "valid": 1}
    assert summary["split_game_overlaps"]["train__valid"] == ["game-a"]
    assert summary["cross_game_benchmark_ready"] is False
    assert summary["offline_object_role_training_ready"] is True
    assert tuple(summary["object_classes"]) == OBJECT_CLASSES
