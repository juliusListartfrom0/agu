"""Offline audit helpers for the licensed E-BARD object-role crop release.

The release contains image crops and conversational labels rather than full
broadcast frames.  It can therefore support an offline object-role/VLM head,
but it must not be treated as a temporally independent ball detector or as a
runtime truth source.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

OBJECT_CLASSIFICATION_SCHEMA = "agu.e-bard-object-classification-audit.v1"
OBJECT_CLASSES = ("basketball", "hoop", "player", "referee")
_SPLIT_MANIFESTS = {
    "train": "train_classification_dataset.json",
    "valid": "valid_classification_dataset.json",
    "test": "test_classification_dataset.json",
}
_IMAGE_SUFFIX = ".jpg"
_GAME_ID_PATTERN = re.compile(r"_(\d{6})_obj\d+(?:_\d+)?\.jpg$", re.IGNORECASE)
_MAX_RECORDED_ERRORS = 50


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_image_path(value: object) -> str:
    image = str(value or "").strip().replace("\\", "/")
    while image.startswith("/"):
        image = image[1:]
    if not image or image.startswith("../") or "/../" in image:
        raise ValueError(f"invalid image path: {value!r}")
    return image


def _game_id(image_path: str) -> str:
    match = _GAME_ID_PATTERN.search(Path(image_path).name)
    if match is None:
        return ""
    return Path(image_path).name[: match.start()]


def _row_label(row: Mapping[str, Any]) -> str:
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise ValueError("classification row lacks two conversations")
    answer = conversations[1]
    if not isinstance(answer, Mapping) or str(answer.get("from")) != "gpt":
        raise ValueError("classification row lacks a gpt answer")
    label = str(answer.get("value") or "").strip().lower()
    if label not in OBJECT_CLASSES:
        raise ValueError(f"unsupported object class: {label!r}")
    return label


def _load_manifest(archive: zipfile.ZipFile, member: str) -> list[Mapping[str, Any]]:
    try:
        payload = json.loads(archive.read(member))
    except KeyError as exc:
        raise ValueError(f"missing manifest: {member}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"manifest is not a list: {member}")
    rows: list[Mapping[str, Any]] = []
    for row in payload:
        if not isinstance(row, Mapping):
            raise ValueError(f"manifest row is not an object: {member}")
        rows.append(row)
    return rows


def _decode_image(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as image:
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        return int(image.width), int(image.height)


def summarize_object_classification_archive(archive_path: Path) -> dict[str, Any]:
    """Validate manifests, image references, labels, and split game overlap.

    The function reads the archive without extracting it, decodes each
    referenced JPEG, and returns a JSON-safe summary.  The summary is an
    offline audit artifact; it intentionally contains no runtime answer
    channel or full-game statistics.
    """

    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    split_rows: dict[str, list[Mapping[str, Any]]] = {}
    split_labels: dict[str, Counter[str]] = {}
    split_images: dict[str, list[str]] = {}
    split_games: dict[str, set[str]] = {}
    invalid_rows: list[dict[str, str]] = []
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        duplicate_members = sorted(
            name for name, count in Counter(names).items() if count > 1
        )
        corrupt_member = archive.testzip()
        for split, manifest in _SPLIT_MANIFESTS.items():
            rows = _load_manifest(archive, manifest)
            split_rows[split] = rows
            labels: Counter[str] = Counter()
            images: list[str] = []
            games: set[str] = set()
            for index, row in enumerate(rows):
                try:
                    image = _normalize_image_path(row.get("image"))
                    label = _row_label(row)
                    game = _game_id(image)
                    if not image.lower().endswith(_IMAGE_SUFFIX):
                        raise ValueError("image is not a JPEG")
                    if not game:
                        raise ValueError("image filename lacks a game/frame/object id")
                except (TypeError, ValueError) as exc:
                    if len(invalid_rows) < _MAX_RECORDED_ERRORS:
                        invalid_rows.append(
                            {"split": split, "row": str(index), "error": str(exc)}
                        )
                    continue
                labels[label] += 1
                images.append(image)
                games.add(game)
            split_labels[split] = labels
            split_images[split] = images
            split_games[split] = games

        all_references = [image for images in split_images.values() for image in images]
        reference_counts = Counter(all_references)
        image_member_names = {
            name for name in names if name.lower().endswith(_IMAGE_SUFFIX)
        }
        missing_references = sorted(
            image for image in reference_counts if image not in image_member_names
        )
        unreferenced_members = sorted(image_member_names - set(reference_counts))
        decode_failures: list[dict[str, str]] = []
        dimensions: Counter[str] = Counter()
        for image in sorted(reference_counts):
            try:
                width, height = _decode_image(archive.read(image))
                dimensions[f"{width}x{height}"] += reference_counts[image]
            except (KeyError, OSError, ValueError) as exc:
                if len(decode_failures) < _MAX_RECORDED_ERRORS:
                    decode_failures.append({"image": image, "error": str(exc)})

        split_overlap: dict[str, list[str]] = {}
        split_names = tuple(_SPLIT_MANIFESTS)
        for left_index, left in enumerate(split_names):
            for right in split_names[left_index + 1 :]:
                split_overlap[f"{left}__{right}"] = sorted(
                    split_games[left] & split_games[right]
                )

        class_counts = Counter(
            label for labels in split_labels.values() for label in labels.elements()
        )
        total_examples = sum(len(rows) for rows in split_rows.values())
        valid_examples = sum(sum(labels.values()) for labels in split_labels.values())
        cross_game_ready = bool(
            total_examples
            and not invalid_rows
            and valid_examples == total_examples
            and not missing_references
            and not decode_failures
            and all(not overlap for overlap in split_overlap.values())
        )
        offline_ready = bool(
            total_examples
            and not invalid_rows
            and valid_examples == total_examples
            and not missing_references
            and not decode_failures
            and set(class_counts) == set(OBJECT_CLASSES)
        )
        return {
            "schema_version": OBJECT_CLASSIFICATION_SCHEMA,
            "archive_sha256": _sha256_file(archive_path),
            "archive_size_bytes": archive_path.stat().st_size,
            "archive_member_count": len(names),
            "duplicate_archive_members": duplicate_members,
            "corrupt_archive_member": corrupt_member,
            "object_classes": list(OBJECT_CLASSES),
            "split_example_counts": {
                split: len(rows) for split, rows in split_rows.items()
            },
            "split_label_counts": {
                split: dict(sorted(labels.items()))
                for split, labels in split_labels.items()
            },
            "class_counts": dict(sorted(class_counts.items())),
            "total_examples": total_examples,
            "image_reference_count": len(all_references),
            "unique_image_reference_count": len(reference_counts),
            "duplicate_image_references": sorted(
                image for image, count in reference_counts.items() if count > 1
            ),
            "missing_image_references": missing_references[:_MAX_RECORDED_ERRORS],
            "missing_image_reference_count": len(missing_references),
            "unreferenced_image_members": unreferenced_members[:_MAX_RECORDED_ERRORS],
            "unreferenced_image_member_count": len(unreferenced_members),
            "decode_failures": decode_failures,
            "decode_failure_count": len(decode_failures),
            "decoded_dimensions": dict(sorted(dimensions.items())),
            "invalid_rows": invalid_rows,
            "invalid_row_count": total_examples - valid_examples,
            "split_game_counts": {
                split: len(games) for split, games in split_games.items()
            },
            "split_game_overlaps": split_overlap,
            "game_count": len(set().union(*split_games.values())),
            "cross_game_benchmark_ready": cross_game_ready,
            "offline_object_role_training_ready": offline_ready,
            "runtime_consumable": False,
            "codex_runtime_answer_used": False,
        }
