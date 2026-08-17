"""Build leakage-safe basketball detector datasets from licensed sources."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

_E_BARD_IMAGE = re.compile(
    r"[a-z]{3}-vs-[a-z]{3}-(\d{10})_\d{6}\.jpg"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SPLITS = ("train", "val", "test")
_MANIFEST_SCHEMA = "agu.basketball-detector-dataset.v1"


def parse_e_bard_game_id(filename: str) -> str:
    """Extract the immutable NBA game ID from a canonical E-BARD image name."""

    match = _E_BARD_IMAGE.fullmatch(Path(filename).name)
    if match is None:
        raise ValueError("E-BARD image filename is not canonical")
    return match.group(1)


def assign_e_bard_game_splits(
    game_ids: Iterable[str],
    *,
    validation_game_count: int,
    test_game_count: int,
    seed: str,
) -> dict[str, str]:
    """Assign complete games to deterministic, mutually exclusive splits."""

    values = list(game_ids)
    if len(values) != len(set(values)):
        raise ValueError("E-BARD game IDs must be unique")
    if (
        not values
        or validation_game_count < 1
        or test_game_count < 0
        or validation_game_count + test_game_count >= len(values)
        or not seed
    ):
        raise ValueError("invalid E-BARD game split configuration")
    ordered = sorted(
        values,
        key=lambda game_id: (
            hashlib.sha256(f"{seed}\0{game_id}".encode()).hexdigest(),
            game_id,
        ),
    )
    test_ids = set(ordered[:test_game_count])
    validation_ids = set(
        ordered[test_game_count : test_game_count + validation_game_count]
    )
    return {
        game_id: (
            "test"
            if game_id in test_ids
            else "val"
            if game_id in validation_ids
            else "train"
        )
        for game_id in sorted(values)
    }


def filter_e_bard_ball_labels(lines: Sequence[str]) -> list[str]:
    """Validate all E-BARD YOLO rows and retain only basketball class zero.

    Format source:
    https://docs.ultralytics.com/datasets/detect#ultralytics-yolo-format
    """

    basketball = []
    for line in lines:
        fields = line.split()
        if len(fields) != 5:
            raise ValueError("E-BARD YOLO row must contain five fields")
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError("invalid E-BARD YOLO value") from exc
        if class_id not in range(4):
            raise ValueError("E-BARD YOLO class is outside 0..3")
        if (
            not all(math.isfinite(value) for value in coordinates)
            or not all(0 <= value <= 1 for value in coordinates[:2])
            or not all(0 <= value <= 1 for value in coordinates[2:])
        ):
            raise ValueError("E-BARD YOLO coordinates are not normalized")
        if class_id == 0 and not all(value > 0 for value in coordinates[2:]):
            raise ValueError("E-BARD basketball box must have positive area")
        if class_id == 0:
            basketball.append(line.strip())
    return basketball


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_sha256(value: object, *, field: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _validate_basketball_detector_dataset_manifest(
    manifest: dict[str, Any],
) -> None:
    if manifest.get("schema_version") != _MANIFEST_SCHEMA:
        raise ValueError("invalid basketball detector dataset schema")
    if manifest.get("purpose") != "licensed_game_disjoint_small_ball_training":
        raise ValueError("invalid basketball detector dataset purpose")
    if manifest.get("runtime_consumable") is not False:
        raise ValueError("dataset must not be runtime consumable")
    if manifest.get("codex_runtime_answer_used") is not False:
        raise ValueError("Codex runtime answers must not be used")
    _validate_sha256(
        manifest.get("e_bard_source_manifest_sha256"),
        field="e_bard_source_manifest_sha256",
    )
    _validate_sha256(
        manifest.get("muvy_source_manifest_sha256"),
        field="muvy_source_manifest_sha256",
    )
    if not isinstance(manifest.get("split_seed"), str) or not manifest["split_seed"]:
        raise ValueError("split_seed must be non-empty")

    split_game_ids = manifest.get("e_bard_split_game_ids")
    if not isinstance(split_game_ids, dict) or set(split_game_ids) != set(_SPLITS):
        raise ValueError("E-BARD split game IDs must define train, val, and test")
    game_sets: dict[str, set[str]] = {}
    for split in _SPLITS:
        values = split_game_ids[split]
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) and value for value in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(f"E-BARD {split} game IDs must be unique and non-empty")
        game_sets[split] = set(values)
    if any(
        game_sets[left] & game_sets[right]
        for index, left in enumerate(_SPLITS)
        for right in _SPLITS[index + 1 :]
    ):
        raise ValueError("E-BARD split game IDs must be mutually disjoint")

    examples = manifest.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("dataset examples must be non-empty")
    image_paths: set[str] = set()
    label_paths: set[str] = set()
    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    ball_box_counts: Counter[str] = Counter()
    for example in examples:
        if not isinstance(example, dict):
            raise ValueError("dataset example must be an object")
        source = example.get("source")
        split = example.get("split")
        group_id = example.get("group_id")
        if source not in {"e_bard", "muvy"}:
            raise ValueError("dataset example source is invalid")
        if split not in _SPLITS:
            raise ValueError("dataset example split is invalid")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("dataset example group_id must be non-empty")
        if source == "e_bard" and group_id not in game_sets[split]:
            raise ValueError("E-BARD example is outside its whole-game split")
        if source == "muvy" and split == "test":
            raise ValueError("MUVY examples cannot enter the test split")

        image_path = example.get("image_path")
        label_path = example.get("label_path")
        if (
            not isinstance(image_path, str)
            or not image_path.startswith(f"images/{split}/")
            or image_path in image_paths
        ):
            raise ValueError("dataset image path is invalid or duplicated")
        if (
            not isinstance(label_path, str)
            or not label_path.startswith(f"labels/{split}/")
            or label_path in label_paths
        ):
            raise ValueError("dataset label path is invalid or duplicated")
        image_paths.add(image_path)
        label_paths.add(label_path)
        _validate_sha256(example.get("image_sha256"), field="image_sha256")
        _validate_sha256(example.get("label_sha256"), field="label_sha256")
        ball_box_count = example.get("ball_box_count")
        if (
            not isinstance(ball_box_count, int)
            or isinstance(ball_box_count, bool)
            or ball_box_count < 0
        ):
            raise ValueError("ball_box_count must be a non-negative integer")
        split_counts[split] += 1
        source_counts[source] += 1
        ball_box_counts[split] += ball_box_count

    expected_split_counts = {
        split: split_counts.get(split, 0) for split in _SPLITS
    }
    if manifest.get("split_counts") != expected_split_counts:
        raise ValueError("dataset split counts do not match examples")
    expected_source_counts = {
        source: source_counts.get(source, 0) for source in ("e_bard", "muvy")
    }
    if manifest.get("source_counts") != expected_source_counts:
        raise ValueError("dataset source counts do not match examples")
    expected_ball_box_counts = {
        split: ball_box_counts.get(split, 0) for split in _SPLITS
    }
    if manifest.get("ball_box_counts") != expected_ball_box_counts:
        raise ValueError("dataset ball box counts do not match examples")


def seal_basketball_detector_dataset_manifest(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Validate and hash-bind a licensed, whole-game-isolated dataset manifest."""

    manifest = deepcopy(payload)
    manifest.update(
        {
            "schema_version": _MANIFEST_SCHEMA,
            "purpose": "licensed_game_disjoint_small_ball_training",
            "runtime_consumable": False,
            "codex_runtime_answer_used": False,
        }
    )
    examples = manifest.get("examples", [])
    manifest["split_counts"] = {
        split: sum(example.get("split") == split for example in examples)
        for split in _SPLITS
    }
    manifest["source_counts"] = {
        source: sum(example.get("source") == source for example in examples)
        for source in ("e_bard", "muvy")
    }
    manifest["ball_box_counts"] = {
        split: sum(
            example.get("ball_box_count", 0)
            for example in examples
            if example.get("split") == split
        )
        for split in _SPLITS
    }
    manifest.pop("artifact_sha256", None)
    _validate_basketball_detector_dataset_manifest(manifest)
    manifest["artifact_sha256"] = _canonical_sha256(manifest)
    return manifest


def verify_basketball_detector_dataset_manifest(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Verify a sealed dataset manifest and return an isolated copy."""

    verified = deepcopy(manifest)
    claimed_sha256 = verified.pop("artifact_sha256", None)
    _validate_sha256(claimed_sha256, field="artifact_sha256")
    _validate_basketball_detector_dataset_manifest(verified)
    if _canonical_sha256(verified) != claimed_sha256:
        raise ValueError("basketball detector dataset artifact hash mismatch")
    verified["artifact_sha256"] = claimed_sha256
    return verified
