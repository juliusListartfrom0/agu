#!/usr/bin/env python3
"""Build a hard-linked, whole-game-isolated E-BARD + MUVY ball dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from app.analysis.basketball_detector_dataset import (
    assign_e_bard_game_splits,
    filter_e_bard_ball_labels,
    parse_e_bard_game_id,
    seal_basketball_detector_dataset_manifest,
)

_SPLITS = ("train", "val", "test")
_E_BARD_EXPECTED_CLASSES = {
    "basketball": 1496,
    "hoop": 1565,
    "player": 15296,
    "referee": 3853,
}
_E_BARD_EXPECTED_SPLITS = {"train": 1440, "val": 180, "test": 180}
# Both counts are sealed, hash-bound reviews: v1 is the historical 177-box
# screen and v2 is the conservative 74-box cross-event review.  Keeping the
# allow-list explicit prevents an arbitrary local manifest from silently
# entering the combined training materializer.
_MUVY_SEALED_IMAGE_COUNTS = frozenset({74, 169})


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _hardlink(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    os.link(source, temporary)
    temporary.replace(destination)
    if source.stat().st_ino != destination.stat().st_ino:
        raise ValueError(f"hard-link verification failed: {destination}")


def _verify_e_bard_manifest(root: Path) -> tuple[dict[str, Any], str]:
    path = root / "source-manifest.json"
    manifest = _read_json(path)
    expected = {
        "schema_version": "agu.public-dataset-download.v1",
        "dataset_id": "e-bard-detection",
        "purpose": "offline_detection_pretraining_and_validation_only",
        "runtime_consumable": False,
        "source_revision": "00563215490c9a9642797b1495ea178535b3f59c",
        "license": "CC-BY-4.0",
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"unexpected E-BARD source manifest {field}")
    layout = manifest.get("yolo_layout")
    if (
        not isinstance(layout, dict)
        or layout.get("root") != "extracted/yolo"
        or layout.get("classes")
        != ["basketball", "hoop", "player", "referee"]
        or layout.get("split_image_counts") != _E_BARD_EXPECTED_SPLITS
        or layout.get("annotation_counts") != _E_BARD_EXPECTED_CLASSES
    ):
        raise ValueError("unexpected E-BARD YOLO layout")
    archive = manifest.get("archive")
    if (
        not isinstance(archive, dict)
        or archive.get("sha256")
        != "4b0a5ef8fd25565714e6b36a7020bc68b1cc2765afdc82a3b3d7a099e5c2ab81"
        or archive.get("size_bytes") != 662567020
    ):
        raise ValueError("unexpected E-BARD archive identity")
    return manifest, _file_sha256(path)


def _verify_muvy_manifest(root: Path) -> tuple[dict[str, Any], str]:
    path = root / "manifest.json"
    manifest = _read_json(path)
    claimed = manifest.get("artifact_sha256")
    unsigned = dict(manifest)
    unsigned.pop("artifact_sha256", None)
    if (
        manifest.get("schema_version") != "agu.muvy-ball-yolo.v1"
        or manifest.get("purpose")
        != "licensed_public_small_ball_detector_training"
        or manifest.get("runtime_consumable") is not False
        or manifest.get("codex_runtime_answer_used") is not False
        or manifest.get("source_license") != "CC-BY-4.0"
        or claimed != _canonical_sha256(unsigned)
    ):
        raise ValueError("invalid or unsealed MUVY source manifest")
    data_yaml = root / "data.yaml"
    if _file_sha256(data_yaml) != manifest.get("data_yaml_sha256"):
        raise ValueError("MUVY data YAML hash mismatch")
    examples = manifest.get("examples")
    image_count = manifest.get("image_count")
    box_count = manifest.get("box_count")
    actual_box_count = (
        sum(len(example.get("labels", [])) for example in examples)
        if isinstance(examples, list)
        else None
    )
    if (
        not isinstance(examples, list)
        or image_count not in _MUVY_SEALED_IMAGE_COUNTS
        or len(examples) != image_count
        or not isinstance(box_count, int)
        or actual_box_count != box_count
    ):
        raise ValueError("unexpected MUVY example count")
    return manifest, _file_sha256(path)


def _scan_e_bard(
    root: Path,
) -> tuple[list[dict[str, Any]], Counter[int]]:
    yolo_root = root / "extracted" / "yolo"
    records: list[dict[str, Any]] = []
    class_counts: Counter[int] = Counter()
    seen_images: set[str] = set()
    for original_split in _SPLITS:
        image_dir = yolo_root / original_split / "images"
        label_dir = yolo_root / original_split / "labels"
        images = sorted(image_dir.glob("*.jpg"))
        if len(images) != _E_BARD_EXPECTED_SPLITS[original_split]:
            raise ValueError(f"unexpected E-BARD {original_split} image count")
        for image in images:
            if image.name in seen_images:
                raise ValueError(f"duplicate E-BARD image name: {image.name}")
            seen_images.add(image.name)
            label = label_dir / f"{image.stem}.txt"
            if not label.is_file():
                raise ValueError(f"missing E-BARD label: {label}")
            lines = label.read_text(encoding="utf-8").splitlines()
            for line in lines:
                fields = line.split()
                if fields:
                    class_counts[int(fields[0])] += 1
            ball_lines = filter_e_bard_ball_labels(lines)
            records.append(
                {
                    "game_id": parse_e_bard_game_id(image.name),
                    "image": image,
                    "label": label,
                    "ball_lines": ball_lines,
                }
            )
    expected_by_id = {
        index: _E_BARD_EXPECTED_CLASSES[name]
        for index, name in enumerate(
            ("basketball", "hoop", "player", "referee")
        )
    }
    if len(records) != 1800 or dict(class_counts) != expected_by_id:
        raise ValueError("E-BARD extracted annotations do not match manifest")
    return records, class_counts


def _materialize_e_bard(
    records: list[dict[str, Any]],
    assignments: dict[str, str],
    output: Path,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda row: row["image"].name):
        split = assignments[record["game_id"]]
        stem = f"ebard-{record['image'].stem}"
        image_relative = Path("images") / split / f"{stem}.jpg"
        label_relative = Path("labels") / split / f"{stem}.txt"
        image_output = output / image_relative
        label_output = output / label_relative
        _hardlink(record["image"], image_output)
        label_text = "".join(f"{line}\n" for line in record["ball_lines"])
        _write_text(label_output, label_text)
        examples.append(
            {
                "source": "e_bard",
                "group_id": record["game_id"],
                "split": split,
                "source_image_path": record["image"].relative_to(
                    record["image"].parents[4]
                ).as_posix(),
                "source_label_path": record["label"].relative_to(
                    record["label"].parents[4]
                ).as_posix(),
                "image_path": image_relative.as_posix(),
                "image_sha256": _file_sha256(image_output),
                "label_path": label_relative.as_posix(),
                "label_sha256": _file_sha256(label_output),
                "ball_box_count": len(record["ball_lines"]),
            }
        )
    return examples


def _materialize_muvy(
    root: Path,
    manifest: dict[str, Any],
    output: Path,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for record in manifest["examples"]:
        split = record.get("split")
        if split not in {"train", "val"}:
            raise ValueError("MUVY source split must be train or val")
        source_image_relative = Path(record["image_path"])
        source_label_relative = Path(record["label_path"])
        source_image = root / source_image_relative
        source_label = root / source_label_relative
        if (
            _file_sha256(source_image) != record.get("image_sha256")
            or _file_sha256(source_label) != record.get("label_sha256")
        ):
            raise ValueError("MUVY source example hash mismatch")
        ball_lines = filter_e_bard_ball_labels(
            source_label.read_text(encoding="utf-8").splitlines()
        )
        if len(ball_lines) != len(record.get("labels", [])):
            raise ValueError("MUVY label content does not match manifest")
        stem = f"muvy-{source_image.stem}"
        image_relative = Path("images") / split / f"{stem}.jpg"
        label_relative = Path("labels") / split / f"{stem}.txt"
        _hardlink(source_image, output / image_relative)
        _hardlink(source_label, output / label_relative)
        examples.append(
            {
                "source": "muvy",
                "group_id": record["event"],
                "split": split,
                "source_image_path": source_image_relative.as_posix(),
                "source_label_path": source_label_relative.as_posix(),
                "image_path": image_relative.as_posix(),
                "image_sha256": record["image_sha256"],
                "label_path": label_relative.as_posix(),
                "label_sha256": record["label_sha256"],
                "ball_box_count": len(ball_lines),
            }
        )
    return examples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e-bard-root", type=Path, required=True)
    parser.add_argument("--muvy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-game-count", type=int, required=True)
    parser.add_argument("--test-game-count", type=int, required=True)
    parser.add_argument("--seed", required=True)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"output directory is not empty: {args.output}")
    e_bard_manifest, e_bard_manifest_sha = _verify_e_bard_manifest(
        args.e_bard_root
    )
    muvy_manifest, muvy_manifest_sha = _verify_muvy_manifest(args.muvy_root)
    records, _ = _scan_e_bard(args.e_bard_root)
    game_ids = sorted({record["game_id"] for record in records})
    assignments = assign_e_bard_game_splits(
        game_ids,
        validation_game_count=args.validation_game_count,
        test_game_count=args.test_game_count,
        seed=args.seed,
    )

    for split in _SPLITS:
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)
    examples = _materialize_e_bard(records, assignments, args.output)
    examples.extend(_materialize_muvy(args.muvy_root, muvy_manifest, args.output))
    examples.sort(key=lambda row: (row["split"], row["image_path"]))

    data_yaml = "\n".join(
        [
            f"path: {args.output.resolve()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: basketball",
            "  1: hoop",
            "  2: player",
            "  3: referee",
            "",
        ]
    )
    _write_text(args.output / "data.yaml", data_yaml)
    split_game_ids = {
        split: sorted(
            game_id
            for game_id, assigned_split in assignments.items()
            if assigned_split == split
        )
        for split in _SPLITS
    }
    manifest = seal_basketball_detector_dataset_manifest(
        {
            "e_bard_source_manifest_sha256": e_bard_manifest_sha,
            "muvy_source_manifest_sha256": muvy_manifest_sha,
            "split_seed": args.seed,
            "e_bard_split_game_ids": split_game_ids,
            "source_records": {
                "e_bard": {
                    "url": e_bard_manifest["source_repository"],
                    "revision": e_bard_manifest["source_revision"],
                    "license": e_bard_manifest["license"],
                },
                "muvy": {
                    "url": muvy_manifest["source_record_url"],
                    "artifact_sha256": muvy_manifest["artifact_sha256"],
                    "license": muvy_manifest["source_license"],
                },
            },
            "data_yaml_sha256": _file_sha256(args.output / "data.yaml"),
            "examples": examples,
        }
    )
    _write_text(
        args.output / "manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "artifact_sha256": manifest["artifact_sha256"],
                "split_counts": manifest["split_counts"],
                "source_counts": manifest["source_counts"],
                "ball_box_counts": manifest["ball_box_counts"],
                "e_bard_game_counts": {
                    split: len(split_game_ids[split]) for split in _SPLITS
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
