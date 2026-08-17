#!/usr/bin/env python3
"""Materialize the sealed Infactory subset as a hard-linked YOLO dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from app.analysis.infactory_ball_tracking import verify_infactory_ball_tracking_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    os.link(source, temporary)
    temporary.replace(destination)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    manifest = verify_infactory_ball_tracking_manifest(
        json.loads(args.manifest.read_text(encoding="utf-8"))
    )
    if args.output.resolve() == args.root.resolve():
        raise ValueError("YOLO output must be a child directory, not the source root")
    examples: list[dict[str, Any]] = manifest["examples"]
    materialized: list[dict[str, Any]] = []
    for example in examples:
        split = str(example["split"])
        stem = f"infactory-{example['video_source']}-{int(example['frame_index']):06d}"
        image_relative = Path("images") / split / f"{stem}.jpg"
        label_relative = Path("labels") / split / f"{stem}.txt"
        source_image = args.root / str(example["file_path"])
        destination_image = args.output / image_relative
        _hardlink(source_image, destination_image)
        if example["label_path"] is None:
            _write_text(args.output / label_relative, "")
        else:
            _hardlink(args.root / str(example["label_path"]), args.output / label_relative)
        materialized.append(
            {
                "source_file_path": example["file_path"],
                "source_image_sha256": example["image_sha256"],
                "split": split,
                "image_path": image_relative.as_posix(),
                "label_path": label_relative.as_posix(),
                "visibility": example["visibility"],
            }
        )

    data_yaml = "\n".join(
        [
            f"path: {args.output.resolve()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: ball",
            "",
        ]
    )
    _write_text(args.output / "data.yaml", data_yaml)
    derived = {
        "schema_version": "agu.infactory-ball-tracking-yolo.v1",
        "purpose": "offline_noncommercial_tiny_ball_auxiliary_pretraining_only",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "source_manifest_sha256": manifest["artifact_sha256"],
        "data_yaml": "data.yaml",
        "examples": materialized,
    }
    derived["artifact_sha256"] = _canonical_sha256(derived)
    _write_text(
        args.output / "manifest.json",
        json.dumps(derived, ensure_ascii=False, indent=2) + "\n",
    )
    print(
        json.dumps(
            {
                "artifact_sha256": derived["artifact_sha256"],
                "source_manifest_sha256": manifest["artifact_sha256"],
                "examples": len(materialized),
            },
            sort_keys=True,
        )
    )
    return 0


def _canonical_sha256(value: object) -> str:
    import hashlib

    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
