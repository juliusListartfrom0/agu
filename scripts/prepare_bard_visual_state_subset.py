#!/usr/bin/env python3
"""Select and materialize a small, same-game-paired BARD validation subset."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.bard_visual_state import (  # noqa: E402
    build_bard_visual_state_subset_manifest,
)
from scripts.build_pbp_visual_state_manifest import (  # noqa: E402
    load_sealed_blind_hashes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, action="append", required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--acquisition-output", type=Path, required=True)
    parser.add_argument("--sealed-blind-acquisition", type=Path)
    parser.add_argument("--materialize", action="store_true")
    return parser.parse_args()


def load_benchmark_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                source_path = str(raw.get("files") or "")
                normalized = PurePosixPath(source_path).as_posix()
                while normalized.startswith("../"):
                    normalized = normalized[3:]
                actions = ast.literal_eval(str(raw.get("actions_name") or ""))
                if not isinstance(actions, list):
                    raise ValueError("BARD benchmark actions must be a list")
                rows.append(
                    {
                        "path": normalized,
                        "actions": [
                            str(action.get("action") or "")
                            for action in actions
                            if isinstance(action, dict)
                        ],
                    }
                )
    return rows


def prepare_subset(
    *,
    repository: Path,
    benchmark_paths: list[Path],
    materialize: bool,
    sealed_blind_acquisition_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repository = repository.resolve()
    revision = _git(repository, "rev-parse", "HEAD")
    rows = load_benchmark_rows(benchmark_paths)
    blob_by_path = _git_blob_map(repository)
    selection = build_bard_visual_state_subset_manifest(
        rows=rows,
        git_blob_sha1_by_path=blob_by_path,
        source_revision=revision,
        benchmark_sha256s=[_file_sha256(path) for path in benchmark_paths],
    )
    selected_paths = [str(row["path"]) for row in selection["examples"]]
    if materialize:
        subprocess.run(
            ["git", "sparse-checkout", "add", "--stdin"],
            cwd=repository,
            input="".join(f"/{path}\n" for path in selected_paths),
            text=True,
            check=True,
        )
    blind_hashes = (
        set(load_sealed_blind_hashes(sealed_blind_acquisition_path))
        if sealed_blind_acquisition_path is not None
        else set()
    )
    acquired = []
    for row in selection["examples"]:
        relative = str(row["path"])
        path = (repository / relative).resolve()
        if not path.is_relative_to(repository):
            raise ValueError("BARD media path escapes the repository")
        if not path.is_file():
            raise ValueError(
                "BARD selected media is absent; rerun with --materialize"
            )
        if _git(repository, "hash-object", relative) != row["git_blob_sha1"]:
            raise ValueError("BARD media does not match its selected Git blob")
        media_sha = _file_sha256(path)
        if media_sha in blind_hashes:
            raise ValueError("sealed blind video cannot enter the BARD subset")
        frame_count, fps, duration = _video_metadata(path)
        acquired.append(
            {
                **row,
                "sha256": media_sha,
                "size_bytes": path.stat().st_size,
                "frame_count": frame_count,
                "fps": fps,
                "duration_seconds": duration,
            }
        )
    acquisition: dict[str, Any] = {
        "schema_version": "agu.bard-visual-state-acquisition.v1",
        "purpose": "offline_broadcast_visual_state_pretraining",
        "runtime_consumable": False,
        "truth_used_for_training_only": True,
        "codex_runtime_answer_used": False,
        "selection_artifact_sha256": selection["artifact_sha256"],
        "source_revision": revision,
        "license": "CC-BY-4.0",
        "sealed_blind_video_sha256s": sorted(blind_hashes),
        "examples": acquired,
        "summary": {
            "examples": len(acquired),
            "free_throw": sum(
                row["state"] == "free_throw" for row in acquired
            ),
            "field_goal": sum(
                row["state"] == "field_goal" for row in acquired
            ),
            "size_bytes": sum(int(row["size_bytes"]) for row in acquired),
            "duration_seconds": sum(
                float(row["duration_seconds"]) for row in acquired
            ),
        },
    }
    acquisition["artifact_sha256"] = _canonical_sha256(acquisition)
    return selection, acquisition


def _git_blob_map(repository: Path) -> dict[str, str]:
    result = subprocess.run(
        ["git", "ls-tree", "-r", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    output = {}
    for line in result.stdout.splitlines():
        metadata, path = line.split("\t", 1)
        _mode, object_type, object_sha = metadata.split()
        if object_type == "blob":
            output[path] = object_sha
    return output


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _video_metadata(path: Path) -> tuple[int, float, float]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open BARD clip: {path.name}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if frame_count <= 0 or fps <= 0:
        raise ValueError(f"invalid BARD clip metadata: {path.name}")
    return frame_count, fps, frame_count / fps


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
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    selection, acquisition = prepare_subset(
        repository=args.repository,
        benchmark_paths=args.benchmark,
        materialize=args.materialize,
        sealed_blind_acquisition_path=args.sealed_blind_acquisition,
    )
    _write_json(args.selection_output, selection)
    _write_json(args.acquisition_output, acquisition)
    print(
        json.dumps(
            {
                "selection_output": str(args.selection_output),
                "selection_sha256": selection["artifact_sha256"],
                "acquisition_output": str(args.acquisition_output),
                "acquisition_sha256": acquisition["artifact_sha256"],
                "summary": acquisition["summary"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
