#!/usr/bin/env python3
"""Expand a sealed causal-closure selection into a review-frame spec.

This adapter does not choose windows or inspect labels.  It verifies an
externally pinned selection, binds every selected window to the exact source
manifest bytes, and materialises the frozen half-open sampling rule expected
by ``build_vru_causal_review.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_vru_causal_closure_selection import (  # noqa: E402
    verify_causal_closure_selection,
)

SPEC_SCHEMA = "agu.vru-causal-review-spec.v1"
MANIFEST_SCHEMA = "agu.vru-basketball-source-manifest.v1"
PURPOSE = "offline_continuous_game_manual_causal_review_closure"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
PATH_SAFE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--expected-selection-artifact-sha256",
        required=True,
        help="externally recorded artifact SHA-256 for the sealed selection",
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_spec_from_closure_selection(
    *,
    selection_path: Path,
    expected_selection_artifact_sha256: str,
    source_manifest_path: Path,
) -> dict[str, Any]:
    """Verify one frozen selection and expand it into exact frame indexes."""

    expected_selection_sha = _require_sha(
        expected_selection_artifact_sha256,
        "expected selection artifact SHA-256",
    )
    selection = verify_causal_closure_selection(
        _read_mapping(selection_path, "causal closure selection")
    )
    if selection["artifact_sha256"] != expected_selection_sha:
        raise ValueError("expected selection artifact SHA-256 does not match")

    manifest, source_manifest_sha = _read_mapping_with_sha(
        source_manifest_path, "VRU source manifest"
    )
    if selection["source_manifest_sha256"] != source_manifest_sha:
        raise ValueError(
            "source manifest file SHA-256 does not match the sealed selection"
        )
    sources = _load_manifest_sources(source_manifest_path, manifest)

    policy = selection["selection"]
    sample_count = _positive_int(policy["sample_count"], "selection sample count")
    sample_rate_hz = _positive_float(
        policy["sample_rate_hz"], "selection sample rate"
    )
    window_seconds = _positive_float(
        policy["window_seconds"], "selection window seconds"
    )
    if policy["interval_semantics"] != "half_open":
        raise ValueError("closure selection sampling must remain half-open")
    if not math.isclose(
        sample_count / sample_rate_hz,
        window_seconds,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("closure selection sample count does not cover its window")

    examples: list[dict[str, Any]] = []
    for window in selection["windows"]:
        source_id = window["source_id"]
        source = sources.get(source_id)
        if source is None or not _window_matches_source(window, source):
            raise ValueError(
                f"window/manifest source binding mismatch: {window['review_id']}"
            )
        frame_indexes = _sample_frame_indexes(
            window,
            sample_count=sample_count,
            sample_rate_hz=sample_rate_hz,
        )
        examples.append(
            {
                "review_id": window["review_id"],
                "clip_id": window["clip_id"],
                "frame_indexes": frame_indexes,
            }
        )

    return {
        "schema_version": SPEC_SCHEMA,
        "purpose": PURPOSE,
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "labels_hidden_from_reviewer": True,
        "radius_seconds": window_seconds / 2.0,
        "sample_period_seconds": 1.0 / sample_rate_hz,
        "sample_count": sample_count,
        "interval_semantics": "half_open",
        "source_selection_artifact_sha256": selection["artifact_sha256"],
        "expected_selection_artifact_sha256": expected_selection_sha,
        "source_manifest_sha256": source_manifest_sha,
        "examples": examples,
    }


def _load_manifest_sources(
    path: Path, manifest: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unsupported VRU source manifest")
    if manifest.get("runtime_consumable") is not False:
        raise ValueError("VRU source manifest must remain offline-only")
    if manifest.get("codex_runtime_answer_used") is not False:
        raise ValueError("VRU source manifest cannot be a runtime answer channel")
    videos = manifest.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("VRU source manifest requires videos")

    sources: dict[str, dict[str, Any]] = {}
    clip_ids: set[str] = set()
    for value in videos:
        if not isinstance(value, Mapping):
            raise ValueError("VRU source manifest video rows must be objects")
        source_id = _path_safe_id(value.get("location"), "manifest source ID")
        clip_id = _path_safe_id(value.get("clip_id"), "manifest clip ID")
        if source_id in sources or clip_id in clip_ids:
            raise ValueError("manifest source and clip IDs must be unique")
        filename = _safe_manifest_video_filename(value.get("path"), path)
        sources[source_id] = {
            "source_id": source_id,
            "clip_id": clip_id,
            "source_video_sha256": _require_sha(
                value.get("sha256"), "manifest source video SHA-256"
            ),
            "source_video_filename": filename,
            "source_fps": _positive_float(value.get("fps"), "manifest source FPS"),
            "frame_count": _positive_int(
                value.get("frame_count"), "manifest source frame count"
            ),
        }
        clip_ids.add(clip_id)
    return sources


def _window_matches_source(
    window: Mapping[str, Any], source: Mapping[str, Any]
) -> bool:
    return all(
        window.get(field) == source.get(field)
        for field in (
            "source_id",
            "clip_id",
            "source_video_sha256",
            "source_video_filename",
            "source_fps",
            "frame_count",
        )
    )


def _sample_frame_indexes(
    window: Mapping[str, Any],
    *,
    sample_count: int,
    sample_rate_hz: float,
) -> list[int]:
    start_seconds = float(window["start_seconds"])
    end_seconds = float(window["end_seconds"])
    fps = float(window["source_fps"])
    frame_count = int(window["frame_count"])
    anchor_frame = int(window["anchor_frame"])
    indexes = [
        round((start_seconds + index / sample_rate_hz) * fps)
        for index in range(sample_count)
    ]
    if (
        len(indexes) != sample_count
        or indexes != sorted(set(indexes))
        or indexes[0] < 0
        or indexes[-1] >= frame_count
        or indexes[sample_count // 2] != anchor_frame
        or indexes[-1] >= round(end_seconds * fps)
    ):
        raise ValueError(
            "closure selection samples must be unique, bounded, half-open, "
            "and anchor-aligned"
        )
    return indexes


def _safe_manifest_video_filename(value: Any, manifest_path: Path) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("manifest video path must be a path-safe relative path")
    relative_path = Path(value)
    if relative_path.is_absolute():
        raise ValueError("manifest video path must be a path-safe relative path")
    filename = relative_path.name
    if not filename or filename in {".", ".."} or Path(filename).name != filename:
        raise ValueError("manifest video path must contain a path-safe filename")

    resolved = (manifest_path.parent / relative_path).resolve()
    manifest_root = manifest_path.parent.resolve()
    repository_data_root = (ROOT / "dataset").resolve()
    if not (
        _is_relative_to(resolved, manifest_root)
        or _is_relative_to(resolved, repository_data_root)
    ):
        raise ValueError("manifest video path escapes its allowed roots")
    return filename


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _path_safe_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or PATH_SAFE_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a path-safe scalar")
    return value


def _require_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _positive_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite and positive")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return number


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _read_mapping(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object")
    return payload


def _read_mapping_with_sha(
    path: Path, description: str
) -> tuple[dict[str, Any], str]:
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object")
    return payload, hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output in {args.selection.resolve(), args.source_manifest.resolve()}:
        raise ValueError("causal review spec output must not overwrite an input")
    artifact = build_spec_from_closure_selection(
        selection_path=args.selection,
        expected_selection_artifact_sha256=(
            args.expected_selection_artifact_sha256
        ),
        source_manifest_path=args.source_manifest,
    )
    _write_json_atomic(args.output, artifact)
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "examples": len(artifact["examples"]),
                "source_selection_artifact_sha256": artifact[
                    "source_selection_artifact_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
