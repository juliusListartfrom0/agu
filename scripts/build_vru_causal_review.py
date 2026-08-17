#!/usr/bin/env python3
"""Build a label-hidden, frame-hash-bound VRU causal review plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from typing import Any

import cv2

# Source-frame hashing is an offline evidence step. Keep OpenCV single-threaded
# so a bounded review build cannot contend with AGU services or violate the
# project's local resource guard while seeking AV1 frames.
cv2.setNumThreads(1)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.vru_causal_review import (  # noqa: E402
    build_vru_causal_review_plan,
    validate_vru_causal_review_id,
    verify_vru_causal_review_plan,
)

SPEC_SCHEMA = "agu.vru-causal-review-spec.v1"
CONTINUOUS_SPEC_PURPOSE = "offline_single_continuous_game_manual_causal_review_closure"
LEGACY_SHARED_CONTINUOUS_SPEC_PURPOSE = "offline_continuous_game_manual_causal_review_closure"
SINGLE_CONTINUOUS_SELECTION_PROTOCOL = "predeclared_single_continuous_source_geometry_only_v1"
CONTINUOUS_SPEC_FIELDS = {
    "schema_version",
    "purpose",
    "runtime_consumable",
    "codex_runtime_answer_used",
    "labels_hidden_from_reviewer",
    "radius_seconds",
    "sample_period_seconds",
    "sample_count",
    "interval_semantics",
    "source_selection_artifact_sha256",
    "expected_selection_artifact_sha256",
    "source_manifest_sha256",
    "examples",
}
# Strict closure plans sample at 8 FPS and support source videos through 60 FPS.
# A larger gap starts a new seek so arbitrary sparse specs cannot trigger an
# unbounded sequential decode across a full game.
MAX_SEQUENTIAL_FRAME_GAP = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_plan(*, manifest_path: Path, spec_path: Path) -> dict[str, Any]:
    spec = _read_json(spec_path)
    manifest = _read_json(manifest_path)
    examples = spec.get("examples")
    is_single_continuous_manifest = manifest.get("selection") == SINGLE_CONTINUOUS_SELECTION_PROTOCOL
    is_old_single_continuous_spec = (
        spec.get("purpose") == LEGACY_SHARED_CONTINUOUS_SPEC_PURPOSE
        and isinstance(examples, list)
        and bool(examples)
        and all(isinstance(row, Mapping) and set(row) == {"review_id", "frame_indexes"} for row in examples)
    )
    if spec.get("purpose") == CONTINUOUS_SPEC_PURPOSE or is_old_single_continuous_spec or is_single_continuous_manifest:
        raise ValueError("use the atomic continuous selection adapter for selection-bound specs")
    return _build_plan_from_spec(manifest_path=manifest_path, spec=spec)


def build_plan_from_verified_continuous_spec(
    *,
    manifest_path: Path,
    spec: Mapping[str, Any],
    source_selection_artifact_sha256: str,
    expected_source_manifest_sha256: str,
    source_clip_id: str,
) -> dict[str, Any]:
    """Decode and seal a receipt-verified continuous selection without a file gap."""

    if set(spec) != CONTINUOUS_SPEC_FIELDS:
        raise ValueError("continuous selection review spec fields are not canonical")
    if (
        spec.get("schema_version") != SPEC_SCHEMA
        or spec.get("purpose") != CONTINUOUS_SPEC_PURPOSE
        or spec.get("runtime_consumable") is not False
        or spec.get("codex_runtime_answer_used") is not False
        or spec.get("labels_hidden_from_reviewer") is not True
        or spec.get("sample_count") != 64
        or spec.get("sample_period_seconds") != 0.125
        or spec.get("radius_seconds") != 4.0
        or spec.get("interval_semantics") != "half_open"
        or spec.get("source_selection_artifact_sha256") != source_selection_artifact_sha256
        or spec.get("expected_selection_artifact_sha256") != source_selection_artifact_sha256
        or spec.get("source_manifest_sha256") != expected_source_manifest_sha256
    ):
        raise ValueError("continuous selection review spec receipt is invalid")
    if _file_sha256(manifest_path) != expected_source_manifest_sha256:
        raise ValueError("continuous selection source manifest receipt changed")
    examples = spec.get("examples")
    if not isinstance(examples, list) or len(examples) != 24:
        raise ValueError("continuous selection review spec requires exactly 24 examples")
    for row in examples:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"review_id", "frame_indexes"}
            or not isinstance(row["frame_indexes"], list)
            or len(row["frame_indexes"]) != 64
        ):
            raise ValueError("continuous selection review examples are not canonical")
    plan = _build_plan_from_spec(
        manifest_path=manifest_path,
        spec=dict(spec),
        forced_clip_id=source_clip_id,
        source_selection_artifact_sha256=source_selection_artifact_sha256,
    )
    try:
        verified_plan = verify_vru_causal_review_plan(
            plan,
            expected_artifact_sha256=plan.get("artifact_sha256"),
        )
    except ValueError as exc:
        raise ValueError("final continuous review plan is invalid") from exc
    if (
        verified_plan.get("source_selection_artifact_sha256") != source_selection_artifact_sha256
        or verified_plan.get("source_manifest_sha256") != expected_source_manifest_sha256
    ):
        raise ValueError("final continuous review plan receipt binding changed")
    return verified_plan


def _build_plan_from_spec(
    *,
    manifest_path: Path,
    spec: Mapping[str, Any],
    forced_clip_id: str | None = None,
    source_selection_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != "agu.vru-basketball-source-manifest.v1":
        raise ValueError("unsupported VRU source manifest")
    if manifest.get("runtime_consumable") is not False:
        raise ValueError("VRU source manifest must remain offline-only")
    if manifest.get("codex_runtime_answer_used") is not False:
        raise ValueError("VRU source manifest cannot be a runtime answer channel")
    videos = manifest.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("VRU source manifest requires videos")
    videos_by_clip: dict[str, dict[str, Any]] = {}
    for row in videos:
        if not isinstance(row, dict):
            raise ValueError("VRU source manifest video rows must be objects")
        clip_id = str(row.get("clip_id") or "")
        if not clip_id or clip_id in videos_by_clip:
            raise ValueError("VRU source manifest clip IDs must be unique")
        videos_by_clip[clip_id] = row

    if spec.get("schema_version") not in (None, SPEC_SCHEMA):
        raise ValueError("unsupported VRU causal review spec")
    examples = spec.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("VRU causal review spec requires examples")
    planned: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    source_sha256s: dict[Path, str] = {}
    source_captures: dict[Path, Any] = {}
    try:
        for row in examples:
            if not isinstance(row, dict):
                raise ValueError("VRU causal review spec rows must be objects")
            review_id = validate_vru_causal_review_id(row.get("review_id"))
            clip_id = forced_clip_id or str(row.get("clip_id") or "")
            frame_indexes = row.get("frame_indexes")
            if not review_id or review_id in seen_ids:
                raise ValueError("VRU causal review IDs must be unique")
            if clip_id not in videos_by_clip:
                raise ValueError(f"unknown VRU clip: {clip_id}")
            if not isinstance(frame_indexes, list) or len(frame_indexes) < 3:
                raise ValueError(f"{review_id} requires at least three frame indexes")
            try:
                indexes = [int(index) for index in frame_indexes]
            except (TypeError, ValueError):
                raise ValueError(f"{review_id} frame indexes must be integers") from None
            if indexes != sorted(set(indexes)):
                raise ValueError(f"{review_id} frame indexes must be sorted and unique")

            video = videos_by_clip[clip_id]
            frame_count = _positive_int(video.get("frame_count"), "frame count")
            if any(index < 0 or index >= frame_count for index in indexes):
                raise ValueError(f"{review_id} frame index is outside the source video")
            video_path = _resolve_manifest_video_path(
                manifest_path=manifest_path,
                value=video.get("path"),
            )
            source_video_sha256 = source_sha256s.get(video_path)
            if source_video_sha256 is None:
                source_video_sha256 = _file_sha256(video_path)
                source_sha256s[video_path] = source_video_sha256
            if source_video_sha256 != str(video.get("sha256") or ""):
                raise ValueError(f"VRU source video hash mismatch: {video_path.name}")
            cap = source_captures.get(video_path)
            if cap is None:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    cap.release()
                    raise ValueError(f"unable to open VRU source video: {video_path}")
                source_captures[video_path] = cap
            planned.append(
                {
                    "review_id": review_id,
                    "source_video_sha256": source_video_sha256,
                    "source_video_filename": video_path.name,
                    "source_fps": float(video.get("fps")),
                    "frame_count": frame_count,
                    "frame_indexes": indexes,
                    "frame_sha256s": _frame_sha256s(cap, video_path, indexes),
                }
            )
            seen_ids.add(review_id)
    finally:
        for cap in source_captures.values():
            cap.release()

    return build_vru_causal_review_plan(
        source_manifest_sha256=_file_sha256(manifest_path),
        examples=planned,
        source_selection_artifact_sha256=source_selection_artifact_sha256,
    )


def _frame_sha256s(cap: Any, path: Path, indexes: list[int]) -> dict[str, str]:
    """Hash sorted frame indexes in bounded sequential decode runs."""

    output: dict[str, str] = {}
    for run in _bounded_sequential_runs(indexes):
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, run[0]):
            raise ValueError(f"unable to seek VRU source frame {run[0]}: {path}")
        wanted = set(run)
        current = run[0]
        last = run[-1]
        while current <= last:
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"unable to decode VRU source frame {current}: {path}")
            if current in wanted:
                output[str(current)] = hashlib.sha256(frame.tobytes()).hexdigest()
            current += 1
    return dict(sorted(output.items(), key=lambda item: int(item[0])))


def _bounded_sequential_runs(indexes: list[int]) -> list[list[int]]:
    runs = [[indexes[0]]]
    for index in indexes[1:]:
        if index - runs[-1][-1] > MAX_SEQUENTIAL_FRAME_GAP:
            runs.append([])
        runs[-1].append(index)
    return runs


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return payload


def _resolve_manifest_video_path(*, manifest_path: Path, value: Any) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("VRU source video path must be a safe relative scalar")
    relative = Path(value)
    if relative.is_absolute() or PureWindowsPath(value).drive or any(part in {".", ".."} for part in relative.parts):
        raise ValueError("VRU source video path must be relative to the manifest root")
    root = manifest_path.parent.resolve(strict=True)
    try:
        resolved = (root / relative).resolve(strict=True)
    except OSError as exc:
        raise ValueError("VRU source video is missing inside the manifest root") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("VRU source video path must remain inside the manifest root") from exc
    if not resolved.is_file():
        raise ValueError("VRU source video must be a regular file")
    return resolved


def _freeze_paths(
    manifest_path: Path,
    spec_path: Path,
    output_path: Path,
) -> tuple[Path, Path, Path]:
    manifest = manifest_path.resolve(strict=True)
    spec = spec_path.resolve(strict=True)
    manifest_payload = _read_json(manifest)
    videos = manifest_payload.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("VRU source manifest requires videos")
    source_paths = tuple(
        _resolve_manifest_video_path(manifest_path=manifest, value=row.get("path"))
        for row in videos
        if isinstance(row, Mapping)
    )
    if len(source_paths) != len(videos):
        raise ValueError("VRU source manifest video rows must be objects")
    protected = (manifest, spec, *source_paths)
    output = output_path.resolve(strict=False)
    if output.as_posix().casefold() in {path.as_posix().casefold() for path in protected} or (
        output.exists() and any(os.path.samefile(output, path) for path in protected)
    ):
        raise ValueError("review plan output must not alias an input")
    return manifest, spec, output


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    manifest, spec, output = _freeze_paths(
        args.manifest,
        args.spec,
        args.output,
    )
    artifact = build_plan(manifest_path=manifest, spec_path=spec)
    _atomic_write_json(output, artifact)
    print(
        json.dumps(
            {
                "output": str(output),
                "examples": len(artifact["examples"]),
                "artifact_sha256": artifact["artifact_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
