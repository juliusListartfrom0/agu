#!/usr/bin/env python3
"""Expand label-free source/time anchors into a causal review spec."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="SOURCE_ID=SECONDS",
        help="repeat once per anchor; labels are never accepted",
    )
    parser.add_argument("--radius-seconds", type=float, default=4.0)
    parser.add_argument("--sample-period-seconds", type=float, default=0.2)
    parser.add_argument(
        "--sample-count",
        type=int,
        help="strict half-open sample count; omitted preserves legacy inclusive sampling",
    )
    parser.add_argument(
        "--review-tag",
        default="v7",
        help="short review batch tag used in review IDs and purpose (default: v7)",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _positive(value: float, field: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field} must be positive")
    return value


def _parse_candidates(values: list[str]) -> list[tuple[str, float]]:
    parsed: list[tuple[str, float]] = []
    seen: set[tuple[str, float]] = set()
    for raw in values:
        source_id, separator, seconds_text = raw.partition("=")
        if separator != "=" or not source_id or not seconds_text:
            raise ValueError("--candidate must use SOURCE_ID=SECONDS")
        try:
            seconds = float(seconds_text)
        except ValueError as exc:
            raise ValueError(f"invalid candidate seconds: {raw}") from exc
        if (
            not math.isfinite(seconds)
            or seconds < 0.0
            or (source_id, seconds) in seen
        ):
            raise ValueError("candidate IDs/times must be unique and non-negative")
        seen.add((source_id, seconds))
        parsed.append((source_id, seconds))
    return parsed


def build_spec(
    *,
    manifest_path: Path,
    candidates: list[str],
    radius_seconds: float,
    sample_period_seconds: float,
    sample_count: int | None = None,
    review_tag: str = "v7",
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "agu.vru-basketball-source-manifest.v1":
        raise ValueError("unsupported VRU source manifest")
    videos = manifest.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("VRU source manifest requires videos")
    radius = _positive(radius_seconds, "radius_seconds")
    period = _positive(sample_period_seconds, "sample_period_seconds")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", review_tag):
        raise ValueError("review_tag must contain lowercase letters, digits, '_' or '-'")
    if period > radius * 2.0:
        raise ValueError("sample_period_seconds is too large for the review span")
    by_source: dict[str, dict[str, Any]] = {}
    for video in videos:
        if not isinstance(video, dict):
            raise ValueError("VRU source manifest video rows must be objects")
        source_id = str(video.get("location") or video.get("clip_id") or "")
        if SOURCE_ID_PATTERN.fullmatch(source_id) is None:
            raise ValueError("source ID must be a path-safe slug")
        if source_id in by_source:
            raise ValueError("source IDs must be unique")
        by_source[source_id] = video

    offsets: list[float] = []
    if sample_count is None:
        raw_steps = (radius * 2.0) / period
        steps = int(round(raw_steps))
        if not math.isclose(raw_steps, steps, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "sample_period_seconds must evenly cover the review span"
            )
        for index in range(steps + 1):
            offsets.append(round(-radius + index * period, 6))
    else:
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count < 3
        ):
            raise ValueError("sample_count must contain at least three samples")
        if not math.isclose(
            sample_count * period,
            radius * 2.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("sample_count and sample_period_seconds must exactly cover the review span")
        for index in range(sample_count):
            offsets.append(round(-radius + index * period, 6))
    examples: list[dict[str, Any]] = []
    seen_review_ids: set[str] = set()
    seen_physical_samples: set[tuple[str, tuple[int, ...]]] = set()
    for source_id, center_seconds in _parse_candidates(candidates):
        if source_id not in by_source:
            raise ValueError(f"unknown source_id: {source_id}")
        video = by_source[source_id]
        fps = _positive(float(video.get("fps")), "source FPS")
        duration = _positive(
            float(video.get("duration_seconds")), "source duration"
        )
        if center_seconds - radius < 0.0 or center_seconds + radius > duration:
            raise ValueError(f"candidate is outside source bounds: {source_id}={center_seconds}")
        sample_times = [center_seconds + offset for offset in offsets]
        frame_indexes = [round(timestamp * fps) for timestamp in sample_times]
        if frame_indexes != sorted(set(frame_indexes)):
            raise ValueError("sample period produces duplicate source frame indexes")
        if sample_count is not None and frame_indexes[-1] >= round(
            (center_seconds + radius) * fps
        ):
            raise ValueError(
                "strict sample quantization reaches the half-open right boundary"
            )
        inferred_frame_count = round(duration * fps)
        frame_count = video.get("frame_count", inferred_frame_count)
        if (
            isinstance(frame_count, bool)
            or not isinstance(frame_count, int)
            or frame_count < 1
        ):
            raise ValueError("source frame count must be a positive integer")
        if (
            sample_times[0] < 0.0
            or sample_times[-1] >= duration
            or frame_indexes[0] < 0
            or frame_indexes[-1] >= frame_count
        ):
            raise ValueError("sampled frames are outside the source review span")
        tag = f"{center_seconds:g}".replace(".", "p")
        review_id = f"{source_id}-{tag}s-{review_tag}"
        physical_key = (source_id, tuple(frame_indexes))
        if review_id in seen_review_ids or physical_key in seen_physical_samples:
            raise ValueError(
                "candidates alias to the same review ID or physical frame sequence"
            )
        seen_review_ids.add(review_id)
        seen_physical_samples.add(physical_key)
        examples.append(
            {
                "review_id": review_id,
                "clip_id": str(video.get("clip_id") or ""),
                "frame_indexes": frame_indexes,
            }
        )
    artifact: dict[str, Any] = {
        "schema_version": "agu.vru-causal-review-spec.v1",
        "purpose": f"offline_continuous_game_manual_causal_review_{review_tag}",
        "runtime_consumable": False,
        "codex_runtime_answer_used": False,
        "labels_hidden_from_reviewer": True,
        "radius_seconds": radius,
        "sample_period_seconds": period,
        "examples": examples,
    }
    if sample_count is not None:
        artifact["sample_count"] = sample_count
        artifact["interval_semantics"] = "half_open"
    return artifact


def main() -> int:
    args = parse_args()
    artifact = build_spec(
        manifest_path=args.manifest,
        candidates=args.candidate,
        radius_seconds=args.radius_seconds,
        sample_period_seconds=args.sample_period_seconds,
        sample_count=args.sample_count,
        review_tag=args.review_tag,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": args.output.as_posix(), "examples": len(artifact["examples"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
