#!/usr/bin/env python3
"""Build a sealed, non-runtime public basketball research plan."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.public_research_datasets import (  # noqa: E402
    build_nba_research_plan,
    build_public_research_catalog,
    check_youtube_oembed,
    probe_youtube_media,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--benchmark",
        action="append",
        required=True,
        metavar="TARGET=ENROLLMENT[,ENROLLMENT...]",
    )
    parser.add_argument("--repository-revision")
    parser.add_argument("--verify-public-video", action="store_true")
    parser.add_argument("--probe-media", action="store_true")
    parser.add_argument("--yt-dlp-executable", default="yt-dlp")
    parser.add_argument("--media-root", type=Path)
    parser.add_argument(
        "--media-overrides",
        type=Path,
        help="JSON mapping from game slug to a verified external media source.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mapping = _parse_benchmarks(args.benchmark)
    revision = args.repository_revision or _git_revision(args.dataset_root)
    metadata_visibility: dict[str, bool] = {}
    playback_probes: dict[str, dict[str, object]] = {}
    local_media_paths: dict[str, Path] = {}
    media_overrides = (
        _load_media_overrides(args.media_overrides) if args.media_overrides else {}
    )
    video_ids = _video_ids(args.dataset_root, mapping)
    if args.verify_public_video:
        metadata_visibility = {
            video_id: check_youtube_oembed(video_id) for video_id in video_ids
        }
    if args.probe_media:
        playback_probes = {
            video_id: probe_youtube_media(
                video_id, executable=args.yt_dlp_executable
            )
            for video_id in video_ids
        }
    if args.media_root:
        local_media_paths = _find_local_media(args.media_root, video_ids)
    plan = build_nba_research_plan(
        args.dataset_root,
        benchmark_enrollment_games=mapping,
        repository_revision=revision,
        video_metadata_visibility=metadata_visibility,
        video_playback_probes=playback_probes,
        local_media_paths=local_media_paths,
        media_overrides=media_overrides,
    )
    _write_json(args.output, plan)
    if args.catalog_output:
        _write_json(args.catalog_output, build_public_research_catalog())
    print(
        json.dumps(
            {
                "output": str(args.output),
                "plan_sha256": plan["plan_sha256"],
                "benchmark_count": plan["benchmark_count"],
                "two_game_metadata_gate_ready": plan["two_game_metadata_gate_ready"],
                "two_game_playback_gate_ready": plan["two_game_playback_gate_ready"],
                "two_game_media_gate_ready": plan["two_game_media_gate_ready"],
                "roster_coverage": {
                    item["benchmark"]["slug"]: item["active_roster_coverage"]
                    for item in plan["benchmarks"]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _parse_benchmarks(values: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for value in values:
        target, separator, enrollment_value = value.partition("=")
        enrollment = [item.strip() for item in enrollment_value.split(",") if item.strip()]
        if not separator or not target.strip() or not enrollment:
            raise ValueError(f"invalid --benchmark value: {value}")
        if target.strip() in result:
            raise ValueError(f"duplicate benchmark target: {target.strip()}")
        result[target.strip()] = enrollment
    return result


def _git_revision(dataset_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(dataset_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _video_ids(dataset_root: Path, mapping: dict[str, list[str]]) -> list[str]:
    slugs = sorted(set(mapping) | {slug for values in mapping.values() for slug in values})
    result: list[str] = []
    for slug in slugs:
        metadata_path = dataset_root / "games" / slug / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        result.append(str(metadata["source_game"]["id"]))
    return result


def _find_local_media(media_root: Path, video_ids: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for video_id in video_ids:
        matches = [
            path
            for path in media_root.glob(f"{video_id}.*")
            if path.is_file() and path.suffix not in {".part", ".json"}
        ]
        if len(matches) > 1:
            raise ValueError(f"multiple local media files found for {video_id}")
        if matches:
            result[video_id] = matches[0]
    return result


def _load_media_overrides(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("media overrides must be a JSON object keyed by game slug")
    result: dict[str, dict[str, object]] = {}
    for slug, value in payload.items():
        if not isinstance(slug, str) or not slug or not isinstance(value, dict):
            raise ValueError("media overrides must map non-empty slugs to JSON objects")
        result[slug] = dict(value)
    return result


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
