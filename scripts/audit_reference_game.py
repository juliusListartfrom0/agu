#!/usr/bin/env python3
"""Build a normalized reference-assisted audit package for one basketball game."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.reference_audit import (  # noqa: E402
    aggregate_player_stats,
    build_raw_frame_index,
    load_summary_stats,
    match_reference_clip,
    normalize_detail_rows,
    read_csv_rows,
    reconcile_player_stats,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-field-accuracy", type=float, default=0.95)
    parser.add_argument("--fingerprint", action="store_true")
    parser.add_argument("--raw-sample-fps", type=float, default=4.0)
    parser.add_argument("--reference-sample-fps", type=float, default=1.0)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--scene-threshold", type=float, default=0.15)
    parser.add_argument("--minimum-scene-seconds", type=float, default=0.5)
    parser.add_argument("--max-hamming-distance", type=float, default=56.0)
    parser.add_argument("--minimum-hamming-margin", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    detail_path = args.reference_dir / "明细数据表.csv"
    summary_paths = [args.reference_dir / "黑队数据.csv", args.reference_dir / "白队数据.csv"]
    raw_videos = sorted(args.raw_dir.glob("*.mov"))
    if len(raw_videos) != 6:
        raise SystemExit(f"Expected 6 raw period videos, found {len(raw_videos)}")

    events = normalize_detail_rows(read_csv_rows(detail_path))
    actual = aggregate_player_stats(events)
    expected, source_rows = load_summary_stats(summary_paths)
    reconciliation = reconcile_player_stats(actual, expected)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "event_ledger.jsonl", events)
    write_json(
        args.output_dir / "player_box_score.json",
        {
            "schema_version": "reference_audit_v1",
            "players": [
                {"team": team, "player": player, **stats, "reference_summary": source_rows[f"{team}:{player}"]}
                for (team, player), stats in sorted(actual.items())
            ],
        },
    )
    if args.fingerprint:
        raw_index = build_raw_frame_index(
            raw_videos,
            sample_fps=args.raw_sample_fps,
            ffmpeg_bin=args.ffmpeg_bin,
        )
        reference_videos = sorted(
            path
            for path in args.reference_dir.glob("*/*")
            if path.suffix.lower() in {".mov", ".mp4"}
        )
        matches = [
            match_reference_clip(
                path,
                raw_index,
                reference_fps=args.reference_sample_fps,
                max_hamming_distance=args.max_hamming_distance,
                minimum_margin=args.minimum_hamming_margin,
                ffmpeg_bin=args.ffmpeg_bin,
                ffprobe_bin=args.ffprobe_bin,
                scene_threshold=args.scene_threshold,
                minimum_scene_seconds=args.minimum_scene_seconds,
            )
            for path in reference_videos
        ]
        sample_count = sum(int(item["sample_count"]) for item in matches)
        accepted_count = sum(int(item["accepted_sample_count"]) for item in matches)
        scene_count = sum(int(item["scene_count"]) for item in matches)
        localized_scene_count = sum(int(item["localized_scene_count"]) for item in matches)
        write_json(
            args.output_dir / "reference_raw_matches.json",
            {
                "schema_version": "reference_fingerprint_v1",
                "raw_sample_fps": args.raw_sample_fps,
                "reference_sample_fps": args.reference_sample_fps,
                "scene_threshold": args.scene_threshold,
                "minimum_scene_seconds": args.minimum_scene_seconds,
                "max_hamming_distance": args.max_hamming_distance,
                "minimum_hamming_margin": args.minimum_hamming_margin,
                "sample_count": sample_count,
                "accepted_sample_count": accepted_count,
                "accepted_sample_rate": accepted_count / sample_count if sample_count else 0.0,
                "scene_count": scene_count,
                "localized_scene_count": localized_scene_count,
                "localized_scene_rate": localized_scene_count / scene_count if scene_count else 0.0,
                "clips": matches,
            },
        )
        print(
            f"fingerprint_samples={sample_count} accepted={accepted_count} "
            f"rate={accepted_count / sample_count if sample_count else 0.0:.4f} "
            f"localized_scenes={localized_scene_count}/{scene_count}"
        )
    write_json(
        args.output_dir / "reconciliation.json",
        {
            **reconciliation,
            "accuracy_scope": "expanded detail ledger versus supplied player summary counting fields",
            "raw_video_review_accuracy": None,
            "raw_video_review_status": "pending_fingerprint_localization_and_codex_review",
        },
    )
    write_json(
        args.output_dir / "manifest.json",
        {
            "schema_version": "reference_audit_v1",
            "reference_dir": str(args.reference_dir),
            "raw_videos": [str(path) for path in raw_videos],
            "event_count": len(events),
            "player_count": len(actual),
            "field_accuracy": reconciliation["field_accuracy"],
            "accuracy_claim": "reference_reconciliation_only",
        },
    )
    print(f"events={len(events)} players={len(actual)} field_accuracy={reconciliation['field_accuracy']:.4f}")
    if reconciliation["field_accuracy"] < args.minimum_field_accuracy:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
