#!/usr/bin/env python3
"""Build deterministic side-by-side Codex review pages from fingerprint matches."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.reference_audit import write_json  # noqa: E402

DEFAULT_SAMPLE_COUNTS = {
    "player_highlight": 15,
    "player_missed": 15,
    "team_highlight": 10,
    "foul_violation": 10,
    "defense": 10,
}


def classify_clip(path: Path) -> str:
    if path.name.startswith("missed_"):
        return "player_missed"
    if path.name.startswith("防守"):
        return "defense"
    if path.name.startswith("违例"):
        return "foul_violation"
    if path.name in {"highlight_白队.mov", "highlight_黑队.mov"}:
        return "team_highlight"
    return "player_highlight"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", default="agu-reference-codex-review-v1")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matches = json.loads(args.matches_json.read_text(encoding="utf-8"))
    candidates: dict[str, list[dict[str, object]]] = defaultdict(list)
    for clip in matches["clips"]:
        reference_path = Path(clip["reference_video"])
        category = classify_clip(reference_path)
        for scene_index, segment in enumerate(clip["segments"]):
            if segment["status"] != "localized":
                continue
            key = f"{args.seed}|{reference_path}|{scene_index}"
            candidates[category].append(
                {
                    "sort_key": hashlib.sha256(key.encode("utf-8")).hexdigest(),
                    "category": category,
                    "reference_video": str(reference_path),
                    "scene_index": scene_index,
                    **segment,
                }
            )

    selected: list[dict[str, object]] = []
    for category, count in DEFAULT_SAMPLE_COUNTS.items():
        selected.extend(sorted(candidates[category], key=lambda item: item["sort_key"])[:count])
    selected.sort(key=lambda item: (item["category"], item["sort_key"]))

    pair_dir = args.output_dir / "pairs"
    page_dir = args.output_dir / "pages"
    pair_dir.mkdir(parents=True, exist_ok=True)
    page_dir.mkdir(parents=True, exist_ok=True)
    queue: list[dict[str, object]] = []
    for index, item in enumerate(selected, start=1):
        reference_time = (float(item["reference_start"]) + float(item["reference_end"])) / 2
        raw_time = (float(item["raw_start"]) + float(item["raw_end"])) / 2
        pair_path = pair_dir / f"pair_{index:03d}.jpg"
        subprocess.run(
            [
                args.ffmpeg_bin,
                "-y",
                "-v",
                "error",
                "-ss",
                f"{reference_time:.3f}",
                "-i",
                str(item["reference_video"]),
                "-ss",
                f"{raw_time:.3f}",
                "-i",
                str(item["raw_video"]),
                "-filter_complex",
                "[0:v]scale=480:270[ref];[1:v]scale=480:270[raw];[ref][raw]vstack=inputs=2",
                "-frames:v",
                "1",
                str(pair_path),
            ],
            check=True,
        )
        queue.append(
            {
                "sample_id": f"sample_{index:03d}",
                "category": item["category"],
                "reference_video": item["reference_video"],
                "reference_time": reference_time,
                "raw_video": item["raw_video"],
                "raw_time": raw_time,
                "mean_distance": item["mean_distance"],
                "mean_margin": item["mean_margin"],
                "evidence_image": str(pair_path),
                "codex_decision": "pending",
                "codex_reason": "",
            }
        )

    for page_index in range((len(queue) + 5) // 6):
        first = page_index * 6 + 1
        page_pairs = [pair_dir / f"pair_{index:03d}.jpg" for index in range(first, min(first + 6, len(queue) + 1))]
        inputs = []
        for pair_path in page_pairs:
            inputs.extend(["-i", str(pair_path)])
        labels = "".join(f"[{index}:v]" for index in range(len(page_pairs)))
        layout = "|".join(
            f"{(index % 3) * 488}_{(index // 3) * 548}" for index in range(len(page_pairs))
        )
        page_path = page_dir / f"page_{page_index + 1:02d}.jpg"
        subprocess.run(
            [
                args.ffmpeg_bin,
                "-y",
                "-v",
                "error",
                *inputs,
                "-filter_complex",
                f"{labels}xstack=inputs={len(page_pairs)}:layout={layout}:fill=black",
                "-frames:v",
                "1",
                str(page_path),
            ],
            check=True,
        )

    write_json(
        args.output_dir / "review_queue.json",
        {
            "schema_version": "codex_reference_review_v1",
            "seed": args.seed,
            "sample_counts": DEFAULT_SAMPLE_COUNTS,
            "sample_size": len(queue),
            "layout": "Each pair is reference frame on top and matched raw frame below; six pairs per page.",
            "samples": queue,
        },
    )
    print(f"samples={len(queue)} pages={(len(queue) + 5) // 6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
