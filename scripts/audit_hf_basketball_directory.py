#!/usr/bin/env python3
"""Seal a bounded Hugging Face basketball-directory discovery audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.hf_dataset_sweep import (  # noqa: E402
    build_hf_basketball_sweep,
    fetch_hf_basketball_search,
)
from app.analysis.public_research_datasets import build_public_research_catalog  # noqa: E402

TRIAGE_OVERRIDES = {
    "shesel08/basketball-segments": "unlicensed_segment_video_no_event_schema",
    "qimingli/video_human_select_basketball": "short_video_clips_no_declared_license_or_event_labels",
    "zhichengai/basketball_v0": "ccby_robotics_episode_buffer_not_broadcast_game",
    "gem/sportsett_basketball": "mit_table_to_text_statistics_no_video",
    "sumeetn/sportsmot-basketball-detection": "player_detection_candidate_no_causal_outcomes",
    "qleandataset/video-basketball-match": "existing_gated_academic_candidate",
    "bestwjh/vru_basketball": "existing_ccby_scene_only_no_event_labels",
    "rschirem/basketball-events": "duplicate_of_existing_saveerjain_research_only_release",
    "uniquedata/basketball_tracking": "existing_70_image_ball_sample_noncommercial_no_derivative",
    "onurulu17/turkish_basketball_super_league_dataset": "tabular_statistics_only",
    "anranzz/pnp_basketball": "robotics_episode_data_not_broadcast_game",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--generated-on", default="2026-08-08")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs/public_research/hf_basketball_directory_sweep_2026-08-08.json"),
    )
    args = parser.parse_args()
    rows = fetch_hf_basketball_search(limit=args.limit)
    catalog_ids: list[str] = []
    for item in build_public_research_catalog()["sources"]:
        source_url = str(item.get("source_url") or "")
        marker = "/datasets/"
        if marker in source_url:
            catalog_ids.append(source_url.split(marker, 1)[1].strip("/"))
    audit = build_hf_basketball_sweep(
        rows,
        generated_on=args.generated_on,
        catalog_dataset_ids=catalog_ids,
        triage_overrides=TRIAGE_OVERRIDES,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "result_count": audit["result_count"],
                "catalog_match_count": audit["catalog_match_count"],
                "payload_downloads_performed": audit["payload_downloads_performed"],
                "audit_sha256": audit["audit_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
