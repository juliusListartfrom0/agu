#!/usr/bin/env python3
"""Seal a metadata-only gate for continuous causal basketball sources.

The candidate table is intentionally small and explicit.  This command does
not download source payloads; it records why each candidate is or is not
eligible for a later, separately authorized acquisition step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.analysis.public_research_datasets import (
    build_continuous_causal_source_gate_audit,
    build_public_research_catalog,
)


def _candidate(
    catalog_by_id: dict[str, dict[str, Any]],
    dataset_id: str,
    *,
    declared_size_bytes: int | None,
    size_scope: str,
    rights_cleared: bool,
    broadcast_diverse: bool,
    continuous_five_by_five_video: bool,
    subsecond_ball_hand_rim_outcome_labels: bool,
    exhaustive_non_shot_hard_negatives: bool,
    evidence: list[str],
) -> dict[str, Any]:
    source = catalog_by_id[dataset_id]
    return {
        "dataset_id": dataset_id,
        "source_url": source["source_url"],
        "source_revision": source.get("source_revision") or "unversioned-metadata-audit",
        "declared_size_bytes": declared_size_bytes,
        "size_scope": size_scope,
        "data_license": source.get("data_license"),
        "rights_cleared": rights_cleared,
        "broadcast_diverse": broadcast_diverse,
        "continuous_five_by_five_video": continuous_five_by_five_video,
        "subsecond_ball_hand_rim_outcome_labels": subsecond_ball_hand_rim_outcome_labels,
        "exhaustive_non_shot_hard_negatives": exhaustive_non_shot_hard_negatives,
        "evidence": evidence,
    }


def current_candidates() -> list[dict[str, Any]]:
    """Return the currently known candidates with conservative declarations."""

    catalog = {
        item["dataset_id"]: item
        for item in build_public_research_catalog()["sources"]
    }
    return [
        _candidate(
            catalog,
            "nba-streaming-2026",
            declared_size_bytes=None,
            size_scope="paper_only_no_public_payload_or_hash_manifest",
            rights_cleared=False,
            broadcast_diverse=False,
            continuous_five_by_five_video=True,
            subsecond_ball_hand_rim_outcome_labels=False,
            exhaustive_non_shot_hard_negatives=False,
            evidence=[
                "https://arxiv.org/abs/2608.09200",
                "https://arxiv.org/html/2608.09200",
                "analysis_outputs/public_research/nba_streaming_release_audit_2026-08-12.json",
            ],
        ),
        _candidate(
            catalog,
            "sportsmot",
            declared_size_bytes=36_100_000_000,
            size_scope="declared_full_image_sequence_release",
            rights_cleared=False,
            broadcast_diverse=False,
            continuous_five_by_five_video=False,
            subsecond_ball_hand_rim_outcome_labels=False,
            exhaustive_non_shot_hard_negatives=False,
            evidence=[
                "https://github.com/MCG-NJU/SportsMOT",
                "https://huggingface.co/datasets/Lekim89/sportsmot",
            ],
        ),
        _candidate(
            catalog,
            "spacejam-basketball-actions",
            declared_size_bytes=None,
            size_scope="repository_and_external_google_drive_dataset",
            rights_cleared=False,
            broadcast_diverse=False,
            continuous_five_by_five_video=False,
            subsecond_ball_hand_rim_outcome_labels=False,
            exhaustive_non_shot_hard_negatives=False,
            evidence=[
                "https://github.com/simonefrancia/SpaceJam",
            ],
        ),
        _candidate(
            catalog,
            "basketevent-player-grounded",
            declared_size_bytes=85_794_901,
            size_scope="bounded_valid_trajectory_json_audit",
            rights_cleared=False,
            broadcast_diverse=True,
            continuous_five_by_five_video=False,
            subsecond_ball_hand_rim_outcome_labels=False,
            exhaustive_non_shot_hard_negatives=False,
            evidence=[
                "https://huggingface.co/datasets/zaywas/BasketEvent",
                "https://arxiv.org/abs/2607.21267",
                "dataset/public_sources/basketevent_v1/manifest.json",
            ],
        ),
        _candidate(
            catalog,
            "nba-rebounds-anticipation",
            declared_size_bytes=None,
            size_scope="paper_only_no_payload",
            rights_cleared=False,
            broadcast_diverse=False,
            continuous_five_by_five_video=False,
            subsecond_ball_hand_rim_outcome_labels=False,
            exhaustive_non_shot_hard_negatives=False,
            evidence=["https://arxiv.org/abs/2512.15386"],
        ),
        _candidate(
            catalog,
            "nba-games-v1",
            declared_size_bytes=63_619_934,
            size_scope="bounded_metadata_snapshot_no_video",
            rights_cleared=False,
            broadcast_diverse=False,
            continuous_five_by_five_video=False,
            subsecond_ball_hand_rim_outcome_labels=False,
            exhaustive_non_shot_hard_negatives=False,
            evidence=[
                "https://huggingface.co/datasets/choucsan/NBA_Games",
                "dataset/public_sources/nba_games_v1/manifest.json",
            ],
        ),
        _candidate(
            catalog,
            "basketball-events-research-only",
            declared_size_bytes=46_738_499,
            size_scope="bounded_annotation_and_sample_clip_audit",
            rights_cleared=False,
            broadcast_diverse=False,
            continuous_five_by_five_video=False,
            subsecond_ball_hand_rim_outcome_labels=False,
            exhaustive_non_shot_hard_negatives=False,
            evidence=[
                "https://huggingface.co/datasets/saveerjain/basketball-events",
                "dataset/public_sources/basketball_events_v1/manifest.json",
            ],
        ),
        _candidate(
            catalog,
            "muvy-basketball-v1",
            declared_size_bytes=34_211_695,
            size_scope="bounded_two_view_media_audit",
            rights_cleared=True,
            broadcast_diverse=False,
            continuous_five_by_five_video=False,
            subsecond_ball_hand_rim_outcome_labels=False,
            exhaustive_non_shot_hard_negatives=False,
            evidence=[
                "https://zenodo.org/records/13883315",
                "dataset/public_sources/muvy_v1/media_manifest.json",
            ],
        ),
        _candidate(
            catalog,
            "vru-basketball",
            declared_size_bytes=None,
            size_scope="bounded_source_subset_audit",
            rights_cleared=True,
            broadcast_diverse=False,
            continuous_five_by_five_video=False,
            subsecond_ball_hand_rim_outcome_labels=False,
            exhaustive_non_shot_hard_negatives=False,
            evidence=[
                "https://huggingface.co/datasets/BestWJH/VRU_Basketball",
                "analysis_outputs/public_research/vru_basketball_audit_v1.json",
            ],
        ),
        _candidate(
            catalog,
            "multi-event-video-mev",
            declared_size_bytes=76_009_107_602,
            size_scope="full_repo_video_shards_and_metadata_declared",
            rights_cleared=False,
            broadcast_diverse=False,
            continuous_five_by_five_video=False,
            subsecond_ball_hand_rim_outcome_labels=False,
            exhaustive_non_shot_hard_negatives=False,
            evidence=[
                "https://huggingface.co/datasets/MultiEventVideo/MEV",
                "analysis_outputs/public_research/online_candidate_supplement_2026-08-09.json",
            ],
        ),
        _candidate(
            catalog,
            "uvy-basketball",
            declared_size_bytes=3_274_165_269,
            size_scope="declared_multi-sport_zip_bounded_basketball_range_subset",
            rights_cleared=True,
            broadcast_diverse=False,
            continuous_five_by_five_video=False,
            subsecond_ball_hand_rim_outcome_labels=False,
            exhaustive_non_shot_hard_negatives=False,
            evidence=[
                "https://zenodo.org/records/21303900",
                "dataset/public_sources/uvy_v1/manifest.json",
                "analysis_outputs/public_research/uvy_basketball_audit_2026-08-08.json",
            ],
        ),
        _candidate(
            catalog,
            "basket-skill-estimation",
            declared_size_bytes=None,
            size_scope="paper_and_metadata_only",
            rights_cleared=False,
            broadcast_diverse=False,
            continuous_five_by_five_video=False,
            subsecond_ball_hand_rim_outcome_labels=False,
            exhaustive_non_shot_hard_negatives=False,
            evidence=[
                "https://arxiv.org/abs/2503.20781",
                "https://github.com/yulupan00/BASKET",
            ],
        ),
        _candidate(
            catalog,
            "nsva-2022",
            declared_size_bytes=23_167,
            size_scope="two_parquet_metadata_tables_only",
            rights_cleared=False,
            broadcast_diverse=False,
            continuous_five_by_five_video=False,
            subsecond_ball_hand_rim_outcome_labels=False,
            exhaustive_non_shot_hard_negatives=False,
            evidence=[
                "https://huggingface.co/datasets/sportsvision/nsva_subset",
                "https://github.com/jackwu502/NSVA",
                "dataset/public_sources/nsva_source/source-manifest.json",
                "dataset/public_sources/nsva_subset_v1/source-manifest.json",
                "analysis_outputs/public_research/nsva_event_text_audit_v1.json",
            ],
        ),
        _candidate(
            catalog,
            "gamecommbench-basketball",
            declared_size_bytes=21_761_000_000,
            size_scope="declared_basketball_video_payload_metadata_only",
            rights_cleared=False,
            broadcast_diverse=False,
            continuous_five_by_five_video=False,
            subsecond_ball_hand_rim_outcome_labels=False,
            exhaustive_non_shot_hard_negatives=False,
            evidence=[
                "https://huggingface.co/datasets/A4Blind/GCB",
                "dataset/public_sources/gcb_basketball_v1/source-manifest.json",
                "analysis_outputs/public_research/gcb_basketball_event_audit_v1.json",
            ],
        ),
        _candidate(
            catalog,
            "wikimedia-hctv-full-games",
            declared_size_bytes=52_343_404_172,
            size_scope="23_cc_by_full_game_file_inventory_one_bounded_seed_retained",
            rights_cleared=True,
            broadcast_diverse=False,
            continuous_five_by_five_video=True,
            subsecond_ball_hand_rim_outcome_labels=False,
            exhaustive_non_shot_hard_negatives=False,
            evidence=[
                "https://commons.wikimedia.org/wiki/Category:Videos_of_basketball_in_the_United_States",
                "dataset/public_sources/wikimedia_hctv_fullgame_v1/manifest.json",
                "analysis_outputs/public_research/wikimedia_hctv_fullgame_audit_2026-08-09.json",
            ],
        ),
        _candidate(
            catalog,
            "wikimedia-vtv-full-game",
            declared_size_bytes=2_228_583_591,
            size_scope="one_public_domain_vtv_full_game_seed_retained",
            rights_cleared=True,
            broadcast_diverse=False,
            continuous_five_by_five_video=True,
            subsecond_ball_hand_rim_outcome_labels=False,
            exhaustive_non_shot_hard_negatives=False,
            evidence=[
                "https://commons.wikimedia.org/wiki/File:Superliga_Profesional_de_Baloncesto_-_Spartans_de_Distrito_Capital_Vs._Trotamundos_de_Carabobo.webm",
                "dataset/public_sources/wikimedia_vtv_fullgame_v1/manifest.json",
                "analysis_outputs/public_research/wikimedia_vtv_fullgame_audit_2026-08-09.json",
            ],
        ),
        _candidate(
            catalog,
            "wikimedia-hctv-randolph-full-game",
            declared_size_bytes=2_152_005_777,
            size_scope="one_cc_by_hctv_full_game_seed_retained",
            rights_cleared=True,
            broadcast_diverse=False,
            continuous_five_by_five_video=True,
            subsecond_ball_hand_rim_outcome_labels=False,
            exhaustive_non_shot_hard_negatives=False,
            evidence=[
                "https://commons.wikimedia.org/wiki/File:Boys_Varsity_Basketball_v._Randolph_-_February_17,_2026.webm",
                "https://www.youtube.com/watch?v=-VeEs0aVr5M",
                "dataset/public_sources/wikimedia_hctv_randolph_v1/manifest.json",
                "analysis_outputs/public_research/wikimedia_hctv_randolph_audit_2026-08-09.json",
            ],
        ),
        _candidate(
            catalog,
            "wikimedia-hctv-vtv-continuous-seeds",
            declared_size_bytes=3_644_828_060,
            size_scope="two_independent_full_game_seeds_retained",
            rights_cleared=True,
            broadcast_diverse=True,
            continuous_five_by_five_video=True,
            subsecond_ball_hand_rim_outcome_labels=False,
            exhaustive_non_shot_hard_negatives=False,
            evidence=[
                "analysis_outputs/public_research/wikimedia_hctv_vtv_cross_source_audit_2026-08-09.json",
                "dataset/public_sources/wikimedia_hctv_fullgame_v1/manifest.json",
                "dataset/public_sources/wikimedia_vtv_fullgame_v1/manifest.json",
            ],
        ),
        _candidate(
            catalog,
            "wikimedia-hctv-hazen-randolph-vtv-continuous-seeds",
            declared_size_bytes=5_796_833_837,
            size_scope="three_independent_full_game_seeds_retained",
            rights_cleared=True,
            broadcast_diverse=True,
            continuous_five_by_five_video=True,
            subsecond_ball_hand_rim_outcome_labels=False,
            exhaustive_non_shot_hard_negatives=False,
            evidence=[
                "analysis_outputs/public_research/wikimedia_hctv_hazen_randolph_vtv_cross_source_audit_2026-08-09.json",
                "dataset/public_sources/wikimedia_hctv_fullgame_v1/manifest.json",
                "dataset/public_sources/wikimedia_hctv_randolph_v1/manifest.json",
                "dataset/public_sources/wikimedia_vtv_fullgame_v1/manifest.json",
            ],
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs/public_research/continuous_causal_source_gate_2026-08-08.json"),
    )
    parser.add_argument("--generated-on", default="2026-08-08")
    args = parser.parse_args()

    audit = build_continuous_causal_source_gate_audit(
        current_candidates(), generated_on=args.generated_on
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_count": len(audit["sources"]),
                "eligible_source_count": audit["eligible_source_count"],
                "audit_sha256": audit["audit_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
